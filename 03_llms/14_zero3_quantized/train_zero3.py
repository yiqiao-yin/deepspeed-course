"""ZeRO-3 BF16 vs NF4: model-loading diagnostic (GitHub issue #124).

This stage measures parameter representation immediately after from_pretrained,
BEFORE deepspeed.initialize(). It does not establish post-engine replication,
training quality, or a ZeRO-3 speed/memory advantage.

CPU check:
    python train_zero3.py --arm bf16 --plan
Linux, 2 GPUs (one arm per separate launch):
    deepspeed --num_gpus=2 train_zero3.py --arm bf16
    deepspeed --num_gpus=2 train_zero3.py --arm nf4
"""

import argparse
import json
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
CONFIGS = {"bf16": "ds_config_bf16.json", "nf4": "ds_config_nf4.json"}
LAB_DIR = Path(__file__).resolve().parent


def load_config(arm):
    """Reject configurations that cannot represent this experiment."""
    path = LAB_DIR / CONFIGS[arm]
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("zero_optimization", {}).get("stage") != 3:
        raise ValueError("Both arms must use ZeRO stage 3")
    if config.get("bf16", {}).get("enabled") is not True:
        raise ValueError("Both arms must enable BF16 compute")
    if config.get("train_micro_batch_size_per_gpu") != 1:
        raise ValueError("Expected micro-batch size 1 per GPU")
    return config


def initialize_distributed():
    """Initialize processes; reject single-rank or misassigned devices."""
    import torch
    import deepspeed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use --plan for CPU-only checks")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Missing LOCAL_RANK; launch through deepspeed --num_gpus=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK does not correspond to a visible GPU")
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world < 2:
        raise RuntimeError(f"At least two distributed ranks required; found {world}")
    # This checks device indices, not independently verified physical GPU UUIDs.
    assignments = [None] * world
    torch.distributed.all_gather_object(assignments, {
        "rank": rank, "local_rank": local_rank,
        "hostname": __import__("socket").gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    if len({(a["hostname"], a["local_rank"]) for a in assignments}) != world:
        raise RuntimeError("Distributed ranks share a host/local device assignment")
    return {"rank": rank, "local_rank": local_rank, "world_size": world,
            "device": f"cuda:{local_rank}", "rank_assignments": assignments}


def _logical_count(param):
    """Use ZeRO metadata, or original NF4 dimensions where available."""
    ds_count = getattr(param, "ds_numel", None)
    if ds_count is not None:
        return int(ds_count)
    quant_state = getattr(param, "quant_state", None)
    shape = getattr(quant_state, "shape", None)
    if shape is not None:
        count = 1
        for dim in shape:
            count *= int(dim)
        return count
    return int(param.numel())


def parameter_partition_details(model):
    """Per-parameter observations, not a proof of cross-rank replication."""
    result = []
    for name, param in model.named_parameters():
        local = int(param.numel())
        logical = _logical_count(param)
        ds_id = getattr(param, "ds_id", None)
        quant_state = getattr(param, "quant_state", None)
        is_quantized = quant_state is not None or type(param).__name__ == "Params4bit"
        # Distinguish packed NF4 storage from a local ZeRO shard.
        has_shard_metadata = ds_id is not None and getattr(param, "ds_numel", None) is not None
        result.append({
            "name": name, "logical_elements": logical,
            "local_elements": local,
            "local_bytes": local * param.element_size(),
            "device": str(param.device), "dtype": str(param.dtype),
            "requires_grad": bool(param.requires_grad),
            "is_quantized": bool(is_quantized),
            "deepspeed_managed": bool(ds_id is not None),
            "has_shard_metadata": bool(has_shard_metadata),
            "ds_id": int(ds_id) if ds_id is not None else None,
            "ds_status": str(getattr(param, "ds_status", "not_present")),
            "ds_numel": (int(param.ds_numel) if hasattr(param, "ds_numel") else None),
        })
    return result


def classify_partition_state(row):
    """Classify observed parameter storage without inferring replication."""

    if row["has_shard_metadata"]:
        if row["local_elements"] < row["ds_numel"]:
            return "locally_partitioned"

        if row["local_elements"] == row["ds_numel"]:
            return "full_size_deepspeed_managed"

        return "unexpected_deepspeed_size"

    if row["is_quantized"]:
        return "quantized_without_shard_metadata"

    return "unmanaged_parameter"

def parameter_stats(model):
    """Aggregate diagnostics while excluding NF4 packing from shard inference."""
    details = parameter_partition_details(model)
    return {
        "logical_parameters": sum(r["logical_elements"] for r in details),
        "local_elements": sum(r["local_elements"] for r in details),
        "local_parameter_bytes": sum(r["local_bytes"] for r in details),
        "deepspeed_managed_tensors": sum(r["deepspeed_managed"] for r in details),
        "partitioned_tensors": sum(r["has_shard_metadata"] and r["local_elements"] < r["ds_numel"] for r in details),
        "quantized_tensors": sum(r["is_quantized"] for r in details),
        "frozen_parameters": sum(r["logical_elements"] for r in details if not r["requires_grad"]),
        "trainable_parameters": sum(r["logical_elements"] for r in details if r["requires_grad"]),
        "parameter_details": details,
    }


def load_model(arm, config, local_rank):
    """Register ZeRO-3 before model creation; keep HF config alive."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.integrations import HfDeepSpeedConfig

    ds_config = HfDeepSpeedConfig(config)
    kwargs = {"dtype": torch.bfloat16}
    if arm == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # NF4 packed weights need a concrete CUDA destination. This remains
        # subject to validation on the selected Linux/DeepSpeed versions.
        kwargs["device_map"] = {"": f"cuda:{local_rank}"}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.config.use_cache = False
    return model, ds_config


def save_rank_results(stats, output_dir="results"):
    """Write rank-specific JSON; never label it as a validated research result."""
    root = Path(output_dir)
    if not root.is_absolute():
        root = LAB_DIR / root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stats['arm']}_rank{stats['rank']}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")
    print(f"Saved {path}", flush=True)
    return path

def attach_lora(model, arm):
    """Freeze the base model and attach identical LoRA adapters."""

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if arm == "nf4":
        model = prepare_model_for_kbit_training(model)
    else:
        for param in model.parameters():
            param.requires_grad_(False)

        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(
        getattr(p, "ds_numel", p.numel())
        for p in model.parameters()
        if p.requires_grad
    )

    if trainable == 0:
        raise RuntimeError(
            "LoRA initialization failed: no trainable parameters."
        )

    return model

def measure_model_loading(arm, config, output_dir):
    """Measure parameter state before and after DeepSpeed engine initialization."""
    import torch
    from importlib.metadata import version

    distributed = initialize_distributed()
    device = distributed["local_rank"]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_bytes = torch.cuda.memory_allocated(device)
    
    model, hf_ds_config = load_model(arm, config, device)
    
    # Record parameter statistics before attaching LoRA.
    before_lora = parameter_stats(model)

    # Attach identical LoRA adapters to both experimental arms.
    model = attach_lora(model, arm)

    # Record parameter statistics after attaching LoRA.
    after_lora = parameter_stats(model)
    
    stats = parameter_stats(model)
    stats["before_lora"] = before_lora
    stats["after_lora"] = after_lora

    # Record parameter distribution before engine initilization.
    before_engine = parameter_stats(model)

    # Initialize DeepSpeed ZeRO-3.
    import deepspeed

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable LoRA parameters were found."
        )

    # Initialize the DeepSpeed engine using trainable LoRA parameters.
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_parameters,
        config=config,
    )
    #Measure parameter distribution after engine initialization.
    after_engine = parameter_stats(engine.module)

    stats["before_engine"] = before_engine
    stats["after_engine"] = after_engine
    
    torch.cuda.synchronize(device)
    stats.update({
        "arm": arm, "model": MODEL_ID, "zero_stage_configured": 3,
        "rank": distributed["rank"], "world_size": distributed["world_size"],
        "local_rank": device, "rank_assignments": distributed["rank_assignments"],
        "gpu_name": torch.cuda.get_device_name(device),
        "measurement_stage": "after_deepspeed_engine_initialization",
        "verified_post_engine": True,
        "start_allocated_gpu_bytes": start_bytes,
        "allocated_gpu_bytes": torch.cuda.memory_allocated(device),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "versions": {name: version(name) for name in
                     ("transformers", "deepspeed", "torch", "bitsandbytes", "peft")},
    })
    # hf_ds_config must remain alive while model is loaded and inspected.
    assert hf_ds_config is not None
    save_rank_results(stats, output_dir=output_dir)
    torch.distributed.barrier()
    del optimizer, engine, model, hf_ds_config
    torch.distributed.destroy_process_group()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--plan", action="store_true", help="CPU-only config check")
    parser.add_argument("--output-dir", default="results")
    # DeepSpeed launchers may pass the local rank through CLI rather than env.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args = parser.parse_args()
    if args.local_rank is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    config = load_config(args.arm)
    if args.plan:
        print(f"Model: {MODEL_ID}\nArm: {args.arm}\nZeRO stage: 3")
        print("PASS: configuration structure valid. GPU measurements: NOT PERFORMED")
        return
    measure_model_loading(args.arm, config, args.output_dir)


if __name__ == "__main__":
    main()
