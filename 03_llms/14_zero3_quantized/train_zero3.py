"""ZeRO-3 BF16 vs NF4: model-loading diagnostic (GitHub issue #124).

This initialization-only experiment measures parameter representation before
and after LoRA attachment and after deepspeed.initialize(). It makes no claims
about cross-rank replication, training quality, or ZeRO-3 speed or memory gains.

CPU check:
    python train_zero3.py --arm bf16 --plan
Linux, 2 GPUs (one arm per separate launch):
    deepspeed --num_gpus=2 train_zero3.py --arm bf16
    deepspeed --num_gpus=2 train_zero3.py --arm nf4
"""

import argparse
import json
import math
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


def parameter_partition_details(model):
    """Report each rank's parameter views and primary ZeRO backing shards."""
    result = []
    for name, param in model.named_parameters():
        local = int(param.numel())
        ds_id = getattr(param, "ds_id", None)
        ds_numel = getattr(param, "ds_numel", None)
        ds_aligned = getattr(param, "ds_numel_aligned", None)
        shard = getattr(param, "ds_tensor", None)
        shard_elements = int(shard.numel()) if shard is not None else None
        partition_elements = getattr(shard, "ds_numel", None)
        quant_state = getattr(param, "quant_state", None)
        if quant_state is None:
            quant_state = getattr(shard, "quant_state", None)
        quant_shape = getattr(quant_state, "shape", None)
        is_quantized = quant_state is not None or type(param).__name__ == "Params4bit"
        quant_storage = getattr(param, "quant_storage", None)
        has_zero_quant_scale = getattr(shard, "ds_quant_scale", None) is not None

        # A Params4bit shape describes the original weights. ZeRO's ds_numel
        # describes the representation it partitioned, which can already be packed.
        logical = math.prod(quant_shape) if quant_shape is not None else ds_numel
        padding_size = getattr(param, "padding_size", None)
        padding = None
        if ds_aligned is not None and ds_numel is not None:
            padding = int(ds_aligned - ds_numel)
        elif callable(padding_size):
            padding = int(padding_size())

        row = {
            "name": name, "logical_elements": int(logical) if logical is not None else local,
            # These existing fields describe the parameter view, even when gathered.
            "local_elements": local,
            "local_bytes": local * param.element_size(),
            "device": str(param.device), "dtype": str(param.dtype),
            "requires_grad": bool(param.requires_grad),
            "is_quantized": is_quantized,
            "quant_type": getattr(quant_state, "quant_type", None) or getattr(param, "quant_type", None),
            "quant_shape": list(quant_shape) if quant_shape is not None else None,
            "quant_blocksize": getattr(quant_state, "blocksize", None),
            "quant_nested": getattr(quant_state, "nested", None),
            "quant_storage_dtype": str(quant_storage) if quant_storage is not None else None,
            "deepspeed_managed": ds_id is not None,
            "has_shard_metadata": ds_id is not None and ds_numel is not None and shard is not None,
            "ds_id": int(ds_id) if ds_id is not None else None,
            "ds_status": str(getattr(param, "ds_status", "not_present")),
            "ds_numel": int(ds_numel) if ds_numel is not None else None,
            "ds_numel_aligned": int(ds_aligned) if ds_aligned is not None else None,
            "ds_padding_elements": padding,
            "shard_elements": shard_elements,
            "shard_bytes": shard_elements * shard.element_size() if shard is not None else None,
            "shard_device": str(shard.device) if shard is not None else None,
            "shard_dtype": str(shard.dtype) if shard is not None else None,
            "shard_status": str(getattr(shard, "status", "not_present")),
            "shard_partition_elements": int(partition_elements) if partition_elements is not None else None,
            "shard_has_zero_quant_scale": has_zero_quant_scale,
        }
        row["partition_state"] = classify_partition_state(row)
        result.append(row)
    return result


def classify_partition_state(row):
    """Compare ZeRO's partition capacity with the size ZeRO partitioned."""
    if row["deepspeed_managed"]:
        if row["shard_elements"] is None:
            return "missing_deepspeed_shard"
        if not row["has_shard_metadata"]:
            return "incomplete_deepspeed_shard_metadata"

        partition_elements = row["shard_partition_elements"]
        if partition_elements is None:
            if row["is_quantized"] or row["shard_has_zero_quant_scale"] or not row["shard_elements"]:
                return "incomplete_deepspeed_shard_metadata"
            partition_elements = row["shard_elements"]

        if partition_elements < row["ds_numel"]:
            return "locally_partitioned"
        if partition_elements == row["ds_numel"]:
            return "full_size_deepspeed_managed"
        return "unexpected_deepspeed_size"

    if row["is_quantized"]:
        return "quantized_without_shard_metadata"
    return "unmanaged_parameter"


def parameter_stats(model):
    """Aggregate tensor payload bytes separately for views and primary shards.

    Quantization metadata and allocator overhead are outside these totals.
    """
    details = parameter_partition_details(model)
    return {
        "logical_parameters": sum(r["logical_elements"] for r in details),
        "local_elements": sum(r["local_elements"] for r in details),
        "local_parameter_bytes": sum(r["local_bytes"] for r in details),
        "shard_elements": sum(r["shard_elements"] for r in details if r["shard_elements"] is not None),
        "shard_bytes": sum(r["shard_bytes"] for r in details if r["shard_bytes"] is not None),
        "deepspeed_managed_tensors": sum(r["deepspeed_managed"] for r in details),
        "partitioned_tensors": sum(r["partition_state"] == "locally_partitioned" for r in details),
        "missing_shard_tensors": sum(r["partition_state"] == "missing_deepspeed_shard" for r in details),
        "quantized_tensors": sum(r["is_quantized"] for r in details),
        "frozen_parameters": sum(r["logical_elements"] for r in details if not r["requires_grad"]),
        "trainable_parameters": sum(r["logical_elements"] for r in details if r["requires_grad"]),
        "parameter_details": details,
    }


def load_model(arm, config):
    """Register ZeRO-3 before model creation; keep HF config alive."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.integrations import HfDeepSpeedConfig

    ds_config = HfDeepSpeedConfig(config)
    kwargs = {"dtype": torch.bfloat16}
    if arm == "nf4":
        # Transformers uses the current CUDA device set during rank initialization.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
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
    
    model, hf_ds_config = load_model(arm, config)
    
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
