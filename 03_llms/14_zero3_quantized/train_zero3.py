"""ZeRO-3 BF16 vs NF4: model-loading diagnostic (GitHub issue #124).

This initialization-only experiment measures parameter representation before
and after LoRA attachment and after deepspeed.initialize(). It makes no claims
about cross-rank replication, training quality, or ZeRO-3 speed or memory gains.

CPU check:
    python train_zero3.py --arm bf16 --plan
Linux, 2 GPUs (one arm per separate launch):
    deepspeed --num_gpus=2 train_zero3.py --arm bf16
    deepspeed --num_gpus=2 train_zero3.py --arm nf4

Each launch reserves a fresh subdirectory under --output-dir, shared by its
ranks. Use measurements_complete.json to identify runs with all rank reports.
The manifest records collection completion. Scientific validation is separate.
A null logical count means the original element count was unavailable.
"""

import argparse
import json
import math
import os
import tempfile
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
        param_quant_state = getattr(param, "quant_state", None)
        shard_quant_state = getattr(shard, "quant_state", None)

        def quant_metadata(field, fallback=None):
            value = getattr(param_quant_state, field, None)
            if value is None:
                value = getattr(shard_quant_state, field, None)
            return fallback if value is None else value

        quant_shape = quant_metadata("shape")
        quant_type = quant_metadata(
            "quant_type", getattr(param, "quant_type", None) or getattr(shard, "quant_type", None),
        )
        bnb_quantized = getattr(param, "bnb_quantized", None)
        is_quantized = (
            param_quant_state is not None or shard_quant_state is not None
            or quant_type in ("nf4", "fp4") or bnb_quantized is True
            or any(cls.__name__ == "Params4bit" for cls in type(param).__mro__)
        )
        quant_storage = getattr(param, "quant_storage", None)
        if quant_storage is None:
            quant_storage = getattr(shard, "quant_storage", None)
        has_zero_quant_scale = getattr(shard, "ds_quant_scale", None) is not None

        # ZeRO's ds_numel can describe an already packed representation.
        if is_quantized:
            logical = math.prod(quant_shape) if quant_shape is not None else None
        else:
            logical = ds_numel if ds_id is not None else local

        padding_size = getattr(param, "padding_size", None)
        padding = None
        if ds_aligned is not None and ds_numel is not None:
            padding = int(ds_aligned - ds_numel)
        elif callable(padding_size):
            padding = int(padding_size())
        partition_world_size = getattr(param, "ds_zero_partition_world_size", None)

        row = {
            "name": name, "logical_elements": int(logical) if logical is not None else None,
            # These existing fields describe the parameter view, even when gathered.
            "local_elements": local,
            "local_bytes": local * param.element_size(),
            "device": str(param.device), "dtype": str(param.dtype),
            "requires_grad": bool(param.requires_grad),
            # Quantization metadata or a 4-bit wrapper can precede packing.
            "is_quantized": is_quantized,
            "bnb_quantized": bnb_quantized,
            "quant_type": quant_type,
            "quant_shape": list(quant_shape) if quant_shape is not None else None,
            "quant_blocksize": quant_metadata("blocksize", getattr(param, "blocksize", None)),
            "quant_nested": quant_metadata("nested"),
            "quant_compress_statistics_configured": getattr(param, "compress_statistics", None),
            "quant_storage_dtype": str(quant_storage) if quant_storage is not None else None,
            "deepspeed_managed": ds_id is not None,
            "has_shard_metadata": ds_id is not None and ds_numel is not None and shard is not None,
            "ds_id": int(ds_id) if ds_id is not None else None,
            "ds_status": str(getattr(param, "ds_status", "not_present")),
            "ds_numel": int(ds_numel) if ds_numel is not None else None,
            "ds_numel_aligned": int(ds_aligned) if ds_aligned is not None else None,
            "ds_padding_elements": padding,
            "ds_partition_world_size": (
                int(partition_world_size) if partition_world_size is not None else None
            ),
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

        full_elements = row["ds_numel"]
        padding = row["ds_padding_elements"]
        group_size = row["ds_partition_world_size"]
        aligned = full_elements + padding if padding is not None else None

        if full_elements < 0 or partition_elements < 0 or (padding is not None and padding < 0):
            return "unexpected_deepspeed_size"
        if group_size is not None:
            if group_size < 1:
                return "unexpected_deepspeed_size"
            expected_capacity = (full_elements + group_size - 1) // group_size
            if partition_elements != expected_capacity or (
                aligned is not None and aligned != expected_capacity * group_size
            ):
                return "unexpected_deepspeed_size"
        elif aligned is not None:
            if partition_elements == 0:
                return "empty_deepspeed_parameter" if aligned == 0 else "unexpected_deepspeed_size"
            group_size, remainder = divmod(aligned, partition_elements)
            if group_size < 1 or remainder or padding >= group_size:
                return "unexpected_deepspeed_size"
        elif 0 < partition_elements < full_elements:
            return "locally_partitioned"
        elif partition_elements != full_elements:
            return "unexpected_deepspeed_size"

        if full_elements == 0:
            return "empty_deepspeed_parameter"
        if group_size is None:
            return "incomplete_deepspeed_shard_metadata"
        if group_size > 1:
            return "locally_partitioned"
        return "full_size_deepspeed_managed"

    if row["is_quantized"]:
        return "quantized_without_shard_metadata"
    return "unmanaged_parameter"


def parameter_stats(model):
    """Aggregate tensor payload bytes separately for views and primary shards.

    Quantization metadata and allocator overhead are outside these totals.
    """
    details = parameter_partition_details(model)
    trainable = [row for row in details if row["requires_grad"]]
    frozen = [row for row in details if not row["requires_grad"]]

    def logical_total(rows):
        counts = [row["logical_elements"] for row in rows]
        return None if None in counts else sum(counts)

    return {
        "logical_parameters": logical_total(details),
        "known_logical_parameters": sum(
            row["logical_elements"] for row in details if row["logical_elements"] is not None
        ),
        "unknown_logical_parameter_tensors": sum(row["logical_elements"] is None for row in details),
        "local_elements": sum(r["local_elements"] for r in details),
        "local_parameter_bytes": sum(r["local_bytes"] for r in details),
        "shard_elements": sum(r["shard_elements"] for r in details if r["shard_elements"] is not None),
        "shard_bytes": sum(r["shard_bytes"] for r in details if r["shard_bytes"] is not None),
        "deepspeed_managed_tensors": sum(r["deepspeed_managed"] for r in details),
        "partitioned_tensors": sum(r["partition_state"] == "locally_partitioned" for r in details),
        "missing_shard_tensors": sum(r["partition_state"] == "missing_deepspeed_shard" for r in details),
        "quantized_tensors": sum(r["is_quantized"] for r in details),
        "frozen_parameters": logical_total(frozen),
        "trainable_parameters": logical_total(trainable),
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


def create_run_directory(output_dir, arm):
    """Reserve an isolated results directory on the ranks' shared filesystem."""
    root = Path(output_dir)
    if not root.is_absolute():
        root = LAB_DIR / root
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{arm}-", dir=root.resolve()))


def save_rank_results(stats, run_dir):
    """Publish one rank's measurements atomically inside its run directory."""
    path = run_dir / f"{stats['arm']}_rank{stats['rank']}.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")
    temporary.replace(path)
    return path


def mark_measurements_complete(run_dir, arm, world_size):
    """Publish a manifest only after every expected rank file exists."""
    rank_files = [f"{arm}_rank{rank}.json" for rank in range(world_size)]
    missing = [name for name in rank_files if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot mark measurements complete; missing rank files: {missing}")
    path = run_dir / "measurements_complete.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump({
            "run_id": run_dir.name, "arm": arm, "world_size": world_size, "rank_files": rank_files,
        }, file, indent=2)
        file.write("\n")
    temporary.replace(path)


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
    rank = distributed["rank"]
    run_directory = [str(create_run_directory(output_dir, arm)) if rank == 0 else None]
    torch.distributed.broadcast_object_list(run_directory, src=0)
    run_dir = Path(run_directory[0])

    device = distributed["local_rank"]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_bytes = torch.cuda.memory_allocated(device)

    model, hf_ds_config = load_model(arm, config)
    before_lora = {"measurement_stage": "after_model_load_before_lora", **parameter_stats(model)}
    model = attach_lora(model, arm)
    before_engine = {"measurement_stage": "after_lora_before_engine", **parameter_stats(model)}

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
    after_engine = {
        "measurement_stage": "after_deepspeed_engine_initialization",
        **parameter_stats(engine.module),
    }

    torch.cuda.synchronize(device)
    stats = {
        "run_id": run_dir.name,
        "arm": arm, "model": MODEL_ID, "zero_stage_configured": 3,
        "rank": rank, "world_size": distributed["world_size"],
        "local_rank": device, "rank_assignments": distributed["rank_assignments"],
        "gpu_name": torch.cuda.get_device_name(device),
        "before_lora": before_lora, "before_engine": before_engine, "after_engine": after_engine,
        "gpu_memory": {
            "start_measurement_stage": "before_model_load",
            "end_measurement_stage": "after_deepspeed_engine_initialization",
            "start_allocated_gpu_bytes": start_bytes,
            "allocated_gpu_bytes": torch.cuda.memory_allocated(device),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        },
        "versions": {name: version(name) for name in
                     ("transformers", "deepspeed", "torch", "bitsandbytes", "peft")},
    }
    # Keep HfDeepSpeedConfig alive until all measurements have been taken.
    del optimizer, engine, model, hf_ds_config
    path = save_rank_results(stats, run_dir)
    print(f"Saved {path}", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        mark_measurements_complete(run_dir, arm, distributed["world_size"])
        print(f"All rank measurements saved in {run_dir}", flush=True)
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
