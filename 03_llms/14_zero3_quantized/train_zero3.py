"""Investigate DeepSpeed ZeRO-3 parameter partitioning.

Compares BF16 and NF4 model representations.

GitHub Issue:
https://github.com/yiqiao-yin/deepspeed-course/issues/124

GPU measurements have not yet been performed.
"""

import os
import argparse
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-4B"

CONFIGS = {
    "bf16": "ds_config_bf16.json",
    "nf4": "ds_config_nf4.json",
}

def initialize_distributed():
    """Initialize DeepSpeed and require at least two GPU ranks."""

    import torch
    import deepspeed

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for ZeRO-3 parameter measurements."
        )

    deepspeed.init_distributed()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if world_size < 2:
        raise RuntimeError(
            "This experiment requires at least two GPU ranks."
            f"Detected world size: {world_size}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": f"cuda:{local_rank}",
    }


def load_config(arm):
    """Load and validate a DeepSpeed configuration."""

    config_path = Path(__file__).resolve().parent / CONFIGS[arm]

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    stage = config.get("zero_optimization", {}).get("stage")

    if stage != 3:
        raise ValueError(
            f"Expected ZeRO-3, but configuration specifies stage {stage}"
        )

    return config


def parameter_stats(model):
    """Measure logical size, local storage, and ZeRO-3 metadata.
    
    These are observations, not proof of cross-rank replication.
    Actual distribution must be checked across multiple GPU ranks.
    """
    stats = {
        "logical_parameters": 0,
        "local_elements": 0,
        "deepspeed_managed_tensors": 0,
        "partitioned_tensors": 0,
        "quantized_tensors": 0,
        "local_parameter_bytes": 0,
    }

    for name, param in model.named_parameters():
        local_count = param.numel()
        logical_count = getattr(param, "ds_numel", local_count)

        # NF4 stores weights in a packed format. Recover the original
        # dimensions where quantization metadata is available.
        quant_state = getattr(param, "quant_state", None)

        if quant_state is not None:
            stats["quantized_tensors"] += 1

            original_shape = getattr(quant_state, "shape", None)
            if original_shape is not None:
                logical_count = 1
                for dimension in original_shape:
                    logical_count *= dimension

        stats["logical_parameters"] += logical_count
        stats["local_elements"] += local_count
        stats["local_parameter_bytes"] += (
            local_count * param.element_size()
        )

        if hasattr(param, "ds_id"):
            stats["deepspeed_managed_tensors"] += 1

            if local_count < logical_count:
                stats["partitioned_tensors"] += 1

    return stats

def parameter_partition_details(model):
    """Collect parameter-level evidence of ZeRO-3 partitioning.

    Logical sizes and local storage are reported separately.
    This function does not claim cross-rank replication.
    """
    details = []

    for name, param in model.named_parameters():
        quant_state = getattr(param, "quant_state", None)

        logical_count = getattr(param, "ds_numel", param.numel())

        if quant_state is not None:
            original_shape = getattr(quant_state, "shape", None)

            if original_shape is not None:
                logical_count = 1
                for dimension in original_shape:
                    logical_count *= dimension

        ds_id = getattr(param, "ds_id", None)
        ds_status = getattr(param, "ds_status", None)

        details.append({
            "name": name,
            "logical_elements": logical_count,
            "local_elements": param.numel(),
            "local_bytes": param.numel() * param.element_size(),
            "requires_grad": bool(param.requires_grad),
            "is_quantized": quant_state is not None,
            "deepspeed_managed": ds_id is not None,
            "ds_id": ds_id,
            "ds_status": str(ds_status) if ds_status is not None else None,
        })

    return details

def load_model(arm, config):
    """Load Qwen3-4B with ZeRO-3 registered before model construction.

    The BF16 and NF4 arms use the same DeepSpeed stage.
    Quantization is controlled by the model-loading arguments.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.integrations import HfDeepSpeedConfig

    # Register ZeRO-3 before calling from_pretrained().
    # Keep this object alive throughout model loading.
    ds_config = HfDeepSpeedConfig(config)

    kwargs = {
        "dtype": torch.bfloat16,
    }

    if arm == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **kwargs,
    )

    model.config.use_cache = False

    return model, ds_config

def measure_model_loading(arm, config):
    """Load a model and collect preliminary per-rank measurements."""

    import torch

    distributed = initialize_distributed()

    rank = distributed["rank"]
    local_rank = distributed["local_rank"]

    model, ds_config = load_model(arm, config)

    # Keep the DeepSpeed configuration alive during measurement.
    assert ds_config is not None

    stats = parameter_stats(model)

    # Capture parameter-level evidence before DeepSpeed engine initialization.
    details = parameter_partition_details(model)

    stats["parameter_details"] = details
    stats["frozen_parameters"] = sum(
        row["logical_elements"]
        for row in details
        if not row["requires_grad"]
    )
    stats["trainable_parameters"] = sum(
        row["logical_elements"]
        for row in details
        if row["requires_grad"]
    )
    stats["managed_frozen_tensors"] = sum(
        1 for row in details
        if row["deepspeed_managed"] and not row["requires_grad"]
    )

    torch.cuda.synchronize(local_rank)

    stats.update({
        "arm": arm,
        "rank": rank,
        "world_size": distributed["world_size"],
        "zero_stage": 3,
        "allocated_gpu_bytes": torch.cuda.memory_allocated(local_rank),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(local_rank),
        "measurement_stage": "after_model_loading",
    })

    print(
        f"Rank {rank} | {arm} | "
        f"Logical parameters: {stats['logical_parameters']:,} | "
        f"Local elements: {stats['local_elements']:,} | "
        f"Partitioned tensors: {stats['partitioned_tensors']}",
        flush=True,
    )

    return model, stats, ds_config

def save_rank_results(stats, output_dir="results"):
    """Save measurements from one GPU rank as JSON.

    Each rank writes to a separate file to avoid overwriting results.
    """

    output_path = Path(__file__).resolve().parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    arm = stats["arm"]
    rank = stats["rank"]

    filename = output_path / f"{arm}_rank{rank}.json"

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(
        f"Saved {arm} measurements for rank {rank}: {filename}",
        flush=True,
    )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arm",
        choices=["bf16", "nf4"],
        required=True
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate configuration without loading the model."
    )

    args = parser.parse_args()

    config = load_config(args.arm)

    if args.plan:

        print("ZeRO-3 experiment configuration")
        print("--------------------------------")
        print(f"Model: {MODEL_ID}")
        print(f"Representation: {args.arm}")
        print(f"ZeRO stage: {config['zero_optimization']['stage']}")
        print("GPU measurements: NOT PERFORMED")
        return

    # GPU execution path: requires a distributed Linux environment.
    model, stats, ds_config = measure_model_loading(args.arm, config)

    save_rank_results(stats)

    print(
        f"Rank {stats['rank']}: "
        f"model-loading measurement completed.",
        flush=True,
    )

    # Preserve the model and DeepSpeed configuration until measurement ends.
    del model, ds_config


if __name__ == "__main__":
    main()