"""Investigate DeepSpeed ZeRO-3 parameter partitioning.

Compares BF16 and NF4 model representations.

GitHub Issue:
https://github.com/yiqiao-yin/deepspeed-course/issues/124

GPU measurements have not yet been performed.
"""

import argparse
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-4B"

CONFIGS = {
    "bf16": "ds_config_bf16.json",
    "nf4": "ds_config_nf4.json",
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
    """Inspect logical and locally stored parameter counts."""

    logical_parameters = 0
    local_elements = 0
    partitioned_tensors = 0

    for parameter in model.parameters():

        logical_parameters += getattr(
            parameter,
            "ds_numel",
            parameter.numel()
        )

        local_elements += parameter.numel()

        if hasattr(parameter, "ds_id"):
            partitioned_tensors += 1

    return {
        "logical_parameters": logical_parameters,
        "local_elements": local_elements,
        "deepspeed_managed_tensors": partitioned_tensors,
    }


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

    raise NotImplementedError(
        "Distributed GPU model loading and parameter measurement "
        "will be implemented after validating the experiment design."
    )


if __name__ == "__main__":
    main()