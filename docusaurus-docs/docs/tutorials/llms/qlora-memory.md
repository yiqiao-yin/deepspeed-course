---
sidebar_position: 19
---

# QLoRA vs LoRA: Measuring the Frozen Base

The companion lab in `03_llms/13_qlora/` trains the same Qwen3-4B adapters
twice: once with bf16 base weights, once with NF4 four-bit base weights.
It asks an infrastructure question: how much **allocated GPU memory per
rank** does each path use? The example does not test model quality.

## Why ZeRO-2 for both?

ZeRO-2 partitions optimizer state and gradients, while each GPU holds a
copy of the base parameters. The adapters and input stay the same between
arms. In the pinned Transformers version, quantized models may bypass the
ZeRO-3 parameter-partition initialization path. Comparing quantized ZeRO-3
to ordinary ZeRO-3 could silently compare different partition strategies.

With about 4.02 billion base parameters, bf16 storage is roughly 8.04 GB
and ideal 4-bit storage roughly 2.01 GB. Neither number is a VRAM
prediction: scales, embeddings, activations, CUDA allocation and optimizer
state require additional space. Budget each GPU separately.

## Try the comparison

With a suitable GPU and from inside the lab folder:

```bash
uv sync
uv run --locked deepspeed --num_gpus=1 train_qlora.py --max-steps 20
```

The 20-step option caps **each arm**. `comparison.json` records peak
allocated bytes, actual stored tensor bytes, logical and trainable parameter
counts, and the first/last losses for each rank. The peak counter resets
before each load. Packed NF4 parameters use their original quantization
shape for logical counts; stored bytes also include quantization state.
The run rejects identical representations or adapters and rejects an NF4
memory peak that is not smaller on a rank. For the SLURM and RunPod commands,
see the lab [README](https://github.com/yiqiao-yin/deepspeed-course/tree/main/03_llms/13_qlora).

Without a GPU, `--plan` shows only the ideal weight arithmetic. The offline
test checks the comparison invariants:

```bash
uv run --no-project python 03_llms/13_qlora/train_qlora.py --plan
uv run tests/test_qlora.py
```

## Measured GPU results

Five single-GPU benchmark runs were completed on October 26, 2026:
three using an NVIDIA RTX 3090 and two using an RTX 4090. Each run
trained both configurations for 40 optimizer steps.

Across all five runs, the measured peak PyTorch-allocated GPU memory was
8.19 GiB for LoRA and 3.94 GiB for QLoRA, representing a **51.86% reduction**
in peak allocated memory.

Training-loop durations varied between runs. These measurements demonstrate
memory savings under the tested configuration but do not establish
equivalent model quality, generalization, or a statistically reliable
training-speed advantage.

The companion lab README contains the individual benchmark results,
original execution logs, JSON measurements, and statistical analysis.
