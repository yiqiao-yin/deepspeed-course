---
sidebar_position: 5
---

# GPT-OSS Fine-tuning

Fine-tune GPT-OSS-20B on multilingual reasoning with LoRA and DeepSpeed.

## Overview

This example demonstrates:
- Large model training (20B parameters)
- LoRA parameter-efficient fine-tuning
- Hardware-specific configurations
- HuggingFace Hub integration

**Model:** GPT-OSS-20B
**Task:** Multilingual reasoning

## Quick Start

```bash
cd 07_huggingface_openai_gpt_oss_finetune_sft/lora

# For 4x A100/RTX 4090
deepspeed --num_gpus=4 train_ds.py

# For 2x RTX 3070 (8GB)
deepspeed --num_gpus=2 train_ds_mistral7b.py

# For H200/H100
deepspeed --num_gpus=8 train_ds_h200.py
```

## Hardware Configurations

### Consumer GPUs (8GB)

`train_ds_mistral7b.py` - Mistral-7B variant:

```python
# Optimized for 2x RTX 3070
model = "mistralai/Mistral-7B-v0.1"
lora_r = 8
batch_size = 1
gradient_accumulation = 16
```

### Professional GPUs

`train_ds.py` - Full GPT-OSS-20B:

```python
# For 4x A100 or RTX 4090
model = "openai/gpt-oss-20b"
lora_r = 16
batch_size = 2
gradient_accumulation = 4
```

### Datacenter GPUs

`train_ds_h200.py` - Maximum throughput:

```python
# For H200/H100/RTX 5090
model = "openai/gpt-oss-20b"
lora_r = 32
batch_size = 8
gradient_accumulation = 2
```

## DeepSpeed Configuration

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

## LoRA Settings

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
)
```

## SLURM Script

```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --mem=256G

source ~/myenv/bin/activate
deepspeed --num_gpus=4 train_ds.py
```

## Memory Guide

| Model | GPUs | Config | VRAM/GPU |
|-------|------|--------|----------|
| Mistral-7B | 2x 3070 | LoRA+Offload | 8 GB |
| GPT-OSS-20B | 4x 4090 | LoRA | 24 GB |
| GPT-OSS-20B | 4x A100 | LoRA | 40 GB |
| GPT-OSS-20B | 8x H100 | Full | 80 GB |

## Next Steps

- [Multi-Agent](/docs/tutorials/huggingface/multi-agent) - Ensemble training
- [Video Training](/docs/tutorials/multimodal/video-text-training) - Multimodal
