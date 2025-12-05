---
sidebar_position: 4
---

# GRPO Training

Memory-efficient GRPO (Group Relative Policy Optimization) fine-tuning for mathematical reasoning.

## Overview

This example demonstrates:
- GRPO algorithm implementation
- LoRA for parameter efficiency
- ZeRO-2 with CPU offloading
- Training on consumer GPUs (8GB+)

**Model:** Qwen-1.5B (distilled)
**Dataset:** GSM8K (8K math reasoning samples)
**Target:** 8GB GPU support

## Quick Start

```bash
cd 06_huggingface_grpo

# SLURM submission
sbatch run_deepspeed.sh

# Direct execution
deepspeed --num_gpus=1 grpo_gsm8k_train.py
```

## What is GRPO?

Group Relative Policy Optimization improves over standard RLHF:
- Groups multiple responses per prompt
- Computes relative rewards within groups
- More stable training than PPO

```python
# GRPO computes rewards relative to group mean
group_rewards = compute_rewards(responses)  # [batch, num_generations]
relative_rewards = group_rewards - group_rewards.mean(dim=1, keepdim=True)
```

## LoRA Configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Benefits:
- Only trains ~1% of parameters
- Fits on 8GB GPUs
- Fast iterations

## DeepSpeed Configuration

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 1
}
```

## GSM8K Dataset

Mathematical word problems:

```
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends. How many eggs does
she have left?

A: Janet starts with 16 eggs. She eats 3. So she has 16 - 3 = 13.
#### 13
```

## Memory Requirements

| Configuration | GPU Memory | System RAM |
|--------------|------------|------------|
| Full model | 24 GB | 32 GB |
| LoRA + ZeRO-2 | 12 GB | 32 GB |
| LoRA + Offload | 8 GB | 64 GB |

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Base LR | 5e-5 |
| LoRA rank | 16 |
| Batch size | 1 per GPU |
| Gradient accum | 8 |
| Epochs | 3 |

## Next Steps

- [GPT-OSS Fine-tuning](/docs/tutorials/huggingface/gpt-oss-finetuning) - Larger models
- [Multi-Agent](/docs/tutorials/huggingface/multi-agent) - Ensemble learning
