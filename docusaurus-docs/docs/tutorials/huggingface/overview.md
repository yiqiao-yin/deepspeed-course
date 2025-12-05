---
sidebar_position: 1
---

# HuggingFace Integration

Guide for fine-tuning large language models with DeepSpeed and HuggingFace.

## Overview

This section covers integrating DeepSpeed with the HuggingFace ecosystem:
- Transformers library
- TRL (Transformer Reinforcement Learning)
- Accelerate
- PEFT (Parameter-Efficient Fine-Tuning)

## Integration Options

### 1. HuggingFace Trainer + DeepSpeed

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # Enable DeepSpeed
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 2. Accelerate + DeepSpeed

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    deepspeed_plugin=deepspeed_plugin,
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

### 3. TRL + DeepSpeed

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        deepspeed="ds_config.json",
    ),
)
```

## DeepSpeed Configuration for LLMs

### ZeRO Stage 2 (Common)

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### ZeRO Stage 3 (Large Models)

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

## Model Size Guidelines

| Model Size | ZeRO Stage | GPU Memory | Recommended GPUs |
|------------|------------|------------|------------------|
| < 1B | Stage 2 | 8-16 GB | 1x RTX 3070/4070 |
| 1-7B | Stage 2 + Offload | 16-24 GB | 2x RTX 4090 |
| 7-20B | Stage 3 | 40-80 GB | 2-4x A100 |
| 20-70B | Stage 3 + Offload | 80+ GB | 4-8x A100/H100 |
| 70B+ | Stage 3 + Offload | 512+ GB | 8x H100/H200 |

## LoRA for Efficiency

Parameter-efficient fine-tuning with LoRA:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

Benefits:
- Trains only 0.1-1% of parameters
- Dramatically reduces memory usage
- Faster training iterations
- Easy model merging

## Example Workflow

### 1. Setup

```bash
# Install dependencies
uv pip install transformers accelerate trl peft deepspeed

# Clone example
cd 05_huggingface
```

### 2. Configure DeepSpeed

```json
// ds_config.json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2
}
```

### 3. Train

```bash
# With DeepSpeed launcher
deepspeed --num_gpus=2 train_ds.py

# With Accelerate
accelerate launch --config_file accelerate_config.yaml train.py
```

## Common Issues

### OOM with Large Models

1. Enable CPU offloading
2. Reduce batch size
3. Use gradient checkpointing
4. Switch to ZeRO Stage 3

### Slow Training

1. Disable unnecessary offloading
2. Enable communication overlap
3. Increase batch size if memory allows
4. Use faster GPUs

## Available Examples

| Example | Model | Task | Location |
|---------|-------|------|----------|
| [TRL Function Calling](/docs/tutorials/huggingface/trl-function-calling) | Qwen3-0.6B | Tool use | `05_huggingface_trl/` |
| [OCR Vision-Language](/docs/tutorials/huggingface/ocr-vision-language) | Qwen2-VL-2B | OCR | `05_huggingface_ocr/` |
| [GRPO Training](/docs/tutorials/huggingface/grpo-training) | Qwen-1.5B | Math reasoning | `06_huggingface_grpo/` |
| [GPT-OSS Fine-tuning](/docs/tutorials/huggingface/gpt-oss-finetuning) | GPT-OSS-20B | Multilingual | `07_huggingface_*/` |
| [Multi-Agent](/docs/tutorials/huggingface/multi-agent) | Qwen-1.5B | GRPO ensemble | `07_*_multi_agency/` |

## Next Steps

- [TRL Function Calling](/docs/tutorials/huggingface/trl-function-calling) - First HuggingFace example
- [Hardware Requirements](/docs/guides/hardware-requirements) - GPU selection guide
