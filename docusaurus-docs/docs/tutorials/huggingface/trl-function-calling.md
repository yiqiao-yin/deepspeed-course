---
sidebar_position: 2
---

# TRL Function Calling

Fine-tune Qwen3-0.6B for function calling using TRL's SFTTrainer with DeepSpeed.

## Overview

This example demonstrates:
- TRL SFTTrainer integration with DeepSpeed
- Function calling / tool use training
- ZeRO-2 optimization
- Multiple inference modes

**Model:** Qwen/Qwen3-0.6B
**Task:** Learning to call functions with proper arguments

## Quick Start

```bash
cd 05_huggingface_trl

# Training (2 GPUs)
deepspeed --num_gpus=2 train_trl_deepspeed.py

# Inference
python inference_trl_model.py --mode sample
```

## Training Data Format

The training data uses a chat format with tool definitions:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You have access to tools..."},
      {"role": "user", "content": "What's the weather in Tokyo?"},
      {"role": "assistant", "content": "{\"tool\": \"get_weather\", \"args\": {\"city\": \"Tokyo\"}}"}
    ]
  }
]
```

## DeepSpeed Configuration

```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2}
}
```

## Inference Modes

- **sample**: Generate multiple examples
- **single**: Process one query
- **interactive**: Chat interface

## Next Steps

- [OCR Vision-Language](/docs/tutorials/huggingface/ocr-vision-language) - Multimodal training
- [GRPO Training](/docs/tutorials/huggingface/grpo-training) - Reinforcement learning
