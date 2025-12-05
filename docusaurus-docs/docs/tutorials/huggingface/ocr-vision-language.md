---
sidebar_position: 3
---

# OCR Vision-Language

Fine-tune Qwen2-VL-2B for OCR and vision-language tasks with DeepSpeed.

## Overview

This example demonstrates:
- Vision-language model training
- Frame extraction from videos
- Multi-GPU DeepSpeed training
- Optimized for 2x RTX 4000-series GPUs

**Model:** Qwen2-VL-2B-Instruct
**Task:** Optical character recognition and image understanding

## Quick Start

```bash
cd 05_huggingface_ocr

# SLURM submission
sbatch submit_job.sh

# Direct execution
deepspeed --num_gpus=2 train_ds.py
```

## Hardware Requirements

| GPU | VRAM | Batch Size | Notes |
|-----|------|------------|-------|
| 2x RTX 4090 | 48 GB | 2 | Recommended |
| 2x RTX 4080 | 32 GB | 1 | With offloading |
| 1x A100 | 80 GB | 4 | Single GPU |

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
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 1
}
```

## Vision Processing

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Process image
inputs = processor(
    text="What text is in this image?",
    images=image,
    return_tensors="pt"
)
```

## Next Steps

- [GRPO Training](/docs/tutorials/huggingface/grpo-training) - Reinforcement learning
- [Hardware Guide](/docs/guides/hardware-requirements) - GPU selection
