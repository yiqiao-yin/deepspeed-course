---
sidebar_position: 2
---

# Video-Speech Training

Fine-tune LongCat-Flash-Omni for video-speech-to-speech tasks.

## Overview

This example demonstrates:
- Multimodal (video + audio) model training
- MoE (Mixture of Experts) architecture
- LoRA fine-tuning for efficiency
- ZeRO-3 with aggressive CPU offloading

**Model:** LongCat-Flash-Omni (560B params, 27B activated)
**Task:** Video + input audio → output audio

## Quick Start

```bash
cd 09_vss

# Standard setup (8+ GPUs)
deepspeed --num_gpus=8 train_ds.py

# 2x B200 setup
./run_2xB200.sh
```

## Hardware Requirements

### Minimum (8x H100)

| Resource | Requirement |
|----------|-------------|
| GPUs | 8x H100 (80GB each) |
| System RAM | 512 GB |
| Storage | 1.1 TB for weights |
| Network | 400 Gbps InfiniBand |

### 2x B200 Configuration

Optimized for smaller setups:

```bash
# Use conservative config
./run_2xB200.sh
```

## Data Format

```
data/train/
├── 01/
│   ├── in.mp4      # Input video
│   ├── in.wav      # Input audio
│   └── out.wav     # Target output audio
├── 02/
│   ├── in.mp4
│   ├── in.wav
│   └── out.wav
└── ...
```

## DeepSpeed Configuration

### Standard (8+ GPUs)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### 2x B200 (Aggressive Offload)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 4
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5
    },
    "sub_group_size": 1e8
  }
}
```

## LoRA Configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "audio_encoder", "audio_decoder"
    ],
    lora_dropout=0.05,
)
```

## Storage Check

Before training, verify storage:

```bash
./check_storage.sh
```

Expected output:
```
Model weights: 1.1 TB
Available space: 2.0 TB
Status: OK
```

## Memory Analysis (2x B200)

| Component | Memory |
|-----------|--------|
| Model (activated) | 54 GB |
| Optimizer states | 108 GB |
| Gradients | 54 GB |
| Activations | 20 GB |
| **Total** | 236 GB |
| **Per GPU** | 118 GB |

With B200's 192 GB VRAM, this leaves headroom for batch processing.

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| LoRA rank | 8 |
| Batch size | 1 per GPU |
| Gradient accum | 16 |
| Warmup steps | 100 |

## Troubleshooting

### OOM Errors

1. Reduce batch size
2. Increase offloading
3. Lower LoRA rank

### Slow Training

1. Check NVLink/InfiniBand
2. Reduce CPU offloading if RAM allows
3. Use larger sub_group_size

## Next Steps

- [Hardware Requirements](/docs/guides/hardware-requirements) - Detailed GPU guide
- [Troubleshooting](/docs/reference/troubleshooting) - Common issues
