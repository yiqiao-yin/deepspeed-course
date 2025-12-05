---
sidebar_position: 1
---

# Video-Text Training

Train vision-language models for video understanding with DeepSpeed.

## Overview

This example provides two frameworks:
1. **LLaVA Video Trainer** - Vision-language video understanding (7B params)
2. **Seq2Seq Video Trainer** - Text-to-text video metadata (600M params)

## Quick Start

```bash
cd 08_vtt/hf_ds_vtt_test2

# LLaVA (vision-language)
cd llava_video_trainer
./run_training.sh

# Seq2Seq (text-only)
cd seq2seq_video_trainer
./run_training.sh
```

## Framework Comparison

| Feature | LLaVA | Seq2Seq |
|---------|-------|---------|
| Model | LLaVA 7B | NLLB 600M |
| Input | Video frames + text | Text only |
| Output | Text | Text |
| VRAM | 40+ GB | 8 GB |
| DeepSpeed Config | Auto-generated | External file |

## LLaVA Video Trainer

### Features
- Extracts and processes video frames
- Vision encoder + language model
- Automatic DeepSpeed config generation
- W&B tracking support

### Usage

```bash
cd llava_video_trainer

# Training
python video_training_script.py \
    --video_dir /path/to/videos \
    --num_gpus 4

# With W&B
export WANDB_API_KEY="your_key"
./run_training.sh
```

### Hardware Requirements

- Minimum: 2x RTX 4090 (48 GB total)
- Recommended: 4x A100 (160 GB total)

## Seq2Seq Video Trainer

### Features
- Text-only generation from video metadata
- External DeepSpeed configuration
- Lower resource requirements

### Usage

```bash
cd seq2seq_video_trainer

# Training
deepspeed --num_gpus=2 video_text_trainer.py \
    --deepspeed_config ds_config.json
```

### DeepSpeed Config

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2
  },
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2
}
```

## Data Format

### Video Input

```
data/
├── video_001.mp4
├── video_001.json  # metadata
├── video_002.mp4
└── video_002.json
```

### Metadata Format

```json
{
  "video_id": "video_001",
  "description": "A cat playing with a ball",
  "duration": 10.5,
  "frames": 300
}
```

## Next Steps

- [Video Speech Training](/docs/tutorials/multimodal/video-speech-training) - Audio+Video
- [Hardware Guide](/docs/guides/hardware-requirements) - GPU selection
