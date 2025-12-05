---
sidebar_position: 3
---

# RunPod Setup

Guide for interactive DeepSpeed development on RunPod.

## Understanding RunPod

RunPod provides dedicated GPU pods:

```
You SSH directly into YOUR pod
    ↓
Pod has dedicated GPU(s)
    ↓
Run code immediately
```

**Key difference from CoreWeave**: No job scheduling - your GPUs are always available.

## Recommended Image

Use the PyTorch image for best compatibility:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

## Pod Configurations

### High-Performance (Multi-GPU)

Best for distributed training:

| Spec | Value |
|------|-------|
| GPUs | 8x H200 SXM |
| VRAM | 1128 GB |
| RAM | 2008 GB |
| vCPU | 224 |
| Cost | ~$30/hr |

### Cost-Effective (Development)

Best for prototyping:

| Spec | Value |
|------|-------|
| GPUs | 10x A40 |
| VRAM | 480 GB |
| RAM | 500 GB |
| vCPU | 90 |
| Cost | ~$4/hr |

## Getting Started

### 1. Create Pod

1. Go to RunPod dashboard
2. Select GPU type and count
3. Choose the PyTorch image
4. Set disk size (80 GB minimum)
5. Launch pod

### 2. Connect

```bash
# SSH (get command from dashboard)
ssh root@<pod-ip> -p <port>

# Or use web terminal
```

### 3. Set Up Environment

```bash
# Clone repository
git clone https://github.com/yiqiao-yin/deepspeed-course.git
cd deepspeed-course

# Create environment
uv venv myenv
source myenv/bin/activate

# Install dependencies
uv pip install torch deepspeed wandb
```

## Running Training

### Direct Execution

```bash
# Navigate to example
cd 01_basic_neuralnet

# Single GPU
deepspeed --num_gpus=1 train_ds.py

# Multi-GPU
deepspeed --num_gpus=4 train_ds.py
```

### With W&B

```bash
export WANDB_API_KEY="your_key"
deepspeed --num_gpus=2 train_ds.py
```

### Jupyter Lab

```bash
# Start Jupyter (usually pre-installed)
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# Access via exposed port in dashboard
```

## Comparison: RunPod vs CoreWeave

| Aspect | RunPod | CoreWeave |
|--------|--------|-----------|
| Access | Immediate | Queue-based |
| Billing | Pod lifetime | Compute time |
| GPUs | Dedicated | Shared cluster |
| Best for | Development | Production |
| Learning curve | Low | Medium |

## Tips

### Persistent Storage

Use the `/workspace` directory for persistent files:

```bash
cd /workspace
git clone ...
```

### Stop vs Terminate

- **Stop**: Keeps data, pauses billing
- **Terminate**: Deletes everything

### Cost Management

1. Stop pods when not in use
2. Use smaller GPUs for development
3. Scale up only for final training

## Troubleshooting

### CUDA Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or use ZeRO-3
```

### Pod Won't Start

- Check GPU availability in region
- Try different GPU type
- Reduce disk size request

## Next Steps

- [Quick Start](/docs/getting-started/quick-start) - First training run
- [Hardware Guide](/docs/guides/hardware-requirements) - GPU selection
