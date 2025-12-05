---
sidebar_position: 2
---

# CoreWeave Setup

Guide for using CoreWeave's HPC cluster for DeepSpeed training.

## Understanding CoreWeave

CoreWeave is a shared multi-user HPC cluster:

```
Login Nodes (where you SSH)
    ↓
SLURM Scheduler (resource manager)
    ↓
Compute Nodes (where jobs run)
    ↓
Your GPU workload
```

**Key point**: When you SSH in, you're on a login node without GPUs. You must submit jobs to access GPUs.

## Getting Started

### 1. SSH Access

```bash
ssh username@coreweave.cloud
```

### 2. Check Available Partitions

```bash
sinfo

# Example output:
# PARTITION    AVAIL  TIMELIMIT  NODES  STATE
# h200-low*    up     4:00:00    50     idle
# a100-high    up     24:00:00   20     idle
```

### 3. Set Up Environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv ~/myenv
source ~/myenv/bin/activate

# Install dependencies
uv pip install torch deepspeed wandb
```

## Submitting Jobs

### Basic Job

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=h200-low
#SBATCH --time=02:00:00
#SBATCH --job-name=deepspeed

source ~/myenv/bin/activate
deepspeed --num_gpus=2 train_ds.py
```

### With W&B Tracking

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=h200-low
#SBATCH --time=02:00:00

export WANDB_API_KEY="your_key"

source ~/myenv/bin/activate
deepspeed --num_gpus=2 train_ds.py
```

## Available GPU Types

| Partition | GPU | VRAM | Max Time |
|-----------|-----|------|----------|
| h200-low | H200 | 141 GB | 4 hours |
| h100-low | H100 | 80 GB | 4 hours |
| a100-high | A100 | 80 GB | 24 hours |
| rtx4090-low | RTX 4090 | 24 GB | 4 hours |

## Interactive Development

For debugging or short experiments:

```bash
# Request interactive session
srun --gres=gpu:1 --partition=h200-low --time=01:00:00 --pty bash

# Now you have GPU access
nvidia-smi
python train.py
```

## Best Practices

1. **Use low-priority partitions** for development
2. **Checkpoint frequently** (4-hour time limits)
3. **Use W&B** for remote monitoring
4. **Test small** before scaling up

## Cost Optimization

- **Pay for compute time only**: No idle charges
- **Use spot instances**: Lower priority, lower cost
- **Batch jobs**: Submit overnight for less contention

## Next Steps

- [SLURM Deployment](/docs/guides/slurm-deployment) - General SLURM guide
- [RunPod Setup](/docs/guides/runpod-setup) - Interactive alternative
