---
sidebar_position: 1
---

# SLURM Deployment

Guide for running DeepSpeed training on SLURM-based HPC clusters.

## Overview

SLURM (Simple Linux Utility for Resource Management) is used by HPC clusters like CoreWeave for job scheduling. This guide covers:
- SLURM basics
- Job submission
- Resource management
- Monitoring

## Basic SLURM Workflow

```bash
# 1. Submit job
sbatch run_deepspeed.sh

# 2. Monitor queue
squeue -u $USER

# 3. Check output
tail -f logs/training_*.out

# 4. Cancel if needed
scancel <job_id>
```

## SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed_train
#SBATCH --gres=gpu:2              # Number of GPUs
#SBATCH --partition=h200-low      # Partition name
#SBATCH --time=04:00:00           # Max runtime
#SBATCH --mem=64G                 # Memory
#SBATCH --cpus-per-task=16        # CPU cores
#SBATCH --output=logs/%x_%j.out   # Output file
#SBATCH --error=logs/%x_%j.err    # Error file

# Create logs directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Activate environment
source ~/myenv/bin/activate

# Set W&B key (optional)
export WANDB_API_KEY="your_key"

# Run training
deepspeed --num_gpus=2 train_ds.py

echo "End: $(date)"
```

## Common SLURM Commands

### Job Management

```bash
# Submit job
sbatch script.sh

# View your jobs
squeue -u $USER

# Detailed job info
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Job Information

```bash
# Job details
scontrol show job <job_id>

# Why is job pending?
squeue -j <job_id> --start

# Job history
sacct -u $USER

# Detailed history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

### Output Files

```bash
# List output files
ls -lt slurm-*.out

# View latest output
cat slurm-$(ls -t slurm-*.out | head -1)

# Follow output in real-time
tail -f slurm-<job_id>.out

# Search for errors
grep -i error slurm-<job_id>.out
```

## Resource Specifications

### GPU Options

```bash
#SBATCH --gres=gpu:1          # 1 GPU (any type)
#SBATCH --gres=gpu:a100:4     # 4 A100 GPUs
#SBATCH --gres=gpu:h100:8     # 8 H100 GPUs
```

### Memory

```bash
#SBATCH --mem=64G             # Total memory
#SBATCH --mem-per-cpu=4G      # Per CPU core
#SBATCH --mem-per-gpu=32G     # Per GPU
```

### Time Limits

```bash
#SBATCH --time=00:30:00       # 30 minutes
#SBATCH --time=04:00:00       # 4 hours
#SBATCH --time=1-00:00:00     # 1 day
```

## Interactive Sessions

For debugging or development:

```bash
# Request interactive GPU session
srun --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash

# Now you're on a compute node with GPU access
nvidia-smi
python train.py
```

## Multi-Node Training

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64

# Get master address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Launch with DeepSpeed
deepspeed --num_gpus=8 \
          --num_nodes=2 \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          train_ds.py
```

## GPU Monitoring

Create a monitoring job:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu_monitor

while true; do
    nvidia-smi
    echo "---"
    sleep 1
done
```

Key metrics to watch:
- **GPU Util**: Should be 90-100% during training
- **Memory**: Watch for OOM
- **Temperature**: Should stay under 85C
- **Power**: Indicates GPU load

## Troubleshooting

### Job Stuck in Queue

```bash
# Check why
squeue -j <job_id> --start

# Common reasons:
# - Resources unavailable
# - Priority queue
# - Maintenance
```

### Job Failed

```bash
# Check exit code
sacct -j <job_id> --format=ExitCode

# View error log
cat slurm-<job_id>.err
```

### GPU Not Detected

```bash
# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Check nvidia-smi
nvidia-smi

# Verify in Python
python -c "import torch; print(torch.cuda.device_count())"
```

## Best Practices

1. **Test locally first**: Run small tests before big jobs
2. **Use checkpointing**: Save progress regularly
3. **Request appropriate resources**: Don't over-request
4. **Monitor actively**: Check jobs early
5. **Clean up**: Remove old output files

## Next Steps

- [CoreWeave Setup](/docs/guides/coreweave-setup) - CoreWeave-specific guide
- [RunPod Setup](/docs/guides/runpod-setup) - Interactive alternative
