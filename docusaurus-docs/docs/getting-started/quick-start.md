---
sidebar_position: 2
---

# Quick Start

Get up and running with DeepSpeed in minutes.

## Hello World with DeepSpeed

Let's verify your setup with a simple GPU computation:

### Step 1: Create Test Script

```python
# hello.py
import torch

print("Hello from Python!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Simple GPU computation test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiplication test passed!")
    print(f"Result shape: {z.shape}")
```

### Step 2: Run Directly (RunPod/Interactive)

```bash
python hello.py
```

### Step 3: Run via SLURM (CoreWeave)

Create a batch script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=h200-low
#SBATCH --time=00:10:00
#SBATCH --job-name=hello_world

source ~/myenv/bin/activate
python hello.py
```

Submit and monitor:

```bash
sbatch run_hello.sh
squeue -u $USER
cat slurm-*.out
```

## Your First DeepSpeed Training

Navigate to the basic neural network example:

```bash
cd 01_basic_neuralnet
```

### Direct Execution (RunPod)

```bash
# Single GPU
deepspeed --num_gpus=1 train_ds.py

# Multiple GPUs
deepspeed --num_gpus=2 train_ds.py
```

### SLURM Submission (CoreWeave)

```bash
# Submit the pre-configured job
sbatch run_deepspeed.sh

# Monitor progress
squeue -u $USER
tail -f logs/basic_nn_*.out
```

## Understanding the Output

A successful run shows:

```
[INFO] DeepSpeed configuration:
  - ZeRO stage: 2
  - FP16 enabled: True
  - Batch size: 32

Epoch 1/100:
  Loss: 0.1234
  Learning rate: 0.001

Epoch 2/100:
  Loss: 0.0567
  ...

Training complete!
  Final loss: 0.0012
  Parameters converged: ✓
```

## Key DeepSpeed Commands

```bash
# Single GPU training
deepspeed --num_gpus=1 train.py

# Multi-GPU training (same node)
deepspeed --num_gpus=4 train.py

# With custom config
deepspeed --num_gpus=2 train.py --deepspeed_config ds_config.json

# Multi-node (requires hostfile)
deepspeed --hostfile=hostfile.txt train.py
```

## Common DeepSpeed Configuration

Each example includes a `ds_config.json`:

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

## Next Steps

- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Understand memory optimization
- [Basic Neural Network](/docs/tutorials/basic/neural-network) - First complete example
- [SLURM Deployment](/docs/guides/slurm-deployment) - Production cluster setup
