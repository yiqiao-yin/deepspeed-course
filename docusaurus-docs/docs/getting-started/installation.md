---
sidebar_position: 1
---

# Installation

Set up your environment for DeepSpeed training.

## Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch 2.0+

## Using uv (Recommended)

The modern `uv` package manager provides 10-100x faster dependency resolution:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv myenv
source myenv/bin/activate

# Install core dependencies
uv pip install torch deepspeed

# Install additional packages
uv pip install numpy pandas matplotlib wandb yfinance scikit-learn
```

## Using pip

Traditional pip installation:

```bash
# Create virtual environment
python -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install torch deepspeed
pip install numpy pandas matplotlib wandb
```

## Verify Installation

Test your setup:

```python
import torch
import deepspeed

print(f"PyTorch version: {torch.__version__}")
print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Clone the Repository

```bash
git clone https://github.com/yiqiao-yin/deepspeed-course.git
cd deepspeed-course
```

## Platform-Specific Setup

### SLURM Clusters (CoreWeave)

On SLURM clusters, activate your environment in batch scripts:

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=h200-low
#SBATCH --time=02:00:00

source ~/myenv/bin/activate
deepspeed --num_gpus=2 train.py
```

### RunPod

RunPod pods typically have PyTorch pre-installed. Use the recommended image:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

Then install DeepSpeed:

```bash
pip install deepspeed wandb
```

## Optional: Weights & Biases

For experiment tracking:

```bash
pip install wandb
wandb login  # Enter your API key from https://wandb.ai/authorize
```

## Next Steps

- [Quick Start](/docs/getting-started/quick-start) - Run your first training
- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Learn about memory optimization
