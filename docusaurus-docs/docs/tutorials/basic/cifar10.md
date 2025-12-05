---
sidebar_position: 3
---

# CIFAR-10 CNN

Train a CNN on CIFAR-10 dataset achieving 81% accuracy with DeepSpeed.

## Overview

This example demonstrates:
- Training on real image data (CIFAR-10)
- Multi-GPU setup with SLURM
- SGD optimizer with BatchNorm
- Production-ready training pipeline

**Achievement:** 81% accuracy on CIFAR-10 test set

## Quick Start

```bash
cd 02_basic_convnet_cifar10_examples

# SLURM submission
sbatch run_deepspeed.sh

# Direct execution (2 GPUs)
deepspeed --num_gpus=2 cifar10_deepspeed.py
```

## Dataset

CIFAR-10 contains:
- 60,000 32x32 color images
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images
- 10,000 test images

## DeepSpeed Configuration

```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 64,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "SGD",
    "params": {
      "lr": 0.01,
      "momentum": 0.9,
      "weight_decay": 5e-4
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

## Key Improvements

### SGD with Momentum

SGD often outperforms Adam for image classification:

```json
{
  "optimizer": {
    "type": "SGD",
    "params": {
      "lr": 0.01,
      "momentum": 0.9,
      "weight_decay": 5e-4
    }
  }
}
```

### Batch Normalization

Stabilizes training and allows higher learning rates:

```python
self.bn1 = nn.BatchNorm2d(32)
self.bn2 = nn.BatchNorm2d(64)
```

### Data Augmentation

Improves generalization:

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
```

## SLURM Configuration

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=cifar10

source ~/myenv/bin/activate
deepspeed --num_gpus=2 cifar10_deepspeed.py
```

## Expected Results

| Metric | Value |
|--------|-------|
| Final Accuracy | ~81% |
| Training Time | ~30 min (2 GPUs) |
| Epochs | 100 |
| Best Epoch | ~80-90 |

## Model Improvement Strategies

See `MODEL_IMPROVEMENT_STRATEGY.md` for detailed techniques:

1. **Architecture improvements**
   - Deeper networks (ResNet-style)
   - Skip connections
   - Dropout regularization

2. **Training improvements**
   - Learning rate scheduling
   - Label smoothing
   - Mixup augmentation

3. **Data improvements**
   - More aggressive augmentation
   - Cutout regularization
   - AutoAugment

## Troubleshooting

### Low Accuracy

- Increase training epochs
- Adjust learning rate schedule
- Add more data augmentation

### Slow Training

- Enable FP16
- Use ZeRO Stage 2
- Increase batch size

## Next Steps

- [Basic RNN](/docs/tutorials/basic/rnn) - Time series prediction
- [Bayesian Neural Networks](/docs/tutorials/intermediate/bayesian-nn) - Uncertainty estimation
