---
sidebar_position: 2
---

# Basic ConvNet

Train an enhanced CNN for image classification using DeepSpeed on synthetic MNIST-like data.

## Overview

This example demonstrates:
- CNN architecture with DeepSpeed
- Kaiming/He weight initialization
- Learning rate scheduling (warmup + cosine decay)
- Early stopping and gradient monitoring
- Real-time accuracy tracking

**Task:** 10-class classification on 28x28 grayscale images

## Quick Start

```bash
cd 02_basic_convnet

# Single GPU
deepspeed --num_gpus=1 train_ds.py

# Multi-GPU
deepspeed --num_gpus=2 train_ds.py
```

## Model Architecture

```python
class CNNModelEnhanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self._initialize_weights()  # Kaiming initialization
```

**Architecture Flow:**
```
Input [1, 28, 28]
    ↓ Conv1 + ReLU + Pool
[16, 14, 14]
    ↓ Conv2 + ReLU + Pool
[32, 7, 7]
    ↓ Flatten
[1568]
    ↓ FC1 + ReLU
[128]
    ↓ FC2
[10] (classes)
```

## Training Enhancements

### Kaiming Initialization

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

Benefits:
- Designed for ReLU activations
- Prevents vanishing/exploding gradients
- Faster convergence

### Learning Rate Schedule

```python
def get_lr_schedule(epoch, initial_lr=0.001, warmup_epochs=5, total_epochs=50):
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr * 0.5 * (1 + cos(progress * pi))
```

### Early Stopping

```python
patience_limit = 15
min_improvement = 1e-5

if avg_loss < best_loss - min_improvement:
    best_loss = avg_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience_limit:
        break  # Stop training
```

## DeepSpeed Configuration

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 (initial) |
| LR Schedule | Warmup + Cosine |
| Warmup Epochs | 5 |
| Total Epochs | 50 |
| Early Stopping | 15 epochs patience |
| Batch Size | 32 |
| Parameters | ~208,000 |

## Gradient Monitoring

The script tracks gradient norms:

```python
total_norm = 0.0
for p in model_engine.module.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
```

**Healthy patterns:**
- Gradual decrease and stabilization
- Values typically 0.01 - 1.0

**Problem indicators:**
- Sudden spikes: gradient explosion
- Near zero: vanishing gradients

## Expected Output

```
Epoch 49 Summary:
  - Avg Loss: 2.145678
  - Accuracy: 15.75%
  - Avg Grad Norm: 0.118765

Note: With synthetic random data, expect "Poor" quality.
With real MNIST, expect 95-99% accuracy.
```

## Using Real MNIST

Replace synthetic data with actual MNIST:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform
)
```

## Next Steps

- [CIFAR-10 CNN](/docs/tutorials/basic/cifar10) - Real dataset example
- [Basic RNN](/docs/tutorials/basic/rnn) - Time series prediction
