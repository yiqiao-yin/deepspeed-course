---
sidebar_position: 1
---

# Basic Neural Network

Train a simple linear regression model using DeepSpeed for distributed training.

## Overview

This example demonstrates:
- DeepSpeed initialization and training loop
- FP16 mixed precision training
- Multi-GPU distributed training
- Optional W&B experiment tracking

**Model:** Simple linear regression learning `y = 2x + 1`

## Quick Start

```bash
cd 01_basic_neuralnet

# Single GPU
deepspeed --num_gpus=1 train_ds.py

# Multi-GPU
deepspeed --num_gpus=2 train_ds.py
```

## Model Architecture

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
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
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Epochs | 30 |
| Batch Size | 32 |
| Loss Function | MSE |
| Precision | FP16 |

## Expected Output

```
Epoch 29/30 Summary: Avg Loss = 0.000123
  Learned Weight: 1.999876
  Learned Bias: 1.000234

Parameter Estimation Errors:
  Weight Error: 0.000124 (0.01%)
  Bias Error: 0.000234 (0.02%)

Model Quality: Excellent!
```

## Key Concepts

### DeepSpeed Initialization

```python
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model_engine(inputs)
        loss = criterion(outputs, targets)

        model_engine.backward(loss)
        model_engine.step()
```

## Optional: Weights & Biases

Enable experiment tracking:

```bash
export WANDB_API_KEY="your_api_key"
deepspeed --num_gpus=1 train_ds_enhanced.py
```

The script works without W&B - it simply skips tracking.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 16
}
```

### FP16 Errors

Disable mixed precision:
```json
{
  "fp16": {
    "enabled": false
  }
}
```

## Next Steps

- [Basic ConvNet](/docs/tutorials/basic/convnet) - Image classification
- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Memory optimization
