---
sidebar_position: 1
---

# DeepSpeed Configuration

Reference for DeepSpeed configuration options.

## Basic Structure

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "optimizer": {...},
  "scheduler": {...},
  "fp16": {...},
  "bf16": {...},
  "zero_optimization": {...},
  "gradient_clipping": 1.0
}
```

## Batch Size Configuration

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4
}
```

**Relationship:**
```
train_batch_size = micro_batch × gradient_accum × num_gpus
32 = 8 × 4 × 1
```

**Auto mode** (for HuggingFace):
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

## Optimizer

### Adam

```json
{
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0
    }
  }
}
```

### AdamW

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "weight_decay": 0.01
    }
  }
}
```

### SGD

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

## Learning Rate Scheduler

### Warmup

```json
{
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-4,
      "warmup_num_steps": 100
    }
  }
}
```

### Warmup + Decay

```json
{
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-3,
      "warmup_num_steps": 100,
      "total_num_steps": 10000
    }
  }
}
```

## Mixed Precision

### FP16

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

### BF16

```json
{
  "bf16": {
    "enabled": true
  }
}
```

**Note**: BF16 requires Ampere (A100) or newer GPUs.

## ZeRO Optimization

### Stage 1

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

### Stage 2

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  }
}
```

### Stage 2 + CPU Offload

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### Stage 3

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e6,
    "stage3_prefetch_bucket_size": 5e5,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

## Gradient Clipping

```json
{
  "gradient_clipping": 1.0
}
```

## Complete Examples

### Basic Training (FP16 + ZeRO-2)

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {"lr": 1e-3}
  },
  "fp16": {"enabled": true},
  "zero_optimization": {"stage": 2}
}
```

### HuggingFace Integration

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu", "pin_memory": true}
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

### Large Model (ZeRO-3 + Offload)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0
}
```

## Validation

DeepSpeed validates configurations at startup. Common errors:

- **Batch size mismatch**: Check the batch size formula
- **Missing optimizer**: Required unless using HuggingFace
- **Invalid ZeRO stage**: Must be 0, 1, 2, or 3

## Next Steps

- [Troubleshooting](/docs/reference/troubleshooting) - Common issues
- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Stage details
