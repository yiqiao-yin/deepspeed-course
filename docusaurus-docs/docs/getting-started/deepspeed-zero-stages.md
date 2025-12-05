---
sidebar_position: 3
---

# DeepSpeed ZeRO Stages

Understanding Zero Redundancy Optimizer (ZeRO) for efficient distributed training.

## What is ZeRO?

ZeRO (Zero Redundancy Optimizer) is DeepSpeed's memory optimization technology that enables training of models with billions of parameters by partitioning model states across GPUs.

## The Three Stages

### ZeRO Stage 1: Optimizer State Partitioning

**What it does:** Partitions optimizer states (momentum, variance in Adam) across GPUs.

**Memory savings:** ~4x reduction in optimizer memory

**Use case:** When optimizer states dominate memory usage

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

**Example:** Training with Adam optimizer
- Without ZeRO: Each GPU stores full optimizer states
- With ZeRO-1: Optimizer states split across GPUs

### ZeRO Stage 2: Gradient Partitioning

**What it does:** Adds gradient partitioning to Stage 1.

**Memory savings:** ~8x reduction in memory for gradients + optimizer states

**Use case:** Medium to large models, most common choice

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```

**This is used in most examples in this course.**

### ZeRO Stage 3: Parameter Partitioning

**What it does:** Partitions model parameters across GPUs in addition to Stage 2.

**Memory savings:** Linear scaling with number of GPUs

**Use case:** Very large models (billions of parameters)

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
    }
  }
}
```

## Comparison Table

| Feature | Stage 1 | Stage 2 | Stage 3 |
|---------|---------|---------|---------|
| Optimizer states partitioned | Yes | Yes | Yes |
| Gradients partitioned | No | Yes | Yes |
| Parameters partitioned | No | No | Yes |
| Communication overhead | Low | Medium | High |
| Memory efficiency | Good | Better | Best |
| Recommended for | Small models | Medium models | Large models |

## When to Use Each Stage

### ZeRO Stage 1
- Models that fit on a single GPU
- Want minimal communication overhead
- Optimizer memory is the bottleneck

### ZeRO Stage 2 (Most Common)
- Models that almost fit on a single GPU
- Good balance of memory efficiency and speed
- **Recommended starting point for most users**

### ZeRO Stage 3
- Models too large for single GPU even with Stage 2
- Multi-billion parameter models
- Willing to trade speed for memory efficiency

## CPU Offloading

ZeRO-Offload extends any stage to use CPU memory:

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

**Benefits:**
- Train larger models on limited GPU memory
- Essential for consumer GPUs (8-16GB)

**Trade-offs:**
- Slower due to CPU-GPU data transfer
- Requires sufficient system RAM

## Practical Examples from This Course

### Basic Examples (Stages 1-2)

```json
// 01_basic_neuralnet/ds_config.json
{
  "zero_optimization": {
    "stage": 2
  },
  "fp16": {
    "enabled": true
  }
}
```

### HuggingFace with LoRA (Stage 2 + Offload)

```json
// 06_huggingface_grpo/ds_config.json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

### Large Multimodal Models (Stage 3)

```json
// 09_vss/ds_config.json
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
    }
  }
}
```

## Choosing the Right Stage

```
Start with ZeRO Stage 2
         │
         ▼
    Fits in memory?
    ┌────┴────┐
   Yes        No
    │          │
    ▼          ▼
  Done!    Add CPU offload
              │
              ▼
         Fits now?
         ┌───┴───┐
        Yes      No
         │        │
         ▼        ▼
       Done!   Use Stage 3
```

## Additional Resources

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload Paper](https://arxiv.org/abs/2101.06840)
