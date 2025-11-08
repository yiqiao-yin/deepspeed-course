# Hardware Requirements for Llama-3.2-3B-Instruct Fine-tuning

This document provides detailed GPU requirements and recommendations for fine-tuning the **Llama-3.2-3B-Instruct** model (approximately 3 billion parameters) with DeepSpeed.

## Model Overview

- **Model**: `unsloth/Llama-3.2-3B-Instruct`
- **Parameters**: ~3.2 billion
- **Base Memory (FP32)**: ~12 GB (model weights only)
- **Base Memory (FP16)**: ~6 GB (model weights only)
- **Training Memory**: Additional 2-3x for optimizer states, gradients, and activations

## Memory Requirements Breakdown

### Without DeepSpeed (Standard PyTorch)

| Component | FP32 | FP16/BF16 |
|-----------|------|-----------|
| Model Weights | 12 GB | 6 GB |
| Gradients | 12 GB | 6 GB |
| Optimizer States (Adam) | 24 GB | 12 GB |
| Activations (batch_size=8) | ~4 GB | ~2 GB |
| **Total** | **~52 GB** | **~26 GB** |

### With DeepSpeed ZeRO

| ZeRO Stage | Memory per GPU (2 GPUs) | Memory per GPU (4 GPUs) |
|------------|-------------------------|-------------------------|
| Stage 1 | ~20 GB (FP16) | ~16 GB (FP16) |
| Stage 2 | ~14 GB (FP16) | ~10 GB (FP16) |
| Stage 3 | ~10 GB (FP16) | ~6 GB (FP16) |

## GPU Compatibility Matrix

### ✅ Highly Recommended (Excellent Performance)

| GPU Model | VRAM | Quantity | ZeRO Stage | Batch Size | Notes |
|-----------|------|----------|------------|------------|-------|
| **H200** | 141 GB | 2x | 1 or 2 | 16-32 | Massive overkill; can run much larger models |
| **H100 (SXM)** | 80 GB | 2x | 1 or 2 | 16-32 | Overkill; excellent for larger models |
| **A100 (80GB)** | 80 GB | 2x | 1 or 2 | 16-32 | Ideal for this model size |
| **H100 (PCIe)** | 80 GB | 2x | 1 or 2 | 16-32 | Excellent performance |

**Recommendation**: These GPUs are overpowered for 3B models. Consider using them for 7B-70B models instead.

### ✅ Recommended (Great Performance)

| GPU Model | VRAM | Quantity | ZeRO Stage | Batch Size | Notes |
|-----------|------|----------|------------|------------|-------|
| **RTX 5090** | 32 GB | 2x | 1 or 2 | 8-16 | Perfect for 3B models; great value |
| **RTX 4090** | 24 GB | 2x | 2 | 4-8 | Works well with ZeRO-2 |
| **A100 (40GB)** | 40 GB | 2x | 1 or 2 | 8-16 | Professional option |
| **A6000** | 48 GB | 2x | 1 or 2 | 8-16 | Professional workstation |
| **L40S** | 48 GB | 2x | 1 or 2 | 8-16 | Data center GPU |

**Recommendation**: RTX 5090 (2x) or RTX 4090 (2x) offer the best price-performance for 3B models.

### ⚠️ Feasible (Requires Optimization)

| GPU Model | VRAM | Quantity | ZeRO Stage | Batch Size | Notes |
|-----------|------|----------|------------|------------|-------|
| **RTX 3090** | 24 GB | 2x | 2 | 4-8 | Reduce batch size; enable gradient checkpointing |
| **RTX A5000** | 24 GB | 2x | 2 | 4-8 | Professional card; similar to 3090 |
| **V100 (32GB)** | 32 GB | 2x | 2 | 4-8 | Older data center GPU; works but slower |
| **RTX 4080** | 16 GB | 2x | 2 or 3 | 2-4 | Tight fit; requires careful optimization |

**Recommendation**: Use ZeRO-2 or ZeRO-3, reduce `per_device_train_batch_size` to 2-4, and enable gradient checkpointing.

### ⚠️ Challenging (Very Tight Fit)

| GPU Model | VRAM | Quantity | ZeRO Stage | Batch Size | Notes |
|-----------|------|----------|------------|------------|-------|
| **RTX 3080** | 10 GB | 2x | 3 | 1-2 | Very tight; requires aggressive optimization |
| **RTX 2080 Ti** | 11 GB | 2x | 3 | 1-2 | Old GPU; barely feasible |
| **RTX 4070 Ti** | 12 GB | 2x | 3 | 1-2 | Consumer GPU; marginal |

**Recommendation**: Use ZeRO-3, batch size 1-2, gradient accumulation (8-16 steps), gradient checkpointing, and consider LoRA/QLoRA instead of full fine-tuning.

### ❌ Not Recommended

| GPU Model | VRAM | Quantity | Reason |
|-----------|------|----------|--------|
| **RTX 2000 Series** | 6-8 GB | 2x | Insufficient VRAM even with ZeRO-3 |
| **GTX 1080 Ti** | 11 GB | 2x | Too old; lacks tensor cores; barely works |
| **RTX 3070** | 8 GB | 2x | Insufficient VRAM for full fine-tuning |
| **RTX 3060** | 12 GB | 2x | Slow performance; not recommended |

**Alternative**: For GPUs with <12 GB VRAM, consider:
- **LoRA/QLoRA**: Parameter-efficient fine-tuning (reduces memory by ~4x)
- **Smaller models**: Try Llama-3.2-1B-Instruct instead
- **Cloud GPUs**: Rent A100/H100 instances on RunPod, Vast.ai, or Lambda Labs

## Detailed GPU Analysis

### 2x H200 (141 GB each)

**Feasibility**: ✅✅✅ Massive overkill

- **Memory Available**: 282 GB total
- **Model Requirements**: ~14 GB (ZeRO-2) per GPU
- **Recommendation**: These GPUs are designed for 70B-405B parameter models. For 3B models, you can:
  - Run batch size of 64+ per GPU
  - Train multiple models simultaneously
  - Use for much larger models (Llama-70B, Mixtral 8x7B, etc.)

**Configuration**:
```json
{
  "zero_optimization": {"stage": 1},
  "fp16": {"enabled": true}
}
```
```python
per_device_train_batch_size = 32  # Can go much higher
```

---

### 2x RTX 5090 (32 GB each)

**Feasibility**: ✅✅ Perfect match

- **Memory Available**: 64 GB total
- **Model Requirements**: ~14 GB (ZeRO-2) per GPU
- **Recommendation**: Ideal for 3B-7B models with excellent performance

**Configuration**:
```json
{
  "zero_optimization": {"stage": 2},
  "fp16": {"enabled": true}
}
```
```python
per_device_train_batch_size = 8  # Can increase to 16
```

---

### 2x RTX 4090 (24 GB each)

**Feasibility**: ✅ Recommended

- **Memory Available**: 48 GB total
- **Model Requirements**: ~14 GB (ZeRO-2) per GPU
- **Recommendation**: Works great with ZeRO-2; popular choice for researchers

**Configuration**:
```json
{
  "zero_optimization": {"stage": 2},
  "fp16": {"enabled": true},
  "gradient_accumulation_steps": 2
}
```
```python
per_device_train_batch_size = 4  # Increase to 8 if memory allows
```

---

### 2x RTX 2080 Ti (11 GB each)

**Feasibility**: ❌ Not recommended

- **Memory Available**: 22 GB total
- **Model Requirements**: ~10 GB (ZeRO-3) per GPU
- **Issues**:
  - Very tight memory fit
  - Old architecture (Turing, 2018)
  - Lacks modern tensor core features
  - Training will be extremely slow

**Recommendation**:
- **Don't use for full fine-tuning** - memory will be constantly maxed out
- **Alternative**: Use QLoRA (4-bit quantization) + LoRA
  - Reduces memory from ~10 GB to ~4 GB per GPU
  - 4x faster training with minimal quality loss
  - Enables batch size 2-4 instead of 1

**QLoRA Configuration**:
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

---

## Quick Selection Guide

**Choose your GPU setup:**

| Your Hardware | Recommended Config | Expected Training Time (50 epochs) |
|---------------|-------------------|-------------------------------------|
| 2x H200/H100/A100 (80GB) | ZeRO-1, FP16, batch=16 | ~3-4 hours |
| 2x RTX 5090 | ZeRO-2, FP16, batch=8 | ~5-6 hours |
| 2x RTX 4090 | ZeRO-2, FP16, batch=4 | ~6-8 hours |
| 2x RTX 3090/4080 | ZeRO-2, FP16, batch=4 | ~8-10 hours |
| 2x RTX 3080/4070 Ti | ZeRO-3, FP16, batch=2 | ~12-16 hours |
| <12 GB VRAM | QLoRA/LoRA | ~8-12 hours (LoRA only) |

## Optimization Tips

### For Limited VRAM (<16 GB per GPU)

1. **Enable Gradient Checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Reduce Sequence Length**:
   ```python
   max_seq_length = 512  # Instead of 2048
   ```

3. **Gradient Accumulation**:
   ```json
   {
     "gradient_accumulation_steps": 8
   }
   ```
   Effective batch size = `per_device_train_batch_size` × `num_gpus` × `gradient_accumulation_steps`

4. **Mixed Precision**:
   ```json
   {
     "fp16": {
       "enabled": true,
       "loss_scale": 0,
       "loss_scale_window": 1000
     }
   }
   ```

5. **CPU Offloading** (ZeRO-3 only):
   ```json
   {
     "zero_optimization": {
       "stage": 3,
       "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
       }
     }
   }
   ```

## Monitoring Memory Usage

During training, monitor GPU memory:

```bash
watch -n 1 nvidia-smi
```

Check for:
- **Allocated Memory**: Should be <90% of total VRAM
- **Memory Spikes**: Indicates potential OOM risk
- **GPU Utilization**: Should be >80% during training

## Summary

- **H200/H100/A100**: Overkill for 3B, perfect for 70B+
- **RTX 5090/4090**: Ideal sweet spot for 3B-7B models
- **RTX 3090/3080**: Feasible with ZeRO-2/3 and optimization
- **RTX 2080 Ti and below**: Use QLoRA instead of full fine-tuning

For the best experience with Llama-3.2-3B-Instruct, use **2x RTX 4090** or **2x RTX 5090** with **ZeRO-2** optimization.
