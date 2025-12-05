---
sidebar_position: 2
---

# Troubleshooting

Common issues and solutions for DeepSpeed training.

## Memory Issues

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```json
{"train_micro_batch_size_per_gpu": 1}
```

2. **Enable CPU offloading:**
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

3. **Use ZeRO Stage 3:**
```json
{"zero_optimization": {"stage": 3}}
```

4. **Enable gradient checkpointing:**
```python
model.gradient_checkpointing_enable()
```

### Host Memory Exhausted

**Error:**
```
RuntimeError: Host memory exhausted
```

**Solutions:**

1. Increase system RAM
2. Reduce offloading buffer sizes:
```json
{
  "zero_optimization": {
    "offload_optimizer": {
      "device": "cpu",
      "buffer_count": 2
    }
  }
}
```

## Configuration Issues

### Batch Size Mismatch

**Error:**
```
AssertionError: Check batch related parameters
```

**Solution:**
Ensure formula holds:
```
train_batch_size = micro_batch × gradient_accum × num_gpus
```

Example fix:
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4
}
// With 1 GPU: 8 × 4 × 1 = 32 ✓
```

### Missing Optimizer

**Error:**
```
DeepSpeedConfigError: optimizer must be specified
```

**Solution:**
Add optimizer config:
```json
{
  "optimizer": {
    "type": "Adam",
    "params": {"lr": 1e-3}
  }
}
```

Or use `"auto"` with HuggingFace:
```json
{
  "optimizer": {"type": "AdamW", "params": {"lr": "auto"}}
}
```

## Training Issues

### Loss is NaN

**Causes:**
- Learning rate too high
- Gradient explosion
- FP16 overflow

**Solutions:**

1. **Reduce learning rate:**
```json
{"optimizer": {"params": {"lr": 1e-5}}}
```

2. **Enable gradient clipping:**
```json
{"gradient_clipping": 1.0}
```

3. **Use loss scaling:**
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 12
  }
}
```

### Training Very Slow

**Causes:**
- Excessive CPU offloading
- Small batch size
- Communication overhead

**Solutions:**

1. **Reduce offloading if RAM allows:**
```json
{"zero_optimization": {"stage": 2}}  // Remove offload
```

2. **Increase batch size:**
```json
{"train_micro_batch_size_per_gpu": 8}
```

3. **Enable overlap:**
```json
{
  "zero_optimization": {
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

### Gradients Exploding (RNNs)

**Error:**
```
Grad norm: inf
```

**Solutions:**

1. **Lower gradient clipping:**
```json
{"gradient_clipping": 0.5}
```

2. **Reduce learning rate**

3. **Check weight initialization:**
```python
nn.init.orthogonal_(lstm.weight_hh_l0)
```

## Multi-GPU Issues

### NCCL Errors

**Error:**
```
NCCL Error: unhandled cuda error
```

**Solutions:**

1. **Set NCCL debug:**
```bash
export NCCL_DEBUG=INFO
```

2. **Check GPU visibility:**
```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

3. **Try different NCCL settings:**
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Hanging at Initialization

**Causes:**
- Port conflict
- Network issues
- GPU mismatch

**Solutions:**

1. **Change master port:**
```bash
deepspeed --master_port=29501 train.py
```

2. **Check all GPUs accessible:**
```python
import torch
print(torch.cuda.device_count())
```

## FP16/BF16 Issues

### FP16 Not Supported

**Error:**
```
RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
```

**Solution:**
Disable FP16 or use BF16:
```json
{"fp16": {"enabled": false}}
// or
{"bf16": {"enabled": true}}
```

### BF16 Not Available

**Error:**
```
BF16 is not supported on this hardware
```

**Solution:**
Use FP16 instead (requires Ampere+ for BF16):
```json
{"fp16": {"enabled": true}, "bf16": {"enabled": false}}
```

## HuggingFace Integration

### Tokenizer Padding

**Error:**
```
ValueError: Padding side mismatch
```

**Solution:**
```python
tokenizer.padding_side = "left"  # For generation
tokenizer.pad_token = tokenizer.eos_token
```

### Model Loading OOM

**Solution:**
Load in lower precision:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

## Debugging Tips

### Enable Verbose Logging

```bash
export DEEPSPEED_LOG_LEVEL=debug
```

### Check Configuration

```python
import deepspeed
print(deepspeed.runtime.config.DeepSpeedConfig("ds_config.json"))
```

### Profile Memory

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Getting Help

1. Check [DeepSpeed GitHub Issues](https://github.com/microsoft/DeepSpeed/issues)
2. Review [DeepSpeed Documentation](https://www.deepspeed.ai/)
3. Search [HuggingFace Forums](https://discuss.huggingface.co/)

## Next Steps

- [DeepSpeed Config Reference](/docs/reference/deepspeed-config)
- [Hardware Requirements](/docs/guides/hardware-requirements)
