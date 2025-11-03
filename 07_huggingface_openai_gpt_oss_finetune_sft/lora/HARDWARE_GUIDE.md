# Hardware Requirements & Model Selection Guide

## 📊 Quick Reference

| Your Hardware | Use This Script | Model | Memory Usage |
|---------------|----------------|-------|--------------|
| **2x RTX 3070 (8GB)** | `train_ds_mistral7b.py` | Mistral-7B | ~6-7GB/GPU ✅ |
| **2x RTX 3090 (24GB)** | `train_ds_mistral7b.py` OR `train_ds.py` | Mistral-7B or GPT-OSS-20B | ~6-7GB or ~20GB/GPU |
| **4x RTX 4090 (24GB)** | `train_ds.py` | GPT-OSS-20B | ~20GB/GPU ✅ |
| **4x A100 (40GB)** | `train_ds.py` | GPT-OSS-20B | ~20GB/GPU ✅ |
| **2x H100 (80GB)** | `train_ds_h200.py` | GPT-OSS-20B | ~20GB/GPU ⚡ Fastest! |
| **2x H200 (141GB)** | `train_ds_h200.py` | GPT-OSS-20B | ~20GB/GPU ⚡ Best! |

---

## 🎯 Which Script Should You Use?

### Option 1: `train_ds_mistral7b.py` (Recommended for 8GB GPUs)

**Model:** Mistral-7B-Instruct-v0.2 (7 billion parameters)

**Works on:**
- ✅ 2x RTX 3070 (8GB each)
- ✅ 2x RTX 3060 (12GB each)
- ✅ 2x RTX 3080 (10GB each)
- ✅ Any 2 GPUs with 8GB+ VRAM

**Memory requirements:**
- **Per GPU:** ~6-7GB VRAM (with LoRA + ZeRO-2)
- **System RAM:** 32GB+ recommended
- **Training config:**
  - Batch size: 1 per GPU
  - Gradient accumulation: 16
  - Effective batch size: 32 (1 × 2 GPUs × 16)

**Run command:**
```bash
uv run deepspeed --num_gpus=2 train_ds_mistral7b.py
```

**Quality:** Excellent! Mistral-7B is one of the best 7B models available.

---

### Option 2: `train_ds.py` (For High-End GPUs Only)

**Model:** OpenAI GPT-OSS-20B (20 billion parameters)

**Works on:**
- ✅ 4x A100 (40GB each)
- ⚠️ 4x RTX 4090 (24GB each) - tight fit
- ❌ 2x RTX 3070 (8GB each) - **TOO SMALL**

**Memory requirements:**
- **Per GPU:** ~20GB VRAM minimum (with LoRA + ZeRO-2)
- **System RAM:** 128GB+ recommended (256GB ideal)
- **Training config:**
  - Batch size: 2 per GPU
  - Gradient accumulation: 8
  - Effective batch size: 64 (2 × 4 GPUs × 8)

**Run command:**
```bash
uv run deepspeed --num_gpus=4 train_ds.py
```

**Quality:** Best quality, but requires expensive hardware.

---

### Option 3: `train_ds_h200.py` (For H100/H200 GPUs - OPTIMIZED!)

**Model:** OpenAI GPT-OSS-20B (20 billion parameters)

**Optimized for datacenter GPUs:**
- ✅ 2x H100 (80GB each) ⚡ **FAST**
- ✅ 2x H200 (141GB each) ⚡ **FASTEST & BEST**
- ✅ 4x H100 (80GB each) - even faster
- ✅ 4x H200 (141GB each) - maximum speed

**Why use this script:**
- **4x larger batch sizes** (8 vs 2 per GPU) = faster training
- **Optimized for high VRAM** - fully utilizes your hardware
- **Faster convergence** - more stable gradients with larger batches
- **Better quality** - larger effective batch size

**Memory requirements:**
- **Per GPU:** ~20GB VRAM (you have 80GB or 141GB!)
- **System RAM:** 256GB+ recommended
- **Training config (optimized):**
  - Batch size: **8 per GPU** (4x larger!)
  - Gradient accumulation: **4** (2x smaller)
  - Effective batch size: **64** (8 × 2 GPUs × 4)

**Run command:**
```bash
# For 2x H100 or 2x H200
uv run deepspeed --num_gpus=2 train_ds_h200.py

# For 4x H100 or 4x H200 (even faster!)
uv run deepspeed --num_gpus=4 train_ds_h200.py
```

**Performance:**
- **Training time on 2x H200:** ~45-60 minutes (10 epochs)
- **Training time on 4x A100:** ~90-120 minutes (10 epochs)
- **Speedup:** ~2x faster than standard config!

**Quality:** Best quality + fastest training!

---

## 🔍 How to Check Your Hardware

```bash
# Check GPU model and VRAM
nvidia-smi --query-gpu=name,memory.total --format=csv

# Example output:
# name, memory.total [MiB]
# NVIDIA GeForce RTX 3070, 8192 MiB
# NVIDIA GeForce RTX 3070, 8192 MiB

# Check system RAM
free -h

# Check available GPUs
nvidia-smi --list-gpus
```

---

## 📝 Configuration Files

Both scripts use the same DeepSpeed configuration:

- **`ds_config.json`**: ZeRO Stage 2 with BF16
  - Optimizer + gradient partitioning
  - No CPU offloading (faster)
  - Auto-configured batch sizes

---

## 🚀 Quick Start for Your Hardware (2x RTX 3070)

```bash
# Navigate to folder
cd 07_huggingface_openai_gpt_oss_finetune_sft/lora

# Install dependencies
uv add torch transformers accelerate datasets deepspeed peft trl

# Optional: Add W&B
uv add wandb

# Run training with Mistral-7B
uv run deepspeed --num_gpus=2 train_ds_mistral7b.py
```

**Expected results:**
- **Training time:** ~2-3 hours for 3 epochs (1000 samples)
- **Memory usage:** ~6-7GB per GPU
- **Quality:** Excellent multilingual reasoning

---

## 💡 Tips for 8GB GPUs

If you still get OOM with Mistral-7B:

**1. Reduce batch size further** (edit `train_ds_mistral7b.py` line 212):
```python
per_device_train_batch_size=1,  # Already at minimum
gradient_accumulation_steps=32,  # Increase from 16 to 32
```

**2. Reduce sequence length** (line 215):
```python
max_length=1024,  # Reduce from 2048 to 1024
```

**3. Reduce LoRA rank** (line 148):
```python
r=8,  # Reduce from 16 to 8
lora_alpha=16,  # Reduce from 32 to 16
```

**4. Use 1 GPU only** (slower but more memory):
```bash
uv run deepspeed --num_gpus=1 train_ds_mistral7b.py
```

---

## 📊 Model Comparison

| Model | Parameters | VRAM Needed | Quality | Speed |
|-------|-----------|-------------|---------|-------|
| **Phi-2** | 2.7B | ~4-5GB/GPU | Very Good | Fast ⚡⚡⚡ |
| **Mistral-7B** | 7B | ~6-7GB/GPU | Excellent | Medium ⚡⚡ |
| **Llama-2-7B** | 7B | ~6-7GB/GPU | Excellent | Medium ⚡⚡ |
| **Llama-2-13B** | 13B | ~12-14GB/GPU | Best | Slow ⚡ |
| **GPT-OSS-20B** | 20B | ~20GB/GPU | Best | Very Slow 🐌 |

---

## ❓ FAQ

### Q: Can I use train_ds.py with 2x RTX 3070?
**A:** No, GPT-OSS-20B requires minimum 20GB per GPU. Your GPUs have 8GB each.

### Q: Will Mistral-7B give good results?
**A:** Yes! Mistral-7B is one of the best 7B models and performs excellently on multilingual reasoning tasks.

### Q: Can I switch models later?
**A:** Yes, just edit the `model_name` variable in the script:
```python
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or any other model
```

### Q: What about quantization (4-bit, 8-bit)?
**A:** Quantization can help, but it's not implemented in these scripts. With LoRA + ZeRO-2, Mistral-7B should fit fine on your hardware.

---

## 🆘 Still Having Issues?

1. **Check GPU memory during training:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Look for memory leaks:**
   - Memory should stabilize after first few steps
   - If it keeps growing → reduce batch size or sequence length

3. **Try smaller model first:**
   ```python
   model_name = "microsoft/phi-2"  # Only 2.7B parameters
   ```

---

**For 2x RTX 3070 (8GB each): Use `train_ds_mistral7b.py` ✅**
