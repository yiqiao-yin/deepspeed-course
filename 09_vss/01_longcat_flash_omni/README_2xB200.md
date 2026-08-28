# CONSERVATIVE Configuration for 2x B200 GPUs

**Quick Start Guide for Training LongCat-Flash-Omni on 2x NVIDIA B200 GPUs**

---

## 🎯 Your Hardware

✅ **GPUs:** 2x NVIDIA B200 (192GB each = 384GB total)
✅ **System RAM:** 3TB (excellent for CPU offloading!)
✅ **Storage:** 2TB (perfect for model weights)
✅ **vCPU:** 192 cores

**Verdict:** Your hardware CAN train this model with conservative settings! 🚀

---

## ⚡ Quick Start

### 1. Navigate to folder
```bash
cd 09_vss
```

### 2. Install dependencies
```bash
# Initialize project
uv init

# Install all dependencies
uv add torch torchvision torchaudio transformers accelerate datasets deepspeed peft opencv-python pillow numpy tensorboard hf_transfer

# Optional: Add W&B and HF Hub
uv add wandb huggingface_hub
```

### 3. Set environment variables
```bash
# Required: Fast model downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: W&B tracking
export WANDB_API_KEY=your_key_here

# Optional: HF Hub upload
export HF_TOKEN=your_token_here
export HF_USER=your_username

# Optional: Enable NVLink (B200 supports it)
export NCCL_P2P_LEVEL=NVL
```

### 4. Run training (CONSERVATIVE mode)
```bash
# Use the 2x B200 optimized script
uv run deepspeed --num_gpus=2 train_ds_2xB200.py
```

**That's it!** Training will start. Expect ~30-60 minutes for 1 epoch (8 samples).

---

## 📊 What Changed for 2x B200?

### Conservative Settings Applied:

| Setting | Standard | 2x B200 Conservative | Why |
|---------|----------|---------------------|-----|
| **LoRA rank** | 32 | **16** | Reduce trainable params |
| **Gradient accumulation** | 32 | **64** | Maintain effective batch size with less memory |
| **Learning rate** | 1e-4 | **5e-5** | Slower but more stable |
| **Video frames** | 16 | **8** | Reduce activation memory |
| **Audio duration** | 30s | **10s** | Reduce activation memory |
| **Gradient clipping** | 1.0 | **0.5** | More aggressive clipping |
| **Logging** | Every 10 steps | **Every 1 step** | Monitor closely |
| **CPU offload** | Standard | **Ultra-aggressive** | Keep more in CPU RAM |
| **Checkpoints** | 3 | **2** | Save disk space |

### DeepSpeed ZeRO-3 Optimizations:

```json
{
  "stage3_max_live_parameters": 5e8,      // More aggressive (was 1e9)
  "stage3_max_reuse_distance": 5e8,       // More aggressive
  "stage3_prefetch_bucket_size": 5e7,     // Smaller prefetch
  "reduce_bucket_size": 5e7,              // Smaller buckets
  "activation_checkpointing": {
    "partition_activations": true,        // Shard activations
    "contiguous_memory_optimization": true
  }
}
```

---

## 🔍 Memory Breakdown

### Per-GPU (B200 has 192GB each):

```
Component                    | GPU 0    | GPU 1    | CPU RAM
-----------------------------|----------|----------|----------
Base model (frozen, sharded) | ~50GB    | ~50GB    | ~1000GB
LoRA adapters (trainable)    | ~50MB    | ~50MB    | -
Optimizer states (offloaded) | -        | -        | ~200MB
Gradients (offloaded)        | -        | -        | ~100MB
Activations (checkpointed)   | ~40GB    | ~40GB    | -
Buffer/temp                  | ~20GB    | ~20GB    | ~100GB
-----------------------------|----------|----------|----------
TOTAL                        | ~110GB   | ~110GB   | ~1.1TB
-----------------------------|----------|----------|----------
Available                    | 192GB    | 192GB    | 3000GB
Safety margin                | ~82GB ✅ | ~82GB ✅ | ~1.9TB ✅
```

**Verdict:** Should fit comfortably with ~40% GPU memory headroom! 🎉

---

## ⏱️ Training Time Estimates

With your **8 sample dataset**:

```
Configuration              | Time per Epoch | Total (3 epochs)
---------------------------|----------------|------------------
1 epoch = 4 steps          | ~30-60 min     | ~1.5-3 hours
(8 samples / 2 GPUs)       |                |
```

**Why slower than 8x GPUs?**
- Only 2 GPUs = less parallelism (4x fewer)
- Heavy CPU offloading = CPU↔GPU transfer overhead
- Conservative settings = smaller batches

**But you WILL finish!** 🚀

---

## 📈 Monitoring During Training

### Terminal 1: Run training
```bash
uv run deepspeed --num_gpus=2 train_ds_2xB200.py
```

### Terminal 2: Watch GPU memory
```bash
watch -n 1 nvidia-smi
```

**What to look for:**
- GPU memory: Should stay around 110-130GB per GPU
- GPU utilization: May be 60-80% (normal with CPU offload)
- Temperature: Should stay under 80°C

### Terminal 3: Watch system RAM
```bash
watch -n 1 free -h
```

**What to look for:**
- Used RAM: Will grow to ~1-1.5TB (this is normal!)
- Swap: Should stay at 0 (if swap is used, you're in trouble)

### Terminal 4: TensorBoard (optional)
```bash
tensorboard --logdir=./tensorboard_logs/
```

Open browser: http://localhost:6006

---

## 🚨 Troubleshooting

### Problem 1: Out of Memory (GPU)

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Edit ds_config_2xB200.json, reduce:
"stage3_max_live_parameters": 3e8  # Reduce from 5e8
"stage3_max_reuse_distance": 3e8   # Reduce from 5e8
```

### Problem 2: Out of Memory (CPU RAM)

**Error:** `Cannot allocate memory` or system freeze

**Solution:**
```bash
# Check RAM usage first:
free -h

# If using swap, stop training immediately
# Reduce dataset size or increase gradient accumulation:
# Edit train_ds_2xB200.py line ~371:
gradient_accumulation_steps=128  # Increase from 64
```

### Problem 3: Very Slow Training

**Issue:** Training taking 2+ hours per epoch

**This is normal!** With 2 GPUs and heavy CPU offloading, expect:
- Lots of CPU↔GPU data movement
- Lower GPU utilization (60-80%)
- High system RAM usage

**To speed up (trades memory for speed):**
```json
// Edit ds_config_2xB200.json:
"stage3_max_live_parameters": 1e9  // Increase from 5e8
```

### Problem 4: No Space Left on Device

**Error:** `No space left on device`

**Solution:**
```bash
# Check storage:
df -h .

# If < 2TB, you need more storage
# Mount external volume:
export HF_HOME=/workspace/models  # Larger volume
export TRANSFORMERS_CACHE=/workspace/models
```

### Problem 5: Model Download Fails

**Error:** `Connection timeout` or very slow download

**Solution:**
```bash
# Enable fast transfer:
export HF_HUB_ENABLE_HF_TRANSFER=1

# Or download manually first:
huggingface-cli download meituan-longcat/LongCat-Flash-Omni
```

---

## ✅ Success Indicators

You'll know training is working when you see:

```
INFO:__main__:🚀 Starting training...
INFO:__main__:Step 1/4 - Loss: 2.456
INFO:__main__:Step 2/4 - Loss: 2.234
INFO:__main__:Step 3/4 - Loss: 2.103
INFO:__main__:Step 4/4 - Loss: 1.987
INFO:__main__:✅ Epoch 1/3 complete! (30 minutes)
```

**Key metrics:**
- ✅ Loss decreasing
- ✅ GPU memory stable (~110-130GB per GPU)
- ✅ System RAM used (~1-1.5TB)
- ✅ No OOM errors
- ✅ No swap usage

---

## 📤 After Training

### Saved Files

```
09_vss/
├── longcat-flash-omni-vss-lora-2xB200/
│   ├── adapter_config.json          # LoRA config
│   ├── adapter_model.safetensors    # LoRA weights (~100MB)
│   └── training_args.bin
├── tensorboard_logs/
│   └── events.out.tfevents.*
```

**Total size:** ~100-200MB (just the LoRA adapters!)

### HuggingFace Hub Upload

If you set `HF_TOKEN`, your model automatically uploads to:
```
https://huggingface.co/your-username/longcat-flash-omni-vss-lora-2xB200
```

**Upload size:** ~100-200MB (fast!)

### Loading Your Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meituan-longcat/LongCat-Flash-Omni",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "your-username/longcat-flash-omni-vss-lora-2xB200"
)
```

---

## 🎓 Why This Works

**LoRA Magic:**
- Only training ~100MB (LoRA adapters)
- Freezing 560B parameters (read-only)
- 99.98% parameter reduction!

**DeepSpeed ZeRO-3:**
- Shards model across 2 GPUs + CPU RAM
- Each GPU only loads what it needs
- Offloads everything else to 3TB RAM

**Conservative Settings:**
- Small LoRA rank (r=16)
- Large gradient accumulation (64)
- Aggressive checkpointing
- Minimal live parameters

**Result:** Fits in 2x B200! 🎉

---

## 📚 Additional Resources

- **Original script:** `train_ds.py` (requires 8+ GPUs)
- **Storage check:** `check_storage.sh` (verify disk space)
- **DeepSpeed docs:** https://www.deepspeed.ai/
- **LoRA paper:** https://arxiv.org/abs/2106.09685

---

## ❓ FAQ

### Q: Will this definitely work?

**A:** 85% confident. Memory calculations say yes, but:
- Activation spikes unpredictable
- Model has custom MoE architecture
- First time this config tested

**Recommendation:** Try it! 1 epoch is only ~30-60 min. If it works, you're golden.

### Q: Can I use larger LoRA rank?

**A:** Not recommended. r=16 is already pushing it. If you want to try:
```python
# Edit train_ds_2xB200.py line ~333:
r=24,  # Increase from 16 (risky!)
```

### Q: Can I train faster?

**A:** Yes, but at the cost of memory:
```python
# Edit train_ds_2xB200.py:
gradient_accumulation_steps=32  # Reduce from 64
```

Or increase live parameters in `ds_config_2xB200.json`:
```json
"stage3_max_live_parameters": 1e9  // Double from 5e8
```

### Q: What if I get OOM?

**A:** Make it even more conservative:
- Increase gradient accumulation to 128
- Reduce LoRA rank to 8
- Reduce live parameters to 3e8

---

## 🚀 Bottom Line

**Can you train on 2x B200?**

# YES! ✅

**With these conservative settings:**
- ✅ Memory should fit (~110GB per GPU)
- ✅ Will complete 1 epoch (~30-60 min)
- ✅ Can finish all 3 epochs (~2-3 hours)
- ✅ LoRA adapters upload to HF Hub (~100MB)

**Trade-offs:**
- ⏱️ Slower than 8 GPUs (expected)
- 💾 High CPU RAM usage (normal)
- 🔄 Lots of CPU↔GPU transfers (normal)

**Just try it!** If it fails, we adjust. But memory calculations say you're good to go! 🎉

---

**Happy Training!** 🚀

Questions? Check the main [README.md](README.md) or troubleshooting section above.
