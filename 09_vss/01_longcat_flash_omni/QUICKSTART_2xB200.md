# 🚀 Quick Start: Training on 2x B200 GPUs

**Ultra-fast guide to get training started in 2 minutes!**

---

## ⚡ Super Quick Start

```bash
# 1. Navigate to folder
cd 09_vss

# 2. Run the automated launch script
chmod +x run_2xB200.sh
./run_2xB200.sh
```

**That's it!** The script will:
- ✅ Check your hardware (GPUs, RAM, storage)
- ✅ Verify data folder exists
- ✅ Set optimal environment variables
- ✅ Launch training automatically

---

## 📁 Files for 2x B200 Setup

| File | Purpose |
|------|---------|
| `train_ds_2xB200.py` | Conservative training script for 2 GPUs |
| `ds_config_2xB200.json` | DeepSpeed ZeRO-3 config with aggressive CPU offload |
| `run_2xB200.sh` | Automated launch script (checks everything) |
| `README_2xB200.md` | Complete guide with troubleshooting |
| `QUICKSTART_2xB200.md` | This file |

---

## 🎯 What You Get

**Training will:**
- Use 2x B200 GPUs (192GB each)
- Train LoRA adapters (~100MB) on frozen 560B model
- Complete 1 epoch in ~30-60 minutes (8 samples)
- Complete 3 epochs in ~1.5-3 hours
- Save model locally (~100-200MB)
- Upload to HuggingFace Hub (if HF_TOKEN set)

**Memory usage:**
- GPU: ~110-130GB per GPU ✅
- RAM: ~1-1.5TB (out of 3TB) ✅
- Storage: ~1.2TB (out of 2TB) ✅

---

## 🔧 Optional: Environment Variables

**Before running `run_2xB200.sh`, optionally set:**

```bash
# For HuggingFace Hub upload (recommended)
export HF_TOKEN=your_hf_token_here
export HF_USER=your_username

# For W&B experiment tracking (optional)
export WANDB_API_KEY=your_wandb_key

# Already set by run_2xB200.sh (no need to set manually):
# export HF_HUB_ENABLE_HF_TRANSFER=1
# export NCCL_P2P_LEVEL=NVL
```

---

## 📊 Monitoring During Training

**Terminal 1:** Training running
```bash
./run_2xB200.sh
```

**Terminal 2:** GPU memory
```bash
watch -n 1 nvidia-smi
```

**Terminal 3:** System RAM
```bash
watch -n 1 free -h
```

**Terminal 4:** TensorBoard (optional)
```bash
tensorboard --logdir=./tensorboard_logs/
```

---

## 🎯 Expected Output

```
==================================================
🚀 LongCat-Flash-Omni Training on 2x B200
==================================================

✓ Detected 2 GPUs
✓ Available storage: 2000GB
✓ Total RAM: 3000GB
✓ Found 8 training samples
✓ DeepSpeed config found
✓ HF_TOKEN is set
✓ WANDB_API_KEY is set

==================================================
🚀 Starting Training
==================================================

Configuration:
  - GPUs: 2x B200
  - LoRA rank: 16 (conservative)
  - Batch size: 1 per GPU
  - Gradient accumulation: 64
  - Effective batch size: 128
  - Dataset: 8 samples
  - Expected time: ~30-60 min per epoch

🚀 Launching DeepSpeed training...

[Step 1/4] Loss: 2.456 | GPU: 115GB | RAM: 1.2TB
[Step 2/4] Loss: 2.234 | GPU: 118GB | RAM: 1.3TB
[Step 3/4] Loss: 2.103 | GPU: 120GB | RAM: 1.4TB
[Step 4/4] Loss: 1.987 | GPU: 122GB | RAM: 1.4TB

✅ Epoch 1/3 complete (45 minutes)
...

==================================================
✅ Training completed successfully!
==================================================

Output:
  - Model: ./longcat-flash-omni-vss-lora-2xB200/
  - Logs: ./tensorboard_logs/

Model uploaded to HuggingFace Hub ✓
```

---

## ⚠️ Common Issues

### 1. "No space left on device"
```bash
# Check available space
df -h .

# If < 2TB, mount larger volume or increase container disk
```

### 2. "CUDA out of memory"
```bash
# Edit ds_config_2xB200.json, reduce:
"stage3_max_live_parameters": 3e8  # From 5e8
```

### 3. "Cannot allocate memory" (RAM)
```bash
# Edit train_ds_2xB200.py, increase:
gradient_accumulation_steps=128  # From 64
```

---

## 📖 Need More Details?

- **Complete guide:** [README_2xB200.md](README_2xB200.md)
- **Main README:** [README.md](README.md)
- **Check storage:** `./check_storage.sh`

---

## ✅ Success Checklist

Before training:
- [ ] 2x B200 GPUs available
- [ ] 2TB+ storage available
- [ ] 1TB+ RAM available (3TB recommended)
- [ ] Data in `data/train/` folder
- [ ] Dependencies installed (`uv add ...`)

Optional:
- [ ] `HF_TOKEN` set (for model upload)
- [ ] `WANDB_API_KEY` set (for tracking)

Ready? Run:
```bash
./run_2xB200.sh
```

---

**Good luck!** 🚀

Training on 2x B200 with LoRA is totally feasible. You got this! 💪
