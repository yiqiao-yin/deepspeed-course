# LLaVA Video Trainer 🎥🤖

Vision-Language model training for video understanding using **LLaVA** (Large Language and Vision Assistant) architecture.

## 🎯 Use Case

**Perfect for:** Video understanding tasks where you need the model to actually "see" and comprehend video content through multiple frames.

- Video question answering
- Video captioning with visual understanding
- Multi-frame video analysis
- Vision-language conversation about videos

## ⚡ Quick Start

**Recommended Setup: Using `uv`**

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package installer and project manager (10-100x faster than pip).

```bash
# 1. Install uv
pip install uv

# 2. Navigate to trainer directory
cd llava_video_trainer

# 3. Initialize project
uv init .

# 4. Add all dependencies (including deepspeed)
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer
uv add opencv-python-headless   # REQUIRED for video frame extraction

# 5. Set credentials
export HF_USER_ID=eagle0504
export HF_TOKEN=your_hf_token
export WANDB_API_KEY=your_wandb_key  # Optional

# 6. Run training with DeepSpeed (2 GPUs)
export CUDA_VISIBLE_DEVICES=0,1
uv run deepspeed --num_gpus=2 video_training_script.py
```

**Why `uv`?**
- ⚡ 10-100x faster than pip
- 🔒 Better dependency resolution
- 📦 Creates isolated virtual environments with automatic activation
- 🎯 Reproducible builds with lock files (pyproject.toml + uv.lock)
- 📝 Uses `uv add` to manage dependencies in pyproject.toml
- 🚀 Seamlessly integrates with `deepspeed` launcher via `uv run`

## 🏗️ Model Architecture

- **Model Type**: `LlavaForConditionalGeneration`
- **Base Model**: `llava-hf/llava-interleave-qwen-7b-hf` (7B parameters)
- **Input**: Video frames (5 frames per video, configurable)
- **Output**: Text responses based on visual content
- **Format**: LLaVA conversation format with user/assistant roles

## 📊 Data Format

LLaVA uses a conversation format with multiple image tokens representing video frames:

```json
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this video?"},
        {"type": "image"},  // Frame 1
        {"type": "image"},  // Frame 2
        {"type": "image"},  // Frame 3
        {"type": "image"},  // Frame 4
        {"type": "image"}   // Frame 5
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "There is a cat in the video."}
      ]
    }
  ],
  "video_url": "https://example.com/video.mp4",
  "num_frames": 5
}
```

## ⚙️ DeepSpeed Configuration

**IMPORTANT:** This script **generates its own DeepSpeed config** internally.

The config is created with **`"auto"` values** that automatically sync with `TrainingArguments`:

```python
config = {
    "optimizer": {
        "params": {
            "lr": "auto",          # Syncs with TrainingArguments
            "betas": "auto",
            "weight_decay": "auto"
        }
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    # ...
}
```

**✅ Advantages:**
- No manual config/TrainingArguments synchronization needed
- Automatically handles batch size calculations
- Single source of truth (TrainingArguments)
- Less error-prone

**📝 File:** Config is generated and saved as `ds_config.json` when `create_deepspeed_config()` is called in `main()`.

## 🎬 Video Processing

Frames are decoded with OpenCV and sampled uniformly across the clip:

```python
def extract_frames_from_file(video_path, num_frames):
    # Uniformly spaced indices across the whole clip
    # cv2 decode -> BGR to RGB -> PIL.Image
    # Returns exactly num_frames RGB images
```

**Behaviour:**
- Extracts `num_frames` uniformly-spaced frames (default: 5)
- Accepts local paths, remote video URLs, and still images
- Converts **BGR to RGB** — OpenCV decodes BGR, the vision encoder expects RGB
- Short clips pad by repeating the last good frame, so the count always matches
  the number of image tokens in the prompt
- **Raises on undecodable input. It does NOT substitute a placeholder.**

> ⚠️ **Requires `opencv-python-headless`.** Install it with
> `uv pip install opencv-python-headless`.

### Why it raises instead of falling back

An earlier version returned a placeholder image repeated `num_frames` times
whenever it could not decode the input. That is worse than crashing: every
"video" became N copies of one still image, so the dataset carried **zero
temporal signal** while training ran normally and the loss went down. A crash is
a bug report; a silently degenerate dataset is a wasted GPU-week.

---

## 🧪 Testing This Trainer

The vision path has two tiers of test, and the distinction matters.

### CPU — runs in CI, no GPU, no download

```bash
uv run tests/test_video_frames.py        # from the repository root
```

Covers the structure: frames are genuinely distinct, BGR→RGB conversion is
applied, sampling spans the clip, failures raise, `preprocess_function` unwraps
the processor's batch dimension, and the collator keeps `pixel_values`.

**This is the guard that protects the repository day to day.** It runs on every
push.

### GPU — manual, needs a real model

```bash
uv run --with torch --with transformers --with accelerate \
       --with opencv-python-headless --with pillow \
       tests/gpu/validate_llava_vision_path.py
```

Drives the real `preprocess_function` and `LlavaVideoCollator` against an actual
LLaVA model and asserts that **perturbing `pixel_values` changes the loss** —
the only way to prove the pixels are truly reaching the model. Skips cleanly
with exit 0 when no GPU is present.

### The bug this tiering exists to catch

Structural tests alone were not enough. HuggingFace processors return token
fields **with a batch dimension**:

```python
processed["input_ids"]    # [[t0, t1, ..., t2937]]  -- nested
```

`preprocess_function` appended that without unwrapping, so each example became a
length-1 list containing a list. The collator computed the max length as **1**
and padded every sequence to a single token.

Nothing raised. Shapes broadcast, the forward pass returned a finite loss, and
training looked healthy. It surfaced only when a GPU run asserted on the actual
**sequence length** (1 instead of 2938).

The fix unwraps the batch dimension, and the CPU test now reproduces the nested
shape with a fake processor — so this specific regression is caught in CI
**without needing a GPU**. See [`tests/gpu/README.md`](../../../tests/gpu/README.md).

## 💾 Disk Space Management

This script includes **disk monitoring and cleanup** (important for large 7B model):

```python
# Check disk space before/after training
check_disk_space()  # Shows free space on root and workspace

# Clean up cache to save space
cleanup_cache_files()  # Clears pip cache and /tmp files
```

## 🚀 Running Training

### Method 1: Using `uv` + DeepSpeed (Recommended)

```bash
cd llava_video_trainer

# Step 1: Install uv and setup project
pip install uv
uv init .
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer
uv add opencv-python-headless   # REQUIRED for video frame extraction

# Step 2: Set required environment variables
export HF_USER_ID=eagle0504
export HF_TOKEN=your_hf_token

# Step 3 (Optional): Set W&B tracking
export WANDB_API_KEY=your_wandb_key  # ← Only if you want tracking

# Step 4: Run training with DeepSpeed (2 GPUs example)
export CUDA_VISIBLE_DEVICES=0,1
uv run deepspeed --num_gpus=2 video_training_script.py
```

**Note:** Using `deepspeed` launcher enables proper distributed training with ZeRO optimizations.

### Method 2: Direct DeepSpeed Execution

```bash
cd llava_video_trainer

# Install dependencies with pip
pip install torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer

# Required environment variables
export HF_USER_ID=eagle0504
export HF_TOKEN=your_hf_token

# Optional - for Weights & Biases tracking
export WANDB_API_KEY=your_wandb_key  # ← Only if you want tracking

# Run training with DeepSpeed (2 GPUs example)
export CUDA_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 video_training_script.py
```

**Weights & Biases Tracking (Optional):**

If you set `WANDB_API_KEY`:
```bash
export WANDB_API_KEY=your_key
export CUDA_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 video_training_script.py
```
Output:
```
✅ Weights & Biases enabled. Run: llava-video-20251027-123456
```

If you don't set it:
```bash
export CUDA_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 video_training_script.py
```
Output:
```
ℹ️  Weights & Biases disabled (WANDB_API_KEY not set)
```
**Script still runs perfectly with or without W&B!**

### Method 3: With run_training.sh

```bash
# (Optional) Setup with uv first
pip install uv
uv init .
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer
uv add opencv-python-headless   # REQUIRED for video frame extraction

# Set environment variables
export HF_USER_ID=your_username
export HF_TOKEN=your_token

# Optional - for W&B tracking
export WANDB_API_KEY=your_wandb_key

# Make script executable
chmod +x run_training.sh

# Run with 4 GPUs (default)
./run_training.sh

# Or specify number of GPUs
./run_training.sh 2
```

## 📦 Requirements

### Option 1: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

```bash
# Install uv
pip install uv

# Initialize project (creates pyproject.toml)
cd llava_video_trainer
uv init .

# Add dependencies (updates pyproject.toml and creates uv.lock)
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer
uv add opencv-python-headless   # REQUIRED for video frame extraction
```

### Option 2: Using `uv pip` into an existing environment

```bash
uv pip install torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb hf_transfer
uv pip install opencv-python-headless   # REQUIRED for video frame extraction
```

**Key dependencies:**
- `pillow` - For image processing (PIL)
- `requests` - For downloading video frames
- `transformers` - LLaVA model support
- `wandb` - (Optional) For experiment tracking
- `torch` - PyTorch deep learning framework
- `deepspeed` - Distributed training optimization
- `hf_transfer` - Fast HuggingFace Hub uploads/downloads (Rust-based)

## 🎓 Training Configuration

```python
# Check if wandb is available and configured
use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY") is not None

TrainingArguments(
    output_dir="./llava_video_finetune",
    run_name=f"llava-video-{timestamp}" if use_wandb else None,
    per_device_train_batch_size=1,  # Large model, small batch
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,                       # Mixed precision
    remove_unused_columns=False,     # CRITICAL for multimodal!
    dataloader_num_workers=0,        # Avoid multiprocessing issues
    do_eval=True,
    save_total_limit=2,
    warmup_steps=100,
    weight_decay=0.01,
    report_to=["wandb"] if use_wandb else []  # Optional W&B tracking
)
```

**Important:**
- `remove_unused_columns=False` is **essential** for vision-language models
- `eval_dataset=None` prevents local checkpoint creation (saves disk space)
- Model is pushed directly to Hub after training
- `report_to=["wandb"]` is automatically set if `WANDB_API_KEY` is available

## 💾 Model Saving Strategy

This script uses a **direct-to-Hub** approach to save disk space:

```python
# After training - only model is uploaded
self.save_model_directly_to_hub(
    trainer.model,
    model_repo_id,
    base_model,
    num_samples=len(video_urls)
)

# Uses safetensors with smaller shards
model.push_to_hub(
    model_repo_id,
    safe_serialization=True,
    max_shard_size="2GB"
)
```

**Benefits:**
- No local checkpoint creation during training
- Smaller shards (2GB) reduce memory requirements
- Uses safetensors format (safer, faster)
- **No dataset uploads** - only the trained model is pushed

## 📊 Typical Resource Usage

- **Model Size**: ~14GB (7B parameters in FP16)
- **GPU Memory**: ~16-20GB per GPU (with ZeRO-2)
- **Training Time**: ~5-10 minutes (4 GPUs, 3 epochs, 4 samples)
- **Disk Space**: Monitor actively (script has built-in checks)

## 🆚 When to Use This vs Seq2Seq Trainer

**Use LLaVA Video Trainer when:**
- ✅ You need actual video understanding (not just text about videos)
- ✅ Working with vision-language tasks
- ✅ Need multi-frame visual reasoning
- ✅ Want conversation-style interactions
- ✅ Have sufficient GPU memory (16GB+ per GPU)

**Use Seq2Seq Trainer when:**
- ✅ Text-to-text generation tasks
- ✅ Smaller models (< 1B params)
- ✅ Video metadata processing (not visual)
- ✅ Limited GPU resources

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size or frames per video
num_frames = 3  # Down from 5
per_device_train_batch_size = 1  # Already minimal
```

### Disk Space Issues
```bash
# Script automatically monitors disk space
# Check output for warnings like:
# ⚠️  WARNING: Root filesystem low on space!

# Manual cleanup:
pip cache purge
rm -rf /tmp/*
```

### Processor Errors
The script automatically fixes common LLaVA processor issues:
```python
# Sets pad_token to eos_token if missing
# Copies tokenizer attributes to processor
# You shouldn't need to do anything manually
```

## 📝 Output

**Model:** `{HF_USER_ID}/llava-video-text-model`

The trained model is automatically pushed to HuggingFace Hub with a comprehensive README.

**What gets uploaded:**
- ✅ Trained model weights (safetensors format)
- ✅ Model processor/tokenizer
- ✅ Model card (README.md)
- ❌ Dataset (not uploaded - only used locally for training)

## 🔄 Training Workflow

The script follows this streamlined workflow:

1. **Download** → Downloads video frames from provided URLs
2. **Process** → Creates LLaVA conversation format with 5 frames per video
3. **Train** → Fine-tunes LLaVA model with DeepSpeed
4. **Upload** → Pushes only the trained model to HuggingFace Hub
5. **(Optional)** → Tracks metrics in Weights & Biases if configured

**No dataset uploads** means:
- ✅ Faster workflow
- ✅ No 409/412 repository conflict errors
- ✅ Only your trained model is saved publicly
- ✅ Training data stays local

### Detailed Training Execution Flow

When you run training with DeepSpeed on 4 GPUs, here's what happens:

#### 1. Training Phase (All 4 GPUs)
```
100%|████████| 3/3 [00:01<00:00, 3.06it/s]
✅ Training completed successfully!
```
- All 4 ranks (GPUs 0-3) participate in distributed training
- Training metrics logged to W&B (if enabled)
- Model weights distributed across GPUs via ZeRO-2

#### 2. Cleanup Phase (All 4 GPUs)
```
🧹 Cleaning up disk space after training...
🧹 Cleared pip cache
🧹 Cleared temporary files
🧹 Cleared ./llava_video_finetune
```
- All ranks run cleanup to free disk space
- Removes training artifacts, cache files, and checkpoints
- Prepares for model upload

#### 3. Model Save Phase (Only Rank 0)
```
Rank 0: "💾 Saving trained model..."
Ranks 1, 2, 3: "⏭️ Skipping model save (only rank 0 saves)" → exit
```
- **Only rank 0 saves** to prevent 4x disk usage
- Ranks 1-3 skip saving and exit cleanly
- Prevents "disk quota exceeded" errors

#### 4. Upload to Hub (Only Rank 0)
```
📁 Using temporary directory: /tmp/llava_model_xxxxx
💾 Saving model to temporary directory...
💾 Saving processor to temporary directory...
📤 Uploading model to Hub...

Processing Files (32 / 32): 100% | 16.3GB / 16.3GB
New Data Upload: 100% | 16.3GB / 16.3GB, 412kB/s
```

### Understanding "New Data Upload"

The `upload_folder` function performs **intelligent deduplication**:

**Phase 1: Processing Files**
```
Processing Files (32 / 32): 16.3GB
```
- Scans all 32 files in the temp directory
- Computes SHA256 hash for each file
- Compares hashes with what's already on HuggingFace Hub

**Phase 2: New Data Upload**
```
New Data Upload: 16.3GB / 16.3GB
```
- Only uploads files that **don't already exist** on the Hub
- Skips files with matching hashes (already uploaded)

**Benefits:**

| Benefit | Explanation |
|---------|-------------|
| **Deduplication** | If you update the model but tokenizer/config unchanged, only model weights upload |
| **Resume on Failure** | If upload interrupted, re-running skips already-uploaded files |
| **Bandwidth Savings** | Common files (processor, tokenizer) often unchanged between runs |
| **Time Savings** | Skip re-uploading unchanged files |
| **Cost Savings** | Less data transfer if using metered connections |

**First Training Run:**
```
New Data Upload: 16.3GB / 16.3GB (100%)
```
All files uploaded (first time)

**Second Training Run (same base model):**
```
Processing Files (32 / 32): 16.3GB
New Data Upload: 7.2GB / 16.3GB  ← Only model weights changed!

Skipped (unchanged):
- tokenizer.json
- special_tokens_map.json
- preprocessor_config.json
- config.json
```

### Disk Space Management

**During Training:**
- Peak usage: ~29.9GB on root (for temp model files)
- Training creates NO local checkpoints (`save_strategy="no"`)
- Only temporary model files during upload

**After Training:**
- Temp directory auto-deleted after upload
- Returns to baseline disk usage
- No lingering checkpoint files

## 🎬 Example Usage After Training

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load your fine-tuned model
processor = AutoProcessor.from_pretrained("your-username/llava-video-text-model")
model = LlavaForConditionalGeneration.from_pretrained(
    "your-username/llava-video-text-model",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare conversation with video frames
conversation = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What is happening in this video?"},
        {"type": "image"},
        {"type": "image"},
        {"type": "image"},
        {"type": "image"},
        {"type": "image"}
    ]
}]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Extract frames from your video (5 frames)
video_frames = [...]  # List of 5 PIL.Image objects

# Process and generate
inputs = processor(images=video_frames, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)

print(response)
```

---

**💡 Key Takeaway:** This trainer processes **actual video frames** for **vision-language understanding**, not just text metadata. It generates its own DeepSpeed config with `"auto"` values for easier maintenance.
