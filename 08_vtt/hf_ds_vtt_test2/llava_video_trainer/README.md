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
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb

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

This script **actually processes video frames**:

```python
def download_and_process_video_frames(self, video_url, num_frames):
    # Downloads video
    # Extracts frames as PIL Images
    # Returns List[PIL.Image]
    return [frame1, frame2, frame3, frame4, frame5]
```

**Features:**
- Extracts `num_frames` frames from each video (default: 5)
- Returns PIL Image objects for each frame
- Handles both image URLs and video URLs
- Fallback to placeholder if download fails

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
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb

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
pip install torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb

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
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb

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
uv add torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb
```

### Option 2: Using `pip`

```bash
pip install torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb
```

**Key dependencies:**
- `pillow` - For image processing (PIL)
- `requests` - For downloading video frames
- `transformers` - LLaVA model support
- `wandb` - (Optional) For experiment tracking
- `torch` - PyTorch deep learning framework
- `deepspeed` - Distributed training optimization

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
