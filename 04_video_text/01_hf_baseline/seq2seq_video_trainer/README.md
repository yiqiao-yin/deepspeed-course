# Seq2Seq Video Text Trainer 📝

Text-to-text model training for video metadata using **Seq2Seq** architecture (NLLB translation model as base).

## 🎯 Use Case

**Perfect for:** Text generation tasks where you reference videos by URL but don't need visual processing.

- Video metadata generation
- Text-based video descriptions
- Caption refinement without visual input
- Question-answering about video metadata
- Lightweight video text tasks

## 🏗️ Model Architecture

- **Model Type**: `AutoModelForSeq2SeqLM`
- **Base Model**: `facebook/nllb-200-distilled-600M` (600M parameters)
- **Input**: Text questions/prompts
- **Output**: Text responses
- **Format**: Simple triplets (video_url, question, caption)

## 📊 Data Format

Uses simple question-caption pairs with video URL references:

```json
{
  "video": "https://example.com/video.mp4",
  "question": "What is in this video?",
  "caption": "There is a cat in the video."
}
```

**Note:** Video URLs are stored for reference but **not processed visually**. This is text-to-text generation.

## ⚙️ DeepSpeed Configuration

**IMPORTANT:** This script **uses an external DeepSpeed config file** (`ds_config.json`).

The config file has **hardcoded values** that must be manually synced with `TrainingArguments`:

### `ds_config.json`
```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,              // Must match TrainingArguments!
      "betas": [0.9, 0.999],
      "weight_decay": 0.01
    }
  },
  "train_batch_size": 4,       // Must match calculated batch size!
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  // ...
}
```

### TrainingArguments (must sync)
```python
TrainingArguments(
    per_device_train_batch_size=1,  // Must match config!
    learning_rate=5e-5,              // Must match config!
    weight_decay=0.01,               // Must match config!
    # ...
)
```

**⚠️ Important:**
- If you change `learning_rate` in TrainingArguments, update `lr` in `ds_config.json`
- If you change `per_device_train_batch_size`, recalculate `train_batch_size` in config
- Formula: `train_batch_size = per_device_batch_size × num_gpus × grad_accum_steps`

**📝 File:** The `ds_config.json` file is **already provided** in this folder and must be present before running.

## 🎬 Video Processing

This script **does NOT process video frames**:

```python
def preprocess_function(self, examples):
    # Only processes text captions
    return self.processor(
        text=examples["caption"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
```

**What happens:**
- Video URLs are stored in the dataset
- Only text (questions and captions) are tokenized
- No PIL Image processing
- No frame extraction
- Lightweight and fast

## 🚀 Running Training

### Method 1: With run_training.sh (recommended)

```bash
# Set environment variables
export HF_USER_ID=your_username
export HF_TOKEN=your_token

# Make script executable
chmod +x run_training.sh

# Run with 4 GPUs (default)
./run_training.sh

# Or specify number of GPUs
./run_training.sh 2
```

The `run_training.sh` script will:
- Check for `ds_config.json` presence
- Validate environment variables
- Set up GPU visibility
- Launch training with logs

### Method 2: Direct Execution

```bash
# Set environment variables
export HF_USER_ID=your_username
export HF_TOKEN=your_token
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with DeepSpeed
deepspeed --num_gpus=4 video_text_trainer.py --deepspeed ds_config.json
```

### Method 3: Python Direct

```bash
# Set environment in script (line 591)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

python video_text_trainer.py
```

## 📦 Requirements

```bash
pip install torch datasets transformers trl huggingface_hub accelerate deepspeed
```

**Key difference from LLaVA:**
- ❌ No `pillow` needed (no image processing)
- ❌ No `requests` needed (no frame downloading)
- ✅ Lighter dependencies

## 🎓 Training Configuration

```python
TrainingArguments(
    output_dir="./video_finetune",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5,             # Must match ds_config.json!
    save_strategy="epoch",
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=100,
    weight_decay=0.01               # Must match ds_config.json!
)
```

**Important:**
- Standard checkpointing is enabled (`save_strategy="epoch"`)
- Best model loading is enabled
- Values must sync with `ds_config.json`

## 💾 Model Saving Strategy

This script uses **standard checkpointing** with best model loading:

```python
# During training
# - Checkpoints saved every epoch
# - Best model tracked based on eval_loss
# - Local checkpoints created in output_dir

# After training
trainer.model.push_to_hub(model_repo_id, token=hf_token)
self.processor.push_to_hub(model_repo_id, token=hf_token)
```

**Characteristics:**
- Creates local checkpoints during training
- Loads best model at end
- Standard model saving workflow
- Less disk-space optimized than LLaVA trainer

## 📊 Typical Resource Usage

- **Model Size**: ~2.4GB (600M parameters in FP16)
- **GPU Memory**: ~6-8GB per GPU (with ZeRO-2)
- **Training Time**: ~2-3 minutes (4 GPUs, 3 epochs, 4 samples)
- **Disk Space**: Moderate (checkpoints created)

## 🆚 When to Use This vs LLaVA Trainer

**Use Seq2Seq Video Trainer when:**
- ✅ Text-to-text generation tasks
- ✅ Video metadata processing (not visual content)
- ✅ Smaller models preferred (< 1B params)
- ✅ Limited GPU resources (6-8GB per GPU is enough)
- ✅ Faster training/iteration cycles
- ✅ Simple text-based video understanding

**Use LLaVA Trainer when:**
- ✅ Need actual visual understanding of videos
- ✅ Multi-frame video analysis required
- ✅ Vision-language conversation
- ✅ Have 16GB+ GPU memory
- ✅ Can afford longer training times

## 🔧 Troubleshooting

### Config Mismatch Errors
```
Error: batch size mismatch between TrainingArguments and DeepSpeed config
```

**Solution:** Ensure values match:
```python
# TrainingArguments
per_device_train_batch_size = 1

# ds_config.json
"train_micro_batch_size_per_gpu": 1  # Must match!
```

### Out of Memory
```bash
# Option 1: Reduce batch size
# In ds_config.json:
"train_micro_batch_size_per_gpu": 1

# In TrainingArguments:
per_device_train_batch_size = 1

# Option 2: Enable CPU offload
# In ds_config.json:
"cpu_offload": true
```

### Learning Rate Issues
```bash
# Make sure LR matches in BOTH places:
# TrainingArguments: learning_rate = 5e-5
# ds_config.json: "lr": 5e-5
```

## 📝 Output

**Dataset:** `{HF_USER_ID}/video-text-dataset`
**Model:** `{HF_USER_ID}/video-text-model`

Both are automatically pushed to HuggingFace Hub with READMEs.

## 🎬 Example Usage After Training

```python
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# Load your fine-tuned model
processor = AutoProcessor.from_pretrained("your-username/video-text-model")
model = AutoModelForSeq2SeqLM.from_pretrained("your-username/video-text-model")

# Generate text from video question
inputs = processor(text="What is in this video?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)

print(response)
# Output: "There is a cat in the video."
```

## 📋 Config File Management

### Current Config (ds_config.json)

```json
{
  "bf16": {"enabled": true},
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-05,
      "betas": [0.9, 0.999],
      "eps": 1e-08,
      "weight_decay": 0.01
    }
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1
}
```

### Modifying the Config

**If you change GPU count:**
```json
// For 2 GPUs:
"train_batch_size": 2  // 1 per_device × 2 GPUs × 1 grad_accum

// For 8 GPUs:
"train_batch_size": 8  // 1 per_device × 8 GPUs × 1 grad_accum
```

**If you enable gradient accumulation:**
```json
// For 4 gradient accumulation steps:
"gradient_accumulation_steps": 4,
"train_batch_size": 16  // 1 per_device × 4 GPUs × 4 grad_accum
```

## 🔄 Retry Logic

This script includes **exponential backoff retry** for HuggingFace Hub uploads:

```python
class RetryHandler:
    def exponential_backoff_retry(func, max_retries=5, ...):
        # Automatically retries on rate limiting (429 errors)
        # Delays: 1s, 2s, 4s, 8s, 16s...
        # Max delay: 60s
```

**Protects against:**
- Rate limiting from HuggingFace Hub
- Temporary network issues
- Upload failures

---

**💡 Key Takeaway:** This trainer is for **text-to-text** generation with video metadata references. It uses an **external config file** (`ds_config.json`) with hardcoded values that must be manually synced with `TrainingArguments`. Lighter and faster than LLaVA but no visual processing.
