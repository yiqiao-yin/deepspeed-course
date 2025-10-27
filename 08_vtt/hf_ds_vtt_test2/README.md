# Video-Text-to-Text Training with DeepSpeed 🎥📝

This directory contains **two separate video training frameworks** optimized for different use cases:

1. **LLaVA Video Trainer** - Vision-language understanding with actual video frame processing
2. **Seq2Seq Video Trainer** - Text-to-text generation with video metadata

---

## 📁 Directory Structure

```
08_vtt/hf_ds_vtt_test2/
├── llava_video_trainer/              # LLaVA vision-language trainer
│   ├── video_training_script.py      # LLaVA training script (generates config)
│   ├── run_training.sh               # SLURM/multi-GPU launcher
│   └── README.md                      # Comprehensive LLaVA guide
│
├── seq2seq_video_trainer/            # Seq2Seq text-to-text trainer
│   ├── video_text_trainer.py         # Seq2Seq training script (uses external config)
│   ├── ds_config.json                # External DeepSpeed config file
│   ├── run_training.sh               # SLURM/multi-GPU launcher
│   └── README.md                      # Comprehensive Seq2Seq guide
│
└── README.md                          # This file
```

---

## 🔍 Quick Comparison

| Feature | LLaVA Trainer | Seq2Seq Trainer |
|---------|---------------|-----------------|
| **Use Case** | Video understanding | Text generation about videos |
| **Model** | LLaVA (7B) | NLLB (600M) |
| **Architecture** | `LlavaForConditionalGeneration` | `AutoModelForSeq2SeqLM` |
| **Visual Processing** | ✅ Extracts & processes video frames | ❌ Text-only |
| **Frame Extraction** | ✅ 5 PIL Images per video | ❌ None |
| **Config Generation** | 🔧 Generated internally (`"auto"`) | 📄 External file (hardcoded) |
| **Config File** | Created by script | Must exist (`ds_config.json`) |
| **Model Size** | ~14GB (FP16) | ~2.4GB (FP16) |
| **GPU Memory** | ~16-20GB per GPU | ~6-8GB per GPU |
| **Disk Monitoring** | ✅ Built-in | ❌ None |
| **Dependencies** | PIL, requests, transformers, trl | transformers, trl only |
| **Training Time** | 10-15 min (4 GPUs, 3 epochs) | 2-3 min (4 GPUs, 3 epochs) |
| **Best For** | Visual video understanding | Lightweight text tasks |

---

## 🎯 Which One Should I Use?

### Use **LLaVA Video Trainer** (`llava_video_trainer/`) when:

✅ You need **actual video understanding** through visual frames
✅ Working on **vision-language tasks** (video QA, visual captioning)
✅ Need **multi-frame video analysis**
✅ Building **conversation systems** about visual content
✅ Have **16GB+ GPU memory** available
✅ Can afford **longer training times**

**Example tasks:**
- "What is the cat doing in this video?" (needs to see frames)
- "Describe the scene visually"
- "Count the objects in this video"
- Multi-turn conversations about video content

### Use **Seq2Seq Video Trainer** (`seq2seq_video_trainer/`) when:

✅ Working with **text-only** video tasks
✅ Processing **video metadata** (titles, descriptions, tags)
✅ Need a **smaller, faster** model
✅ Have **limited GPU resources** (6-8GB is enough)
✅ Want **quick iteration cycles**
✅ Don't need actual visual understanding

**Example tasks:**
- "Generate a description from video title"
- "Translate video captions"
- Text-based question answering
- Metadata refinement

---

## 🔧 DeepSpeed Configuration: Key Difference

### LLaVA Trainer: Auto-Generated Config

```python
# In video_training_script.py
def create_deepspeed_config(config_path="ds_config.json"):
    config = {
        "optimizer": {
            "params": {
                "lr": "auto",          # ✅ Syncs with TrainingArguments
                "betas": "auto",
                "weight_decay": "auto"
            }
        },
        "train_batch_size": "auto",    # ✅ Calculated automatically
        "train_micro_batch_size_per_gpu": "auto",
        # ...
    }
```

**✅ Advantages:**
- No manual synchronization needed
- Single source of truth (TrainingArguments)
- Less error-prone
- Easier to modify

**📝 When it's created:** Called in `main()` before training starts

---

### Seq2Seq Trainer: External Config File

```json
// ds_config.json (must exist before running)
{
  "optimizer": {
    "params": {
      "lr": 5e-5,                // ⚠️ Must match TrainingArguments!
      "weight_decay": 0.01
    }
  },
  "train_batch_size": 4,         // ⚠️ Must calculate correctly!
  "train_micro_batch_size_per_gpu": 1
}
```

**⚠️ Important:**
- Values are hardcoded
- Must manually sync with `TrainingArguments`
- Formula: `train_batch_size = per_device_batch × num_gpus × grad_accum_steps`
- If you change learning rate in TrainingArguments, update in config too!

**📝 Location:** `seq2seq_video_trainer/ds_config.json` (must exist)

---

## 🚀 Quick Start

### LLaVA Video Trainer

```bash
cd llava_video_trainer

# Set environment
export HF_USER_ID=your_username
export HF_TOKEN=your_token

# Run training (config generated automatically)
chmod +x run_training.sh
./run_training.sh 4  # 4 GPUs
```

**Output:**
- Dataset: `{HF_USER_ID}/llava-video-text-dataset`
- Model: `{HF_USER_ID}/llava-video-text-model`
- Config: `ds_config.json` (generated)

---

### Seq2Seq Video Trainer

```bash
cd seq2seq_video_trainer

# Set environment
export HF_USER_ID=your_username
export HF_TOKEN=your_token

# Run training (uses existing ds_config.json)
chmod +x run_training.sh
./run_training.sh 4  # 4 GPUs
```

**Output:**
- Dataset: `{HF_USER_ID}/video-text-dataset`
- Model: `{HF_USER_ID}/video-text-model`
- Config: `ds_config.json` (must exist beforehand)

---

## 📊 Resource Requirements

### LLaVA Trainer

```
Model: llava-hf/llava-interleave-qwen-7b-hf
├── Parameters: 7 billion
├── Model Size: ~14GB (FP16)
├── GPU Memory: 16-20GB per GPU (ZeRO-2)
├── Recommended GPUs: 2-4x A100/H100 (40GB+)
├── Training Time: 10-15 min (4 GPUs, 3 epochs, 4 samples)
├── Disk Space: Monitor actively (script has checks)
└── Dependencies: torch, deepspeed, transformers, trl, pillow, requests
```

### Seq2Seq Trainer

```
Model: facebook/nllb-200-distilled-600M
├── Parameters: 600 million
├── Model Size: ~2.4GB (FP16)
├── GPU Memory: 6-8GB per GPU (ZeRO-2)
├── Recommended GPUs: 2-4x RTX 3090/4090 or A40
├── Training Time: 2-3 min (4 GPUs, 3 epochs, 4 samples)
├── Disk Space: Moderate (checkpoints created)
└── Dependencies: torch, deepspeed, transformers, trl
```

---

## 🎓 Training Workflow Comparison

### LLaVA: Vision-Language Pipeline

```
1. Download video → Extract 5 frames → PIL Images
2. Create LLaVA conversation format with <image> tokens
3. Generate DeepSpeed config with "auto" values
4. Train with vision-language model
5. Push directly to Hub (no local checkpoints)
```

### Seq2Seq: Text-Only Pipeline

```
1. Reference video URL → No frame extraction
2. Create simple question-caption pairs
3. Use existing ds_config.json
4. Train with text-to-text model
5. Standard checkpointing + Hub push
```

---

## 📝 Data Format Examples

### LLaVA Format (Vision-Language)

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
      "content": [{"type": "text", "text": "There is a cat in the video."}]
    }
  ],
  "video_url": "https://example.com/video.mp4",
  "num_frames": 5
}
```

### Seq2Seq Format (Text-Only)

```json
{
  "video": "https://example.com/video.mp4",
  "question": "What is in this video?",
  "caption": "There is a cat in the video."
}
```

---

## 🔍 Detailed Documentation

Each subdirectory contains a **comprehensive README** with:

- ✅ Detailed architecture explanation
- ✅ Step-by-step setup instructions
- ✅ Config file management guide
- ✅ Training examples and commands
- ✅ Troubleshooting section
- ✅ Resource requirements
- ✅ Example usage after training

**Navigate to:**
- [`llava_video_trainer/README.md`](./llava_video_trainer/README.md) - LLaVA guide
- [`seq2seq_video_trainer/README.md`](./seq2seq_video_trainer/README.md) - Seq2Seq guide

---

## 🛠️ Troubleshooting Quick Reference

### LLaVA Trainer Issues

| Issue | Solution |
|-------|----------|
| Out of GPU memory | Reduce `num_frames` from 5 to 3 |
| Disk space errors | Script monitors automatically; clean `/tmp` |
| Processor errors | Script auto-fixes LLaVA processor issues |
| Rate limiting | Built-in exponential backoff retry |

### Seq2Seq Trainer Issues

| Issue | Solution |
|-------|----------|
| Batch size mismatch | Sync `ds_config.json` with `TrainingArguments` |
| Learning rate not applied | Check both config file and TrainingArguments |
| Config not found | Ensure `ds_config.json` exists in directory |
| Out of memory | Reduce batch size in both places |

---

## 📚 References

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

---

## 🎯 Summary

| Aspect | LLaVA | Seq2Seq |
|--------|-------|---------|
| **Config Strategy** | Generated with `"auto"` | External hardcoded file |
| **Visual Processing** | ✅ Real frames | ❌ None |
| **Complexity** | High | Low |
| **GPU Requirements** | 16-20GB | 6-8GB |
| **Use Case** | Video understanding | Text generation |
| **Setup Difficulty** | Easy (auto config) | Moderate (manual sync) |

---

**💡 Bottom Line:** Choose **LLaVA** for visual video understanding tasks and **Seq2Seq** for lightweight text-based video tasks. The key difference is that LLaVA processes actual video frames while Seq2Seq only handles text metadata. Configuration-wise, LLaVA auto-generates config while Seq2Seq uses an external file that must be manually synced.

**📖 For detailed instructions, see the README.md in each subdirectory!**
