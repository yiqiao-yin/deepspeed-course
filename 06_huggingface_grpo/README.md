# GRPO Training with DeepSpeed on GSM8K 🧮

**Tested Configuration:** This folder contains a working implementation of GRPO (Group Relative Policy Optimization) training using DeepSpeed ZeRO Stage 1 for fine-tuning language models on the GSM8K math reasoning dataset.

---

## Overview

**GRPO (Group Relative Policy Optimization)** is an advanced reinforcement learning technique for fine-tuning language models. This implementation trains a Qwen-based model on GSM8K (Grade School Math 8K) dataset with custom reward functions that encourage:
- 🤔 **Chain-of-thought reasoning** via `<think>...</think>` tags
- 🎨 **Response diversity** through character variety metrics
- 📊 **Distributed training** with DeepSpeed ZeRO-1 optimization

**Key Features:**
- ✅ **Tested and Working**: Successfully trained with `uv run deepspeed`
- 🚀 **DeepSpeed Integration**: ZeRO Stage 1 with FP16 for efficient training
- 💡 **Custom Reward Functions**: Combined rewards for reasoning tags and diversity
- 📈 **GSM8K Dataset**: Enhanced dataset with 8K training samples
- 🤖 **Qwen Model**: Uses `eagle0504/qwen-distilled-scout-1.5b-instruct-gen2`

---

## Files

```
06_huggingface_grpo/
├── grpo_gsm8k_train.py         # Main training script (GRPO + DeepSpeed)
├── ds_config_zero1.json         # DeepSpeed ZeRO-1 configuration
├── archive/                     # Old experimental scripts
│   ├── README.md               # Original setup instructions
│   ├── ds_config_zero2.json    # Alternative ZeRO-2 config
│   ├── main.py                 # Legacy entry point
│   ├── train_ds.py             # Basic training script
│   ├── train_ds_grpo.py        # Earlier GRPO version
│   ├── train_ds_r1.py          # R1 training variant
│   ├── train_ds_sft.py         # SFT training script
│   └── upload_to_hf.py         # HuggingFace upload utility
└── README.md                    # This file
```

---

## Quick Start

### Prerequisites

**Environment:**
- Python 3.10+
- CUDA 11.8+ or 12.x
- Recommended image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

**Install `uv` (if not already installed):**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Installation

**Step 1: Navigate to folder**
```bash
cd 06_huggingface_grpo
```

**Step 2: Initialize project (if needed)**
```bash
uv init .
```

**Step 3: Install dependencies**
```bash
uv add torch transformers accelerate datasets deepspeed bitsandbytes trl
```

**Step 4: Create symbolic link for config (Important!)**

The script references `ds_config.json` but the file is named `ds_config_zero1.json`. Create a symlink:

```bash
ln -s ds_config_zero1.json ds_config.json
```

Or alternatively, update line 135 in `grpo_gsm8k_train.py`:
```python
deepspeed="ds_config_zero1.json"  # Instead of "ds_config.json"
```

---

## Usage

### Basic Training (Single GPU)

```bash
uv run deepspeed --num_gpus=1 grpo_gsm8k_train.py
```

### Multi-GPU Training (Recommended)

```bash
uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py
```

### Expected Output

```
INFO:__main__:Loading dataset...
INFO:__main__:Formatting dataset...
INFO:__main__:Initializing trainer with DeepSpeed...
INFO:__main__:Starting training...
[Training progress logs...]
INFO:__main__:Training complete.
INFO:__main__:Saving model and tokenizer to ./grpo-trained-qwen-gsm8k
INFO:__main__:Model and tokenizer saved.
```

**Output Model:** `./grpo-trained-qwen-gsm8k/`

---

## Configuration Details

### Training Script: `grpo_gsm8k_train.py`

**Model:**
- Base: `eagle0504/qwen-distilled-scout-1.5b-instruct-gen2`
- Architecture: Qwen 1.5B (distilled variant)
- Task: Math reasoning with chain-of-thought

**Dataset:**
- Source: `eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1`
- Size: 8K training samples
- Format: Question → CoT reasoning → Answer

**Reward Functions:**

1. **`reward_has_think_tags()`**: Returns 1.0 if completion contains `<think>...</think>`, else 0.0
2. **`reward_num_unique_chars()`**: Counts unique characters (promotes diversity)
3. **`reward_combined()`**: Weighted combination (α=0.7 for think tags, β=0.3 for diversity)

**Example Formatted Prompt:**
```
Question: A store sells apples for $1.50 each. If you buy 12 apples, how much will you pay?
<think>12 apples × $1.50 per apple = $18.00</think>
Answer: $18.00
```

### DeepSpeed Config: `ds_config_zero1.json`

**Key Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **ZeRO Stage** | 1 | Optimizer state partitioning |
| **FP16** | Enabled | Half-precision training |
| **Batch Size** | Auto | Dynamically determined by TRL |
| **Optimizer** | AdamW | Learning rate: 5e-5 |
| **Scheduler** | WarmupDecayLR | Cosine decay with warmup |
| **Gradient Clipping** | 1.0 | Prevents gradient explosion |
| **Checkpointing** | CPU | Activation checkpointing on CPU |

**Auto Parameters:**
- `train_batch_size`: Auto-detected by TRL trainer
- `train_micro_batch_size_per_gpu`: Auto-tuned for GPU memory
- `gradient_accumulation_steps`: Auto-calculated
- `warmup_num_steps`: Auto-scaled to dataset size
- `total_num_steps`: Auto-determined from epochs

**Memory Optimization:**
```json
"checkpoint": {
  "partition_activations": true,
  "contiguous_memory_optimization": true,
  "cpu_checkpointing": true
}
```

This configuration enables training 1.5B models on GPUs with ~16GB VRAM.

---

## Training Process

### Step 1: Dataset Loading & Formatting

```python
dataset = load_dataset(
    "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1",
    split="train"
)
dataset = dataset.map(format_gsm8k_example)
```

### Step 2: Trainer Initialization

```python
trainer = GRPOTrainer(
    model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
    reward_funcs=reward_combined,
    train_dataset=dataset,
    deepspeed="ds_config.json"  # ← Must match config file name
)
```

### Step 3: Training & Saving

```python
trainer.train()
trainer.model.save_pretrained("./grpo-trained-qwen-gsm8k")
trainer.tokenizer.save_pretrained("./grpo-trained-qwen-gsm8k")
```

---

## Expected Results

**Training Metrics:**
- **Reward Convergence**: Should increase as model learns to use `<think>` tags
- **Loss Decrease**: Policy loss should gradually decrease
- **Memory Usage**: ~10-14GB per GPU with ZeRO-1 (1.5B model)
- **Training Time**: ~2-4 hours on 2x A100 GPUs (8K samples)

**Model Outputs After Training:**
```
Question: If a train travels 60 mph for 3 hours, how far does it go?
<think>Distance = Speed × Time = 60 mph × 3 hours = 180 miles</think>
Answer: 180 miles
```

---

## Troubleshooting

### Issue: `FileNotFoundError: ds_config.json`

**Solution:**
```bash
ln -s ds_config_zero1.json ds_config.json
```

Or update line 135 in `grpo_gsm8k_train.py`:
```python
deepspeed="ds_config_zero1.json"
```

### Issue: CUDA Out of Memory

**Solution 1: Enable ZeRO-2 (stronger memory optimization)**
```bash
cp archive/ds_config_zero2.json ds_config.json
```

**Solution 2: Reduce batch size manually**
Edit `ds_config_zero1.json`:
```json
"train_micro_batch_size_per_gpu": 1,  # Instead of "auto"
"gradient_accumulation_steps": 16
```

### Issue: Slow Training

**Solution 1: Use more GPUs**
```bash
uv run deepspeed --num_gpus=4 grpo_gsm8k_train.py
```

**Solution 2: Enable BF16 instead of FP16 (if supported)**
Edit `ds_config_zero1.json`:
```json
"fp16": {"enabled": false},
"bf16": {"enabled": true}
```

### Issue: Import Error for `trl`

**Solution:**
```bash
uv add trl  # Ensure TRL is installed
uv pip list | grep trl  # Verify installation
```

---

## Advanced Usage

### Custom Reward Function

Modify `reward_combined()` to experiment with different reward weights:

```python
def reward_combined(completions: List[str], **kwargs) -> List[float]:
    reward_think = reward_has_think_tags(completions, **kwargs)
    reward_unique = reward_num_unique_chars(completions, **kwargs)

    alpha = 0.9  # Increase weight for <think> tags
    beta = 0.1   # Decrease weight for diversity

    return [alpha * r1 + beta * r2 for r1, r2 in zip(reward_think, reward_unique)]
```

### Different Dataset

Replace GSM8K with another dataset:

```python
dataset = load_dataset("your-dataset-name", split="train")
dataset = dataset.map(your_format_function)
```

### Different Model

Change the base model:

```python
trainer = GRPOTrainer(
    model="meta-llama/Llama-2-7b-hf",  # Or any HuggingFace model
    reward_funcs=reward_combined,
    train_dataset=dataset,
    deepspeed="ds_config_zero1.json"
)
```

---

## Archive Contents

The `archive/` folder contains experimental scripts and configurations from earlier development:

- **`README.md`**: Original `uv` project setup instructions
- **`ds_config_zero2.json`**: ZeRO Stage 2 config (more memory efficient)
- **`main.py`**: Simple training entry point
- **`train_ds.py`**: Basic DeepSpeed training example
- **`train_ds_grpo.py`**: Earlier GRPO implementation
- **`train_ds_r1.py`**: R1 (reasoning) training variant
- **`train_ds_sft.py`**: Supervised fine-tuning script
- **`upload_to_hf.py`**: Utility to upload models to HuggingFace Hub

These files are preserved for reference but are **not maintained**.

---

## Resources

**GRPO & Reinforcement Learning:**
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300) (if applicable)
- [Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)

**DeepSpeed:**
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Optimization](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/docs/config-json/)

**GSM8K Dataset:**
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [Enhanced GSM8K Dataset](https://huggingface.co/datasets/eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1)

**Tools:**
- [uv Documentation](https://docs.astral.sh/uv/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Testing Confirmation ✅

This configuration has been **tested and verified working** with:
- **Command:** `uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py`
- **Environment:** RunPod with PyTorch 2.1.0, CUDA 11.8
- **Model:** Qwen 1.5B distilled
- **Dataset:** GSM8K enhanced (8K samples)
- **Result:** Successfully trained and saved to `./grpo-trained-qwen-gsm8k/`

---

## License

This project is released under the MIT License.

---

**Happy Training with GRPO!** 🚀🧮
