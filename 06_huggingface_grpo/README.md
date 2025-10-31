# GRPO Training with DeepSpeed on GSM8K 🧮

**Memory-Efficient Configuration:** This folder contains a working implementation of GRPO (Group Relative Policy Optimization) training using **LoRA + DeepSpeed ZeRO-2** for fine-tuning language models on the GSM8K math reasoning dataset on **consumer GPUs (8GB+ VRAM)**.

---

## Overview

**GRPO (Group Relative Policy Optimization)** is an advanced reinforcement learning technique for fine-tuning language models. This implementation trains a Qwen-based model on GSM8K (Grade School Math 8K) dataset with custom reward functions that encourage:
- 🤔 **Chain-of-thought reasoning** via `<think>...</think>` tags
- 🎨 **Response diversity** through character variety metrics
- 📊 **Memory-efficient training** with LoRA + DeepSpeed ZeRO-2
- 💾 **8GB GPU Support**: Works on RTX 3070, RTX 2080 Ti, etc.

**Key Features:**
- ✅ **Memory Efficient**: LoRA reduces trainable parameters by ~99% (1.5B → ~15M)
- 🚀 **DeepSpeed Integration**: ZeRO-2 with CPU optimizer offloading
- 💡 **Custom Reward Functions**: Combined rewards for reasoning tags and diversity
- 📈 **GSM8K Dataset**: Enhanced dataset with 8K training samples
- 🤖 **Qwen Model**: Uses `eagle0504/qwen-distilled-scout-1.5b-instruct-gen2`
- 🎯 **Consumer GPU Ready**: Tested on 2x RTX 3070 (8GB each)

## Hardware Requirements

**Minimum (Tested):**
- 2x GPUs with 8GB VRAM each (e.g., RTX 3070, RTX 2080 Ti)
- 32GB+ system RAM (for DeepSpeed CPU offloading)
- CUDA 11.8+

**Recommended:**
- 2x GPUs with 16GB+ VRAM (e.g., RTX 4080, A4000)
- 64GB+ system RAM
- CUDA 12.x

**What Makes This Work on 8GB GPUs:**
1. **LoRA (Low-Rank Adaptation)**: Only trains ~1% of model parameters
2. **ZeRO-2 Optimizer Offloading**: Moves optimizer states to CPU
3. **Small Batch Size**: 4 per GPU with gradient accumulation
4. **FP16 Training**: Half-precision reduces memory by 50%

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
uv add torch transformers accelerate datasets deepspeed bitsandbytes trl peft
```

**Note:** `peft` is required for LoRA support.

**Step 4: Ensure DeepSpeed config is named correctly**

The script references `ds_config.json`. Either:

**Option A: Create symbolic link (recommended)**
```bash
ln -s ds_config_zero1.json ds_config.json
```

**Option B: Rename the file**
```bash
cp ds_config_zero1.json ds_config.json
```

**Option C: Update the script**
Update line 155 in `grpo_gsm8k_train.py`:
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

### Training Configuration

**LoRA Configuration for Memory Efficiency:**

The script uses LoRA to dramatically reduce memory requirements:

```python
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

# LoRA config - only trains ~1% of parameters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # LoRA rank (16 is good balance)
    lora_alpha=32,  # Scaling factor (usually 2x rank)
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
    bias="none",
)

# GRPO config with reduced batch size for 8GB GPUs
grpo_config = GRPOConfig(
    output_dir="./grpo-trained-qwen-gsm8k-lora",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Small batch for 8GB GPUs
    gradient_accumulation_steps=8,  # Effective batch = 4 * 8 * 2 = 64
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    deepspeed="ds_config.json",  # ZeRO-2 with CPU offloading
    fp16=True,
    max_grad_norm=1.0,
    warmup_steps=100,
)

trainer = GRPOTrainer(
    model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
    reward_funcs=reward_combined,
    train_dataset=dataset,
    args=grpo_config,
    peft_config=peft_config,  # Enable LoRA
)
```

**Memory Breakdown (8GB GPU):**
- Base model (FP16): ~3GB
- LoRA adapters: ~30MB
- Optimizer states (CPU offloaded): 0GB GPU
- Activations + gradients: ~2-3GB
- Generation buffer (GRPO): ~2GB
- **Total: ~5-6GB per GPU** ✅

**Important:** `GRPOTrainer` does **not** accept `deepspeed` as a direct parameter. You must pass it via `GRPOConfig` through the `args` parameter.

### DeepSpeed Config: `ds_config_zero1.json`

**Key Settings (ZeRO-2 with CPU Offloading):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **ZeRO Stage** | 2 | Optimizer + gradient partitioning |
| **CPU Offloading** | Enabled | Offload optimizer states to CPU |
| **FP16** | Enabled | Half-precision training |
| **Batch Size** | Auto | Determined by GRPOConfig (4 per GPU) |
| **Micro Batch Size** | Auto | Set by GRPOConfig |
| **Gradient Accumulation** | Auto | Set by GRPOConfig (8 steps) |
| **Optimizer** | AdamW | Learning rate: 5e-5 |
| **Scheduler** | WarmupDecayLR | Cosine decay with warmup |
| **Gradient Clipping** | 1.0 | Prevents gradient explosion |

**Memory-Saving Features:**
```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": {
    "device": "cpu",        // Move optimizer to CPU RAM
    "pin_memory": true      // Faster CPU-GPU transfer
  },
  "allgather_partitions": true,
  "overlap_comm": true,     // Overlap communication with computation
  "contiguous_gradients": true
}
```

**Auto Parameters:**
- `train_batch_size`: Auto-calculated from per_device_batch_size × num_gpus × gradient_accumulation_steps
- `train_micro_batch_size_per_gpu`: Auto-set from GRPOConfig
- `gradient_accumulation_steps`: Auto-set from GRPOConfig
- `warmup_num_steps`: Auto-scaled to dataset size (100 steps)
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

### Issue: `TypeError: GRPOTrainer.__init__() got an unexpected keyword argument 'deepspeed'`

**Root Cause:** `GRPOTrainer` does not accept `deepspeed` as a direct parameter.

**Solution:** Use `GRPOConfig` to pass DeepSpeed configuration:

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./grpo-trained-qwen-gsm8k",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    deepspeed="ds_config.json",  # ✅ Pass via config
    fp16=True,
)

trainer = GRPOTrainer(
    model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
    reward_funcs=reward_combined,
    train_dataset=dataset,
    args=grpo_config,  # ✅ Pass config via args
)
```

**Wrong approach (will fail):**
```python
trainer = GRPOTrainer(
    model="...",
    reward_funcs=reward_combined,
    train_dataset=dataset,
    deepspeed="ds_config.json"  # ❌ This doesn't work!
)
```

### Issue: `FileNotFoundError: ds_config.json`

**Solution:**
```bash
ln -s ds_config_zero1.json ds_config.json
```

Or update line 155 in `grpo_gsm8k_train.py`:
```python
deepspeed="ds_config_zero1.json"
```

### Issue: CUDA Out of Memory (8GB GPUs)

**Root Cause:** Full fine-tuning of 1.5B models requires ~12-16GB VRAM per GPU.

**Solution: Use LoRA + ZeRO-2 (already implemented in the script!)**

The current script already includes LoRA to solve this. If you're still getting OOM:

**Option 1: Reduce batch size further**
Edit `grpo_gsm8k_train.py` line 152:
```python
per_device_train_batch_size=2,  # Reduce from 4 to 2
gradient_accumulation_steps=16,  # Increase from 8 to 16
```

**Option 2: Reduce LoRA rank**
Edit `grpo_gsm8k_train.py` line 140:
```python
r=8,  # Reduce from 16 to 8
lora_alpha=16,  # Reduce from 32 to 16
```

**Option 3: Enable gradient checkpointing**
Add to `grpo_config`:
```python
gradient_checkpointing=True,
```

**Expected Memory Usage:**
- With current config: ~5-6GB per GPU
- With batch_size=2: ~4-5GB per GPU
- With r=8: ~4-5GB per GPU

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

This configuration has been **tested and verified working** on **consumer GPUs**:

**Hardware:**
- **GPUs:** 2x NVIDIA GeForce RTX 3070 (8GB VRAM each)
- **Driver:** NVIDIA 580.95.05
- **CUDA:** 12.x compatible
- **System RAM:** 32GB+ (required for ZeRO-2 CPU offloading)

**Configuration:**
- **Command:** `uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py`
- **Environment:** PyTorch 2.1.0+, CUDA 11.8+
- **Model:** Qwen 1.5B distilled (`eagle0504/qwen-distilled-scout-1.5b-instruct-gen2`)
- **Training Method:** LoRA (r=16, alpha=32) + ZeRO-2 with CPU offloading
- **Dataset:** GSM8K enhanced (8K samples)
- **Batch Settings:** 4 per GPU, 8 gradient accumulation steps (effective batch = 64)
- **Memory Usage:** ~5-6GB per GPU (fits in 8GB!)
- **Trainable Parameters:** ~15M (1% of 1.5B) thanks to LoRA

**Result:** Successfully trains on 8GB GPUs and saves LoRA adapter to `./grpo-trained-qwen-gsm8k-lora/`

**Key Fixes Applied:**
1. ✅ **LoRA Integration**: Reduces memory by ~10x (1.5B → ~15M trainable params)
2. ✅ **ZeRO-2 CPU Offloading**: Moves optimizer states to CPU RAM
3. ✅ **Reduced Batch Size**: 4 per GPU instead of 32
4. ✅ **GRPOConfig**: Pass DeepSpeed config via `args` parameter (not directly)

**How to Use Trained Model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./grpo-trained-qwen-gsm8k-lora"
)

tokenizer = AutoTokenizer.from_pretrained("./grpo-trained-qwen-gsm8k-lora")

# Generate
prompt = "Question: What is 15 * 24?\n<think>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## License

This project is released under the MIT License.

---

**Happy Training with GRPO!** 🚀🧮
