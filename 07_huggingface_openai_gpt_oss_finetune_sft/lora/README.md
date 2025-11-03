# GPT-OSS-20B Fine-tuning with LoRA 🚀

**Memory-Efficient Configuration:** This folder contains a complete implementation for fine-tuning **OpenAI GPT-OSS-20B** (20 billion parameters) on the **HuggingFaceH4/Multilingual-Thinking** dataset using **LoRA + DeepSpeed ZeRO-2**.

---

## Overview

**OpenAI GPT-OSS-20B** is a 20 billion parameter open-source language model. This implementation:
- 🎯 **Fine-tunes on multilingual reasoning** tasks
- 💾 **Uses LoRA** for parameter-efficient training (~99% fewer trainable parameters)
- 🚀 **DeepSpeed ZeRO-2** for distributed training across multiple GPUs
- 📊 **Optional W&B integration** for experiment tracking
- 🤗 **Optional HF Hub integration** for model sharing
- ⚡ **BF16 training** for memory and speed optimization

**Key Features:**
- ✅ **LoRA Integration**: Reduces trainable parameters from 20B → ~80M
- ✅ **Optional W&B**: Track experiments if WANDB_API_KEY is set
- ✅ **Optional HF Hub**: Push models if HF_TOKEN is set
- ✅ **Runs without secrets**: Trains and saves locally without tokens
- ✅ **DeepSpeed ZeRO-2**: Efficient multi-GPU training
- ✅ **BF16 Precision**: Faster training with lower memory usage

---

## Hardware Requirements

**Minimum (for 4 GPUs):**
- 4x GPUs with 24GB+ VRAM each (e.g., RTX 3090, RTX 4090, A5000)
- 128GB+ system RAM
- CUDA 11.8+

**Recommended (for optimal performance):**
- 4x GPUs with 40GB+ VRAM (e.g., A100, H100)
- 256GB+ system RAM
- CUDA 12.x

**Memory Breakdown (with LoRA + ZeRO-2):**
- Base model (BF16): ~40GB
- LoRA adapters: ~160MB
- Optimizer states (ZeRO-2 partitioned): ~10GB per GPU (split across 4 GPUs)
- Activations + gradients: ~8GB per GPU
- **Total: ~25-30GB per GPU** (fits in 32GB or 40GB GPUs)

---

## Files

```
lora/
├── train_ds.py          # Main training script (LoRA + DeepSpeed + W&B)
├── ds_config.json       # DeepSpeed ZeRO-2 configuration
├── run_deepspeed.sh     # SLURM batch script for HPC clusters
└── README.md            # This file
```

---

## Quick Start

### Prerequisites

**Environment:**
- Python 3.10+
- CUDA 11.8+ or 12.x
- Recommended: Use `uv` for fast dependency management

**Install `uv` (if not already installed):**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

---

### Installation

**Step 1: Navigate to the lora folder**
```bash
cd 07_huggingface_openai_gpt_oss_finetune_sft/lora
```

**Step 2: Initialize project**
```bash
uv init .
```

**Step 3: Install dependencies**
```bash
# Core dependencies
uv add torch transformers accelerate datasets deepspeed peft trl

# Required: TensorBoard for training logs
uv add tensorboard

# Required: Fast model downloads from HuggingFace
uv add hf_transfer

# Optional: Install W&B for experiment tracking
uv add wandb

# Optional: Install HuggingFace Hub for model uploads
uv add huggingface_hub
```

**Step 4: Verify DeepSpeed config exists**
```bash
ls -l ds_config.json
```

---

## Usage

### Option 1: Local Training (Interactive)

**Basic training (no W&B, no HF Hub):**
```bash
uv run deepspeed --num_gpus=4 train_ds.py
```

**With W&B tracking:**
```bash
# Get your API key from https://wandb.ai/authorize
export WANDB_API_KEY=your_wandb_key_here
uv run deepspeed --num_gpus=4 train_ds.py
```

**With HF Hub pushing:**
```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_hf_token_here
export HF_USER=your_hf_username
uv run deepspeed --num_gpus=4 train_ds.py
```

**With both W&B and HF Hub:**
```bash
export WANDB_API_KEY=your_wandb_key_here
export HF_TOKEN=your_hf_token_here
export HF_USER=your_hf_username
uv run deepspeed --num_gpus=4 train_ds.py
```

**Control options:**
```bash
# Disable pushing to HF Hub (even if token is set)
export PUSH_TO_HUB=false

# Disable evaluation after training
export RUN_EVALUATION=false

uv run deepspeed --num_gpus=4 train_ds.py
```

---

### Option 2: SLURM Batch Jobs (CoreWeave/HPC)

For HPC cluster environments like CoreWeave, use the provided SLURM batch script.

**Step 1: Edit the SLURM script**
```bash
nano run_deepspeed.sh
```

Update the following (all optional):
- `WANDB_API_KEY=<ENTER_KEY_HERE>` → Your W&B API key (or leave as-is to skip)
- `HF_TOKEN=<ENTER_KEY_HERE>` → Your HF token (or leave as-is to skip)
- `HF_USER=your_hf_username` → Your HuggingFace username (if pushing)
- `source ~/myenv/bin/activate` → Path to your virtual environment
- `#SBATCH --partition=h200-low` → Your cluster partition name

**Step 2: Submit the job**
```bash
sbatch run_deepspeed.sh
```

**Step 3: Monitor the job**
```bash
# Check job status
squeue -u $USER

# View real-time logs
tail -f logs/gpt_oss_lora_<job_id>.out
```

---

## Configuration Details

### Training Script: `train_ds.py`

**Model:**
- Base: `openai/gpt-oss-20b`
- Parameters: 20 billion
- Architecture: GPT-style transformer

**Dataset:**
- Source: `HuggingFaceH4/Multilingual-Thinking`
- Task: Multilingual reasoning with chain-of-thought
- Format: Conversational messages with system/user/assistant roles

**LoRA Configuration:**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA rank
    lora_alpha=16,          # Scaling factor
    target_parameters=[     # Target specific MLP expert layers
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
    lora_dropout=0.0,       # No dropout for target_parameters
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Alternative LoRA Config (attention layers):**
```python
# Uncomment in train_ds.py to use this instead
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Training Hyperparameters:**
```python
from trl import SFTConfig

training_args = SFTConfig(
    learning_rate=2e-4,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # Effective batch size = 64
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    gradient_checkpointing=True,
    deepspeed="./ds_config.json",
)
```

**W&B Integration:**
- Automatically detected if `WANDB_API_KEY` is set
- Logs training metrics, loss, learning rate, etc.
- Gracefully disabled if key is not set
- Run name: `gpt-oss-20b-multilingual-{username}`

**HF Hub Integration:**
- Automatically detected if `HF_TOKEN` is set
- Pushes model to `{HF_USER}/gpt-oss-20b-multilingual-reasoner-lora`
- Can be disabled with `PUSH_TO_HUB=false`
- Gracefully disabled if token is not set

---

### DeepSpeed Config: `ds_config.json`

**Key Settings (ZeRO Stage 2 with BF16):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **ZeRO Stage** | 2 | Optimizer + gradient partitioning across GPUs |
| **BF16** | Enabled | Brain float 16 for memory savings |
| **CPU Offloading** | None | Keep optimizer on GPU (faster) |
| **Gradient Accumulation** | Auto | Set by SFTConfig (8 steps) |
| **Optimizer** | AdamW | With auto learning rate |
| **Scheduler** | WarmupLR | Cosine with minimum learning rate |

**Memory-Saving Features:**
```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "overlap_comm": true,           // Overlap communication with computation
  "reduce_scatter": true,         // Efficient gradient reduction
  "contiguous_gradients": true,   // Memory-efficient gradient storage
  "offload_optimizer": {
    "device": "none"              // Keep optimizer on GPU
  }
}
```

**BF16 Configuration:**
```json
"bf16": {
  "enabled": true                 // Brain float 16 for faster training
}
```

**Why BF16 instead of FP16:**
- Better numerical stability for large models (20B parameters)
- No loss scaling required
- Wider dynamic range (same exponent bits as FP32)
- Supported on Ampere+ GPUs (A100, H100, RTX 3090, RTX 4090)

---

## Training Process

### Step 1: Dataset Loading & Preprocessing

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
# Automatically formats conversations using tokenizer's chat template
```

**Example conversation format:**
```json
{
  "messages": [
    {"role": "system", "content": "reasoning language: German"},
    {"role": "user", "content": "¿Cuál es el capital de Australia?"},
    {"role": "assistant", "content": "Die Hauptstadt von Australien ist Canberra..."}
  ]
}
```

### Step 2: Model Initialization with LoRA

```python
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    use_cache=False  # Required for gradient checkpointing
)

# Apply LoRA
from peft import get_peft_model
peft_model = get_peft_model(model, lora_config)

# Print trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: ~80M || all params: 20B || trainable%: ~0.4%
```

### Step 3: Training with SFTTrainer

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./gpt-oss-20b-multilingual-reasoner-lora")
```

### Step 4: Evaluation (Optional)

```python
# Load and merge LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
model = PeftModel.from_pretrained(base_model, "./gpt-oss-20b-multilingual-reasoner-lora")
model = model.merge_and_unload()  # Merge LoRA weights into base model

# Generate response
messages = [
    {"role": "system", "content": "reasoning language: German"},
    {"role": "user", "content": "¿Cuál es el capital de Australia?"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
```

---

## Expected Results

**Training Metrics:**
- **Training time**: ~3-4 hours on 4x A100 GPUs (10 epochs)
- **Loss convergence**: Should decrease from ~2.5 to ~1.2
- **Memory usage**: ~25-30GB per GPU with ZeRO-2
- **Throughput**: ~5-8 samples/sec (depends on GPU)

**Model Outputs After Training:**
```
User: ¿Cuál es el capital de Australia?
System: reasoning language: German
Assistant: Die Hauptstadt von Australien ist Canberra, nicht Sydney wie viele denken.
```

The model should demonstrate:
- Multilingual reasoning (answering in German when prompted)
- Accurate factual knowledge
- Natural conversational flow

---

## Troubleshooting

### Issue: `ImportError: No module named 'wandb'`

**Solution:**
```bash
uv add wandb
```
Or skip W&B by not setting `WANDB_API_KEY`.

### Issue: `ValueError: DeepSpeed config mismatch`

**Root Cause:** DeepSpeed config has hard-coded values that don't match training arguments.

**Solution:** The `ds_config.json` already uses "auto" for most parameters. If you see this error, ensure you're using the provided config file.

### Issue: CUDA Out of Memory

**Root Cause:** 20B model + LoRA requires ~25-30GB per GPU with ZeRO-2.

**Solution 1: Reduce batch size**
Edit `train_ds.py` line 265:
```python
per_device_train_batch_size=1,  # Reduce from 2 to 1
gradient_accumulation_steps=16,  # Increase from 8 to 16
```

**Solution 2: Use fewer GPUs with smaller batch**
```bash
# Use 2 GPUs instead of 4
uv run deepspeed --num_gpus=2 train_ds.py
```

**Solution 3: Reduce LoRA rank**
Edit `train_ds.py` line 150:
```python
r=4,            # Reduce from 8 to 4
lora_alpha=8,   # Reduce from 16 to 8
```

### Issue: `FileNotFoundError: ds_config.json`

**Solution:**
Ensure you're running the script from the `lora` folder:
```bash
cd 07_huggingface_openai_gpt_oss_finetune_sft/lora
uv run deepspeed --num_gpus=4 train_ds.py
```

### Issue: Model fails to load - authentication error

**Solution:**
The `openai/gpt-oss-20b` model may be gated. Get access:
1. Visit https://huggingface.co/openai/gpt-oss-20b
2. Accept the license agreement
3. Set your HF token:
```bash
export HF_TOKEN=your_token_here
```

### Issue: Training is very slow

**Solution 1: Check GPU utilization**
```bash
nvidia-smi -l 1  # Monitor GPU usage
```
GPUs should be at 90-100% utilization. If not:
- Check if CPU is bottleneck (increase `--cpus-per-task`)
- Reduce `dataloader_pin_memory` overhead
- Increase `gradient_accumulation_steps`

**Solution 2: Use BF16 (already enabled)**
The config already uses BF16 for faster training.

**Solution 3: Disable gradient checkpointing (if memory allows)**
Edit `train_ds.py` line 276:
```python
gradient_checkpointing=False,  # Faster but uses more memory
```

---

## Advanced Usage

### Custom Dataset

Replace the Multilingual-Thinking dataset with your own:

```python
# In train_ds.py, modify load_data():
def load_data():
    dataset = load_dataset("your-username/your-dataset", split="train")
    return dataset
```

**Dataset format requirements:**
- Must have a `messages` column with list of dicts
- Each message: `{"role": "system"/"user"/"assistant", "content": "..."}`

### Different Base Model

Change to a different model (must support chat template):

```python
# In train_ds.py, modify main():
model_name = "meta-llama/Llama-2-13b-chat-hf"  # Or any other model
```

**Note:** Adjust memory requirements based on model size.

### Modify LoRA Targets

To target different layers, edit `apply_lora()` in `train_ds.py`:

```python
# Option 1: Target all attention layers
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Option 2: Target all MLP layers
target_modules=["gate_proj", "up_proj", "down_proj"]

# Option 3: Target everything (memory intensive)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
```

### Hyperparameter Tuning

Edit `train_ds.py` training arguments:

```python
training_args = SFTConfig(
    learning_rate=1e-4,              # Try: 1e-5, 5e-5, 2e-4, 5e-4
    num_train_epochs=5,              # Try: 3, 5, 10, 20
    per_device_train_batch_size=4,   # Try: 1, 2, 4, 8
    gradient_accumulation_steps=4,   # Try: 2, 4, 8, 16
    max_length=1024,                 # Try: 512, 1024, 2048, 4096
    warmup_ratio=0.05,               # Try: 0.01, 0.03, 0.05, 0.10
)
```

### Using the Trained Model

**Load LoRA adapter:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./gpt-oss-20b-multilingual-reasoner-lora"
)

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Generate
messages = [
    {"role": "system", "content": "reasoning language: French"},
    {"role": "user", "content": "What is the capital of Japan?"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0])
print(response)
```

**Merge and save (optional):**
```python
# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./gpt-oss-20b-multilingual-merged")
tokenizer.save_pretrained("./gpt-oss-20b-multilingual-merged")

# Now you can load it like any other model
model = AutoModelForCausalLM.from_pretrained("./gpt-oss-20b-multilingual-merged")
```

---

## SLURM Commands Reference

**Submit job:**
```bash
sbatch run_deepspeed.sh
```

**Check job status:**
```bash
squeue -u $USER
```

**Monitor logs:**
```bash
tail -f logs/gpt_oss_lora_<job_id>.out
```

**Cancel job:**
```bash
scancel <job_id>
```

**View job history:**
```bash
sacct -u $USER
```

**Check estimated start time:**
```bash
squeue -j <job_id> --start
```

---

## Resources

**Model:**
- [OpenAI GPT-OSS-20B on HuggingFace](https://huggingface.co/openai/gpt-oss-20b)

**Dataset:**
- [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)

**Libraries:**
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [Weights & Biases](https://docs.wandb.ai/)
- [uv Documentation](https://docs.astral.sh/uv/)

---

## License

This project is released under the MIT License.

---

**Happy Fine-tuning!** 🚀
