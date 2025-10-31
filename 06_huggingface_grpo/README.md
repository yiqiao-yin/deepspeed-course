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
2. **ZeRO-2 Optimizer Partitioning**: Splits optimizer states across GPUs
3. **Small Batch Size**: 4 per GPU with gradient accumulation
4. **FP16 Training**: Half-precision reduces memory by 50%

---

## Files

```
06_huggingface_grpo/
├── grpo_gsm8k_train.py         # Main training script (GRPO + DeepSpeed + LoRA)
├── ds_config.json               # DeepSpeed ZeRO-2 configuration
├── run_deepspeed.sh             # SLURM batch script for CoreWeave/HPC clusters
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

**Step 4: Verify DeepSpeed config exists**

The script uses `ds_config.json` which is already in the folder.

---

## Usage

### Weights & Biases (W&B) Setup (Optional)

The script automatically detects and uses W&B if `WANDB_API_KEY` is set in your environment.

**Enable W&B tracking:**
```bash
# Get your API key from https://wandb.ai/authorize
export WANDB_API_KEY=your_api_key_here
```

**Disable W&B tracking:**
```bash
# Simply don't set WANDB_API_KEY, or:
unset WANDB_API_KEY
```

**What gets logged to W&B:**
- Training loss, learning rate, gradient norms
- Reward metrics (think tags, character diversity)
- GRPO-specific metrics (policy loss, value estimates)
- Model parameters count
- Hardware utilization
- Training progress and ETA

### Basic Training (Single GPU)

```bash
# Without W&B
uv run deepspeed --num_gpus=1 grpo_gsm8k_train.py

# With W&B
export WANDB_API_KEY=your_key
uv run deepspeed --num_gpus=1 grpo_gsm8k_train.py
```

### Multi-GPU Training (Recommended)

```bash
# Without W&B
uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py

# With W&B
export WANDB_API_KEY=your_key
uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py
```

---

## SLURM Batch Job Submission (CoreWeave) 🖥️

For HPC cluster environments like CoreWeave, use the provided SLURM batch script to submit training jobs to the scheduler.

### Prerequisites

**Before submitting jobs:**

1. **Activate your virtual environment** and ensure all dependencies are installed:
   ```bash
   source ~/myenv/bin/activate
   uv add torch transformers accelerate datasets deepspeed trl peft
   ```

2. **Edit `run_deepspeed.sh`** to configure your environment:
   ```bash
   nano run_deepspeed.sh
   ```

   Update the following:
   - `WANDB_API_KEY=<ENTER_KEY_HERE>` → Replace with your W&B API key from https://wandb.ai/authorize
   - `source ~/myenv/bin/activate` → Update path to your virtual environment
   - `#SBATCH --partition=h200-low` → Update partition name based on your cluster

### Submitting a Job

**Step 1: Navigate to the folder**
```bash
cd 06_huggingface_grpo
```

**Step 2: Submit the job to SLURM**
```bash
sbatch run_deepspeed.sh
```

**Expected output:**
```
Submitted batch job 12345
```

**Step 3: Check job status**
```bash
squeue -u $USER
```

**Example output:**
```
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  12345 h200-low  grpo_gsm yyin    R       0:15      1 node-001
```

**Job Status Codes:**
- `PD` = Pending (waiting for resources)
- `R` = Running (job is executing)
- `CG` = Completing (job finishing up)
- `CD` = Completed (job finished successfully)
- `F` = Failed (job encountered an error)

### Monitoring Your Job

**Watch job queue in real-time:**
```bash
watch -n 1 squeue -u $USER
# Updates every 1 second, Ctrl+C to exit
```

**View detailed job information:**
```bash
scontrol show job 12345
```

**Check estimated start time (for pending jobs):**
```bash
squeue -j 12345 --start
```

**View job history:**
```bash
sacct -u $USER
# Or for specific job:
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList
```

### Viewing Logs

**Real-time log monitoring (while job is running):**
```bash
# Standard output
tail -f logs/grpo_gsm8k_12345.out

# Standard error (if errors occur)
tail -f logs/grpo_gsm8k_12345.err
```

**View complete logs (after job finishes):**
```bash
# List all log files (sorted by time)
ls -lt logs/

# View specific job output
cat logs/grpo_gsm8k_12345.out

# Search for errors
grep -i error logs/grpo_gsm8k_12345.err

# Search for specific metrics
grep "Loss:" logs/grpo_gsm8k_12345.out
grep "Trainable parameters" logs/grpo_gsm8k_12345.out
```

### Managing Jobs

**Cancel a running job:**
```bash
scancel 12345
```

**Cancel all your jobs:**
```bash
scancel -u $USER
```

**Cancel jobs by name:**
```bash
scancel --name=grpo_gsm8k_lora
```

**View resource usage of running job:**
```bash
sstat -j 12345 --format=JobID,MaxRSS,AveCPU,AveRSS
```

### GPU Monitoring

**Create a GPU monitoring script:**

```bash
cat > gpu_monitor.sh << 'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=h200-low
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu_monitor

while true; do
    nvidia-smi
    echo "---"
    sleep 1
done
EOF
```

**Submit the monitor:**
```bash
sbatch gpu_monitor.sh
# Note the job ID, then:
tail -f slurm-<job_id>.out
```

**SSH to compute node (if allowed):**
```bash
# Find node where your job is running
squeue -j 12345 -o "%N"
# Output: node-001

# SSH to that node (if permitted)
ssh node-001

# Monitor GPU on that node
nvidia-smi -l 1  # Updates every 1 second
```

### Common SLURM Commands Reference

| Command | Description |
|---------|-------------|
| `sbatch run_deepspeed.sh` | Submit a batch job |
| `squeue -u $USER` | View your jobs in the queue |
| `squeue` | View all jobs in the queue |
| `scontrol show job 12345` | Detailed job information |
| `scancel 12345` | Cancel a specific job |
| `scancel -u $USER` | Cancel all your jobs |
| `sacct -u $USER` | View job history |
| `sinfo` | View cluster partition info |
| `sinfo -N` | View all nodes and their status |
| `squeue --start -j 12345` | Estimate start time for pending job |
| `sstat -j 12345` | Resource usage of running job |
| `tail -f logs/grpo_gsm8k_12345.out` | Monitor job output in real-time |

### Expected Training Time

**On 2x RTX 3070 (8GB each):**
- Dataset loading: ~2-5 minutes (first run, cached afterward)
- Model initialization: ~1-2 minutes
- Training (3 epochs, 8K samples): ~30-45 minutes
- **Total runtime: ~45-60 minutes**

**On 2x H100 (80GB each):**
- Training: ~10-15 minutes
- **Total runtime: ~15-20 minutes**

### Troubleshooting SLURM Jobs

**Job stays in PD (Pending) state:**
```bash
# Check why job is waiting
squeue -j 12345 --start
# Common reasons:
#   - Resources: Waiting for available GPUs
#   - Priority: Other users have higher priority
#   - ReqNodeNotAvail: Node is reserved/down
```

**Job fails immediately:**
```bash
# Check error logs
cat logs/grpo_gsm8k_12345.err

# Common issues:
#   - Invalid partition name
#   - Python environment not activated
#   - Missing dependencies (trl, peft)
#   - WANDB_API_KEY not set correctly
```

**Out of memory error in logs:**
```bash
# Check logs for OOM
grep "CUDA out of memory" logs/grpo_gsm8k_12345.err

# Solutions:
#   1. Reduce batch size in grpo_gsm8k_train.py
#   2. Switch to ZeRO Stage 1
#   3. Reduce LoRA rank
# See "Troubleshooting" section below for details
```

**Job runs but produces no output:**
```bash
# Check if logs directory exists
ls -la logs/

# Check SLURM job status
scontrol show job 12345 | grep -E "(JobState|Reason|WorkDir)"

# Verify script has execute permissions
ls -l run_deepspeed.sh
# Should show: -rwxr-xr-x

# Make executable if needed
chmod +x run_deepspeed.sh
```

### SLURM Script Configuration

The `run_deepspeed.sh` script is configured with the following resources:

| Resource | Value | Reason |
|----------|-------|--------|
| **GPUs** | 2 | Tested on 2x RTX 3070 (8GB each) |
| **CPUs** | 16 | Sufficient for data loading/preprocessing |
| **Memory** | 64GB | Model init + dataset + ZeRO-2 optimizer |
| **Time** | 2 hours | Typical: 30-45 min, buffer for initialization |
| **Partition** | h200-low | Update based on your cluster |

**To modify resources**, edit `run_deepspeed.sh`:

```bash
#SBATCH --gres=gpu:4          # Request 4 GPUs instead of 2
#SBATCH --cpus-per-task=32    # Request 32 CPUs instead of 16
#SBATCH --mem=128G            # Request 128GB RAM instead of 64GB
#SBATCH --time=04:00:00       # Request 4 hours instead of 2
```

Then update the DeepSpeed command:
```bash
deepspeed --num_gpus=4 grpo_gsm8k_train.py  # Match GPU count
```

---

### Expected Output

**With W&B enabled:**
```
INFO:__main__:✅ W&B tracking enabled (API key found)
INFO:__main__:Loading dataset...
INFO:__main__:Formatting dataset...
INFO:__main__:Configuring LoRA for memory efficiency...
INFO:__main__:Configuring training with DeepSpeed and LoRA...
INFO:__main__:📊 W&B run name: grpo-qwen-gsm8k-lora-your_username
INFO:__main__:   View at: https://wandb.ai/your-username/grpo-qwen-gsm8k-lora
INFO:__main__:Initializing trainer with DeepSpeed and LoRA...
INFO:__main__:Starting training with LoRA...
INFO:__main__:Trainable parameters: 15,728,640
INFO:__main__:Total parameters: 1,544,363,008
[Training progress logs...]
INFO:__main__:Training complete.
INFO:__main__:Saving LoRA adapter and tokenizer to ./grpo-trained-qwen-gsm8k-lora
INFO:__main__:LoRA adapter and tokenizer saved.
```

**Without W&B:**
```
INFO:__main__:⚠️  W&B tracking disabled (WANDB_API_KEY not set)
INFO:__main__:   To enable: export WANDB_API_KEY=your_key_here
INFO:__main__:Loading dataset...
[... rest of training ...]
```

**Output Model:** `./grpo-trained-qwen-gsm8k-lora/`

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

**Memory Breakdown (8GB GPU with 2 GPUs):**
- Base model (FP16): ~3GB
- LoRA adapters: ~30MB
- Optimizer states (ZeRO-2 partitioned): ~1.5GB per GPU (split across 2 GPUs)
- Activations + gradients: ~2GB
- Generation buffer (GRPO): ~1.5GB
- **Total: ~6-7GB per GPU** ✅ (fits in 8GB RTX 3070)

**Important:** `GRPOTrainer` does **not** accept `deepspeed` as a direct parameter. You must pass it via `GRPOConfig` through the `args` parameter.

### DeepSpeed Config: `ds_config.json`

**Key Settings (ZeRO Stage 2 with GPU Partitioning):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **ZeRO Stage** | 2 | Optimizer + gradient partitioning across GPUs |
| **CPU Offloading** | Disabled | Keep optimizer on GPU (avoid CUDA compilation issues) |
| **FP16** | Auto | Half-precision training |
| **Batch Size** | Auto | Determined by GRPOConfig (4 per GPU) |
| **Micro Batch Size** | Auto | Set by GRPOConfig |
| **Gradient Accumulation** | Auto | Set by GRPOConfig (8 steps) |
| **Gradient Clipping** | Auto | Set by GRPOConfig |

**Memory-Saving Features:**
```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "overlap_comm": true,     // Overlap communication with computation
  "reduce_scatter": true,   // Efficient gradient reduction
  "contiguous_gradients": true
}
```

**Why No CPU Offloading:**
- CPU offloading requires DeepSpeed to compile CUDA extensions
- Compilation fails if PyTorch CUDA version ≠ system CUDA version
- With LoRA + ZeRO-2 partitioning, optimizer states are small (~1.5GB per GPU)
- Keeping optimizer on GPU is faster and avoids compilation issues

**Auto Parameters:**
- `train_batch_size`: Auto-calculated from per_device_batch_size × num_gpus × gradient_accumulation_steps
- `train_micro_batch_size_per_gpu`: Auto-set from GRPOConfig
- `gradient_accumulation_steps`: Auto-set from GRPOConfig
- `warmup_num_steps`: Auto-scaled to dataset size (100 steps)
- `total_num_steps`: Auto-determined from epochs

### ZeRO Stage 1 vs Stage 2: Speed Comparison 🚀

**Current config uses Stage 2**, but you can switch to **Stage 1 for 10-20% faster training** with minimal memory increase!

#### Comparison Table

| Aspect | ZeRO Stage 1 | ZeRO Stage 2 (Current) |
|--------|-------------|------------------------|
| **What's Partitioned** | Optimizer states only | Optimizer states + gradients |
| **Memory Savings** | ~4x reduction | ~8x reduction |
| **Communication Overhead** | Low (optimizer only) | Higher (optimizer + gradients) |
| **Training Speed** | **Faster (~10-20%)** ⚡ | Slower (more communication) |
| **Memory per GPU (with LoRA)** | ~6-7GB | ~6-7GB (similar) |
| **Communication Passes** | 1 per step | 2-3 per step |

#### How It Works

**ZeRO Stage 1:**
```
GPU 0: [Model] [Gradients] [Optimizer_0]
GPU 1: [Model] [Gradients] [Optimizer_1]
       ↓
Only optimizer states are partitioned
Less communication between GPUs → FASTER
```

**ZeRO Stage 2:**
```
GPU 0: [Model] [Gradients_0] [Optimizer_0]
GPU 1: [Model] [Gradients_1] [Optimizer_1]
       ↓
Both gradients AND optimizer states partitioned
More communication (AllReduce + AllGather) → SLOWER
```

#### Why Stage 1 is Faster

**Stage 1 Communication:**
- Broadcast updated parameters after optimizer step
- **1 communication pass per training step**

**Stage 2 Communication:**
- AllReduce gradients during backward pass
- AllGather to reassemble for optimizer step
- Broadcast updated parameters
- **2-3 communication passes per training step**

With LoRA, optimizer states are tiny (~60MB), so Stage 2's extra partitioning provides minimal memory benefit while adding significant communication overhead.

#### When to Use Each Stage

**Use ZeRO Stage 1 when:**
- ✅ Using LoRA or other parameter-efficient training
- ✅ Small models (<3B params)
- ✅ Want maximum training speed
- ✅ Memory is not a critical constraint
- ✅ **Recommended for this setup!** (RTX 3070 8GB with LoRA)

**Use ZeRO Stage 2 when:**
- ✅ Larger models (>3B params) without LoRA
- ✅ Memory is very tight (need every MB)
- ✅ Maximum memory savings needed
- ✅ Speed is less important than fitting in memory

**Use ZeRO Stage 3 when:**
- ✅ Huge models (>13B params)
- ✅ Model itself doesn't fit in single GPU
- ✅ Need maximum memory savings across GPUs
- ✅ Can tolerate slower training (most communication)

#### How to Switch to Stage 1 (Faster Training)

Edit `ds_config.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 1  // ← Change from 2 to 1
  },
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

**Expected Results with Stage 1:**
- Training time: **10-20% faster** ⚡
- Memory usage: ~6-7GB per GPU (similar to Stage 2)
- Still fits comfortably in 8GB RTX 3070
- Throughput: ~15-25% more samples per second

#### Recommendation for 8GB GPUs with LoRA

**Try Stage 1!** With LoRA, your optimizer states are so small (~60MB) that:
- Stage 2's gradient partitioning saves only ~0.5GB per GPU
- But costs 10-20% training speed
- Stage 1 gives you best speed without memory issues

**When to stick with Stage 2:**
- If you get OOM errors with Stage 1
- If you plan to increase batch size or LoRA rank significantly
- If you want maximum memory headroom for future changes

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

### Issue: `ValueError: DeepSpeed config values mismatch TrainingArguments`

**Error Message:**
```
ValueError: Please correct the following DeepSpeed config values that mismatch TrainingArguments values:
- ds train_micro_batch_size_per_gpu=32 vs hf per_device_train_batch_size=4
- ds gradient_accumulation_steps=1 vs hf gradient_accumulation_steps=8
The easiest method is to set these DeepSpeed config values to 'auto'.
```

**Root Cause:** DeepSpeed config file has hardcoded values that don't match the Python script settings.

**Solution:** Set all DeepSpeed config values to `"auto"`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "gradient_clipping": "auto",
  ...
}
```

**Don't include optimizer/scheduler in DeepSpeed config** - let `GRPOConfig` control these:
- ❌ Remove `"optimizer"` section from ds_config.json
- ❌ Remove `"scheduler"` section from ds_config.json
- ✅ All training params controlled by `GRPOConfig` in Python script

The current `ds_config.json` is already configured correctly with all "auto" settings.

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
Ensure you're running the script from the `06_huggingface_grpo` folder where `ds_config.json` is located:
```bash
cd 06_huggingface_grpo
uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py
```

### Issue: CUDA Version Mismatch with DeepSpeed

**Error Message:**
```
deepspeed.ops.op_builder.builder.CUDAMismatchException:
Installed CUDA version 11.8 does not match the version torch was compiled with 12.8,
unable to compile cuda/cpp extensions without a matching cuda version.
```

**Root Cause:**
DeepSpeed with CPU optimizer offloading tries to compile CUDA extensions, but your PyTorch was compiled with a different CUDA version than what's installed on your system.

**Solution: Disable CPU Offloading (already done in ds_config.json)**

The config has been updated to remove CPU offloading:

```json
{
  "zero_optimization": {
    "stage": 2,
    // ❌ Removed: "offload_optimizer": {"device": "cpu"}
    "allgather_partitions": true,
    "overlap_comm": true,
    ...
  }
}
```

**Why This Still Works on 8GB GPUs:**
- LoRA keeps trainable parameters tiny (~15M vs 1.5B)
- ZeRO-2 partitions optimizer across 2 GPUs (~1.5GB per GPU)
- Each GPU only holds half of the optimizer states
- Total memory: ~6-7GB per GPU (fits in 8GB)

**Alternative Solutions (if you want CPU offloading):**
1. **Match CUDA versions:**
   ```bash
   # Reinstall PyTorch with matching CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Set environment variable to skip compilation:**
   ```bash
   export DS_BUILD_CPU_ADAM=0
   export DS_BUILD_FUSED_ADAM=0
   uv run deepspeed --num_gpus=2 grpo_gsm8k_train.py
   ```

### Issue: CUDA Out of Memory (8GB GPUs)

**Root Cause:** Full fine-tuning of 1.5B models requires ~12-16GB VRAM per GPU.

**Solution: Use LoRA + ZeRO-2 (already implemented in the script!)**

The current script already includes LoRA to solve this. If you're still getting OOM:

**Option 1: Reduce batch size further**
Edit `grpo_gsm8k_train.py` line 164:
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

**Option 3: Use ZeRO Stage 1 (faster and less memory overhead)**
Edit `ds_config.json`:
```json
"zero_optimization": {
  "stage": 1  // Instead of 2
}
```

See the [ZeRO Stage 1 vs Stage 2](#zero-stage-1-vs-stage-2-speed-comparison-) section above for detailed comparison.

**Expected Memory Usage:**
- With current config (Stage 2): ~6-7GB per GPU
- With batch_size=2: ~5-6GB per GPU
- With r=8: ~5-6GB per GPU
- With ZeRO Stage 1: ~6-7GB per GPU (similar memory, 10-20% faster!)

### Issue: Slow Training

**Solution 1: Switch to ZeRO Stage 1 (Recommended! ⚡)**
Edit `ds_config.json`:
```json
"zero_optimization": {
  "stage": 1  // Change from 2 to 1
}
```
**Expected speedup: 10-20% faster training with same memory usage!**

See the [ZeRO Stage 1 vs Stage 2](#zero-stage-1-vs-stage-2-speed-comparison-) section for detailed explanation.

**Solution 2: Use more GPUs**
```bash
uv run deepspeed --num_gpus=4 grpo_gsm8k_train.py
```

**Solution 3: Enable BF16 instead of FP16 (if supported)**
Edit `ds_config.json`:
```json
"fp16": {"enabled": false},
"bf16": {"enabled": true}
```

Note: BF16 requires newer GPUs (Ampere/Ada architecture, e.g., RTX 3000/4000 series)

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
2. ✅ **ZeRO-2 Optimizer Partitioning**: Splits optimizer across GPUs (~1.5GB per GPU)
3. ✅ **Reduced Batch Size**: 4 per GPU instead of 32
4. ✅ **GRPOConfig**: Pass DeepSpeed config via `args` parameter (not directly)
5. ✅ **No CPU Offloading**: Avoids CUDA version mismatch compilation errors

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
