# DeepSpeed Course 🚀

**Author:** Yiqiao Yin
[LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)

### 📖 **[Read the full course → yiqiao-yin.github.io/deepspeed-course](https://yiqiao-yin.github.io/deepspeed-course/)**

The documentation site is the primary way to read this material: 39 pages with
the memory and communication arithmetic derived in full, ~200 cited papers, and
diagrams. This README covers setup and cluster operations.

| Start here | |
|---|---|
| [DeepSpeed ZeRO Stages](https://yiqiao-yin.github.io/deepspeed-course/docs/getting-started/deepspeed-zero-stages) | Why partitioning works and what each stage costs — everything else references this |
| [Basic Neural Network](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/basic/neural-network) | The training loop, losses as likelihoods, CUDA OOM as memory accounting |
| [CIFAR-10](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/basic/cifar10) | A real debugging case study: NaN at 10% accuracy, repaired to 81% |
| [GRPO Training](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/huggingface/grpo-training) | RL with verifiable rewards |
| [Troubleshooting](https://yiqiao-yin.github.io/deepspeed-course/docs/reference/troubleshooting) | Symptom-first diagnosis |

---

> **Folders were reorganised.** Every top-level number now appears once,
> and examples live at `NN_section/NN_topic`. If a link into this repository
> stopped working, [MOVED.md](MOVED.md) maps every old path to its new one.

## Table of Contents

- [Overview](#overview)
  - [Situation Today](#situation-today-)
  - [Problem Statement](#problem-statement-)
  - [Solution](#solution-)
- [Environment & Testing](#environment--testing-)
  - [Package management: uv](#package-management-uv)
  - [What runs locally, and what does not](#what-runs-locally-and-what-does-not)
- [Folder Structure](#folder-structure-)
- [CoreWeave vs RunPod: Understanding the Architectures](#coreweave-vs-runpod-understanding-the-architectures)
  - [CoreWeave: Shared Multi-User HPC Cluster](#coreweave-shared-multi-user-hpc-cluster)
  - [RunPod: Single-User Pod Model](#runpod-single-user-pod-model)
  - [Key Differences](#key-differences-table)
  - [Why CoreWeave Uses SLURM](#why-coreweave-uses-this-model)
  - [Interactive Access on CoreWeave](#can-you-get-interactive-access-on-coreweave)
  - [When to Use Each](#when-to-use-each)
- [Getting Started](#getting-started)
  - [SLURM Batch Jobs (CoreWeave)](#slurm-batch-jobs-coreweave-)
    - [Quick Start Guide](#quick-start-guide)
    - [SLURM Commands Reference](#slurm-commands-reference)
    - [GPU Monitoring](#gpu-monitoring)
    - [Beginner Tutorial](#beginner-tutorial-hello-world)
    - [Virtual Environment Setup](#virtual-environment-setup-with-uv)
  - [Runpod](#runpod-)
- [Example Training Commands](#example-training-commands)
- [Contributing](#contributing-)
- [Resources](#resources)

---

## Overview

### Situation Today 🐢

Training and inference for deep learning models are often slow and resource-intensive, especially as model sizes and dataset complexity grow. This bottleneck impacts productivity and limits experimentation, making it difficult to iterate quickly or deploy models efficiently.

### Problem Statement 🤔

To overcome these challenges, it's essential to leverage multiple GPUs and distributed training. DeepSpeed is a deep learning optimization library that enables faster training, efficient memory usage, and scalable distributed training across multiple GPUs. Using DeepSpeed can significantly reduce training time and improve inference speed, making it possible to work with larger models and datasets.

### Solution 💡

This repository provides a collection of basic frameworks and examples demonstrating how to use DeepSpeed for distributed training and inference. Each folder contains a different neural network architecture or use case, showing how DeepSpeed can be integrated to accelerate workflows.

---

## Security 🔐

This repository is public; your credentials are not. Every key
(`RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`) is read from the environment and
never committed — a test scans every file on every push. The RunPod tooling
never places your API key on rented hardware.

See **[SECURITY.md](SECURITY.md)**.

---

## Environment & Testing 🧪

### Package management: `uv`

Every example uses [`uv`](https://docs.astral.sh/uv/) — not bare `pip` or conda —
and **every example folder is a `uv` project with a committed `uv.lock`**. So
after cloning, one command sets a folder up:

```bash
cd 01_basics/01_neuralnet
uv sync                                   # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=1 train_ds_enhanced.py
```

`uv run` uses the project environment directly, so there is no `activate` step.
Add `uv sync --extra tracking` if you want Weights & Biases; it stays optional.

The lock is the point: everyone who clones resolves to identical versions,
instead of whatever `uv pip install` finds that day. Regenerate deliberately
with `uv lock --upgrade`.

<details>
<summary>Manual route, without the project</summary>

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed
```

The `--index-url` is required: PyPI's default `torch` is a CUDA 13 wheel, and
on a driver older than CUDA 13 it installs cleanly and then reports
`cuda.is_available() == False`. Verified on a driver 550.127 box.
</details>

Each example folder's README has an **Environment & Local Testing** section with
its exact dependencies, GPU requirement, and download size.

**Five examples deliberately skip the `deepspeed` launcher**, each for a stated
reason — using a distributed launcher where there is nothing to distribute is
cargo cult:

| Example | Why |
|---|---|
| `03_huggingface/09_multi_agency` | drives TRL's `GRPOTrainer` directly |
| `04_video_text/04_streaming_memory` | streaming *inference* — sequential, no optimizer |
| `04_video_text/05_video_eval` | evaluation — short `generate()` calls |
| `05_video_speech/03_duplex_streaming` | duplex inference — slices arrive in order |
| `05_video_speech/04_omni_eval` | evaluation — modality-ablation `generate()` calls |

### What runs locally, and what does not

| Examples | Scale | Can you run it on one machine? |
|---|---|---|
| `01_basics`, `02_intermediate` | Synthetic or small data, ≤1M parameters | **Yes** — end to end, in seconds to minutes |
| `03_huggingface`, `04_video_text`, `05_video_speech` | Real models, GBs to 1.1 TB of weights, 2–8 GPUs | **No** — needs real GPU capacity |

For the second group a full run is not a practical way to check a change. The
repository therefore ships **logic tests** that exercise the code paths without a
GPU or a model download:

```bash
./tests/run_all.sh                  # 18 suites, no GPU and no downloads
uv run tests/test_ds_configs.py     # a single suite
```

| Suite | Guards against |
|---|---|
| `test_ds_configs.py` | Config errors across **all 14** `ds_config.json` files — batch-invariant mismatches, fp16/bf16 conflicts, stage-3 checkpoints that cannot reload |
| `test_stock_leakage.py` | Look-ahead bias from fitting a scaler before the train/test split |
| `test_grpo_rewards.py` | A PPO value head under GRPO; surface-form and misaligned rewards |
| `test_video_frames.py` | Frame "extraction" that returns one image repeated |

See [`tests/README.md`](tests/README.md).

## Folder Structure 📁

Five sections, each number used exactly once. Every example lives at
`NN_section/NN_topic` and is self-contained — open one folder and run it
without touching the rest.

```
deepspeed-course/
│
├── 01_basics/             # Runs end to end on one machine, in seconds to minutes
│   ├── 01_neuralnet/            # Fitting y = 2x + 1. Two parameters — the smallest real DeepSpeed run
│   ├── 02_convnet/              # MNIST CNN — the first example with a real dataset
│   ├── 03_convnet_cifar10/      # CIFAR-10: a documented failure-and-recovery, plus 3 modern nets at ~93%
│   └── 04_rnn/                  # LSTM on sequence data
│
├── 02_intermediate/       # Still small, but the modelling questions get harder
│   ├── 01_bayesian_neuralnet/   # Parallel-tempering MCMC — uncertainty, not point estimates
│   └── 02_rnn_stock_data/       # Time-series forecasting, and why most models lose to persistence
│
├── 03_huggingface/        # Real models and real downloads. 04-07 are one argument about what you can delete from RLHF
│   ├── 01_llm_finetuning/       # LLM fine-tuning with ZeRO — the starting point
│   ├── 02_trl_sft/              # TRL supervised fine-tuning for function calling
│   ├── 03_ocr/                  # Vision-language OCR + a measured comparison of 5 modern OCR models
│   ├── 04_reward_model/         # Bradley-Terry reward modelling. This IS the RLHF pipeline
│   ├── 05_dpo/                  # DPO and 5 descendants — deletes the REWARD MODEL
│   ├── 06_grpo/                 # GRPO on GSM8K — deletes the CRITIC
│   ├── 07_online_dpo/           # Online DPO, Nash-MD, XPO — re-adds sampling, needs a judge
│   ├── 08_gpt_oss_lora/         # LoRA SFT of a 20B model
│   └── 09_multi_agency/         # Multi-agent GRPO (drives TRL directly, no DeepSpeed launcher)
│
├── 04_video_text/         # Video in, text out
│   ├── 01_hf_baseline/          # Foundational LLaVA / seq2seq video trainers
│   ├── 02_qwen25vl/             # Qwen2.5-VL — a model that can represent TIME
│   ├── 03_token_compression/    # ToMe, FastV, DyCoke — 'ZeRO for activations'
│   ├── 04_streaming_memory/     # STAR: unbounded video in O(1) memory
│   └── 05_video_eval/           # Did compression break understanding? Reports the TEMPORAL GAP
│
└── 05_video_speech/       # Video AND audio in, speech out
    ├── 01_longcat_omni/         # The frontier: 560B, ~3 TB host RAM
    ├── 02_thinker_talker/       # Two streams onto ONE 40 ms clock, then speech out
    ├── 03_duplex_streaming/     # Listening and watching WHILE speaking
    ├── 04_omni_eval/            # Does it actually use both streams? Reports the FUSION GAIN
    └── data/                    # Shared corpus (44 MB), not duplicated per subtopic
```

**Every example folder has the same six files** (the contract in
[CONTRIBUTING.md](CONTRIBUTING.md)), so this tree does not list them per folder:

| File | Role |
|---|---|
| `train_*.py` | Entry point. Calls `deepspeed.initialize(...)`, starts with `require_gpu()` |
| `ds_config*.json` | DeepSpeed config — ZeRO stage, precision, optimizer, batch sizes |
| `run_deepspeed.sh` | SLURM batch script (`submit_job.sh` / `run_training.sh` in older folders) |
| `README.md` | Standalone walkthrough: hardware, setup, run command, expected output |
| `pyproject.toml` | Makes the folder a `uv` project |
| `uv.lock` | **Committed**, so `cd <folder> && uv sync` works from a fresh clone |

Some folders add `HARDWARE_REQUIREMENTS.md`, `MODEL_IMPROVEMENT_STRATEGY.md`, or
extra scripts; those are documented in their own README.

> Folder paths changed when the sections were introduced. [MOVED.md](MOVED.md)
> maps every old path to its new one.

## CoreWeave vs RunPod: Understanding the Architectures

Before diving into the workflows, it's essential to understand the fundamental differences between these two platforms. This will help you choose the right environment for your needs.

### CoreWeave: Shared Multi-User HPC Cluster

**Architecture:**
```
Login Nodes (where you SSH)
    ↓
SLURM Scheduler (resource manager)
    ↓
Compute Nodes (where jobs run)
    ↓
Your GPU workload
```

**Why you need SLURM:**
1. **Shared Resources**: Hundreds of users competing for GPUs
2. **Fair Scheduling**: SLURM ensures fair allocation based on priority/quota
3. **Resource Isolation**: Prevents users from hogging all GPUs
4. **Queue System**: Your job waits if resources aren't available
5. **Accounting**: Tracks who uses what (billing, quotas)

**What happens when you SSH:**
- You land on a **login node** (no GPUs attached)
- Login nodes are for: submitting jobs, editing files, light tasks
- **Cannot run GPU code directly** - no GPUs available on login nodes
- Must use `sbatch` to request GPU time on compute nodes

---

### RunPod: Single-User Pod Model

**Architecture:**
```
You SSH directly into YOUR pod
    ↓
Pod has dedicated GPU(s)
    ↓
Run code immediately
```

**Why no SLURM needed:**
1. **Pre-allocated**: You rent the entire pod upfront
2. **Dedicated Resources**: Those GPUs are YOURS for the rental period
3. **Single User**: No competition - it's like renting a whole server
4. **Pay-per-use**: You're billed for the entire time pod is running
5. **No scheduling**: Run whatever, whenever - you own the resources

---

### Key Differences Table

| Aspect | CoreWeave (SLURM) | RunPod |
|--------|-------------------|--------|
| **Access Model** | Shared cluster | Dedicated pod |
| **GPU Access** | Request via scheduler | Always available |
| **When you pay** | Only when job runs | Entire pod lifetime |
| **Multi-user** | Yes, hundreds | No, just you |
| **Resource competition** | Yes, queue if busy | No, yours alone |
| **Can run commands directly** | ❌ No (login node only) | ✅ Yes |
| **Best for** | Batch jobs, research clusters | Interactive work, development |

---

### Why CoreWeave Uses This Model

#### **Efficiency Example:**

```bash
# ❌ Bad: Everyone gets dedicated GPUs (RunPod style)
User A: GPU idle 80% of time (editing code)
User B: GPU idle 90% of time (debugging)
User C: GPU idle 70% of time (reading papers)
Total: 3 GPUs, mostly wasted

# ✅ Good: Shared cluster with scheduler (CoreWeave style)
User A: Submit job when ready → GPU used 100%
User B: Submit job when ready → GPU used 100%
User C: Submit job when ready → GPU used 100%
Total: 1 GPU, fully utilized, serves 3 users
```

#### **Cost Model:**
- **CoreWeave**: Pay only for GPU hours used (like AWS Lambda)
- **RunPod**: Pay for entire rental period (like renting a car)

#### **Scale:**
- **SLURM** can manage 10,000+ GPUs across 1000+ nodes
- **RunPod** model would require 1000+ separate pods

---

### Can You Get Interactive Access on CoreWeave?

**Yes!** Use `srun` for interactive sessions:

```bash
# Request interactive shell with 1 GPU for 2 hours
srun --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash

# Now you're on a compute node with GPU access!
nvidia-smi
python

# Run code interactively
python train.py
```

This gives you RunPod-like experience, but:
- ⏱️ You wait in queue if GPUs busy
- ⏰ Session ends after time limit
- 💰 You're charged for the entire interactive session

---

### Analogies

#### **CoreWeave = Airport 🛫**
- You can't just walk onto any plane (login node)
- Need a ticket and boarding pass (SLURM job)
- Wait in line if flights full (queue)
- Efficient: planes stay full

#### **RunPod = Private Jet ✈️**
- You own/rent the jet for the day
- Board anytime, no waiting
- More expensive per person
- Jet might sit idle while you're at lunch

---

### When to Use Each

**Use SLURM/CoreWeave when:**
- ✅ Running batch training jobs (submit and forget)
- ✅ Need massive scale (100+ GPUs)
- ✅ Want cost efficiency (only pay for actual compute)
- ✅ Research/academic environment
- ✅ Jobs can wait in queue
- ✅ Training runs for hours/days

**Use RunPod/Direct Access when:**
- ✅ Need interactive development
- ✅ Debugging code frequently
- ✅ Prototyping/experimenting
- ✅ Can't wait in queue
- ✅ Want simplicity (no SLURM learning curve)
- ✅ Jupyter notebooks for exploration

---

### Bottom Line

You **can** SSH into CoreWeave, but you're on a **login node without GPUs**. To use GPUs, you must:
1. **Batch jobs**: `sbatch script.sh` (submit and check later)
2. **Interactive**: `srun --gres=gpu:1 --pty bash` (wait for GPU, then interactive)

RunPod gives you the GPU immediately because you're **renting the entire pod**. CoreWeave makes you **request GPU time** because it's a **shared cluster**.

**Think of it like:**
- **RunPod** = Renting a whole Airbnb 🏠 (dedicated, always available)
- **CoreWeave** = Using a hotel room 🏨 (book when you need it, efficient)

---

## Getting Started

This repository supports two main deployment environments:
1. **SLURM Batch Jobs (CoreWeave)** - For HPC cluster environments with job scheduling
2. **Runpod** - For interactive development with Jupyter Lab or terminal

---

## SLURM Batch Jobs (CoreWeave) 🚀

**Perfect for:** HPC cluster users with CoreWeave or similar SLURM-based infrastructure. This workflow allows you to submit batch jobs that run in the background, efficiently managing GPU resources across multiple nodes.

### Quick Start Guide

Each training folder (01-04) includes a `run_deepspeed.sh` SLURM batch script optimized for HPC clusters like CoreWeave.

**Basic Workflow:**

```bash
# 1. Navigate to your desired training folder
cd 02_intermediate/02_rnn_stock_data

# 2. Edit the SLURM script to configure your environment
nano run_deepspeed.sh
# Configure:
#   - WANDB_API_KEY (get from https://wandb.ai/authorize)
#   - Virtual environment path (or use uv - see below)

# 3. Submit your job to the SLURM queue
sbatch run_deepspeed.sh

# 4. Check job status
squeue -u $USER

# 5. View output in real-time
tail -f logs/stock_rnn_<job_id>.out
```

**Script Features:**
- ✅ Pre-configured GPU/CPU/memory resources per workload
- ✅ Automatic log directory creation
- ✅ Job information printing (ID, node, GPUs, timestamps)
- ✅ W&B API key integration with placeholder
- ✅ Optimized for CoreWeave/SLURM clusters
- ✅ Multi-GPU support with DeepSpeed

---

### SLURM Commands Reference

**Job Submission & Monitoring**

```bash
# See your jobs in the queue
squeue -u $USER

# More detailed view with job state
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# See all jobs in the queue (entire cluster)
squeue

# Watch the queue in real-time (refreshes every 1 second)
watch -n 1 squeue -u $USER

# Check specific job status (e.g., job ID 34)
squeue -j 34

# Get detailed job information
scontrol show job 34

# See why job is pending (shows estimated start time)
squeue -j 34 --start

# View your job history
sacct -u $USER

# Cancel a job if needed
scancel 34

# Cancel all your jobs
scancel -u $USER
```

**Output File Management**

```bash
# List output files (sorted by modification time)
ls -lt slurm-*.out

# Check most recent output file
ls -lt slurm-*.out | head -1

# View the complete output
cat slurm-34.out

# Tail the output (last 10 lines)
tail slurm-34.out

# Follow output in real-time (useful for monitoring training)
tail -f slurm-34.out

# Search for errors in output
grep -i error slurm-34.out

# Search for specific metrics (e.g., loss)
grep "Loss:" slurm-34.out
```

---

### GPU Monitoring

Monitor GPU utilization in real-time during training to ensure your resources are being used efficiently.

**Create GPU Monitor Script:**

```bash
cat > gpu_monitor.sh << 'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=h200-low
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu_monitor

# Log nvidia-smi output every 0.1 seconds
while true; do
    nvidia-smi
    echo "---"
    sleep 0.1
done
EOF
```

**Submit Monitor Job:**

```bash
sbatch gpu_monitor.sh

# Get the job ID from output, then monitor
squeue -u $USER  # Find job ID (e.g., 34)
tail -f slurm-34.out
```

**Example Output:**

```
Tue Oct 14 17:20:56 2025
+-------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08     Driver Version: 570.172.08   CUDA Version: 12.9 |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf          Pwr  | Memory-Usage         | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H200              | 00000000:19:00.0 Off |                    0 |
| N/A   32C    P0             78W / 700W |      4MiB / 143771MiB |      0%   Default |
+-------------------------------+----------------------+----------------------+
```

**Key Metrics to Watch:**
- **GPU**: Model name (e.g., NVIDIA H200 with 141GB HBM3e)
- **Driver**: Version 570.172.08
- **CUDA**: Toolkit version 12.9
- **Memory**: 4MiB / 143771MiB (usage / total)
- **GPU Util**: 0% means idle, 90-100% means fully utilized
- **Power**: 78W / 700W (current / max TDP)
- **Temp**: 32°C (should stay under 85°C under load)

---

### Beginner Tutorial: Hello World

Test your SLURM setup with a simple PyTorch job that verifies GPU access.

**Step 1: Create Python Script**

```bash
cat > hello.py << 'EOF'
import torch

print("Hello from Python!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current GPU: {torch.cuda.current_device()}")

    # Simple GPU computation test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiplication test passed!")
    print(f"Result shape: {z.shape}")
else:
    print("No GPU detected")
EOF
```

**Step 2: Create SLURM Batch Script**

```bash
cat > run_hello.sh << 'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=h200-low
#SBATCH --time=00:10:00
#SBATCH --job-name=hello_world

# Load modules if needed (uncomment if your cluster uses environment modules)
# module load python/3.10
# module load cuda/12.9

# Activate your virtual environment
source ~/myenv/bin/activate

# Run the Python script
python3 hello.py
EOF
```

**Step 3: Submit the Job**

```bash
sbatch run_hello.sh
```

**Step 4: Check Status and Output**

```bash
# Check job status
squeue -u $USER

# Wait for job to complete, then view output
ls -lt slurm-*.out

# View the results
cat slurm-*.out  # Replace with actual job ID
```

**Expected Output:**

```
Hello from Python!
PyTorch version: 2.0.1
CUDA available: True
GPU count: 1
GPU name: NVIDIA H200
Current GPU: 0
Matrix multiplication test passed!
Result shape: torch.Size([1000, 1000])
```

---

### Virtual Environment Setup with `uv`

Use the modern `uv` package manager for fast dependency installation on SLURM compute nodes.

**Create Virtual Environment:**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# Create a new virtual environment with uv
uv venv myenv

# Activate the environment
source myenv/bin/activate
```

**Install Dependencies:**

```bash
# Install PyTorch with uv (much faster than pip!)
uv pip install torch

# Install DeepSpeed
uv pip install deepspeed

# Install additional dependencies
uv pip install numpy pandas matplotlib wandb yfinance scikit-learn

# Or install from requirements.txt
uv pip install -r requirements.txt
```

**Using in SLURM Scripts:**

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=h200-low
#SBATCH --time=02:00:00
#SBATCH --job-name=my_training

# Activate uv-created virtual environment
source ~/myenv/bin/activate

# Run training with DeepSpeed
deepspeed --num_gpus=2 train_script.py
```

**Deactivate Environment:**

```bash
deactivate  # When you're done
```

**Why use `uv`?**
- ⚡ **10-100x faster** than pip for dependency resolution
- 🔒 **Reproducible** environments with lock files
- 📦 **Unified** tool for venv creation and package management
- 💾 **Smaller** cache and faster installs on shared filesystems

---

## Runpod 🖥️

**Perfect for:** Interactive development, Jupyter notebooks, real-time experimentation, and debugging.

For language models or vision-language models, it is recommended to use the **Runpod PyTorch 2.8.0** image:

`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

### Recommended Configurations

**High-Performance Multi-GPU Setup**

Best for: Large-scale distributed training, multi-node experiments

**Pricing Summary:**
- GPU Cost: $30.32 / hr
- Running Pod Disk Cost: $0.011 / hr
- Stopped Pod Disk Cost: $0.014 / hr

**Pod Summary:**
- 8x H200 SXM (1128 GB VRAM)
- 2008 GB RAM • 224 vCPU
- Total Disk: 80 GB

---

**Cost-Effective Single-GPU Setup**

Best for: Long training runs, single-GPU experiments, development

**Pricing Summary:**
- GPU Cost: $4 / hr
- Running Pod Disk Cost: $0.011 / hr
- Stopped Pod Disk Cost: $0.014 / hr

**Pod Summary:**
- 10x A40 (480 GB VRAM)
- 500 GB RAM • 90 vCPU
- Total Disk: 80 GB

### Running on Runpod

**Terminal-based Training:**

```bash
# Clone the repository
git clone https://github.com/your-repo/deepspeed-course.git
cd deepspeed-course

# Install dependencies with uv (recommended)
uv venv myenv
source myenv/bin/activate
uv pip install torch deepspeed wandb

# Navigate to a training folder
cd 02_intermediate/02_rnn_stock_data

# Run training directly
uv run deepspeed --num_gpus=1 train_rnn_stock_data_ds.py

# Or run with multiple GPUs
uv run deepspeed --num_gpus=2 train_rnn_stock_data_ds.py
```

**Jupyter Lab:**

```bash
# Start Jupyter Lab (usually pre-installed on Runpod)
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# Navigate to the exposed URL and open notebooks
# Follow along with training examples interactively
```

---

## Example Training Commands

**Basic Neural Network (Single GPU):**
```bash
cd 01_basics/01_neuralnet
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=1 train_ds_enhanced.py  # Direct
```

**CIFAR-10 CNN (Multi-GPU):**
```bash
cd 01_basics/03_convnet_cifar10
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 cifar10_deepspeed.py  # Direct
```

**LSTM Time Series (Multi-GPU + ZeRO-2):**
```bash
cd 01_basics/04_rnn
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 train_rnn_deepspeed.py  # Direct
```

**Stock Price RNN (Multi-GPU + Real Data):**
```bash
cd 02_intermediate/02_rnn_stock_data
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 train_rnn_stock_data_ds.py  # Direct
```

---

## Contributing 🤝

**Contributions from anyone are welcome** — you do not need to know the
maintainer or ask permission first. Fork, add your example, open a PR.

New models and training scripts are the most valuable contribution. The one
requirement worth knowing before you start is the **three-platform contract**:
your example must work sensibly for a reader with **no GPU** (fail gracefully
via `require_gpu()`), a reader on **CoreWeave** (`sbatch` + a cheap dry run),
and a reader with a **RunPod API key** (registered in `runpod_ctl.py`, with the
auto-shutdown flags documented).

Scaffold a new example that already satisfies the contract:

```bash
uv run scripts/new_example.py 10_my_topic --title "My Topic" --vram 24
./tests/run_all.sh
```

`uv` for packages, `deepspeed` for training — this is a DeepSpeed course.

**Using Claude Code to contribute is encouraged.** `CONTRIBUTING.md` is written
to double as a spec an agent can follow, and the repo ships a `CLAUDE.md` that
Claude Code loads automatically.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide, or the
[Contributing page](https://yiqiao-yin.github.io/deepspeed-course/docs/contributing)
on the course site.

---

## Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## License

Released under the [MIT License](LICENSE). By contributing you agree your
contribution is licensed under the same terms.

---

**Happy Training with DeepSpeed!** 🚀
