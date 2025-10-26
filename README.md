# DeepSpeed Course 🚀

**Author:** Yiqiao Yin
[LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)

---

## Table of Contents

- [Overview](#overview)
  - [Situation Today](#situation-today-)
  - [Problem Statement](#problem-statement-)
  - [Solution](#solution-)
- [Folder Structure](#folder-structure-)
- [Getting Started](#getting-started)
  - [SLURM Batch Jobs (CoreWeave)](#slurm-batch-jobs-coreweave-)
    - [Quick Start Guide](#quick-start-guide)
    - [SLURM Commands Reference](#slurm-commands-reference)
    - [GPU Monitoring](#gpu-monitoring)
    - [Beginner Tutorial](#beginner-tutorial-hello-world)
    - [Virtual Environment Setup](#virtual-environment-setup-with-uv)
  - [Runpod](#runpod-)

---

## Overview

### Situation Today 🐢

Training and inference for deep learning models are often slow and resource-intensive, especially as model sizes and dataset complexity grow. This bottleneck impacts productivity and limits experimentation, making it difficult to iterate quickly or deploy models efficiently.

### Problem Statement 🤔

To overcome these challenges, it's essential to leverage multiple GPUs and distributed training. DeepSpeed is a deep learning optimization library that enables faster training, efficient memory usage, and scalable distributed training across multiple GPUs. Using DeepSpeed can significantly reduce training time and improve inference speed, making it possible to work with larger models and datasets.

### Solution 💡

This repository provides a collection of basic frameworks and examples demonstrating how to use DeepSpeed for distributed training and inference. Each folder contains a different neural network architecture or use case, showing how DeepSpeed can be integrated to accelerate workflows.

---

## Folder Structure 📁

```
deepspeed-course/
├── 01_basic_neuralnet/
│   ├── train_ds.py                    # Basic neural network training
│   ├── train_ds_enhanced.py           # Enhanced with W&B tracking
│   ├── ds_config.json                 # DeepSpeed configuration
│   ├── run_deepspeed.sh              # SLURM batch script
│   └── README.md                      # Documentation
│
├── 02_basic_convnet/
│   ├── train_ds.py                    # CNN training on synthetic MNIST
│   ├── ds_config.json                 # DeepSpeed configuration
│   ├── run_deepspeed.sh              # SLURM batch script
│   └── README.md                      # Documentation
│
├── 02_basic_convnet_cifar10_examples/
│   ├── cifar10_deepspeed.py          # CIFAR-10 CNN (81% accuracy!)
│   ├── ds_config.json                 # DeepSpeed config (SGD + BatchNorm)
│   ├── run_deepspeed.sh              # SLURM batch script (2 GPUs)
│   ├── MODEL_IMPROVEMENT_STRATEGY.md  # Technical deep dive
│   └── README.md                      # Comprehensive guide
│
├── 03_basic_rnn/
│   ├── train_rnn_deepspeed.py        # LSTM time series prediction
│   ├── ds_config_rnn.json            # DeepSpeed config (ZeRO-2 + FP16)
│   ├── run_deepspeed.sh              # SLURM batch script
│   └── README.md                      # Documentation with W&B guide
│
├── 04_intermediate_rnn_stock_data/
│   ├── train_rnn_stock_data.py       # Single-machine stock RNN training
│   ├── train_rnn_stock_data_ds.py    # DeepSpeed stock RNN with W&B
│   ├── train_rnn_stock_data_config.json # DeepSpeed config (ZeRO-2 + FP16)
│   ├── run_deepspeed.sh              # SLURM batch script (2 GPUs)
│   └── README.md                      # Stock prediction guide with uv setup
│
├── 04_transferlearning/               # (TBD)
├── 05_huggingface/                    # HuggingFace examples
├── 05_huggingface_trl/                # TRL (Transformer Reinforcement Learning)
├── 06_huggingface_grpo/               # GRPO (Group Relative Policy Optimization)
├── 07_huggingface_openai_gpt_oss_finetune_sft/  # SFT examples
├── 07_huggingface_trl_multi_agency/   # Multi-agent systems
├── 08_vtt/                            # Vision-Text-Text models
└── README.md                          # This file
```

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
cd 04_intermediate_rnn_stock_data

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
cd 04_intermediate_rnn_stock_data

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
cd 01_basic_neuralnet
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=1 train_ds_enhanced.py  # Direct
```

**CIFAR-10 CNN (Multi-GPU):**
```bash
cd 02_basic_convnet_cifar10_examples
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 cifar10_deepspeed.py  # Direct
```

**LSTM Time Series (Multi-GPU + ZeRO-2):**
```bash
cd 03_basic_rnn
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 train_rnn_deepspeed.py  # Direct
```

**Stock Price RNN (Multi-GPU + Real Data):**
```bash
cd 04_intermediate_rnn_stock_data
sbatch run_deepspeed.sh  # SLURM
# Or: deepspeed --num_gpus=2 train_rnn_stock_data_ds.py  # Direct
```

---

## Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is released under the MIT License.

---

**Happy Training with DeepSpeed!** 🚀
