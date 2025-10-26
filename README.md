# DeepSpeed Course 🚀

**Author:** Yiqiao Yin  
[LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)

## Situation Today 🐢

Training and inference for deep learning models are often slow and resource-intensive, especially as model sizes and dataset complexity grow. This bottleneck impacts productivity and limits experimentation, making it difficult to iterate quickly or deploy models efficiently.

## Problem Statement 🤔

To overcome these challenges, it's essential to leverage multiple GPUs and distributed training. DeepSpeed is a deep learning optimization library that enables faster training, efficient memory usage, and scalable distributed training across multiple GPUs. Using DeepSpeed can significantly reduce training time and improve inference speed, making it possible to work with larger models and datasets.

## Solution 💡

This repository provides a collection of basic frameworks and examples demonstrating how to use DeepSpeed for distributed training and inference. Each folder contains a different neural network architecture or use case, showing how DeepSpeed can be integrated to accelerate workflows.

### Folder Structure 📁

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
├── 04_transferlearning/               # (TBD)
├── 05_huggingface/                    # HuggingFace examples
├── 05_huggingface_trl/                # TRL (Transformer Reinforcement Learning)
├── 06_huggingface_grpo/               # GRPO (Group Relative Policy Optimization)
├── 07_huggingface_openai_gpt_oss_finetune_sft/  # SFT examples
├── 07_huggingface_trl_multi_agency/   # Multi-agent systems
├── 08_vtt/                            # Vision-Text-Text models
└── README.md                          # This file
```

### Quick Start with SLURM Batch Jobs 🚀

Each training folder (01-03) includes a `run_deepspeed.sh` SLURM batch script for running on HPC clusters like CoreWeave. To use:

```bash
# 1. Navigate to desired folder
cd 02_basic_convnet_cifar10_examples

# 2. Edit the SLURM script to configure:
#    - WANDB_API_KEY (get from https://wandb.ai/authorize)
#    - Virtual environment path
nano run_deepspeed.sh

# 3. Submit to SLURM queue
sbatch run_deepspeed.sh

# 4. Monitor job status
squeue -u $USER

# 5. Check logs
tail -f logs/cifar10_cnn_<job_id>.out
```

**Script Features:**
- ✅ Pre-configured GPU/CPU/memory resources per workload
- ✅ Automatic log directory creation
- ✅ Job information printing (ID, node, GPUs, timestamps)
- ✅ W&B API key integration with placeholder
- ✅ Optimized for CoreWeave/SLURM clusters

Explore each folder for step-by-step guides and code samples to accelerate your deep learning projects with DeepSpeed!

### On Runpod

For language models or vision-language models, it is recommended to use the Runpod image: **Runpod Pytorch 2.8.0**

`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

**Pricing Summary**
- GPU Cost: $30.32 / hr
- Running Pod Disk Cost: $0.011 / hr
- Stopped Pod Disk Cost: $0.014 / hr

**Pod Summary**
- 8x H200 SXM (1128 GB VRAM)
- 2008 GB RAM • 224 vCPU
- Total Disk: 80 GB

For single GPU usage with long training times, it is recommended to use the following Runpod configuration:

**Pricing Summary**
- GPU Cost: $4 / hr
- Running Pod Disk Cost: $0.011 / hr
- Stopped Pod Disk Cost: $0.014 / hr

**Pod Summary**
- 10x A40 (480 GB VRAM)
- 500 GB RAM • 90 vCPU
- Total Disk: 80 GB