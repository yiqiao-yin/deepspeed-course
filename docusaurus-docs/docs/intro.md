---
sidebar_position: 1
slug: /intro
---

# DeepSpeed Course

**Author:** Yiqiao Yin | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)

Welcome to the DeepSpeed Course - a comprehensive guide to distributed deep learning with DeepSpeed.

## Overview

### The Challenge

Training and inference for deep learning models are often slow and resource-intensive, especially as model sizes and dataset complexity grow. This bottleneck impacts productivity and limits experimentation, making it difficult to iterate quickly or deploy models efficiently.

### The Solution

DeepSpeed is a deep learning optimization library that enables faster training, efficient memory usage, and scalable distributed training across multiple GPUs. This repository provides a collection of frameworks and examples demonstrating how to use DeepSpeed for distributed training and inference.

## What You'll Learn

This course takes you from basic neural networks to cutting-edge multimodal AI:

| Level | Topics |
|-------|--------|
| **Basic** | Neural networks, CNNs, RNNs with DeepSpeed |
| **Intermediate** | Bayesian inference, real-world stock prediction |
| **Advanced** | HuggingFace integration, TRL fine-tuning, GRPO |
| **Expert** | Video-language models, video-speech-to-speech |

## Quick Start

Choose your deployment environment:

### Option 1: SLURM Clusters (CoreWeave)

For HPC cluster users with job scheduling:

```bash
# Navigate to any training folder
cd 01_basic_neuralnet

# Submit your job
sbatch run_deepspeed.sh

# Monitor progress
squeue -u $USER
tail -f logs/training_*.out
```

### Option 2: Interactive Development (RunPod)

For direct GPU access and Jupyter notebooks:

```bash
# Clone and setup
git clone https://github.com/yiqiao-yin/deepspeed-course.git
cd deepspeed-course

# Install dependencies
uv venv myenv && source myenv/bin/activate
uv pip install torch deepspeed wandb

# Run training
cd 01_basic_neuralnet
deepspeed --num_gpus=1 train_ds.py
```

## Repository Structure

```
deepspeed-course/
├── 01_basic_neuralnet/          # Simple linear regression
├── 02_basic_convnet/            # CNN on synthetic MNIST
├── 02_basic_convnet_cifar10/    # CIFAR-10 (81% accuracy)
├── 03_basic_rnn/                # LSTM time series
├── 04_bayesian_neuralnet/       # Parallel tempering MCMC
├── 04_intermediate_rnn_stock/   # Real stock prediction
├── 05_huggingface/              # LLM fine-tuning
├── 05_huggingface_trl/          # Function calling
├── 05_huggingface_ocr/          # Vision-language OCR
├── 06_huggingface_grpo/         # GRPO training
├── 07_huggingface_gpt_oss/      # GPT-OSS-20B fine-tuning
├── 07_huggingface_multi_agent/  # Multi-agent systems
├── 08_vtt/                      # Video-text training
└── 09_vss/                      # Video-speech training
```

## Next Steps

- [Installation Guide](/docs/getting-started/installation) - Set up your environment
- [Quick Start](/docs/getting-started/quick-start) - Run your first DeepSpeed training
- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Understand memory optimization
