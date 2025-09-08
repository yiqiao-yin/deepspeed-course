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
├── 01_basic_neuralnet
├── 02_basic_convnet
├── 02_basic_convnet_cifar10_examples
├── 03_basic_rnn
├── 04_transferlearning
├── 05_huggingface
├── 05_huggingface_trl
├── 06_huggingface_grpo
├── 07_huggingface_openai_gpt_oss_finetune_sft
├── 07_huggingface_trl_multi_agency
├── 08_vtt
└── README.md
```

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