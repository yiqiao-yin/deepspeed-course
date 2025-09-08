# DeepSpeed Course

## Situation Today

Training and inference for deep learning models are often slow and resource-intensive, especially as model sizes and dataset complexity grow. This bottleneck impacts productivity and limits experimentation, making it difficult to iterate quickly or deploy models efficiently.

## Problem Statement

To overcome these challenges, it's essential to leverage multiple GPUs and distributed training. DeepSpeed is a deep learning optimization library that enables faster training, efficient memory usage, and scalable distributed training across multiple GPUs. Using DeepSpeed can significantly reduce training time and improve inference speed, making it possible to work with larger models and datasets.

## Solution

This repository provides a collection of basic frameworks and examples demonstrating how to use DeepSpeed for distributed training and inference. Each folder contains a different neural network architecture or use case, showing how DeepSpeed can be integrated to accelerate workflows.

### Folder Structure

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

Explore each folder for step-by-step guides and code samples to accelerate your deep learning projects with