---
sidebar_position: 4
---

# Hardware Requirements

Guide for selecting appropriate hardware for DeepSpeed training.

## GPU Comparison

### Consumer GPUs

| GPU | VRAM | FP16 TFLOPS | Best For |
|-----|------|-------------|----------|
| RTX 3060 | 12 GB | 12.7 | Small models, learning |
| RTX 3070 | 8 GB | 20.3 | Development, LoRA |
| RTX 3080 | 10 GB | 29.8 | Medium models |
| RTX 3090 | 24 GB | 35.6 | Large models |
| RTX 4070 | 12 GB | 29.1 | Efficient training |
| RTX 4080 | 16 GB | 48.7 | Good balance |
| RTX 4090 | 24 GB | 82.6 | Best consumer |

### Professional GPUs

| GPU | VRAM | FP16 TFLOPS | Best For |
|-----|------|-------------|----------|
| A40 | 48 GB | 37.4 | Multi-model |
| A100 | 40/80 GB | 77.9 | Production |
| H100 | 80 GB | 267.6 | Large scale |
| H200 | 141 GB | 267.6 | Massive models |

## Model Size Guidelines

### Memory Estimation

```
Full Training Memory ≈
    Model Size × 4 (weights)
    + Model Size × 4 (gradients)
    + Model Size × 8 (optimizer states for Adam)
    = Model Size × 16
```

### With ZeRO Optimization

| ZeRO Stage | Memory Reduction |
|------------|------------------|
| Stage 1 | ~4x less optimizer memory |
| Stage 2 | ~8x less (optimizer + gradients) |
| Stage 3 | Linear scaling with GPUs |

## Model → GPU Mapping

| Model Size | ZeRO Stage | GPU Config |
|------------|------------|------------|
| < 1B | Stage 2 | 1x RTX 3070 (8GB) |
| 1-3B | Stage 2 | 1x RTX 4090 (24GB) |
| 3-7B | Stage 2 + Offload | 2x RTX 4090 |
| 7-13B | Stage 3 | 4x A100 (40GB) |
| 13-30B | Stage 3 + Offload | 4x A100 (80GB) |
| 30-70B | Stage 3 + Offload | 8x H100 |
| 70B+ | Stage 3 + Offload | 8x H200 |

## Example Configurations

### Basic Examples (01-04)

```
Minimum: 1x RTX 3060 (12GB)
Recommended: 1x RTX 4090 (24GB)
```

### HuggingFace LLMs (05-07)

```
Qwen3-0.6B: 1x RTX 3070 (8GB) with LoRA
Mistral-7B: 2x RTX 4090 (48GB) with LoRA
GPT-OSS-20B: 4x A100 (160GB) with LoRA
```

### Multimodal (08-09)

```
LLaVA 7B: 2x A100 (80GB)
Qwen2-VL: 2x RTX 4090 (48GB)
LongCat-560B: 8x H200 (1.1TB)
```

## System Requirements

### RAM

| Training Type | Minimum RAM |
|--------------|-------------|
| Basic examples | 16 GB |
| HuggingFace (no offload) | 32 GB |
| HuggingFace (with offload) | 64 GB |
| Large models (Stage 3) | 256+ GB |

### Storage

| Use Case | Storage |
|----------|---------|
| Basic training | 50 GB |
| HuggingFace models | 100 GB |
| Large models (70B+) | 500+ GB |
| LongCat-Flash-Omni | 1.1+ TB |

### Network

For multi-node training:
- Minimum: 25 Gbps Ethernet
- Recommended: 100+ Gbps InfiniBand

## Cost Optimization

### Cloud Pricing (Approximate)

| Provider | GPU | $/hour |
|----------|-----|--------|
| RunPod | RTX 4090 | $0.74 |
| RunPod | A100 80GB | $1.89 |
| CoreWeave | H100 | $4.76 |
| AWS | p4d.24xlarge (8x A100) | $32.77 |

### Recommendations

1. **Development**: Use smallest GPU that works
2. **Experimentation**: Use spot/preemptible instances
3. **Production**: Size for throughput, not just fit

## Choosing Your Setup

### Decision Tree

```
Is your model < 1B parameters?
├── Yes → 1x consumer GPU (RTX 3070+)
└── No
    ├── Is it < 7B parameters?
    │   ├── Yes → 2x RTX 4090 or 1x A100
    │   └── No
    │       ├── Is it < 30B parameters?
    │       │   ├── Yes → 4x A100
    │       │   └── No → 8x H100/H200
```

## Next Steps

- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) - Memory optimization
- [Troubleshooting](/docs/reference/troubleshooting) - Common issues
