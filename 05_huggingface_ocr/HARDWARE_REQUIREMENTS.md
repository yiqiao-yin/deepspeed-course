# Hardware Requirements

This document outlines the hardware requirements for running the Vision-Language Model fine-tuning script (`train_ds.py`) across different GPU configurations.

## Minimum Viable Configuration

The script is designed and tested for:
- **GPUs**: 2x RTX 4000-series (e.g., RTX 4090, RTX 4080)
- **VRAM**: 16-24 GB per GPU
- **System RAM**: 32 GB minimum
- **CUDA**: 11.8 or higher
- **Storage**: 50 GB for model weights and checkpoints

## GPU Comparison Table

The following table compares different GPU configurations for running this training script:

| GPU Model | VRAM per GPU | # GPUs | Total VRAM | Batch Size | Grad Accum | Est. Training Time | Recommended Config | Notes |
|-----------|--------------|--------|------------|------------|------------|-------------------|-------------------|-------|
| **RTX 4060 Ti** | 16 GB | 2 | 32 GB | 1 | 8 | ~45 min | `--use-4bit --use-lora` | Minimal viable; requires quantization |
| **RTX 4070** | 12 GB | 2 | 24 GB | 1 | 8 | ~40 min | `--use-4bit --use-lora` | Requires quantization and LoRA |
| **RTX 4080** | 16 GB | 2 | 32 GB | 1 | 4 | ~35 min | `--use-4bit --use-lora` | Recommended minimum |
| **RTX 4090** | 24 GB | 2 | 48 GB | 2 | 4 | ~25 min | `--use-4bit --use-lora` | Good performance |
| **RTX 4090** | 24 GB | 4 | 96 GB | 2 | 2 | ~15 min | `--use-lora` | Can skip quantization |
| **RTX 5060 Ti** | 16 GB | 2 | 32 GB | 1 | 4 | ~40 min | `--use-4bit --use-lora` | Similar to RTX 4080 |
| **RTX 5070** | 16 GB | 2 | 32 GB | 1 | 4 | ~35 min | `--use-4bit --use-lora` | Improved performance vs 4070 |
| **RTX 5080** | 20 GB | 2 | 40 GB | 2 | 4 | ~30 min | `--use-4bit --use-lora` | Better memory headroom |
| **RTX 5090** | 32 GB | 2 | 64 GB | 2 | 2 | ~20 min | `--use-lora` | Can skip quantization |
| **RTX 5090** | 32 GB | 4 | 128 GB | 4 | 2 | ~12 min | Full precision | No quantization needed |
| **A100 40GB** | 40 GB | 2 | 80 GB | 2 | 2 | ~20 min | `--use-lora` | Enterprise option |
| **A100 80GB** | 80 GB | 2 | 160 GB | 4 | 1 | ~15 min | Full precision | No quantization needed |
| **H100 80GB** | 80 GB | 2 | 160 GB | 4 | 1 | ~10 min | Full precision | Best performance |

## Configuration Recommendations

### Budget Configuration (RTX 4060 Ti / RTX 4070)
```bash
uv run deepspeed --num_gpus=2 train_ds.py \
  --use-4bit \
  --use-lora \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```
- **Pros**: Most affordable option
- **Cons**: Requires quantization; slower training
- **Best for**: Learning, testing, small datasets

### Recommended Configuration (RTX 4090 / RTX 5080)
```bash
uv run deepspeed --num_gpus=2 train_ds.py \
  --use-4bit \
  --use-lora \
  --batch-size 2 \
  --gradient-accumulation-steps 4
```
- **Pros**: Good balance of cost and performance
- **Cons**: Still requires quantization for 2B+ models
- **Best for**: Development, medium datasets

### High-Performance Configuration (RTX 5090 / A100)
```bash
uv run deepspeed --num_gpus=2 train_ds.py \
  --use-lora \
  --batch-size 2 \
  --gradient-accumulation-steps 2
```
- **Pros**: No quantization needed; faster training
- **Cons**: Higher cost
- **Best for**: Production, large datasets

### Enterprise Configuration (4x A100/H100)
```bash
uv run deepspeed --num_gpus=4 train_ds.py \
  --batch-size 4 \
  --gradient-accumulation-steps 1
```
- **Pros**: Full precision; fastest training
- **Cons**: Highest cost
- **Best for**: Research, large-scale production

## Memory Requirements Breakdown

For the default Qwen2-VL-2B-Instruct model:

| Component | Memory Usage | With 4-bit | With LoRA |
|-----------|--------------|------------|-----------|
| Model Weights | ~8 GB | ~2 GB | ~8 GB |
| Optimizer States | ~16 GB | ~4 GB | ~1 GB |
| Gradients | ~8 GB | ~2 GB | ~1 GB |
| Activations | ~4 GB | ~2 GB | ~2 GB |
| **Total per GPU** | **~18 GB** | **~5 GB** | **~6 GB** |

Note: With DeepSpeed ZeRO Stage 2, optimizer states and gradients are partitioned across GPUs.

## Scaling Guidelines

### For Larger Models (7B+)
- Add more GPUs (4-8 GPUs recommended)
- Enable DeepSpeed ZeRO Stage 3
- Reduce batch size to 1
- Increase gradient accumulation steps

### For Smaller Models (<2B)
- Can run on single GPU
- Remove DeepSpeed: `--no-deepspeed`
- Increase batch size
- May not need quantization

## System Requirements

### CPU
- 16+ cores recommended for data preprocessing
- Intel Xeon or AMD EPYC preferred

### RAM
- **Minimum**: 32 GB
- **Recommended**: 64 GB
- **Optimal**: 128 GB+

### Storage
- **Minimum**: 50 GB free space
- **Recommended**: 200 GB SSD
- **Optimal**: 1 TB NVMe SSD

### Network (for distributed training)
- **Minimum**: 1 Gbps
- **Recommended**: 10 Gbps Ethernet
- **Optimal**: InfiniBand or 100 Gbps Ethernet

## Cloud/Bare Metal Recommendations

### CoreWeave
- Instance: RTX 4090 x2 or RTX A6000 x2
- Region: Choose closest to your location
- Storage: Persistent volume for checkpoints

### AWS
- Instance: p4d.24xlarge (8x A100 40GB)
- Region: us-east-1 or us-west-2
- Storage: EBS with provisioned IOPS

### Google Cloud
- Instance: a2-highgpu-2g (2x A100 40GB)
- Region: us-central1 or us-west1
- Storage: Persistent SSD

### Lambda Labs
- Instance: 2x RTX 4090 or 2x A100
- Most cost-effective for short training runs

## Troubleshooting Memory Issues

If you encounter OOM (Out of Memory) errors:

1. **Enable quantization**: Add `--use-4bit`
2. **Enable LoRA**: Add `--use-lora`
3. **Reduce batch size**: `--batch-size 1`
4. **Increase gradient accumulation**: `--gradient-accumulation-steps 8`
5. **Enable CPU offloading**: Modify `ds_config.json` to enable CPU offload (slower)
6. **Use smaller model**: Try Qwen2-VL-1.5B or similar

## Performance Optimization Tips

1. **Use NVMe storage** for faster checkpoint loading/saving
2. **Enable tensor cores** (automatic with FP16/BF16)
3. **Use latest CUDA toolkit** for best performance
4. **Monitor GPU utilization** with `nvidia-smi` or `nvtop`
5. **Profile training** with PyTorch Profiler to identify bottlenecks
