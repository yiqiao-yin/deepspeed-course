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

## Understanding DeepSpeed ZeRO Stages

DeepSpeed's **ZeRO (Zero Redundancy Optimizer)** is a memory optimization technique that eliminates redundant memory consumption by partitioning model states across multiple GPUs. Traditional data-parallel training replicates the entire model, optimizer states, and gradients on every GPU, leading to massive memory redundancy. ZeRO addresses this by sharding these components across GPUs while maintaining computational efficiency.

### ZeRO Stage Progression

**ZeRO Stage 1** partitions only the optimizer states across GPUs. Each GPU maintains a portion of the optimizer states (e.g., Adam's momentum and variance) but still keeps full copies of model parameters and gradients. This provides modest memory savings (~4x reduction in optimizer memory) with minimal communication overhead, making it suitable for models that are close to fitting in GPU memory.

**ZeRO Stage 2** extends Stage 1 by additionally partitioning gradients across GPUs. Each GPU only stores gradients for its assigned portion of parameters. This provides more significant memory savings (~8x reduction in optimizer + gradient memory) while introducing slightly more communication during the backward pass. Stage 2 is the sweet spot for most training scenarios, balancing memory efficiency with communication overhead.

**ZeRO Stage 3** partitions everything: optimizer states, gradients, AND model parameters. Each GPU only stores a fraction of the model weights, which are gathered on-demand during forward and backward passes. This enables training models that are far too large to fit on a single GPU but introduces the highest communication overhead. Stage 3 is essential for very large models (13B+) or when GPU memory is severely constrained.

### ZeRO Stages Comparison Table

| Feature | Stage 1 | Stage 2 | Stage 3 |
|---------|---------|---------|---------|
| **Optimizer States** | Partitioned ✓ | Partitioned ✓ | Partitioned ✓ |
| **Gradients** | Replicated | Partitioned ✓ | Partitioned ✓ |
| **Model Parameters** | Replicated | Replicated | Partitioned ✓ |
| **Memory Savings** | ~4x optimizer | ~8x optimizer+gradients | Up to N×GPUs total memory |
| **Communication Overhead** | Low | Medium | High |
| **Best For** | Near-fitting models | Most use cases | Very large models (13B+) |
| **Min GPUs Required** | 2+ | 2+ | 2+ (4+ recommended) |

### Memory Reduction Formula

For a model with **P** parameters:
- **Stage 1**: Saves ~4P bytes (optimizer states only)
- **Stage 2**: Saves ~6P bytes (optimizer + gradients)
- **Stage 3**: Saves ~16P bytes (everything), scaling with number of GPUs

### ZeRO Stage Recommendations by GPU Configuration

| GPU Setup | Model Size | Recommended ZeRO Stage | Rationale |
|-----------|------------|------------------------|-----------|
| **2x RTX 4060 Ti (16GB)** | 2-3B | Stage 2 + 4-bit + LoRA | Limited VRAM; Stage 2 provides good balance |
| **2x RTX 4090 (24GB)** | 2-7B | Stage 2 | Sufficient memory; Stage 2 optimal efficiency |
| **4x RTX 4090 (24GB)** | 7-13B | Stage 2 or Stage 3 | Stage 3 if model barely fits |
| **2x RTX 5090 (32GB)** | 2-7B | Stage 2 | More VRAM; Stage 2 is ideal |
| **4x RTX 5090 (32GB)** | 7-20B | Stage 2 or Stage 3 | Stage 3 for 13B+ models |
| **2x A100 40GB** | 7-13B | Stage 2 | High bandwidth reduces Stage 2 overhead |
| **4x A100 40GB** | 13-20B | Stage 3 | Stage 3 needed for large models |
| **8x A100 40GB** | 20-70B | Stage 3 | Stage 3 essential; consider CPU offload |
| **2x A100 80GB** | 7-20B | Stage 2 | Large VRAM; Stage 2 sufficient |
| **4x A100 80GB** | 20-65B | Stage 3 | Stage 3 for max model size |
| **8x A100 80GB** | 65B-175B | Stage 3 + CPU offload | Stage 3 with offloading for huge models |
| **2x H100 80GB** | 13-30B | Stage 2 or Stage 3 | High bandwidth; Stage 2 preferred if fits |
| **4x H100 80GB** | 30-70B | Stage 3 | Stage 3 for large models |
| **8x H100 80GB** | 70B-200B+ | Stage 3 + optimizations | Stage 3 with NVLink for max throughput |

### When to Use Each Stage

**Use Stage 1 when:**
- Your model almost fits in GPU memory
- You want minimal communication overhead
- Using 2-4 GPUs with high-bandwidth interconnect (NVLink)
- Model size: <3B parameters on consumer GPUs

**Use Stage 2 when:**
- Standard training scenario (most common choice)
- 2-8 GPUs with moderate memory constraints
- Good balance of memory savings and speed needed
- Model size: 2-13B parameters
- **This is the default in this repository**

**Use Stage 3 when:**
- Model cannot fit in GPU memory even with Stage 2
- Training very large models (13B+)
- Have 4+ GPUs to distribute parameters effectively
- Willing to accept communication overhead for memory savings
- Model size: 13B+ parameters

### Communication Overhead Comparison

| ZeRO Stage | Forward Pass | Backward Pass | Optimizer Step | Total Overhead |
|------------|--------------|---------------|----------------|----------------|
| **Stage 1** | No overhead | No overhead | Reduce-scatter | ~5-10% |
| **Stage 2** | No overhead | Reduce-scatter | Reduce-scatter | ~10-15% |
| **Stage 3** | All-gather | Reduce-scatter + All-gather | Reduce-scatter | ~20-30% |

### Modifying ZeRO Stage in This Repository

To change the ZeRO stage, edit `ds_config.json`:

```json
{
  "zero_optimization": {
    "stage": 2,  // Change to 1, 2, or 3
    ...
  }
}
```

**For Stage 3**, additional configuration is recommended:

```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    ...
  }
}
```

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
