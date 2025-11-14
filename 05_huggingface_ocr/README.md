# Vision-Language Model Fine-tuning with DeepSpeed

Minimal Vision-Language Model (VLM) fine-tuning script using DeepSpeed for distributed training on 2 RTX 4000-series NVIDIA GPUs. This example uses the Qwen2-VL-2B-Instruct model for OCR and vision-language tasks.

## Prerequisites

- 2x RTX 4000-series NVIDIA GPUs
- CUDA 11.8 or higher
- Python 3.8+
- `uv` package manager

## Getting Started

### 1. Install uv

If you haven't already installed `uv`, install it first:

```bash
pip install uv
```

### 2. Initialize Project

Create a new project directory and initialize it with `uv`:

```bash
uv init proj
cd proj
```

### 3. Install Dependencies

Install all required packages using `uv add`:

```bash
# Install PyTorch with CUDA support (CUDA 11.8)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face and training libraries
uv add transformers datasets accelerate peft

# Install DeepSpeed and optimization libraries
uv add deepspeed bitsandbytes

# Install image processing
uv add pillow

# Optional: Install Weights & Biases for experiment tracking
uv add wandb
```

### 4. Copy Training Script

Copy the `train_ds.py` file to your project directory:

```bash
cp ../train_ds.py .
```

### 5. Run Training

Run the training script with DeepSpeed using 2 GPUs:

```bash
uv run deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora
```

#### Training Options

The script supports various configuration options:

**Model Configuration:**
- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2-VL-2B-Instruct`)
- `--use-4bit`: Enable 4-bit quantization for reduced memory usage
- `--use-lora`: Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha parameter (default: 16)
- `--lora-dropout`: LoRA dropout rate (default: 0.05)

**Training Configuration:**
- `--batch-size`: Batch size per device (default: 1)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--num-epochs`: Number of training epochs (default: 1)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--output-dir`: Output directory for checkpoints (default: `outputs`)

**System Configuration:**
- `--no-deepspeed`: Disable DeepSpeed (run on single GPU)

#### Example Commands

```bash
# Basic training with default settings
uv run deepspeed --num_gpus=2 train_ds.py

# Training with 4-bit quantization and LoRA
uv run deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora

# Single GPU training (without DeepSpeed)
uv run python train_ds.py --no-deepspeed --use-4bit --use-lora

# Custom configuration
uv run deepspeed --num_gpus=2 train_ds.py \
  --use-4bit \
  --use-lora \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --output-dir ./custom_output
```

## Features

- **DeepSpeed ZeRO Stage 2**: Efficient distributed training with gradient and optimizer state partitioning
- **4-bit Quantization**: Reduced memory footprint using bitsandbytes
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **Gradient Checkpointing**: Memory optimization for large models
- **FP16 Mixed Precision**: Faster training with reduced memory usage
- **Synthetic Dataset**: Built-in sample data generation for testing

## Output

The training script will:
1. Generate a DeepSpeed configuration file (`ds_config.json`)
2. Download and load the specified model
3. Create synthetic training data (or use your custom dataset)
4. Train the model across 2 GPUs
5. Save checkpoints to the output directory

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` to 1
- Increase `--gradient-accumulation-steps`
- Enable `--use-4bit` quantization
- Enable `--use-lora` for parameter-efficient training

### DeepSpeed Initialization Errors
- Ensure CUDA toolkit is properly installed
- Verify both GPUs are accessible: `nvidia-smi`
- Check DeepSpeed installation: `ds_report`

### Model Download Issues
- Ensure stable internet connection
- HuggingFace models may require authentication for some models
- Set `HF_TOKEN` environment variable if needed

## Next Steps

- Replace synthetic dataset with your own OCR/VLM dataset
- Adjust hyperparameters based on your dataset size
- Enable W&B tracking for experiment monitoring
- Fine-tune on domain-specific vision-language tasks

## See Also

- [HARDWARE_REQUIREMENTS.md](./HARDWARE_REQUIREMENTS.md) - GPU requirements and recommendations
- [submit_job.sh](./submit_job.sh) - SLURM job submission script for CoreWeave
