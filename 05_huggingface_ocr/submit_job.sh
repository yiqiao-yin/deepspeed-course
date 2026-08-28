#!/bin/bash
#SBATCH --job-name=vlm-finetune
#SBATCH --output=logs/vlm-finetune-%j.out
#SBATCH --error=logs/vlm-finetune-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

################################################################################
# SLURM Job Submission Script for Vision-Language Model Fine-tuning
#
# This script is designed for CoreWeave bare metal or any SLURM-based cluster
# with 2 GPUs for distributed training using DeepSpeed.
#
# Usage:
#   sbatch submit_job.sh
#
# Monitoring:
#   squeue -u $USER              # Check job status
#   scancel <job-id>             # Cancel job
#   tail -f logs/vlm-finetune-<job-id>.out  # Monitor output
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Avoid tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Optional: Set HuggingFace cache directory
# export HF_HOME=/path/to/hf_cache
# export TRANSFORMERS_CACHE=/path/to/transformers_cache

# Print GPU information
echo "=================================================="
echo "GPU Information:"
nvidia-smi
echo "=================================================="

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
fi

# Create project directory if it doesn't exist
PROJECT_DIR="proj"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Initializing project with uv..."
    uv init $PROJECT_DIR
fi

cd $PROJECT_DIR

# Install dependencies if not already installed
echo "Installing dependencies..."
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv add transformers datasets accelerate peft
uv add deepspeed bitsandbytes
uv add pillow wandb

# Copy training script to project directory
if [ ! -f "train_ds.py" ]; then
    echo "Copying training script..."
    cp ../train_ds.py .
fi

# Configuration variables (modify as needed)
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_EPOCHS=1
LEARNING_RATE=5e-5
OUTPUT_DIR="outputs"
MAX_SAMPLES=10

# Create output directory
mkdir -p $OUTPUT_DIR

# Optional: Login to HuggingFace (uncomment and set your token)
# export HF_TOKEN="your_token_here"
# huggingface-cli login --token $HF_TOKEN

# Optional: Login to Weights & Biases (uncomment and set your API key)
# export WANDB_API_KEY="your_api_key_here"
# wandb login $WANDB_API_KEY

echo "=================================================="
echo "Starting Training with DeepSpeed"
echo "Model: $MODEL_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="

# Run training with DeepSpeed
# Note: deepspeed launcher will automatically use SLURM environment variables
uv run deepspeed --num_gpus=2 train_ds.py \
    --model-name "$MODEL_NAME" \
    --use-4bit \
    --use-lora \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM_STEPS \
    --num-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --output-dir $OUTPUT_DIR \
    --max-samples $MAX_SAMPLES

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    echo "End Time: $(date)"
    echo "=================================================="

    # Optional: Copy results to permanent storage
    # cp -r $OUTPUT_DIR /path/to/permanent/storage/

    # Optional: Upload to HuggingFace Hub
    # uv run python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('$OUTPUT_DIR'); model.push_to_hub('your-username/model-name')"
else
    echo "=================================================="
    echo "Training failed with exit code: $?"
    echo "Check logs for details"
    echo "=================================================="
    exit 1
fi

# Print job statistics
echo "=================================================="
echo "Job Statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,State,Time,Elapsed,MaxRSS,MaxVMSize
echo "=================================================="
