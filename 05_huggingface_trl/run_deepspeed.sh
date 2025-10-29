#!/bin/bash
# SLURM batch script for TRL Function Calling Training with DeepSpeed
# This script fine-tunes Qwen/Qwen3-0.6B on tool-augmented dataset
# for function calling capabilities using TRL's SFTTrainer

#SBATCH --gres=gpu:2
# Request 2 GPUs for distributed training

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=01:00:00
# Maximum wall-clock time: 1 hour
# Training 3 epochs on ~200 samples takes ~10-15 minutes on 2 GPUs

#SBATCH --job-name=trl_qwen_sft_ds
# Job name: TRL Qwen SFT with DeepSpeed

#SBATCH --ntasks-per-node=1
# Number of tasks per node (DeepSpeed handles multi-GPU internally)

#SBATCH --cpus-per-task=8
# Number of CPU cores (8 cores for efficient data loading)

#SBATCH --mem=32G
# Total memory per node: 32 GB
# Qwen3-0.6B model ~1.2GB, but need space for gradients and optimizer states

#SBATCH --output=logs/trl_qwen_%j.out
# Standard output log

#SBATCH --error=logs/trl_qwen_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export HuggingFace token (optional - only needed for gated models)
# Qwen/Qwen3-0.6B is publicly available, so not required
# export HF_TOKEN=<ENTER_KEY_HERE>

# Export Weights & Biases API key for experiment tracking (OPTIONAL)
# Get your API key from: https://wandb.ai/authorize
# Uncomment the line below and add your key to enable W&B tracking
# export WANDB_API_KEY=<ENTER_KEY_HERE>

# Activate Python virtual environment
# Option 1: Using uv (recommended for faster dependency management)
# pip install uv && uv init . && uv add torch transformers trl datasets deepspeed wandb
# source .venv/bin/activate

# Option 2: Using traditional virtualenv
# Update this path to your actual virtual environment location
source ~/myenv/bin/activate

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Verify dataset exists
if [ ! -f "tool_augmented_dataset.json" ]; then
    echo "❌ ERROR: tool_augmented_dataset.json not found!"
    echo "   Please ensure the dataset file is in the current directory."
    exit 1
fi

echo "✅ Dataset file found: tool_augmented_dataset.json"

# Clean up previous training artifacts (optional)
# Uncomment if you want a fresh start each run
# echo "Cleaning up previous training data..."
# rm -rf ./sft_qwen_model ./logs ./wandb
# echo "Cleanup complete."

# Launch TRL training with DeepSpeed
# Features:
# - Model: Qwen/Qwen3-0.6B (600M parameters)
# - Task: Function calling / Tool use
# - Trainer: TRL SFTTrainer
# - Optimizer: AdamW (lr=2e-5, warmup=100 steps)
# - ZeRO Stage 2 optimization for memory efficiency
# - Gradient clipping (1.0)
# - FP32 precision for numerical stability
# - Optional W&B tracking (if WANDB_API_KEY set)
echo ""
echo "🚀 Starting TRL Function Calling Training with DeepSpeed"
echo "   - Model: Qwen/Qwen3-0.6B"
echo "   - Dataset: tool_augmented_dataset.json"
echo "   - Training: 3 epochs with 2 GPUs"
echo "   - Output: ./sft_qwen_model"
echo ""

deepspeed --num_gpus=2 train_trl_deepspeed.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Training completed successfully!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo ""
    echo "📦 Model saved to: ./sft_qwen_model"
    echo ""
    echo "💡 Next Steps:"
    echo "   1. Load model: AutoModelForCausalLM.from_pretrained('./sft_qwen_model')"
    echo "   2. Test with: inference_trl_model.py"
    echo "   3. Check W&B dashboard for metrics (if enabled)"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "❌ Training failed!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo ""
    echo "🔍 Check logs for errors:"
    echo "   - Standard output: logs/trl_qwen_${SLURM_JOB_ID}.out"
    echo "   - Standard error: logs/trl_qwen_${SLURM_JOB_ID}.err"
    echo ""
    exit 1
fi
