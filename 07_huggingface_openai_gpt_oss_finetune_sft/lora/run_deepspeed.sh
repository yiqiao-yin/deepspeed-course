#!/bin/bash
# SLURM batch script for GPT-OSS-20B fine-tuning with LoRA and DeepSpeed
# Fine-tunes OpenAI GPT-OSS-20B on multilingual reasoning using LoRA + ZeRO-2

#SBATCH --gres=gpu:4
# Request 4 GPUs (GPT-OSS-20B is 20B parameters)
# With LoRA, memory usage is ~25-30GB per GPU

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue
# Update this based on your CoreWeave partition names

#SBATCH --time=06:00:00
# Maximum wall-clock time: 6 hours
# Typical runtime: 3-4 hours for 10 epochs on Multilingual-Thinking dataset
# Allocating extra time for dataset download and model initialization

#SBATCH --job-name=gpt_oss_lora
# Job name: GPT-OSS-20B LoRA fine-tuning

#SBATCH --ntasks-per-node=1
# Number of tasks per node (1 for multi-GPU training with DeepSpeed)

#SBATCH --cpus-per-task=32
# Number of CPU cores for data loading and preprocessing
# 32 cores ensures smooth data pipeline

#SBATCH --mem=256G
# Total memory per node: 256 GB
# Needed for:
#   - Model initialization (~80GB for 20B model)
#   - Dataset loading (~10GB)
#   - DeepSpeed optimizer states (~40GB with ZeRO-2)
#   - System overhead (~10GB)

#SBATCH --output=logs/gpt_oss_lora_%j.out
# Standard output log: logs/gpt_oss_lora_<job_id>.out

#SBATCH --error=logs/gpt_oss_lora_%j.err
# Standard error log: logs/gpt_oss_lora_<job_id>.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Optional: Export Weights & Biases API key for experiment tracking
# Get your API key from: https://wandb.ai/authorize
# IMPORTANT: This is optional - script will run without it
export WANDB_API_KEY=<ENTER_KEY_HERE>

# Optional: Export HuggingFace token for pushing models to hub
# Get your token from: https://huggingface.co/settings/tokens
# IMPORTANT: This is optional - script will run without it and save locally
export HF_TOKEN=<ENTER_KEY_HERE>

# Optional: Set HuggingFace username for model uploads
export HF_USER=your_hf_username

# Optional: Disable pushing to hub (even if HF_TOKEN is set)
# export PUSH_TO_HUB=false

# Optional: Disable evaluation after training
# export RUN_EVALUATION=false

# Activate Python virtual environment
# Update this path to your actual virtual environment location
# If using uv: source ~/myenv/bin/activate
source ~/myenv/bin/activate

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=================================================="

# Print configuration
echo "Configuration:"
echo "  - Model: openai/gpt-oss-20b (20B parameters)"
echo "  - Dataset: HuggingFaceH4/Multilingual-Thinking"
echo "  - Method: LoRA (r=8, alpha=16)"
echo "  - DeepSpeed: ZeRO-2 with BF16"
echo "  - Batch size: 2 per GPU, gradient accumulation: 8"
echo "  - Effective batch size: 2 * 4 GPUs * 8 = 64"
echo "  - Epochs: 10"
echo "  - Learning rate: 2e-4"
echo "  - W&B Tracking: $([ -n "$WANDB_API_KEY" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - HF Push: $([ -n "$HF_TOKEN" ] && echo "ENABLED" || echo "DISABLED")"
echo "=================================================="

# Launch training with DeepSpeed
# Configuration:
#   - Model: OpenAI GPT-OSS-20B (20B parameters)
#   - Dataset: HuggingFaceH4/Multilingual-Thinking
#   - Optimization: LoRA (r=8, alpha=16) + ZeRO-2 + BF16
#   - Batch size: 2 per GPU, gradient accumulation: 8
#   - Effective batch size: 2 * 4 GPUs * 8 = 64
#   - Training: 10 epochs
echo "Starting GPT-OSS-20B LoRA fine-tuning with DeepSpeed..."

deepspeed --num_gpus=4 train_ds.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo "Output location: ./gpt-oss-20b-multilingual-reasoner-lora/"
    echo ""
    echo "To use the trained model:"
    echo "  from transformers import AutoModelForCausalLM"
    echo "  from peft import PeftModel"
    echo ""
    echo "  base_model = AutoModelForCausalLM.from_pretrained('openai/gpt-oss-20b')"
    echo "  model = PeftModel.from_pretrained(base_model, './gpt-oss-20b-multilingual-reasoner-lora')"
    echo "  model = model.merge_and_unload()  # Optional: merge LoRA weights"
    echo "=================================================="
else
    echo "=================================================="
    echo "Training failed with exit code: $?"
    echo "End Time: $(date)"
    echo "Check logs for errors:"
    echo "  - stdout: logs/gpt_oss_lora_${SLURM_JOB_ID}.out"
    echo "  - stderr: logs/gpt_oss_lora_${SLURM_JOB_ID}.err"
    echo "=================================================="
    exit 1
fi
