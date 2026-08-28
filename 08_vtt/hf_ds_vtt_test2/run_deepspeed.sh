#!/bin/bash
# SLURM batch script for video-text training (LLaVA or NLLB seq2seq)
# Set TRAINER=llava or TRAINER=seq2seq to choose the path

#SBATCH --gres=gpu:2
# 2 GPUs. LLaVA needs 40GB+ each; seq2seq is far lighter.

#SBATCH --partition=h200-low
# Update this to match your cluster's partition names (check with: sinfo)

#SBATCH --time=06:00:00
# Maximum wall-clock time. LLaVA is slow; visual tokens dominate sequence length.

#SBATCH --job-name=vtt_train

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# CPU cores for the data pipeline. Too few starves the dataloader and leaves
# the GPU idle between batches.

#SBATCH --mem=96G
# Host memory. Video decoding plus model. Raise for more frames.

#SBATCH --output=logs/vtt_train_%j.out
#SBATCH --error=logs/vtt_train_%j.err

mkdir -p logs

echo "=================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Start:    $(date)"
echo "=================================================="

# Activate your environment (built once on a LOGIN node with uv):
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
#   uv pip install deepspeed transformers datasets accelerate trl huggingface_hub
#   uv pip install opencv-python-headless  # REQUIRED for frame extraction
source ~/myenv/bin/activate

# Point the HuggingFace cache at scratch. $HOME is usually a small NFS quota,
# and a multi-GB model download into it fails slowly.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Requires HF credentials; both trainers push results to the Hub.
# export HF_TOKEN=<ENTER_KEY_HERE>
# export HF_USER_ID=<your-username>

# Optional experiment tracking. Leave unset to skip it; the scripts handle that.
# export WANDB_API_KEY=<ENTER_KEY_HERE>

nvidia-smi

TRAINER="${TRAINER:-seq2seq}"
if [ "$TRAINER" = "llava" ]; then
    cd llava_video_trainer
    deepspeed --num_gpus=2 video_training_script.py
else
    cd seq2seq_video_trainer
    deepspeed --num_gpus=2 video_text_trainer.py
fi

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
