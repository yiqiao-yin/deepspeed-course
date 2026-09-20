#!/bin/bash
# SLURM batch script — Qwen3-VL video fine-tuning (DeepSpeed ZeRO-3 + LoRA)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 04_video_text/02_qwen3vl \
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:1
# ONE, and the number is measured rather than guessed. Qwen3-VL-8B is 17.5 GB
# of bf16 weights; this lab measured a floor of 18.8 GB and 11.97 GB per 1,000
# visual tokens, so a single 48 GB card holds roughly 33 frames. Two cards buy
# nothing here unless ZeRO-3 shards the weights, and the floor -- not the
# weights -- is what dominates at these sequence lengths.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=04:00:00

#SBATCH --job-name=qwen3vl_video

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# Video decoding is CPU-bound. Too few cores starves the dataloader and the
# GPUs idle between batches — which looks like a slow model and is not.

#SBATCH --mem=96G

#SBATCH --output=logs/qwen3vl_%j.out
#SBATCH --error=logs/qwen3vl_%j.err

set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:    $(date)"
echo "=================================================="

# Environment, built once on a LOGIN node with uv:
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#   uv pip install deepspeed transformers accelerate peft datasets
#   uv pip install qwen-vl-utils opencv-python-headless
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

# $HOME is usually a small NFS quota and a multi-GB model download into it
# fails slowly. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Optional experiment tracking. Leave COMMENTED — an uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error, because `<` redirects.
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-1}"
MAX_FRAMES="${MAX_FRAMES:-16}"

deepspeed --num_gpus="${NUM_GPUS}" train_qwen3vl.py \
    --deepspeed ds_config.json \
    --max-frames "${MAX_FRAMES}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
