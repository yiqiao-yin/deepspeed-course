#!/bin/bash
# SLURM batch script — Thinker-Talker omni fine-tuning (video+speech in, speech out)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 09_vss/02_thinker_talker \
#         --collect --wait --terminate --yes
#
# No GPU? The algorithm this folder teaches runs on CPU:
#     uv run 09_vss/02_thinker_talker/tmrope.py
#     uv run tests/test_tmrope.py         # 59 checks, no GPU, no download

#SBATCH --gres=gpu:2
# 2 GPUs. Qwen2.5-Omni-3B + LoRA fits ONE 24GB card; two is so the
# ZeRO-3 sharding is worth demonstrating. Raise to 4 for the 7B.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=06:00:00

#SBATCH --job-name=omni_thinker_talker

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# Decoding TWO streams — video frames and 16 kHz audio — is CPU-bound. Too few
# cores starves the dataloader and the GPUs idle between batches.

#SBATCH --mem=128G

#SBATCH --output=logs/omni_thinker_talker_%j.out
#SBATCH --error=logs/omni_thinker_talker_%j.err

set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:    $(date)"
echo "=================================================="

# Environment, built ONCE on a LOGIN node with uv. Compute nodes usually have
# no network egress, so building it inside the job fails.
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
#   uv pip install deepspeed transformers accelerate peft datasets
#   uv pip install librosa soundfile opencv-python-headless
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

NUM_GPUS="${NUM_GPUS:-2}"
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-3B}"

deepspeed --num_gpus="${NUM_GPUS}" train_omni.py \
    --deepspeed ds_config.json \
    --model "${MODEL}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
