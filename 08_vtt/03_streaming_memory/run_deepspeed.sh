#!/bin/bash
# SLURM batch script — streaming video understanding in constant memory
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 08_vtt/03_streaming_memory \
#         --collect --wait --terminate --yes
#
# The memory mechanics need NO GPU and no download — run them anywhere:
#     uv run 08_vtt/03_streaming_memory/stream_infer.py --frames 20000
#     uv run tests/test_star_memory.py

#SBATCH --gres=gpu:1
# ONE GPU. Streaming inference is inherently sequential — frames arrive in
# order — so extra GPUs do not help a single stream. Scale by running more
# streams, not by sharding one.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=02:00:00

#SBATCH --job-name=vtt_streaming

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G
# Modest, and that is the entire claim being demonstrated: host memory does
# not grow with stream length either. If this job's RSS climbs over the run,
# something is retaining frames and the O(1) property is broken.

#SBATCH --output=logs/streaming_%j.out
#SBATCH --error=logs/streaming_%j.err

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
#   uv pip install deepspeed transformers accelerate opencv-python-headless
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Optional experiment tracking. Leave COMMENTED — an uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error, because `<` redirects.
# export WANDB_API_KEY="your_value_here"

nvidia-smi

FRAMES="${FRAMES:-20000}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

# A long stream is the whole demonstration. 20,000 frames at 2 fps is nearly
# three hours of video through a context that never exceeds ~306 tokens.
python stream_infer.py \
    --frames "${FRAMES}" \
    --query-every 2000 \
    --model "${MODEL}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
