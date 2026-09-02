#!/bin/bash
# SLURM batch script — visual token compression, MEASURED against real VRAM
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 04_video_text/03_token_compression \
#         --collect --wait --terminate --yes
#
# No GPU at all? The algorithms are CPU-testable and that is the point:
#     uv run tests/test_token_compression.py
#     uv run 04_video_text/03_token_compression/token_compression.py

#SBATCH --gres=gpu:1
# ONE GPU on purpose. This script measures peak VRAM, and sharding across
# devices would mix ZeRO's saving into a number that is supposed to isolate
# the effect of sequence length. Measure the variable you are changing.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=01:00:00
# Short: this is a measurement sweep, not a training run.

#SBATCH --job-name=vtt_compression

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns its own worker per GPU.

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --output=logs/compression_%j.out
#SBATCH --error=logs/compression_%j.err

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
#   uv pip install deepspeed transformers accelerate peft opencv-python-headless
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Optional experiment tracking. Leave COMMENTED — an uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error, because `<` redirects.
# export WANDB_API_KEY="your_value_here"

nvidia-smi

FRAMES="${FRAMES:-32}"

# Sweep the frame budget. The interesting result is not any single row, it is
# where the compressed curve stops tracking the uncompressed one — that is the
# point at which activations overtook weights as the dominant memory term.
for frames in 8 16 "${FRAMES}"; do
    echo ""
    echo "--- ${frames} frames ---"
    deepspeed --num_gpus=1 train_compressed.py \
        --frames "${frames}" \
        --output "compression_results_${frames}f.json" \
        "$@"
done

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
