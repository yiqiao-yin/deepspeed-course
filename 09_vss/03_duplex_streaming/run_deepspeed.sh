#!/bin/bash
# SLURM batch script — Full-duplex conversation (listening and watching while speaking)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 09_vss/03_duplex_streaming \
#         --collect --wait --terminate --yes
#
# No GPU? The algorithm this folder teaches runs on CPU:
#     uv run 09_vss/03_duplex_streaming/duplex.py
#     uv run tests/test_duplex.py         # 36 checks, no GPU, no download

#SBATCH --gres=gpu:1
# ONE GPU. Duplex inference is inherently sequential — slices arrive in
# order — so extra GPUs do not help a single conversation. Scale by running
# more conversations, not by sharding one.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=02:00:00

#SBATCH --job-name=omni_duplex

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# Decoding TWO streams — video frames and 16 kHz audio — is CPU-bound. Too few
# cores starves the dataloader and the GPUs idle between batches.

#SBATCH --mem=64G

#SBATCH --output=logs/omni_duplex_%j.out
#SBATCH --error=logs/omni_duplex_%j.err

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
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
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

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-3B}"
SLICES="${SLICES:-200}"

# The number that decides whether this is shippable is the WORST-case RTF, not
# the mean. RTF >= 1 is not slowness — the backlog grows without bound and the
# conversation collapses.
python run_duplex.py \
    --model "${MODEL}" \
    --slices "${SLICES}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
