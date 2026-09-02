#!/bin/bash
# SLURM batch script — Omni evaluation — modality ablation (does it use BOTH streams?)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 05_video_speech/04_omni_eval \
#         --collect --wait --terminate --yes
#
# No GPU? The algorithm this folder teaches runs on CPU:
#     uv run 05_video_speech/04_omni_eval/omni_eval.py
#     uv run tests/test_omni_eval.py      # 49 checks, no GPU, no download

#SBATCH --gres=gpu:1
# ONE GPU. Evaluation is a series of short generate() calls; the win comes
# from batching questions, not from sharding the model.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=04:00:00

#SBATCH --job-name=omni_eval

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# Decoding TWO streams — video frames and 16 kHz audio — is CPU-bound. Too few
# cores starves the dataloader and the GPUs idle between batches.

#SBATCH --mem=64G

#SBATCH --output=logs/omni_eval_%j.out
#SBATCH --error=logs/omni_eval_%j.err

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
DATASET="${DATASET:-}"
ASR_WER="${ASR_WER:-0.05}"

# ALWAYS run the simulated harness check first. If the ablation grid cannot
# catch a deliberately modality-ignoring model, every number after it is
# meaningless.
echo ""
echo "--- harness self-check: a video-ignoring model MUST be caught ---"
python omni_eval.py --video-skill 0 --fusion-skill 0 \
    --output eval_selfcheck.json

echo ""
echo "--- real model, full ablation grid ---"
if [ -n "${DATASET}" ]; then
    python omni_eval.py --model "${MODEL}" --dataset "${DATASET}" \
        --asr-wer "${ASR_WER}" --output eval_results.json "$@"
else
    python omni_eval.py --model "${MODEL}" \
        --asr-wer "${ASR_WER}" --output eval_results.json "$@"
fi

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
