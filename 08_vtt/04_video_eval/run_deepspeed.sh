#!/bin/bash
# SLURM batch script — video understanding evaluation across compression ratios
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 08_vtt/04_video_eval \
#         --collect --wait --terminate --yes
#
# The harness itself validates offline, no GPU, no download:
#     uv run 08_vtt/04_video_eval/video_mme_eval.py --dry-run

#SBATCH --gres=gpu:1
# ONE GPU. Evaluation is embarrassingly parallel ACROSS questions, but each
# question is a short generate() call — the win comes from batching questions,
# not from sharding the model.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=03:00:00

#SBATCH --job-name=vtt_eval

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

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
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
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

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${DATASET:-}"

# ALWAYS establish the chance floor first. If the random baseline does not
# land near 25%, the harness is leaking answers and every number after it is
# meaningless. That exact bug shipped in this file once — correlated RNG seeds
# between question generation and guessing scored a random model at 100%.
echo ""
echo "--- chance baseline (must be near 25%) ---"
python video_mme_eval.py --dry-run --output eval_baseline.json

# Then sweep frame budgets. The number to watch is the TEMPORAL GAP: it is
# what widens when compression has thrown away the time axis, and it is
# invisible in the overall average.
for frames in 8 16 32 64; do
    echo ""
    echo "--- ${frames} frames ---"
    if [ -n "${DATASET}" ]; then
        python video_mme_eval.py \
            --model "${MODEL}" \
            --dataset "${DATASET}" \
            --max-frames "${frames}" \
            --output "eval_${frames}f.json" "$@"
    else
        python video_mme_eval.py \
            --model "${MODEL}" \
            --max-frames "${frames}" \
            --output "eval_${frames}f.json" "$@"
    fi
done

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
