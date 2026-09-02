#!/bin/bash
# SLURM batch script — Offline preference optimization (DPO / IPO / CPO / ORPO / SimPO / KTO)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 03_huggingface/05_dpo \
#         --dry-run --collect --wait --terminate --yes
#
# No GPU? the OBJECTIVES run on CPU, and they are the part worth learning:
#     uv run 03_huggingface/05_dpo/preference_losses.py
#     uv run tests/test_preference_losses.py     # 58 checks, no download

#SBATCH --gres=gpu:1
# ONE GPU. Qwen3-0.6B + LoRA fits comfortably; raise to 2 for a 7B.
# Offline PO does no generation, so it is far cheaper than GRPO.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=04:00:00

#SBATCH --job-name=dpo_train

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --output=logs/dpo_train_%j.out
#SBATCH --error=logs/dpo_train_%j.err

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
#   uv pip install deepspeed transformers trl peft accelerate datasets
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
METHOD="${METHOD:-dpo}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# Sweep the family on the same data to see the differences directly. The number
# to compare is rewards/margins, NOT loss — the losses are on different scales
# and IPO is a squared error while the rest are log-sigmoid.
for method in ${METHODS:-$METHOD}; do
    echo ""
    echo "--- ${method} ---"
    deepspeed --num_gpus="${NUM_GPUS}" train_dpo.py \
        --deepspeed ds_config.json \
        --method "${method}" \
        --model "${MODEL}" \
        --output "./${method}-out" \
        "$@"
done

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
