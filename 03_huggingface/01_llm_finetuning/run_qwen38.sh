#!/bin/bash
# =============================================================================
# SLURM batch script — Qwen3.8-27B LoRA fine-tuning
#
# CoreWeave / any SLURM cluster:
#     sbatch run_qwen38.sh --max-steps 20                 # cheap dry run first
#     sbatch run_qwen38.sh                                # the real thing
#     sbatch run_qwen38.sh --lora-scope attention+mlp     # bigger adapter
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 03_huggingface/01_llm_finetuning \
#         --dry-run --collect --wait --terminate --yes
#
# SIZING
# ------
# Qwen3.8-27B is 55.6 GB of bf16 weights. LoRA does NOT reduce what it costs to
# HOLD them — it only removes optimizer state for the frozen base:
#
#     1 x 48GB (A6000/L40S) =  48 GB   NOT ENOUGH
#     2 x 48GB              =  96 GB   works (this script's default)
#     2 x 80GB (A100/H100)  = 160 GB   comfortable
#
# The script computes this itself and refuses before downloading. To see the
# arithmetic without a GPU or a download:
#
#     uv run train_qwen38_ds.py --plan
#     uv run train_qwen38_ds.py --verify-arch
# =============================================================================

#SBATCH --gres=gpu:2
# TWO, and this is a memory floor rather than a preference. 55.6 GB of weights
# do not fit one 48 GB card, so ZeRO-3 must shard the parameters across at
# least two ranks (~28 GB each). On 80 GB cards one would technically hold the
# weights, but leaves little room for activations at this sequence length.

#SBATCH --partition=h200-low
# Any partition with >= 2 x 48 GB cards. Check what you have with: sinfo

#SBATCH --time=03:00:00
# Wall-clock ceiling; the job is killed here. The 55.6 GB download alone can
# take 20+ minutes on a cold cache, so do not trim this for a first run.

#SBATCH --job-name=qwen38_lora

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# Cores for the dataloader and, mostly, for the safetensors download and shard
# loading, which are CPU-bound and painfully slow with too few.

#SBATCH --mem=128G
# Host RAM. ZeRO-3 stages parameters through CPU memory during load; at 55.6 GB
# a 32 GB node thrashes and can OOM the host before the GPUs are ever touched.

#SBATCH --output=logs/qwen38_lora_%j.out
#SBATCH --error=logs/qwen38_lora_%j.err

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
#   uv pip install deepspeed transformers datasets trl peft accelerate
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

# $HOME is usually a small NFS quota and a 55.6 GB download into it fails
# slowly, taking the quota with it. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Credentials, if your cluster needs them. KEEP THESE COMMENTED AND QUOTED.
# An uncommented `export HF_TOKEN=<ENTER_KEY_HERE>` is a bash SYNTAX ERROR —
# `<` is a redirection operator — so the script aborts on that line and never
# reaches the training command. Seven scripts shipped that way once and could
# never run. tests/test_runpod_ctl.py runs `bash -n` over every shell script
# to stop it recurring.
# export HF_TOKEN="your_value_here"
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-2}"

# "$@" forwards sbatch's extra arguments to the training script, so
# `sbatch run_qwen38.sh --max-steps 20` is a real dry run rather than a full job.
deepspeed --num_gpus="${NUM_GPUS}" train_qwen38_ds.py \
    --deepspeed ds_config_qwen38.json \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
