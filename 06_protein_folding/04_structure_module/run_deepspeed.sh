#!/bin/bash
# SLURM batch script — Structure module (IPA vs diffusion)
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 06_protein_folding/04_structure_module \
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:1
# TODO(contributor): justify this number. Why 1 and not 1?

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=02:00:00
# Wall-clock ceiling. The job is killed at this point, so overestimate.

#SBATCH --job-name=structure_module

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# Cores for the data pipeline. Too few starves the dataloader and the GPU
# idles between batches — which looks like a slow model and is not.

#SBATCH --mem=48G

#SBATCH --output=logs/structure_module_%j.out
#SBATCH --error=logs/structure_module_%j.err

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
#   uv pip install deepspeed
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

# $HOME is usually a small NFS quota and a multi-GB model download into it
# fails slowly. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Credentials, if your example needs them. KEEP THESE COMMENTED AND QUOTED.
# An uncommented `export HF_TOKEN=<ENTER_KEY_HERE>` is a bash SYNTAX ERROR —
# `<` is a redirection operator — so the script aborts on that line and never
# reaches the training command. Seven scripts shipped that way once and could
# never run. tests/test_runpod_ctl.py runs `bash -n` over every shell script
# to stop it recurring.
# export HF_TOKEN="your_value_here"
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-1}"

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY RUN: capping at 3 optimizer steps"
    EXTRA_ARGS+=(--max-steps 3 --train-chains 16 --eval-chains 8)
fi

deepspeed --num_gpus="${NUM_GPUS}" train_structure_ds.py \
    --deepspeed_config ds_config.json \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
