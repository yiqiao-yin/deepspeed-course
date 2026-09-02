#!/bin/bash
# SLURM batch script — Learning to Rank
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 02_intermediate/03_learning_to_rank \
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:2
# TWO GPUs, and not because the model needs them -- it is a few thousand
# parameters and fits in a phone. Two ranks is the smallest number that
# exercises the thing this example is about: ranking data is sharded by QUERY,
# never by document, because a pairwise loss or a listwise softmax split across
# devices is a different computation. One rank would never test that.
# Drop to 1 (NUM_GPUS=1 sbatch run_deepspeed.sh) and everything still runs.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=00:30:00
# Wall-clock ceiling; the job is killed at this point. Generous: the full
# --model/--method sweep on synthetic data finishes in a few minutes.

#SBATCH --job-name=learning_to_rank

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=4
# Cores per rank. There is no dataloader here -- the tensors are generated
# once and live on the GPU -- so this only needs to cover the process itself.

#SBATCH --mem=16G
# Host RAM. The dataset is generated in-process with numpy and is a few
# hundred MB at the default sizes -- there is nothing large to hold.

#SBATCH --output=logs/learning_to_rank_%j.out
#SBATCH --error=logs/learning_to_rank_%j.err

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

NUM_GPUS="${NUM_GPUS:-2}"

deepspeed --num_gpus="${NUM_GPUS}" train_learning_to_rank.py \
    --deepspeed ds_config.json \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
