#!/bin/bash
# SLURM batch script — DeepSeek MLA from Scratch
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 03_huggingface/10_deepseek_from_scratch \
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:1
# ONE is enough, and that is the honest number. The largest model here is ~1.6M
# parameters and the comparison is about the KV CACHE, which is an inference
# cost that data parallelism does not change. Two ranks would run the same
# experiment twice as fast without altering a single reported figure.
# NUM_GPUS=2 sbatch run_deepspeed.sh works if you want the throughput.

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=00:30:00
# Wall-clock ceiling. The job is killed at this point, so overestimate.

#SBATCH --job-name=deepseek_from_sc

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=4
# Cores for the data pipeline. Too few starves the dataloader and the GPU
# idles between batches — which looks like a slow model and is not.

#SBATCH --mem=16G

#SBATCH --output=logs/deepseek_from_sc_%j.out
#SBATCH --error=logs/deepseek_from_sc_%j.err

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

deepspeed --num_gpus="${NUM_GPUS}" train_deepseek_from_scratch.py \
    --deepspeed ds_config.json \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
