#!/bin/bash
# SLURM launcher — ZeRO-3 BF16 vs NF4 parameter partitioning
#
# Submit on a SLURM cluster:
#   sbatch run_deepspeed.sh bf16
#   sbatch run_deepspeed.sh nf4
#
# Run directly on a Linux machine with two GPUs:
#   bash run_deepspeed.sh bf16
#   bash run_deepspeed.sh nf4

#SBATCH --gres=gpu:2
#SBATCH --partition=h200-low
#SBATCH --time=02:00:00
#SBATCH --job-name=zero3_quantized
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=zero3_%j.out
#SBATCH --error=zero3_%j.err

set -euo pipefail
cd "$(dirname "$0")"

ARM="${1:-bf16}"

if [[ "$ARM" != "bf16" && "$ARM" != "nf4" ]]; then
    echo "Usage: bash run_deepspeed.sh [bf16|nf4]"
    exit 1
fi

mkdir -p logs results

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export NUM_GPUS=2

echo "=========================================="
echo "ZeRO-3 parameter partitioning experiment"
echo "Arm:  ${ARM}"
echo "GPUs: ${NUM_GPUS}"
echo "Date: $(date)"
echo "=========================================="

nvidia-smi

uv run --locked deepspeed --num_gpus="${NUM_GPUS}" train_zero3.py \
    --arm "${ARM}" \
    --output-dir "results/${ARM}"

echo "Experiment complete: ${ARM}"