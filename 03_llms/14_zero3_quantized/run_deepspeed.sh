#!/bin/bash
# SLURM launcher — ZeRO-3 BF16 vs NF4 parameter partitioning
#
# Submit from this lab directory on a SLURM cluster:
#   sbatch run_deepspeed.sh bf16
#   sbatch run_deepspeed.sh nf4
#
# Run directly on a Linux machine with two GPUs:
#   ./run_deepspeed.sh bf16
#   ./run_deepspeed.sh nf4
#
# Validate the config on a login node or CPU machine without submitting a job:
#   ./run_deepspeed.sh bf16 --plan

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

ARM=
PLAN=false
for arg in "$@"; do
    case "$arg" in
        bf16|nf4)
            if [[ -n "$ARM" ]]; then
                echo "Choose one arm: bf16 or nf4." >&2
                exit 2
            fi
            ARM="$arg"
            ;;
        --plan)
            if [[ "$PLAN" == true ]]; then
                echo "Specify --plan only once." >&2
                exit 2
            fi
            PLAN=true
            ;;
        *)
            printf 'Unsupported argument: %s\nUsage: %s [bf16|nf4] [--plan]\n' "$arg" "$0" >&2
            exit 2
            ;;
    esac
done
ARM="${ARM:-bf16}"

# SLURM may run a spooled copy of this script. Submit from the lab directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
if [[ ! -f train_zero3.py ]]; then
    echo "Launch from the ZeRO-3 lab directory containing train_zero3.py." >&2
    exit 2
fi

if [[ "$PLAN" == true ]]; then
    exec python3 train_zero3.py --arm "$ARM" --plan
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
