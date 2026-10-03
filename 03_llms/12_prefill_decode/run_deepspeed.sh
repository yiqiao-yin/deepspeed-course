#!/bin/bash
#SBATCH --job-name=prefill_decode       # shows in squeue
#SBATCH --nodes=1                       # one node; nothing to distribute
#SBATCH --ntasks-per-node=1             # one process: this is single-GPU inference
#SBATCH --gres=gpu:1                    # one GPU is enough at 0.6B
#SBATCH --cpus-per-task=8               # dataloader-free, but tokenizer wants a few
#SBATCH --mem=32G                       # host RAM for the 1.2 GB download + env
#SBATCH --time=00:30:00                 # calibration is minutes, not hours
#SBATCH --partition=h200-low            # adjust to your cluster's partitions
#SBATCH --output=logs/prefill_decode_%j.out
#SBATCH --error=logs/prefill_decode_%j.err

# Prefill/decode scheduling measurement.
#
# NOTE: there is deliberately no `deepspeed` launcher here. This lab measures
# INFERENCE scheduling -- no optimizer, no gradients, one process. See the
# README section "Why there is no deepspeed launcher here".

mkdir -p logs

# Optional experiment tracking. Keep this COMMENTED and QUOTED: an
# uncommented `export WANDB_API_KEY=<KEY>` is a bash syntax error, because
# `<` redirects, and the script would abort before reaching the run.
# export WANDB_API_KEY="your_key_here"

set -euo pipefail

cd "$(dirname "$0")" || exit 1
uv sync

# Cheap dry run first: proves the model loads and the harness works before
# anything long starts.
uv run serve_bench.py --calibrate --repeats 3 --warmup 1

# The real measurement.
uv run serve_bench.py --calibrate --repeats 9 --warmup 4
uv run serve_bench.py --demo --prompt-len 4096 --chunk 256 --repeats 7
