#!/bin/bash
# =============================================================================
# SLURM batch script — GLM-5.3 LoRA fine-tuning
#
# CoreWeave / any SLURM cluster:
#     sbatch run_glm53.sh --max-steps 20                  # cheap dry run first
#     sbatch run_glm53.sh                                 # the real thing
#     sbatch run_glm53.sh --model zai-org/glm-edge-1.5b-chat   # 1 GPU is enough
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run 03_huggingface/01_llm_finetuning \
#         --dry-run --collect --wait --terminate --yes
#
# READ THIS BEFORE SUBMITTING AT SCALE
# ------------------------------------
# GLM-5.3 is 755.7 GB of fp8 weights. LoRA does NOT reduce what it costs to
# HOLD them — it only removes optimizer state for the frozen base. All 755 GB
# must be resident across your ranks:
#
#     8 x A100 80GB  =   640 GB   NOT ENOUGH
#     8 x H100 80GB  =   640 GB   NOT ENOUGH
#     8 x H200 141GB = 1,128 GB   works
#     8 x B200 180GB = 1,440 GB   comfortable
#
# The script computes this itself and refuses before downloading. To see the
# arithmetic without a GPU or a download:
#
#     uv run train_glm53_ds.py --plan
# =============================================================================

#SBATCH --gres=gpu:8
# EIGHT, and this one is not negotiable for the default model. ZeRO-3 shards
# parameters across ranks, so the constraint is aggregate VRAM: 755 GB of
# weights over 8 x H200 is ~94 GB per rank, which fits with room for
# activations. Fewer ranks means more per rank, and at 4 it does not fit at
# all. Use --model zai-org/glm-edge-1.5b-chat with --gres=gpu:1 to exercise
# the same code path cheaply.

#SBATCH --partition=h200-low
# Must be an H200/B200-class partition for the default model. Check: sinfo

#SBATCH --time=04:00:00
# Wall-clock ceiling; the job is killed here. A 755 GB download alone can take
# over an hour, so do not trim this for the first run.

#SBATCH --job-name=glm53_lora

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=16
# Cores for the dataloader and — mostly — for the safetensors download and
# fp8 dequantisation, which are CPU-bound and painfully slow with too few.

#SBATCH --mem=256G
# Host RAM. ZeRO-3 stages parameters through CPU memory during load; at this
# model size a 64 GB node thrashes and can OOM the host before the GPUs are
# ever touched.

#SBATCH --output=logs/glm53_lora_%j.out
#SBATCH --error=logs/glm53_lora_%j.err

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

# $HOME is usually a small NFS quota, and a 755 GB download into it fails
# slowly and takes the quota with it. Point the cache at scratch — this is not
# optional at this model size.
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

NUM_GPUS="${NUM_GPUS:-8}"

# "$@" forwards sbatch's extra arguments to the training script, so
# `sbatch run_glm53.sh --max-steps 20` is a real dry run rather than a full job.
deepspeed --num_gpus="${NUM_GPUS}" train_glm53_ds.py \
    --deepspeed ds_config_glm53.json \
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
