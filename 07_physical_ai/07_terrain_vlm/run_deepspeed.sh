#!/bin/bash
#SBATCH --job-name=terrain_vlm
#SBATCH --partition=h200-low
#SBATCH --gres=gpu:8                   # the full fine-tune needs all eight
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/terrain_vlm_%j.out
#SBATCH --error=logs/terrain_vlm_%j.err

# ---------------------------------------------------------------------------
# Lab 7: can a model that SEES Martian terrain beat one that walks into it?
#
# THIS IS THE FIRST LAB IN 07_physical_ai THAT ACTUALLY USES DEEPSPEED,
# and the reason is measured rather than assumed. `verify_arch.py`:
#
#       7B LoRA r=16        15.3 GB on ONE card   <- fits already
#       7B full fine-tune  119.7 GB on one card   <- does not
#                           15.0 GB across eight  <- ZeRO-3 earns its place
#
# Labs 1-6 here carry `launcher="python"` because their models are
# 10k-200k parameter policies with nothing to shard, and lab 6 went
# further: it measured GPU physics at 1.50x a CPU worker pool against
# a 5x bar and recommended NOT renting. Using ZeRO-3 on the LoRA arm
# would be the same cargo cult in the other direction, so the LoRA arm
# below runs plain python on one GPU and only the full fine-tune is
# launched with `deepspeed`.
#
# --gres=gpu:8 is sized for the full arm. The cheap arms need one GPU;
# see the dry-run section.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Credentials stay COMMENTED and QUOTED. An uncommented
# `export WANDB_API_KEY=<KEY>` is a bash syntax error -- `<` is a
# redirection operator -- so the script would abort on that line and
# never reach the training command.
# export WANDB_API_KEY="your_key_here"

echo "=============================================================="
echo "  Terrain VLM — does seeing beat walking into it?"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
nproc && free -g | head -2

cd "$(dirname "$0")"
uv sync --frozen

# ---------------------------------------------------------------------------
# 1. The things that need no GPU and no download. Cheap, and the ones
#    worth failing on: a bad label makes every number downstream
#    meaningless.
# ---------------------------------------------------------------------------
uv run make_dataset.py --check
uv run verify_arch.py

# ---------------------------------------------------------------------------
# 2. The dataset. Generated, not downloaded -- a few minutes of CPU.
# ---------------------------------------------------------------------------
if [ ! -f data/train.jsonl ]; then
    uv run make_dataset.py --seeds 400
fi

# ---------------------------------------------------------------------------
# 3. The cheap arms, on ONE GPU and without DeepSpeed, because LoRA on
#    a 3B model has nothing to shard. The control runs FIRST: if the
#    blind arm scores as well as the sighted one, the rest of the
#    lab's numbers do not mean what they appear to.
# ---------------------------------------------------------------------------
DRY="${DRY_RUN:-0}"
CAP=""
if [ "$DRY" = "1" ]; then
    CAP="--max-steps 3 --limit 32"
    echo "  DRY RUN — capped. These are not results."
fi

uv run python train_vlm.py --lora --blind --name blind  $CAP "$@"
uv run python train_vlm.py --lora          --name lora3b $CAP "$@"

uv run evaluate.py --zero-shot            ${CAP:+--limit 32}
uv run evaluate.py --run blind  --blind   ${CAP:+--limit 32}
uv run evaluate.py --run lora3b           ${CAP:+--limit 32}

# ---------------------------------------------------------------------------
# 4. The full fine-tune. THIS is the DeepSpeed part: 119.7 GB of
#    weights and optimizer state that only fits when ZeRO-3 shards it
#    across eight cards.
#
#    Skipped on a dry run and on anything smaller than 8 GPUs, rather
#    than failed -- a reader on one card should still get everything
#    above.
# ---------------------------------------------------------------------------
GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$DRY" != "1" ] && [ "${GPUS:-0}" -ge 8 ]; then
    uv run deepspeed --num_gpus=8 train_vlm.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --deepspeed ds_config.json --name full7b --batch 1 --grad-accum 4 "$@"
    uv run evaluate.py --run full7b --model Qwen/Qwen2.5-VL-7B-Instruct
else
    echo "  full fine-tune skipped (${GPUS:-0} GPUs, needs 8). The LoRA"
    echo "  and control arms above are complete and comparable."
fi

echo "  done."
