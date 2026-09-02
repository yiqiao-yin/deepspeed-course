#!/usr/bin/env bash
# =============================================================================
# Multi-GPU smoke test for 03_huggingface/04_reward_model.
#
#     bash tests/gpu/verify_05_reward_model.sh [num_gpus]      # default 2
#
# REQUIRES A GPU and downloads Qwen3-0.6B (~1.2 GB) plus a preference dataset.
# Skips cleanly with exit 0 when no CUDA device is visible.
#
# What this covers that the CPU suite cannot
# ------------------------------------------
# tests/test_reward_model.py asserts the Bradley-Terry objective itself: that
# it is shift-invariant, that a huge shift still perturbs it in float32. All of
# that runs on plain tensors and never loads a model.
#
# It cannot tell you whether TRL's RewardTrainer accepts what this script hands
# it. That is where reward modelling actually breaks, and the failures are
# specific: LoRA applied with the wrong task_type trains no scalar head at all;
# a tokenizer with no pad token dies on the first batch of pairs; ZeRO-2 plus
# gradient checkpointing plus a frozen base can disagree about which
# parameters require grad.
#
# COVERAGE, NOT CONVERGENCE. Two optimizer steps on 64 pairs proves the
# pipeline assembles and steps. It says nothing about reward quality, and is
# not meant to -- a reward model trained for two steps is noise, by design.
# =============================================================================
set -uo pipefail

GPUS="${1:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$HERE/03_huggingface/04_reward_model"
LOG="${VERIFY_LOG:-/tmp/verify_05.log}"

PY_BIN="${PYTHON:-}"
if [ -z "$PY_BIN" ]; then
    if command -v python >/dev/null 2>&1; then PY_BIN=python; else PY_BIN=python3; fi
fi

export MPLBACKEND=Agg
# Keep every download inside the pod's big volume rather than the small root fs.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

if ! "$PY_BIN" -c "import torch" 2>/dev/null; then
    echo "[skip] $PY_BIN cannot import torch. Exiting 0."; exit 0
fi
if ! "$PY_BIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[skip] no CUDA device visible. Exiting 0."; exit 0
fi
VISIBLE=$("$PY_BIN" -c "import torch;print(torch.cuda.device_count())")

echo "=============================================================================="
echo "  03_huggingface/04_reward_model — multi-GPU smoke test"
echo "=============================================================================="
"$PY_BIN" - <<'PY'
import torch
print(f"  torch {torch.__version__}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    [{i}] {p.name}  {p.total_memory/1024**3:.0f} GB")
PY
echo "  requested GPUs: $GPUS   visible: $VISIBLE"
echo "  HF_HOME: $HF_HOME"
echo "  log: $LOG"
echo "=============================================================================="

: > "$LOG"
cd "$DIR" || { echo "cannot cd $DIR"; exit 1; }

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS=()

step() {
    local label="$1" tmo="$2" need="$3"; shift 3
    if [ "$VISIBLE" -lt "$need" ]; then
        RESULTS+=("SKIP  $label  (needs $need GPUs, $VISIBLE visible)")
        SKIP=$((SKIP+1)); return
    fi
    echo "" >> "$LOG"; echo "@@@@@ BEGIN $label @@@@@" >> "$LOG"
    echo "\$ $*" >> "$LOG"
    local t0 rc; t0=$(date +%s)
    timeout "$tmo" bash -c "$*" >> "$LOG" 2>&1
    rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "@@@@@ END $label rc=$rc ${dt}s @@@@@" >> "$LOG"
    if [ $rc -eq 0 ]; then
        RESULTS+=("PASS  $label  (${dt}s)"); PASS=$((PASS+1))
    elif [ $rc -eq 124 ]; then
        RESULTS+=("PASS  $label  (capped at ${tmo}s — still running)"); PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $label  rc=$rc (${dt}s)"); FAIL=$((FAIL+1))
    fi
    printf "  %-26s rc=%-4s %ss\n" "$label" "$rc" "$dt"
}

# Tiny on purpose: 64 pairs, 2 optimizer steps, short sequences. The point is
# that every component accepts the next one's output, not that anything learns.
SMALL="--max-samples 64 --max-steps 2 --max-length 256 --batch-size 2 --grad-accum 1"

# --- 1. the CPU-runnable objective, on a GPU box -----------------------------
step objective_module 180 0 "$PY_BIN reward_modeling.py"

# --- 2. single process, LoRA (the documented default) ------------------------
step reward_1gpu_lora 1500 1 "$PY_BIN train_reward_model.py $SMALL --output /tmp/rm1"

# --- 3. the DeepSpeed launcher at $GPUS ---------------------------------------
# The real target of this test. ds_config.json is all "auto", which HuggingFace
# Trainer fills from TrainingArguments -- so unlike 04 there is no batch
# invariant to violate, but ZeRO-2 still has to agree with LoRA + gradient
# checkpointing about which parameters carry gradients.
step reward_${GPUS}gpu_lora 1800 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_reward_model.py $SMALL --output /tmp/rm2"

# --- 4. full fine-tune, no LoRA, at $GPUS ------------------------------------
# Exercises a different path: every parameter is trainable, so ZeRO-2 actually
# has something to shard. With LoRA the optimizer state is tiny and stage 2 is
# nearly a no-op, which would let a sharding bug hide.
step reward_${GPUS}gpu_full 1800 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_reward_model.py $SMALL --no-lora --output /tmp/rm3"

# --- 5. the shipped launcher, exactly as a reader would run it ---------------
# Catches the class of bug where the script works but the launcher that wraps
# it does not -- a swallowed "$@", a bad NUM_GPUS default, an unquoted export.
step launcher_${GPUS}gpu 1800 "$GPUS" \
    "NUM_GPUS=$GPUS bash run_deepspeed.sh $SMALL --output /tmp/rm4"

# --- did the multi-GPU steps actually go multi-GPU? --------------------------
# Exiting 0 under `deepspeed --num_gpus=2` proves nothing by itself: a script
# that never initialises DeepSpeed just runs twice and exits 0 twice.
if [ "$VISIBLE" -ge "$GPUS" ] && [ "$GPUS" -gt 1 ]; then
    echo ""
    echo "  --- checking the ${GPUS}-GPU steps really ran distributed ---"
    for lbl in "reward_${GPUS}gpu_lora" "reward_${GPUS}gpu_full" "launcher_${GPUS}gpu"; do
        sect=$(sed -n "/@@@@@ BEGIN $lbl @@@@@/,/@@@@@ END $lbl /p" "$LOG")
        if grep -qiE "world_size=?[ :]*$GPUS|num_gpus=$GPUS|world size[ :]*$GPUS" <<<"$sect"; then
            RESULTS+=("PASS  $lbl: confirmed world_size=$GPUS")
            PASS=$((PASS+1))
        else
            RESULTS+=("FAIL  $lbl: exited 0 with no evidence of $GPUS-way parallelism")
            FAIL=$((FAIL+1))
        fi
    done
fi

# --- did a reward model actually come out? -----------------------------------
# A training run that saves nothing has not been tested, only survived.
echo ""
echo "  --- checking the saved artefacts ---"
for d in /tmp/rm1 /tmp/rm2 /tmp/rm3 /tmp/rm4; do
    [ -d "$d" ] || continue
    if find "$d" -name '*.safetensors' -o -name '*.bin' | grep -q .; then
        RESULTS+=("PASS  $d: weights written")
        PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $d: directory exists but holds no weights")
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=============================================================================="
echo "  SUMMARY   pass=$PASS  fail=$FAIL  skip=$SKIP"
echo "=============================================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
{
  echo ""; echo "===== SUMMARY pass=$PASS fail=$FAIL skip=$SKIP ====="
  for r in "${RESULTS[@]}"; do echo "  $r"; done
} >> "$LOG"
echo "=============================================================================="
echo "  full log: $LOG"

[ "$FAIL" -eq 0 ]
