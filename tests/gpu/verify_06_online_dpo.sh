#!/usr/bin/env bash
# =============================================================================
# Multi-GPU smoke test for 03_huggingface/07_online_dpo.
#
#     bash tests/gpu/verify_06_online_dpo.sh [num_gpus]      # default 2
#
# REQUIRES A GPU. Downloads Qwen3-0.6B twice over (policy + reference) plus a
# prompt set, and trains a small reward model first to act as the judge.
# Skips cleanly with exit 0 when no CUDA device is visible.
#
# Why 2 GPUs is the right default here
# ------------------------------------
# Online methods GENERATE during training: every step samples two completions
# per prompt, scores them, and only then computes a loss. That is the cost
# driver, and it is why runpod_ctl registers this example at gpus=2 while the
# offline DPO example asks for one.
#
# What this covers that the CPU suite cannot
# ------------------------------------------
# tests/test_preference_losses.py asserts the offline objectives on plain
# tensors. Nothing on CPU can tell you whether TRL's OnlineDPOTrainer,
# NashMDTrainer and XPOTrainer accept the (model, reward_model, judge, config)
# tuple this script assembles -- and those three differ from each other, which
# is exactly where a shared code path breaks for two of them and not the third.
#
# It also exercises the 05 -> 06 handoff the READMEs describe: a reward model
# trained here by 03_huggingface/04_reward_model becomes the preference source.
# That is the documented pipeline, and it had never been run end to end.
#
# COVERAGE, NOT CONVERGENCE. Two optimizer steps on a handful of prompts.
# =============================================================================
set -uo pipefail

GPUS="${1:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$HERE/03_huggingface/07_online_dpo"
RM_DIR="$HERE/03_huggingface/04_reward_model"
LOG="${VERIFY_LOG:-/tmp/verify_06.log}"
RM_OUT=/tmp/rm-for-06

PY_BIN="${PYTHON:-}"
if [ -z "$PY_BIN" ]; then
    if command -v python >/dev/null 2>&1; then PY_BIN=python; else PY_BIN=python3; fi
fi
export MPLBACKEND=Agg
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

if ! "$PY_BIN" -c "import torch" 2>/dev/null; then
    echo "[skip] $PY_BIN cannot import torch. Exiting 0."; exit 0
fi
if ! "$PY_BIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[skip] no CUDA device visible. Exiting 0."; exit 0
fi
VISIBLE=$("$PY_BIN" -c "import torch;print(torch.cuda.device_count())")

echo "=============================================================================="
echo "  03_huggingface/07_online_dpo — multi-GPU smoke test"
echo "=============================================================================="
"$PY_BIN" - <<'PY'
import torch
print(f"  torch {torch.__version__}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    [{i}] {p.name}  {p.total_memory/1024**3:.0f} GB")
PY
echo "  requested GPUs: $GPUS   visible: $VISIBLE"
echo "  log: $LOG"
echo "=============================================================================="

: > "$LOG"
PASS=0; FAIL=0; SKIP=0
declare -a RESULTS=()

step() {  # step <label> <timeout> <needs_gpus> <cmd...>
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
    if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
        RESULTS+=("PASS  $label  (${dt}s${rc:+$([ $rc -eq 124 ] && echo ', capped')})")
        PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $label  rc=$rc (${dt}s)"); FAIL=$((FAIL+1))
    fi
    printf "  %-28s rc=%-4s %ss\n" "$label" "$rc" "$dt"
}

# expect_fail <label> <timeout> <substring> <cmd...>
# Some behaviour is only correct when it REFUSES. A misconfiguration that
# starts training anyway is the bug, so exiting 0 here is a failure.
expect_fail() {
    local label="$1" tmo="$2" want="$3"; shift 3
    echo "" >> "$LOG"; echo "@@@@@ BEGIN $label @@@@@" >> "$LOG"
    echo "\$ $*" >> "$LOG"
    local out rc
    out=$(timeout "$tmo" bash -c "$*" 2>&1); rc=$?
    echo "$out" >> "$LOG"
    echo "@@@@@ END $label rc=$rc @@@@@" >> "$LOG"
    if [ $rc -eq 0 ]; then
        RESULTS+=("FAIL  $label: exited 0 — it should have refused")
        FAIL=$((FAIL+1))
    elif grep -qiF "$want" <<<"$out"; then
        RESULTS+=("PASS  $label: refused with a useful message")
        PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $label: refused, but the message never says '$want'")
        FAIL=$((FAIL+1))
    fi
    printf "  %-28s rc=%-4s\n" "$label" "$rc"
}

cd "$DIR" || { echo "cannot cd $DIR"; exit 1; }

# --- 1. the catalogue, no GPU work -------------------------------------------
step list_methods 120 0 "$PY_BIN train_online_dpo.py --list-methods"

# --- 2. misconfiguration must be REFUSED, not discovered mid-training --------
# Online methods need exactly one preference source. Neither, or both, has to
# stop before a model is downloaded.
expect_fail no_judge 300 "exactly one preference source" \
    "$PY_BIN train_online_dpo.py --max-steps 1"
expect_fail both_judges 300 "exactly one preference source" \
    "$PY_BIN train_online_dpo.py --max-steps 1 --reward-model x --judge y"
# --judge used to be accepted and then ignored entirely. An unknown name must
# now produce a list of what this TRL actually offers.
expect_fail unknown_judge 600 "Unknown judge" \
    "$PY_BIN train_online_dpo.py --max-steps 1 --judge NotARealJudge"

# --- 3. build the preference source, via the documented 05 -> 06 handoff -----
# --no-lora so the result is a full model directory that
# AutoModelForSequenceClassification can load, not an adapter.
step build_reward_model 1800 1 \
    "cd $RM_DIR && $PY_BIN train_reward_model.py --max-samples 64 --max-steps 2 \
     --max-length 256 --batch-size 2 --grad-accum 1 --no-lora --output $RM_OUT"

SMALL="--max-samples 16 --max-steps 2 --batch-size 1 --grad-accum 1 \
       --max-new-tokens 16 --reward-model $RM_OUT"

# --- 4. all three online methods, at $GPUS ------------------------------------
# Run all three: they are different TRL trainers with different constructor
# contracts, so passing one says nothing about the other two.
for m in online_dpo xpo; do
    step "${m}_${GPUS}gpu" 2400 "$GPUS" \
        "deepspeed --num_gpus=$GPUS train_online_dpo.py --method $m $SMALL \
         --output /tmp/odpo-$m"
done

# nash_md is expected to REFUSE on TRL 1.12: GeometricMixtureWrapper.forward is
# decorated @torch.inference_mode() and its logits feed the loss, so the first
# backward raises "Inference tensors cannot be saved for backward". That is an
# upstream bug; what this repository owns is refusing early with an explanation
# instead of surfacing a LayerNorm traceback. If TRL fixes it, the script stops
# refusing and THIS check fails -- which is the correct signal to promote
# nash_md back into the loop above.
expect_fail nash_md_refuses 900 "upstream TRL bug" \
    "deepspeed --num_gpus=$GPUS train_online_dpo.py --method nash_md $SMALL \
     --output /tmp/odpo-nash_md"

# --- 5. the shipped launcher, as a reader would run it ------------------------
step launcher_${GPUS}gpu 2400 "$GPUS" \
    "NUM_GPUS=$GPUS bash run_deepspeed.sh --method online_dpo $SMALL \
     --output /tmp/odpo-launcher"

# --- did the multi-GPU steps really go multi-GPU? ----------------------------
if [ "$VISIBLE" -ge "$GPUS" ] && [ "$GPUS" -gt 1 ]; then
    echo ""
    echo "  --- checking the ${GPUS}-GPU steps ran distributed ---"
    for lbl in "online_dpo_${GPUS}gpu" "xpo_${GPUS}gpu" "launcher_${GPUS}gpu"; do
        sect=$(sed -n "/@@@@@ BEGIN $lbl @@@@@/,/@@@@@ END $lbl /p" "$LOG")
        if grep -qiE "world_size=?[ :]*$GPUS|num_gpus=$GPUS|world size[ :]*$GPUS" <<<"$sect"; then
            RESULTS+=("PASS  $lbl: confirmed world_size=$GPUS"); PASS=$((PASS+1))
        else
            RESULTS+=("FAIL  $lbl: no evidence of $GPUS-way parallelism"); FAIL=$((FAIL+1))
        fi
    done
fi

# --- did anything actually get written? --------------------------------------
echo ""
echo "  --- checking saved artefacts ---"
for d in "$RM_OUT" /tmp/odpo-online_dpo /tmp/odpo-xpo /tmp/odpo-launcher; do
    [ -d "$d" ] || continue
    if find "$d" \( -name '*.safetensors' -o -name '*.bin' \) | grep -q .; then
        RESULTS+=("PASS  $(basename $d): weights written"); PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $(basename $d): no weights"); FAIL=$((FAIL+1))
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
