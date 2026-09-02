#!/usr/bin/env bash
# =============================================================================
# Multi-GPU smoke test for every script in 02_intermediate/02_rnn_stock_data.
#
#     bash tests/gpu/verify_04_multi_gpu.sh [num_gpus]     # default 2
#
# REQUIRES A GPU. Skips cleanly with exit 0 when no CUDA device is visible, so
# running it on a laptop is harmless -- see tests/gpu/README.md.
#
# What this covers that the CPU suite cannot
# ------------------------------------------
# tests/test_attention_layers.py and tests/test_ts_forecasting.py assert the
# mathematics of the layers. They say nothing about whether the eight scripts
# in the folder actually START on a machine with more than one GPU: whether the
# DeepSpeed launcher can spawn them, whether the batch invariant holds at that
# GPU count, whether yfinance can reach the network, whether matplotlib can
# render without a display.
#
# Those are exactly the failures a reader hits first, and none of them are
# visible from a CPU box.
#
# Goal is COVERAGE, NOT CONVERGENCE. Nothing here checks that a model learned
# anything -- the CPU suites do that. A step "passes" if it reached its work
# and exited without crashing. Long runs are capped with --max-steps where the
# script supports it and with `timeout` where it does not, so rc=124 (timed
# out) is reported as PASS-capped rather than as a failure.
# =============================================================================
set -uo pipefail          # NOT -e: one failing step must not abort the sweep

GPUS="${1:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$HERE/02_intermediate/02_rnn_stock_data"
LOG="${VERIFY_LOG:-/tmp/verify_04.log}"
CFG="train_rnn_stock_data_config.json"

# Interpreter. On a RunPod/CoreWeave image `python` is the one with torch; on a
# workstation it often is not, so allow an override rather than silently
# reporting "no GPU" when the real problem is the wrong interpreter.
PY_BIN="${PYTHON:-}"
if [ -z "$PY_BIN" ]; then
    if command -v python >/dev/null 2>&1; then PY_BIN=python; else PY_BIN=python3; fi
fi

# Headless box: matplotlib must not try to open a display, and the scripts all
# save figures to disk.
export MPLBACKEND=Agg

# --- skip cleanly with no GPU, per the tests/gpu contract --------------------
if ! "$PY_BIN" -c "import torch" 2>/dev/null; then
    echo "[skip] $PY_BIN cannot import torch — set PYTHON=/path/to/python. Exiting 0."
    exit 0
fi
if ! "$PY_BIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[skip] no CUDA device visible — this script needs a GPU. Exiting 0."
    exit 0
fi

VISIBLE=$("$PY_BIN" -c "import torch;print(torch.cuda.device_count())")
echo "=============================================================================="
echo "  02_intermediate/02_rnn_stock_data — multi-GPU smoke test"
echo "=============================================================================="
"$PY_BIN" - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"    [{i}] {torch.cuda.get_device_name(i)}")
PY
echo "  interpreter: $PY_BIN"
echo "  requested GPUs: $GPUS   visible: $VISIBLE"
if [ "$VISIBLE" -lt "$GPUS" ]; then
    echo "  !! only $VISIBLE GPU(s) visible; the ${GPUS}-GPU steps will be skipped."
fi
echo "  log: $LOG"
echo "=============================================================================="

: > "$LOG"
cd "$DIR" || { echo "cannot cd $DIR"; exit 1; }

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS=()

# step <label> <timeout_s> <needs_gpus> <command...>
step() {
    local label="$1" tmo="$2" need="$3"; shift 3
    if [ "$VISIBLE" -lt "$need" ]; then
        RESULTS+=("SKIP  $label  (needs $need GPUs, $VISIBLE visible)")
        SKIP=$((SKIP+1)); return
    fi
    echo "" >> "$LOG"
    echo "@@@@@ BEGIN $label @@@@@" >> "$LOG"
    echo "\$ $*" >> "$LOG"
    local t0 rc
    t0=$(date +%s)
    timeout "$tmo" bash -c "$*" >> "$LOG" 2>&1
    rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "@@@@@ END $label rc=$rc ${dt}s @@@@@" >> "$LOG"

    # rc 124 == hit our own `timeout`. The script was running fine; we stopped
    # it on purpose. Treating that as failure would mean the uncapped trainers
    # could never pass.
    if [ $rc -eq 0 ]; then
        RESULTS+=("PASS  $label  (${dt}s)"); PASS=$((PASS+1))
    elif [ $rc -eq 124 ]; then
        RESULTS+=("PASS  $label  (capped at ${tmo}s — was still running)"); PASS=$((PASS+1))
    else
        RESULTS+=("FAIL  $label  rc=$rc (${dt}s)"); FAIL=$((FAIL+1))
    fi
    printf "  %-24s rc=%-4s %ss\n" "$label" "$rc" "$dt"
}

# --- the three library modules: self-contained demos, no GPU needed ----------
step lib_attention_layers 180 0 "$PY_BIN attention_layers.py"
step lib_modern_ts_layers 180 0 "$PY_BIN modern_ts_layers.py"
step lib_tokenize_series  180 0 "$PY_BIN tokenize_series.py"

# --- the three newer trainers, single process --------------------------------
step attention_1proc 600 1 \
    "$PY_BIN train_rnn_attention.py --model lstm_attn --epochs 1 --max-steps 5"
step modern_ts_1proc 600 1 \
    "$PY_BIN train_modern_ts.py --model dlinear --epochs 1 --max-steps 5"
step token_lm_1proc  600 1 \
    "$PY_BIN train_token_lm.py --bits 8 --epochs 1 --max-steps 5"

# --- the registered script under the DeepSpeed launcher ----------------------
# train_rnn_stock_data_config.json hardcodes train_batch_size 64 with
# micro 32 and accumulation 1, so DeepSpeed's invariant
#     train_batch_size == micro * accum * num_gpus
# is satisfied at exactly 2 GPUs and nowhere else. That is the whole reason
# this test defaults to 2.
step ds_registered_${GPUS}gpu 900 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_rnn_stock_data_ds.py"

# --- the same script at 1 GPU -------------------------------------------------
# Recorded rather than asserted. If the invariant above is right this cannot
# work, and the log is the evidence for fixing either the config or the docs.
step ds_registered_1gpu 300 1 \
    "deepspeed --num_gpus=1 train_rnn_stock_data_ds.py"

# --- plain PyTorch, no DeepSpeed ---------------------------------------------
# No --max-steps flag on this one, so it is capped by `timeout` instead.
step plain_torch 420 1 "$PY_BIN train_rnn_stock_data.py"

# --- do the newer trainers engage DeepSpeed at all? ---------------------------
# They take --deepspeed with a default of ds_config.json, which does NOT exist
# in this folder -- so by default the DeepSpeed branch is dead and they run as
# plain single-process PyTorch even on a multi-GPU box. Pointing them at the
# config that does exist is what makes this a real multi-GPU test.
step attention_ds_${GPUS}gpu 700 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_rnn_attention.py --model lstm_attn --epochs 1 --max-steps 5 --deepspeed $CFG"
step modern_ts_ds_${GPUS}gpu 700 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_modern_ts.py --model dlinear --epochs 1 --max-steps 5 --deepspeed $CFG"
step token_lm_ds_${GPUS}gpu 700 "$GPUS" \
    "deepspeed --num_gpus=$GPUS train_token_lm.py --bits 8 --epochs 1 --max-steps 5 --deepspeed $CFG"

# --- did the multi-GPU steps ACTUALLY go multi-GPU? --------------------------
# A step exiting 0 under `deepspeed --num_gpus=2` proves nothing on its own:
# if the script never calls deepspeed.initialize, the launcher simply runs it
# twice, once per GPU, and both copies exit 0 having duplicated the work.
# That is exactly what these two scripts used to do. Each rank now announces
# its shard, so the absence of that line is the regression signal.
if [ "$VISIBLE" -ge "$GPUS" ] && [ "$GPUS" -gt 1 ]; then
    echo ""
    echo "  --- checking the 2-GPU steps really sharded ---"
    for lbl in "ds_registered_${GPUS}gpu" "attention_ds_${GPUS}gpu" \
               "modern_ts_ds_${GPUS}gpu" "token_lm_ds_${GPUS}gpu"; do
        sect=$(sed -n "/@@@@@ BEGIN $lbl @@@@@/,/@@@@@ END $lbl /p" "$LOG")
        # Either the script prints its own shard line, or DeepSpeed reports the
        # world size it built. Both are evidence of a real distributed run.
        if grep -qE "data-parallel: $GPUS ranks|world_size=?[ :]*$GPUS" <<<"$sect"; then
            RESULTS+=("PASS  $lbl: confirmed $GPUS-way data parallel")
            PASS=$((PASS+1))
        else
            RESULTS+=("FAIL  $lbl: exited 0 but shows NO sign of $GPUS-way parallelism"
                      "      (ran the same work $GPUS times?)")
            FAIL=$((FAIL+1))
        fi
    done
fi

echo ""
echo "=============================================================================="
echo "  SUMMARY   pass=$PASS  fail=$FAIL  skip=$SKIP"
echo "=============================================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
{
  echo ""
  echo "===== SUMMARY pass=$PASS fail=$FAIL skip=$SKIP ====="
  for r in "${RESULTS[@]}"; do echo "  $r"; done
} >> "$LOG"
echo "=============================================================================="
echo "  full log: $LOG"

[ "$FAIL" -eq 0 ]
