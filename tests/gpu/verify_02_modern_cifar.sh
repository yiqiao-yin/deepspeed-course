#!/usr/bin/env bash
# =============================================================================
# Train the modern CIFAR-10 architectures and REPORT WHAT THEY REACH.
#
#     bash tests/gpu/verify_02_modern_cifar.sh [num_gpus] [epochs]
#
# REQUIRES A GPU. Downloads CIFAR-10 (~170 MB). Skips with exit 0 on CPU.
#
# Unlike the other harnesses in this directory, the point here is not only that
# the scripts run -- it is the NUMBER. CONTRIBUTING.md forbids publishing an
# accuracy the repository has not measured, so this is what measures it.
# =============================================================================
set -uo pipefail
GPUS="${1:-2}"
EPOCHS="${2:-30}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$HERE/02_basic_convnet_cifar10_examples"
LOG="${VERIFY_LOG:-/tmp/verify_02_modern.log}"
exec > >(tee "$LOG") 2>&1

PY_BIN="${PYTHON:-}"; [ -z "$PY_BIN" ] && { command -v python >/dev/null && PY_BIN=python || PY_BIN=python3; }
"$PY_BIN" -c "import torch" 2>/dev/null || { echo "[skip] no torch"; exit 0; }
"$PY_BIN" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[skip] no CUDA device"; exit 0; }
VISIBLE=$("$PY_BIN" -c "import torch;print(torch.cuda.device_count())")

echo "=============================================================================="
"$PY_BIN" -c "
import torch
print(f'  torch {torch.__version__}  devices={torch.cuda.device_count()}')
[print(f'    [{i}] {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "  requested $GPUS GPUs, $EPOCHS epochs"
echo "=============================================================================="
cd "$DIR" || exit 1

PASS=0; FAIL=0
declare -a RESULTS=()

# Report progress as it happens. The pod's container can restart mid-run -- if
# it does, everything written to the log is lost and the only evidence left is
# what was already sent. NTFY_TOPIC is set by the RunPod driver; locally this
# is a no-op.
say() {
    echo ">>> $*"
    [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 15 -d "$*" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
    return 0
}

# The catalogue must work with no GPU work at all.
say "starting: $GPUS GPUs, $EPOCHS epochs, 3 models"
if timeout 300 "$PY_BIN" train_modern_cifar10.py --list-models >/dev/null 2>&1; then
    RESULTS+=("PASS  --list-models"); PASS=$((PASS+1))
else RESULTS+=("FAIL  --list-models"); FAIL=$((FAIL+1)); fi

# A capped run proves the pipeline before any real training is paid for.
if timeout 900 deepspeed --num_gpus="$GPUS" train_modern_cifar10.py \
      --model resnet9 --max-steps 3 --epochs 1 >/dev/null 2>&1; then
    RESULTS+=("PASS  dry run (--max-steps 3)"); PASS=$((PASS+1))
else RESULTS+=("FAIL  dry run"); FAIL=$((FAIL+1)); fi

say "dry run done (pass=$PASS fail=$FAIL)"

for m in resnet9 cifarnet wrn_16_8; do
    echo ""
    echo "########## $m ##########"
    say "training $m ..."
    t0=$(date +%s)
    timeout 5400 deepspeed --num_gpus="$GPUS" train_modern_cifar10.py \
        --model "$m" --epochs "$EPOCHS" 2>&1 | tail -45
    rc=${PIPESTATUS[0]}; dt=$(( $(date +%s) - t0 ))
    acc=$(grep -E "^  FINAL" "$LOG" | tail -1)
    if [ $rc -eq 0 ]; then
        RESULTS+=("PASS  $m trained (${dt}s)"); PASS=$((PASS+1))
        say "$m OK ${dt}s | ${acc:-no FINAL line}"
    else
        RESULTS+=("FAIL  $m rc=$rc (${dt}s)"); FAIL=$((FAIL+1))
        say "$m FAILED rc=$rc after ${dt}s | $(tail -3 "$LOG" | tr '\n' ' ' | head -c 300)"
    fi
done

echo ""
echo "=============================================================================="
echo "  SUMMARY  pass=$PASS fail=$FAIL"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "  --- measured accuracy (grep FINAL above for the authoritative lines) ---"
grep -E "FINAL|with mirror TTA" "$LOG" | sed 's/^/  /' || true
echo "=============================================================================="
[ "$FAIL" -eq 0 ]
