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
DIR="$HERE/01_basics/03_convnet_cifar10"
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

# --- fetch CIFAR-10 exactly ONCE ---------------------------------------------
# The canonical host (cs.toronto.edu) served a rented pod at ~78 kB/s, so the
# 170 MB archive took over half an hour -- longer than any single training step
# was allowed, so every model died mid-download and the run produced nothing.
#
# Two fixes, both necessary. Download before the loop rather than inside each
# run, and put it on /workspace, which is the pod's VOLUME and survives a
# container restart; the default ./data lives on the container filesystem and
# is wiped, so a restart re-downloaded from zero.
DATA_DIR="${CIFAR_DATA_DIR:-/workspace/cifar10-data}"
mkdir -p "$DATA_DIR"
if [ -d "$DATA_DIR/cifar-10-batches-py" ]; then
    say "CIFAR-10 already present in $DATA_DIR"
else
    for attempt in 1 2 3; do
        say "downloading CIFAR-10 (attempt $attempt) — this host may be slow"
        if timeout 3000 "$PY_BIN" - "$DATA_DIR" <<'PYEOF'
import sys, torchvision
root = sys.argv[1]
torchvision.datasets.CIFAR10(root=root, train=True, download=True)
torchvision.datasets.CIFAR10(root=root, train=False, download=True)
print("download complete")
PYEOF
        then break; fi
        say "attempt $attempt failed"
    done
fi
if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    say "ABORT: could not fetch CIFAR-10 after 3 attempts"
    echo "  Could not download CIFAR-10; nothing below would be meaningful."
    exit 1
fi
say "CIFAR-10 ready"
if timeout 300 "$PY_BIN" train_modern_cifar10.py --list-models >/dev/null 2>&1; then
    RESULTS+=("PASS  --list-models"); PASS=$((PASS+1))
else RESULTS+=("FAIL  --list-models"); FAIL=$((FAIL+1)); fi

# A capped run proves the pipeline before any real training is paid for.
if timeout 900 deepspeed --num_gpus="$GPUS" train_modern_cifar10.py \
      --model resnet9 --max-steps 3 --epochs 1 --data-dir "$DATA_DIR" >/dev/null 2>&1; then
    RESULTS+=("PASS  dry run (--max-steps 3)"); PASS=$((PASS+1))
else RESULTS+=("FAIL  dry run"); FAIL=$((FAIL+1)); fi

say "dry run done (pass=$PASS fail=$FAIL)"

for m in resnet9 cifarnet wrn_16_8; do
    echo ""
    echo "########## $m ##########"
    say "training $m ..."
    t0=$(date +%s)
    timeout 5400 deepspeed --num_gpus="$GPUS" train_modern_cifar10.py \
        --model "$m" --epochs "$EPOCHS" --data-dir "$DATA_DIR" 2>&1 | tail -45
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
