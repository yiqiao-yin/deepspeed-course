#!/usr/bin/env bash
# =============================================================================
# Run the modern OCR comparison and REPORT WHAT IT MEASURES.
#
#     bash tests/gpu/verify_05_ocr_models.sh [num_gpus] [pages]
#
# REQUIRES A GPU and downloads several models (~10 GB total). Skips on CPU.
#
# As with the CIFAR-10 harness, the point is the NUMBER: CONTRIBUTING.md
# forbids publishing an accuracy the repository has not measured, so this is
# what measures it. Each model is run separately and a failure in one is
# recorded rather than aborting the rest -- a comparison that stops at the
# first awkward processor contract is worth nothing.
# =============================================================================
set -uo pipefail
GPUS="${1:-1}"
PAGES="${2:-16}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$HERE/05_huggingface_ocr"
LOG="${VERIFY_LOG:-/tmp/verify_05_ocr.log}"
exec > >(tee "$LOG") 2>&1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PY_BIN="${PYTHON:-}"; [ -z "$PY_BIN" ] && { command -v python >/dev/null && PY_BIN=python || PY_BIN=python3; }
"$PY_BIN" -c "import torch" 2>/dev/null || { echo "[skip] no torch"; exit 0; }
"$PY_BIN" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[skip] no CUDA device"; exit 0; }

say() {
    echo ">>> $*"
    [ -n "${NTFY_TOPIC:-}" ] && curl -s -m 15 -d "$*" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
    return 0
}

echo "=============================================================================="
"$PY_BIN" -c "
import torch
print(f'  torch {torch.__version__}  devices={torch.cuda.device_count()}')
[print(f'    [{i}] {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "  pages: $PAGES"
echo "=============================================================================="
cd "$DIR" || exit 1

PASS=0; FAIL=0
declare -a RESULTS=()

say "OCR comparison starting: $PAGES pages"

if timeout 300 "$PY_BIN" run_modern_ocr.py --list-models >/dev/null 2>&1; then
    RESULTS+=("PASS  --list-models"); PASS=$((PASS+1))
else RESULTS+=("FAIL  --list-models"); FAIL=$((FAIL+1)); fi

# Each model separately: one bad processor contract must not lose the others.
for m in got-ocr2 florence-2-base qwen2-vl-2b qwen2.5-vl-3b deepseek-ocr; do
    say "running $m ..."
    t0=$(date +%s)
    timeout 3600 "$PY_BIN" run_modern_ocr.py --models "$m" --max-samples "$PAGES" 2>&1 | tail -30
    rc=${PIPESTATUS[0]}; dt=$(( $(date +%s) - t0 ))
    line=$(grep -E "^  $m " "$LOG" | tail -1)
    if [ $rc -eq 0 ]; then
        RESULTS+=("PASS  $m (${dt}s)"); PASS=$((PASS+1))
        say "$m done ${dt}s | ${line:-no summary row}"
    else
        RESULTS+=("FAIL  $m rc=$rc (${dt}s)"); FAIL=$((FAIL+1))
        say "$m FAILED rc=$rc | $(tail -3 "$LOG" | tr '\n' ' ' | head -c 250)"
    fi
done

# --- is 0.0000 real, or is the harness not scoring anything? ----------------
# A benchmark that reports a perfect score has to be able to show a non-perfect
# one, or a perfect model and a broken pipeline look identical. Same model,
# same pages, deliberately degraded: the error rate MUST move.
say "sanity: does a degraded page produce a non-zero CER?"
for how in blur small; do
    timeout 1200 "$PY_BIN" run_modern_ocr.py --models qwen2-vl-2b \
        --max-samples 6 --degrade "$how" 2>&1 | tail -12
done
clean=$(grep -A6 'qwen2-vl-2b —' "$LOG" | grep 'CER (pooled)' | head -1 | grep -oE '[0-9.]+$')
say "degrade sanity done (clean run was CER $clean)"

# --- the two models that need an older transformers --------------------------
# deepseek-ocr wants LlamaFlashAttention2, removed after 4.47; florence-2's
# remote config predates a field generate() now reads. Both work on 4.47.1, and
# pinning the main environment there would break Qwen2.5-VL -- so they get their
# own environment instead of being written off as "did not run".
LEGACY="${LEGACY_TRANSFORMERS:-4.47.1}"
say "building pinned env (transformers==$LEGACY) for the remote-code models"
if uv venv --system-site-packages /workspace/legacy-tf >/dev/null 2>&1 && \
   uv pip install --python /workspace/legacy-tf/bin/python -q \
       "transformers==$LEGACY" addict easydict matplotlib timm einops \
       >/dev/null 2>&1; then
    LEGACY_PY=/workspace/legacy-tf/bin/python
    # VERIFY the pinned env is the one the script will import, from inside the
    # same interpreter that runs it. Printing the installed version proves
    # nothing about what gets imported when site-packages ordering surprises
    # you -- which is exactly what happened, and cost several runs.
    ACTUAL=$("$LEGACY_PY" -c "import transformers,os;print(transformers.__version__, os.path.dirname(transformers.__file__))")
    say "pinned env imports: $ACTUAL"
    case "$ACTUAL" in
        "$LEGACY "*) : ;;
        *) say "WRONG transformers in the pinned env ($ACTUAL) -- skipping"
           RESULTS+=("FAIL  pinned env imports $ACTUAL, wanted $LEGACY")
           FAIL=$((FAIL+1)); LEGACY_PY="" ;;
    esac
    [ -n "$LEGACY_PY" ] && for m in florence-2-base deepseek-ocr; do
        say "running $m on transformers==$LEGACY ..."
        t0=$(date +%s)
        timeout 2400 "$LEGACY_PY" run_modern_ocr.py --models "$m" \
            --max-samples "$PAGES" 2>&1 | tail -25
        rc=${PIPESTATUS[0]}; dt=$(( $(date +%s) - t0 ))
        if [ $rc -eq 0 ]; then
            RESULTS+=("PASS  $m on transformers==$LEGACY (${dt}s)"); PASS=$((PASS+1))
            say "$m (pinned) done ${dt}s | $(grep -E "^  $m " "$LOG" | tail -1)"
        else
            RESULTS+=("FAIL  $m even on transformers==$LEGACY rc=$rc"); FAIL=$((FAIL+1))
            say "$m (pinned) FAILED | $(tail -3 "$LOG" | tr '\n' ' ' | head -c 220)"
        fi
    done
else
    RESULTS+=("SKIP  pinned-transformers env could not be built")
    say "could not build the pinned env"
fi

echo ""
echo "=============================================================================="
echo "  SUMMARY  pass=$PASS fail=$FAIL"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "  --- measured ---"
grep -E "^  (got-ocr2|florence-2-base|qwen2-vl-2b|qwen2.5-vl-3b|deepseek-ocr) " "$LOG" | sed 's/^/  /'
echo "=============================================================================="
[ "$FAIL" -eq 0 ]
