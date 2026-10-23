#!/usr/bin/env bash
# Is "bigger is worse" a scaling fact or my learning rate?
#
# The ladder used lr=1e-3 for every rung. Large models want less,
# and gru_xl's first epoch hit a loss of 231,748 before recovering
# -- that is a diverging optimiser, not a statement about capacity.
# Re-run the biggest trainable rungs at 3e-4 and 1e-4 and see
# whether the ordering survives. If it does, the scaling claim
# stands; if it does not, the ladder was measuring the optimiser.
set -u
cd "$(dirname "$0")"
mkdir -p logs runs
for LR in 3e-4 1e-4; do
  for CFG in "mlp xxl" "gru xl" "gru large"; do
    set -- $CFG
    NAME="$1_$2_lr$LR"
    [ -f "runs/$NAME/summary.json" ] && { echo "SKIP $NAME"; continue; }
    echo "START $NAME  $(date -u +%H:%M:%S)"
    uv run train_dyn.py --kind "$1" --size "$2" --lr "$LR" \
        --epochs 8 --batch 256 --name "$NAME" > "logs/$NAME.log" 2>&1 \
      && echo "OK    $NAME" || echo "FAIL  $NAME"
  done
done
echo "LR CHECK COMPLETE"
