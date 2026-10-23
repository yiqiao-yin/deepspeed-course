#!/usr/bin/env bash
# Every rung of the size ladder, same data, same epochs, same batch.
#
# A rung that OOMs is a DATA POINT, not a lost run -- topic 7 lab 6
# learned that when MJX's abort() took the interpreter down and
# erased two other backends' results with it. Each rung runs in its
# own process and its failure is recorded rather than fatal.
#
# Equal epochs across rungs on purpose: a scaling comparison where
# the big model also got more passes is not a scaling comparison.
set -u
cd "$(dirname "$0")"
mkdir -p logs runs
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-256}"
for KIND in mlp gru; do
  for SIZE in tiny small medium large xl xxl; do
    NAME="${KIND}_${SIZE}"
    if [ -f "runs/$NAME/summary.json" ]; then echo "SKIP  $NAME"; continue; fi
    echo "START $NAME  $(date -u +%H:%M:%S)"
    if uv run train_dyn.py --kind "$KIND" --size "$SIZE" \
         --epochs "$EPOCHS" --batch "$BATCH" --name "$NAME" \
         > "logs/$NAME.log" 2>&1; then
      echo "OK    $NAME  $(grep -o 'peak GPU *[0-9.]* GB' "logs/$NAME.log" | tail -1)"
    else
      rc=$?
      if grep -qi "out of memory\|CUDA out of memory" "logs/$NAME.log"; then
        echo "OOM   $NAME  — recorded, not a lost run"
        echo '{"name":"'"$NAME"'","oom":true}' > "runs/$NAME/oom.json" 2>/dev/null \
          || { mkdir -p "runs/$NAME"; echo '{"name":"'"$NAME"'","oom":true}' > "runs/$NAME/oom.json"; }
      else
        echo "FAIL  $NAME rc=$rc -- see logs/$NAME.log"
      fi
    fi
  done
done
echo "LADDER COMPLETE"
