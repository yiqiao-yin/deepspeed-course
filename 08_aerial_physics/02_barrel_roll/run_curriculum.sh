#!/bin/bash
# Reproduce lab 2: learn the ROLL first, then add the landing.
#
#     ./run_curriculum.sh          # stage A then three stage-B seeds
#
# THE ORDER IS THE RESULT. Do not swap these stages.
#
# Measured, 96 held-out throws each:
#
#   warm started from a LANDING policy   roll  0.0%   land 85-97%
#   trained from scratch                 roll 95.0%   land  9%
#
# Neither half transfers to the other by accident, and which one a
# run ends up with is decided entirely by how it was initialised.
#
# The reason is asymmetric. A landing prior is the reflex "null the
# body rates", which destroys the launch rotation in ~0.3 s -- before
# enough roll accumulates for the shaping to pay anything. It is
# actively incompatible with the new task. A ROLLING prior is "do not
# fight the rotation", which costs the landing nothing: the car still
# has 5.6 s of clock left after the turn, and landing is the half
# lab 1 already proved learnable.
#
# So stage A is the roll and stage B is the landing. Stage A is trained
# here from scratch; it reaches ~95% roll completion on ~104 kg.
#
# Self-healing: each job is retried until it writes a summary, so a
# lost session costs that run's progress and nothing else.
set -u
cd "$(dirname "$0")"

JOBS="0 1 2"

needs () { [ -f "runs/roll2land_s$1/summary.json" ] && return 1 || return 0; }

train () {
  local seed=$1 name="roll2land_s$1"
  needs "$seed" || { echo "skip $name"; return 0; }
  rm -rf "runs/$name"
  uv run train_roll.py --rig reveal --name "$name" --seed "$seed" \
      --init-from runs/reveal_scratch2 --total-steps 3000000 \
      --n-envs 16 --eval-episodes 96 > "logs_${name}.txt" 2>&1
  if needs "$seed"; then echo "FAILED $name"; tail -3 "logs_${name}.txt"
  else echo "ok $name"; fi
}
export -f train needs

# Stage A: the roll, from scratch. Skipped if already present.
if [ ! -f runs/reveal_scratch2/summary.json ]; then
  echo "=== STAGE A: learning the roll, from scratch ==="
  uv run train_roll.py --rig reveal --name reveal_scratch2 --seed 0 \
      --total-steps 3000000 --n-envs 16 --eval-episodes 96 \
      > logs_reveal_scratch2.txt 2>&1
fi

echo "=== STAGE B: adding the landing, warm started ==="
for pass in 1 2 3; do
  todo=""
  for s in $JOBS; do needs "$s" && todo="$todo $s"; done
  [ -z "${todo// /}" ] && break
  echo "=== pass $pass, seeds:$todo"
  for s in $todo; do echo "$s"; done | xargs -P 2 -L 1 bash -c 'train "$0"'
done
echo "ALLDONE_MARKER"
