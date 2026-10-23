#!/usr/bin/env python3
"""
The null model: predict that nothing changes.

    uv run baseline.py

At 100 Hz consecutive observations differ by very little, so
`o_{t+1} = o_t` is already a decent predictor and every learned
number has to be read against it. A model that cannot beat this has
learned nothing, and a one-step error that looks small may be
entirely this effect.

A second null is included: the mean observed delta, applied every
step. It is what a model that learned only the average drift would
do, and it is the harder of the two to beat.
"""
from __future__ import annotations
import numpy as np

def main() -> int:
    tr = np.load("data/train.npz", allow_pickle=True)
    va = np.load("data/val.npz", allow_pickle=True)
    mean_delta = (tr["nxt"] - tr["obs"]).mean(0, keepdims=True)

    print("=" * 74)
    print("  NULL MODELS — what 'learned nothing' scores")
    print("=" * 74)
    one_hold = float(np.mean(np.linalg.norm(va["nxt"] - va["obs"], axis=1)))
    one_drift = float(np.mean(np.linalg.norm(
        va["nxt"] - va["obs"] - mean_delta, axis=1)))
    print(f"  1-step,  hold still      {one_hold:.4f}")
    print(f"  1-step,  mean drift      {one_drift:.4f}")

    for H in (10, 50):
        errs_h, errs_d = [], []
        for s, e in zip(va["ep_start"], va["ep_end"]):
            if e - s <= H + 1:
                continue
            o0 = va["obs"][s]
            truth = va["obs"][s + H]
            errs_h.append(np.linalg.norm(o0 - truth))
            errs_d.append(np.linalg.norm(o0 + H * mean_delta[0] - truth))
        print(f"  {H}-step, hold still      {np.mean(errs_h):.4f}")
        print(f"  {H}-step, mean drift      {np.mean(errs_d):.4f}")
    print()
    print("  Any learned model must be read against these. Beating the")
    print("  1-step hold is nearly free; beating the 50-step one is not.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
