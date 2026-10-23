#!/usr/bin/env python3
"""
Collate the size ladder and judge it against the bar.

    uv run ladder.py
    uv run ladder.py --json ladder.json

THE BAR, WRITTEN DOWN BEFORE THE MEASUREMENT
---------------------------------------------
A model 1000x larger must cut 50-step rollout error by at least
HALF, or the sharding machinery is not earning its place in this
topic. Topic 7 lab 6 set its bar the same way -- "5x the CPU worker
pool or renting a GPU is not worth it" -- and reported the FAILS it
got. The bar is printed here whichever way the answer falls.

WHY 50-STEP ERROR AND NOT ONE-STEP
-----------------------------------
At 100 Hz consecutive observations barely differ, so "predict no
change" already scores 0.22 on the one-step metric. Every model
beats that easily and the number says nothing. Open-loop rollout --
feeding the model its own prediction for 50 steps -- is what any
downstream use would actually depend on, and it is where the models
separate.

Both are shown, against both null models, because the gap between
them is the finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

# Measured by baseline.py on the same validation split. Quoted here
# so the table is readable standalone; `uv run baseline.py`
# recomputes them.
NULL = {"one_hold": 0.2185, "one_drift": 0.2180,
        "roll_hold": 1.6334, "roll_drift": 1.4847}


def load_runs(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*/")):
        s = d / "summary.json"
        o = d / "oom.json"
        if s.exists():
            out.append(json.loads(s.read_text()))
        elif o.exists():
            r = json.loads(o.read_text())
            r["oom"] = True
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    runs = load_runs(HERE / a.runs)
    done = [r for r in runs if not r.get("oom") and "params" in r]
    done.sort(key=lambda r: (r["kind"], r["params"]))

    print("=" * 80)
    print("  THE SIZE LADDER")
    print("=" * 80)
    print(f"  null, predict no change:  1-step {NULL['one_hold']:.4f}   "
          f"50-step {NULL['roll_hold']:.3f}")
    print(f"  null, mean drift:         1-step {NULL['one_drift']:.4f}   "
          f"50-step {NULL['roll_drift']:.3f}")
    print()
    print(f"  {'kind':<5}{'size':<8}{'params':>13}{'1-step':>9}"
          f"{'50-step':>9}{'vs null':>9}{'peak GB':>9}{'min':>7}")
    for r in runs:
        if r.get("oom"):
            print(f"  {r['name']:<13}{'OUT OF MEMORY — recorded, not a lost run':>58}")
            continue
        if "params" not in r:
            continue
        vs = NULL["roll_hold"] / r["rollout_err"] if r["rollout_err"] else 0.0
        print(f"  {r['kind']:<5}{r['size']:<8}{r['params']:>13,}"
              f"{r['one_step_err']:>9.4f}{r['rollout_err']:>9.3f}"
              f"{vs:>8.2f}x{r['peak_gpu_gb']:>9.2f}"
              f"{r['wall_seconds'] / 60:>7.1f}")

    print("\n" + "=" * 80)
    print("  DOES SIZE BUY ACCURACY?")
    print("=" * 80)
    verdict = []
    for kind in ("mlp", "gru"):
        ks = [r for r in done if r["kind"] == kind]
        if len(ks) < 2:
            continue
        small, big = ks[0], ks[-1]
        grew = big["params"] / small["params"]
        # Lower is better, so the improvement is small/big.
        cut = (small["rollout_err"] / big["rollout_err"]
               if big["rollout_err"] else float("inf"))
        best = min(ks, key=lambda r: r["rollout_err"])
        print(f"\n  {kind}: {small['params']:,} -> {big['params']:,} "
              f"({grew:,.0f}x the parameters)")
        print(f"    50-step error {small['rollout_err']:.3f} -> "
              f"{big['rollout_err']:.3f}   ({cut:.2f}x better)")
        print(f"    best rung is {best['size']} at {best['params']:,} "
              f"params, error {best['rollout_err']:.3f}")
        ok = grew >= 1000 and cut >= 2.0
        verdict.append((kind, grew, cut, ok))
        print(f"    BAR (1000x must halve the error): "
              f"{'MET' if ok else 'NOT MET'}")

    print("\n" + "=" * 80)
    if verdict and not any(v[3] for v in verdict):
        print("  VERDICT: the bar is NOT met.")
        print()
        print("  Scaling the dynamics model does not buy accuracy on this")
        print("  system, so there is nothing here for ZeRO to be worth")
        print("  sharding -- which is the same answer labs 1 to 5 gave")
        print("  about the policy, now measured rather than asserted for")
        print("  the one model in this topic big enough to ask.")
    elif verdict:
        print("  VERDICT: the bar IS met — size buys accuracy here, and")
        print("  the sharding question is live for this model.")
    print("=" * 80)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"null": NULL, "runs": runs,
             "verdict": [{"kind": k, "grew": g, "cut": c, "met": o}
                         for k, g, c, o in verdict]}, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
