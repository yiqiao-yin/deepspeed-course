#!/usr/bin/env python3
"""
Score the trained checkpoints properly, on enough episodes to mean something.

    uv run evaluate.py --episodes 120

WHY THIS IS SEPARATE FROM TRAINING
----------------------------------
`train_nav.py` evaluates every few iterations on 12 episodes, which is
the right trade DURING training: it is a progress signal, and running
120 episodes every checkpoint would cost more than the training does.

It is the wrong number to PUBLISH. Twelve episodes quantises the result
to 8.3%, so the three seeds of each arm came back as 0%, 8% and 17% --
literally 0, 1 and 2 episodes -- and the granularity of the measurement
was the same size as the effect being measured. Two arms whose means
landed on exactly 8.3% apiece cannot be compared at that resolution,
and reporting "no difference" from it would say more about the
evaluation than about the policies.

So this re-scores the SAME checkpoints on many more episodes, and
reports a Wilson interval with each rate so the reader can see whether
two arms actually separate. Nothing is retrained; this file cannot
manufacture a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(run: str):
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((RUNS / run / "summary.json").read_text())
    ck = torch.load(RUNS / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], 7)
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm, meta


def score(run: str, episodes: int, goal_range: float | None) -> dict:
    import numpy as np
    import torch

    from nav_env import NavWorld

    net, norm, meta = load(run)
    env = NavWorld(mode=meta["mode"], flat=meta["flat"], seed=0,
                   goal_range=goal_range)
    arrived = fell = 0
    effs, lefts = [], []
    for i in range(episodes):
        # The SAME episode seeds for every arm, so the two are compared
        # on identical maps rather than on their own lucky draws.
        obs, _ = env.reset(seed=20_000 + i)
        while True:
            with torch.no_grad():
                t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(t).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        arrived += bool(info["arrived"])
        fell += bool(info["fell"])
        lefts.append(info["to_goal"])
        if info["arrived"] and info["travelled"] > 0.1:
            effs.append(min(info["route"] / info["travelled"], 1.0))
    lo, hi = wilson(arrived, episodes)
    return {"run": run, "mode": meta["mode"], "n": episodes,
            "arrived_k": arrived, "arrived": arrived / episodes,
            "ci": [round(lo, 3), round(hi, 3)],
            "fell": fell / episodes, "to_goal": float(np.mean(lefts)),
            "efficiency": float(np.mean(effs)) if effs else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="nav2_")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--goal-range", type=float, default=7.0,
                    help="must match what the runs were TRAINED on, or this "
                         "measures a distribution shift instead of a policy")
    ap.add_argument("--out", default="runs/evaluation.json")
    a, _ = ap.parse_known_args()
    sys.path.insert(0, str(HERE))

    import numpy as np

    runs = sorted(Path(p).parent.name
                  for p in glob.glob(str(RUNS / f"{a.prefix}*/policy.pt")))
    if not runs:
        print(f"  no checkpoints matching {a.prefix}*", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"  EVALUATION — {a.episodes} episodes per arm, identical maps")
    print("=" * 78)
    rows = []
    for r in runs:
        row = score(r, a.episodes, a.goal_range)
        rows.append(row)
        print(f"  {r:<24} {row['arrived_k']:>3}/{row['n']} = "
              f"{row['arrived']:5.1%}  [{row['ci'][0]:5.1%},{row['ci'][1]:5.1%}]"
              f"   eff {row['efficiency']:4.0%}  left {row['to_goal']:4.1f} m")

    print()
    for mode in ("blind", "privileged"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        k = sum(r["arrived_k"] for r in sel)
        n = sum(r["n"] for r in sel)
        lo, hi = wilson(k, n)
        print(f"  {mode:<12} pooled {k:>4}/{n} = {k/n:5.1%}  "
              f"[{lo:5.1%}, {hi:5.1%}]   "
              f"per-seed {[format(r['arrived'], '.0%') for r in sel]}")

    (HERE / a.out).write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {a.out}")
    print("  Pooling across seeds is an UPPER bound on the evidence: episodes")
    print("  within a seed share a policy, so they are not independent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
