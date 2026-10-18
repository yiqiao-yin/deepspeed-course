#!/usr/bin/env python3
"""
Score a trained policy on HELD-OUT throws, against the baseline.

    uv run evaluate.py runs/r_reveal
    uv run evaluate.py runs/r_reveal --episodes 96 --compare

HELD OUT MEANS HELD OUT
------------------------
Training seeds the pool from `seed * 1000`; evaluation starts at
900,000 and never overlaps. A policy scored on the throws it trained
on is a memorisation test, and on a task with a randomised initial
tumble it is an easy mistake to make silently -- the numbers simply
come out better and nothing complains.

WILSON, NOT A BARE FRACTION
----------------------------
48 episodes is not many. 39/48 and 44/48 look like a clear ordering
and their 95% intervals overlap heavily. Topic 7 published a
comparison whose intervals overlapped and had to walk it back; the
interval is printed next to every rate here so the reader can see
for themselves whether the ordering survives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_policy(run_dir):
    import torch

    from ppo import ActorCritic, RunningNorm

    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "policy.pt", map_location="cpu",
                    weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["act_dim"])
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(ck["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm


def evaluate(run_dir, episodes: int = 48, seed0: int = 900_000,
             **env_kw) -> dict:
    """Mean action, no sampling: a deployed policy does not explore."""
    import torch

    from aerial_env import AerialEnv

    net, norm = load_policy(run_dir)
    env = AerialEnv(**env_kw)
    tally: dict[str, int] = {}
    sinks, tilts, gas, ok = [], [], [], 0
    for ep in range(episodes):
        o = env.reset(seed0 + ep)
        while True:
            with torch.no_grad():
                x = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
                a = net.distribution(x).mean.squeeze(0).numpy()
            o, _r, d, _ = env.step(np.clip(a, 0.0, 1.0))
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gas.append(env.gas_used)
        if env.touchdown:
            sinks.append(env.touchdown["sink"])
            tilts.append(env.touchdown["tilt"])
        if env.landed_ok():
            ok += 1
    lo, hi = wilson(ok, episodes)
    return {"episodes": episodes, "landed_ok": ok, "rate": ok / episodes,
            "ci": [lo, hi], "tally": tally,
            "sink": float(np.mean(sinks)) if sinks else float("nan"),
            "tilt": float(np.mean(tilts)) if tilts else float("nan"),
            "gas": float(np.mean(gas))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--compare", action="store_true",
                    help="score the classical baseline on the SAME throws")
    a, _ = ap.parse_known_args()

    run = Path(a.run)
    meta = json.loads((run / "summary.json").read_text())
    kw = dict(rig_name=meta["rig"], preset=meta["preset"],
              flat=meta.get("flat", False))
    if kw["preset"] == "flat":
        kw["preset"] = "rolling"

    print("=" * 74)
    print(f"  EVALUATE — {run.name}: rig {meta['rig']}, "
          f"terrain {meta['preset']}")
    print("=" * 74)
    r = evaluate(run, a.episodes, a.seed0, **kw)
    lo, hi = r["ci"]
    print(f"  PPO        landed cleanly {r['landed_ok']:>3}/{r['episodes']}  "
          f"{r['rate']:>5.0%}  [{lo:.0%}, {hi:.0%}]   "
          f"sink {r['sink']:.2f} m/s   gas {r['gas']:.0f} kg")
    for k in sorted(r["tally"]):
        print(f"               {k:<14} {r['tally'][k]:>3}")

    if a.compare:
        from classical import run as run_pd

        b = run_pd(meta["rig"], kw["preset"], kw["flat"], a.episodes,
                   seed0=a.seed0)
        blo, bhi = wilson(b["landed_ok"], b["episodes"])
        print(f"\n  classical  landed cleanly {b['landed_ok']:>3}/"
              f"{b['episodes']}  {b['rate']:>5.0%}  [{blo:.0%}, {bhi:.0%}]   "
              f"sink {b['sink']:.2f} m/s   gas {b['gas']:.0f} kg")
        overlap = not (lo > bhi or blo > hi)
        print(f"\n  intervals {'OVERLAP — no measurable difference at this '
                              'sample size' if overlap else 'are DISJOINT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
