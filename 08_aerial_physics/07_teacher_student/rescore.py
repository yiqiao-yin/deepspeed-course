#!/usr/bin/env python3
"""
Re-fly every arm down ONE common list of held-out launches.

    uv run rescore.py                     # all arms, 120 launches
    uv run rescore.py --episodes 200

WHY THIS SCRIPT EXISTS
-----------------------
`train_policy.py` and `distill.py` each score their own arm when
they finish, and they were given different episode counts -- 48 for
the PPO arms, 120 for the distilled ones. Both numbers are correct
and the two sets cannot be compared, because `compare.py` pairs
arms episode by episode and there is no pairing between a list of
48 launches and a list of 120.

`compare.py` REFUSED to report rather than quietly comparing the
summary rates, which is the behaviour that matters: an unpaired
comparison of two different sets of launches would have produced a
perfectly reasonable-looking table on terrain whose per-spot
success varies from 8% to 83%.

So this script re-scores everything on one seed list and rewrites
each run's `episodes.json`. It changes no weights and trains
nothing; it only makes the arms commensurable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--world", default="earth")
    a, _ = ap.parse_known_args()

    import numpy as np

    from render_arms import load_any, sense_of
    from teach_env import TeachEnv

    root = HERE / a.runs
    dirs = sorted(d for d in root.glob("*")
                  if (d / "policy.pt").exists()
                  and (d / "summary.json").exists())
    if not dirs:
        raise SystemExit(f"no runs under {root}")

    print("=" * 78)
    print(f"  RESCORE — {len(dirs)} arms on {a.episodes} identical launches "
          f"(seeds {a.seed0}..{a.seed0 + a.episodes - 1})")
    print("=" * 78)

    for d in dirs:
        act, odim, kind = load_any(d)
        env = TeachEnv(a.rig, a.preset, sense=sense_of(d), seed=0,
                       world=a.world)
        if env.obs_dim != odim:
            raise ValueError(
                f"{d.name}: policy wants {odim} inputs, this env gives "
                f"{env.obs_dim}. Wrong rig or preset — refusing to score "
                f"it rather than producing a number for the wrong task.")
        seeds, hits, miss, outs = [], [], [], []
        for ep in range(a.episodes):
            obs = env.reset(a.seed0 + ep)
            done = False
            while not done:
                obs, _r, done, _i = env.step(np.clip(act(obs), 0.0, 1.0))
            seeds.append(a.seed0 + ep)
            hits.append(env.outcome == "on_target")
            outs.append(env.outcome)
            miss.append(env.miss())
        (d / "episodes.json").write_text(json.dumps(
            {"seeds": seeds, "on_target": hits, "outcome": outs,
             "miss": miss}))
        s = json.loads((d / "summary.json").read_text())
        s["rescored"] = {"episodes": a.episodes, "seed0": a.seed0,
                         "rate": sum(hits) / a.episodes,
                         "miss": float(np.mean(miss))}
        (d / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"  {d.name:<13} {kind:<24} on target "
              f"{sum(hits) / a.episodes:>6.1%}   miss "
              f"{np.mean(miss):>5.2f} m   crash "
              f"{outs.count('crash') / a.episodes:>5.1%}")

    print("\n  every arm now has the same seed list; `uv run compare.py` "
          "can pair them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
