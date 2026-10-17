#!/usr/bin/env python3
"""
Rank the searched bodies on fresh episodes, with intervals.

    uv run rank_bodies.py                    # the table
    uv run rank_bodies.py --episodes 48      # tighter intervals

WHY NOT JUST READ THE TRAINING SUMMARIES
-----------------------------------------
Because the trainer's own evaluation is 12 episodes, where one
episode is 8.3% and a single lucky run looks like a finding. Lab 8
published "the hexapod never fell in 36 episodes" off a 12-episode
number and had to retract it.

So every body is re-rolled here on the SAME fresh seeds, and every
proportion carries a Wilson interval. Bodies whose intervals overlap
are not ranked against each other; the table says so rather than
implying an order that the data does not support.

WHAT IS COMPARED, AND WHAT IS NOT
----------------------------------
Every run in the search holds terrain, task, seed, budget,
observation mode and integrator fixed. Only the body changes.

These numbers are NOT comparable with labs 5-8, which ran under the
RK4 integrator. `implicitfast` is 3.93x faster and is what MuJoCo
recommends for contact-rich models, but it measurably changes
behaviour -- the same policy walks 7.9 m instead of 5.9 m. The
re-baselined `if_*` runs exist so this lab can compare against the
five known bodies inside its own physics.

THE HEADLINE METRIC IS DISPLACEMENT, NOT PATH LENGTH
-----------------------------------------------------
Arrival is useless here -- no body arrives reliably and every
interval includes zero. The obvious substitute is "distance walked",
and it is a trap.

Path length sums every centimetre the torso moves, so a robot
thrashing on the spot accumulates it as fast as a robot travelling.
Measured over 24 episodes:

    body        path   displacement   ratio
    if_biped   16.8 m        1.4 m     0.09
    bs_m2      14.3 m        3.3 m     0.23
    bs_m6      15.8 m        3.0 m     0.19
    if_spider8  9.2 m        1.1 m     0.12

The biped "walks" the furthest and ends 1.4 m from where it started.
It is the WORST traveller in the set and path length ranks it first.

So the headline here is DISPLACEMENT -- straight-line start to end --
with `efficiency` (displacement / path) beside it, because the two
together say whether a body is going somewhere or merely moving.
Path length is kept as a secondary column, demoted rather than
deleted, since it is what lab 8 reported and the pages must be
comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def roll(run: str, episodes: int, seed0: int) -> dict | None:
    import torch

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm

    sdir = LAB5 / "runs" / run
    if not (sdir / "summary.json").exists():
        return None
    meta = json.loads((sdir / "summary.json").read_text())
    ck = torch.load(sdir / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta["act_dim"],
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    # The BODY comes from the run's own summary. Rebuilding it from a
    # default would score a 25-actuator policy against a 9-actuator
    # robot, which raises -- loudly, and only because the widths
    # differ. A body whose widths happened to match would run and be
    # silently wrong.
    layout = None
    if meta.get("body_spec"):
        from robot import make_layout
        layout = make_layout(**meta["body_spec"])
    env = NavWorld(mode=meta["mode"], flat=meta.get("flat", False), seed=0,
                   relief_scale=meta.get("relief_scale", 1.0),
                   upright_w=meta.get("upright_w", 0.0),
                   morphology=meta.get("morphology", "biped"),
                   integrator=meta.get("integrator"), layout=layout)
    assert env.obs_dim == meta["obs_dim"], (
        f"{run}: rebuilt body has obs {env.obs_dim}, policy expects "
        f"{meta['obs_dim']} — the summary does not describe this policy")

    fell = arrived = 0
    dist, disp, remain = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        start = env.pos().copy()
        prev = start.copy()
        travelled = 0.0
        while True:
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(act)
            now = env.pos()
            travelled += float(np.linalg.norm(now - prev))
            prev = now.copy()
            if term or trunc:
                break
        fell += int(info["fell"])
        arrived += int(info["arrived"])
        dist.append(travelled)
        disp.append(float(np.linalg.norm(prev - start)))
        remain.append(info["to_goal"])

    flo, fhi = wilson(fell, episodes)
    return {"run": run, "act_dim": int(env.act_dim),
            "obs_dim": int(env.obs_dim),
            "legs": len(layout["legs"]) if layout else None,
            "posture": (layout or {}).get("kind"),
            "spec": meta.get("body_spec"),
            "episodes": episodes,
            "walked_m": float(np.mean(dist)),
            "walked_sd": float(np.std(dist)),
            "displacement_m": float(np.mean(disp)),
            "displacement_sd": float(np.std(disp)),
            "efficiency": float(np.mean(disp) / max(np.mean(dist), 1e-9)),
            "fell": fell / episodes, "fell_ci": [flo, fhi],
            "arrived": arrived / episodes,
            "to_goal_m": float(np.mean(remain)),
            "train_s": meta.get("wall_seconds")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="bs_")
    ap.add_argument("--also", default="if_biped,if_quadruped,if_hexapod,"
                                      "if_spider6,if_spider8",
                    help="re-baselined known bodies, for context")
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--json", default="runs/ranking.json")
    a, _ = ap.parse_known_args()

    runs = sorted(p.name for p in (LAB5 / "runs").glob(f"{a.prefix}*")
                  if (p / "summary.json").exists())
    runs += [r for r in a.also.split(",") if r
             and (LAB5 / "runs" / r / "summary.json").exists()]
    if not runs:
        print(f"\n  no runs matching {a.prefix}*. Train some first:\n"
              f"      uv run search_bodies.py --stage count\n")
        return 1

    print("=" * 84)
    print(f"  BODY SEARCH — {len(runs)} bodies, {a.episodes} fresh "
          f"episodes each, implicitfast")
    print("=" * 84)
    print(f"  {'body':<16} {'legs':>4} {'post':<7} {'act':>4} "
          f"{'moved':>8} {'eff':>6} {'path':>8} {'falls':>16} {'left':>7}")

    rows = []
    for r in runs:
        got = roll(r, a.episodes, a.seed0)
        if got is None:
            continue
        rows.append(got)
        fc = (f"{got['fell']:>5.1%} [{got['fell_ci'][0]:.0%},"
              f"{got['fell_ci'][1]:.0%}]")
        print(f"  {r:<16} {str(got['legs'] or '-'):>4} "
              f"{str(got['posture'] or '-'):<7} {got['act_dim']:>4} "
              f"{got['displacement_m']:>7.1f}m {got['efficiency']:>6.2f} "
              f"{got['walked_m']:>7.1f}m {fc:>16} "
              f"{got['to_goal_m']:>6.1f}m", flush=True)

    if not rows:
        return 1
    rows.sort(key=lambda r: -r["displacement_m"])
    best = rows[0]
    print(f"\n  travels furthest: {best['run']} at "
          f"{best['displacement_m']:.1f} m displaced "
          f"({best['act_dim']} actuators, efficiency "
          f"{best['efficiency']:.2f})")
    by_path = max(rows, key=lambda r: r["walked_m"])
    if by_path["run"] != best["run"]:
        print(f"  (path length would have crowned {by_path['run']} at "
              f"{by_path['walked_m']:.1f} m -- it ends "
              f"{by_path['displacement_m']:.1f} m from its start)")

    # Overlapping intervals are not a ranking. Say so.
    near = [r for r in rows[1:]
            if abs(r["displacement_m"] - best["displacement_m"]) <
            (best["displacement_sd"] + r["displacement_sd"])
            / math.sqrt(a.episodes)]
    if near:
        print(f"  statistically indistinguishable from it: "
              f"{', '.join(r['run'] for r in near)}")

    steadiest = min(rows, key=lambda r: r["fell"])
    print(f"  steadiest: {steadiest['run']} at {steadiest['fell']:.1%} falls")
    if steadiest["run"] != best["run"]:
        print("  -> distance and stability are won by DIFFERENT bodies, "
              "as in lab 8.")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
