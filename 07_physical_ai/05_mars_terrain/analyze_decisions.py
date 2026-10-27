#!/usr/bin/env python3
"""
Does the robot actually TELL THE TWO OBSTACLE CLASSES APART?

    uv run analyze_decisions.py --run m2_blind_s0
    uv run analyze_decisions.py --run m2_privileged_s1 --episodes 60

THE QUESTION THIS LAB EXISTS TO ANSWER, AND HAS NEVER ASKED
------------------------------------------------------------
Part 1 built two classes of landform and measured that they afford
different things: a dune's edge is blocked on 4-6% of its ring, a
scarp's on 17-24%. Part 2 trained a policy that reaches the goal about
one episode in seven.

Neither of them checked the thing in between. A robot can reach B at
15% while treating every obstacle identically -- going around
everything, or blundering over everything and surviving the ones that
happen to be climbable. Arrival rate cannot distinguish those, and
arrival rate is all we have measured.

So this script watches BEHAVIOUR rather than outcome. For every
landform the robot comes near, it records whether the robot went OVER
it or AROUND it, and compares that against what the landform affords:

    climbable (dune, hill)   going over is correct and cheaper
    blocked   (scarp, mesa)  going over is impossible; around is correct

The headline is the DIFFERENCE between the two cross rates. A policy
that reads terrain crosses dunes far more often than scarps. A policy
that does not shows the same rate for both, whatever its arrival score.

WHY THIS NEEDS NO TRAINING
---------------------------
It replays checkpoints that already exist. Any policy can be measured
this way in a couple of minutes, which is the whole point: it is the
cheap experiment that should have come before the expensive ones. This
lab has already spent six hours establishing that a wrongly-specified
task fails, when a ten-minute measurement would have said so.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# How close the robot's path must come for the landform to count as
# "met". Scaled to the landform, not absolute: a 7 m dune ridge and a
# 2 m butte are not encountered at the same distance.
NEAR = 1.9
# Inside this fraction of the landform's own radius counts as ON it.
ON = 0.85


def load(run: str):
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta.get("act_dim", 7),
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm, meta


def encounters(track: list[tuple[float, float]], landforms) -> list[dict]:
    """
    Which landforms did this path meet, and did it go over or around?

    `track` is the robot's (x, y) history. For each landform we ask two
    questions in order, because they are not the same question:

      met     did the path come within NEAR metres of the footprint?
      crossed did the path enter the footprint itself?

    A landform the robot never approached tells us nothing about its
    preferences, so it is excluded rather than counted as "avoided" --
    counting distant obstacles as successful avoidance is how you
    manufacture a competence that was never exercised.
    """
    if not track:
        return []
    P = np.asarray(track)
    out = []
    for kind, cx, cy, rad, height in landforms:
        d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
        closest = float(d.min())
        if closest > rad + NEAR:
            continue                      # never came near it
        out.append(dict(kind=kind, radius=float(rad), height=float(height),
                        closest=closest,
                        crossed=bool(closest < rad * ON)))
    return out


def analyse(run: str, episodes: int, seed0: int = 30_000) -> dict:
    import torch

    from mars_env import NavWorld

    net, norm, meta = load(run)
    env = NavWorld(mode=meta["mode"], flat=meta.get("flat", False),
                   seed=0, relief_scale=meta.get("relief_scale", 1.0),
                   upright_w=meta.get("upright_w", 0.0))

    tally: dict[str, dict] = {}
    arrived = 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        track = []
        while True:
            with torch.no_grad():
                t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(t).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(act)
            track.append((info["x"], info["y"]))
            if term or trunc:
                break
        arrived += bool(info["arrived"])
        for e in encounters(track, env.map.landforms):
            # Collapse to the two AFFORDANCE classes. The lab's claim is
            # about climbable vs blocked, not about dunes vs hills.
            cls = "climbable" if e["kind"] in ("dune", "hill") else "blocked"
            t_ = tally.setdefault(cls, {"met": 0, "crossed": 0})
            t_["met"] += 1
            t_["crossed"] += e["crossed"]

    out = {"run": run, "mode": meta["mode"], "episodes": episodes,
           "arrived": arrived / episodes}
    for cls in ("climbable", "blocked"):
        t_ = tally.get(cls, {"met": 0, "crossed": 0})
        out[cls] = dict(met=t_["met"], crossed=t_["crossed"],
                        rate=(t_["crossed"] / t_["met"]) if t_["met"] else float("nan"))
    a, b = out["climbable"]["rate"], out["blocked"]["rate"]
    out["discrimination"] = (a - b) if (a == a and b == b) else float("nan")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="repeatable; compare several policies")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default=None)
    a, _ = ap.parse_known_args()
    sys.path.insert(0, str(HERE))

    print("=" * 78)
    print(f"  DOES IT TELL THE CLASSES APART?  {a.episodes} episodes each")
    print("=" * 78)
    print(f"  {'run':<20}{'arrived':>9}{'climbable':>22}{'blocked':>22}"
          f"{'diff':>8}")
    rows = []
    for run in a.run:
        r = analyse(run, a.episodes)
        rows.append(r)
        c, b = r["climbable"], r["blocked"]
        cl, ch = wilson(c["crossed"], c["met"])
        bl, bh = wilson(b["crossed"], b["met"])
        print(f"  {run:<20}{r['arrived']:>8.0%}"
              f"  {c['crossed']:>4}/{c['met']:<4} ={c['rate']:>5.0%}"
              f" [{cl:.0%},{ch:.0%}]"
              f"  {b['crossed']:>4}/{b['met']:<4} ={b['rate']:>5.0%}"
              f" [{bl:.0%},{bh:.0%}]"
              f"  {r['discrimination']:>+6.0%}")

    print()
    print("  `diff` is the claim: how much MORE often the robot crosses a")
    print("  climbable landform than a blocked one. Near zero means the")
    print("  policy treats them the same, whatever its arrival rate --")
    print("  and arrival rate alone could never have told you.")
    print()
    print("  A caution on `blocked`: a non-zero cross rate there is not")
    print("  necessarily a mistake. `traversable()` is conservative --")
    print("  one measured episode walked 3.4 m where the planner routed")
    print("  7.6 m -- so some 'blocked' edges are climbable in practice.")

    if a.out:
        (HERE / a.out).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
