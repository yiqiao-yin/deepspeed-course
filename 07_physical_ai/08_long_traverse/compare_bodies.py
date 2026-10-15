#!/usr/bin/env python3
"""
Five bodies, one task: the morphology table, with intervals.

    uv run compare_bodies.py                 # the table, CPU, seconds
    uv run compare_bodies.py --episodes 60   # tighter intervals
    uv run compare_bodies.py --json runs/bodies.json

THE CLAIM UNDER TEST
--------------------
Three labs reached the same wall from three directions -- lab 5 part
4 (0% on scree), lab 7 (perception cannot help), lab 8 (the traverse
stalls) -- and each time the body was the suspect. This measures it:
same terrain, same task, same 1.5M-step budget, same seed, only the
machine changes.

MORPHOLOGICAL COMPUTATION, AND WHY IT IS FALSIFIABLE HERE
----------------------------------------------------------
Pfeifer and Bongard's term for structure doing work a controller
would otherwise do; the canonical demonstration is McGeer's passive
dynamic walker, which walks down a slope with no actuators and no
software. The framing is contested -- Hoffmann and Mueller argue
"offloading computation" is a metaphor and the walker is simply
well-shaped physics -- and this lab does not need to settle that.

It needs the prediction, which is sharp either way:

    A sprawling body is statically stable, so its policy should not
    have to LEARN balance. spider8 carries 25 actuators against the
    biped's 9 -- a far harder search -- and should still hold its own
    at an identical step budget. If the biped wins at equal budget,
    the body bought nothing here.

THE TRADE-OFF, WHICH IS NOT OURS AND IS WELL ESTABLISHED
---------------------------------------------------------
Comparative biomechanics has measured this in animals: sprawling
posture gives stability and manoeuvrability but costs energy,
because the feet sit far from the centre of mass so muscles resist
large joint moments and the body sways laterally. Erect posture
loads limbs in compression, recovers pendular and spring energy, and
costs less -- at the price of static stability. Larger tetrapods
tend to erect postures for exactly that reason.

So the expectation is a TRADE, not a winner: sprawlers should fall
less and travel less. A lab that only reported falls would show
half of it.

WHAT THIS LAB CANNOT CLAIM
--------------------------
Anything about evolution. A MuJoCo robot can illustrate why a
trade-off exists; it is not evidence about how tetrapods actually
evolved, and no number here should be read that way.
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

BODIES = ["biped", "quadruped", "hexapod", "spider6", "spider8"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def roll(body: str, run: str, episodes: int, seed0: int) -> dict:
    """Fresh episodes, so the interval is ours and not the trainer's."""
    import torch

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm

    meta = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
    ck = torch.load(LAB5 / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta["act_dim"],
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    env = NavWorld(mode=meta["mode"], flat=meta.get("flat", False), seed=0,
                   relief_scale=meta.get("relief_scale", 1.0),
                   upright_w=meta.get("upright_w", 0.0), morphology=body)

    arrived = fell = 0
    dist, remain, down_frac = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        prev = env.pos().copy()
        travelled, down, steps = 0.0, 0, 0
        while True:
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(act)
            steps += 1
            now = env.pos()
            travelled += float(np.linalg.norm(now - prev))
            prev = now.copy()
            down += int(env.fallen())
            if term or trunc:
                break
        arrived += int(info["arrived"])
        fell += int(info["fell"])
        dist.append(travelled)
        remain.append(info["to_goal"])
        down_frac.append(down / max(steps, 1))

    lo, hi = wilson(arrived, episodes)
    flo, fhi = wilson(fell, episodes)
    return {"body": body, "act_dim": int(env.act_dim),
            "obs_dim": int(env.obs_dim), "episodes": episodes,
            "arrived": arrived / episodes, "arrived_ci": [lo, hi],
            "fell": fell / episodes, "fell_ci": [flo, fhi],
            "walked_m": float(np.mean(dist)),
            "to_goal_m": float(np.mean(remain)),
            "down_fraction": float(np.mean(down_frac)),
            "stance_m": float(env._stand_clearance)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=",".join(BODIES))
    ap.add_argument("--prefix", default="morph_")
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--json", default="runs/bodies.json")
    a, _ = ap.parse_known_args()

    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()]
    rows = []
    print("=" * 86)
    print("  MORPHOLOGY — same terrain, same task, same budget, "
          "only the body changes")
    print("=" * 86)
    print(f"  {'body':<10} {'act':>4} {'stance':>7} {'arrived':>18} "
          f"{'falls':>18} {'walked':>8} {'left':>7}")
    for b in bodies:
        run = f"{a.prefix}{b}"
        if not (LAB5 / "runs" / run / "summary.json").exists():
            print(f"  {b:<10}  no trained policy at runs/{run}")
            continue
        r = roll(b, run, a.episodes, a.seed0)
        rows.append(r)
        ac = f"{r['arrived']:>5.1%} [{r['arrived_ci'][0]:.0%},{r['arrived_ci'][1]:.0%}]"
        fc = f"{r['fell']:>5.1%} [{r['fell_ci'][0]:.0%},{r['fell_ci'][1]:.0%}]"
        print(f"  {b:<10} {r['act_dim']:>4} {r['stance_m']:>6.2f}m "
              f"{ac:>18} {fc:>18} {r['walked_m']:>7.1f}m "
              f"{r['to_goal_m']:>6.1f}m")

    if len(rows) >= 2:
        base = rows[0]
        print(f"\n  against the baseline ({base['body']}):")
        for r in rows[1:]:
            df = r["fell"] - base["fell"]
            dw = r["walked_m"] - base["walked_m"]
            print(f"    {r['body']:<10} falls {df:+6.1%}   "
                  f"walked {dw:+6.1f} m   "
                  f"({r['act_dim']/base['act_dim']:.1f}x the actuators, "
                  f"same step budget)")
        # The trade-off, stated only if it is actually present.
        sprawl = [r for r in rows if r["stance_m"] < 0.30]
        if sprawl:
            s = sprawl[-1]
            safer = s["fell"] < base["fell"]
            slower = s["walked_m"] < base["walked_m"]
            if safer and slower:
                print("\n  TRADE-OFF PRESENT: the sprawling body falls less "
                      "AND travels less,\n  which is the stability/efficiency "
                      "trade comparative biomechanics\n  reports for sprawling "
                      "versus erect posture in animals.")
            elif safer and not slower:
                print("\n  NO TRADE-OFF HERE: the sprawling body falls less "
                      "and travels at least\n  as far. The expected energetic "
                      "penalty did not appear at this scale.")
            else:
                print("\n  PREDICTION FAILED: the sprawling body did not even "
                      "fall less.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
