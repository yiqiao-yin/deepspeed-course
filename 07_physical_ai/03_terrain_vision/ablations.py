#!/usr/bin/env python3
"""
The reviewer's questions: is it the camera, and does it generalise?

    uv run ablations.py                 # everything
    uv run ablations.py --only heldout

WHY THIS FILE EXISTS SEPARATELY FROM train_student.py
-----------------------------------------------------
`train_student.py` answers "did this student learn". These are the
questions an outside reader asks next, and all three were missing when
the lab first shipped:

  1. CONFOUND.    The headline compared a VISION student against a BLIND
                  PPO policy. Those differ in two things at once -- the
                  camera, and being distilled from a privileged oracle.
                  A proprioception-only student, same teacher and same
                  objective, is the cell that separates them. It is
                  trained by `train_student.py --no-depth` and scored
                  here beside the others.

  2. GENERALISATION. Training and evaluation both drew the stair rise
                  from [0.06, 0.11]. Interpolation and extrapolation are
                  different results, and a perception claim in
                  particular should say which one it is making.

  3. INTERVALS.   24 episodes is a binomial sample, not a number. Wilson
                  intervals are reported so the reader can see whether
                  two arms actually separate.

Nothing here retrains anything. Every arm is a checkpoint produced
elsewhere, so this file cannot manufacture a result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"

# Training drew the rise from here; see vision_env.RISE_MIN/RISE_MAX.
TRAIN_LO, TRAIN_HI = 0.06, 0.11


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    95% Wilson score interval for a binomial rate.

    Wilson rather than the normal approximation because the rates here
    reach 0% and 100%, where the normal interval has zero width and is
    simply wrong.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_student(run: str):
    import torch
    import torch.nn as nn

    from train_student import build

    ck = torch.load(RUNS / run / "student.pt", weights_only=False)
    net = build(torch, nn, ck["proprio_dim"], 6, ck["width"],
                use_depth=ck.get("use_depth", True))
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck.get("use_depth", True)


def rollouts(net, use_depth, kind, *, episodes, rise=None, mode="vision",
             mean_img=None, base_seed=8000) -> int:
    """Episodes clearing the terrain. `rise=None` keeps the training range."""
    import numpy as np
    import torch

    from vision_env import TerrainWorld

    ok = 0
    for i in range(episodes):
        env = TerrainWorld(kind=kind, privileged=False, depth=use_depth,
                           seed=0, fixed_rise=rise)
        obs, _ = env.reset(seed=base_seed + i)
        while True:
            if not use_depth:
                img = np.zeros((64, 64), np.float32)
            elif mode == "blank":
                img = np.zeros((64, 64), np.float32)
            elif mode == "mean":
                img = mean_img
            else:
                img = env.depth()
            with torch.no_grad():
                a = net(torch.as_tensor(img).unsqueeze(0),
                        torch.as_tensor(obs[:15], dtype=torch.float32
                                        ).unsqueeze(0)).squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ok += bool(info["past_event"])
    return ok


# `student_poor` is a 2-epoch checkpoint kept ONLY to render an
# under-trained animation. It is not a seed of anything, and it matches
# the same `runs/student*` glob as the real arms -- which pulled the
# published vision mean from 89% to 77% the first time this was
# aggregated. An artifact that merely matches a pattern is not a result.
EXCLUDE = {"student_poor"}
SEEDS = {"student": 0, "student_s1": 1, "student_s2": 2,
         "student_blind": 0, "student_blind_s1": 1, "student_blind_s2": 2}


def consolidate() -> dict:
    """
    Gather every arm into one file for the figures to read.

    Deliberately NOT a re-evaluation: `train_student.py` already scored
    each arm under every camera condition and wrote it to
    `runs/<tag>/summary.json`. Re-rolling it here would take an hour and
    would let two numbers for the same thing diverge. This just collects
    what the runs produced.
    """
    import glob

    arms = {}
    for f in sorted(glob.glob(str(RUNS / "student*/summary.json"))):
        name = Path(f).parent.name
        if name in EXCLUDE or name not in SEEDS:
            continue
        d = json.loads(Path(f).read_text())
        arms[name] = {"use_depth": d.get("use_depth", True),
                      "seed": SEEDS[name], "params": d["params"],
                      "epochs": d["epochs"], "rollout": d["rollout"],
                      "eval_episodes": d.get("eval_episodes", 24)}

    for prefix, key in (("v3_priv", "privileged_ppo"), ("v3_blind", "blind_ppo")):
        runs = [json.loads(Path(p).read_text())
                for p in sorted(glob.glob(str(RUNS / f"{prefix}_s*/summary.json")))]
        if runs:
            arms[key] = {"algorithm": "ppo", "seeds": [
                {k: r["final"][f"past_{k}"] for k in ("flat", "up", "down")}
                for r in runs]}

    print("=" * 78)
    print("  CONSOLIDATED ARMS")
    print("=" * 78)
    for tag, want in (("vision BC", True), ("blind BC", False)):
        ups = sorted(a["rollout"]["vision"]["up"] for n, a in arms.items()
                     if n.startswith("student") and a.get("use_depth") is want)
        if ups:
            mean = sum(ups) / len(ups)
            print(f"  {tag:<16} up: {[f'{u:.0%}' for u in ups]}  mean {mean:5.1%}")
    for key, tag in (("privileged_ppo", "privileged PPO"), ("blind_ppo", "blind PPO")):
        if key in arms:
            ups = sorted(s["up"] for s in arms[key]["seeds"])
            print(f"  {tag:<16} up: {[f'{u:.0%}' for u in ups]}  "
                  f"mean {sum(ups)/len(ups):5.1%}")
    if EXCLUDE & {Path(f).parent.name for f in glob.glob(str(RUNS / "student*/summary.json"))}:
        print(f"  (excluded from every aggregate: {sorted(EXCLUDE)})")
    return arms


def arms_table(a) -> dict:
    """Every student arm, per terrain, with intervals."""
    import numpy as np

    from terrain import KINDS

    d = np.load(HERE / "data" / "bc.npz")
    mean_img = d["depth"].mean(axis=0).astype(np.float32)

    runs = [r.name for r in sorted(RUNS.iterdir())
            if (r / "student.pt").exists()]
    print("=" * 78)
    print("  ARMS — episodes clearing the terrain, with 95% Wilson intervals")
    print("=" * 78)

    out = {}
    for run in runs:
        net, use_depth = load_student(run)
        out[run] = {"use_depth": use_depth, "terrain": {}}
        label = "vision" if use_depth else "BLIND (no camera)"
        print(f"\n  {run}  [{label}]")
        for k in KINDS:
            ok = rollouts(net, use_depth, k, episodes=a.episodes)
            lo, hi = wilson(ok, a.episodes)
            out[run]["terrain"][k] = {"k": ok, "n": a.episodes,
                                      "rate": ok / a.episodes,
                                      "ci": [round(lo, 3), round(hi, 3)]}
            print(f"    {k:<6} {ok:>3}/{a.episodes} = {ok/a.episodes:5.1%}"
                  f"   [{lo:5.1%}, {hi:5.1%}]")
    return out


def camera_ablations(a) -> dict:
    """
    Three camera conditions on the vision student.

    `blank` is kept only so the reader can see it is the LOOSE control:
    all-zeros depth means "a wall at 0.8 m", which the network never saw,
    so failing there confounds lost information with OOD brittleness.
    `mean` is in-distribution and carries no per-episode signal, which is
    the comparison the claim actually needs.
    """
    import numpy as np

    from terrain import KINDS

    d = np.load(HERE / "data" / "bc.npz")
    mean_img = d["depth"].mean(axis=0).astype(np.float32)
    net, use_depth = load_student(a.student)
    if not use_depth:
        print("  (that run has no camera; nothing to ablate)")
        return {}

    print()
    print("=" * 78)
    print(f"  CAMERA ABLATIONS on {a.student}")
    print("=" * 78)
    print(f"  {'terrain':<8}{'real depth':>14}{'TRAIN MEAN':>14}{'zeros (OOD)':>14}")
    out = {}
    for k in KINDS:
        row = {}
        for mode in ("vision", "mean", "blank"):
            ok = rollouts(net, True, k, episodes=a.episodes, mode=mode,
                          mean_img=mean_img)
            lo, hi = wilson(ok, a.episodes)
            row[mode] = {"k": ok, "n": a.episodes, "rate": ok / a.episodes,
                         "ci": [round(lo, 3), round(hi, 3)]}
        out[k] = row
        print(f"  {k:<8}{row['vision']['rate']:>13.0%}"
              f"{row['mean']['rate']:>13.0%}{row['blank']['rate']:>13.0%}")
    v = np.mean([out[k]["vision"]["rate"] for k in KINDS])
    m = np.mean([out[k]["mean"]["rate"] for k in KINDS])
    print(f"\n  real depth vs training mean: {v:.0%} vs {m:.0%}  "
          f"({v - m:+.0%})")
    print("  The MEAN column is the one that carries the argument. The zeros")
    print("  column is reported for continuity and is a weaker control.")
    return out


def heldout(a) -> dict:
    """
    Does it work on stairs steeper than any it trained on?

    Training drew the rise from [0.06, 0.11]. This walks the rise out to
    0.15 and reports where performance falls over. A policy that only
    interpolates is a different and much smaller claim than one that
    extrapolates, and the lab did not previously say which it had.
    """
    from terrain import KINDS

    net, use_depth = load_student(a.student)
    rises = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    print()
    print("=" * 78)
    print(f"  HELD-OUT GEOMETRY on {a.student}   "
          f"(trained on rise {TRAIN_LO}-{TRAIN_HI})")
    print("=" * 78)
    hdr = "".join(f"{r:>8.2f}" for r in rises)
    print(f"  {'terrain':<8}{hdr}")
    print(f"  {'':<8}" + "".join(
        f"{('  in' if TRAIN_LO <= r <= TRAIN_HI else ' OUT'):>8}" for r in rises))
    out = {}
    for k in KINDS:
        if k == "flat":
            continue                       # flat has no rise to vary
        row = {}
        cells = []
        for r in rises:
            ok = rollouts(net, use_depth, k, episodes=a.heldout_episodes,
                          rise=r)
            row[f"{r:.2f}"] = {"k": ok, "n": a.heldout_episodes,
                               "rate": ok / a.heldout_episodes}
            cells.append(f"{ok / a.heldout_episodes:>7.0%} ")
        out[k] = row
        print(f"  {k:<8}" + "".join(cells))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="student",
                    help="which run to ablate (default: the seed-0 vision one)")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--heldout-episodes", type=int, default=12)
    ap.add_argument("--only", choices=("consolidate", "arms", "camera",
                                       "heldout"), default=None,
                    help="`arms` and `camera` RE-ROLL what train_student.py "
                         "already measured and take about an hour; they are "
                         "an independent re-verification, not part of the "
                         "normal path.")
    ap.add_argument("--out", default="runs/ablations.json")
    a, _ = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    result = {}
    if a.only in (None, "consolidate", "heldout"):
        result["arms"] = consolidate()
    if a.only == "arms":
        result["reverified_arms"] = arms_table(a)
    if a.only == "camera":
        result["reverified_camera"] = camera_ablations(a)
    if a.only in (None, "heldout"):
        result["heldout"] = heldout(a)

    # Preserve sections an earlier invocation produced, so `--only` does
    # not silently delete the slow one someone already paid for.
    path = HERE / a.out
    if path.exists():
        prev = json.loads(path.read_text())
        prev.update(result)
        result = prev
    path.write_text(json.dumps(result, indent=2))

    # The figures read this one; keep it beside the raw file.
    if "arms" in result:
        (RUNS / "results.json").write_text(json.dumps(
            {"arms": result["arms"], "heldout": result.get("heldout", {}),
             "excluded": sorted(EXCLUDE),
             "note": "Written by ablations.py, read by make_figures.py. "
                     "No number here is typed by hand."}, indent=2))
        print("  wrote runs/results.json  (what make_figures.py reads)")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
