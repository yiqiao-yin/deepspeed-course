#!/usr/bin/env python3
"""
Render a behaviour-cloning dataset from the privileged teacher.

    uv run collect.py --episodes 120

WHY THIS IS A SEPARATE STEP
---------------------------
Rendering inside a training loop costs 250x. Measured on this machine:
7,077 physics steps/s with no camera, 79 frames/s with depth. An
on-policy vision run would take eight hours instead of four minutes.

Behaviour cloning moves that cost out of the loop. The teacher acts on
privileged state (no rendering, full speed), we render ONCE alongside it,
and the student then trains supervised on a fixed dataset -- which is
also the first thing in this category where a GPU has real work to do.

WHAT GOES IN A SAMPLE
---------------------
    depth      64x64, clipped to a FIXED 0.8-3.5 m window
    proprio    15 numbers the robot genuinely has
    action     6 torques -- the label
    terrain    0/1/2 -- NEVER an input; it is how we later test whether
               the student's encoder actually learned to see

COVERAGE
--------
Rollouts use the teacher plus exploration noise. A dataset of perfect
trajectories teaches a student nothing about recovering, and the first
time it drifts off the teacher's line it has never seen the state it is
in. The noise is the cheap half of the fix; one DAgger round is the
other, and `--relabel` does that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def load_policy(run: str):
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], 6)
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=None,
                    help="run directory; defaults to the best v3_priv_* seed")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--noise", type=float, default=0.15,
                    help="exploration noise on the teacher's action, so the "
                         "dataset contains recovery states and not just the "
                         "ideal line")
    ap.add_argument("--every", type=int, default=2,
                    help="keep every Nth step; consecutive frames are nearly "
                         "identical and cost render time for no new signal")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--fovy", type=float, default=45.0,
                    help="camera field of view in degrees. The dataset has "
                         "to be re-rendered per FOV -- the depth image is "
                         "exactly what changes -- so this is the expensive "
                         "axis of the sweep.")
    ap.add_argument("--kinds", default=None,
                    help="comma-separated terrains to collect from. "
                         "Defaults to the published three.")
    ap.add_argument("--out", default="data/bc.npz")
    a, _ = ap.parse_known_args()
    collect(a)


def collect(a) -> None:
    import numpy as np
    import torch

    from terrain import KINDS, STONE_KINDS
    from vision_env import TerrainWorld

    # The terrain set the TEACHER was trained on decides the
    # observation width, so it has to match or the policy will
    # not load. `--kinds stones` collects only stones but still
    # declares the four-terrain observation.
    want = ([k.strip() for k in a.kinds.split(',')] if a.kinds
            else list(KINDS))
    obs_kinds = STONE_KINDS if 'stones' in want else KINDS

    if a.teacher is None:
        cands = sorted(p.name for p in (HERE / "runs").iterdir()
                       if p.name.startswith("v3_priv_")
                       and (p / "summary.json").exists())
        if not cands:
            print("  no teacher found. Train one first:\n"
                  "      uv run train_teacher.py --name v3_priv_s0",
                  file=sys.stderr)
            sys.exit(1)
        a.teacher = max(cands, key=lambda n: json.loads(
            (HERE / "runs" / n / "summary.json").read_text())["final"]["past_all"])

    net, norm, meta = load_policy(a.teacher)
    rng = np.random.default_rng(0)
    print("=" * 74)
    print(f"  collecting from {a.teacher}  "
          f"(past_all {meta['final']['past_all']:.0%})")
    print("=" * 74)

    D, P, A, T = [], [], [], []
    t0 = time.time()
    for ep in range(a.episodes):
        kind = want[ep % len(want)]            # balanced across terrains
        env = TerrainWorld(kind=kind, privileged=True, depth=True,
                           depth_res=a.res, seed=1000 + ep,
                           fovy=a.fovy, kinds=obs_kinds)
        obs, _ = env.reset(seed=1000 + ep)
        step = 0
        while True:
            with torch.no_grad():
                t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                clean = net.distribution(t).mean.squeeze(0).numpy()
            act = np.clip(clean + rng.normal(0, a.noise, 6), -1, 1)

            if step % a.every == 0:
                D.append(env.depth())
                # Only what a real robot has: the privileged tail is dropped.
                P.append(obs[:15].copy())
                A.append(clean.copy())        # label is the CLEAN action
                T.append(want.index(kind))

            obs, _, term, trunc, _ = env.step(act)
            step += 1
            if term or trunc:
                break
        if (ep + 1) % 20 == 0:
            print(f"  {ep+1:>4}/{a.episodes} episodes   {len(D):>6} frames   "
                  f"{time.time()-t0:5.0f}s")

    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, depth=np.asarray(D, dtype=np.float32),
        proprio=np.asarray(P, dtype=np.float32),
        action=np.asarray(A, dtype=np.float32),
        terrain=np.asarray(T, dtype=np.int64))
    counts = np.bincount(np.asarray(T), minlength=len(want))
    print()
    print(f"  wrote {out.relative_to(HERE)}  {len(D):,} frames  "
          f"{out.stat().st_size/1e6:.1f} MB  in {time.time()-t0:.0f}s")
    print(f"  per terrain: " + "  ".join(f"{k} {c}" for k, c in zip(want, counts)))
    print()
    print("  NOTE: the action label is the teacher's CLEAN action, while the")
    print("  action actually EXECUTED carried noise. That is deliberate --")
    print("  the student should learn what the teacher would do here, not")
    print("  reproduce the exploration that got it here.")


if __name__ == "__main__":
    main()
