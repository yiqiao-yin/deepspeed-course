#!/usr/bin/env python3
"""
Where does a training step actually go?

    uv run profile_training.py                      # the breakdown
    uv run profile_training.py --morphology biped

WHY THIS COMES FIRST
--------------------
Lab 9 wants to TRAIN MANY BODIES, which is only affordable if
training is cheap. Lab 6 measured where the money goes and the answer
moved between bodies:

    biped     physics 59% of a training step
    spider8   physics 21%

So for the bodies that actually walk, four fifths of the time is
NOT physics -- it is the learner and the plumbing around it. That is
good news, because the learner is ours to fix and the simulator is
not. But "the learner" is three or four different things and
optimising the wrong one is free disappointment.

This measures them separately, with the same discipline lab 5's
`bench_compute.py` used to find 3.9x sitting unclaimed in a serial
`for` loop: time the parts, then attack the biggest.

WHAT IS TIMED
-------------
    env step        MuJoCo physics, across the worker pool
    policy forward  action selection during rollout (no grad)
    GAE             advantage computation, numpy
    PPO update      the backward passes, which is the only part
                    where a GPU could plausibly help
    obs normalise   the running mean/var update

Each is timed in isolation at the batch sizes the real loop uses, so
the numbers add up to something comparable with a real step rather
than to a microbenchmark nobody runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morphology", default="spider8")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a, _ = ap.parse_known_args()

    import torch

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm
    from vecenv import ParallelEnvs

    probe = NavWorld(mode="blind", flat=False, seed=0,
                     morphology=a.morphology)
    obs_dim, act_dim = probe.obs_dim, probe.act_dim
    print("=" * 74)
    print(f"  WHERE A TRAINING STEP GOES — {a.morphology}, "
          f"obs {obs_dim}, act {act_dim}")
    print("=" * 74)

    net = ActorCritic(obs_dim, act_dim, hidden=a.hidden).to(a.device)
    norm = RunningNorm(obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    per = max(1, a.n_envs // a.workers)
    pool = ParallelEnvs(a.workers, per, 0, mode="blind", flat=False,
                        morphology=a.morphology)
    n = pool.n
    times: dict = {}

    try:
        obs = pool.reset()
        acts = np.zeros((n, act_dim))
        for _ in range(5):                      # spin up
            pool.step(acts)

        # ---- 1. physics -------------------------------------------
        t0 = time.time()
        for _ in range(a.rollout):
            obs, _r, _d = pool.step(acts)
        times["env step"] = time.time() - t0

        # ---- 2. policy forward, as the rollout does it ------------
        x = torch.as_tensor(norm(obs), dtype=torch.float32)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(a.rollout):
                net.distribution(x).sample()
        times["policy forward"] = time.time() - t0

        # ---- 3. observation normalisation -------------------------
        t0 = time.time()
        for _ in range(a.rollout):
            norm.update(obs)
            norm(obs)
        times["obs normalise"] = time.time() - t0

        # ---- 4. GAE -----------------------------------------------
        rew = np.random.randn(a.rollout, n).astype(np.float32)
        val = np.random.randn(a.rollout + 1, n).astype(np.float32)
        done = np.zeros((a.rollout, n), dtype=bool)
        t0 = time.time()
        adv = np.zeros_like(rew)
        last = np.zeros(n, dtype=np.float32)
        for t in reversed(range(a.rollout)):
            nz = 1.0 - done[t]
            delta = rew[t] + 0.99 * val[t + 1] * nz - val[t]
            last = delta + 0.99 * 0.95 * nz * last
            adv[t] = last
        times["GAE"] = time.time() - t0

        # ---- 5. the PPO update ------------------------------------
        N = a.rollout * n
        bo = torch.randn(N, obs_dim, device=a.device)
        ba = torch.randn(N, act_dim, device=a.device)
        bl = torch.randn(N, device=a.device)
        bA = torch.randn(N, device=a.device)
        bR = torch.randn(N, device=a.device)
        mb = N // a.minibatches
        t0 = time.time()
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=a.device)
            for k in range(a.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                lp, ent, v = net.evaluate(bo[idx], ba[idx])
                ratio = (lp - bl[idx]).exp()
                loss = (-torch.min(ratio * bA[idx],
                                   ratio.clamp(0.8, 1.2) * bA[idx]).mean()
                        + 0.5 * ((v - bR[idx]) ** 2).mean()
                        - 0.001 * ent.mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        times["PPO update"] = time.time() - t0
    finally:
        pool.close()

    total = sum(times.values())
    steps = a.rollout * n
    print(f"\n  one iteration = {a.rollout} rollout steps x {n} envs "
          f"= {steps:,} transitions\n")
    print(f"  {'stage':<18} {'seconds':>9} {'share':>8}")
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:>9.2f} {v/total:>7.1%}")
    print(f"  {'TOTAL':<18} {total:>9.2f}")
    print(f"\n  throughput        {steps/total:>9,.0f} transitions/s")

    big = max(times, key=times.get)
    print(f"\n  biggest cost: {big} at {times[big]/total:.0%}.")
    if big == "env step":
        print("  Physics-bound. More CPU workers, or revisit GPU physics.")
    else:
        print("  NOT physics-bound. More CPU workers will not help much;")
        print("  the win is in the learner, which is ours to change.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"morphology": a.morphology, "obs_dim": obs_dim,
             "act_dim": act_dim, "n_envs": n, "rollout": a.rollout,
             "times": times, "total": total,
             "transitions_per_s": steps / total}, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
