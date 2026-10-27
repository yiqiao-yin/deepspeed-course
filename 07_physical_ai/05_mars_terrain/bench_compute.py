#!/usr/bin/env python3
"""
What should we actually rent? Measure first, book second.

    uv run bench_compute.py                 # ~3 minutes, no training
    uv run bench_compute.py --envs 8,16,32,64,128

WHY THIS SCRIPT EXISTS BEFORE ANY GPU IS BOOKED
------------------------------------------------
Part 4 ended needing more compute, and the obvious move -- rent a GPU
box and run the trainer on it -- is measurably wrong. The policy here
is 12,175 parameters. Benchmarked locally:

    batch 16     CPU  55 us     CUDA  255 us    GPU 4.6x SLOWER
    batch 4096   CPU 507 us     CUDA  262 us    GPU 1.9x faster

At the batch size PPO actually uses during rollout, the GPU loses to
kernel-launch overhead, and the forward pass is 55 us against ~277 us
of MuJoCo physics per step. Making the network infinitely fast would
buy about 1.2x.

So the question is not "GPU or not". It is **where the wall-clock
goes**, and that has three possible answers with three different
machines behind them:

    physics-bound, parallelises   -> rent CORES, not a GPU
    physics-bound, does not       -> rewrite for MJX, then rent a GPU
    update-bound                  -> bigger batches, then a GPU helps

This script answers it in three minutes on whatever box it is run on,
and prints the projected hours for a 10M-step run. Run it locally,
run it on the pod for 15 minutes before committing to anything long,
and compare. A rented box that turns out to be slower than the
laptop is a thing that happens, and it is cheap to find out.

NOTHING HERE TRAINS. It cannot produce a result, only a rate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def machine() -> dict:
    """What we are actually running on. Printed so a log is attributable."""
    info = {"platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:                                  # noqa: BLE001
        info["torch"] = f"unavailable: {e}"
    return info


def bench_physics(n_envs: int, steps: int = 400) -> float:
    """
    Env steps per second with `n_envs` environments stepped in sequence.

    This is the quantity that decides everything. PPO collects
    `n_envs x rollout` transitions per iteration by stepping each
    environment, and MuJoCo runs on the CPU, so this is where the
    wall-clock lives.
    """
    sys.path.insert(0, str(HERE))
    from mars_env import NavWorld

    envs = [NavWorld(mode="blind", seed=i, upright_w=1.0)
            for i in range(n_envs)]
    for i, e in enumerate(envs):
        e.reset(seed=1000 + i)
    a = np.zeros(envs[0].act_dim)

    for e in envs[:2]:                      # warm the caches
        for _ in range(20):
            e.step(a)

    t0 = time.time()
    for _ in range(steps):
        for e in envs:
            e.step(a)
    dt = time.time() - t0
    return n_envs * steps / dt


def bench_update(batch: int, hidden: int = 64, iters: int = 40) -> dict:
    """One PPO-shaped forward+backward, CPU and (if present) GPU."""
    import torch

    sys.path.insert(0, str(HERE))
    from ppo import ActorCritic

    out = {}
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        net = ActorCritic(25, 7, hidden=hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        x = torch.randn(batch, 25, device=dev)
        act = torch.randn(batch, 7, device=dev)
        for _ in range(5):
            lp, ent, v = net.evaluate(x, act)
            (lp.mean() + v.mean()).backward()
            opt.step(); opt.zero_grad()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            lp, ent, v = net.evaluate(x, act)
            (lp.mean() + v.mean()).backward()
            opt.step(); opt.zero_grad()
        if dev == "cuda":
            torch.cuda.synchronize()
        out[dev] = (time.time() - t0) / iters * 1e3     # ms
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", default="4,16,32,64",
                    help="comma-separated environment counts to try")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--target-steps", type=int, default=10_000_000,
                    help="the run we are costing, in env steps")
    ap.add_argument("--out", default=None)
    a, _ = ap.parse_known_args()

    info = machine()
    print("=" * 78)
    print("  WHAT SHOULD WE RENT?  (nothing here trains)")
    print("=" * 78)
    for k, v in info.items():
        print(f"  {k:<12} {v}")

    print("\n  -- PHYSICS: env steps/s as environments are added --")
    print(f"  {'n_envs':>7}{'steps/s':>12}{'per-env':>10}{'scaling':>10}")
    rows, base = [], None
    for n in [int(x) for x in a.envs.split(",")]:
        r = bench_physics(n, a.steps)
        base = base or r
        rows.append({"n_envs": n, "steps_per_s": r})
        print(f"  {n:>7}{r:>12,.0f}{r/n:>10,.0f}{r/base:>9.2f}x")

    print("\n  -- POLICY UPDATE: one PPO minibatch, ms --")
    upd = {}
    for b in (256, 4096):
        u = bench_update(b)
        upd[b] = u
        cpu = u["cpu"]
        gpu = u.get("cuda")
        line = f"  batch {b:<6} cpu {cpu:7.2f} ms"
        if gpu:
            line += f"   cuda {gpu:7.2f} ms   {'GPU ' + format(cpu/gpu, '.1f') + 'x faster' if gpu < cpu else 'CPU ' + format(gpu/cpu, '.1f') + 'x faster'}"
        print(line)

    best = max(rows, key=lambda r: r["steps_per_s"])
    hours = a.target_steps / best["steps_per_s"] / 3600
    print("\n" + "=" * 78)
    print(f"  best physics rate: {best['steps_per_s']:,.0f} steps/s "
          f"at n_envs={best['n_envs']}")
    print(f"  {a.target_steps:,} steps would take {hours:.1f} h "
          f"on this machine, per arm")
    print("=" * 78)
    print("""
  HOW TO READ THIS

  Physics scaling near 1.0x as n_envs grows means stepping is
  serialised on one core: adding environments buys nothing, and the
  machine to rent is one with FASTER cores, or the code needs MJX.

  Physics scaling that keeps climbing means more cores help, and the
  machine to rent is a many-core CPU box -- a GPU would sit idle.

  The update row decides whether a GPU is worth anything at all. If
  CPU wins at batch 256, the rollout phase gains nothing from a GPU;
  only very large minibatches would.
""")
    if a.out:
        (HERE / a.out).write_text(json.dumps(
            {"machine": info, "physics": rows,
             "update_ms": {str(k): v for k, v in upd.items()},
             "target_steps": a.target_steps,
             "projected_hours": hours}, indent=2))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
