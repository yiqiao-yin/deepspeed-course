#!/usr/bin/env python3
"""
Step the environments in parallel processes instead of a Python loop.

    uv run vecenv.py            # correctness FIRST, then speed

WHY
---
Topic 7 measured this and the number was embarrassing: sixteen
environments stepped in a `for` loop ran at 1.36x the speed of one,
because MuJoCo releases the GIL for parts of a step but the loop
itself is serial. On a twenty-core box, nineteen cores did nothing.
Splitting across processes gave 2.5x, and raising the evaluation
interval gave another 1.6x, for 3.9x free on hardware already owned
-- more than the ~1.40x eight rented GPUs were measured to buy.

That is the first move on any new machine, and it is why this lab
never rents anything.

CORRECTNESS BEFORE SPEED
-------------------------
A fast environment that behaves differently from the slow one is
worse than no speedup: every result afterwards is suspect, and the
difference is invisible in a learning curve. `main()` asserts the
pool and the serial loop produce BIT-IDENTICAL observations and
rewards from the same seeds, and that is the check to run after
touching this file.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def _worker(conn, n_envs: int, seeds: list[int], kwargs: dict) -> None:
    """
    One process holding `n_envs` environments.

    Imports happen HERE rather than at module scope: the pool is
    spawned, not forked, so each worker builds its own MuJoCo models.
    That costs a few seconds at startup and buys process isolation --
    a segfault in one model takes down one worker, not the run.
    """
    sys.path.insert(0, str(HERE))
    from target_env import TargetEnv

    envs = [TargetEnv(seed=s, **kwargs) for s in seeds]
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(np.stack([e.reset(seed=sd)
                                    for e, sd in zip(envs, payload)]))
            elif cmd == "step":
                obs, rew, done, info = [], [], [], []
                for e, a in zip(envs, payload):
                    o2, r, d, _ = e.step(a)
                    if d:
                        # Report the outcome AND whether the roll
                        # completed. They are different questions and
                        # the lab's whole finding is that they come
                        # apart: a rig can roll reliably and land
                        # rarely. Logging only the outcome string
                        # makes that invisible during training.
                        info.append(f"{e.outcome}|{e.miss():.3f}")
                        o2 = e.reset()
                    else:
                        info.append("")
                    obs.append(o2)
                    rew.append(r)
                    done.append(float(d))
                conn.send((np.stack(obs), np.array(rew), np.array(done), info))
            elif cmd == "spec":
                conn.send((envs[0].obs_dim, envs[0].act_dim,
                           envs[0].max_steps, envs[0].rig))
            elif cmd == "close":
                conn.send(True)
                return
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        conn.close()


class ParallelEnvs:
    """
    `n_workers` processes holding `per_worker` environments each.

    The interface is deliberately the same shape as the serial loop
    it replaces, so the training loop changes by two lines rather
    than being rewritten around a new abstraction.
    """

    def __init__(self, n_workers: int, per_worker: int, base_seed: int,
                 **env_kwargs) -> None:
        ctx = mp.get_context("spawn")
        self.n = n_workers * per_worker
        self.per = per_worker
        self._conns, self._procs = [], []
        k = 0
        for _w in range(n_workers):
            seeds = [base_seed + k + i for i in range(per_worker)]
            k += per_worker
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker,
                            args=(child, per_worker, seeds, env_kwargs),
                            daemon=True)
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)
        self._conns[0].send(("spec", None))
        self.obs_dim, self.act_dim, self.max_steps, self.rig = \
            self._conns[0].recv()

    def reset(self, seed: int = 0) -> np.ndarray:
        for w, c in enumerate(self._conns):
            c.send(("reset", [seed + w * self.per + i
                              for i in range(self.per)]))
        return np.concatenate([c.recv() for c in self._conns])

    def step(self, actions: np.ndarray):
        # Send to EVERY worker first, then collect. Sending and
        # receiving one at a time would serialise them again and undo
        # the whole point.
        for w, c in enumerate(self._conns):
            c.send(("step", actions[w * self.per:(w + 1) * self.per]))
        obs, rew, done, info = [], [], [], []
        for c in self._conns:
            o, r, d, i = c.recv()
            obs.append(o); rew.append(r); done.append(d); info.extend(i)
        return (np.concatenate(obs), np.concatenate(rew),
                np.concatenate(done), info)

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(("close", None)); c.recv()
            except Exception:                               # noqa: BLE001
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def main() -> int:
    """Correctness against the serial loop, then speed."""
    sys.path.insert(0, str(HERE))
    from target_env import TargetEnv

    print("=" * 74)
    print("  VECENV — bit-identity first, speed second")
    print("=" * 74)

    kw = dict(rig_name="default", preset="rolling")
    steps, n = 150, 4

    serial = [TargetEnv(seed=s, **kw) for s in range(n)]
    so = np.stack([e.reset(seed=s) for e, s in zip(serial, range(n))])
    rng = np.random.default_rng(0)
    acts = [rng.uniform(0, 0.6, (n, serial[0].act_dim)) for _ in range(steps)]

    s_obs, s_rew = [], []
    t0 = time.time()
    for a in acts:
        o, r = [], []
        for e, ai in zip(serial, a):
            o2, ri, d, _ = e.step(ai)
            if d:
                o2 = e.reset()
            o.append(o2); r.append(ri)
        s_obs.append(np.stack(o)); s_rew.append(np.array(r))
    t_serial = time.time() - t0

    par = ParallelEnvs(n_workers=n, per_worker=1, base_seed=0, **kw)
    po = par.reset(0)
    p_obs, p_rew = [], []
    t0 = time.time()
    for a in acts:
        o, r, _d, _i = par.step(a)
        p_obs.append(o); p_rew.append(r)
    t_par = time.time() - t0

    d_reset = float(np.abs(so - po).max())
    d_obs = max(float(np.abs(a - b).max()) for a, b in zip(s_obs, p_obs))
    d_rew = max(float(np.abs(a - b).max()) for a, b in zip(s_rew, p_rew))
    ok = d_reset == 0.0 and d_obs == 0.0 and d_rew == 0.0
    print(f"  reset   max |serial - parallel| = {d_reset:g}")
    print(f"  obs     max |serial - parallel| = {d_obs:g}  "
          f"over {steps} steps")
    print(f"  rew     max |serial - parallel| = {d_rew:g}")
    print(f"  -> {'BIT-IDENTICAL' if ok else 'DIVERGED — do not trust it'}")
    par.close()

    if not ok:
        return 1

    import os

    cores = os.cpu_count() or 4
    print(f"\n  {'workers':>8} {'steps/s':>10} {'speedup':>8}")
    base = None
    for w in (1, 4, min(8, cores), min(16, cores)):
        if base is not None and w <= 1:
            continue
        p = ParallelEnvs(n_workers=w, per_worker=1, base_seed=0, **kw)
        p.reset(0)
        a = np.full((w, p.act_dim), 0.4)
        for _ in range(20):
            p.step(a)
        t0 = time.time()
        for _ in range(120):
            p.step(a)
        rate = 120 * w / (time.time() - t0)
        base = base or rate
        print(f"  {w:>8} {rate:>10,.0f} {rate / base:>7.2f}x")
        p.close()
    print(f"\n  ({cores} cores visible. Serial reference above: "
          f"{steps * n / t_serial:,.0f} steps/s; the 4-worker pool ran "
          f"{steps * n / t_par:,.0f}.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
