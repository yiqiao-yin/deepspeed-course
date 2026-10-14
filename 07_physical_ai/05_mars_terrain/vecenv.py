#!/usr/bin/env python3
"""
Step the environments in parallel processes instead of a Python loop.

    uv run vecenv.py            # correctness + speed check

WHY: THE MACHINE WAS 95% IDLE
------------------------------
`train_mars.py` held 16 environments and stepped them in a `for` loop
in one process. MuJoCo releases the GIL for parts of a step but the
loop itself is serial, so on a 20-core box nineteen cores did nothing.
Measured with `bench_compute.py`:

    n_envs    4     16     32     64
    steps/s  1614  1950   1978   2202      <- 16x the envs, 1.36x the work

That is the signature of serialisation, and it is why Part 4 needed
"more compute": the compute was already there.

Split across processes instead:

    workers    1      4      8     16
    steps/s  1708   6589  11352  15949      <- 9.3x

Same box, same code, no GPU, no money. A 10M-step run goes from about
three hours to roughly twenty minutes.

WHAT THIS IS NOT
-----------------
It is not a speedup of the physics -- each env steps at exactly the
rate it did before. It is the removal of an accidental bottleneck,
which is the cheapest kind of compute there is and the reason to
measure before renting anything. `bench_compute.py` exists to make
that the first move on any new machine.

CORRECTNESS BEFORE SPEED
-------------------------
A fast environment that behaves differently from the slow one is
worse than no speedup, because every result afterwards is suspect and
the difference is invisible in a learning curve. `main()` below
asserts that the parallel pool and the serial loop produce
BIT-IDENTICAL observations and rewards over 200 steps from the same
seeds, and that is the check to run after touching this file.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def _worker(conn, n_envs: int, seeds: list[int], kwargs: dict) -> None:
    """
    One process holding `n_envs` environments.

    Imports happen HERE rather than at module scope: the pool is
    spawned, not forked, so each worker builds its own MuJoCo models
    and loads its own copy of the task pool. That costs a few seconds
    at startup and buys process isolation -- a segfault in one MuJoCo
    model takes down one worker, not the run.
    """
    sys.path.insert(0, str(HERE))
    from mars_env import NavWorld

    envs = [NavWorld(seed=s, **kwargs) for s in seeds]
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                out = [e.reset(seed=sd)[0] for e, sd in zip(envs, payload)]
                conn.send(np.stack(out))
            elif cmd == "step":
                obs, rew, done = [], [], []
                for e, a in zip(envs, payload):
                    o2, r, te, tr, _ = e.step(a)
                    if te or tr:
                        # Auto-reset, exactly as the serial loop does:
                        # the observation returned is the FIRST of the
                        # next episode, and `done` marks the boundary
                        # so GAE does not bootstrap across it.
                        o2, _ = e.reset()
                    obs.append(o2); rew.append(r); done.append(float(te or tr))
                conn.send((np.stack(obs), np.array(rew), np.array(done)))
            elif cmd == "spec":
                conn.send((envs[0].obs_dim, envs[0].act_dim,
                           envs[0].max_steps))
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

    The interface is deliberately the same shape as the serial loop it
    replaces -- `reset()` returns stacked observations, `step(actions)`
    returns `(obs, rewards, dones)` -- so the training loop changes by
    two lines rather than being rewritten around a new abstraction.
    """

    def __init__(self, n_workers: int, per_worker: int, base_seed: int,
                 **env_kwargs) -> None:
        ctx = mp.get_context("spawn")
        self.n = n_workers * per_worker
        self.per = per_worker
        self._conns, self._procs = [], []
        k = 0
        for w in range(n_workers):
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
        self.obs_dim, self.act_dim, self.max_steps = self._conns[0].recv()

    def reset(self, seed: int = 0) -> np.ndarray:
        for w, c in enumerate(self._conns):
            c.send(("reset", [seed + w * self.per + i
                              for i in range(self.per)]))
        return np.concatenate([c.recv() for c in self._conns])

    def step(self, actions: np.ndarray):
        # Send to every worker FIRST, then collect. Sending and
        # receiving one worker at a time would serialise them again
        # and undo the whole point.
        for w, c in enumerate(self._conns):
            c.send(("step", actions[w * self.per:(w + 1) * self.per]))
        obs, rew, done = [], [], []
        for c in self._conns:
            o, r, d = c.recv()
            obs.append(o); rew.append(r); done.append(d)
        return (np.concatenate(obs), np.concatenate(rew),
                np.concatenate(done))

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
    import time

    sys.path.insert(0, str(HERE))
    from mars_env import NavWorld

    kw = dict(mode="blind", upright_w=1.0)
    N = 8

    print("=" * 74)
    print("  PARALLEL ENVS: identical to the serial loop, then faster")
    print("=" * 74)

    serial = [NavWorld(seed=100 + i, **kw) for i in range(N)]
    so = np.stack([e.reset(seed=500 + i)[0] for i, e in enumerate(serial)])
    par = ParallelEnvs(n_workers=4, per_worker=2, base_seed=100, **kw)
    po = par.reset(seed=500)

    ok = np.allclose(so, po, atol=0, rtol=0)
    print(f"  reset observations bit-identical       {'PASS' if ok else 'FAIL'}"
          f"   max |diff| {np.abs(so - po).max():.2e}")

    rng = np.random.default_rng(0)
    worst_o = worst_r = 0.0
    for t in range(200):
        acts = rng.uniform(-1, 1, (N, serial[0].act_dim))
        s_o, s_r = [], []
        for e, a in zip(serial, acts):
            o2, r, te, tr, _ = e.step(a)
            if te or tr:
                o2, _ = e.reset()
            s_o.append(o2); s_r.append(r)
        p_o, p_r, _ = par.step(acts)
        worst_o = max(worst_o, float(np.abs(np.stack(s_o) - p_o).max()))
        worst_r = max(worst_r, float(np.abs(np.array(s_r) - p_r).max()))
    print(f"  200 steps, observations identical      "
          f"{'PASS' if worst_o == 0 else 'FAIL'}   max |diff| {worst_o:.2e}")
    print(f"  200 steps, rewards identical           "
          f"{'PASS' if worst_r == 0 else 'FAIL'}   max |diff| {worst_r:.2e}")
    par.close()

    print("\n  -- throughput --")
    a0 = np.zeros((N, serial[0].act_dim))
    t0 = time.time()
    for _ in range(150):
        for e in serial:
            e.step(a0[0])
    ser = N * 150 / (time.time() - t0)
    print(f"  serial, {N} envs           {ser:>9,.0f} steps/s")
    for nw, pw in ((4, 4), (8, 4), (16, 4)):
        p = ParallelEnvs(n_workers=nw, per_worker=pw, base_seed=0, **kw)
        p.reset(seed=0)
        acts = np.zeros((nw * pw, p.act_dim))
        for _ in range(10):
            p.step(acts)
        t0 = time.time()
        for _ in range(150):
            p.step(acts)
        r = nw * pw * 150 / (time.time() - t0)
        print(f"  {nw:>2} workers x {pw} envs      {r:>9,.0f} steps/s"
              f"   {r/ser:>5.1f}x")
        p.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
