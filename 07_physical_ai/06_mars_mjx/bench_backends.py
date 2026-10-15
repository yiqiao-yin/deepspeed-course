#!/usr/bin/env python3
"""
Three physics backends on ONE model, and an ablation that explains the gap.

    uv run bench_backends.py                  # the whole thing, ~6 min
    uv run bench_backends.py --skip-mjx       # Warp and CPU only
    uv run bench_backends.py --ablate-only    # just the why

WHAT THIS LAB IS FOR
--------------------
Lab 5 ended needing samples. The obvious next move is GPU physics:
MJX and MuJoCo Warp both advertise tens of thousands of environments
stepping at once, and the published numbers are large enough that
renting a card looks like a formality.

This repository has lost that bet before. `bench_compute.py` in lab 5
asked the same question about the POLICY and found the GPU 1.7x
slower than the CPU, while 3.9x sat unclaimed in a serial `for` loop
and a too-frequent evaluation. So the rule here is to measure the
machine before renting one, and that is all this script does.

THE FALSIFIER, WRITTEN BEFORE THE RUN
-------------------------------------
    GPU physics is worth renting a card for if a GPU backend steps
    this model at >= 5x the CPU worker pool.

Five is not arbitrary: below that, Amdahl eats it. Physics is one
part of a training step, and `--amdahl` measures what fraction,
so a 5x on the physics alone is already a much smaller number end to
end. Anything less and the honest recommendation is to stay on the
CPU, which is a result the lab is willing to publish.

WHY AN ABLATION AND NOT JUST A NUMBER
-------------------------------------
A throughput table that disagrees with the published benchmarks is
not yet a finding -- it could be our card, our batch size, or our
measurement. The ablation turns it into one by changing a single
property of the model at a time: the number of rocks, the height
field's resolution, and finally whether there is a height field at
all. Whichever one moves the number is the explanation.
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

# The bar, fixed before any measurement. See the docstring.
WORTH_RENTING = 5.0


def build_model(seed: int = 0, *, rocks: int | None = None,
                hfield_res: int | None = None, plane: bool = False):
    """
    The lab 5 Mars course, with one property optionally overridden.

    Every knob here exists for the ablation, and each changes exactly
    one thing about the scene so a throughput difference can be
    attributed to it.
    """
    import tempfile

    import mujoco

    sys.path.insert(0, str(LAB5))
    import mars
    from robot import hfield_png, model_xml

    task = mars.make_task(seed, route_band=(6.0, 11.0), min_detour=1.25,
                          forced=True, scale="course")
    if task is None:
        raise SystemExit("no solvable task at that seed")

    h = task.heights - task.heights.min()
    if hfield_res is not None and hfield_res != h.shape[0]:
        # Nearest-neighbour resample. The SURFACE is meant to stay the
        # same shape; only how finely it is tabulated changes.
        idx = np.linspace(0, h.shape[0] - 1, hfield_res).round().astype(int)
        h = h[np.ix_(idx, idx)]
    if plane:
        # A flat height field, not a `plane` geom: that keeps the
        # collider type identical and changes only the GEOMETRY, so a
        # difference cannot be blamed on swapping collision routines.
        h = np.zeros_like(h)

    rock_list = task.rocks if rocks is None else task.rocks[:rocks]
    tmp = Path(tempfile.mkdtemp())
    hfield_png(h, tmp / "t.png")
    xml = model_xml("t.png", task.start, task.goal,
                    relief=max(float(h.max()), 1e-3), rocks=rock_list)
    model = mujoco.MjModel.from_xml_string(
        xml, {"t.png": (tmp / "t.png").read_bytes()})
    return model, task


def mean_contacts(model, steps: int = 300) -> float:
    """
    Contacts per step, on the CPU, with the robot settling under
    gravity.

    This is the MECHANISM behind the throughput ablation, and it is
    measured on the same models rather than argued for: a height field
    is diced into triangles on contact, so relief multiplies the
    number of contact pairs a solver has to handle. Cheap enough to
    report beside every variant, and it makes the throughput column
    checkable against something physical.
    """
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    n = []
    for _ in range(steps):
        mujoco.mj_step(model, d)
        n.append(int(d.ncon))
    return float(np.mean(n))


def cpu_rate(model, steps: int = 300) -> float:
    """One environment, stepped the way lab 5 steps it."""
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for _ in range(20):                      # warm the caches
        mujoco.mj_step(model, d)
    t0 = time.time()
    for _ in range(steps):
        mujoco.mj_step(model, d)
    return steps / (time.time() - t0)


def cpu_pool_rate(workers: int = 16, n_envs: int = 64, flat: bool = False,
                  steps: int = 200) -> float | None:
    """
    The REAL baseline: lab 5's worker pool, not one environment.

    Comparing a GPU against a single CPU environment is the comparison
    that makes every GPU look good, and it is not the one anybody
    faces. Lab 5 already runs 16 processes and proved them
    bit-identical to the serial loop, so that is what a GPU has to
    beat.
    """
    sys.path.insert(0, str(LAB5))
    try:
        from vecenv import ParallelEnvs
    except Exception as e:                                  # noqa: BLE001
        print(f"      (worker pool unavailable: {type(e).__name__})")
        return None

    per = max(1, n_envs // workers)
    pool = ParallelEnvs(workers, per, 0, mode="blind", flat=flat)
    try:
        pool.reset()
        acts = np.zeros((pool.n, pool.act_dim), dtype=np.float64)
        for _ in range(10):                  # spin the processes up
            pool.step(acts)
        t0 = time.time()
        for _ in range(steps):
            pool.step(acts)
        return pool.n * steps / (time.time() - t0)
    finally:
        pool.close()


def mjx_rate(model, n: int, steps: int = 60) -> float | None:
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    d0 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d0)
    mujoco.mj_forward(model, d0)
    try:
        mx = mjx.put_model(model)
        batch = jax.vmap(lambda _: mjx.put_data(model, d0))(jp.arange(n))
        step_v = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        batch = step_v(mx, batch)
        jax.block_until_ready(batch)
        t0 = time.time()
        for _ in range(steps):
            batch = step_v(mx, batch)
        jax.block_until_ready(batch)
        return n * steps / (time.time() - t0)
    except Exception as e:                                  # noqa: BLE001
        print(f"      MJX {n:>5} envs  FAILED  {type(e).__name__}: "
              f"{str(e)[:90]}")
        return None


def warp_rate(model, n: int, steps: int = 60) -> float | None:
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp

    d0 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d0)
    mujoco.mj_forward(model, d0)
    try:
        m = mjwarp.put_model(model)
        d = mjwarp.put_data(model, d0, nworld=n)
        mjwarp.step(m, d)
        wp.synchronize()
        t0 = time.time()
        for _ in range(steps):
            mjwarp.step(m, d)
        wp.synchronize()
        return n * steps / (time.time() - t0)
    except Exception as e:                                  # noqa: BLE001
        print(f"      Warp {n:>5} envs  FAILED  {type(e).__name__}: "
              f"{str(e)[:90]}")
        return None


def amdahl_share(run: str, repeats: int = 3) -> dict | None:
    """
    What fraction of a TRAINING step is physics?

    This is the number that decides whether a physics speedup matters
    at all, and it is the one most often left out of a GPU comparison.

    Both halves come from the SAME configuration, which is the only
    way the ratio means anything: the training rate is lab 5's own
    recorded wall clock for `run`, and the physics rate is measured
    here with that run's worker count, environment count and terrain.
    Measuring the pool at 16 envs and dividing by a 64-env training
    run would invent a speedup out of a mismatch.
    """
    summary = LAB5 / "runs" / run / "summary.json"
    if not summary.exists():
        return None
    try:
        s = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return None
    steps, secs = s.get("total_steps"), s.get("wall_seconds")
    workers, envs = s.get("workers"), s.get("n_envs")
    if not (steps and secs and workers and envs):
        return None

    train = steps / secs
    # Repeats: the pool rate moved 16.4k -> 19.2k between two runs of
    # this script, which swings the share from 33% to 28%. A single
    # sample would publish whichever one it happened to get.
    rates = [r for r in (cpu_pool_rate(workers, envs, flat=s.get("flat", False))
                         for _ in range(repeats)) if r]
    if not rates:
        return None
    phys = float(np.median(rates))
    shares = sorted(train / r for r in rates)
    # Time per step: training = physics + everything else.
    return {"run": run, "workers": workers, "n_envs": envs,
            "flat": s.get("flat", False), "train_steps_per_s": train,
            "physics_steps_per_s": phys, "physics_runs": rates,
            "share": train / phys,
            "share_lo": shares[0], "share_hi": shares[-1]}


def device_report() -> dict:
    """
    What hardware is actually going to run this, asked of each
    backend rather than of `nvidia-smi`.

    A card being present is not the question. The question is whether
    JAX and Warp can see it, and they fail that independently: a torch
    pinned to the wrong CUDA index, a driver older than the wheel, or
    a bare `CUDA_VISIBLE_DEVICES=""` all leave `nvidia-smi` showing a
    healthy GPU while the backend quietly computes on the CPU.
    """
    names, gpu = [], False
    try:
        import jax

        for d in jax.devices():
            names.append(f"jax:{d.platform}")
            gpu |= d.platform in ("gpu", "cuda", "rocm")
    except Exception as e:                                  # noqa: BLE001
        names.append(f"jax:unavailable({type(e).__name__})")
    try:
        import warp as wp

        wp.init()
        d = str(wp.get_preferred_device())
        names.append(f"warp:{d}")
        gpu |= not d.startswith("cpu")
    except Exception as e:                                  # noqa: BLE001
        names.append(f"warp:unavailable({type(e).__name__})")
    return {"summary": ", ".join(names), "gpu": gpu}


def run_isolated(kind: str, n: int, seed: int, steps: int = 60,
                 **kw) -> dict:
    """
    Time one backend at one env count in a SEPARATE PROCESS.

    Not defensiveness for its own sake. MJX on a 16 GB card does not
    raise a catchable Python exception when it runs out of memory at
    1024 environments -- XLA's allocator hits a `Check failed` and
    calls `abort()`, which takes the interpreter down with it. In
    process, that meant one backend failing erased the other
    backend's results and the ablation that explains them.

    Isolating each measurement turns that crash into what it actually
    is: a data point. "MJX died at 1024 envs" is a finding about MJX
    on this hardware, and it belongs in the table rather than in a
    stack trace.
    """
    import subprocess

    spec = json.dumps({"kind": kind, "n": n, "seed": seed,
                       "steps": steps, "kw": kw})
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--child", spec],
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": "timed out after 900 s"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    if p.returncode < 0:
        return {"ok": False, "why": f"process killed by signal "
                                    f"{-p.returncode} (allocator abort)"}
    tail = (p.stderr.strip().splitlines() or ["no output"])[-1][:90]
    return {"ok": False, "why": f"exit {p.returncode}: {tail}"}


def child(spec: str) -> int:
    """The other side of `run_isolated`. Prints one JSON line."""
    s = json.loads(spec)
    model, _ = build_model(s["seed"], **s["kw"])
    fn = {"mjx": mjx_rate, "warp": warp_rate}[s["kind"]]
    try:
        r = fn(model, s["n"], s.get("steps", 60))
    except Exception as e:                                  # noqa: BLE001
        r = None
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
    res = ({"ok": True, "steps_per_s": r, "contacts": mean_contacts(model)}
           if r else {"ok": False, "why": "backend returned no rate"})
    print("__RESULT__" + json.dumps(res))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", default="64,256,1024,4096")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pool-envs", type=int, default=64,
                    help="environments across the pool, as lab 5 runs it")
    ap.add_argument("--amdahl-run", default="v3_flat_blind",
                    help="lab 5 run whose wall clock sets the physics share")
    ap.add_argument("--skip-mjx", action="store_true")
    ap.add_argument("--skip-warp", action="store_true")
    ap.add_argument("--ablate-only", action="store_true")
    ap.add_argument("--ablate-envs", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repeats per ablation variant")
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="timed steps per measurement; lower = cheaper")
    ap.add_argument("--dry-run", action="store_true",
                    help="cheapest path that still exercises every stage: "
                         "one env count, one repeat, few steps. For "
                         "validating a cluster allocation without burning "
                         "it.")
    ap.add_argument("--child", default=None,
                    help=argparse.SUPPRESS)      # internal: see run_isolated
    a, _ = ap.parse_known_args()

    if a.child:
        return child(a.child)

    if a.dry_run:
        # Every stage still runs -- baselines, both backends, the
        # ablation, Amdahl -- just at the smallest size that proves
        # the harness works. A dry run that skips stages validates
        # nothing about the stages it skipped.
        a.envs, a.repeats, a.max_steps, a.ablate_envs = "64", 1, 10, 64
        print("  DRY RUN — 64 envs, 1 repeat, 10 timed steps. "
              "These numbers are NOT results.\n")

    counts = [int(x) for x in a.envs.split(",")]
    out: dict = {"seed": a.seed, "env_counts": counts}

    print("=" * 74)
    print("  GPU PHYSICS FOR THE MARS COURSE — does it pay?")
    print("=" * 74)
    print(f"  falsifier: a GPU backend must reach {WORTH_RENTING:.0f}x the "
          f"CPU worker pool to justify renting a card.\n")

    # WHICH DEVICE, stated before any number.
    #
    # With no CUDA device visible, Warp falls back to CPU and reports
    # throughput perfectly happily. Those numbers look exactly like
    # GPU numbers and are not, which is the quiet-wrongness this
    # course exists to avoid -- so the device is printed, and a
    # CPU-only run is labelled as one rather than left to be inferred.
    devices = device_report()
    print(f"  devices: {devices['summary']}")
    if not devices["gpu"]:
        print("  NOTE: no GPU visible. The 'GPU' backends below will run on "
              "the CPU and\n        their numbers are NOT a GPU "
              "measurement — treat this run as a\n        smoke test of the "
              "harness, not as evidence about hardware.")
    out["devices"] = devices

    model, task = build_model(a.seed)
    print(f"\n  model: {task.extent:.0f} m course, {len(task.rocks)} rocks, "
          f"{model.ngeom} geoms, nq={model.nq} nu={model.nu}\n")

    if not a.ablate_only:
        print("  [1] baselines (CPU)")
        one = cpu_rate(model)
        print(f"      1 env, serial        {one:>10,.0f} steps/s")
        pool = cpu_pool_rate(a.workers, a.pool_envs)
        if pool:
            print(f"      {a.pool_envs} envs / {a.workers} workers "
                  f"{pool:>10,.0f} steps/s   {pool/one:>6.2f}x")
        out["cpu_one"] = one
        out["cpu_pool"] = pool
        ref = pool or one
        ref_name = f"{a.workers}-worker pool" if pool else "1 CPU env"

        for name, kind, skip in (("MJX", "mjx", a.skip_mjx),
                                 ("Warp", "warp", a.skip_warp)):
            if skip:
                continue
            print(f"\n  [{'2' if kind == 'mjx' else '3'}] {name}")
            best, rows = 0.0, {}
            for n in counts:
                res = run_isolated(kind, n, a.seed, a.max_steps)
                if not res.get("ok"):
                    print(f"      {n:>5} envs            FAILED — "
                          f"{res.get('why')}")
                    rows[n] = res
                    break          # larger counts cannot help
                r = res["steps_per_s"]
                best = max(best, r)
                rows[n] = res
                print(f"      {n:>5} envs            {r:>10,.0f} steps/s"
                      f"   {r/ref:>6.2f}x {ref_name}")
            out[f"{kind}_best"] = best
            out[f"{kind}_rows"] = rows

    # ---------------------------------------------------------------
    # The ablation. One property at a time.
    # ---------------------------------------------------------------
    if a.skip_warp:
        print("\n  (ablation needs Warp; skipped)")
    else:
        n = a.ablate_envs
        print(f"\n  [4] why — one property at a time, Warp at {n} envs")
        variants = [
            ("rocks   0", dict(rocks=0)),
            ("rocks  75", dict(rocks=None)),
            ("hfield  45^2", dict(hfield_res=45)),
            ("hfield 180^2", dict(hfield_res=180)),
            ("hfield FLAT", dict(plane=True)),
        ]
        ab = {}
        # Repeats, because one timed call is a sample and not a
        # measurement. The decisive comparison here turns on a ratio
        # near 1, so the SPREAD decides whether there is an effect at
        # all -- reporting a single number would be the whole error.
        print(f"      {'variant':<14} {'steps/s (median of ' + str(a.repeats) + ')':>24}"
              f"  {'spread':>9}  {'contacts':>9}")
        for label, kw in variants:
            runs, contacts = [], None
            for _ in range(a.repeats):
                res = run_isolated("warp", n, a.seed, a.max_steps, **kw)
                if res.get("ok"):
                    runs.append(res["steps_per_s"])
                    contacts = res["contacts"]
            if not runs:
                print(f"      {label:<14} {'FAILED':>24}")
                continue
            med = float(np.median(runs))
            spread = (max(runs) - min(runs)) / med if len(runs) > 1 else 0.0
            ab[label] = {"steps_per_s": med, "runs": runs,
                         "spread": spread, "contacts": contacts}
            print(f"      {label:<14} {med:>24,.0f}  {spread:>8.1%}  "
                  f"{contacts:>9.1f}")
        out["ablation"] = ab
        if "hfield FLAT" in ab and "hfield 180^2" in ab:
            f, g = ab["hfield FLAT"], ab["hfield 180^2"]
            ratio = f["steps_per_s"] / g["steps_per_s"]
            crat = g["contacts"] / max(f["contacts"], 1e-9)
            noise = max(f["spread"], g["spread"])
            print(f"\n      flat vs relief: {ratio:.2f}x throughput for "
                  f"{crat:.1f}x the contacts,")
            print(f"      against a run-to-run spread of {noise:.1%}.")
            if abs(ratio - 1.0) <= noise:
                print("      That difference is INSIDE the noise: on this "
                      "backend the terrain\n      is exonerated — relief is "
                      "not what the throughput is going on.")
            else:
                print("      Relief costs real throughput, but far less than "
                      "the contact count\n      alone would suggest.")
            out["hfield_cost"] = ratio
            out["contact_ratio"] = crat
            out["ablation_noise"] = noise

    am = amdahl_share(a.amdahl_run, a.repeats)
    if am:
        s = am["share"]
        print(f"\n  [5] Amdahl — measured on lab 5 run `{am['run']}` "
              f"({am['workers']} workers, {am['n_envs']} envs, "
              f"{'flat' if am['flat'] else 'Mars relief'})")
        print(f"      training        {am['train_steps_per_s']:>10,.0f} "
              f"steps/s   (its own recorded wall clock)")
        print(f"      physics alone   {am['physics_steps_per_s']:>10,.0f} "
              f"steps/s   (the same pool, measured here)")
        if s < 1.0:
            lo, hi = am["share_lo"], am["share_hi"]
            print(f"      physics is {lo:.0%}-{hi:.0%} of a training step "
                  f"across {a.repeats} repeats, so an\n      INFINITELY fast "
                  f"physics engine gives at most "
                  f"{1/(1-hi):.1f}x end to end.")
        else:
            print("      physics alone measured no faster than training, "
                  "which cannot be:\n      treat the share as unresolved "
                  "rather than reporting a number.")
        out["amdahl"] = am

    # The verdict, against the bar written before the run.
    ref = out.get("cpu_pool") or out.get("cpu_one")
    best = max(out.get("mjx_best") or 0.0, out.get("warp_best") or 0.0)
    if ref and best:
        speedup = best / ref
        verdict = "HOLDS" if speedup >= WORTH_RENTING else "FAILS"
        print(f"\n  falsifier: best GPU backend is {speedup:.2f}x the CPU "
              f"pool, bar was {WORTH_RENTING:.0f}x  ->  {verdict}")
        out["speedup"] = speedup
        out["verdict"] = verdict

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
