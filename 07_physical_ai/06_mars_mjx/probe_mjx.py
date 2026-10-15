#!/usr/bin/env python3
"""
Can MJX run OUR terrain, and how fast? Measure before building a lab.

    uv run probe_mjx.py              # the feasibility gate
    uv run probe_mjx.py --envs 4096

WHY THIS RUNS BEFORE ANY LAB CODE
----------------------------------
Lab 5 ended needing samples, not a bigger network: 10M steps changed
nothing because the robot falls on most episodes and never collects a
usable trajectory. MJX is the honest answer -- GPU physics, thousands
of environments -- but it is also a rewrite, and this repository has a
standing rule against paying for a thing before measuring it.

Three questions, in the order that can kill the lab fastest:

  1. Does MJX accept the model AT ALL? Our terrain is an `hfield` with
     box rocks on it. MJX implements HFIELD, BOX, CAPSULE and MESH, so
     this should work -- but "should" is what `--check` exists to
     replace. CYLINDER collides only with primitives in MJX, and our
     goalpost is a cylinder; it is `contype="0"` so it should not
     matter, and that is also checked rather than assumed.

  2. Does it produce the SAME physics as the CPU engine? A faster
     simulator that behaves differently is not a speedup, it is a
     different experiment. Lab 5's `vecenv.py` had to prove
     bit-identity against the serial loop before anyone trusted it;
     MJX cannot be bit-identical -- different solver implementation,
     float32 by default -- so the bar here is "close enough that a
     policy trained in one works in the other", measured as
     trajectory divergence rather than asserted.

  3. Is it actually faster ON THIS MODEL? MJX's headline numbers come
     from simple scenes. Ours has a 180x180 height field and 75
     colliding boxes, and MuJoCo caps height-field contacts at 50 per
     geom pair (mjMAXCONPAIR). A scene that spends its time in
     contact generation may not vectorise the way a published
     benchmark does.

Nothing here trains. It produces a go/no-go and a throughput number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"


def build_cpu_model(seed: int = 0):
    """The Mars course model, exactly as lab 5 builds it."""
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
    tmp = Path(tempfile.mkdtemp())
    hfield_png(h, tmp / "t.png")
    xml = model_xml("t.png", task.start, task.goal,
                    relief=max(float(h.max()), 1e-3), rocks=task.rocks)
    model = mujoco.MjModel.from_xml_string(
        xml, {"t.png": (tmp / "t.png").read_bytes()})
    return model, task


def q1_does_it_load(model) -> bool:
    """Question 1: will MJX accept this model?"""
    import mujoco
    from mujoco import mjx

    print("  [1] does MJX accept the model?")
    kinds = {}
    for g in range(model.ngeom):
        t = int(model.geom_type[g])
        kinds[mujoco.mjtGeom(t).name] = kinds.get(mujoco.mjtGeom(t).name, 0) + 1
    print(f"      geoms: {kinds}")
    try:
        mx = mjx.put_model(model)
        print(f"      PASS   model on device, nq={model.nq} nu={model.nu} "
              f"ngeom={model.ngeom}")
        return mx
    except Exception as e:                                  # noqa: BLE001
        print(f"      FAIL   {type(e).__name__}: {str(e)[:200]}")
        return None


def q2_same_physics(model, mx, steps: int = 300) -> None:
    """
    Question 2: how far do CPU and GPU trajectories diverge?

    Not asserted equal -- MJX uses a different solver implementation
    and float32, so exact agreement is not on offer. What matters is
    whether a policy trained on one transfers to the other, and the
    proxy for that is how fast an identical open-loop rollout drifts
    apart.
    """
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    print("  [2] does it produce the same physics?")
    d_cpu = mujoco.MjData(model)
    mujoco.mj_resetData(model, d_cpu)
    mujoco.mj_forward(model, d_cpu)

    dx = mjx.put_data(model, d_cpu)
    step_fn = jax.jit(mjx.step)

    ctrl = np.zeros(model.nu)
    drift = []
    for i in range(steps):
        d_cpu.ctrl[:] = ctrl
        mujoco.mj_step(model, d_cpu)
        dx = dx.replace(ctrl=jp.array(ctrl))
        dx = step_fn(mx, dx)
        if (i + 1) % 100 == 0:
            gq = np.asarray(dx.qpos)
            drift.append(float(np.abs(gq - d_cpu.qpos).max()))
            print(f"      after {i+1:>4} steps: max |qpos difference| "
                  f"{drift[-1]:.2e}")
    if drift and drift[-1] < 0.05:
        print("      PASS   trajectories stay close enough to transfer")
    else:
        print("      NOTE   they diverge; a policy may need retraining "
              "rather than porting")


def q3_throughput(model, mx, env_counts: list[int], steps: int = 60) -> None:
    """Question 3: is it faster ON THIS MODEL, and how does it scale?"""
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    print("  [3] throughput")
    # CPU reference: one environment, the way lab 5 runs it.
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for _ in range(20):
        mujoco.mj_step(model, d)
    t0 = time.time()
    for _ in range(300):
        mujoco.mj_step(model, d)
    cpu_rate = 300 / (time.time() - t0)
    print(f"      CPU, 1 env           {cpu_rate:>10,.0f} steps/s")

    d0 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d0)
    mujoco.mj_forward(model, d0)

    step_v = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    for n in env_counts:
        try:
            batch = jax.vmap(lambda _: mjx.put_data(model, d0))(jp.arange(n))
            batch = step_v(mx, batch)          # compile
            jax.block_until_ready(batch)
            t0 = time.time()
            for _ in range(steps):
                batch = step_v(mx, batch)
            jax.block_until_ready(batch)
            dt = time.time() - t0
            rate = n * steps / dt
            print(f"      MJX, {n:>5} envs      {rate:>10,.0f} steps/s"
                  f"   {rate/cpu_rate:>7.1f}x CPU")
        except Exception as e:                              # noqa: BLE001
            print(f"      MJX, {n:>5} envs      FAILED  "
                  f"{type(e).__name__}: {str(e)[:120]}")
            break


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", default="64,256,1024,4096")
    ap.add_argument("--seed", type=int, default=0)
    a, _ = ap.parse_known_args()

    import jax

    print("=" * 74)
    print("  MJX FEASIBILITY GATE — nothing here trains")
    print("=" * 74)
    print(f"  jax {jax.__version__}  devices {jax.devices()}")

    model, task = build_cpu_model(a.seed)
    print(f"  lab 5 course: {task.extent:.0f} m arena, route "
          f"{task.route_len:.1f} m, {len(task.rocks)} rocks, "
          f"{len(task.landforms)} landforms\n")

    mx = q1_does_it_load(model)
    if mx is None:
        print("\n  STOP. MJX cannot run this model; the lab needs a "
              "different terrain representation.")
        return 1
    print()
    q2_same_physics(model, mx)
    print()
    q3_throughput(model, mx, [int(x) for x in a.envs.split(",")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
