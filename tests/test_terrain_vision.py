#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2"]
# ///
"""
The head is cosmetic, and the camera must never see it.

`07_physical_ai/03_terrain_vision` grew its robot a large BD-1 head for
looks. That is a dangerous kind of change, because the head is not a
decal: it is a rigid body carrying 19% of the robot's mass AND the depth
camera the whole lab is about. Two failure modes follow, and neither
announces itself.

  MASS.   MuJoCo derives mass from geom volume, so enlarging the box
          multiplies its mass. The policy would still run, still look
          fine, and quietly be driving a different robot than the one
          1.5M training steps were spent on.

  VISION. The camera is mounted INSIDE this body. Decoration that
          reaches into its field of view puts a constant blob in every
          depth frame the student ever receives. Training would still
          converge, the animation would look better than before, and
          the lab's central measurement would be wrong.

The vision check here is GEOMETRIC, not a render comparison, which makes
it both exact and free of any OpenGL dependency: the decoration is
rigidly attached to the camera, so if every vertex of every head geom
lies behind the camera's view plane in the head frame, no head geom can
appear in any frame, in any pose, ever. A render comparison could only
ever sample the poses it happened to try.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "07_physical_ai" / "03_terrain_vision"

# Measured from the model BEFORE the head was enlarged, and pinned in the
# lab's `<inertial>` element. If a future edit changes the robot's
# dynamics, these are what notices.
HEAD_MASS = 4.252544
TOTAL_MASS = 22.686363
HEAD_INERTIA = (0.02797353, 0.02100553, 0.01533584)

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def head_geoms(model, mujoco):
    """Every geom belonging to the `head` body."""
    h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    return h, [g for g in range(model.ngeom) if model.geom_bodyid[g] == h]


def camera_frame(model, mujoco):
    """
    The camera's origin and viewing direction, in the HEAD body frame.

    MuJoCo cameras look down their own -z axis. `cam_mat0` is not usable
    here because it is a world-frame quantity computed at the model's
    reference pose; `cam_quat` is the local orientation we want.
    """
    c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    pos = model.cam_pos[c].copy()
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.cam_quat[c])
    return pos, -mat.reshape(3, 3)[:, 2]          # -z column = view direction


def worst_intrusion(model, mujoco, geoms) -> tuple[float, int]:
    """
    How far the most forward head vertex reaches past the camera plane.

    Negative is safe. The number returned is in metres and is the signed
    distance of the worst corner of the worst geom, measured along the
    viewing direction -- so it is also the margin a future edit has to
    play with before the decoration shows up in the data.
    """
    cam_pos, view = camera_frame(model, mujoco)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)], dtype=float)

    worst, which = -np.inf, -1
    for g in geoms:
        # geom_aabb is (center, half-extent) in the geom's own frame, and
        # is exact for a box and tight for a capsule or cylinder.
        c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        R = model.geom_quat[g]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, R)
        mat = mat.reshape(3, 3)
        pts = (mat @ (c + corners * half).T).T + model.geom_pos[g]
        d = float(np.max((pts - cam_pos) @ view))
        if d > worst:
            worst, which = d, g
    return worst, which


def check_bootstrap() -> None:
    """Exercise `ensure_prerequisites` in all three states, without running it."""
    import types
    from unittest import mock

    sys.path.insert(0, str(LAB))
    import train_student                                        # noqa: E402

    def calls(has_teacher: bool, has_data: bool) -> list[str]:
        """Which scripts does the bootstrap decide to run?"""
        ran: list[str] = []
        args = types.SimpleNamespace(data="data/bc.npz")

        class FakePath:
            """Just enough Path to answer `/`, `glob` and `exists`."""

            def __init__(self, name: str = "", exists: bool = True) -> None:
                self._name, self._exists = name, exists

            def __truediv__(self, other):
                o = str(other)
                return FakePath(o, has_data if o.endswith(".npz") else True)

            def __str__(self) -> str:
                return self._name

            def glob(self, _pattern):
                return iter([1] if has_teacher else [])

            def exists(self) -> bool:
                return self._exists

        with mock.patch.object(train_student, "HERE", FakePath()), \
             mock.patch("subprocess.run",
                        side_effect=lambda c, **k: ran.append(str(c[1]))):
            train_student.ensure_prerequisites(args)
        return ran

    both, data_only, neither = (calls(True, True), calls(True, False),
                                calls(False, False))
    check("bootstrap runs nothing when both artifacts exist",
          both == [], f"ran {both}")
    check("bootstrap renders the dataset when only it is missing",
          data_only == ["collect.py"], f"ran {data_only}")
    check("bootstrap trains a teacher AND collects when both are missing",
          neither == ["train_teacher.py", "collect.py"], f"ran {neither}")


def check_controls() -> None:
    """
    The controls that this lab shipped without, and the numbers they gave.

    The first version compared a VISION student against a BLIND PPO
    policy and credited the whole gap to the camera. Those arms differ
    in the camera AND in how they were trained, so the comparison could
    not separate them -- and roughly a third of the gap turned out to
    belong to the training method.

    What is pinned here is not the result (three seeds cannot establish
    it) but the SHAPE of the experiment, because that is what regressed:

      - the control arm has to exist at all, and be reachable by a flag
      - the ablation has to include an IN-DISTRIBUTION condition; the
        all-zeros frame reads as "a wall at 0.8 m" and knocks out even
        `flat`, which needs no camera
      - flat and descent must still be reported, because they are the
        columns showing the camera is worth nothing there
    """
    import json

    lab_src = (LAB / "train_student.py").read_text()
    check("a no-camera student is buildable",
          "--no-depth" in lab_src and "use_depth" in lab_src)
    check("the ablation includes an in-distribution control",
          '"mean"' in lab_src and "mean_img" in lab_src)

    f = LAB / "runs" / "results.json"
    if not f.exists():
        print("  SKIP  runs/results.json absent (artifacts are gitignored)")
        return
    r = json.loads(f.read_text())["arms"]

    blind = [a["rollout"]["vision"] for n, a in r.items()
             if n.startswith("student") and a.get("use_depth") is False]
    vis = [a["rollout"]["vision"] for n, a in r.items()
           if n.startswith("student") and a.get("use_depth") is True]
    check("both student arms have three seeds",
          len(blind) == 3 and len(vis) == 3,
          f"blind {len(blind)}, vision {len(vis)}")

    # The load-bearing negative result: no camera, still perfect.
    for t in ("flat", "down"):
        ok = all(a[t] == 1.0 for a in blind)
        check(f"the camera-less student still clears `{t}` on every seed", ok,
              f"{[a[t] for a in blind]}")

    # And the undertrained animation checkpoint must stay out of the mean.
    check("the under-trained checkpoint is excluded from the aggregate",
          "student_poor" not in r,
          "it is a 2-epoch artifact for a GIF, not a seed")


def main() -> int:
    import mujoco

    sys.path.insert(0, str(LAB))
    from terrain import world                                   # noqa: E402

    print("=" * 74)
    print("  The BD-1 head must be cosmetic, and invisible to its own camera")
    print("=" * 74)

    model = mujoco.MjModel.from_xml_string(world("flat", 0.09, 0.40))
    h, geoms = head_geoms(model, mujoco)

    # -- the head must not have changed the robot --------------------------
    check("head mass is the pre-enlargement value",
          abs(model.body_mass[h] - HEAD_MASS) < 1e-6,
          f"{model.body_mass[h]:.6f} vs {HEAD_MASS}")
    check("total robot mass is unchanged",
          abs(model.body_mass.sum() - TOTAL_MASS) < 1e-5,
          f"{model.body_mass.sum():.6f} vs {TOTAL_MASS}")
    check("head inertia tensor is unchanged",
          np.allclose(model.body_inertia[h], HEAD_INERTIA, atol=1e-8),
          np.array2string(model.body_inertia[h], precision=8))

    # A decorative geom that still collides would change contact dynamics
    # even while contributing no mass, which is the subtler half.
    noncolliding = all(model.geom_contype[g] == 0
                       and model.geom_conaffinity[g] == 0 for g in geoms)
    check(f"all {len(geoms)} head geoms are non-colliding", noncolliding)

    # -- the head must be invisible to the camera --------------------------
    margin, worst_g = worst_intrusion(model, mujoco, geoms)
    check("no head geom reaches the camera's view plane", margin < 0,
          f"worst vertex {margin * 100:+.2f} cm "
          f"(geom {worst_g}; negative = behind the camera)")

    # -- watch the checker fail --------------------------------------------
    #
    # A check that has never rejected anything is not a check. Four in
    # this repository shipped unable to fail, so the intrusion test is
    # run here against a head that deliberately pokes a lens forward
    # past the camera. If this counterexample does not trip it, the
    # PASS above means nothing.
    bad = world("flat", 0.09, 0.40).replace(
        '<camera name="eye"',
        '<geom type="sphere" pos="0.22 0 -0.05" size="0.05" '
        'density="0" contype="0" conaffinity="0"/>\n        <camera name="eye"')
    bm = mujoco.MjModel.from_xml_string(bad)
    _, bgeoms = head_geoms(bm, mujoco)
    bmargin, _ = worst_intrusion(bm, mujoco, bgeoms)
    check("the same check REJECTS a lens poking into frame", bmargin > 0,
          f"worst vertex {bmargin * 100:+.2f} cm")

    # -- the terrains must stay indistinguishable from proprioception ------
    #
    # The approach slab is what makes the lab a vision lab: if it ever
    # differs between terrains, the robot can feel which one it is on and
    # the camera becomes decoration.
    import re
    slabs = {k: re.search(r'name="approach"[^/]*', world(k, 0.09, 0.40)).group()
             for k in ("flat", "up", "down")}
    check("the approach slab is identical on all three terrains",
          len(set(slabs.values())) == 1)

    # -- the bootstrap that makes a fresh pod work ------------------------
    #
    # Neither runs/ nor the 23 MB rendered dataset is committed, so the
    # registered RunPod command would otherwise land on a pod with
    # nothing to train on -- a lab is its COMMAND, not just its code.
    # `--auto` is what closes that, so it is checked rather than
    # trusted: the branches are exercised with subprocess stubbed, which
    # costs milliseconds instead of the 25 minutes a real bootstrap
    # takes.
    check_bootstrap()
    check_controls()

    print()
    if _fails:
        print(f"  {len(_fails)} FAILED: {', '.join(_fails)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
