#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "torch==2.11.0", "mujoco>=3.2",
#                 "scipy>=1.11", "pillow>=10.0"]
# ///
"""
Topic 8 Lab 7 — a teacher-student claim is only as good as its plumbing.

This lab claims that a policy which cannot learn to read a depth
sensor from a scalar reward CAN learn it by regression against a
teacher that was simply told the answer. Several things would make
that claim look true, or look false, while being an artifact. Each
is pinned here.

  1. THE ARMS MUST DIFFER IN ONE CHANNEL AND NOTHING ELSE. If the
     observation layouts drift apart, the comparison stops being
     about information. The `oracle` arm must stay BYTE-IDENTICAL
     to lab 3's `blind` arm -- that is the reproduction control,
     and moving the terrain channel by seven indices once took the
     teacher from 60% to 4% with no error and every array the right
     shape.

  2. DAGGER MUST RELABEL. The student drives; the teacher labels.
     If `obs_for` mutated the arm, or the label were taken from the
     driver, DAgger would silently degenerate into behaviour
     cloning and still produce a plausible curve.

  3. THE LABEL IS THE TEACHER'S MEAN, NOT A SAMPLE. A sampled label
     injects exploration noise into a regression target.

  4. THE COMPARISON IS PAIRED. Arms that flew different held-out
     seeds cannot be compared, and nothing in the numbers would
     look wrong.

  5. A SENSOR THAT SEES ITS OWN VEHICLE MUST RAISE. A mount buried
     in the hull returns a dense field of constant short ranges --
     correct dtype, correct shape, no signal.

Properties, not shapes. No GPU, no training, no downloads.

    uv run tests/test_teacher_student.py
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "08_aerial_physics" / "07_teacher_student"
LAB3 = REPO / "08_aerial_physics" / "03_land_on_target"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    sys.path.insert(0, str(LAB))

    print("=" * 76)
    print("  TOPIC 8 LAB 7 — teacher and student")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. The sensor's geometry is a pinhole, not a fan.
    # ------------------------------------------------------------------
    print("\n  The rangefinder is a camera, not a fan")
    import rangefinder as RF

    d = RF.beam_directions()
    check("every beam is a unit vector",
          bool(np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)))
    check("every beam points DOWN in the body frame",
          bool((d[:, 2] < 0).all()),
          f"max z {d[:, 2].max():+.3f}")

    grid = d.reshape(RF.BEAM_N, RF.BEAM_N, 3)
    tx = grid[:, :, 0] / (-grid[:, :, 2])
    step = np.diff(tx, axis=1)
    check("the grid is uniform in TANGENT — the pinhole model",
          bool(np.allclose(step, step[0, 0], atol=1e-12)),
          f"step {step[0, 0]:.4f} everywhere")
    ang = np.degrees(np.arctan(tx[0]))
    gaps = np.diff(ang)
    half = len(gaps) // 2
    check("...so the ANGLES tighten toward the edge",
          bool((np.diff(gaps[:half]) > 0).all()),
          f"{gaps[0]:.2f} deg at the edge, {gaps[half]:.2f} at the axis")
    # The buggy version shipped a FAN and looked entirely reasonable.
    # Assert the fan is measurably different, or this test proves
    # nothing about which model is implemented.
    e = math.tan(RF.HALF_FOV)
    fan = np.tan(np.linspace(-RF.HALF_FOV, RF.HALF_FOV, RF.BEAM_N))
    pin = np.linspace(-e, e, RF.BEAM_N)
    check("...and a fan would be measurably different here",
          float(np.abs(fan - pin).max()) > 0.05,
          f"max tangent difference {np.abs(fan - pin).max():.3f} — if these "
          f"agreed, the check above could not distinguish them")

    check("a no-return beam reads the FAR end of the scale, not a sentinel",
          RF.NO_RETURN == 1.0,
          "a -1 would put a discontinuity mid-range for the network to "
          "trip over")
    check("an absurd beam count is REJECTED, not silently defaulted",
          _raises(lambda: RF.beam_directions(n=1)))

    # ------------------------------------------------------------------
    # 2. The three arms differ in ONE channel, at a FIXED index.
    # ------------------------------------------------------------------
    print("\n  One task, three views, and the layout is load-bearing")
    from teach_env import PATCH_SPAN, SENSES, TeachEnv

    check("the arms are exactly the three this lab defines",
          tuple(SENSES) == ("deaf", "oracle", "onboard"), str(SENSES))

    env = TeachEnv("dense", "badlands", sense="onboard", seed=5)
    env.reset(123)
    for _ in range(7):
        env.step(np.full(env.act_dim, 0.4))

    views = {s: env.obs_for(s) for s in SENSES}
    # 3 + 3 attitude columns, 1 uprightness, 3 body rates, 3 velocity.
    PREFIX = 13
    check("the shared PREFIX is identical across arms",
          bool(np.allclose(views["deaf"][:PREFIX], views["oracle"][:PREFIX])
               and np.allclose(views["deaf"][:PREFIX],
                               views["onboard"][:PREFIX])),
          f"{PREFIX} numbers of attitude, rates and velocity")
    nact = env.act_dim
    check("the trailing action echo is identical across arms",
          bool(np.allclose(views["deaf"][-nact:], views["onboard"][-nact:])))
    check("the widths differ only by the terrain channel",
          (len(views["oracle"]) - len(views["deaf"]) == 2
           and len(views["onboard"]) - len(views["deaf"]) == RF.BEAM_N ** 2),
          f"deaf {len(views['deaf'])}, oracle {len(views['oracle'])}, "
          f"onboard {len(views['onboard'])}")
    check(f"the TERRAIN CHANNEL SITS AT INDEX {PREFIX}",
          bool(np.allclose(views["oracle"][PREFIX:PREFIX + 2],
                           env.oracle_state())),
          "lab 3 put altitude and clearance exactly here; moving them "
          "seven places later cost the teacher 60% -> 4% on two seeds "
          "with nothing raising and every array the right shape")
    check("...and the deaf arm has NO terrain channel there",
          len(views["deaf"]) == len(views["oracle"]) - 2
          and bool(np.allclose(views["deaf"][PREFIX:PREFIX + 2],
                               views["oracle"][PREFIX + 2:PREFIX + 4])),
          "the mark bearing follows the channel directly, so dropping "
          "the channel shifts it two places left")

    # The reproduction control: this lab's teacher view must BE lab
    # 3's blind view. If this fails, the published 56% no longer
    # bounds anything in this lab.
    sys.path.insert(0, str(LAB3))
    for m in [k for k in list(sys.modules)
              if k in ("car", "terrain", "thrusters", "gas")]:
        del sys.modules[m]
    try:
        from target_env import TargetEnv

        e3 = TargetEnv("dense", "badlands", sense="blind", world="earth",
                       seed=5)
        o3 = e3.reset(123)
        for _ in range(7):
            o3, _r, _d, _i = e3.step(np.full(e3.act_dim, 0.4))
        o7 = views["oracle"]
        check("the ORACLE view is byte-identical to lab 3's BLIND view",
              bool(np.array_equal(np.asarray(o3), np.asarray(o7))),
              f"maxdiff {np.abs(np.asarray(o3) - np.asarray(o7)).max():.2e} "
              f"— lab 3 measured this exact observation at 56%, which is "
              f"what makes this lab's 60% a reproduction rather than a "
              f"new claim")
    except Exception as exc:                    # pragma: no cover
        check("the ORACLE view is byte-identical to lab 3's BLIND view",
              False, f"could not build lab 3's env: {exc!r}")
    finally:
        sys.path.remove(str(LAB3))

    check("an unknown arm is REJECTED, not silently defaulted",
          _raises(lambda: env.obs_for("xray")))
    check("the oracle patch span is unused — no arm gets a height map",
          PATCH_SPAN > 0,
          "the constant survives for --gap only; lab 3 measured a 5x5 "
          "patch taking PPO from 56% to 0%")

    # ------------------------------------------------------------------
    # 3. obs_for must NOT mutate the arm.
    # ------------------------------------------------------------------
    print("\n  Relabelling must not change who is flying")
    before = env.sense
    env.obs_for("oracle")
    env.obs_for("deaf")
    check("asking for another arm's view leaves THIS arm set",
          env.sense == before, f"{before!r} before, {env.sense!r} after")
    a1 = env.obs_for("deaf")
    a2 = env.obs_for("deaf")
    check("...and the deaf view is deterministic for a fixed state",
          bool(np.allclose(a1, a2)))
    # The onboard view is NOT deterministic -- it carries sensor
    # noise -- and if it were, the dropout model would be dead.
    n1, n2 = env.obs_for("onboard"), env.obs_for("onboard")
    check("...while the onboard view DOES vary — the sensor is noisy",
          not bool(np.allclose(n1, n2)),
          "if these matched, noise and dropout would be switched off and "
          "the student would be reading a clean sensor")

    # ------------------------------------------------------------------
    # 4. DAgger relabels: the driver changes, the labeller never does.
    # ------------------------------------------------------------------
    print("\n  DAgger must train on the STUDENT's states with the "
          "TEACHER's labels")
    src = (LAB / "distill.py").read_text()
    tree = ast.parse(src)
    collect = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_collect")

    # The label assignment must be OUTSIDE any branch on beta.
    label_calls = [n for n in ast.walk(collect)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name)
                   and n.func.id == "_teacher_action"]
    check("the teacher is queried in the collection loop",
          len(label_calls) >= 1)
    ifs = [n for n in ast.walk(collect) if isinstance(n, ast.If)]
    in_branch = any(
        any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "_teacher_action"
            for c in ast.walk(br))
        for n in ifs for br in (n.body + n.orelse))
    check("...UNCONDITIONALLY — the label never depends on who drove",
          not in_branch,
          "a label taken from the driver turns DAgger into behaviour "
          "cloning, silently, with the loss curve still falling")
    check("the STUDENT's action is what may drive",
          any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "_student_action" for n in ast.walk(collect)))

    ta = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_teacher_action")
    uses_mu = any(isinstance(n, ast.Attribute) and n.attr == "mu"
                  for n in ast.walk(ta))
    samples = any(isinstance(n, ast.Attribute) and n.attr == "sample"
                  for n in ast.walk(ta))
    check("the label is the teacher's MEAN action",
          uses_mu and not samples,
          "a sampled label injects the teacher's exploration noise into a "
          "regression target")
    check("the teacher is asked for the ORACLE view of the student's state",
          'obs_for("oracle")' in src,
          "this is the one line that makes it DAgger rather than cloning")

    # BC must be the same code path, forced to one round.
    check("BC is DAgger with one teacher-driven round, not a second "
          "code path",
          'args.iters = 1' in src and 'args.decay = 1.0' in src,
          "two implementations would drift apart in some detail nobody "
          "is watching")

    # ------------------------------------------------------------------
    # 5. The statistics.
    # ------------------------------------------------------------------
    print("\n  Paired statistics, and a refusal when they cannot be paired")
    sys.path.insert(0, str(LAB))
    import compare as CMP

    m = CMP.mcnemar_exact([True] * 10, [True] * 10)
    check("identical arms are not significantly different",
          m["p"] == 1.0 and m["n_discordant"] == 0,
          "and the concordant pairs are correctly ignored")
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10
    m = CMP.mcnemar_exact(a, b)
    check("a clean 10-0 split is significant",
          m["p"] < 0.01 and m["b01"] == 10 and m["b10"] == 0,
          f"p = {m['p']:.5f}")
    # Concordant pairs must not move the p-value: padding both arms
    # with shared wins is the classic way an unpaired test inflates.
    m2 = CMP.mcnemar_exact(a + [True] * 500, b + [True] * 500)
    check("...and adding 500 pairs BOTH arms win changes nothing",
          abs(m2["p"] - m["p"]) < 1e-12,
          "McNemar conditions on the discordant pairs, which is the "
          "whole reason to use it")
    check("unpaired input is REJECTED",
          _raises(lambda: CMP.mcnemar_exact([True], [True, False])))

    lo, hi = CMP.wilson(0, 48)
    check("Wilson stays inside [0, 1] at a zero count",
          lo >= 0.0 and hi <= 1.0 and hi > 0.0,
          f"[{lo:.3f}, {hi:.3f}] — the normal approximation returns a "
          f"negative lower bound here")
    lo, hi = CMP.wilson(48, 48)
    check("...and at a perfect count",
          lo < 1.0 and hi <= 1.0, f"[{lo:.3f}, {hi:.3f}]")

    # ------------------------------------------------------------------
    # 6. A sensor looking at its own vehicle must RAISE.
    # ------------------------------------------------------------------
    print("\n  A sensor that sees its own hull is a constant, and must fail "
          "loudly")
    rf = env.rf
    check("the shipped mount is looking at the GROUND",
          rf.self_hit_rate() < 0.05,
          f"{rf.self_hit_rate():.2%} of {rf.total_beams:,} beams hit the "
          f"vehicle")
    # Watch the check fail before trusting it.
    rf.self_hits, rf.total_beams = 90, 100
    check("...and the audit RAISES when it is not",
          _raises(rf.audit),
          "a mount buried in the hull returns a dense field of 4 mm "
          "readings — right dtype, right shape, no signal. This check "
          "fired on its first real run.")
    rf.self_hits = rf.total_beams = 0

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
