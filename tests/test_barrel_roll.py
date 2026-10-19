#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0", "scipy>=1.11"]
# ///
"""
Topic 8 Lab 2 — a roll must be a roll, and shaping must not pay to stall.

The lab's claim is that a cold-gas car can be trained to complete a
full 360 degree barrel roll and then land. Two things could make that
claim true on paper and false in fact, and both are tested here.

  1. THE ROLL TEST MUST NOT BE FAKEABLE. If "rolled" were just the
     integral of body roll rate, a policy could bank credit by
     tumbling: rotations do not commute, so that integral is a
     geometric angle only when the manoeuvre is roll-dominant. The
     env therefore requires BOTH the integral AND a genuine passage
     through inverted, and the tests below construct a tumble that
     satisfies the first and not the second.

  2. POTENTIAL-BASED SHAPING MUST NOT SUBSIDISE DOING NOTHING. With
     gamma < 1, holding station collects (gamma - 1) * Phi every
     step. A NEGATIVE potential therefore pays an agent to stall.
     This lab measured it: at Phi ~ -15 the subsidy was +0.075 per
     step, about +60 per episode for free, and roll completion fell
     from 21% to 12% while timeouts climbed to 86%. Stage B's
     potential is non-negative and the test pins the sign.

Properties, not shapes. No GPU, no training, no downloads.

    uv run tests/test_barrel_roll.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "08_aerial_physics" / "02_barrel_roll"
LAB1 = REPO / "08_aerial_physics" / "01_aerial_stabilization"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mujoco

    import roll_env as RE
    import thrusters as TH

    print("=" * 76)
    print("  TOPIC 8 LAB 2 — barrel roll")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. The roll test cannot be faked.
    # ------------------------------------------------------------------
    print("\n  A roll must actually be a roll")
    env = RE.RollEnv("reveal", flat=True)
    env.reset(0)
    check("a fresh episode has not rolled", not env.rolled())
    check("...and progress starts near zero", env.roll_progress() < 0.2,
          f"{env.roll_progress():.2f}")

    # Integral satisfied, never inverted -> must NOT count.
    env.roll_acc = 3.0 * RE.ROLL_TARGET
    env.went_inverted = False
    check("spinning past 360 WITHOUT going inverted does not count",
          not env.rolled(), "the integral alone is not a barrel roll")
    env.went_inverted = True
    check("...and counts once the car has genuinely been inverted",
          env.rolled())

    # Inverted but short of a full turn -> must NOT count.
    env.roll_acc = 0.6 * RE.ROLL_TARGET
    check("passing inverted WITHOUT completing the turn does not count",
          not env.rolled(), "half a roll is not a roll")

    # Progress saturates: a policy that keeps spinning must not keep
    # banking shaping for it.
    env.roll_acc = 9.0 * RE.ROLL_TARGET
    check("roll progress SATURATES at one turn",
          env.roll_progress() == 1.0,
          "or a policy is paid to spin forever")
    env.roll_acc = -RE.ROLL_TARGET
    env.went_inverted = True
    check("rolling the other way counts too (the sign is free)",
          env.rolled())

    # The inversion flag must correspond to real geometry.
    env.reset(1)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    check("upright is not inverted", env.up_dot() > 0.9,
          f"up_dot {env.up_dot():.2f}")
    env.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]        # 180 deg about x
    mujoco.mj_forward(env.model, env.data)
    check("a 180 degree roll puts the body axis below the horizon",
          env.up_dot() < RE.INVERT_COS,
          f"up_dot {env.up_dot():.2f} vs threshold {RE.INVERT_COS}")

    # ------------------------------------------------------------------
    # 2. Shaping must charge for stalling, not pay for it.
    # ------------------------------------------------------------------
    print("\n  Shaping must not subsidise doing nothing")
    GAMMA = 0.995
    roll_env = RE.RollEnv("reveal", flat=True, require_roll=True)
    roll_env.reset(3)
    phi = roll_env._potential()
    check("the roll stage's potential is NON-NEGATIVE",
          phi >= 0.0, f"Phi = {phi:.2f}")
    check("...so holding station COSTS rather than pays",
          (GAMMA - 1.0) * phi <= 0.0,
          f"leak {(GAMMA - 1.0) * phi:+.4f} per step")

    # The counterexample, kept permanently: the shape this lab
    # shipped first, and what it was worth.
    bad_phi = -15.0
    check("a NEGATIVE potential would pay to stall — the bug, pinned",
          (GAMMA - 1.0) * bad_phi > 0.0,
          f"would pay {(GAMMA - 1.0) * bad_phi:+.3f}/step, "
          f"{(GAMMA - 1.0) * bad_phi * 800:+.0f} per episode")

    # Progress must move the potential, or it teaches nothing.
    roll_env.roll_acc = 0.0
    roll_env.went_inverted = False
    lo = roll_env._potential()
    roll_env.roll_acc = RE.ROLL_TARGET
    roll_env.went_inverted = True
    hi = roll_env._potential()
    check("completing the roll RAISES the potential", hi > lo + 5.0,
          f"{lo:.1f} -> {hi:.1f}")
    roll_env.roll_acc = 4.0 * RE.ROLL_TARGET
    check("...and over-rolling raises it no further",
          abs(roll_env._potential() - hi) < 1e-9)

    # ------------------------------------------------------------------
    # 3. Stage A must be lab 1's task, not an invented variant.
    # ------------------------------------------------------------------
    print("\n  Stage A reproduces lab 1 rather than inventing a task")
    land = RE.RollEnv("reveal", flat=True, require_roll=False)
    check("the landing stage uses lab 1's 6 second clock",
          land.max_steps == int(RE.MAX_SECONDS_LAND * RE.CTRL_HZ) == 600,
          f"{land.max_steps} steps")
    check("the roll stage gets 8 seconds — a 360 turn needs 0.9 s of it",
          roll_env.max_steps == 800)
    alts, vzs = [], []
    for s in range(40):
        land.reset(s)
        alts.append(land.altitude())
        vzs.append(land.vel()[2])
    check("...and lab 1's launch envelope, 2.5 to 7.5 m",
          2.0 < min(alts) and max(alts) < 8.0,
          f"{min(alts):.1f}-{max(alts):.1f} m")
    check("...with lab 1's upward velocity, 1 to 5 m/s",
          0.9 < min(vzs) and max(vzs) < 5.1,
          f"{min(vzs):.1f}-{max(vzs):.1f} m/s")
    check("the landing stage's potential is lab 1's, negative and all",
          land._potential() < 0.0,
          f"Phi = {land._potential():.2f} — reproduced deliberately")

    # The observation is IDENTICAL across stages, which is what makes
    # the warm start a load rather than a surgery.
    check("both stages share one observation width — warm start is safe",
          land.obs_dim == roll_env.obs_dim == 33,
          f"{land.obs_dim} vs {roll_env.obs_dim}")
    check("roll state is in the observation in BOTH stages",
          len(land._obs()) == len(roll_env._obs()),
          "without it the task is not Markov")

    # ------------------------------------------------------------------
    # 4. Scoring must not let the policy decline the stunt.
    # ------------------------------------------------------------------
    print("\n  A gentle landing without the roll must score below one with")
    e = RE.RollEnv("reveal", flat=True, require_roll=True)
    e.reset(5)
    e.outcome = "landed"
    e.roll_acc, e.went_inverted = 0.0, False
    e.data.qvel[:3] = [0.0, 0.0, -0.2]
    e.data.qvel[3:6] = [0.0, 0.0, 0.0]
    e.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(e.model, e.data)
    no_roll = e._terminal()
    nm_no = e.outcome
    e.outcome = "landed"
    e.roll_acc, e.went_inverted = RE.ROLL_TARGET, True
    rolled = e._terminal()
    check("rolled-and-landed outscores landed-without-rolling",
          rolled > no_roll + 50.0, f"{rolled:.0f} vs {no_roll:.0f}")
    check("...and the two are named differently",
          nm_no == "landed_no_roll" and e.outcome == "rolled_and_landed",
          f"{nm_no} / {e.outcome}")
    e.outcome = "crash"
    crash_rolled = e._terminal()
    e.roll_acc, e.went_inverted = 0.0, False
    crash_plain = e._terminal()
    check("a crash AFTER rolling still beats a crash before it",
          crash_rolled > crash_plain,
          f"{crash_rolled:.0f} vs {crash_plain:.0f} — keeps a gradient")
    check("...but both remain firmly negative",
          crash_rolled < -20.0 and crash_plain < -20.0)

    # ------------------------------------------------------------------
    # 5. The hardware prediction this lab is built on.
    # ------------------------------------------------------------------
    print("\n  The rigs, and what lab 1's audit predicts for them")
    for nm, pure in (("minimal", 0.0), ("sixlift", 0.0)):
        d = TH.audit(nm, TH.rig(nm))
        check(f"{nm} has NO pure roll couple",
              d["roll_pure_Nm"] < 1.0, f"{d['roll_pure_Nm']:.0f} N·m")
        check(f"...but plenty of FREE roll torque — it can still roll",
              d["roll_free_Nm"] > 1e4, f"{d['roll_free_Nm']:,.0f} N·m")
    d = TH.audit("reveal", TH.rig("reveal"))
    check("reveal's pure couple is about half its free torque",
          0.4 < d["purity"] < 0.7, f"purity {d['purity']:.0%}")
    check("...and completes a 360 in under a second at full couple",
          d["min_roll_time_s"] < 1.0, f"{d['min_roll_time_s']:.2f} s")
    check("the roll fits inside the 8 second clock with room to land",
          d["min_roll_time_s"] * 2 < RE.MAX_SECONDS * 0.5,
          "even at twice the theoretical minimum")

    # ------------------------------------------------------------------
    # 6. Propellant: the roll has to be affordable.
    # ------------------------------------------------------------------
    print("\n  The budget")
    hov = TH.hover_seconds()
    check("the tank holds seconds of hover, not minutes",
          3.0 < hov < 12.0, f"{hov:.1f} s")
    check("...which is close to the episode clock — the margin is thin",
          hov < RE.MAX_SECONDS * 1.5,
          f"{hov:.1f} s of gas against a {RE.MAX_SECONDS:.0f} s episode")
    tk = RE.tank_state(TH.gas_mass())
    check("a full tank delivers full thrust", tk["frac"] > 0.95,
          f"{tk['frac']:.0%}")
    low = RE.tank_state(TH.gas_mass() * 0.05)
    check("a nearly empty one does not", low["frac"] < 0.5,
          f"{low['frac']:.0%} — thrust collapses after regulator dropout")

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
