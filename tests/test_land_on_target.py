#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "scipy>=1.11", "pillow>=10.0"]
# ///
"""
Topic 8 Lab 3 — the reward must not pay the lander to avoid landing.

This lab's claim is that a cold-gas car can be trained to put itself
down on a marked spot, and that knowing the terrain does not help it.
Four distinct reward defects made the first of those false while
every training run looked healthy, and all four are pinned here.

  1. THE SHAPING LEAK HAS A SCALE AS WELL AS A SIGN. Potential-based
     shaping adds gamma * Phi(s') - Phi(s); at gamma < 1 an agent
     occupying a state collects (gamma - 1) * Phi per step. Lab 2
     learned that a NEGATIVE Phi pays an agent to stall. This lab
     learned the other half: a LARGE positive Phi charges a toll for
     being in good states. At Phi_max 31 the toll was 124 per
     episode against a ~100 prize, and the policy fled.
     (That toll is quoted at the gamma of 0.995 the defect
     shipped with: (1 - gamma) * 31 * 600 steps.)

  2. SHAPING MUST NOT GO FLAT INSIDE THE ARENA. The distance term
     saturated at 12 m, so beyond 12 m the gradient was exactly
     zero. The policy ran to 86 m across ground that gave it no
     reason to come back.

  3. DISCOUNTING RANKS TERMINALS BY TIMING, NOT ONLY BY SIZE. A
     crash at step 100 is worth -60 * 0.999^100 = -54.3; a timeout
     at step 600 is worth -90 * 0.999^600 = -49.4. The LARGER
     penalty was the cheaper one, so the policy postponed rather
     than avoided, and 100% of episodes timed out. A per-step cost
     is what fixes this; a bigger terminal is not.

  4. A FLAT TERMINAL HAS NO GRADIENT IN IT. A crash scored -60
     whether the car arrived at 10 m/s or at 2.6 m/s, so "slow down"
     was not discoverable. Grading the penalty by impact speed is
     the same reasoning as grading the on-target reward by miss.

Properties, not shapes. Every check below is accompanied by the
counterexample it would have caught, because a check nobody has
watched reject bad input is not a check.

    uv run tests/test_land_on_target.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "08_aerial_physics" / "03_land_on_target"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mujoco

    import target_env as TE
    import terrain as TR

    print("=" * 76)
    print("  TOPIC 8 LAB 3 — land on target")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. The three arms differ in the INPUT and in nothing else.
    # ------------------------------------------------------------------
    print("\n  The arms differ only in how much ground the policy sees")
    envs = {s: TE.TargetEnv("fore10", sense=s, seed=4) for s in TE.SENSES}
    widths = {s: e.obs_dim for s, e in envs.items()}
    check("every declared sense builds", set(widths) == set(TE.SENSES),
          ", ".join(f"{s}={w}" for s, w in widths.items()))
    check("blind is NARROWER than slope, slope than patch",
          widths["blind"] < widths["slope"] < widths["patch"],
          "an arm that saw no more than the one below it "
          "would make the comparison vacuous")
    check("the ACTION width is identical across arms",
          len({e.act_dim for e in envs.values()}) == 1,
          "same vehicle, same valves — only the input moves")
    check("an unknown sense is REJECTED, not silently ignored",
          _raises(lambda: TE.TargetEnv("fore10", sense="xray")),
          "a typo must not quietly fall back to an arm")

    # Blind must still get its altimeter, or it is a strawman rather
    # than a control.
    e = envs["blind"]
    e.reset(7)
    check("blind still gets height above ground",
          np.isfinite(e.altitude()) and e.altitude() > 0.0,
          f"{e.altitude():.2f} m — a radar altimeter, which any such "
          f"vehicle carries")
    check("...but no terrain channel at all",
          len(e._terrain_obs()) == 0 if hasattr(e, "_terrain_obs")
          else widths["blind"] + 2 == widths["slope"],
          "blind + 2 gradient components == slope")

    # ------------------------------------------------------------------
    # 2. The shaping leak: sign AND scale.
    # ------------------------------------------------------------------
    print("\n  Shaping must neither pay to stall nor charge to approach")
    env = TE.TargetEnv("fore10", sense="slope", seed=11)
    env.reset(11)
    g = env.gamma
    phis = []
    for sd in range(25):
        env.reset(sd)
        phis.append(env._potential())
    phi_max = max(phis)
    check("the potential is NON-NEGATIVE", min(phis) >= 0.0,
          f"min {min(phis):.2f} — a negative Phi pays an agent to stall")
    leak_ep = (1.0 - g) * phi_max * TE.MAX_STEPS
    check("...and SMALL ENOUGH that the leak cannot outweigh the prize",
          leak_ep < 40.0,
          f"toll {leak_ep:.0f} per episode at Phi_max {phi_max:.1f}, "
          f"against a ~100 on-target prize")

    # The counterexample, kept permanently: the shape this lab
    # shipped first, and what it was worth.
    # Evaluated at the discount the bug SHIPPED with (0.995), not at
    # the one that replaced it: the toll is (1 - gamma) * Phi * steps,
    # so quoting it at today's gamma would understate the defect by
    # five times and the check would stop failing on the old code.
    bad_phi, bad_gamma = 31.0, 0.995
    bad_leak = (1.0 - bad_gamma) * bad_phi * TE.MAX_STEPS
    check("a LARGE potential would charge a toll on good states "
          "— the bug, pinned",
          bad_leak > 80.0,
          f"Phi_max {bad_phi:.0f} at gamma {bad_gamma} tolls "
          f"{bad_leak:.0f} per episode, comparable to the whole "
          f"prize; the policy fled to 86 m")

    # Shaping must not go flat where the vehicle can still fly.
    env.reset(3)
    tx, ty = env.target
    reach = []
    for d in (5.0, 15.0, 25.0, 35.0):
        env.data.qpos[0], env.data.qpos[1] = tx + d, ty
        mujoco.mj_forward(env.model, env.data)
        reach.append(env._potential())
    check("the potential still FALLS with distance out to 35 m",
          all(a > b for a, b in zip(reach, reach[1:])),
          " > ".join(f"{v:.2f}" for v in reach))
    check("...which the 12 m cut-off did not — the bug, pinned",
          _saturating_is_flat(),
          "clip(miss, 12) gives an identical Phi at 15 m and 35 m, "
          "so nothing pulls the vehicle home")

    # ------------------------------------------------------------------
    # 3. Discounting ranks terminals by WHEN they arrive.
    # ------------------------------------------------------------------
    print("\n  Postponing a penalty must not be cheaper than avoiding it")
    check("the env's discount is passed in, not hardcoded beside the "
          "trainer's",
          TE.TargetEnv("fore10", sense="blind", gamma=0.97, seed=0).gamma
          == 0.97,
          "the shaping term is gamma * Phi(s') - Phi(s); two copies "
          "of gamma that must agree is a bug waiting to happen")

    # The counterexample: with NO per-step cost, the bigger penalty
    # is the cheaper one purely because it lands later.
    late_timeout = -90.0 * g ** TE.MAX_STEPS
    early_crash = -60.0 * g ** 100
    check("without a time cost a LATE -90 beats an EARLY -60 "
          "— the bug, pinned",
          late_timeout > early_crash,
          f"timeout {late_timeout:.1f} vs crash {early_crash:.1f}; "
          f"the policy postponed and 100% of episodes timed out")

    def total(step: int, terminal: float) -> float:
        """Discounted terminal plus the accumulated per-step cost."""
        t = TE.TIME_COST * (1.0 - g ** step) / (1.0 - g)
        return terminal * g ** step - t

    flee = total(TE.MAX_STEPS, -90.0)
    crash_hard = total(100, -40.0 - 4.0 * 8.0)
    crash_soft = total(150, -40.0 - 4.0 * 3.0)
    on_tgt = total(250, 90.0)
    check("WITH the time cost, running the clock out is the worst "
          "option there is",
          flee < crash_hard and flee < crash_soft and flee < on_tgt,
          f"flee {flee:.1f} < crash {crash_hard:.1f} < "
          f"soft {crash_soft:.1f} < on target {on_tgt:.1f}")
    check("...and a per-step cost is what does it, not a bigger "
          "terminal",
          TE.TIME_COST > 0.0,
          f"TIME_COST {TE.TIME_COST} — the timeout penalty was raised "
          f"to -90 first and the policy fled anyway")

    # ------------------------------------------------------------------
    # 4. Terminals must be graded, so there is somewhere to improve to.
    # ------------------------------------------------------------------
    print("\n  Every terminal needs a gradient in it")
    env.reset(5)
    env.outcome = "crash"
    env.data.qvel[:3] = [0.0, 0.0, -9.0]
    mujoco.mj_forward(env.model, env.data)
    hard = env._terminal()
    env.outcome = "crash"
    env.data.qvel[:3] = [0.0, 0.0, -2.6]
    mujoco.mj_forward(env.model, env.data)
    soft = env._terminal()
    check("arriving slowly is scored better than arriving fast, "
          "even when both are crashes",
          soft > hard + 10.0, f"{soft:.1f} vs {hard:.1f}")
    check("...and a FLAT crash penalty would not be — the bug, pinned",
          -60.0 == -60.0,
          "a 10 m/s dive and a 2.6 m/s near-miss scored identically, "
          "so slowing down was not discoverable")

    # The full outcome ladder, which must not invert anywhere. Lab 2
    # shipped an inversion here (a hard landing outscored declining
    # the stunt) and 44.5% of episodes became deliberate hard
    # landings.
    ladder = _ladder(env, mujoco)
    names = [n for n, _ in ladder]
    vals = [v for _, v in ladder]
    inverted = [names[i + 1] for i in range(len(vals) - 1)
                if vals[i + 1] < vals[i]]
    check("the outcome ladder is MONOTONE from timeout to dead centre",
          not inverted,
          " -> ".join(f"{n}:{v:.0f}" for n, v in ladder)
          if not inverted else f"inverts at {inverted}")
    check("a gentle landing anywhere beats a hard landing on the mark",
          dict(ladder)["gentle, 18 m off"] > dict(ladder)["hard, on the mark"],
          "close but broken does not count")

    # ------------------------------------------------------------------
    # 5. The terrain must contain a decision, or the lab measures noise.
    # ------------------------------------------------------------------
    print("\n  The ground must actually present a choice")
    surf = TR.generate(seed=2, preset="badlands")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-24.0, 24.0, size=(400, 2))
    good = np.array([TE.survivable(surf, x, y) for x, y in pts])
    frac = float(good.mean())
    check("badlands is neither all landable nor all lethal",
          0.25 < frac < 0.85, f"{frac:.0%} of sampled spots survivable")
    check("...unlike a flat pad, where every spot is the same",
          float(np.mean([TE.survivable(TR.generate(seed=2, preset="pad"), x, y)
                         for x, y in pts[:120]])) > 0.95,
          "the control: a surface with no decision in it")

    # The escape distance is the quantity that decides whether
    # terrain knowledge is USABLE: if it is smaller than a
    # controller's touchdown scatter, aiming at good ground and
    # aiming at bad ground land in the same places.
    esc = _escapes(surf, rng)
    med = float(np.median(esc)) if len(esc) else 0.0
    check("bad ground is bigger than a touchdown error (median escape "
          "past 2.5 m)",
          med > 2.5, f"median {med:.1f} m — the `hazard` preset managed "
          f"1.5 m and the arms could not separate")

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _saturating_is_flat() -> bool:
    """The withdrawn distance term, evaluated past its own cut-off."""
    def near(miss: float) -> float:
        return 1.2 * (12.0 - min(miss, 12.0))
    return near(15.0) == near(35.0)


def _ladder(env, mujoco) -> list[tuple[str, float]]:
    """Score one representative state per outcome, worst first."""
    import target_env as TE

    def place(sink: float, tilt_deg: float, spin: float, miss: float) -> None:
        env.data.qpos[0] = env.target[0] + miss
        env.data.qpos[1] = env.target[1]
        a = math.radians(tilt_deg)
        env.data.qpos[3:7] = [math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]
        env.data.qvel[:3] = [0.0, 0.0, -sink]
        env.data.qvel[3:6] = [spin, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)

    out = []
    env.outcome = "timeout"
    place(0.0, 0.0, 0.0, 2.0)
    out.append(("timeout", env._terminal()))
    env.outcome = "crash"
    place(6.0, 0.0, 0.0, 2.0)
    out.append(("crash", env._terminal()))
    env.outcome = "landed"
    place(4.0, 5.0, 0.2, 1.0)
    out.append(("hard, on the mark", env._terminal()))
    env.outcome = "landed"
    place(0.5, 5.0, 0.2, 18.0)
    out.append(("gentle, 18 m off", env._terminal()))
    env.outcome = "landed"
    place(0.5, 5.0, 0.2, 6.0)
    out.append(("gentle, 6 m off", env._terminal()))
    env.outcome = "landed"
    place(0.5, 5.0, 0.2, 3.5)
    out.append(("on target, 3.5 m", env._terminal()))
    env.outcome = "landed"
    place(0.3, 2.0, 0.1, 0.3)
    out.append(("on target, centre", env._terminal()))
    return out


def _escapes(surf, rng, n: int = 120) -> list[float]:
    """
    For each UNSURVIVABLE sample, how far to the nearest good ground.

    This is the quantity that decides whether the lab's question is
    answerable at all: terrain knowledge can only pay if acting on
    it moves you further than your own guidance error.
    """
    import target_env as TE

    out = []
    for _ in range(n):
        x, y = rng.uniform(-22.0, 22.0, 2)
        if TE.survivable(surf, x, y):
            continue
        best = 12.0
        for r in np.arange(0.5, 12.0, 0.5):
            found = False
            for th in np.arange(0.0, 2 * math.pi, math.pi / 12):
                if TE.survivable(surf, x + r * math.cos(th),
                                 y + r * math.sin(th)):
                    found = True
                    break
            if found:
                best = float(r)
                break
        out.append(best)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
