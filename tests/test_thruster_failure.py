#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "scipy>=1.11", "pillow>=10.0"]
# ///
"""
Topic 8 Lab 5 — a fault must be felt, not read, and inaction must be worst.

The lab's claim is that a policy trained with in-flight thruster
failures survives them far better than one that never saw a failure,
and that no cheap static authority check predicts which failures are
survivable. Several defects would make the first true on paper and
false in fact, and each is pinned here with its counterexample.

  1. THE FAULT MUST NOT LEAK INTO THE OBSERVATION. It is applied to
     the REALISED control, after valve lag and thrust scaling, and
     never to `u_act` -- which is what the policy commanded and what
     the observation reports. If the fault touched `u_act` the
     policy would be handed a perfect fault detector it never
     earned, and the `--tell` arm would stop being a comparison.

  2. PROPELLANT MUST FOLLOW THE REALISED OPENING. A latched nozzle
     keeps spending whether or not the policy asks. Billing the
     command hides half of why stuck-open is worse than stuck-shut.

  3. THE WORST TERMINAL MUST BE THE ONE INACTION REACHES. Lab 4 put
     the crash at the bottom, correctly for a gate course. Ported to
     a pure landing task it taught the policy never to try: crashes
     fell 97% -> 3%, timeouts rose 3% -> 97%, landings never above
     1% across 4M steps. Here a hover IS inaction, so the timeout
     sits at the bottom.

  4. THE FAULT MUST BE A PROPERTY OF THE SEED, or the two arms are
     not scored on the same breakages and the comparison is void.

Properties, not shapes. No GPU, no training, no downloads.

    uv run tests/test_thruster_failure.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAB = REPO / "08_aerial_physics" / "05_thruster_failure"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mujoco

    import fail_env as FE
    import fault
    import thrusters as TH

    print("=" * 76)
    print("  TOPIC 8 LAB 5 — thruster failure")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. The fault is felt, not read.
    # ------------------------------------------------------------------
    print("\n  The fault must reach the physics and not the observation")
    e = FE.FailEnv("reveal", flat=True, fail_mode="open",
                   fail_at=(0.05, 0.06), seed=1)
    e.reset(1)
    k = e.fault["nozzle"]
    cmd = np.zeros(e.act_dim)           # ask for NOTHING
    for _ in range(20):
        e.step(cmd)
    check("a latched nozzle fires even when the policy commands zero",
          e.fault_active and e.gas_used > 0.0,
          f"{e.gas_used:.2f} kg spent on a zero command")
    check("...and the OBSERVATION still reports the zero command",
          float(np.max(e.u_act)) < 1e-6,
          f"max u_act {float(np.max(e.u_act)):.2e} — the gap between "
          f"command and effect IS the fault")
    check("the fault channel is silent unless --tell",
          float(np.abs(e._fault_obs()).max()) == 0.0,
          "off by default: the vehicle infers the break from dynamics")

    told = FE.FailEnv("reveal", flat=True, fail_mode="open",
                      fail_at=(0.05, 0.06), tell=True, seed=1)
    told.reset(1)
    for _ in range(20):
        told.step(np.zeros(told.act_dim))
    check("...and reports the nozzle and mode when --tell is on",
          float(np.abs(told._fault_obs()).max()) > 0.0
          and told._fault_obs()[told.fault["nozzle"]] == 1.0)
    check("both arms share an observation width",
          e.obs_dim == told.obs_dim == len(e._obs()),
          f"{e.obs_dim} — a policy can load into either, which is what "
          f"makes the comparison a comparison")

    # ------------------------------------------------------------------
    # 2. Propellant follows the realised opening.
    # ------------------------------------------------------------------
    print("\n  A latched nozzle drains the tank")
    def spend(mode, cmd_level):
        en = FE.FailEnv("reveal", flat=True, fail_mode=mode,
                        fail_at=(0.05, 0.06), seed=2)
        en.reset(2)
        for _ in range(60):
            en.step(np.full(en.act_dim, cmd_level))
        return en.gas_used

    shut_gas, open_gas = spend("shut", 0.0), spend("open", 0.0)
    check("stuck OPEN spends propellant on a zero command, stuck SHUT "
          "does not",
          open_gas > 0.1 and shut_gas < 1e-9,
          f"open {open_gas:.2f} kg vs shut {shut_gas:.2e} kg")

    # The dynamic signature, AVERAGED OVER NOZZLES rather than drawn
    # once. A single seed said 6.26 rad/s against 1.36 and that was
    # an artifact of which nozzle happened to break: for four of the
    # ten it is the DEAD nozzle that spins the vehicle more. The
    # mean is the claim; the per-nozzle spread is the caveat.
    def spin_after(mode, k):
        en = FE.FailEnv("reveal", flat=True, fail_mode=mode,
                        fail_at=(0.05, 0.06), seed=11)
        en.reset(11)
        en.fault["nozzle"] = k
        for _ in range(60):
            en.step(np.full(en.act_dim, 0.35))
        return float(np.linalg.norm(en.omega()))

    sh = [spin_after("shut", k) for k in range(e.act_dim)]
    op = [spin_after("open", k) for k in range(e.act_dim)]
    check("stuck OPEN spins the vehicle harder ON AVERAGE",
          float(np.mean(op)) > float(np.mean(sh)),
          f"{np.mean(op):.2f} vs {np.mean(sh):.2f} rad/s "
          f"({np.mean(op) / np.mean(sh):.2f}x)")
    check("...but NOT for every nozzle — the per-draw claim is false",
          any(a > b for a, b in zip(sh, op)),
          f"{sum(1 for a, b in zip(sh, op) if a > b)} of {len(sh)} "
          f"nozzles spin MORE when dead than when latched")

    # ------------------------------------------------------------------
    # 3. The fault belongs to the seed.
    # ------------------------------------------------------------------
    print("\n  Both arms must meet the same breakages")
    for sd in (1, 7, 23):
        a = FE.FailEnv("reveal", flat=True, seed=0)
        b = FE.FailEnv("reveal", flat=True, seed=999)
        a.reset(sd)
        b.reset(sd)
        check(f"seed {sd} draws the same fault regardless of env seed",
              a.fault == b.fault, f"{a.fault}")
    none = FE.FailEnv("reveal", flat=True, fail_p=0.0, seed=3)
    none.reset(3)
    check("fail_p 0 means no fault at all", none.fault is None)
    check("...and the fault never fires at t=0",
          FE.FailEnv("reveal", flat=True, seed=4).reset(4) is not None
          and all(FE.FailEnv("reveal", flat=True, seed=s).fault["at"] > 0
                  for s in range(5)),
          "a vehicle born broken lets the policy fold the fault into "
          "its launch reflex instead of detecting it")

    # ------------------------------------------------------------------
    # 4. The ladder: inaction must be the worst outcome.
    # ------------------------------------------------------------------
    print("\n  The outcome reachable by doing nothing must be worst")
    env = FE.FailEnv("reveal", flat=True, seed=9)
    env.reset(9)

    def term(outcome, sink=0.0, tilt=0.0, spin=0.0):
        env.outcome = outcome
        a = math.radians(tilt)
        env.data.qpos[3:7] = [math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]
        env.data.qvel[:3] = [0.0, 0.0, -sink]
        env.data.qvel[3:6] = [spin, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)
        return env._terminal()

    t_out = term("timeout")
    c_hard = term("crash", sink=8.0)
    c_soft = term("crash", sink=1.0)
    l_hard = term("landed", sink=4.0, tilt=5.0)
    l_ok = term("landed", sink=0.3, tilt=2.0)
    ladder = [("timeout", t_out), ("crash hard", c_hard),
              ("crash soft", c_soft), ("landed hard", l_hard),
              ("landed well", l_ok)]
    bad = [b[0] for a, b in zip(ladder, ladder[1:]) if b[1] < a[1]]
    check("the ladder is monotone, worst first", not bad,
          " < ".join(f"{n} {v:.0f}" for n, v in ladder))
    check("TIMEOUT is the worst outcome, below even a hard crash",
          t_out < c_hard, f"{t_out:.0f} vs {c_hard:.0f}")
    check("the crash is GRADED by impact speed",
          c_soft > c_hard + 10.0,
          f"{c_soft:.0f} at 1 m/s vs {c_hard:.0f} at 8 m/s — on a "
          f"failure nothing can save, arriving slowly is the only "
          f"improvement left")

    # The counterexample, kept permanently: lab 4's ordering here.
    g, T = env.gamma, FE.TIME_COST
    def total(step, terminal):
        return terminal * g ** step - T * (1 - g ** step) / (1 - g)
    l4_hover = total(FE.MAX_STEPS, -90.0)
    l4_crash = total(200, -120.0 - 4.0 * 6.0)
    check("lab 4's ordering would make hovering BEAT crashing "
          "— the bug, pinned",
          l4_hover > l4_crash,
          f"hover {l4_hover:.0f} vs crash {l4_crash:.0f}; measured "
          f"consequence was 97% timeouts and under 1% landings")
    now_hover = total(FE.MAX_STEPS, t_out)
    now_crash = total(200, c_hard)
    check("...and this lab's ordering does not",
          now_hover < now_crash,
          f"hover {now_hover:.0f} vs crash {now_crash:.0f}")

    # ------------------------------------------------------------------
    # 5. Shaping: non-negative and small.
    # ------------------------------------------------------------------
    print("\n  Shaping must neither pay to stall nor charge to approach")
    phis = []
    for sd in range(20):
        env.reset(sd)
        phis.append(env._potential())
    check("the potential is NON-NEGATIVE", min(phis) >= 0.0,
          f"min {min(phis):.2f}")
    phi_max = 3.0 + 2.0
    toll = (1.0 - env.gamma) * phi_max * FE.MAX_STEPS
    check("...and its leak is small against the landing prize",
          toll < 10.0,
          f"toll {toll:.1f} per episode at Phi_max {phi_max:.0f}, "
          f"against a landing worth about 80")

    # ------------------------------------------------------------------
    # 6. The geometry, and what it does and does not say.
    # ------------------------------------------------------------------
    print("\n  The survivability audit")
    rv = fault.survey("reveal", "earth")
    check("`reveal` can hover when healthy", rv["healthy"])
    check("stuck OPEN is at least as fatal as stuck SHUT",
          len(rv["open_fatal"]) >= len(rv["shut_fatal"]),
          f"{len(rv['open_fatal'])}/{rv['n']} open vs "
          f"{len(rv['shut_fatal'])}/{rv['n']} shut")

    sx = fault.survey("sixlift", "mars")
    check("sixlift on Mars survives losing ANY one nozzle",
          len(sx["shut_fatal"]) == 0, f"{len(sx['shut_fatal'])}/{sx['n']}")
    check("...and is killed by ANY one sticking on",
          len(sx["open_fatal"]) == sx["n"],
          f"{len(sx['open_fatal'])}/{sx['n']} — redundancy is not "
          f"symmetric, and a review asking only 'what if one fails?' "
          f"passes this vehicle")

    rig = TH.rig("reveal")
    lifts = [k for k, t in enumerate(rig) if t.role == "lift"]
    zero = [k for k in range(rv["n"]) if rv["lift_shut"][k] < 1e-6]
    check("losing ANY corner lift nozzle zeroes torque-free lift",
          sorted(zero) == sorted(lifts),
          f"{zero} are exactly the {len(lifts)} lift nozzles — three "
          f"corners all pushing up cannot balance roll AND pitch")
    check("...while the healthy rig has plenty",
          rv["lift_healthy"] > 2.0, f"{rv['lift_healthy']:.2f}x weight")

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
