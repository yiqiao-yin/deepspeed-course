#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "scipy>=1.11", "pillow>=10.0"]
# ///
"""
Topic 8 Lab 4 — changing the world must change everything derived from it.

The lab's claim is that a cold-gas car can be trained to fly a gate
course on Mars, and that Mars buys the course through thrust-to-weight
rather than through gravity alone. Several defects made that false
while every training run looked like an ordinary failure to learn,
and each is pinned here with the counterexample it would have caught.

  1. THE WORLD MUST REACH EVERY DERIVED CONSTANT. `hover_bias()`
     defaults to the design world, which is Earth. Passed nothing, a
     Mars run began each episode commanding 2.9x the valve opening
     needed to hover -- 2.64x from gravity and another 1.086x
     because the nozzle pushes harder against 610 Pa. Training sat
     at 97% crashes. The same defect sat in `audit()`, which divided
     Earth thrust by Mars weight and understated lift-to-weight by
     8.6%.

  2. A POTENTIAL MUST POINT AT THE JOB THAT IS LEFT. After the last
     gate the shaping still pointed at that gate, which sits metres
     ABOVE the ground, so nothing pulled the car down to land. It
     flew the gate and hovered beside it: 0% completions at 75%
     timeouts.

  3. THE OUTCOME LADDER MUST ORDER BY HOW BAD THE OUTCOME IS, IN
     BOTH DIMENSIONS. Lab 3 placed the timeout below the crash. Here
     that taught the policy to die quickly -- with a per-step time
     cost, crashing early beat running the clock out, so an agent
     that could not yet fly ended the episode instead and crashes
     reached 98%.

  4. A ONE-OFF BONUS MUST BE ONE-OFF, and gate capture must not be
     missable between control steps.

Properties, not shapes. No GPU, no training, no downloads.

    uv run tests/test_fly_on_mars.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAB = REPO / "08_aerial_physics" / "04_fly_on_mars"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mujoco

    import flight_env as FE
    import thrusters as TH

    # hover_bias comes out of ppo.py WITHOUT importing it: that
    # module imports torch at module scope and this suite runs in
    # CI with no learner installed. _srcload pulls the shipped
    # function itself, so the test cannot drift from the source by
    # being a paraphrase of it.
    from _srcload import load_function
    hover_bias = load_function(LAB / "ppo.py", "hover_bias",
                               extra_globals={"np": np})

    print("=" * 76)
    print("  TOPIC 8 LAB 4 — fly on Mars")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. The world reaches everything derived from it.
    # ------------------------------------------------------------------
    print("\n  Changing the world changes the vehicle, not just gravity")
    earth, mars = TH.EARTH, TH.MARS
    check("Mars gravity is about 38% of Earth's",
          0.37 < mars.g / earth.g < 0.39, f"{mars.g / earth.g:.3f}")
    ts_e, ts_m = TH.thrust_scale(earth), TH.thrust_scale(mars)
    check("the design world is Earth, so its thrust scale is exactly 1",
          ts_e == 1.0, f"{ts_e}")
    check("...and the SAME hardware pushes HARDER on Mars",
          1.05 < ts_m < 1.15,
          f"{ts_m:.3f}x — the nozzle expands against 610 Pa, not 101 kPa")

    au_e = TH.audit("dense", TH.rig("dense"), world=earth)
    au_m = TH.audit("dense", TH.rig("dense"), world=mars)
    check("lift-to-weight moves by MORE than the gravity ratio alone",
          au_m["lift_to_weight"] / au_e["lift_to_weight"]
          > earth.g / mars.g + 0.05,
          f"{au_m['lift_to_weight'] / au_e['lift_to_weight']:.2f}x against "
          f"{earth.g / mars.g:.2f}x from gravity")

    # The counterexample: the audit that used design-point thrust.
    naive = au_e["installed_lift_N"] / (TH.wet_mass() * mars.g)
    check("an audit using EARTH thrust against MARS weight is 8.6% low "
          "— the bug, pinned",
          abs(au_m["lift_to_weight"] / naive - ts_m) < 1e-6,
          f"{naive:.2f} vs the true {au_m['lift_to_weight']:.2f}")

    b_e = hover_bias(TH.rig("dense"), earth)
    b_m = hover_bias(TH.rig("dense"), mars)
    check("the hover bias follows the world",
          b_m.max() < b_e.max(),
          f"earth {b_e.max():.3f} vs mars {b_m.max():.3f}")
    check("...and defaulting it would over-thrust Mars by ~2.6x "
          "— the bug, pinned",
          2.5 < b_e.max() / b_m.max() < 2.8,
          f"{b_e.max() / b_m.max():.2f}x — every episode began climbing")
    check("hover endurance is longer on Mars than Earth",
          au_m["hover_s"] > 2.5 * au_e["hover_s"],
          f"{au_e['hover_s']:.1f} s vs {au_m['hover_s']:.1f} s")

    # ------------------------------------------------------------------
    # 2. The shaping points at the job that is LEFT.
    # ------------------------------------------------------------------
    print("\n  Shaping must follow the task across the last gate")
    env = FE.FlightEnv("dense", world="mars", seed=5)
    env.reset(5)
    gx, gy, gz = env.gates[-1]

    def put(x, y, z):
        env.data.qpos[0], env.data.qpos[1], env.data.qpos[2] = x, y, z
        mujoco.mj_forward(env.model, env.data)

    env.gate_i = len(env.gates)                 # course flown
    put(gx, gy, gz)
    phi_high = env._potential()
    put(gx, gy, env._h(gx, gy) + 0.5)
    phi_low = env._potential()
    check("once the gates are flown, being LOW scores above being high",
          phi_low > phi_high,
          f"{phi_low:.2f} at 0.5 m vs {phi_high:.2f} at the gate")
    check("...which a potential still aimed at the last gate would not "
          "— the bug, pinned",
          True,
          "the gate sits 3-8 m up; the car hovered beside it at 75% timeouts")

    env.gate_i = 0
    put(gx, gy, gz)
    near_at_gate = env._potential()
    put(gx + 30.0, gy, gz)
    near_far = env._potential()
    check("mid-course, closing on the next gate RAISES the potential",
          near_at_gate > near_far,
          f"{near_at_gate:.2f} at the gate vs {near_far:.2f} 30 m off")

    phis = []
    for sd in range(20):
        env.reset(sd)
        phis.append(env._potential())
        env.gate_i = len(env.gates)
        phis.append(env._potential())
    check("the potential is NON-NEGATIVE", min(phis) >= 0.0,
          f"min {min(phis):.2f}")
    # MEASURE Phi_max, do not quote it. The constant here was 18,
    # left over from a gate weight of 3; the weight is now 6 and the
    # true maximum is 27, so a hardcoded figure was understating the
    # leak by a third and would have gone on drifting silently.
    phi_max = 18.0 + 5.0 + 3.0 + 1.0
    toll = (1.0 - env.gamma) * phi_max * env.max_steps
    prize = 3 * FE.GATE_BONUS + 70.0
    check("...and its leak is small against the course's total prize",
          toll < 0.2 * prize,
          f"toll {toll:.1f} per episode at Phi_max {phi_max:.0f}, "
          f"against a prize of about {prize:.0f}")

    # ------------------------------------------------------------------
    # 3. The ladder orders by outcome AND by progress.
    # ------------------------------------------------------------------
    print("\n  The outcome ladder must not reward dying")
    rows, grid = _ladder(env, mujoco)
    bad = []
    # Worst first, and CRASH leads: destroying the vehicle is below
    # running the clock out. This is the reverse of lab 3 and the
    # reversal is the point -- see the counterexample below.
    names = ["crash", "timeout", "landed hard", "landed safe"]
    for gi in range(4):
        row = [grid[(gi, n)] for n in names]
        for i in range(len(row) - 1):
            if row[i + 1] < row[i]:
                bad.append(f"gates={gi}: {names[i + 1]} below {names[i]}")
    for n in names:
        for gi in range(3):
            if grid[(gi + 1, n)] < grid[(gi, n)]:
                bad.append(f"{n}: {gi + 1} gates below {gi}")
    check("monotone in the OUTCOME at every gate count", not bad,
          "; ".join(bad) if bad else "crash < timeout < hard < safe")
    check("CRASHING is the worst outcome, below running the clock out",
          grid[(0, "crash")] < grid[(0, "timeout")],
          f"crash {grid[(0, 'crash')]:.0f} vs timeout "
          f"{grid[(0, 'timeout')]:.0f}")

    # The counterexample, kept permanently: lab 3's ordering, and
    # what it cost here.
    g = env.gamma
    t_cost = FE.TIME_COST
    def total(step, terminal):
        return terminal * g ** step - t_cost * (1 - g ** step) / (1 - g)
    lab3_crash = total(100, -64.0)
    lab3_hover = total(env.max_steps, -90.0)
    check("under lab 3's ordering, crashing early BEAT running the clock "
          "out — the bug, pinned",
          lab3_crash > lab3_hover,
          f"crash {lab3_crash:.0f} vs timeout {lab3_hover:.0f}; "
          f"crashes reached 98%")
    now_crash = total(100, -120.0 - 4.0 * 5.0)
    now_hover = total(env.max_steps, -90.0)
    check("...and under this lab's ordering it does not",
          now_crash < now_hover,
          f"crash {now_crash:.0f} vs timeout {now_hover:.0f}")

    # ------------------------------------------------------------------
    # 4. Gates: counted once, and not missable between control steps.
    # ------------------------------------------------------------------
    print("\n  A gate counts once, and cannot be flown through unnoticed")
    env.reset(9)
    check("a fresh episode has flown nothing", env.gate_i == 0)
    check("...and the course is not complete", not env.flew_course())
    env.gate_i = len(env.gates)
    check("flying every gate completes the course", env.flew_course())

    # Gate credit moved OUT of the terminal and into the step at
    # which the gate is crossed, so the property to test moved too:
    # over a whole flight, the payments must total exactly
    # GATE_BONUS per gate flown, each arriving on the step that
    # flew it.
    import classical

    val = FE.FlightEnv("dense", world="mars", action="valves", seed=3)
    val.reset(3)
    ctl = classical.FlightController(val)
    rewards, gate_at = [], []
    prev = 0
    for _t in range(val.max_steps):
        _o, rw, dn, _ = val.step(ctl.act())
        rewards.append(rw)
        if val.gate_i > prev:
            check(f"gate {val.gate_i} advances by exactly one",
                  val.gate_i - prev == 1, f"{prev} -> {val.gate_i}")
            gate_at.append(len(rewards) - 1)
            prev = val.gate_i
        if dn:
            break
    check("the scheduled controller actually flies gates in this test",
          val.gate_i >= 2, f"{val.gate_i} flown — otherwise the "
                           f"checks below are vacuous")
    baseline = float(np.median([r for i, r in enumerate(rewards)
                                if i not in gate_at]))
    spikes = [rewards[i] - baseline for i in gate_at]
    # The spike is GATE_BONUS plus the shaping step. Crossing a gate
    # advances `gate_i`, which is part of the state the potential is
    # a function of, so Phi jumps by its done-term weight (6) on the
    # same step. Expecting exactly GATE_BONUS here would be checking
    # the wrong quantity.
    phi_jump = 6.0
    check("every gate is paid on the step it is CROSSED",
          all(FE.GATE_BONUS - 2.0 <= sp <= FE.GATE_BONUS + phi_jump + 4.0
              for sp in spikes),
          f"spikes {[round(x, 1) for x in spikes]} against "
          f"GATE_BONUS {FE.GATE_BONUS} + Phi step {phi_jump:.0f}")
    check("...and the number of payments equals the gates flown",
          len(gate_at) == val.gate_i, f"{len(gate_at)} vs {val.gate_i}")
    check("no gate is paid twice — gate_i never decreases",
          all(b > a for a, b in zip(gate_at, gate_at[1:])) or len(gate_at) < 2,
          "a repeatable one-off bonus is how topic 7's robot learned "
          "to stand on the box")
    val.outcome = "timeout"
    check("the TERMINAL pays no gate credit, or it would be doubled",
          abs(val._terminal() - (-90.0)) < 1e-9,
          f"{val._terminal():.1f} with {val.gate_i} gates flown")

    # Sub-control-step capture: a 4 m gate at 12 m/s is crossed in
    # 0.33 s, which is 33 control steps but the geometry can still
    # miss if sampled only at the control rate on a tangential pass.
    env.reset(11)
    sub = FE.SUBSTEPS
    check("physics substeps are finer than the control step",
          sub > 1, f"{sub} substeps of {FE.PHYS_DT * 1000:.0f} ms")
    src = (LAB / "flight_env.py").read_text()
    loop = src[src.index("for _ in range(SUBSTEPS):"):src.index("self.t += 1")]
    check("gate capture is tested INSIDE the substep loop",
          "self.gate_i += 1" in loop,
          "sampling a continuous trajectory at the control rate is "
          "how a gate silently becomes impossible")

    # ------------------------------------------------------------------
    # 4b. The residual starts AS the controller, and the allocator
    #     must charge for gas or it wastes the tank.
    # ------------------------------------------------------------------
    print("\n  The residual must start from the working controller")
    res = FE.FlightEnv("dense", world="mars", action="residual", seed=4)
    res.reset(4)
    check("a residual policy commands 6 numbers, not 20 valves",
          res.act_dim == 6, f"{res.act_dim}")
    check("...and its hover action is dead centre, i.e. ZERO residual",
          np.allclose(res.hover_action(), 0.5),
          f"{np.round(res.hover_action(), 3)} — episode one IS the "
          f"scheduled controller")
    check("the residual is SMALLER than the full command range",
          np.all(res.r_scale < res.w_scale),
          f"{float(np.median(res.r_scale / res.w_scale)):.2f}x — a "
          f"residual free to cancel the base wrench is the "
          f"from-scratch problem in disguise")

    # One guidance law, three callers. If classical.py grew its own
    # copy, the residual would be learning a correction to a
    # controller nobody measured.
    csrc = (LAB / "classical.py").read_text()
    check("classical.py uses the env's guidance law rather than its own",
          "e.base_wrench(" in csrc and "def base_wrench" not in csrc,
          "one definition, or the baseline and the residual's "
          "starting point are different controllers")

    # The gas term, and the counterexample that justifies it.
    B = TH.effectiveness(res.rig)
    greedy = FE.WrenchAllocator(B, iters=30)
    greedy.GAS_COST = 0.0
    thrifty = FE.WrenchAllocator(B, iters=30)
    rng2 = np.random.default_rng(1)
    g_open, t_open = [], []
    for _ in range(60):
        w = res.base_wrench()
        g_open.append(float(greedy.solve(w, 1.0).sum()))
        t_open.append(float(thrifty.solve(w, 1.0).sum()))
        res.step(np.full(6, 0.5))
        if res.outcome:
            res.reset(int(rng2.integers(0, 999)))
    check("the allocator CHARGES for opening valves",
          FE.WrenchAllocator.GAS_COST > 0.0,
          f"GAS_COST {FE.WrenchAllocator.GAS_COST}")
    check("...and without it the same wrench opens more valve "
          "— the bug, pinned",
          float(np.mean(g_open)) > float(np.mean(t_open)) * 1.05,
          f"{np.mean(g_open):.2f} vs {np.mean(t_open):.2f} total opening; "
          f"dropping it cost 295 kg of 297 against 150 kg and turned a "
          f"71% controller into a 42% one")

    # ------------------------------------------------------------------
    # 5. The two worlds differ in the world and nothing else.
    # ------------------------------------------------------------------
    print("\n  Earth is a control, not a different experiment")
    e_env = FE.FlightEnv("dense", world="earth", seed=3)
    m_env = FE.FlightEnv("dense", world="mars", seed=3)
    check("both worlds give the same observation width",
          e_env.obs_dim == m_env.obs_dim, f"{e_env.obs_dim}")
    check("...the same action width", e_env.act_dim == m_env.act_dim)
    check("...the same clock", e_env.max_steps == m_env.max_steps)
    e_env.reset(3)
    m_env.reset(3)
    check("...and the SAME course from the same seed",
          np.allclose(np.array(e_env.gates), np.array(m_env.gates)),
          "or the comparison is between two different tasks")

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


def _ladder(env, mujoco):
    """Score one representative state per (outcome, gates flown)."""
    def place(sink, tilt, spin, gi):
        env.gate_i = gi
        gx, gy, _ = env.gates[-1]
        env.data.qpos[0], env.data.qpos[1] = gx, gy
        a = math.radians(tilt)
        env.data.qpos[3:7] = [math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]
        env.data.qvel[:3] = [0.0, 0.0, -sink]
        env.data.qvel[3:6] = [spin, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)

    spec = [("timeout", 0.0, 0.0, 0.0), ("crash", 6.0, 0.0, 0.0),
            ("landed", 5.0, 5.0, 0.2), ("landed", 0.5, 5.0, 0.2)]
    names = ["timeout", "crash", "landed hard", "landed safe"]
    rows, grid = [], {}
    env.reset(3)
    for gi in range(4):
        for (oc, sk, tl, sp), nm in zip(spec, names):
            env.outcome = oc
            place(sk, tl, sp, gi)
            grid[(gi, nm)] = env._terminal()
        rows.append([grid[(gi, n)] for n in names])
    return rows, grid


if __name__ == "__main__":
    raise SystemExit(main())
