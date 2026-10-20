#!/usr/bin/env python3
"""
Does the simulation obey the equations it claims to? Residuals, measured.

    uv run verify.py                   # the whole audit
    uv run verify.py --world mars
    uv run verify.py --only momentum

WHY THIS FILE IS THE MOST IMPORTANT ONE HERE
---------------------------------------------
A learning curve cannot tell you that your physics is wrong. It will
rise just as happily against a broken simulator, and the policy will
exploit whatever you built rather than whatever you meant. Every
number this lab publishes is downstream of the integrator being
right, so the integrator gets audited before anything is published --
and the audit prints RESIDUALS rather than the word "passed", because
a tolerance nobody can see is a tolerance nobody checked.

This lab has already paid for skipping it once: it shipped a
compressibility of 1.95 chosen to make an arithmetic come out, and
every mass, impulse and endurance figure was wrong by a fifth.

WHAT IS CHECKED, AND WHAT WOULD BREAK IT
-----------------------------------------
  1. NEWTON. In free flight, m*a must equal the applied force. With
     mass leaving the vehicle this is the rocket equation, and the
     thing that gets it wrong is writing d(mv)/dt = F -- which adds
     a phantom force of v*dm/dt, about 400 N here.

  2. ANGULAR MOMENTUM. Ballistic, in vacuum, with no thrust: H in
     the WORLD frame is conserved. Gravity acts at the centre of
     mass so it exerts no torque. This one check covers the
     quaternion kinematics, the inertia tensor, and the gyroscopic
     coupling from four spinning wheels at once.

  3. ENERGY. Same conditions: kinetic plus rotational plus potential
     is constant. Catches a dissipative integrator, which is the
     usual way a tumble quietly damps itself.

  4. PROPELLANT. Mass consumed must equal the integral of throat
     flux times throat area. If thrust and mass flow are computed
     from different models, a policy can buy impulse it never paid
     for -- and nothing in the reward would notice.

  5. IMPULSE. The momentum change the thrusters produced must match
     the integral of their force. This is the end-to-end check that
     the wrench MuJoCo applied is the wrench the gas model sold.

  6. THERMODYNAMICS. Tank pressure, enthalpy and the nozzle solution
     against CoolProp itself, when it is installed.

Residuals are reported against the SCALE of the quantity, because
"3.8 N·m" means nothing without knowing the peak torque was 11 kN·m.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FAILS: list[str] = []


def report(name: str, resid: float, scale: float, tol: float,
           unit: str = "") -> None:
    rel = resid / max(abs(scale), 1e-12)
    ok = rel <= tol
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}")
    print(f"          residual {resid:.4g} {unit} against a "
          f"{scale:.4g} {unit} scale  =  {rel:.2e}  (tol {tol:.0e})")
    if not ok:
        FAILS.append(name)


def _prep(env, vacuum=True, free_wheels=False):
    """Put the vehicle somewhere these checks are meaningful."""
    import mujoco

    if vacuum:
        env.model.opt.density = 0.0
        env.model.opt.viscosity = 0.0
    if free_wheels:
        # Wheel hinges carry damping, which DISSIPATES energy. An
        # energy-conservation check against a model with a dashpot in
        # it is a check of the dashpot.
        env.model.dof_damping[:] = 0.0
    env.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return env


def _free_flight(env, steps, action, warmup: int = 0):
    """
    Roll forward recording SYSTEM quantities, not chassis ones.

    The distinction cost this audit five false failures. The vehicle
    is five bodies -- a chassis and four wheels that spin on their
    own hinges -- so the chassis's mass is 1662 kg of a 1750 kg
    machine, and its inertia tensor knows nothing about four wheels
    at a 1.35 m lever arm or about their spin. Hand-rolling
    `R @ (I * w)` measured a quantity that is not conserved and was
    never supposed to be.

    MuJoCo already computes the right things: `mj_subtreeVel` fills
    the centre-of-mass velocity and the angular momentum about it for
    every subtree, and `mjENBL_ENERGY` fills potential and kinetic
    energy for the whole system.
    """
    import mujoco

    # Let the valves settle first. They are a 25 ms first-order lag
    # behind a 10 ms transport delay, so a force sampled at the start
    # of an interval is not the force applied across it -- the Newton
    # and impulse residuals read 34% and 20% until the ramp was
    # allowed to finish. Thrust is only a constant once u_act has
    # converged.
    for _ in range(warmup):
        env.step(action)
    rows = []
    seen_contact = 0
    for _ in range(steps):
        d = env.data
        mujoco.mj_forward(env.model, d)
        seen_contact = max(seen_contact, int(d.ncon))
        mujoco.mj_subtreeVel(env.model, d)
        rows.append(dict(
            t=d.time,
            # The model's own total. `_apply_mass` has already put
            # the gas INTO body_mass[chassis], so adding m_gas again
            # double-counts it -- a 14% error, which is exactly the
            # size of the Newton residual it produced.
            m=float(env.model.body_mass.sum()),
            vcom=d.subtree_linvel[env.chassis].copy(),
            H=d.subtree_angmom[env.chassis].copy(),
            Epot=float(d.energy[0]), Ekin=float(d.energy[1]),
            qfrc=d.qfrc_actuator[:6].copy()))
        env.step(action)
    if seen_contact:
        raise AssertionError(
            f"the vehicle touched something ({seen_contact} contacts) "
            f"during a FREE-FLIGHT check — the residual would be a "
            f"measurement of the contact solver, not the integrator")
    return rows


def check_newton(env, world) -> None:
    """M*a_com = F_thrust + M*g, in free flight with mass leaving."""
    import mujoco

    _prep(env)
    env.reset(11)
    env.data.qpos[2] = 400.0                 # see check_angular_momentum
    env.data.qvel[:] = 0.0
    env.data.qvel[3:6] = [0.4, -0.2, 0.3]
    mujoco.mj_forward(env.model, env.data)

    u = np.zeros(env.act_dim)
    u[[i for i, t in enumerate(env.rig) if t.role == "lift"]] = 0.55
    rows = _free_flight(env, 120, u, warmup=80)

    worst = scale = 0.0
    for k in range(2, len(rows) - 2):
        dt = rows[k + 1]["t"] - rows[k - 1]["t"]
        a = (rows[k + 1]["vcom"] - rows[k - 1]["vcom"]) / dt
        m = rows[k]["m"]
        f_pred = rows[k]["qfrc"][:3] + np.array([0, 0, -m * world.g])
        worst = max(worst, float(np.abs(m * a - f_pred).max()))
        scale = max(scale, float(np.abs(f_pred).max()))
    report("Newton: M*a_com = F_thrust + M*g  (vacuum, mass dropping)",
           worst, scale, 2e-3, "N")


def check_angular_momentum(env) -> None:
    """H about the system CoM is constant for a ballistic tumble."""
    import mujoco

    _prep(env, free_wheels=True)
    env.reset(5)
    # 400 m, not 60. At 60 m the car reaches the ground 3.4 s into a
    # 4 s test and the last 0.6 s is a BOUNCE: energy fell 88% and
    # the check reported a broken integrator. A conservation test
    # that silently acquires a contact is measuring the contact.
    env.data.qpos[2] = 400.0
    env.data.qvel[:3] = 0.0
    env.data.qvel[3:6] = [2.4, -1.1, 0.8]    # a real tumble
    mujoco.mj_forward(env.model, env.data)

    rows = _free_flight(env, 400, np.zeros(env.act_dim))
    H = np.array([r["H"] for r in rows])
    report("angular momentum about the system CoM is conserved "
           "(vacuum, ballistic)",
           float(np.abs(H - H[0]).max()), float(np.abs(H[0]).max()),
           1e-5, "kg m^2/s")


def check_energy(env, world) -> None:
    """Potential + kinetic constant, ballistic in vacuum, no dampers."""
    import mujoco

    _prep(env, free_wheels=True)
    env.reset(7)
    env.data.qpos[2] = 400.0
    env.data.qvel[:3] = [3.0, -1.0, 6.0]
    env.data.qvel[3:6] = [1.5, 0.6, -0.9]
    mujoco.mj_forward(env.model, env.data)

    rows = _free_flight(env, 400, np.zeros(env.act_dim))
    E = np.array([r["Epot"] + r["Ekin"] for r in rows])
    report("energy conserved in ballistic flight (vacuum, no damping)",
           float(np.abs(E - E[0]).max()), float(np.abs(E).max()),
           1e-5, "J")


def check_propellant(env) -> None:
    """Gas consumed == integral of throat flux x throat area."""
    import mujoco

    import target_env as AE

    _prep(env)
    env.reset(3)
    env.data.qpos[2] = 400.0
    mujoco.mj_forward(env.model, env.data)

    u = np.full(env.act_dim, 0.7)
    m_start = env.m_gas
    predicted = 0.0
    for _ in range(300):
        # Integrate the same way the environment does, from the
        # tabulated flux rather than from thrust / a constant v_e.
        for _ in range(AE.SUBSTEPS):
            tk = AE.tank_state(env.m_gas - predicted * 0.0, env.beta,
                               env.world.name)
            predicted += (float(np.sum(env.u_act * env.at)) * tk["G"]
                          * AE.PHYS_DT)
            break
        env.step(u)
    used = m_start - env.m_gas
    report("propellant: mass used == integral of G_t * A_t",
           abs(used - env.gas_used), used, 1e-12, "kg")

    # And the independent cross-check: impulse delivered vs the
    # effective exhaust velocity that was sold for it.
    print(f"          (consumed {used:.2f} kg of {m_start:.0f} in "
          f"{300 / AE.CTRL_HZ:.1f} s at 70% on every valve)")


def check_impulse(env, world) -> None:
    """Momentum change == integral of thruster force, in free fall."""
    import mujoco

    _prep(env)
    env.reset(13)
    env.data.qpos[2] = 400.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    u = np.zeros(env.act_dim)
    u[[i for i, t in enumerate(env.rig) if t.role == "lift"]] = 0.8
    rows = _free_flight(env, 200, u, warmup=80)

    # sum m*dv, minus the gravity impulse, against sum F*dt
    dp = np.zeros(3)
    imp = np.zeros(3)
    for k in range(len(rows) - 1):
        dt = rows[k + 1]["t"] - rows[k]["t"]
        m = rows[k]["m"]
        dp += m * (rows[k + 1]["vcom"] - rows[k]["vcom"])
        dp -= np.array([0, 0, -m * world.g]) * dt
        imp += rows[k]["qfrc"][:3] * dt
    report("impulse: sum m*dv (less weight) == sum F_thrust*dt",
           float(np.abs(dp - imp).max()), float(np.abs(imp).max()),
           5e-3, "N s")


def check_wrench(env) -> None:
    """The matrix the lab reasons with IS the one MuJoCo applies."""
    import mujoco

    import thrusters as TH

    B = TH.effectiveness(env.rig)
    env.reset(0)
    d = env.data
    # IDENTITY ATTITUDE, deliberately. `qfrc_actuator[:3]` for a free
    # joint is a WORLD-frame force while B is a BODY-frame wrench;
    # they are the same numbers only when the body frame is the world
    # frame. Comparing them at the random tumbled attitude `reset`
    # produces reported a 51% error in a matrix that is exact.
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    worst, name = 0.0, ""
    for k, t in enumerate(env.rig):
        d.ctrl[:] = 0
        d.ctrl[env.aid[k]] = 1.0
        mujoco.mj_forward(env.model, d)
        e = float(np.abs(d.qfrc_actuator[:6] - B[:, k]).max())
        if e > worst:
            worst, name = e, t.name
    report("control effectiveness matrix == MuJoCo's applied wrench",
           worst, float(np.abs(B).max()), 1e-6, "N")


def check_integrators(rig_name: str, world_name: str) -> None:
    """
    Which integrator, and the answer is not topic 7's.

    Topic 7 adopted `implicitfast` after measuring it 3.9x faster
    than RK4 on a legged robot. That robot has feet on the ground
    almost every step, which is the contact-rich case the
    documentation recommends `implicitfast` for.

    This vehicle is BALLISTIC for most of its episode -- the
    energy-conserving case the same documentation reserves RK4 for.
    Same measurement, opposite answer, because the system changed.
    """
    import time

    import mujoco

    import car as CAR

    print("\n  Integrator, measured on a 4 s tumble in vacuum, no contacts:")
    print(f"  {'integrator':<14} {'dH/H':>11} {'dE/E':>11} {'steps/s':>10}")
    for integ in ("Euler", "implicitfast", "implicit", "RK4"):
        m, _r, _s = CAR.build(rig_name, flat=True, start_z=400.0,
                              integrator=integ)
        m.opt.density = 0.0
        m.opt.viscosity = 0.0
        m.dof_damping[:] = 0.0
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0, 0, 400]
        d.qvel[3:6] = [2.4, -1.1, 0.8]
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        ch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        H0 = d.subtree_angmom[ch].copy()
        E0 = float(d.energy.sum())
        t0 = time.time()
        for _ in range(2000):
            mujoco.mj_step(m, d)
        el = time.time() - t0
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        dH = float(np.abs(d.subtree_angmom[ch] - H0).max()
                   / max(np.abs(H0).max(), 1e-12))
        dE = abs((float(d.energy.sum()) - E0) / E0)
        star = "  <- in use" if integ == "RK4" else ""
        print(f"  {integ:<14} {dH:>11.2e} {dE:>11.2e} "
              f"{2000 / el:>10,.0f}{star}")
    print("  A 1% drift in angular momentum per four seconds is not a")
    print("  rounding error on a task whose whole content is arresting a")
    print("  tumble: the policy would spend propellant cancelling it.")


def check_thermo(world) -> None:
    """The tables against CoolProp, if it is installed."""
    import gas

    import thrusters as TH

    try:
        from CoolProp.CoolProp import PropsSI as P
    except ImportError:
        print("  SKIP  thermodynamics vs CoolProp — not installed")
        print("          (build-time only: "
              "`uv run --extra tables gen_tables.py --verify`)")
        return

    # Sample the OPERATING ENVELOPE, not an arbitrary rectangle.
    #
    # The worst error anywhere on the grid is 2.2%, on the saturation
    # dome around 103 K, where p(rho, T) has a kink that no bilinear
    # scheme handles. Whether that matters depends entirely on
    # whether the vehicle goes there.
    #
    # The first version of this comment asserted it does not -- "the
    # tank bottoms out at 145 K" -- which was a guess, and this
    # sampler immediately found 97 K and failed the check. It was
    # right to: at beta = 0 the tank really does cross the dome and
    # condense its propellant. `gas.blowdown_T` now REFUSES that
    # region, so the envelope below is the one the model will
    # actually produce.
    rng = np.random.default_rng(0)
    rho0 = TH.gas_mass() / TH.TANK_V
    ep, eh, where = [], [], (0.0, 0.0, 0.0)
    skipped = 0
    for _ in range(400):
        m = rng.uniform(3.0, TH.gas_mass())
        beta = rng.uniform(0.0, 1.0)
        r = m / TH.TANK_V
        try:
            T = gas.blowdown_T(r, rho0, TH.T_TANK, beta)
        except ValueError:
            skipped += 1            # two-phase; the model refuses it
            continue
        ref = P("P", "D", r, "T", T, "Nitrogen")
        e = abs(gas.tank_p(r, T) - ref) / ref
        if e > where[0]:
            where = (e, r, T)
        ep.append(e)
        eh.append(abs(gas.tank_h(r, T) - P("H", "D", r, "T", T, "Nitrogen")))
    report("tank pressure vs CoolProp, over the blowdown envelope",
           float(np.max(ep)), 1.0, 1e-3, "")
    print(f"          worst at rho = {where[1]:.0f} kg/m3, T = {where[2]:.0f} K; "
          f"mean {np.mean(ep):.3%}")
    print(f"          {skipped} of 400 draws were REFUSED as two-phase "
          f"(low beta, low fill) —\n          "
          f"a perfectly insulated cold-gas tank condenses its own "
          f"propellant")
    print(f"          (the model REFUSES them rather than interpolating "
          f"across the dome)")

    # The headline number, recomputed from the reference EOS.
    rho0 = P("D", "P", TH.TANK_P0, "T", TH.T_TANK, "Nitrogen")
    report("tank mass vs CoolProp directly",
           abs(TH.gas_mass() - rho0 * TH.TANK_V), rho0 * TH.TANK_V,
           2e-3, "kg")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--rig", default="reveal")
    ap.add_argument("--only", default="")
    a, _ = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    import thrusters as TH
    from target_env import TargetEnv

    world = TH.WORLDS[a.world]
    print("=" * 78)
    print(f"  PHYSICS AUDIT — rig {a.rig}, {world.name} "
          f"(g = {world.g:.3f}, p_amb = {world.p_amb / 1e3:.2f} kPa)")
    print("=" * 78)
    print("  Residuals, not verdicts. A tolerance nobody can see is a")
    print("  tolerance nobody checked.\n")

    def want(k):
        return (not a.only) or a.only == k

    if want("wrench"):
        check_wrench(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world))
    if want("newton"):
        check_newton(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world), world)
    if want("momentum"):
        check_angular_momentum(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world))
    if want("energy"):
        check_energy(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world), world)
    if want("propellant"):
        check_propellant(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world))
    if want("impulse"):
        check_impulse(TargetEnv(a.rig, flat=True, allow_flat=True, world=a.world), world)
    if want("thermo"):
        check_thermo(world)
    if want("integrators"):
        check_integrators(a.rig, a.world)

    print("\n" + "=" * 78)
    if FAILS:
        print(f"  {len(FAILS)} RESIDUAL(S) OVER TOLERANCE: {', '.join(FAILS)}")
    else:
        print("  every residual inside tolerance")
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
