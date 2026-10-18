#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0", "scipy>=1.11"]
# ///
"""
Topic 8 Lab 1 — the rig is parameters, and the physics must be real.

What is worth testing here is decided by what could be quietly wrong
rather than by what is easy to assert:

  1. THE EFFECTIVENESS MATRIX IS THE LAB'S WHOLE ARGUMENT. The claim
     "placement beats count" is read off B before any training, so if
     B disagrees with the simulator every conclusion is decoration.
     It is therefore checked AGAINST MUJOCO: push one nozzle, read
     the generalised force, compare to the column.

  2. ACTUATORS RESOLVE BY NAME, AND A MISS MUST RAISE. Topic 7
     shipped `ctrl[6] = yaw` three times; each time the index
     silently addressed something else, the robot stood perfectly
     still at a 0% fall rate, and three runs were blamed on the
     observation width. A name filter that returns nothing makes
     `ctrl[[]] = []` a no-op.

  3. THE REAL-GAS ARITHMETIC. The ideal gas law says 600 L at 689 bar
     holds 475 kg of nitrogen, which is denser than LIQUID nitrogen.
     The whole lab is sized from the corrected 244 kg, so the
     correction is asserted — including that it points the way it
     does.

  4. A RANK-DEFICIENT RIG MUST BE DETECTABLE. `waisted` cannot
     produce a pitch torque at any command. That is a hardware fact
     no policy can learn around, and the audit has to say so.

  5. COSMETICS MUST NOT CARRY MASS. The 24-slice lofted hull is
     decoration; it once added 394 kg of inertia nobody chose, which
     would have been invisible in every learning curve.

No GPU, no training, no downloads. Properties, not shapes.

    uv run tests/test_aerial_stabilization.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "08_aerial_physics" / "01_aerial_stabilization"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mujoco

    import car as CAR
    import terrain
    import thrusters as TH

    print("=" * 76)
    print("  TOPIC 8 LAB 1 — aerial stabilization")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. Propellant arithmetic — the number the lab is sized from.
    # ------------------------------------------------------------------
    print("\n  The tank")
    m_real = TH.gas_mass_from_tank()
    m_ideal = TH.TANK_P0 / (TH.R_N2 * TH.T_TANK) * TH.TANK_V
    check("the stated gas mass is the one the constants imply",
          abs(m_real - TH.M_GAS_0) < 1.0, f"{m_real:.1f} vs {TH.M_GAS_0}")
    check("the ideal gas law OVERSTATES it, and by a lot",
          m_ideal > m_real * 1.5, f"ideal {m_ideal:.0f} kg vs real {m_real:.0f}")
    check("...to a density above LIQUID nitrogen (807 kg/m³) — the tell",
          m_ideal / TH.TANK_V > 790, f"{m_ideal / TH.TANK_V:.0f} kg/m³")
    check("Isp 73 s gives the exhaust velocity used",
          abs(TH.VE_IDEAL - 73.0 * 9.80665) < 1e-6)
    hov = TH.hover_seconds()
    check("the whole tank buys SECONDS of hover, not minutes",
          4.0 < hov < 15.0, f"{hov:.1f} s")

    import aerial_env as AE

    p0, t0, a0 = AE.tank_state(TH.M_GAS_0)
    p1, t1, a1 = AE.tank_state(TH.M_GAS_0 * 0.25)
    check("pressure falls as gas leaves", p1 < p0 * 0.5,
          f"{p0/1e5:.0f} → {p1/1e5:.0f} bar")
    check("and the tank COOLS doing it (adiabatic blowdown)",
          t1 < t0 - 30.0, f"{t0:.0f} → {t1:.0f} K")
    _p, t_iso, _a = AE.tank_state(TH.M_GAS_0 * 0.25, beta=1.0)
    check("beta=1 is isothermal — the knob does what it says",
          abs(t_iso - TH.T_TANK) < 1.0, f"{t_iso:.0f} K")
    check("availability saturates at 1 while the regulator holds", a0 == 1.0)
    check("and collapses when the tank falls below dropout",
          AE.tank_state(6.0)[2] < 0.35, f"{AE.tank_state(6.0)[2]:.2f}")

    # ------------------------------------------------------------------
    # 2. The rig generator.
    # ------------------------------------------------------------------
    print("\n  A rig is parameters")
    r10 = TH.rig("reveal")
    check("the quoted ten nozzles come out of the generator",
          len(r10) == 10, f"{len(r10)}")
    check("every nozzle has a unique name",
          len({t.name for t in r10}) == 10)
    check("every direction is a unit vector",
          all(abs(np.linalg.norm(t.direction) - 1) < 1e-9 for t in r10))
    for bad, why in (({"pairs": 0}, "zero stations"),
                     ({"aft": -1}, "negative aft pairs")):
        try:
            TH.make_rig(**bad)
            check(f"make_rig REFUSES {why}", False)
        except ValueError:
            check(f"make_rig REFUSES {why}", True)

    # Equal installed thrust is the control the sweep depends on.
    tot = {}
    for nm in ("minimal", "reveal", "wide", "dense"):
        rr = TH.rig(nm)
        tot[nm] = sum(t.fmax * t.direction[2] for t in rr if t.role == "lift")
    check("every rig is built at EQUAL installed lift — the control",
          max(tot.values()) - min(tot.values()) < 1.0,
          ", ".join(f"{k} {v/1000:.1f}kN" for k, v in tot.items()))

    # ------------------------------------------------------------------
    # 3. Authority: the claim the lab makes before training.
    # ------------------------------------------------------------------
    print("\n  What a rig can and cannot do")
    B = TH.effectiveness(r10)
    check("B is 6 x N", B.shape == (6, 10), str(B.shape))
    check("reveal spans all six wrench axes",
          np.linalg.matrix_rank(B, tol=1e-6) == 6)

    roll_free = TH.max_wrench(B, 3)
    roll_pure, u = TH.max_pure_couple(B, 0)
    check("a PURE couple is strictly weaker than a free-force one",
          0 < roll_pure < roll_free,
          f"{roll_pure:,.0f} vs {roll_free:,.0f} N·m")
    if u is not None:
        w = B @ u
        check("...and the pure solution really has no net force",
              float(np.abs(w[:3]).max()) < 1.0,
              f"residual {float(np.abs(w[:3]).max()):.3e} N")

    # The counterexample, kept permanently: lift-only rigs cannot
    # produce a couple at all, and the audit must SAY so rather than
    # returning a plausible number.
    for nm in ("minimal", "sixlift"):
        d = TH.audit(nm, TH.rig(nm))
        check(f"{nm} (lift only) has NO pure roll couple",
              d["roll_pure_Nm"] < 1.0, f"{d['roll_pure_Nm']:.1f} N·m")
        check(f"...so its 360° roll time is infinite, not a number",
              not math.isfinite(d["min_roll_time_s"]))

    dw = TH.audit("waisted", TH.rig("waisted"))
    dm = TH.audit("minimal", TH.rig("minimal"))
    check("waisted is rank-DEFICIENT — an axis is unreachable",
          dw["rank"] == 5, f"rank {dw['rank']}")
    check("...specifically pitch, because it has no fore-aft arm",
          dw["pitch_pure_Nm"] < 1.0, f"{dw['pitch_pure_Nm']:.1f} N·m")
    check("...while keeping full roll at the SAME nozzle count as minimal",
          dw["n_thrusters"] == dm["n_thrusters"] and dw["roll_pure_Nm"] > 1e3,
          f"both N={dw['n_thrusters']}, waisted roll "
          f"{dw['roll_pure_Nm']:,.0f} N·m")

    # The lab's thesis, checkable in closed form.
    d10, d20 = TH.audit("reveal", TH.rig("reveal")), \
        TH.audit("dense", TH.rig("dense"))
    check("DOUBLING the nozzle count buys no extra roll authority",
          abs(d10["roll_pure_Nm"] - d20["roll_pure_Nm"]) < 1.0,
          f"10 → {d10['roll_pure_Nm']:,.0f}, 20 → {d20['roll_pure_Nm']:,.0f}")
    dn = TH.audit("narrow", TH.rig("narrow"))
    check("...but HALVING the track halves it, at the same count",
          dn["roll_pure_Nm"] < d10["roll_pure_Nm"] * 0.6,
          f"{dn['roll_pure_Nm']:,.0f} vs {d10['roll_pure_Nm']:,.0f}")

    # ------------------------------------------------------------------
    # 4. The model agrees with the matrix. This is the load-bearing one.
    # ------------------------------------------------------------------
    print("\n  MuJoCo agrees with the matrix")
    model, rig_, _s = CAR.build("reveal", flat=True)
    ids = CAR.resolve(model, rig_)
    check("all ten actuators resolve by name", len(ids) == 10)
    check("and they are distinct", len(set(ids.tolist())) == 10)

    data = mujoco.MjData(model)
    worst = 0.0
    for k in range(len(rig_)):
        data.ctrl[:] = 0
        data.ctrl[ids[k]] = 1.0
        mujoco.mj_forward(model, data)
        worst = max(worst, float(np.abs(data.qfrc_actuator[:6] - B[:, k]).max()))
    scale = float(np.abs(B).max())
    check("the wrench MuJoCo applies IS the column thrusters.py claims",
          worst / scale < 1e-6, f"{worst:.4f} N on a {scale:,.0f} N scale")

    # A miss must raise rather than silently doing nothing.
    class _Ghost:
        name = "nozzle_that_does_not_exist"
    try:
        CAR.resolve(model, list(rig_) + [_Ghost()])
        check("resolving a missing actuator RAISES", False,
              "it returned quietly — this is the topic-7 bug class")
    except (KeyError, ValueError):
        check("resolving a missing actuator RAISES", True)

    # ------------------------------------------------------------------
    # 5. Mass, inertia, and cosmetics that must not weigh anything.
    # ------------------------------------------------------------------
    print("\n  The vehicle")
    check("dry mass is the stated dry mass — the 24-slice loft is free",
          abs(float(model.body_mass.sum()) - TH.M_DRY) < 1.0,
          f"{float(model.body_mass.sum()):.1f} kg vs {TH.M_DRY:.0f}")
    m_wide, _r, _ = CAR.build("wide", flat=True)
    check("...and it does not change when the rig does",
          abs(float(m_wide.body_mass.sum()) - TH.M_DRY) < 1.0,
          f"{float(m_wide.body_mass.sum()):.1f} kg")
    I = TH.inertia()
    check("gas adds to ROLL and YAW but not pitch (the tanks are outboard)",
          I[0] > TH.I_DRY[0] and I[2] > TH.I_DRY[2]
          and abs(I[1] - TH.I_DRY[1]) < 1e-9,
          f"{np.round(I, 1)}")

    # ------------------------------------------------------------------
    # 6. The environment's gates must be able to fire.
    # ------------------------------------------------------------------
    print("\n  The task")
    env = AE.AerialEnv("reveal", flat=True)
    check("observation width is DERIVED, not declared",
          env.obs_dim == len(env._obs()), f"{env.obs_dim}")
    check("action width is the nozzle count",
          env.act_dim == len(env.rig) == 10)
    e6 = AE.AerialEnv("wide", flat=True)
    check("...and it follows the rig rather than a default argument",
          e6.act_dim == 14, f"{e6.act_dim}")

    check("wheels and hull are told apart BY NAME",
          len(env.wheel_geoms) == 4 and len(env.hull_geoms) >= 1,
          f"{len(env.wheel_geoms)} wheels, {len(env.hull_geoms)} hull")
    check("they are disjoint sets",
          not (env.wheel_geoms & env.hull_geoms))

    # Clearance must read the LOWEST point, not the centre of mass.
    # Topic 7 measured a body half a limb short by confusing them.
    env.reset(0)
    env.data.qpos[:3] = [0, 0, 3.0]
    env.data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    up_clear = env.clearance()
    ang = math.radians(60)
    env.data.qpos[3:7] = [math.cos(ang / 2), math.sin(ang / 2), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    check("tilting the car LOWERS its clearance at the same CG height",
          env.clearance() < up_clear - 0.3,
          f"{up_clear:.2f} m upright → {env.clearance():.2f} m at 60°")
    check("clearance is below the CG height (it is a corner, not the centre)",
          up_clear < 3.0)

    # Potential-based shaping must telescope, or it is not
    # potential-based and the optimality guarantee is gone.
    env.reset(3)
    phis = []
    for _ in range(12):
        phis.append(env._prev_phi)
        env.step(np.full(env.act_dim, 0.4))
    check("the shaping potential moves with the state",
          len(set(np.round(phis, 6))) > 1)
    env.reset(3)
    lo_phi = env._potential()
    env.data.qpos[2] += 4.0
    mujoco.mj_forward(env.model, env.data)
    check("...and is LOWER when higher up — the agent is told which way "
          "is down", env._potential() < lo_phi,
          f"{lo_phi:.2f} → {env._potential():.2f}")

    # ------------------------------------------------------------------
    # 7. Terrain.
    # ------------------------------------------------------------------
    print("\n  The ground")
    flat = terrain.generate(0, "pad")
    rough = terrain.generate(0, "rough")
    check("the `pad` preset really is flat",
          float(np.ptp(flat.heights)) < 1e-9)
    check("`rough` is not, and carries rocks",
          float(np.ptp(rough.heights)) > 1.0 and len(rough.rocks) > 0,
          f"relief {float(np.ptp(rough.heights)):.2f} m, "
          f"{len(rough.rocks)} rocks")
    check("the landing pad at the origin is flattened in every preset",
          max(rough.slope_deg_at(0, 0), rough.slope_deg_at(1, 1)) < 1.0,
          f"{rough.slope_deg_at(0, 0):.2f}°")
    check("...but it is rough away from it",
          max(rough.slope_deg_at(x, y)
              for x in (12, 18) for y in (-8, 9)) > 3.0)
    check("no rock is dropped on the pad",
          all(math.hypot(r.x, r.y) > 8.0 for r in rough.rocks))
    check("generation is deterministic in the seed",
          float(np.abs(terrain.generate(5, "rough").heights
                       - terrain.generate(5, "rough").heights).max()) == 0.0)
    check("and different seeds differ",
          float(np.abs(terrain.generate(1, "rough").heights
                       - terrain.generate(2, "rough").heights).max()) > 0.1)

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
