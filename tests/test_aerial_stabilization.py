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

    import aerial_env as AE
    import car as CAR
    import terrain
    import thrusters as TH

    print("=" * 76)
    print("  TOPIC 8 LAB 1 — aerial stabilization")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. Propellant arithmetic — the number the lab is sized from.
    # ------------------------------------------------------------------
    print("\n  The tank — a reference equation of state, not a guess")
    import gas

    m_real = TH.gas_mass()
    m_ideal = TH.TANK_P0 / (gas.R_N2 * TH.T_TANK) * TH.TANK_V
    rho0 = m_real / TH.TANK_V
    Z = gas.compressibility(rho0, TH.T_TANK)
    check("compressibility at 689 bar is ~1.60, from CoolProp's EOS",
          abs(Z - 1.602) < 0.01, f"Z = {Z:.3f}")
    check("...which this lab first GUESSED at 1.95 and got 244 kg",
          abs(m_real - 296.8) < 1.5, f"{m_real:.1f} kg")
    check("the ideal gas law overstates the fill by ~60%",
          1.5 < m_ideal / m_real < 1.75, f"ideal {m_ideal:.0f} kg")
    check("...to a density above LIQUID nitrogen (807 kg/m³) — the tell",
          m_ideal / TH.TANK_V > 790, f"{m_ideal / TH.TANK_V:.0f} kg/m³")

    # Blowdown: the tank cools, and that is not a detail.
    t_full = gas.blowdown_T(rho0, rho0, TH.T_TANK, 0.30)
    t_low = gas.blowdown_T(0.1 * rho0, rho0, TH.T_TANK, 0.30)
    check("the tank COOLS as it empties (isentropic blowdown)",
          t_low < t_full - 80.0, f"{t_full:.0f} K → {t_low:.0f} K")
    check("beta = 1 is isothermal — the knob does what it says",
          abs(gas.blowdown_T(0.1 * rho0, rho0, TH.T_TANK, 1.0)
              - TH.T_TANK) < 1.0)

    # The saturation guard. A perfectly insulated cold-gas tank
    # condenses its own propellant, and the single-phase tables do
    # not describe that -- so the model must REFUSE it rather than
    # interpolate across the dome and return a plausible pressure.
    try:
        gas.blowdown_T(0.1 * rho0, rho0, TH.T_TANK, 0.0)
        check("a fully adiabatic blowdown is REFUSED as two-phase", False,
              "it returned a number — the dome was interpolated across")
    except ValueError as e:
        check("a fully adiabatic blowdown is REFUSED as two-phase",
              "two-phase" in str(e).lower() or "saturation" in str(e).lower())
    check("...but the default beta = 0.30 is comfortably superheated",
          gas.superheat(0.1 * rho0, t_low) > gas.SUPERHEAT_MIN,
          f"{gas.superheat(0.1 * rho0, t_low):.0f} K of margin")
    check("above the critical density there is NO saturation line",
          gas.T_sat(gas.RHO_CRIT + 50) == 0.0
          and gas.superheat(gas.RHO_CRIT + 50, 290.0) is None,
          "superheat is undefined, not 290 K")

    # The nozzle. Effective exhaust velocity is not a constant, and
    # the pressure term is why Mars and Earth differ on identical
    # hardware.
    h0 = gas.tank_h(rho0, TH.T_TANK)
    ce_e = gas.nozzle_ce(TH.P_REG, h0, "earth")
    ce_m = gas.nozzle_ce(TH.P_REG, h0, "mars")
    check("Isp is in the published cold-gas range (55-75 s)",
          55.0 < ce_e / TH.G0 < 75.0, f"{ce_e / TH.G0:.1f} s on earth")
    check("the SAME nozzle does better on Mars — (p_e - p_a)A_e",
          ce_m > ce_e * 1.05,
          f"{ce_m / TH.G0:.1f} s vs {ce_e / TH.G0:.1f} s")
    low = gas.tank_h(0.15 * rho0,
                     gas.blowdown_T(0.15 * rho0, rho0, TH.T_TANK, 0.30))
    check("...and c_e FALLS through the burn — it is not a constant",
          gas.nozzle_ce(TH.P_REG, low, "earth") < ce_e * 0.85,
          f"{gas.nozzle_ce(TH.P_REG, low, 'earth'):.0f} m/s late "
          f"vs {ce_e:.0f} at the start")
    check("the nozzle exit stays above nitrogen's triple point",
          gas.nozzle_Te(TH.P_REG, h0, "earth") > gas.T_TRIPLE,
          f"{gas.nozzle_Te(TH.P_REG, h0, 'earth'):.0f} K vs "
          f"{gas.T_TRIPLE:.0f} K")

    imp = TH.total_impulse()
    check("total impulse is integrated along the blowdown, not m*c_e",
          imp < m_real * ce_e * 0.9,
          f"{imp / 1000:.0f} kN·s vs the {m_real * ce_e / 1000:.0f} "
          f"a constant c_e would claim")
    hov = TH.hover_seconds()
    check("the whole tank buys SECONDS of hover, not minutes",
          3.0 < hov < 12.0, f"{hov:.1f} s")

    # Worlds must be coherent. The first version ran Earth gravity
    # with Martian air density, which is not any planet.
    print("\n  Worlds")
    check("gravity, air density and ambient pressure travel together",
          TH.MARS.g < TH.EARTH.g and TH.MARS.rho_air < TH.EARTH.rho_air
          and TH.MARS.p_amb < TH.EARTH.p_amb)
    check("Mars gravity is Mars gravity", abs(TH.MARS.g - 3.72) < 0.02,
          f"{TH.MARS.g:.3f} m/s²")
    m_e, _r, _s = CAR.build("reveal", flat=True, world=TH.EARTH)
    m_m, _r, _s = CAR.build("reveal", flat=True, world=TH.MARS)
    check("...and the MODEL gets them, not just the bookkeeping",
          abs(m_e.opt.gravity[2] + TH.EARTH.g) < 1e-4
          and abs(m_m.opt.gravity[2] + TH.MARS.g) < 1e-4,
          f"{m_e.opt.gravity[2]:.3f} vs {m_m.opt.gravity[2]:.3f}")
    check("the same hardware hovers longer on Mars",
          TH.hover_seconds(TH.MARS) > TH.hover_seconds(TH.EARTH) * 2.0,
          f"{TH.hover_seconds(TH.MARS):.1f} s vs "
          f"{TH.hover_seconds(TH.EARTH):.1f} s")

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
    check("every nozzle has a real THROAT AREA, not just a thrust",
          all(t.throat_area > 0 for t in r10)
          and all(abs(t.fmax - t.throat_area * TH.design_point()[0]
                      * TH.design_point()[1]) < 1.0 for t in r10),
          f"{min(t.throat_area for t in r10) * 1e6:.0f}–"
          f"{max(t.throat_area for t in r10) * 1e6:.0f} mm²")
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

    # RANK IS NOT REACHABILITY. The counterexample is kept here
    # permanently: `reveal` is rank 6 and can produce no backward
    # force at all, because every horizontal nozzle pushes the same
    # way and commands cannot go negative. Measured against the
    # classical controller, every rig that can brake lands >=98% and
    # every rig that cannot tops out at 83% -- a 12-nozzle rig
    # beating an 18-nozzle one. The lab shipped `rank` as its
    # headline hardware metric and it was the wrong invariant.
    print("\n  Rank is not reachability")
    r_rev = TH.rig("reveal")
    B_rev = TH.effectiveness(r_rev)
    check("reveal is FULL RANK",
          int(np.linalg.matrix_rank(B_rev, tol=1e-6)) == 6)
    check("...and can produce ZERO backward force anyway",
          TH.max_wrench(B_rev, 0, -1) < 1.0,
          f"{TH.max_wrench(B_rev, 0, -1):.1f} N of braking")
    check("...which `dead_axes` names and `rank` cannot",
          "-Fx" in TH.dead_axes(B_rev), str(TH.dead_axes(B_rev)))
    r_fore = TH.rig("fore10")
    check("adding ONE fore pair fixes it, at +2 nozzles",
          TH.max_wrench(TH.effectiveness(r_fore), 0, -1)
          > 0.9 * TH.wet_mass() * TH.EARTH.g,
          f"{TH.max_wrench(TH.effectiveness(r_fore), 0, -1):,.0f} N")
    check("...while an 18-nozzle rig without one still cannot brake",
          "-Fx" in TH.dead_axes(TH.effectiveness(TH.rig("dense8"))),
          "dense8, 18 nozzles, rank 6, no braking")
    check("reach is signed and spans all six axes",
          TH.reach(B_rev).shape == (6, 2))
    check("...and the polytope is one-sided where the hardware is",
          TH.reach(B_rev)[0, 0] == 0.0 and TH.reach(B_rev)[0, 1] > 1e3)
    check("waisted has NO pitch torque in either direction",
          set(TH.dead_axes(TH.effectiveness(TH.rig("waisted"))))
          >= {"-Ty", "+Ty"},
          str(TH.dead_axes(TH.effectiveness(TH.rig("waisted")))))

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
    # The bug that made the simulation wrong rather than merely
    # mis-reported: MuJoCo divides accumulated mass moments by
    # body_subtreemass to get subtree_com, and the whole
    # composite-rigid-body dynamics are expressed about that point.
    e_mass = AE.AerialEnv("reveal", flat=True)
    e_mass.reset(0)
    want = float(e_mass.model.body_mass.sum())
    check("body_subtreemass tracks body_mass as gas is spent",
          abs(float(e_mass.model.body_subtreemass[e_mass.chassis]) - want)
          < 1e-6,
          f"subtreemass {float(e_mass.model.body_subtreemass[e_mass.chassis]):.1f} "
          f"vs total {want:.1f}")
    for _ in range(40):
        e_mass.step(np.full(e_mass.act_dim, 0.8))
    want2 = float(e_mass.model.body_mass.sum())
    check("...and keeps tracking it after 40 steps of burning",
          abs(float(e_mass.model.body_subtreemass[e_mass.chassis]) - want2)
          < 1e-6 and want2 < want - 1.0,
          f"{want:.1f} → {want2:.1f} kg")

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
    check("the integrator is RK4 — ballistic flight, not contact-rich",
          int(env.model.opt.integrator) == 1,
          f"mjtIntegrator {int(env.model.opt.integrator)}")
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
