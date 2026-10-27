#!/usr/bin/env python3
"""
Generate the real-gas property tables. Authoring tool — needs CoolProp.

    uv run --extra tables gen_tables.py
    uv run --extra tables gen_tables.py --verify      # error vs CoolProp

WHY TABLES AND NOT A CORRELATION
---------------------------------
At 689 bar nitrogen is nowhere near ideal, and the simple corrections
are not good enough to publish a headline number from:

    model                        Z at 689 bar, 293 K    mass in 600 L
    ideal gas                            1.000              475 kg
    van der Waals                        1.463              325 kg
    a hand-picked Z = 1.95               1.950              244 kg
    CoolProp (Span-Wagner EOS)           1.602              297 kg

The third line is what this lab shipped first. It was a guess, it was
22% wrong, and every number downstream of it was wrong with it. The
fix is not a better guess — it is a reference equation of state.

CoolProp is a build-time dependency only. The tables it writes are
committed, so a learner runs the lab without it, which matters
because the whole course installs from a committed lock.

WHAT IS TABULATED, AND IN WHAT COORDINATES
-------------------------------------------
Two 2-D grids, each chosen so the physics it feeds is a pure lookup:

  TANK, over (density, temperature):
      p     tank pressure
      h     specific enthalpy — the quantity the regulator conserves
      s     specific entropy  — the quantity blowdown conserves

  CHAMBER AND NOZZLE, over (chamber pressure, inlet enthalpy):
      G_t   throat mass flux, kg/(m^2 s)
      c_e   EFFECTIVE exhaust velocity, per world
      T_e   exit temperature — watched, because it approaches the
            temperature at which the propellant freezes
      p_e   exit pressure

(p_c, h) is the right parametrisation for the second grid because a
regulator is an isenthalpic throttle: enthalpy is what survives it,
so the chamber state is a function of the pressure it holds and the
enthalpy that arrived. Tabulating against tank temperature instead
would need a third axis the moment the regulator drops out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FLUID = "Nitrogen"

# Grids. Ranges are set by where the vehicle actually operates: the
# tank runs 520 kg/m^3 down to empty and cools to ~90 K doing it.
# 260 points, not 120. Pressure is strongly nonlinear in density
# at the high-density end, and bilinear interpolation there was
# 2.3% off -- fine for a plot, not for the number the whole lab
# is sized from.
RHO = np.linspace(2.0, 530.0, 260)          # kg/m^3
T_TANK = np.linspace(75.0, 320.0, 100)      # K
# Chamber pressure and inlet enthalpy, ranged over what the vehicle
# ACTUALLY visits. Measured along the blowdown: 29-302 kJ/kg (the low
# end is a nearly empty adiabatic tank at 82 K, the high end an
# isothermal one). The first version guessed -120 to 340 kJ/kg and
# 43% of the grid fell outside the fluid's valid range, so nearly
# half the table was fill rather than physics.
PC = np.geomspace(0.8e5, 46.0e5, 88)        # Pa, chamber
H_IN = np.linspace(0.20e5, 3.10e5, 92)      # J/kg, enthalpy into the chamber

# Ambient pressures. The nozzle is NOT expanded to ambient -- it is
# expanded to whatever its area ratio gives, and the pressure term
# (p_e - p_a) A_e is then added. That is the correct thrust equation,
# and it is also the only one that survives Mars: expanding N2
# isentropically to 610 Pa takes it below its triple point (63.15 K),
# so "expand to ambient" is not merely inaccurate there, it is
# thermodynamically impossible.
WORLDS = {"earth": 101325.0, "mars": 610.0}
AREA_RATIO = 5.0


def nozzle(pc: float, h_in: float, eps: float, p_amb: float, P):
    """
    Real-gas converging-diverging nozzle, from first principles.

    Throat is found as the maximum of the mass flux along the
    isentrope -- which IS the sonic condition, and finding it that
    way needs no speed of sound and no gamma. The exit is the
    supersonic root of G(p_e) = G_t / eps.
    """
    try:
        Tc = P("T", "P", pc, "H", h_in, FLUID)
        sc = P("S", "P", pc, "T", Tc, FLUID)
        hc = P("H", "P", pc, "T", Tc, FLUID)
    except Exception:                                       # noqa: BLE001
        return None

    ps = np.linspace(0.35 * pc, 0.92 * pc, 120)
    best = (0.0, pc)
    for p in ps:
        try:
            h = P("H", "P", p, "S", sc, FLUID)
            if h >= hc:
                continue
            g = P("D", "P", p, "S", sc, FLUID) * math_sqrt(2.0 * (hc - h))
        except Exception:                                   # noqa: BLE001
            continue
        if g > best[0]:
            best = (g, p)
    g_t, p_t = best
    if g_t <= 0.0:
        return None

    # SEPARATION. A fixed-area-ratio bell at low chamber pressure is
    # grossly over-expanded, and the ideal calculation then returns a
    # NEGATIVE effective exhaust velocity -- a thruster that pulls
    # backwards when you open its valve. Real flow does not do that:
    # it detaches from the wall and the exit plane moves inboard to
    # roughly p_e = 0.4 p_a (Summerfield). That floor is enforced
    # here, so a draining tank loses thrust instead of reversing it.
    p_sep = 0.40 * p_amb
    lo, hi = max(300.0, p_sep), p_t          # above the triple point
    if lo >= hi:
        lo = min(300.0, 0.5 * p_t)
    pe = p_t
    for _ in range(60):
        pe = 0.5 * (lo + hi)
        try:
            h = P("H", "P", pe, "S", sc, FLUID)
            ve = math_sqrt(max(2.0 * (hc - h), 0.0))
            ge = P("D", "P", pe, "S", sc, FLUID) * ve
        except Exception:                                   # noqa: BLE001
            lo = pe
            continue
        if ge > g_t / eps:
            hi = pe
        else:
            lo = pe
    try:
        h = P("H", "P", pe, "S", sc, FLUID)
        ve = math_sqrt(max(2.0 * (hc - h), 0.0))
        rho_e = P("D", "P", pe, "S", sc, FLUID)
        Te = P("T", "P", pe, "S", sc, FLUID)
    except Exception:                                       # noqa: BLE001
        return None
    if ve <= 1.0 or rho_e <= 0.0:
        return None
    # F = mdot*ve + (pe - pa)*Ae, and Ae = mdot/(rho_e ve), so the
    # EFFECTIVE exhaust velocity is F/mdot:
    c_e = ve + (pe - p_amb) / (rho_e * ve)
    return g_t, c_e, Te, pe


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "n2_tables.npz"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--eps", type=float, default=AREA_RATIO)
    a, _ = ap.parse_known_args()

    from CoolProp.CoolProp import PropsSI as P

    print("=" * 76)
    print(f"  REAL-GAS TABLES — {FLUID}, area ratio {a.eps}")
    print("=" * 76)

    # --- tank grid --------------------------------------------------
    nP = np.zeros((len(T_TANK), len(RHO)))
    nH = np.zeros_like(nP)
    nS = np.zeros_like(nP)
    for j, T in enumerate(T_TANK):
        for i, r in enumerate(RHO):
            nP[j, i] = P("P", "D", r, "T", T, FLUID)
            nH[j, i] = P("H", "D", r, "T", T, FLUID)
            nS[j, i] = P("S", "D", r, "T", T, FLUID)
    # Saturation temperature against density. The single-phase
    # tables are only valid ABOVE this line, and a fully adiabatic
    # blowdown crosses it: at beta = 0 the tank is two-phase below
    # about 60 kg remaining. That is real -- an ideally insulated
    # cold-gas tank condenses its own propellant -- and the model
    # has to say so rather than interpolate across the dome.
    TSAT = np.zeros(len(RHO))
    for i, r in enumerate(RHO):
        try:
            TSAT[i] = P("T", "D", r, "Q", 1, FLUID)
        except Exception:                                   # noqa: BLE001
            TSAT[i] = 0.0
    out_tsat = TSAT

    print(f"  tank grid     {nP.shape}  "
          f"rho {RHO[0]:.0f}-{RHO[-1]:.0f} kg/m3, "
          f"T {T_TANK[0]:.0f}-{T_TANK[-1]:.0f} K")

    # --- nozzle grid, per world -------------------------------------
    out = {"rho": RHO, "T_tank": T_TANK, "p": nP, "h": nH, "s": nS,
           "T_sat": out_tsat,
           "pc": PC, "h_in": H_IN, "eps": a.eps}
    for world, pa in WORLDS.items():
        G = np.zeros((len(H_IN), len(PC)))
        C = np.zeros_like(G)
        TE = np.zeros_like(G)
        PE = np.zeros_like(G)
        bad = 0
        for j, h in enumerate(H_IN):
            for i, pc in enumerate(PC):
                r = nozzle(pc, h, a.eps, pa, P)
                if r is None:
                    bad += 1
                    continue
                G[j, i], C[j, i], TE[j, i], PE[j, i] = r
        # Fill failures by nearest valid along the pressure axis so a
        # lookup off the edge of physics degrades rather than
        # returning zero thrust and looking like an empty tank.
        for arr in (G, C, TE, PE):
            for j in range(arr.shape[0]):
                row = arr[j]
                ok = row > 0
                if ok.any():
                    arr[j] = np.interp(np.arange(len(row)),
                                       np.flatnonzero(ok), row[ok])
        out[f"G_{world}"] = G
        out[f"ce_{world}"] = C
        out[f"Te_{world}"] = TE
        out[f"pe_{world}"] = PE
        print(f"  {world:<6} grid  {G.shape}  "
              f"c_e {C[C > 0].min():.0f}-{C.max():.0f} m/s, "
              f"{bad} points outside the fluid's range")

    np.savez_compressed(a.out, **out)
    sz = Path(a.out).stat().st_size / 1024
    print(f"\n  wrote {Path(a.out).name}  ({sz:.0f} KB)")

    # --- the headline, recomputed -----------------------------------
    rho0 = P("D", "P", 689.5e5, "T", 293.15, FLUID)
    z0 = 689.5e5 / (rho0 * 296.8 * 293.15)
    print(f"\n  600 L at 689.5 bar, 293.15 K:")
    print(f"    real   {rho0:.1f} kg/m3 -> {rho0 * 0.6:.1f} kg   (Z = {z0:.3f})")
    print(f"    ideal  {689.5e5 / (296.8 * 293.15):.1f} kg/m3 -> "
          f"{689.5e5 / (296.8 * 293.15) * 0.6:.1f} kg")

    if a.verify:
        import gas

        gas.reload(a.out)
        rng = np.random.default_rng(0)
        ep, eh, ec = [], [], []
        for _ in range(400):
            r = rng.uniform(RHO[0] + 1, RHO[-1] - 1)
            T = rng.uniform(T_TANK[0] + 1, T_TANK[-1] - 1)
            ep.append(abs(gas.tank_p(r, T) - P("P", "D", r, "T", T, FLUID))
                      / P("P", "D", r, "T", T, FLUID))
            eh.append(abs(gas.tank_h(r, T) - P("H", "D", r, "T", T, FLUID)))
        for _ in range(200):
            pc = float(np.exp(rng.uniform(np.log(1.0e5), np.log(44e5))))
            h = rng.uniform(0.3e5, 3.0e5)
            ref = nozzle(pc, h, a.eps, WORLDS["earth"], P)
            if ref is None:
                continue
            # Relative to |ref|. The first version divided by a
            # SIGNED reference that could be negative (an
            # over-expanded nozzle), so the "mean absolute error"
            # printed as -9.3% -- a number no absolute error can
            # take, and the tell that the metric was wrong.
            ec.append(abs(gas.nozzle_ce(pc, h, "earth") - ref[1])
                      / max(abs(ref[1]), 1.0))
        print(f"\n  INTERPOLATION ERROR vs CoolProp, off-grid:")
        print(f"    tank pressure      {np.mean(ep):.3%} mean, "
              f"{np.max(ep):.3%} worst")
        print(f"    tank enthalpy      {np.mean(eh):.0f} J/kg mean")
        print(f"    exhaust velocity   {np.mean(ec):.3%} mean, "
              f"{np.max(ec):.3%} worst")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
