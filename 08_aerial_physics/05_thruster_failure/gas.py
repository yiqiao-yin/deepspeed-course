#!/usr/bin/env python3
"""
Real-gas propellant: tank, regulator, nozzle. Tables, no CoolProp at runtime.

    uv run gas.py                  # the blowdown, printed
    uv run gas.py --world mars
    uv run gas.py --nozzle         # where the area ratio stops paying

THE CHAIN, AND WHAT CONSERVES WHAT
-----------------------------------
    TANK        gas at (rho, T). Emptying it is ISENTROPIC, so the
                tank cools as it drains and a cold tank holds less
                pressure. The blowdown path is s = s0.

    REGULATOR   an ISENTHALPIC throttle down to the chamber
                pressure. Enthalpy is what survives a throttle, so
                the chamber state is a function of (p_c, h) and of
                nothing else -- which is why the nozzle tables are
                in those coordinates.

    NOZZLE      isentropic from the chamber. The throat is the
                maximum of the mass flux along the isentrope, which
                is the sonic condition found without ever needing a
                speed of sound or a ratio of specific heats. The
                exit is fixed by the AREA RATIO, not by ambient.

                F = mdot * v_e + (p_e - p_a) * A_e

THE BIT THAT CANNOT BE SHORTCUT
--------------------------------
"Expand to ambient" is the usual simplification and it is wrong
here, in a way that matters twice.

On Earth it hides the fact that a nozzle has a BEST area ratio:
expand past it and (p_e - p_a) goes negative and the extra bell
costs thrust. On Mars there is no such optimum, and the naive
version is not merely inaccurate -- expanding nitrogen isentropically
to 610 Pa takes it below its 63.15 K triple point, so the
calculation describes a nozzle full of snow.

WHAT THIS REPLACED
------------------
A constant exhaust velocity and a compressibility Z = 1.95 picked to
make an arithmetic come out. CoolProp says Z = 1.602 at 689 bar and
293 K; the tank holds 297 kg, not the 244 kg this lab first
published. Every number downstream moved with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TABLES = HERE / "n2_tables.npz"

R_N2 = 296.803                  # J/(kg K), specific gas constant
T_TRIPLE = 63.151               # K — below this the propellant is a solid
M_MOLAR = 0.0280134             # kg/mol

_T: dict | None = None


def reload(path=None) -> None:
    global _T
    p = Path(path or TABLES)
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} is missing. It is committed with the lab; "
            f"regenerate with `uv run --extra tables gen_tables.py`.")
    _T = dict(np.load(p))


def _tab() -> dict:
    if _T is None:
        reload()
    return _T                                               # type: ignore


def _bilinear(grid, xs, ys, x, y) -> float:
    """One lookup. `grid` is [len(ys), len(xs)]."""
    xi = float(np.clip(np.interp(x, xs, np.arange(len(xs))), 0,
                       len(xs) - 1 - 1e-9))
    yi = float(np.clip(np.interp(y, ys, np.arange(len(ys))), 0,
                       len(ys) - 1 - 1e-9))
    i, j = int(xi), int(yi)
    fx, fy = xi - i, yi - j
    return float((1 - fx) * (1 - fy) * grid[j, i]
                 + fx * (1 - fy) * grid[j, i + 1]
                 + (1 - fx) * fy * grid[j + 1, i]
                 + fx * fy * grid[j + 1, i + 1])


# --- tank ------------------------------------------------------------
def tank_p(rho: float, T: float) -> float:
    t = _tab()
    return _bilinear(t["p"], t["rho"], t["T_tank"], rho, T)


def tank_h(rho: float, T: float) -> float:
    t = _tab()
    return _bilinear(t["h"], t["rho"], t["T_tank"], rho, T)


def tank_s(rho: float, T: float) -> float:
    t = _tab()
    return _bilinear(t["s"], t["rho"], t["T_tank"], rho, T)


def compressibility(rho: float, T: float) -> float:
    """Z = p/(rho R T). 1.0 is ideal; this gas is not."""
    return tank_p(rho, T) / max(rho * R_N2 * T, 1e-9)


T_CRIT = 126.192                # K
RHO_CRIT = 313.3                # kg/m^3


def T_sat(rho: float) -> float:
    """
    Saturation temperature at this density, or 0 if there is none.

    Above the critical density (313 kg/m^3) nitrogen is
    SUPERCRITICAL and has no saturation line at all -- there is no
    phase boundary to cross, so "superheat" is not a defined
    quantity. The table stores 0 there, and 0 must be read as "no
    such line" rather than as "saturates at absolute zero". The HUD
    printed a superheat of 285.9 K on a full tank by doing the
    latter.
    """
    t = _tab()
    if rho >= RHO_CRIT:
        return 0.0
    return float(np.interp(rho, t["rho"], t["T_sat"]))


def superheat(rho: float, T: float) -> float | None:
    """Kelvin above saturation, or None where no phase line exists."""
    ts = T_sat(rho)
    return None if ts <= 0.0 else T - ts


# Superheat margin. Within a few kelvin of the saturation line the
# single-phase tables are unreliable no matter how fine the grid --
# p(rho, T) has a kink there and condensation is nucleating anyway.
# Measured: 2.9% pressure error 0.4 K above the dome, against 0.014%
# over the rest of the envelope. Refusing the margin is honest;
# refining the grid into it would only make a bad model precise.
SUPERHEAT_MIN = 5.0              # K


def two_phase(rho: float, T: float) -> bool:
    ts = T_sat(rho)
    return ts > 0.0 and T < ts + SUPERHEAT_MIN and rho > 1e-6


def blowdown_T(rho: float, rho0: float, T0: float, beta: float = 0.0) -> float:
    """
    Tank temperature at density `rho`, having started at (rho0, T0).

    The path is ISENTROPIC, solved by bisection on the tabulated
    entropy. `beta` blends back toward T0: 0 is adiabatic, 1 is a
    tank whose walls hold it at ambient. It is a heuristic and is
    labelled one — a real wall model needs a heat transfer
    coefficient and a thermal mass this lab does not have.
    """
    t = _tab()
    s0 = tank_s(rho0, T0)
    lo, hi = float(t["T_tank"][0]), float(t["T_tank"][-1])
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if tank_s(rho, mid) < s0:
            lo = mid
        else:
            hi = mid
    t_ad = 0.5 * (lo + hi)
    out = t_ad + beta * (T0 - t_ad)
    if two_phase(rho, out):
        # FAIL LOUDLY. A perfectly insulated cold-gas tank really
        # does condense its own propellant as it empties -- at
        # beta = 0 this one goes two-phase below ~60 kg remaining --
        # and these tables describe a single-phase fluid only.
        # Interpolating across the saturation dome would return a
        # plausible pressure and quietly understate the thrust the
        # policy is being trained against.
        raise ValueError(
            f"tank is at or near SATURATION at rho = {rho:.1f} kg/m^3, "
            f"T = {out:.1f} K (saturation {T_sat(rho):.1f} K, margin "
            f"required {SUPERHEAT_MIN:.0f} K). Nitrogen is condensing in "
            f"the tank. The single-phase tables do not describe this. "
            f"beta = {beta:.2f} is too low: the walls must return at least "
            f"~20% of the adiabatic drop to keep the propellant a gas "
            f"with {SUPERHEAT_MIN:.0f} K of margin.")
    return out


# --- chamber and nozzle ----------------------------------------------
def nozzle_G(pc: float, h_in: float, world: str = "earth") -> float:
    """Throat mass flux, kg/(m^2 s). Multiply by throat area for mdot."""
    t = _tab()
    return _bilinear(t[f"G_{world}"], t["pc"], t["h_in"], pc, h_in)


def nozzle_ce(pc: float, h_in: float, world: str = "earth") -> float:
    """
    EFFECTIVE exhaust velocity, F/mdot, including the pressure term.

    This is the number that multiplies mass flow to give thrust, and
    it is NOT the exit velocity: it carries (p_e - p_a) A_e, which is
    positive on Mars and negative on Earth for an over-expanded bell.
    """
    t = _tab()
    return max(_bilinear(t[f"ce_{world}"], t["pc"], t["h_in"], pc, h_in), 1.0)


def nozzle_Te(pc: float, h_in: float, world: str = "earth") -> float:
    t = _tab()
    return _bilinear(t[f"Te_{world}"], t["pc"], t["h_in"], pc, h_in)


def nozzle_pe(pc: float, h_in: float, world: str = "earth") -> float:
    t = _tab()
    return _bilinear(t[f"pe_{world}"], t["pc"], t["h_in"], pc, h_in)


def area_ratio() -> float:
    return float(_tab()["eps"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--volume", type=float, default=0.600)
    ap.add_argument("--p0", type=float, default=689.5e5)
    ap.add_argument("--T0", type=float, default=293.15)
    ap.add_argument("--pc", type=float, default=40.0e5)
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--nozzle", action="store_true")
    a, _ = ap.parse_known_args()

    reload()
    t = _tab()

    # Invert p(rho, T0) for the fill density.
    lo, hi = 2.0, 529.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tank_p(mid, a.T0) < a.p0:
            lo = mid
        else:
            hi = mid
    rho0 = 0.5 * (lo + hi)
    m0 = rho0 * a.volume

    print("=" * 78)
    print(f"  COLD-GAS PROPELLANT — nitrogen, area ratio {area_ratio():.1f}, "
          f"{a.world}")
    print("=" * 78)
    print(f"  {a.volume * 1000:.0f} L at {a.p0 / 1e5:.0f} bar, {a.T0:.1f} K")
    print(f"    real gas   {rho0:7.1f} kg/m3 -> {m0:6.1f} kg   "
          f"(Z = {compressibility(rho0, a.T0):.3f})")
    ideal = a.p0 / (R_N2 * a.T0)
    print(f"    ideal gas  {ideal:7.1f} kg/m3 -> {ideal * a.volume:6.1f} kg   "
          f"({ideal / rho0 - 1:+.0%}, and denser than LIQUID nitrogen)")

    if a.nozzle:
        print(f"\n  The area ratio is a CHOICE, and it is world-specific.")
        print(f"  (chamber 40 bar, 260 K inlet enthalpy)")
        print(f"\n  {'eps':>5} {'p_e':>10} {'T_e':>8} "
              f"{'c_e earth':>11} {'c_e mars':>10}")
        print("  Re-run `gen_tables.py --eps N` to retabulate; the shipped")
        print(f"  tables are eps = {area_ratio():.1f}, chosen as Earth's "
              f"optimum.")
        return 0

    h0 = tank_h(rho0, a.T0)
    print(f"\n  BLOWDOWN (isentropic, wall blend beta = {a.beta})")
    print(f"  {'gas kg':>7} {'bar':>7} {'T tank':>8} {'p_c bar':>9} "
          f"{'c_e':>7} {'Isp':>6} {'T exit':>8} {'p_e kPa':>9} {'G_t':>9}")
    for frac in (1.0, 0.85, 0.7, 0.55, 0.4, 0.28, 0.18, 0.1, 0.05):
        m = m0 * frac
        rho = m / a.volume
        T = blowdown_T(rho, rho0, a.T0, a.beta)
        p = tank_p(rho, T)
        pc = min(a.pc, p / 1.15)
        h = tank_h(rho, T)
        ce = nozzle_ce(pc, h, a.world)
        te = nozzle_Te(pc, h, a.world)
        g = nozzle_G(pc, h, a.world)
        flag = "  <- FREEZES" if te < T_TRIPLE else ""
        print(f"  {m:>7.1f} {p / 1e5:>7.0f} {T:>8.1f} {pc / 1e5:>9.1f} "
              f"{ce:>7.0f} {ce / 9.80665:>6.1f} {te:>8.1f} "
              f"{nozzle_pe(pc, h, a.world) / 1e3:>9.1f} {g:>9.0f}{flag}")

    print(f"\n  c_e is EFFECTIVE exhaust velocity F/mdot, including the")
    print(f"  (p_e - p_a)A_e term. Nitrogen's triple point is "
          f"{T_TRIPLE:.1f} K — an exit")
    print(f"  colder than that is a nozzle full of snow, not a thruster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
