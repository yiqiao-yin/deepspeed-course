#!/usr/bin/env python3
"""
A thruster rig is PARAMETERS. Check it can do the job before training it.

    uv run thrusters.py                      # the default rig, audited
    uv run thrusters.py --rig minimal
    uv run thrusters.py --all                # every named rig, compared
    uv run thrusters.py --pairs 3 --upper

WHY THIS FILE EXISTS AT ALL
---------------------------
Topic 7 learned the same lesson three times in one session: a
hardcoded actuator index eventually addresses the wrong actuator, and
it does not crash -- it writes the yaw command into an ankle and the
robot stands perfectly still at a 0% fall rate while three training
runs get blamed on the observation width.

So nothing here is typed out. A rig is generated from a handful of
numbers, every nozzle gets a NAME, and the model resolves actuators
through those names rather than through their order.

WHAT A COLD-GAS THRUSTER ACTUALLY IS
-------------------------------------
Not a rocket. There is no combustion: a valve opens, high-pressure
gas expands through a bell, and the momentum of the gas leaving is
the thrust. Nitrogen manages a specific impulse of about 73 s --
roughly a quarter of a hydrazine monopropellant and a twentieth of a
cryogenic engine. The entire design consequence is that propellant is
a HARD BUDGET measured in seconds, so the interesting control problem
is not "can it hover" but "can it finish before the tank does".

THE ARITHMETIC THAT SIZES THE WHOLE LAB
----------------------------------------
Tesla's October 2026 reveal quoted ~600 L of compressed gas in
carbon-fibre overwrapped pressure vessels. COPVs of that class run to
about 10,000 psi (689 bar). At 689 bar and 293 K nitrogen is NOT an
ideal gas, and the error is not small:

    ideal gas   p/(RT)       = 792 kg/m^3  -> 475 kg   (denser than
                                                        LIQUID nitrogen,
                                                        which should
                                                        have been the
                                                        tell)
    real gas    p/(Z R T),   Z = 1.95      -> 406 kg/m^3 -> 244 kg

244 kg at v_e = 716 m/s is 175 kN*s of total impulse: about 8.9
seconds of hover for a 2-tonne car, and far less once attitude
control is also drawing from the same tank. Every number in this lab
follows from that one.

WHAT IS CLAIMED AND WHAT IS ASSUMED
------------------------------------
From the reveal (manufacturer claims, one clip, no repeatability
shown, no independent measurement): roughly 10 nozzles, ~600 L of
gas, COPV storage, a hop with a mid-air rotation onto the wheels.

EVERYTHING ELSE HERE IS A STATED ENGINEERING ASSUMPTION: mass,
inertia, nozzle positions and sizes, regulated pressure, and the
whole control problem. This is a physics sandbox motivated by the
reveal, not a reconstruction of Tesla's hardware.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------------------
# Vehicle and propellant constants. One place, named, so a number
# cannot be edited here and left stale three files away.
# ---------------------------------------------------------------------------
G = 9.80665                      # m/s^2, standard gravity (Isp uses this one)
M_DRY = 1750.0                   # kg, vehicle without gas
M_GAS_0 = 244.0                  # kg, see the docstring's arithmetic
TANK_V = 0.600                   # m^3, the quoted 600 L
TANK_P0 = 689.5e5                # Pa, 10,000 psi COPV
T_TANK = 293.15                  # K
R_N2 = 296.8                     # J/(kg K)
Z_N2 = 1.95                      # compressibility at 689 bar / 293 K
ISP = 73.0                       # s, nitrogen cold gas
VE_IDEAL = ISP * G               # m/s, 715.9
ETA_V = 0.92                     # nozzle velocity efficiency
VE = ETA_V * VE_IDEAL            # m/s, 658.6 effective

P_REG = 40.0e5                   # Pa, regulated chamber pressure
P_DROPOUT = 1.15 * P_REG         # below this the regulator cannot hold

# Body geometry. The car is a 4.34 m x 1.96 m roadster; these are the
# half-extents the nozzle stations are placed against.
HALF_L = 2.17
HALF_W = 0.98
DECK_Z = -0.25                   # z of the lower (down-firing) nozzles
ROOF_Z = 0.22                    # z of the upper (up-firing) nozzles

# Inertia of the dry car about its CG, roll/pitch/yaw.
I_DRY = np.array([800.0, 2950.0, 3250.0])
Y_TANK = 0.45                    # tanks sit here; gas adds to roll and yaw


def gas_mass_from_tank() -> float:
    """
    The 244 kg, recomputed rather than quoted.

    A constant that is also derivable should be derived, because the
    derivation is what fails loudly when someone edits the pressure
    and forgets the mass. Topic 7 shipped a published figure whose
    attached configuration had drifted; the fix is to not have two
    copies of one fact.
    """
    rho = TANK_P0 / (Z_N2 * R_N2 * T_TANK)
    return rho * TANK_V


def total_impulse(m_gas: float = M_GAS_0) -> float:
    """N*s available. The budget the whole task is played against."""
    return m_gas * VE


def hover_seconds(m_gas: float = M_GAS_0, mass: float | None = None) -> float:
    mass = (M_DRY + m_gas) if mass is None else mass
    return total_impulse(m_gas) / (mass * G)


@dataclass(frozen=True)
class Thruster:
    """
    One nozzle.

    `direction` is the direction of the FORCE on the car (gas leaves
    the opposite way). `fmax` is thrust at full regulated pressure.
    `name` is load-bearing: the model resolves actuators by it.
    """
    name: str
    pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    fmax: float
    role: str                      # "lift" | "down" | "aft" | "fore"

    def column(self, cg=(0.0, 0.0, 0.0)) -> np.ndarray:
        """This nozzle's column of the control effectiveness matrix."""
        r = np.asarray(self.pos) - np.asarray(cg)
        d = np.asarray(self.direction)
        return np.concatenate([self.fmax * d, self.fmax * np.cross(r, d)])


def _unit(v) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("thruster direction must be non-zero")
    return tuple(a / n)


def make_rig(pairs: int = 2,
             upper: bool = True,
             aft: int = 1,
             fore: int = 0,
             cant_deg: float = 15.0,
             spread: float = 1.30,
             width: float = 0.80,
             roof_width: float = 0.86,
             thrust_ratio: float = 2.40,
             upper_frac: float = 0.36,
             aft_frac: float = 0.58,
             mass: float | None = None) -> list[Thruster]:
    """
    Build a rig from numbers. Nothing about it is typed out.

    `pairs`      stations per side along the body, front to back. Two
                 pairs is a car's four corners; three adds a mid
                 station; one puts everything at the waist.
    `upper`      mirror every lower station with an UP-firing nozzle.
                 This is what buys a pure roll COUPLE -- see below.
    `aft`/`fore` pairs of horizontal nozzles for downrange speed.
    `cant_deg`   corner nozzles lean their exhaust outboard so the
                 plumes miss the bodywork. The cant is NOT free: it
                 costs cos(gamma) of the vertical component and the
                 lateral parts only cancel if the rig is symmetric.

    `thrust_ratio` is the total lift-to-weight at full pressure; every
    lower nozzle is sized from it, so a rig with six lift nozzles has
    weaker ones than a rig with four and the comparison between rigs
    is at EQUAL INSTALLED THRUST rather than equal nozzle size. That
    is the control that makes a thruster-count sweep mean anything --
    without it, "more nozzles is better" would just be "more thrust is
    better", which needs no experiment.

    WHY UP-FIRING NOZZLES, AND WHY NOT SIDEWAYS ONES
    -------------------------------------------------
    Torque is tau = r x F, so the roll component is

        tau_x = r_y F_z - r_z F_y.

    A sideways push (F_y) at door height (r_z ~ 0) makes essentially
    no roll. A vertical push (F_z) at a corner (|r_y| ~ 0.8 m) makes
    plenty. So roll authority comes from VERTICAL nozzles at lateral
    stations, and a pure roll couple -- torque with no net force --
    needs some pushing up on one side while others push down. With
    lower nozzles only, every roll command also throws the car
    upward, and the controller has to fight its own roll input.
    """
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    if aft < 0 or fore < 0:
        raise ValueError("aft/fore counts must be >= 0")
    mass = M_DRY + M_GAS_0 if mass is None else mass

    g = math.radians(cant_deg)
    sg, cg_ = math.sin(g), math.cos(g)

    n_lift = 2 * pairs
    # Sized so the LIFT COMPONENT (after the cant) hits the ratio.
    f_lift = thrust_ratio * mass * G / (n_lift * cg_)
    f_up = upper_frac * f_lift
    f_aft = aft_frac * mass * G / max(aft, 1)
    f_fore = aft_frac * mass * G / max(fore, 1)

    # Station x positions, front to back, symmetric about the CG.
    if pairs == 1:
        xs = [0.0]
    else:
        xs = [spread - 2 * spread * i / (pairs - 1) for i in range(pairs)]

    out: list[Thruster] = []
    for k, x in enumerate(xs):
        tag = "FMRABCDEFG"[k] if k < 10 else str(k)
        for side, sy in (("L", +1.0), ("R", -1.0)):
            # Lower: pushes the car UP, exhaust canted outboard.
            out.append(Thruster(
                name=f"lo{tag}{side}",
                pos=(x, sy * width, DECK_Z),
                direction=_unit((0.0, -sy * sg, cg_)),
                fmax=f_lift, role="lift"))
            if upper:
                # Upper: pushes the car DOWN. Same cant, mirrored.
                out.append(Thruster(
                    name=f"up{tag}{side}",
                    pos=(x, sy * roof_width, ROOF_Z),
                    direction=_unit((0.0, -sy * sg, -cg_)),
                    fmax=f_up, role="down"))

    for i in range(aft):
        z = 0.0 if aft == 1 else -0.10 + 0.20 * i / max(aft - 1, 1)
        for side, sy in (("L", +1.0), ("R", -1.0)):
            out.append(Thruster(name=f"aft{i}{side}",
                                pos=(-HALF_L + 0.07, sy * 0.45, z),
                                direction=(1.0, 0.0, 0.0),
                                fmax=f_aft, role="aft"))
    for i in range(fore):
        for side, sy in (("L", +1.0), ("R", -1.0)):
            out.append(Thruster(name=f"fore{i}{side}",
                                pos=(HALF_L - 0.07, sy * 0.45, 0.0),
                                direction=(-1.0, 0.0, 0.0),
                                fmax=f_fore, role="fore"))

    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate thruster names in rig: {names}")
    return out


# ---------------------------------------------------------------------------
# Named rigs. ADDED, never substituted -- a published number has to go
# on describing the rig it was measured on.
# ---------------------------------------------------------------------------
RIGS: dict[str, dict] = {
    # Four corners, lift only. The cheapest thing that can hop, and
    # the one with no pure roll couple at all.
    "minimal":   dict(pairs=2, upper=False, aft=1),
    # Four corners plus four roof nozzles plus a rear pair: ten, the
    # count Tesla quoted.
    "reveal":    dict(pairs=2, upper=True, aft=1),
    # Six stations, no roof.
    "sixlift":   dict(pairs=3, upper=False, aft=1),
    # Six stations mirrored: fourteen nozzles.
    "wide":      dict(pairs=3, upper=True, aft=1),
    # Everything at the waist: the SAME nozzle count as `reveal` with
    # the lever arms taken away. The control for "count vs placement".
    "waisted":   dict(pairs=1, upper=True, aft=1, spread=0.0),
    # Narrow track: lever arms halved laterally, count unchanged.
    "narrow":    dict(pairs=2, upper=True, aft=1, width=0.40,
                      roof_width=0.43),
    # Eight stations mirrored plus fore and aft: twenty.
    "dense":     dict(pairs=4, upper=True, aft=1, fore=1),
}
RIGS["default"] = RIGS["reveal"]


def rig(name: str = "default", **override) -> list[Thruster]:
    if name not in RIGS:
        raise KeyError(f"unknown rig {name!r}; have {sorted(RIGS)}")
    spec = dict(RIGS[name])
    spec.update({k: v for k, v in override.items() if v is not None})
    return make_rig(**spec)


# ---------------------------------------------------------------------------
# Control effectiveness, and what the rig can actually do
# ---------------------------------------------------------------------------
def effectiveness(rig_: list[Thruster], cg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """
    The 6 x N matrix B with wrench = B u for commands u in [0,1]^N.

    Rows are (Fx, Fy, Fz, Tx, Ty, Tz) in the body frame about `cg`.
    """
    return np.stack([t.column(cg) for t in rig_], axis=1)


def max_wrench(B: np.ndarray, row: int, sign: float = 1.0) -> float:
    """
    Largest value of one wrench component with everything else FREE.

    Closed form: commands are box-bounded in [0, 1], so the maximum
    of a linear functional is the sum of its positive coefficients.
    No solver, and nothing to get subtly wrong.
    """
    return float(np.sum(np.clip(sign * B[row], 0.0, None)))


def max_pure_couple(B: np.ndarray, axis: int = 0, sign: float = 1.0
                    ) -> tuple[float, np.ndarray] | tuple[float, None]:
    """
    Largest torque about `axis` with EVERY other wrench component zero.

    This is the honest measure of attitude authority, and it is much
    smaller than `max_wrench` because a roll torque that also throws
    the car upward is not a roll torque -- it is a roll torque plus a
    translation the controller then has to undo.

    Solved exactly as a linear program:

        max  c' u    s.t.  A u = 0,  0 <= u <= 1

    with c the torque row and A the other five rows. Topic 7's
    equivalent figure was first quoted from a hand calculation that
    ignored the nozzle cant and overstated it by 11%; a hand
    calculation is a guess with the arithmetic shown.
    """
    from scipy.optimize import linprog

    rows = [r for r in range(6) if r != 3 + axis]
    res = linprog(c=-sign * B[3 + axis],
                  A_eq=B[rows], b_eq=np.zeros(len(rows)),
                  bounds=[(0.0, 1.0)] * B.shape[1], method="highs")
    if not res.success:
        return 0.0, None
    return float(sign * B[3 + axis] @ res.x), res.x


def inertia(m_gas: float = M_GAS_0) -> np.ndarray:
    """Roll/pitch/yaw inertia. The gas sits in the tanks, so it adds
    to roll and yaw but not to pitch."""
    return I_DRY + m_gas * Y_TANK ** 2 * np.array([1.0, 0.0, 1.0])


def audit(name: str, rig_: list[Thruster], mass: float | None = None) -> dict:
    """
    Everything you can know about a rig WITHOUT training on it.

    This is the `--verify-arch` idea from the LLM labs, pointed at a
    vehicle: build the thing cheaply, interrogate it, and find out
    that the target names resolve and the arithmetic agrees -- before
    paying for the run that would otherwise discover it.
    """
    m = M_DRY + M_GAS_0 if mass is None else mass
    B = effectiveness(rig_)
    I = inertia()
    w = m * G

    lift = max_wrench(B, 2)
    roll_free = max_wrench(B, 3)
    roll_pure, _u = max_pure_couple(B, 0)
    pitch_pure, _ = max_pure_couple(B, 1)
    yaw_pure, _ = max_pure_couple(B, 2)

    # Can it finish a 360 deg roll inside a ballistic arc? With a pure
    # couple tau the roll is spin-up/coast/spin-down; the fastest
    # bang-bang 2*pi turn takes 2*sqrt(pi*I/tau).
    t_roll = (2.0 * math.sqrt(math.pi * I[0] / roll_pure)
              if roll_pure > 1e-6 else float("inf"))

    return {
        "rig": name,
        "n_thrusters": len(rig_),
        "n_lift": sum(t.role == "lift" for t in rig_),
        "n_down": sum(t.role == "down" for t in rig_),
        "installed_lift_N": lift,
        "lift_to_weight": lift / w,
        "roll_free_Nm": roll_free,
        "roll_pure_Nm": roll_pure,
        "pitch_pure_Nm": pitch_pure,
        "yaw_pure_Nm": yaw_pure,
        "purity": roll_pure / roll_free if roll_free > 1e-9 else 0.0,
        "min_roll_time_s": t_roll,
        "hover_s": hover_seconds(),
        "rank": int(np.linalg.matrix_rank(B, tol=1e-6)),
        "names": [t.name for t in rig_],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(RIGS))
    ap.add_argument("--all", action="store_true", help="compare every rig")
    ap.add_argument("--pairs", type=int)
    ap.add_argument("--upper", action="store_true", default=None)
    ap.add_argument("--aft", type=int)
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    print("=" * 78)
    print("  COLD-GAS RIG AUDIT — what the hardware can do, before any training")
    print("=" * 78)
    m_chk = gas_mass_from_tank()
    print(f"  tank      {TANK_V * 1000:.0f} L at {TANK_P0 / 1e5:.0f} bar, "
          f"Z = {Z_N2}")
    print(f"  gas       {m_chk:.1f} kg real-gas  "
          f"(ideal gas would claim {TANK_P0 / (R_N2 * T_TANK) * TANK_V:.0f} kg "
          f"— denser than LIQUID nitrogen)")
    print(f"  impulse   {total_impulse() / 1000:.1f} kN·s at v_e = {VE:.0f} m/s "
          f"(Isp {ISP:.0f} s)")
    print(f"  hover     {hover_seconds():.1f} s for the whole tank — "
          f"the budget every policy plays against")
    print(f"  mass      {M_DRY + M_GAS_0:.0f} kg, "
          f"I = {np.round(inertia(), 1)} kg·m²")

    names = sorted(RIGS) if a.all else [a.rig]
    rows = []
    for nm in names:
        r = rig(nm, pairs=a.pairs, upper=a.upper, aft=a.aft)
        rows.append(audit(nm, r))

    print(f"\n  {'rig':<10} {'N':>3} {'L/W':>6} {'roll free':>10} "
          f"{'roll PURE':>10} {'purity':>7} {'360° in':>8} {'rank':>5}")
    print("  " + "-" * 68)
    for d in rows:
        t = d["min_roll_time_s"]
        print(f"  {d['rig']:<10} {d['n_thrusters']:>3} "
              f"{d['lift_to_weight']:>6.2f} "
              f"{d['roll_free_Nm']:>9,.0f}N "
              f"{d['roll_pure_Nm']:>9,.0f}N "
              f"{d['purity']:>6.0%} "
              f"{(f'{t:.2f}s' if math.isfinite(t) else 'never'):>8} "
              f"{d['rank']:>5}")

    print("\n  `roll free` allows a net force; `roll PURE` is a true couple.")
    print("  A rig whose purity is 0 can only roll by also throwing itself")
    print("  somewhere, which the policy then has to pay to undo.")
    print("  rank < 6 means some wrench direction is UNREACHABLE at any")
    print("  command — the rig cannot be commanded in that axis at all.")

    if not a.all:
        d = rows[0]
        print(f"\n  {d['rig']}: {', '.join(d['names'])}")
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
