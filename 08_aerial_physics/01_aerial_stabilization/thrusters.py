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
# Worlds. Gravity, air and AMBIENT PRESSURE travel together.
#
# The first version of this lab ran Earth gravity (9.81) with Martian
# air density (0.02) on Mars terrain, which is not any planet. The
# three numbers are now one object so they cannot drift apart, and
# ambient pressure is in it because a nozzle's thrust depends on it:
# the (p_e - p_a)A_e term is the difference between Isp 60.6 and 65.8
# for the SAME hardware.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class World:
    name: str
    g: float                     # m/s^2
    rho_air: float               # kg/m^3
    p_amb: float                 # Pa


EARTH = World("earth", 9.80665, 1.225, 101325.0)
MARS = World("mars", 3.72076, 0.020, 610.0)
WORLDS = {w.name: w for w in (EARTH, MARS)}

G0 = 9.80665                     # m/s^2, the standard gravity Isp uses

# ---------------------------------------------------------------------------
# Vehicle and propellant constants.
# ---------------------------------------------------------------------------
M_DRY = 1750.0                   # kg, vehicle without gas
TANK_V = 0.600                   # m^3, the quoted 600 L
TANK_P0 = 689.5e5                # Pa, 10,000 psi COPV
T_TANK = 293.15                  # K
P_REG = 40.0e5                   # Pa, regulated chamber pressure
REG_HEADROOM = 1.15              # the regulator needs this much above p_c

# Design point for sizing nozzles: a full tank on EARTH. The rig is
# then a fixed piece of hardware, and flying it on Mars is a
# different question rather than a different vehicle.
DESIGN_WORLD = EARTH


def gas_mass_from_tank(volume: float = TANK_V, p0: float = TANK_P0,
                       T0: float = T_TANK) -> float:
    """
    How much nitrogen is actually in the tank.

    Derived from the reference equation of state, never quoted. This
    lab first published 244 kg from a hand-picked compressibility of
    1.95; CoolProp says Z = 1.602 and the answer is 297 kg. A
    constant that is also derivable should be derived, because the
    derivation is what fails loudly when someone edits the pressure.
    """
    import gas

    lo, hi = 2.0, 529.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gas.tank_p(mid, T0) < p0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * volume


M_GAS_0 = None                   # filled on first use; see gas_mass()


def gas_mass() -> float:
    global M_GAS_0
    if M_GAS_0 is None:
        M_GAS_0 = gas_mass_from_tank()
    return M_GAS_0


def wet_mass() -> float:
    return M_DRY + gas_mass()


# Body geometry. The car is a 4.34 m x 1.96 m roadster; these are the
# half-extents the nozzle stations are placed against.
HALF_L = 2.17
HALF_W = 0.98
DECK_Z = -0.25                   # z of the lower (down-firing) nozzles
ROOF_Z = 0.22                    # z of the upper (up-firing) nozzles

# Inertia of the dry car about its CG, roll/pitch/yaw.
I_DRY = np.array([800.0, 2950.0, 3250.0])
Y_TANK = 0.45                    # tanks sit here; gas adds to roll and yaw


def design_point(world=None):
    """
    (throat mass flux, effective exhaust velocity) at a full tank.

    Everything about nozzle SIZE is referred to this, so the gear
    MuJoCo is given is a real design-point thrust and the runtime
    scaling is a pure ratio.
    """
    import gas

    world = world or DESIGN_WORLD
    rho0 = gas_mass() / TANK_V
    h0 = gas.tank_h(rho0, T_TANK)
    return (gas.nozzle_G(P_REG, h0, world.name),
            gas.nozzle_ce(P_REG, h0, world.name))


def total_impulse(m_gas: float | None = None, world=None) -> float:
    """
    N*s available, integrated along the ACTUAL blowdown.

    Not m * v_e. Effective exhaust velocity falls from 594 m/s to
    353 m/s on Earth as the tank cools and the regulator drops out,
    so a constant-v_e figure overstates the budget by about a fifth.
    """
    import gas

    world = world or DESIGN_WORLD
    m0 = gas_mass() if m_gas is None else m_gas
    rho0 = gas_mass() / TANK_V
    ms = np.linspace(m0, 0.5, 400)
    tot = 0.0
    for k in range(len(ms) - 1):
        m = 0.5 * (ms[k] + ms[k + 1])
        rho = m / TANK_V
        T = gas.blowdown_T(rho, rho0, T_TANK, 0.30)
        p = gas.tank_p(rho, T)
        pc = min(P_REG, p / REG_HEADROOM)
        ce = gas.nozzle_ce(pc, gas.tank_h(rho, T), world.name)
        tot += ce * (ms[k] - ms[k + 1])
    return tot


def hover_seconds(world=None) -> float:
    world = world or DESIGN_WORLD
    return total_impulse(world=world) / (wet_mass() * world.g)


@dataclass(frozen=True)
class Thruster:
    """
    One nozzle.

    `direction` is the direction of the FORCE on the car (gas leaves
    the opposite way). `name` is load-bearing: the model resolves
    actuators by it.

    The SIZE of a nozzle is its THROAT AREA, not its thrust. Thrust
    is an outcome -- it falls as the tank cools and the regulator
    drops out, and it differs between Earth and Mars for the same
    hardware. `fmax` is kept as the design-point thrust because
    MuJoCo's actuator gear is a static number, and the environment
    scales the command by the ratio of current to design conditions.
    """
    name: str
    pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    throat_area: float             # m^2 -- the real size
    fmax: float                    # N at the design point; the gear
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
    mass = wet_mass() if mass is None else mass
    g_t0, ce0 = design_point()        # full tank, Earth, 40 bar

    def area_for(force_n: float) -> float:
        """Throat area that delivers `force_n` at the design point."""
        return force_n / (g_t0 * ce0)

    g = math.radians(cant_deg)
    sg, cg_ = math.sin(g), math.cos(g)

    n_lift = 2 * pairs
    # Sized so the LIFT COMPONENT (after the cant) hits the ratio.
    f_lift = thrust_ratio * mass * DESIGN_WORLD.g / (n_lift * cg_)
    f_up = upper_frac * f_lift
    f_aft = aft_frac * mass * DESIGN_WORLD.g / max(aft, 1)
    f_fore = aft_frac * mass * DESIGN_WORLD.g / max(fore, 1)

    # Station x positions, front to back, symmetric about the CG.
    if pairs == 1:
        xs = [0.0]
    else:
        xs = [spread - 2 * spread * i / (pairs - 1) for i in range(pairs)]

    out: list[Thruster] = []
    for k, x in enumerate(xs):
        # Station letters per ROW COUNT, not a shared alphabet. A
        # two-pair rig is Front and Rear; indexing one string gave it
        # "F" and "M" -- a middle station on a car that has none.
        # Topic 7 shipped exactly this, naming a two-pair body's rows
        # front and mid instead of front and hind.
        tag = {1: ["C"], 2: ["F", "R"], 3: ["F", "M", "R"],
               4: ["F", "A", "B", "R"]}.get(
            len(xs), [str(i) for i in range(len(xs))])[k]
        for side, sy in (("L", +1.0), ("R", -1.0)):
            # Lower: pushes the car UP, exhaust canted outboard.
            out.append(Thruster(
                name=f"lo{tag}{side}",
                pos=(x, sy * width, DECK_Z),
                direction=_unit((0.0, -sy * sg, cg_)),
                throat_area=area_for(f_lift), fmax=f_lift, role="lift"))
            if upper:
                # Upper: pushes the car DOWN. Same cant, mirrored.
                out.append(Thruster(
                    name=f"up{tag}{side}",
                    pos=(x, sy * roof_width, ROOF_Z),
                    direction=_unit((0.0, -sy * sg, -cg_)),
                    throat_area=area_for(f_up), fmax=f_up, role="down"))

    for i in range(aft):
        z = 0.0 if aft == 1 else -0.10 + 0.20 * i / max(aft - 1, 1)
        for side, sy in (("L", +1.0), ("R", -1.0)):
            out.append(Thruster(name=f"aft{i}{side}",
                                pos=(-HALF_L + 0.07, sy * 0.45, z),
                                direction=(1.0, 0.0, 0.0),
                                throat_area=area_for(f_aft), fmax=f_aft, role="aft"))
    for i in range(fore):
        for side, sy in (("L", +1.0), ("R", -1.0)):
            out.append(Thruster(name=f"fore{i}{side}",
                                pos=(HALF_L - 0.07, sy * 0.45, 0.0),
                                direction=(-1.0, 0.0, 0.0),
                                throat_area=area_for(f_fore), fmax=f_fore, role="fore"))

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
    # The two controls that pull `dense` apart. It was the only rig
    # with FORE nozzles, so its result confounded "more nozzles" with
    # "a thrust direction nothing else has" -- an ablation in which
    # two things moved. These move one each.
    #   dense8   dense's eight stations, WITHOUT the fore pair
    #   fore10   reveal's two stations, WITH a fore pair
    "dense8":    dict(pairs=4, upper=True, aft=1, fore=0),
    "fore10":    dict(pairs=2, upper=True, aft=1, fore=1),
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


AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def reach(B: np.ndarray) -> np.ndarray:
    """
    The achievable wrench POLYTOPE, summarised as its extent on each
    axis: a 6 x 2 array of (most negative, most positive).

    WHY RANK IS THE WRONG TEST, AND THIS IS THE RIGHT ONE
    ------------------------------------------------------
    A nozzle cannot fire backwards. Commands are box-constrained to
    [0, 1], so the set of achievable wrenches is

        { B u : 0 <= u <= 1 }

    which is a convex POLYTOPE (a zonotope), not a subspace. The rank
    of B asks whether the six axes are spanned when the coefficients
    may be any real number -- including negative ones the hardware
    cannot produce. It is necessary, and it is nowhere near
    sufficient.

    This lab shipped `rank` as its headline hardware metric and it
    was wrong. The `reveal` rig is rank 6 and can produce exactly
    ZERO backward force: every one of its horizontal nozzles pushes
    the same way. It can accelerate and it cannot brake. Measured
    against the classical controller, every rig that can brake lands
    at least 98% of the time and every rig that cannot tops out at
    83% -- a 12-nozzle rig beating an 18-nozzle one by 15 points.

    Because the bound is linear, each extent is a closed form: the
    maximum of a linear functional over a box is the sum of its
    positive coefficients. No solver.
    """
    return np.stack([[-np.sum(np.clip(-B[k], 0.0, None)),
                      np.sum(np.clip(B[k], 0.0, None))]
                     for k in range(6)])


def dead_axes(B: np.ndarray, tol: float = 1.0) -> list[str]:
    """
    Directions the rig cannot produce AT ALL, signed.

    Returns e.g. ['-Fx'] for a rig that can push forward but never
    brake. This is the check `rank` cannot make.
    """
    r = reach(B)
    out = []
    for k, nm in enumerate(AXES):
        if abs(r[k, 0]) < tol:
            out.append(f"-{nm}")
        if abs(r[k, 1]) < tol:
            out.append(f"+{nm}")
    return out


def inertia(m_gas: float | None = None) -> np.ndarray:
    """Roll/pitch/yaw inertia. The gas sits in the tanks, so it adds
    to roll and yaw but not to pitch."""
    m_gas = gas_mass() if m_gas is None else m_gas
    return I_DRY + m_gas * Y_TANK ** 2 * np.array([1.0, 0.0, 1.0])


def audit(name: str, rig_: list[Thruster], mass: float | None = None,
          world=None) -> dict:
    """
    Everything you can know about a rig WITHOUT training on it.

    This is the `--verify-arch` idea from the LLM labs, pointed at a
    vehicle: build the thing cheaply, interrogate it, and find out
    that the target names resolve and the arithmetic agrees -- before
    paying for the run that would otherwise discover it.
    """
    world = world or DESIGN_WORLD
    m = wet_mass() if mass is None else mass
    B = effectiveness(rig_)
    I = inertia()
    w = m * world.g

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
        "hover_s": hover_seconds(world),
        "world": world.name,
        "throat_mm2": [t.throat_area * 1e6 for t in rig_],
        "rank": int(np.linalg.matrix_rank(B, tol=1e-6)),
        "brake_N": float(-reach(B)[0, 0]),
        "brake_per_w": float(-reach(B)[0, 0]) / w,
        "dead_axes": dead_axes(B),
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
    ap.add_argument("--world", default="earth", choices=sorted(WORLDS))
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    world = WORLDS[a.world]
    g_t0, ce0 = design_point(world)
    m0 = gas_mass()
    print("=" * 78)
    print(f"  COLD-GAS RIG AUDIT — {world.name}, "
          f"g = {world.g:.2f} m/s², p_amb = {world.p_amb / 1e3:.1f} kPa")
    print("=" * 78)
    print(f"  tank      {TANK_V * 1000:.0f} L at {TANK_P0 / 1e5:.0f} bar, "
          f"{T_TANK:.1f} K")
    import gas as _gas
    rho0 = m0 / TANK_V
    print(f"  gas       {m0:.1f} kg  (Z = "
          f"{_gas.compressibility(rho0, T_TANK):.3f}; the ideal gas law "
          f"claims {TANK_P0 / (_gas.R_N2 * T_TANK) * TANK_V:.0f} kg,")
    print(f"            which is denser than LIQUID nitrogen)")
    print(f"  design    p_c {P_REG / 1e5:.0f} bar, area ratio "
          f"{_gas.area_ratio():.1f}, c_e {ce0:.0f} m/s "
          f"(Isp {ce0 / G0:.1f} s), G_t {g_t0:,.0f} kg/m²s")
    imp = total_impulse(world=world)
    print(f"  impulse   {imp / 1000:.1f} kN·s integrated along the real "
          f"blowdown")
    print(f"            (a constant-c_e figure would claim "
          f"{m0 * ce0 / 1000:.1f} kN·s — "
          f"{m0 * ce0 / imp - 1:+.0%})")
    print(f"  hover     {hover_seconds(world):.1f} s for the whole tank — "
          f"the budget every policy plays against")
    print(f"  mass      {wet_mass():.0f} kg, "
          f"I = {np.round(inertia(), 1)} kg·m²")

    names = sorted(RIGS) if a.all else [a.rig]
    rows = []
    for nm in names:
        r = rig(nm, pairs=a.pairs, upper=a.upper, aft=a.aft)
        rows.append(audit(nm, r, world=world))

    print(f"\n  {'rig':<10} {'N':>3} {'roll PURE':>10} {'360° in':>8} "
          f"{'rank':>5} {'brake/W':>8}  cannot produce")
    print("  " + "-" * 74)
    for d in rows:
        t = d["min_roll_time_s"]
        dead = ", ".join(d["dead_axes"]) or "—"
        print(f"  {d['rig']:<10} {d['n_thrusters']:>3} "
              f"{d['roll_pure_Nm']:>9,.0f}N "
              f"{(f'{t:.2f}s' if math.isfinite(t) else 'never'):>8} "
              f"{d['rank']:>4}/6 {d['brake_per_w']:>8.2f}  {dead}")

    print("\n  RANK IS NOT THE TEST. A nozzle cannot fire backwards, so")
    print("  commands live in [0,1] and the achievable wrench set is a")
    print("  convex POLYTOPE, not a subspace. `reveal` is rank 6 and can")
    print("  produce zero backward force — it accelerates and cannot brake.")
    print("  The last column is what rank misses.")
    print("\n  `roll PURE` is a true couple (torque with no net force). A rig")
    print("  with none can only roll by also throwing itself somewhere.")

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
