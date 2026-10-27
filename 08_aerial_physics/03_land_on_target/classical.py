#!/usr/bin/env python3
"""
The baseline: pick the nearest good ground, fly to it, land. No learning.

    uv run classical.py --episodes 96
    uv run classical.py --blind        # ignore terrain, aim at the mark

WHY A LAB ABOUT PERCEPTION OPENS WITH A CONTROLLER THAT SEES EVERYTHING
------------------------------------------------------------------------
The question is whether terrain knowledge pays. The engineered
controller is the upper bound on what PERFECT knowledge buys, because
it reads the height field directly: it searches outward from the mark
for the nearest spot that is under 12 degrees and clear of boulders,
and flies there.

`--blind` is the same controller with the search removed -- it aims
at the mark itself, whatever is on it. The gap between those two is
how much terrain knowledge is WORTH before any policy is trained, and
it is the number the learned arms have to be read against.

It also derisks the task. If the seeing controller cannot land on the
mark, the task is impossible rather than hard.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import thrusters as TH
from target_env import TargetEnv, CTRL_HZ, survivable

# Weights: attitude rows matter more than force rows, and forces and
# torques are normalised to comparable scale first (newtons and
# newton-metres are not interchangeable units).
W_DIAG = np.array([0.5, 0.5, 1.0, 3.0, 3.0, 3.0]) / \
    np.array([5e4, 5e4, 5e4, 1.5e4, 1.5e4, 1.5e4])
RIDGE = 2e-3
GAS_COST = 4e-3


class Allocator:
    """FISTA with adaptive restart, warm-started between steps."""

    def __init__(self, B: np.ndarray) -> None:
        self.B = B
        self.W = np.diag(W_DIAG)
        self.WB = self.W @ B
        H = 2.0 * (self.WB.T @ self.WB)
        self.L = float(np.linalg.eigvalsh(H).max()) + 2.0 * RIDGE
        self.u = np.zeros(B.shape[1])

    def solve(self, w: np.ndarray, avail: float = 1.0,
              iters: int = 120) -> np.ndarray:
        a = max(avail, 1e-3)
        Wb = self.W @ w
        x = self.u.copy()
        y = x.copy()
        t = 1.0
        L = (a * a) * self.L
        for _ in range(iters):
            g = (2.0 * a * self.WB.T @ (a * (self.WB @ y) - Wb)
                 + 2.0 * RIDGE * y + GAS_COST)
            xn = np.clip(y - g / L, 0.0, 1.0)
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            # O'Donoghue-Candes restart: if the momentum points
            # uphill, drop it. Without this the badly conditioned
            # Hessian makes the iterate oscillate for hundreds of
            # steps and the allocation is visibly wrong mid-flight.
            if float((y - xn) @ (xn - x)) > 0.0:
                tn, yn = 1.0, xn
            else:
                yn = xn + ((t - 1.0) / tn) * (xn - x)
            x, y, t = xn, yn, tn
        self.u = x
        return x


class TargetController:
    """Choose a spot, fly to it, put the car down on it."""

    def __init__(self, env, wn: float = 8.0, zeta: float = 0.9,
                 kz: float = 2.2, blind: bool = False,
                 max_divert: float = 6.0) -> None:
        self.env = env
        self.alloc = Allocator(TH.effectiveness(env.rig))
        self.wn, self.zeta, self.kz = wn, zeta, kz
        self.I = TH.inertia()
        self.blind = blind
        self.max_divert = max_divert
        self.spot = self._choose()
        # Published for the renderer: the map draws the gap
        # between this and the mark, which IS the divert.
        env._render_spot = self.spot

    def _choose(self):
        """
        Nearest survivable ground to the mark, by outward search.

        Chosen ONCE at the start rather than re-evaluated step: a
        controller that keeps changing its mind about where to land
        oscillates and burns the tank. Committing early is also what
        the task demands of the learned policies, which have three
        seconds of flight.
        """
        e = self.env
        tx, ty = e.target
        if self.blind:
            return (tx, ty)
        if survivable(e.surf, tx, ty):
            return (tx, ty)
        # Only divert as far as the budget allows. Diverting is
        # not free -- it costs lateral travel, fuel, and the
        # target circle -- so 'go to the nearest safe ground'
        # is a different policy from 'go there IF it is close'.
        for R in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            if R > self.max_divert:
                break
            best, bestd = None, 1e9
            for k in range(max(8, int(R * 10))):
                a = 2.0 * math.pi * k / max(8, int(R * 10))
                x, y = tx + R * math.cos(a), ty + R * math.sin(a)
                if survivable(e.surf, x, y):
                    d = math.hypot(x - tx, y - ty)
                    if d < bestd:
                        best, bestd = (x, y), d
            if best is not None:
                return best
        return (tx, ty)

    def act(self) -> np.ndarray:
        e = self.env
        R = e.rotmat()
        omega = e.omega()
        mass = e.m_dry_body + e.m_gas
        p = e.pos()

        bz = R[:, 2]
        axis_w = np.cross(bz, np.array([0.0, 0.0, 1.0]))
        sn = float(np.linalg.norm(axis_w))
        cs = float(np.clip(bz @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        err_b = R.T @ ((axis_w / sn * math.atan2(sn, cs))
                       if sn > 1e-9 else np.zeros(3))
        Kp, Kd = self.wn ** 2, 2.0 * self.zeta * self.wn
        tau = self.I * (Kp * err_b - Kd * omega)

        # Lateral guidance to the chosen spot, and a descent rate that
        # slows as the gap closes -- descending fast while still 5 m
        # off the mark lands you wherever you happen to be.
        ex = self.spot[0] - p[0]
        ey = self.spot[1] - p[1]
        gap = math.hypot(ex, ey)
        alt = e.altitude()
        # Hold altitude while still closing on the spot -- but only
        # for the first 55% of the clock. The first version held
        # whenever the gap exceeded 1.2 m, and since the controller
        # settles to a ~1.4 m miss it held forever: 75% of episodes
        # timed out hovering a metre off the ground. A guidance law
        # that waits for a tolerance it cannot reach never lands.
        early = self.env.t < 0.55 * self.env.max_steps
        hold = 1.0 if (gap > 1.2 and alt < 3.0 and early) else 0.0
        v_target = -np.clip(1.0 * max(alt - 0.9, 0.0) + 0.8, 0.0, 2.1) \
            * (0.15 if hold else 1.0)
        az = self.kz * (v_target - e.vel()[2])
        vh = e.vel()[:2]
        ax = 1.1 * np.clip(ex, -3.0, 3.0) - 1.9 * vh[0]
        ay = 1.1 * np.clip(ey, -3.0, 3.0) - 1.9 * vh[1]
        f_w = np.array([mass * ax, mass * ay, mass * (e.world.g + az)])

        f_b = R.T @ f_w
        tk = __import__("target_env").tank_state(e.m_gas, e.beta,
                                                 e.world.name)
        return self.alloc.solve(np.concatenate([f_b, tau]), tk["frac"])


def run(rig_name="dense", preset="badlands", episodes=96, wn=8.0,
        seed0=900_000, verbose=False, blind=False,
        max_divert=6.0, **kw) -> dict:
    env = TargetEnv(rig_name, preset)
    tally: dict[str, int] = {}
    misses, gas, ok, alive = [], [], 0, 0
    for ep in range(episodes):
        env.reset(seed0 + ep)
        ctl = TargetController(env, wn=wn, blind=blind,
                               max_divert=max_divert)
        while True:
            _o, _r, d, _ = env.step(ctl.act())
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gas.append(env.gas_used)
        if env.touchdown:
            misses.append(env.touchdown["miss"])
            alive += 1
        if env.success():
            ok += 1
        if verbose:
            print(f"    ep{ep:>3} {env.outcome:<12} miss {env.miss():>5.1f} m "
                  f"gas {env.gas_used:>6.1f}")
    return {"rig": rig_name, "episodes": episodes, "blind": blind,
            "landed_ok": ok, "rate": ok / episodes,
            "survived": alive / episodes, "tally": tally,
            "miss": float(np.mean(misses)) if misses else float("nan"),
            "gas": float(np.mean(gas))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--wn", type=float, default=8.0)
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--max-divert", type=float, default=6.0,
                    help="how far the seeing controller will go "
                         "to reach safe ground")
    ap.add_argument("--blind", action="store_true",
                    help="ignore terrain, aim at the mark")
    ap.add_argument("--rate", type=float, default=2.6,
                    help="commanded roll rate, rad/s. 2.6 matches the "
                         "launch: measured 38%% roll-and-land on 252 kg "
                         "against 25%% on 271 kg at 5.0. Fighting the "
                         "rotation the ramp already gave you costs gas "
                         "twice, once to add it and once to remove it.")
    ap.add_argument("--sweep-gain", action="store_true")
    ap.add_argument("--gains", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a, _ = ap.parse_known_args()

    if a.gains:
        print("=" * 74)
    print(f"  ENGINEERED BASELINE — {'BLIND (aims at the mark)' if a.blind else 'SEEING (picks good ground)'}")
    print("=" * 74)
    d = run(a.rig, a.preset, a.episodes, a.wn, verbose=a.verbose,
            blind=a.blind, max_divert=a.max_divert)
    print(f"\n  on target (<= 4 m, gentle)  {d['landed_ok']:>3}/{d['episodes']}"
          f"  ({d['rate']:.0%})")
    print(f"  survived at all             {d['survived']:.0%}")
    for k in sorted(d["tally"]):
        print(f"    {k:<14} {d['tally'][k]:>3}")
    print(f"\n  mean miss  {d['miss']:.2f} m")
    print(f"  gas used   {d['gas']:.1f} kg of {TH.gas_mass():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
