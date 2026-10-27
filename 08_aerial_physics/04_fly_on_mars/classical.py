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
import flight_env
from flight_env import FlightEnv, CTRL_HZ

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


class FlightController:
    """
    Fly the gates, then land. No learning, no plan -- a guidance law.

    Two modes and one switch between them. While gates remain it
    steers at the NEXT gate in three dimensions with a speed cap;
    once the course is flown it reverts to lab 3's descent law over
    the last gate's ground point.

    This exists to answer a question before any policy is trained:
    is the course flyable at all with the propellant available? A
    learned result against an unflyable course measures the course.
    """

    def __init__(self, env, wn: float = 4.0, zeta: float = 0.9,
                 kz: float = 2.2, v_cruise: float = 9.0,
                 kv: float = 1.2) -> None:
        self.env = env
        self.alloc = Allocator(TH.effectiveness(env.rig))
        self.I = TH.inertia()
        self.wn, self.zeta, self.kz = wn, zeta, kz
        self.v_cruise = v_cruise
        self.kv = kv

    def act(self) -> np.ndarray:
        """Allocate the env's guidance law with the exact solver."""
        e = self.env
        w = e.base_wrench(wn=self.wn, zeta=self.zeta, kz=self.kz,
                          v_cruise=self.v_cruise, kv=self.kv)
        tk = flight_env.tank_state(e.m_gas, e.beta, e.world.name)
        return self.alloc.solve(w, tk["frac"])


def run(rig_name="dense", preset="rough", episodes=64, wn=4.0,
        seed0=900_000, verbose=False, world="mars",
        v_cruise=9.0, kv=1.2, **kw) -> dict:
    # The scheduled controller emits VALVE openings directly, so
    # it needs the valve interface whatever the learner uses.
    env = FlightEnv(rig_name, preset, world=world,
                    action="valves", **kw)
    tally: dict[str, int] = {}
    gates, gas, ok = [], [], 0
    for ep in range(episodes):
        env.reset(seed0 + ep)
        ctl = FlightController(env, wn=wn, v_cruise=v_cruise, kv=kv)
        while True:
            _o, _r, d, _ = env.step(ctl.act())
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gates.append(env.gate_i)
        gas.append(env.gas_used)
        ok += env.success()
        if verbose:
            print(f"    ep {ep:>3}: {env.outcome:<13} "
                  f"{env.gate_i}/{len(env.gates)} gates, "
                  f"{env.gas_used:.0f} kg")
    print(f"\n=== SCHEDULED CONTROLLER — {world} ===")
    for k in sorted(tally):
        print(f"  {k:<14} {tally[k]:>3}  ({tally[k] / episodes:.0%})")
    print(f"  flew the course   {ok}/{episodes}  ({ok / episodes:.0%})")
    print(f"  mean gates        {np.mean(gates):.2f} of {len(env.gates)}")
    print(f"  gas used          {np.mean(gas):.1f} kg of {TH.gas_mass():.0f}")
    return {"rate": ok / episodes, "gates": float(np.mean(gates)),
            "gas": float(np.mean(gas)), "tally": tally}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="rough")
    ap.add_argument("--world", default="mars", choices=("earth", "mars"))
    ap.add_argument("--episodes", type=int, default=64)
    # wn=4, not 8. At 8 the attitude loop chatters the valves and
    # the controller burned 288 kg of 297 while flying 2.5 gates; at
    # 4 it flies 3.0 gates on 183 kg. The gains were costing more
    # propellant than the course did.
    ap.add_argument("--wn", type=float, default=4.0)
    ap.add_argument("--v-cruise", type=float, default=9.0)
    ap.add_argument("--kv", type=float, default=1.2)
    ap.add_argument("--verbose", action="store_true")
    a, _ = ap.parse_known_args()
    run(a.rig, a.preset, a.episodes, a.wn, verbose=a.verbose,
        world=a.world, v_cruise=a.v_cruise, kv=a.kv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
