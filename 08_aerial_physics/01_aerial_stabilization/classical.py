#!/usr/bin/env python3
"""
The baseline: PD attitude control plus a thrust allocator. No learning.

    uv run classical.py                      # score it
    uv run classical.py --rig minimal --episodes 60
    uv run classical.py --gains              # the gain design, shown

WHY A LAB ABOUT RL OPENS WITH A CONTROLLER THAT DOES NOT LEARN
---------------------------------------------------------------
Because "PPO learned to land the car" is not a result. It is a result
only against something, and the something has to be the obvious
engineering answer rather than a random policy. Attitude control of a
rigid body with thrusters is a SOLVED problem -- it has been flown on
spacecraft since the 1960s -- so a learned policy that merely matches
it has demonstrated nothing except that the simulator is
differentiable in spirit.

This also derisks the environment. If the classical controller cannot
land, the task is broken rather than hard, and no amount of training
would have told you which. Topic 7 built three policies against a
task whose robot could not be commanded at all.

WHAT IT DOES
------------
1. Pick a desired wrench: enough vertical force to arrest the fall,
   and a PD torque that drives the attitude error to zero.
2. Hand that 6-vector to an allocator, which finds valve commands
   whose wrench is closest to it.

THE ALLOCATION PROBLEM
----------------------
    min   || W (B u - w) ||^2 + lambda ||u||^2 + c 1'u
    s.t.  0 <= u <= 1

B is 6 x N with N > 6, so the system is over-actuated and the null
space is where the choice lives. The ridge picks one solution out of
it; the linear term is a propellant cost that prefers fewer, better
placed jets to many half-open ones. Solved with projected gradient
plus Nesterov momentum and adaptive restart -- the box constraint
rules out a closed form, and the Hessian here is badly conditioned
(the rig has a four-dimensional null space), which is exactly the
case where plain momentum stalls.

GAIN DESIGN
-----------
Each attitude axis closes as a double integrator, so with

    tau = I (-Kp e - Kd omega),      Kp = wn^2,  Kd = 2 zeta wn

the loop is a standard second-order system. `wn` cannot be raised
freely: the valve lag plus transport delay is about 35 ms of phase,
and past roughly 12 rad/s the loop rings and then diverges. That
ceiling is measured, not assumed -- `--sweep-gain` shows it.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import thrusters as TH
from aerial_env import AerialEnv, CTRL_HZ

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


class PDController:
    """Level the car, arrest the sink, do not waste gas."""

    def __init__(self, env: AerialEnv, wn: float = 8.0,
                 zeta: float = 0.9, kz: float = 2.2) -> None:
        self.env = env
        self.B = TH.effectiveness(env.rig)
        self.alloc = Allocator(self.B)
        self.wn, self.zeta, self.kz = wn, zeta, kz
        self.I = TH.inertia()

    def act(self) -> np.ndarray:
        e = self.env
        R = e.rotmat()
        omega = e.omega()

        # Attitude error as a rotation vector toward upright. The
        # shortest rotation taking the body z axis onto world z is
        # about the axis (body_z x world_z).
        bz = R[:, 2]
        axis_w = np.cross(bz, np.array([0.0, 0.0, 1.0]))
        s = float(np.linalg.norm(axis_w))
        c = float(np.clip(bz @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        ang = math.atan2(s, c)
        err_w = (axis_w / s * ang) if s > 1e-9 else np.zeros(3)
        err_b = R.T @ err_w          # error expressed in the body frame

        Kp, Kd = self.wn ** 2, 2.0 * self.zeta * self.wn
        tau = self.I * (Kp * err_b - Kd * omega)

        # Vertical: hold a descent rate that gets slower near the
        # ground, so the car arrives with something left rather than
        # hovering until the tank is dry.
        alt = e.altitude()
        # Descend at a rate proportional to the height left, floored
        # so it keeps moving and capped at the sink gate. Tuned
        # against the CLOCK as well as the gate: the first version
        # descended at 0.35 + 0.55h and timed out 13 runs in 16 while
        # landing the other 3 at a flawless 0.30 m/s. A controller
        # that lands perfectly and usually does not arrive is not a
        # baseline, it is a hover.
        v_target = -np.clip(1.05 * max(alt - 0.9, 0.0) + 0.9, 0.0,
                            0.85 * 2.5)
        az = self.kz * (v_target - e.vel()[2])
        mass = e.m_dry_body + e.m_gas
        # Bleed off horizontal velocity too. The canted corner
        # nozzles and the aft pair give some lateral authority; the
        # allocator works out how much of this it can actually honour
        # and the weighting makes it give way to attitude when the
        # two compete.
        vh = e.vel()[:2]
        f_w = np.array([-0.9 * mass * vh[0], -0.9 * mass * vh[1],
                        mass * (9.80665 + az)])
        f_b = R.T @ f_w

        _p, _t, avail = __import__("aerial_env").tank_state(e.m_gas, e.beta)
        return self.alloc.solve(np.concatenate([f_b, tau]), avail)


def run(rig_name="default", preset="rolling", flat=False, episodes=48,
        wn=8.0, seed0=0, verbose=False) -> dict:
    env = AerialEnv(rig_name, preset, flat=flat)
    tally: dict[str, int] = {}
    sinks, tilts, gas, ok = [], [], [], 0
    for ep in range(episodes):
        env.reset(seed0 + ep)
        pd = PDController(env, wn=wn)
        while True:
            _o, _r, d, _ = env.step(pd.act())
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gas.append(env.gas_used)
        if env.touchdown:
            sinks.append(env.touchdown["sink"])
            tilts.append(env.touchdown["tilt"])
        if env.landed_ok():
            ok += 1
        if verbose:
            print(f"    ep{ep:>3} {env.outcome:<13} gas {env.gas_used:>6.1f} kg")
    return {"rig": rig_name, "preset": env.preset, "episodes": episodes,
            "wn": wn, "landed_ok": ok, "rate": ok / episodes,
            "tally": tally,
            "sink": float(np.mean(sinks)) if sinks else float("nan"),
            "tilt": float(np.mean(tilts)) if tilts else float("nan"),
            "gas": float(np.mean(gas))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--wn", type=float, default=8.0)
    ap.add_argument("--sweep-gain", action="store_true")
    ap.add_argument("--gains", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a, _ = ap.parse_known_args()

    if a.gains:
        print("=" * 74)
        print("  GAIN DESIGN — second order per axis, tau = I(Kp e - Kd w)")
        print("=" * 74)
        print(f"  {'wn':>5} {'Kp':>9} {'Kd':>8} {'settle ~4/(z wn)':>18}")
        for wn in (4, 6, 8, 10, 12, 16, 22):
            print(f"  {wn:>5} {wn ** 2:>9.0f} {2 * 0.9 * wn:>8.1f} "
                  f"{4 / (0.9 * wn):>17.2f}s")
        print(f"\n  valve lag {25} ms + delay {10} ms ≈ 35 ms of phase. "
              f"Run --sweep-gain for where it actually breaks.")
        return 0

    if a.sweep_gain:
        print("=" * 74)
        print(f"  GAIN SWEEP — rig {a.rig}, {a.episodes} episodes each")
        print("=" * 74)
        print(f"  {'wn':>5} {'landed ok':>10} {'sink':>7} {'tilt':>7} {'gas':>7}")
        for wn in (4, 6, 8, 10, 12, 16, 22):
            d = run(a.rig, a.preset, a.flat, a.episodes, wn)
            print(f"  {wn:>5} {d['rate']:>9.0%} {d['sink']:>6.2f}m/s "
                  f"{d['tilt']:>6.1f}° {d['gas']:>6.1f}kg")
        print("\n  The loop has 35 ms of actuator phase in it. Raising the")
        print("  bandwidth past what that supports does not fail politely.")
        return 0

    print("=" * 74)
    print(f"  CLASSICAL BASELINE — rig {a.rig}, "
          f"terrain {'flat' if a.flat else a.preset}, wn {a.wn}")
    print("=" * 74)
    d = run(a.rig, a.preset, a.flat, a.episodes, a.wn, verbose=a.verbose)
    print(f"\n  landed cleanly  {d['landed_ok']}/{d['episodes']}  "
          f"({d['rate']:.0%})")
    for k in sorted(d["tally"]):
        print(f"    {k:<14} {d['tally'][k]:>3}")
    print(f"\n  touchdown sink  {d['sink']:.2f} m/s")
    print(f"  touchdown tilt  {d['tilt']:.1f}°")
    print(f"  gas used        {d['gas']:.1f} kg of {TH.M_GAS_0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
