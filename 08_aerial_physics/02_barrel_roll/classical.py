#!/usr/bin/env python3
"""
The baseline: a scheduled barrel roll, then a PD landing. No learning.

    uv run classical.py --episodes 48
    uv run classical.py --rig minimal          # the zero-purity rig
    uv run classical.py --rate 4.5             # commanded roll rate

WHY A LAB ABOUT RL OPENS WITH A CONTROLLER THAT DOES NOT LEARN
---------------------------------------------------------------
"PPO learned a barrel roll" is a result only against something, and
the something has to be the obvious engineering answer. A scheduled
roll followed by an attitude hold is what an aerospace team would
fly, and rigid-body attitude control with thrusters has been flown
since the 1960s.

It also derisks the task. If the scheduled version cannot roll and
land, the task is impossible rather than hard, and six hours of
training would not have told you which.

THE TWO PHASES
--------------
  ROLL     rate control. Command omega_x at a fixed rate and drive
           omega_y and omega_z to zero, plus enough vertical force to
           limit the fall. Attitude is deliberately NOT regulated
           here: the reference is rotating, and a controller that
           fights for upright cannot roll.

  RECOVER  the moment the turn completes, switch to attitude control
           toward upright and run lab 1's descent profile.

The switch is where a scheduled controller is weakest. It commits to
a rate before it knows whether it has the authority or the altitude
to finish. A learned policy can change its mind.

THE ALLOCATION PROBLEM
----------------------
    min   || W (B u - w) ||^2 + lambda ||u||^2 + c 1'u
    s.t.  0 <= u <= 1

B is 6 x N with N > 6, so the system is over-actuated and the null
space is where the choice lives. Solved with projected gradient plus
Nesterov momentum and adaptive restart; the box constraint rules out
a closed form.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import thrusters as TH
from roll_env import RollEnv, CTRL_HZ

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


class RollController:
    """Phase 1 rolls at a commanded rate. Phase 2 lands."""

    def __init__(self, env, wn: float = 8.0, zeta: float = 0.9,
                 kz: float = 2.2, rate: float = 2.6,
                 k_rate: float = 6.0) -> None:
        self.env = env
        self.B = TH.effectiveness(env.rig)
        self.alloc = Allocator(self.B)
        self.wn, self.zeta, self.kz = wn, zeta, kz
        self.rate, self.k_rate = rate, k_rate
        self.I = TH.inertia()

    def act(self) -> np.ndarray:
        e = self.env
        R = e.rotmat()
        omega = e.omega()
        mass = e.m_dry_body + e.m_gas

        if e.require_roll and not e.rolled():
            # RATE control about the body x axis. The sign follows the
            # launch spin, so the controller finishes the turn the car
            # already started instead of reversing it, which would
            # cost most of the budget.
            sgn = 1.0 if e.roll_acc >= 0.0 else -1.0
            tau = self.I * (self.k_rate
                            * (np.array([sgn * self.rate, 0.0, 0.0]) - omega))
            # Hold altitude only loosely: supporting full weight while
            # inverted is impossible, so ask for what is useful and
            # let the allocator decline the rest.
            az = self.kz * (-1.0 - e.vel()[2])
            f_w = np.array([0.0, 0.0, mass * (e.world.g + az)])
        else:
            bz = R[:, 2]
            axis_w = np.cross(bz, np.array([0.0, 0.0, 1.0]))
            sn = float(np.linalg.norm(axis_w))
            cs = float(np.clip(bz @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
            ang = math.atan2(sn, cs)
            err_b = R.T @ ((axis_w / sn * ang) if sn > 1e-9 else np.zeros(3))
            Kp, Kd = self.wn ** 2, 2.0 * self.zeta * self.wn
            tau = self.I * (Kp * err_b - Kd * omega)

            alt = e.altitude()
            v_target = -np.clip(1.05 * max(alt - 0.9, 0.0) + 0.9, 0.0,
                                0.85 * 2.5)
            az = self.kz * (v_target - e.vel()[2])
            vh = e.vel()[:2]
            f_w = np.array([-0.9 * mass * vh[0], -0.9 * mass * vh[1],
                            mass * (e.world.g + az)])

        f_b = R.T @ f_w
        tk = __import__("roll_env").tank_state(e.m_gas, e.beta, e.world.name)
        return self.alloc.solve(np.concatenate([f_b, tau]), tk["frac"])


def run(rig_name="default", preset="rolling", flat=False, episodes=48,
        wn=8.0, seed0=0, verbose=False, world="earth", rate=4.0) -> dict:
    env = RollEnv(rig_name, preset, flat=flat, world=world)
    tally: dict[str, int] = {}
    sinks, tilts, gas, ok, rolled = [], [], [], 0, 0
    for ep in range(episodes):
        env.reset(seed0 + ep)
        ctl = RollController(env, wn=wn, rate=rate)
        while True:
            _o, _r, d, _ = env.step(ctl.act())
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gas.append(env.gas_used)
        rolled += int(env.rolled())
        if env.touchdown:
            sinks.append(env.touchdown["sink"])
            tilts.append(env.touchdown["tilt"])
        if env.success():
            ok += 1
        if verbose:
            print(f"    ep{ep:>3} {env.outcome:<18} "
                  f"roll {env.roll_progress():>6.1%} "
                  f"inv {int(env.went_inverted)} gas {env.gas_used:>6.1f}")
    return {"rig": rig_name, "preset": env.preset, "episodes": episodes,
            "wn": wn, "roll_cmd": rate,
            "landed_ok": ok, "rate": ok / episodes,
            "rolled": rolled, "roll_rate": rolled / episodes,
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
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
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
            d = run(a.rig, a.preset, a.flat, a.episodes, wn,
                    world=a.world)
            print(f"  {wn:>5} {d['rate']:>9.0%} {d['sink']:>6.2f}m/s "
                  f"{d['tilt']:>6.1f}° {d['gas']:>6.1f}kg")
        print("\n  The loop has 35 ms of actuator phase in it. Raising the")
        print("  bandwidth past what that supports does not fail politely.")
        return 0

    print("=" * 74)
    print(f"  SCHEDULED ROLL + PD LANDING — rig {a.rig}, "
          f"rate {a.rate} rad/s, wn {a.wn}")
    print("=" * 74)
    d = run(a.rig, a.preset, a.flat, a.episodes, a.wn,
            verbose=a.verbose, world=a.world, rate=a.rate)
    print(f"\n  completed the roll   {d['rolled']:>3}/{d['episodes']}  "
          f"({d['roll_rate']:.0%})")
    print(f"  rolled AND landed    {d['landed_ok']:>3}/{d['episodes']}  "
          f"({d['rate']:.0%})")
    for k in sorted(d["tally"]):
        print(f"    {k:<20} {d['tally'][k]:>3}")
    print(f"\n  touchdown sink  {d['sink']:.2f} m/s")
    print(f"  touchdown tilt  {d['tilt']:.1f} deg")
    print(f"  gas used        {d['gas']:.1f} kg of {TH.gas_mass():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
