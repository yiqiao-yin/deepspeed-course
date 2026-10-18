#!/usr/bin/env python3
"""
The task: thrown into the air tumbling, land on your wheels.

    uv run aerial_env.py                   # a random-policy sanity run
    uv run aerial_env.py --rig minimal --episodes 20
    uv run aerial_env.py --budget           # what the tank can and cannot buy

THE TASK, AND WHY IT IS THIS ONE
---------------------------------
The car leaves a ramp at a randomised attitude with a randomised
tumble, somewhere between a gentle pitch and a rate that would put it
on its roof. It has a few seconds of air and a tank that holds about
eight seconds of hover. It has to arrive wheels-down, slowly enough
to survive, on ground it did not choose.

That is "aerial stabilization" with nothing else mixed in. There is
no driving, no ramp model and no downrange goal, because topic 7
spent a lab discovering that mixing terrain with policy leaves you
unable to say which one broke.

PROPELLANT IS THE WHOLE GAME
-----------------------------
A cold-gas thruster has a specific impulse of about 73 s. A quadrotor
can dither its attitude all day; this vehicle spends mass every
millisecond a valve is open, and the tank is sized in SECONDS. So the
reward charges for gas, and running dry is a terminal state rather
than a gentle degradation -- which is exactly what makes the problem
interesting rather than a hover task with extra steps.

WHAT IS MODELLED, AND WHAT IS NOT
----------------------------------
Modelled: 6-DOF rigid-body flight, real-gas tank blowdown with
Joule-Thomson-ish cooling, a regulator that drops out when the tank
falls below it, thrust proportional to chamber pressure, valve lag
and transport delay, mass and inertia that shrink as gas leaves,
wheel contact, hull strikes, thin Martian air.

NOT modelled, and stated rather than hidden: plume impingement on the
ground, nozzle over/under-expansion beyond a flat efficiency, tank
wall heat transfer as anything but a blend parameter, structural
loads, suspension travel beyond MuJoCo's contact solver, and any
aerodynamic surface. See the README's simplifications table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import car as CAR
import terrain
import thrusters as TH

CTRL_HZ = 100.0                      # policy decisions per second
PHYS_DT = 0.002                      # MuJoCo timestep
SUBSTEPS = int(round(1.0 / CTRL_HZ / PHYS_DT))
MAX_SECONDS = 6.0
MAX_STEPS = int(MAX_SECONDS * CTRL_HZ)

VALVE_TAU = 0.025                    # s, first-order valve lag
DELAY_S = 0.010                      # s, transport delay
GAMMA_N2 = 1.40
BETA_WALL = 0.30                     # share of the adiabatic drop the walls return

# Landing gates. A 2-tonne car arriving faster than this is a write-off
# whatever the attitude says.
SINK_OK = 2.5                        # m/s
TILT_OK = 25.0                       # degrees from upright
SPIN_OK = 1.2                        # rad/s at touchdown


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tank_state(m_gas: float, beta: float = BETA_WALL) -> tuple[float, float, float]:
    """
    Pressure, temperature and availability for the gas left.

    Real gas, because the ideal law is 25% wrong at this density and
    the error points the flattering way. Z is interpolated linearly
    in density between 1.0 at vacuum and `Z_N2` at the full tank --
    a two-point fit to the compressibility, which is crude and is
    SAID to be crude, rather than a table this lab would then have to
    verify.

    Temperature follows an adiabatic expansion blended toward ambient
    by `beta`: emptying a tank cools it, a cold tank gives less
    pressure, and a policy that fires everything early pays for it
    later. beta = 0 is adiabatic, beta = 1 is isothermal.
    """
    m_gas = max(m_gas, 0.0)
    rho = m_gas / TH.TANK_V
    rho0 = TH.M_GAS_0 / TH.TANK_V
    z = 1.0 + (TH.Z_N2 - 1.0) * (rho / rho0)
    t_ad = TH.T_TANK * max(rho / rho0, 1e-6) ** (GAMMA_N2 - 1.0)
    t = t_ad + beta * (TH.T_TANK - t_ad)
    p = z * rho * TH.R_N2 * t
    avail = float(np.clip(p / TH.P_DROPOUT, 0.0, 1.0))
    return p, t, avail


class AerialEnv:
    """
    One vehicle, one throw, one landing attempt.

    EVERY dimension is derived, never typed: `act_dim` comes from the
    rig, `obs_dim` from assembling one observation and measuring it.
    Topic 7 shipped `random_policy(rng, n=7)` -- a default action
    width that no longer matched the body -- and a sweep that reported
    "FAILED training" for rigs the parser simply rejected.
    """

    def __init__(self, rig_name: str = "default", preset: str = "rolling",
                 seed: int = 0, flat: bool = False,
                 roll: bool = False, beta: float = BETA_WALL,
                 integrator: str = "implicitfast",
                 toss: float = 1.0, **rig_kw) -> None:
        import mujoco

        self._mj = mujoco
        self.rig_name = rig_name
        self.preset = "flat" if flat else preset
        self.roll_task = roll
        self.beta = beta
        self.toss = toss
        self.rng = np.random.default_rng(seed)

        self.model, self.rig, self.surf = CAR.build(
            rig_name, preset, seed=seed, flat=flat, start_z=6.0,
            integrator=integrator, **rig_kw)
        self.data = mujoco.MjData(self.model)
        self.aid = CAR.resolve(self.model, self.rig)
        self.fmax = np.array([t.fmax for t in self.rig])
        self.act_dim = len(self.rig)

        self.chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "chassis")
        if self.chassis < 0:
            raise KeyError("chassis body missing from the model")
        # Geoms resolved BY NAME. A wheel touching the ground is a
        # landing; anything else touching it is a crash, and telling
        # them apart by geom INDEX is the bug topic 7 shipped three
        # times.
        self.wheel_geoms = set()
        self.hull_geoms = set()
        for g in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm is None:
                continue
            if nm.endswith("_tyre"):
                self.wheel_geoms.add(g)
            elif nm.startswith("hull_"):
                self.hull_geoms.add(g)
        if len(self.wheel_geoms) != 4:
            raise ValueError(f"expected 4 named wheel colliders, "
                             f"found {len(self.wheel_geoms)}")
        if not self.hull_geoms:
            raise ValueError("no hull colliders found — a crash could "
                             "never be detected")
        self.ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        "ground")

        self.m_dry_body = float(self.model.body_mass[self.chassis])
        self.I_body0 = self.model.body_inertia[self.chassis].copy()
        self._delay_n = max(1, int(round(DELAY_S * CTRL_HZ)))

        self.reset(seed)
        self.obs_dim = len(self._obs())
        self.max_steps = MAX_STEPS

    # -- state -------------------------------------------------------
    def pos(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def rotmat(self) -> np.ndarray:
        return _quat_to_mat(self.quat())

    def vel(self) -> np.ndarray:
        return self.data.qvel[:3].copy()

    def omega(self) -> np.ndarray:
        """Body-frame angular velocity. MuJoCo stores the free joint's
        rotational velocity in the BODY frame already."""
        return self.data.qvel[3:6].copy()

    def up_dot(self) -> float:
        """Cosine of the tilt: +1 upright, -1 on the roof."""
        return float(self.rotmat()[2, 2])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.up_dot(), -1, 1))))

    def ground_z(self) -> float:
        p = self.pos()
        return 0.0 if self.surf is None else self.surf.height_at(p[0], p[1])

    def altitude(self) -> float:
        return float(self.pos()[2] - self.ground_z())

    def clearance(self) -> float:
        """Lowest point of the car above the ground directly under it.

        This is the number that decides whether a tilted car has
        actually hit, and it is NOT the CG height. Topic 7 measured a
        robot half a limb short by reading a capsule's centre instead
        of its tip and recorded a 100% false-fall rate for it.
        """
        R, p = self.rotmat(), self.pos()
        lo = math.inf
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    c = np.array([sx * 2.05, sy * 0.94, sz * 0.30])
                    w = p + R @ c
                    gz = (0.0 if self.surf is None
                          else self.surf.height_at(w[0], w[1]))
                    lo = min(lo, w[2] - gz)
        return float(lo)

    # -- contacts ----------------------------------------------------
    def _touching(self) -> tuple[int, bool]:
        """(wheels in contact, any hull contact)."""
        wheels, hull = set(), False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.wheel_geoms:
                    wheels.add(g)
                elif g in self.hull_geoms:
                    hull = True
        return len(wheels), hull

    # -- gas ---------------------------------------------------------
    def _apply_mass(self) -> None:
        """Mass and inertia shrink as gas leaves. 244 kg of 1994 is
        12%, and the roll inertia moves more than the mass does
        because the tanks sit outboard."""
        self.model.body_mass[self.chassis] = self.m_dry_body + self.m_gas
        extra = self.m_gas * TH.Y_TANK ** 2
        self.model.body_inertia[self.chassis] = (
            self.I_body0 + np.array([extra, 0.0, extra]))

    # -- episode -----------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)

        r = self.rng
        # The throw. Randomised so a policy cannot memorise one arc.
        alt = 5.0 + 2.5 * r.uniform(-1, 1) * self.toss
        gz = 0.0 if self.surf is None else self.surf.height_at(0.0, 0.0)
        self.data.qpos[:3] = [0.0, 0.0, gz + alt]

        # A random tumble: a rotation of up to 50 deg off upright, and
        # a body rate of up to 3 rad/s mostly in roll. That upper rate
        # takes ~1.0 s to put the car on its roof, which is inside the
        # flight time -- so doing nothing genuinely loses.
        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = math.radians(r.uniform(0, 50) * self.toss)
        self.data.qpos[3:7] = [math.cos(ang / 2), *(math.sin(ang / 2) * axis)]
        self.data.qvel[:3] = [r.uniform(-1.5, 6.0) * self.toss,
                              r.uniform(-1.0, 1.0) * self.toss,
                              r.uniform(1.0, 5.0) * self.toss]
        spin = r.normal(size=3) * np.array([1.0, 0.45, 0.35])
        self.data.qvel[3:6] = spin / max(np.linalg.norm(spin), 1e-9) * \
            r.uniform(0.5, 3.0) * self.toss

        self.m_gas = TH.M_GAS_0
        self._apply_mass()
        self.u_act = np.zeros(self.act_dim)
        self._hist = [np.zeros(self.act_dim) for _ in range(self._delay_n)]
        self.t = 0
        self.gas_used = 0.0
        self.roll_acc = 0.0
        self._prev_roll_sign = None
        self.outcome = ""
        self._settled = 0
        self.touchdown = None
        self._prev_phi = self._potential()
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self) -> np.ndarray:
        R = self.rotmat()
        _p, _t, avail = tank_state(self.m_gas, self.beta)
        alt = self.altitude()
        # Local slope under the car, as a gradient. A landing has to
        # match ground the car did not pick.
        if self.surf is None:
            gx = gy = 0.0
        else:
            p = self.pos()
            d = 1.0
            gx = (self.surf.height_at(p[0] + d, p[1])
                  - self.surf.height_at(p[0] - d, p[1])) / (2 * d)
            gy = (self.surf.height_at(p[0], p[1] + d)
                  - self.surf.height_at(p[0], p[1] - d)) / (2 * d)
        return np.concatenate([
            R[:, 0], R[:, 1],                     # 6: attitude, continuous
            np.array([R[2, 2]]),                  # 1: uprightness, explicitly
            self.omega() / 5.0,                   # 3
            self.vel() / 10.0,                    # 3
            [alt / 6.0, self.clearance() / 6.0],  # 2
            [gx, gy],                             # 2
            [self.m_gas / TH.M_GAS_0, avail],     # 2
            [1.0 - self.t / MAX_STEPS],           # 1
            self.u_act,                           # act_dim
        ]).astype(np.float64)

    def step(self, action: np.ndarray):
        mujoco = self._mj
        u_cmd = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        self._hist.append(u_cmd)
        u_delayed = self._hist.pop(0)

        rew = 0.0
        done = False
        for _ in range(SUBSTEPS):
            # Valve lag, then availability, then the actual command.
            self.u_act += (u_delayed - self.u_act) * (PHYS_DT / VALVE_TAU)
            self.u_act = np.clip(self.u_act, 0.0, 1.0)
            _p, _t, avail = tank_state(self.m_gas, self.beta)
            if self.m_gas <= 0.05:
                avail = 0.0
            eff = self.u_act * avail
            self.data.ctrl[self.aid] = eff

            thrust = float(np.sum(eff * self.fmax))
            dm = thrust / TH.VE * PHYS_DT
            dm = min(dm, self.m_gas)
            self.m_gas -= dm
            self.gas_used += dm
            self._apply_mass()

            mujoco.mj_step(self.model, self.data)

            n_wheel, hull = self._touching()
            if hull and self.altitude() < 2.0:
                self.outcome = "crash"
                done = True
                break
            # LANDED means down on the wheels and no longer falling.
            # It deliberately does NOT require the car to be stopped:
            # the first version tested total speed < 1.5 m/s, and the
            # throw hands the car up to 6 m/s of downrange velocity,
            # so a car sitting squarely on all four wheels rolling
            # forward scored as "still flying" and timed out. Eleven
            # runs in sixteen. Wheels are for exactly that.
            if n_wheel >= 3 and abs(self.vel()[2]) < 1.5:
                self._settled += 1
                if self._settled >= 5:
                    self.outcome = "landed"
                    done = True
                    break
            else:
                self._settled = 0

        # Roll progress, unwrapped, for the barrel-roll variant.
        self.roll_acc += float(self.omega()[0]) * (SUBSTEPS * PHYS_DT)
        self.t += 1

        rew += self._reward(done)
        if not done and self.t >= MAX_STEPS:
            self.outcome = self.outcome or "timeout"
            done = True
        if done and self.outcome in ("landed", "crash", "timeout"):
            rew += self._terminal()
        return self._obs(), rew, done, {}

    def _potential(self) -> float:
        """
        Potential for shaping: high when low and upright.

        POTENTIAL-BASED shaping (Ng, Harada & Russell 1999) adds
        gamma*Phi(s') - Phi(s) to the reward, which provably leaves
        the optimal policy unchanged while still telling the agent
        which way is downhill. That guarantee is the reason to use
        this shape rather than a hand-added "descend!" bonus, which
        would change what optimal MEANS and can be collected by
        diving into the ground.
        """
        # The altitude weight is 2.5, not 0.9. At 0.9 a metre of
        # descent was worth less than a tenth of what staying level
        # paid, so the policy banked the level bonus and hovered to
        # the clock -- landings went 1.0%, 0.5%, 0.2%, 0.0% over
        # 550k steps while crashes fell 98% to 2%. It had learned
        # the first half of the task and was being paid to stop.
        return -2.5 * min(self.altitude(), 8.0) + 3.0 * self.up_dot()

    def _reward(self, done: bool) -> float:
        """
        Dense, and every term is here because leaving it out broke
        something. Weights are in the README's table.

        THE TERM THAT HAD TO BE ADDED, AND WHY
        ---------------------------------------
        The first version had an alive bonus and an uprightness bonus
        and no altitude term at all. PPO drove crashes from 98% down
        to 44% over 490k steps and turned every one of them into a
        TIMEOUT: it learned to stabilise and then hover for the full
        six seconds, because hovering banked a certain +62 while
        attempting a landing risked -60.

        That was not a bug. It was the correct answer to the question
        actually asked. The reward said "be upright and stay alive",
        and the policy did exactly that, forever.
        """
        dt = SUBSTEPS * PHYS_DT
        up = self.up_dot()
        w = self.omega()
        r = 0.0
        r += 2.0 * up * dt                      # be upright
        r -= 0.05 * float(w @ w) * dt           # stop tumbling
        r -= 0.004 * float(np.sum(self.u_act)) * dt   # gas costs
        r -= 0.02 * max(0.0, abs(self.vel()[2]) - 4.0) * dt   # do not dive
        r += 0.05 * dt                          # alive, but barely
        phi = self._potential()
        r += 0.995 * phi - self._prev_phi       # shaping, see above
        self._prev_phi = phi
        return r

    def _terminal(self) -> float:
        v = self.vel()
        sink = abs(min(v[2], 0.0))
        tilt = self.tilt_deg()
        spin = float(np.linalg.norm(self.omega()))
        if self.outcome == "crash":
            return -60.0
        if self.outcome == "timeout":
            # Still airborne when the clock ran out. This is a
            # FAILURE, not a draw. The first version priced it at -10
            # against a crash's -60, which made hovering the rational
            # play -- and that is precisely what the policy found.
            return -40.0 + 6.0 * self.up_dot()
        self.touchdown = dict(sink=sink, tilt=tilt, spin=spin)
        if sink <= SINK_OK and tilt <= TILT_OK and spin <= SPIN_OK:
            self.outcome = "landed_ok"
            return (60.0 + 10.0 * (1.0 - sink / SINK_OK)
                    + 10.0 * (1.0 - tilt / TILT_OK))
        self.outcome = "landed_hard"
        return -5.0

    def landed_ok(self) -> bool:
        return self.outcome == "landed_ok"


def random_policy(rng: np.random.Generator, env: AerialEnv) -> np.ndarray:
    """Width comes from the ENV, not from a default argument."""
    return rng.uniform(0.0, 0.6, env.act_dim)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--budget", action="store_true")
    a, _ = ap.parse_known_args()

    if a.budget:
        print("=" * 74)
        print("  THE TANK — what it can and cannot buy")
        print("=" * 74)
        print(f"  {'gas kg':>7} {'bar':>8} {'K':>7} {'avail':>7}")
        for m in (244, 200, 150, 100, 60, 30, 12, 4):
            p, t, av = tank_state(float(m))
            print(f"  {m:>7} {p / 1e5:>8.0f} {t:>7.0f} {av:>6.0%}")
        print(f"\n  regulator drops out below {TH.P_DROPOUT / 1e5:.0f} bar; "
              f"below that thrust falls with tank pressure.")
        print(f"  whole tank  = {TH.total_impulse() / 1000:.0f} kN·s "
              f"= {TH.hover_seconds():.1f} s of hover")
        return 0

    env = AerialEnv(a.rig, a.preset, flat=a.flat)
    print("=" * 74)
    print(f"  ENV — rig {a.rig} ({env.act_dim} thrusters), "
          f"terrain {env.preset}")
    print("=" * 74)
    print(f"  obs {env.obs_dim}   act {env.act_dim}   "
          f"{CTRL_HZ:.0f} Hz, {SUBSTEPS} physics steps per decision, "
          f"max {MAX_SECONDS:.0f} s")

    rng = np.random.default_rng(0)
    tally: dict[str, int] = {}
    for ep in range(a.episodes):
        env.reset(ep)
        total = 0.0
        while True:
            _o, r, d, _ = env.step(random_policy(rng, env))
            total += r
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1

    print(f"\n  {a.episodes} episodes under a RANDOM policy:")
    for k in sorted(tally):
        print(f"    {k:<14} {tally[k]:>3}  ({tally[k] / a.episodes:.0%})")
    print("\n  A random policy should almost never land cleanly. If it")
    print("  does, the task is too easy and nothing trained on it means")
    print("  anything — that is the check, not the throughput.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
