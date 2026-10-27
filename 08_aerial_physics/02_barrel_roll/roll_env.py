#!/usr/bin/env python3
"""
The task: one full barrel roll in the air, then land on the wheels.

    uv run roll_env.py                  # random-policy sanity run
    uv run roll_env.py --budget         # can the tank even afford a roll?
    uv run roll_env.py --rig minimal --episodes 20

THE MANOEUVRE
-------------
The car is launched upward and tumbling. Before it comes back down it
must rotate a full 360 degrees about its own long axis -- a barrel
roll -- and then arrive on its wheels slowly enough to survive.

Lab 1 asked only for the landing. The roll changes the problem in a
way that is not merely "harder": it forces the vehicle to SPEND
attitude authority on a goal that conflicts with the landing, on a
clock, out of a tank that is also paying for the landing.

HOW THE ROLL IS MEASURED, AND WHY TWO WAYS
-------------------------------------------
Rotations do not commute, so the integral of body roll rate is only
a geometric angle when the manoeuvre is roll-DOMINANT. Relying on it
alone would let a policy bank credit by tumbling chaotically. So a
roll counts only if BOTH hold:

  1. the unwrapped integral of body-frame omega_x reaches 2*pi, and
  2. the car actually passes through INVERTED on the way -- its own
     up-axis points below the horizon at some instant.

(2) is a geometric event that no amount of yaw or pitch thrashing
can fake, and (1) ensures the rotation went one way round rather
than wobbling across inverted twice. The env also records peak
|omega_y| and |omega_z| so the roll-dominance assumption behind (1)
can be checked rather than assumed; `--budget` prints it.

WHAT THIS LAB PREDICTS BEFORE TRAINING
---------------------------------------
From lab 1's rig audit: a rig's PURE roll couple (torque with no net
force) can be far smaller than its FREE roll torque. `minimal` and
`sixlift` have 17.7 kN*m of free roll torque and ZERO pure couple --
every newton-metre of roll they produce arrives with a shove.

So the prediction is not that they cannot roll. It is that they roll
and then cannot land, having spent the flight undoing the
translation their own roll command caused. Purity, not authority,
should predict the outcome.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import car as CAR
import terrain
import thrusters as TH

CTRL_HZ = 100.0
PHYS_DT = 0.002
SUBSTEPS = int(round(1.0 / CTRL_HZ / PHYS_DT))

# Eight seconds, not lab 1's six. A 360 degree roll takes at least
# 0.90 s at `reveal`'s full pure couple and realistically twice that
# once the policy is also holding attitude, so lab 1's clock would
# have made the task impossible rather than hard -- and an impossible
# task trains to the same flat zero as a broken one.
MAX_SECONDS = 8.0            # with a roll to fit in
MAX_SECONDS_LAND = 6.0       # lab 1's clock, for the landing stage
MAX_STEPS = int(MAX_SECONDS * CTRL_HZ)

VALVE_TAU = 0.025
DELAY_S = 0.010
BETA_WALL = 0.30

SINK_OK = 2.5                 # m/s
TILT_OK = 25.0                # degrees from upright
SPIN_OK = 1.2                 # rad/s at touchdown
ROLL_TARGET = 2.0 * math.pi   # one full turn
INVERT_COS = -0.30            # body up-axis this far below horizontal


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tank_state(m_gas: float, beta: float = BETA_WALL,
               world: str = "earth") -> dict:
    """Propellant state from the reference equation of state. See lab 1."""
    import gas

    m_gas = max(m_gas, 0.0)
    rho = m_gas / TH.TANK_V
    rho0 = TH.gas_mass() / TH.TANK_V
    T = gas.blowdown_T(rho, rho0, TH.T_TANK, beta) if rho > 1e-6 else 80.0
    p = gas.tank_p(rho, T) if rho > 1e-6 else 0.0
    pc = min(TH.P_REG, p / TH.REG_HEADROOM)
    h = gas.tank_h(rho, T) if rho > 1e-6 else 0.0
    if pc <= gas._tab()["pc"][0] or m_gas <= 0.05:
        return dict(p=p, T=T, pc=0.0, ce=1.0, G=0.0, Te=0.0, frac=0.0, h=h)
    G0, ce0 = TH.design_point()
    ce = gas.nozzle_ce(pc, h, world)
    G = gas.nozzle_G(pc, h, world)
    return dict(p=p, T=T, pc=pc, ce=ce, G=G,
                Te=gas.nozzle_Te(pc, h, world), h=h,
                frac=float(np.clip((G * ce) / (G0 * ce0), 0.0, 1.5)))


class RollEnv:
    """One launch, one barrel roll, one landing attempt."""

    def __init__(self, rig_name: str = "default", preset: str = "rolling",
                 seed: int = 0, flat: bool = False, beta: float = BETA_WALL,
                 integrator: str = "RK4", world: str = "earth",
                 require_roll: bool = True, toss: float = 1.0,
                 launch: str = "stunt", alt_w: float = 1.5,
                 **rig_kw) -> None:
        import mujoco

        self._mj = mujoco
        self.rig_name = rig_name
        self.preset = "flat" if flat else preset
        self.beta = beta
        self.world = TH.WORLDS[world]
        self.require_roll = require_roll
        self.launch = launch
        self.alt_w = alt_w
        self.toss = toss
        self.rng = np.random.default_rng(seed)

        self.model, self.rig, self.surf = CAR.build(
            rig_name, preset, seed=seed, flat=flat, start_z=12.0,
            integrator=integrator, world=self.world, **rig_kw)
        self.data = mujoco.MjData(self.model)
        self.aid = CAR.resolve(self.model, self.rig)
        self.fmax = np.array([t.fmax for t in self.rig])
        self.at = np.array([t.throat_area for t in self.rig])
        self.act_dim = len(self.rig)

        self.chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "chassis")
        self.wheel_geoms, self.hull_geoms = set(), set()
        for g in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm is None:
                continue
            if nm.endswith("_tyre"):
                self.wheel_geoms.add(g)
            elif nm.startswith("hull_"):
                self.hull_geoms.add(g)
        if len(self.wheel_geoms) != 4 or not self.hull_geoms:
            raise ValueError("wheel/hull colliders did not resolve by name")

        self.m_dry_body = float(self.model.body_mass[self.chassis])
        self._wheel_mass = float(self.model.body_mass.sum()
                                 - self.model.body_mass[self.chassis])
        self._delay_n = max(1, int(round(DELAY_S * CTRL_HZ)))

        # The clock is lab 1's 6 s when there is no roll to fit
        # in, 8 s when there is. Stage A is then the task lab 1
        # already proved learnable, not a longer variant of it.
        self.max_steps = int((MAX_SECONDS if require_roll
                              else MAX_SECONDS_LAND) * CTRL_HZ)
        self.reset(seed)
        self.obs_dim = len(self._obs())

    # -- state -------------------------------------------------------
    def pos(self):
        return self.data.qpos[:3].copy()

    def quat(self):
        return self.data.qpos[3:7].copy()

    def rotmat(self):
        return _quat_to_mat(self.quat())

    def vel(self):
        return self.data.qvel[:3].copy()

    def omega(self):
        return self.data.qvel[3:6].copy()

    def up_dot(self) -> float:
        return float(self.rotmat()[2, 2])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.up_dot(), -1, 1))))

    def ground_z(self) -> float:
        p = self.pos()
        return 0.0 if self.surf is None else self.surf.height_at(p[0], p[1])

    def altitude(self) -> float:
        return float(self.pos()[2] - self.ground_z())

    def clearance(self) -> float:
        """Lowest corner of the car above the ground under it."""
        R, p = self.rotmat(), self.pos()
        lo = math.inf
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    w = p + R @ np.array([sx * 2.05, sy * 0.94, sz * 0.30])
                    gz = (0.0 if self.surf is None
                          else self.surf.height_at(w[0], w[1]))
                    lo = min(lo, w[2] - gz)
        return float(lo)

    def roll_progress(self) -> float:
        """Fraction of the required turn completed, 0 to 1."""
        return float(np.clip(abs(self.roll_acc) / ROLL_TARGET, 0.0, 1.0))

    def rolled(self) -> bool:
        """Both tests must pass. See the module docstring."""
        return (abs(self.roll_acc) >= ROLL_TARGET) and self.went_inverted

    # -- contacts ----------------------------------------------------
    def _touching(self):
        wheels, hull = set(), False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.wheel_geoms:
                    wheels.add(g)
                elif g in self.hull_geoms:
                    hull = True
        return len(wheels), hull

    def _apply_mass(self) -> None:
        self.model.body_mass[self.chassis] = self.m_dry_body + self.m_gas
        extra = self.m_gas * TH.Y_TANK ** 2
        self.model.body_inertia[self.chassis] = (
            self.I_body0 + np.array([extra, 0.0, extra]))
        tot = float(self.model.body_mass[self.chassis]) + self._wheel_mass
        self.model.body_subtreemass[self.chassis] = tot
        self.model.body_subtreemass[0] = tot

    # -- episode -----------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)
        if not hasattr(self, "I_body0"):
            self.I_body0 = self.model.body_inertia[self.chassis].copy()

        r = self.rng
        # A LAUNCH, not lab 1's toss. The car needs air time to turn
        # over: ~10 m up with 6-11 m/s of climb buys 3.5-4.5 s, and a
        # 360 degree roll costs at least 0.9 s of that.
        gz = 0.0 if self.surf is None else self.surf.height_at(0.0, 0.0)
        if not self.require_roll:
            # LAB 1'S EXACT LAUNCH. Stage A is not "lab 2 without the
            # roll" -- it is the task lab 1 reached 89% on, reproduced
            # so the curriculum starts from something known to be
            # learnable. Five attempts at inventing an easier stage A
            # all reached 0%, and every probe short enough to afford
            # was also too short to see lab 1's own breakout at 1.5M.
            alt = 5.0 + 2.5 * r.uniform(-1, 1)
            vz = r.uniform(1.0, 5.0)
        elif self.launch == "ramp":
            alt = r.uniform(1.5, 11.0)
            vz = r.uniform(0.0, 6.5 * (alt / 11.0))
        elif self.launch == "stunt":
            alt, vz = 9.0 + 2.0 * r.uniform(-1, 1), r.uniform(6.0, 11.0)
        else:
            alt, vz = 6.0 + 2.0 * r.uniform(-1, 1), r.uniform(3.0, 6.5)
        self.data.qpos[:3] = [0.0, 0.0, gz + alt]
        ang = math.radians(r.uniform(0, 50 if not self.require_roll
                                     else 30) * self.toss)
        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        self.data.qpos[3:7] = [math.cos(ang / 2), *(math.sin(ang / 2) * axis)]
        self.data.qvel[:3] = [r.uniform(-1.5, 6.0) * self.toss,
                              r.uniform(-1.0, 1.0) * self.toss,
                              vz]
        # A small initial roll rate, signed, so there is a direction to
        # commit to. Not enough to complete the turn on its own: at
        # 1.2 rad/s the car would need 5.2 s of free rotation.
        if not self.require_roll:
            sp = r.normal(size=3) * np.array([1.0, 0.45, 0.35])
            spin = sp / max(np.linalg.norm(sp), 1e-9) * r.uniform(0.5, 3.0)
        else:
            # LAUNCHED ALREADY ROLLING, at 2.2 to 3.2 rad/s.
            #
            # A car leaving a ramp sideways is rotating; it does not
            # start level and then decide to spin. At 0.6-1.2 rad/s
            # the first version needed 5-10 s to complete one turn --
            # longer than the episode -- so the policy had to
            # GENERATE the whole rotation, which is the exact
            # opposite of what stage A taught it (null the body
            # rates). It never attempted a roll in 2M steps.
            #
            # At 2.7 rad/s a free turn takes 2.3 s of the 8 s clock,
            # so the manoeuvre becomes what the real stunt is: do not
            # fight the rotation, let one turn complete, then arrest
            # it and land. Angular momentum is conserved in flight,
            # so doing nothing keeps rolling -- the skill is stopping
            # at exactly one turn, not starting.
            spin = np.array([r.uniform(2.2, 3.2) * r.choice([-1.0, 1.0]),
                             r.uniform(-0.3, 0.3), r.uniform(-0.3, 0.3)])
        self.data.qvel[3:6] = spin

        self.m_gas = TH.gas_mass()
        self.tank = tank_state(self.m_gas, self.beta, self.world.name)
        self._apply_mass()
        self.u_act = np.zeros(self.act_dim)
        self._hist = [np.zeros(self.act_dim) for _ in range(self._delay_n)]
        self.t = 0
        self.gas_used = 0.0
        self.roll_acc = 0.0
        self.went_inverted = False
        self.roll_done_step = None
        self.peak_wyz = 0.0
        self.outcome = ""
        self._settled = 0
        self.touchdown = None
        mujoco.mj_forward(self.model, self.data)
        self._prev_phi = self._potential()
        return self._obs()

    def _obs(self) -> np.ndarray:
        R = self.rotmat()
        tk = self.tank
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
            R[:, 0], R[:, 1],                       # 6 attitude
            [R[2, 2]],                              # 1 uprightness
            self.omega() / 5.0,                     # 3
            self.vel() / 10.0,                      # 3
            [self.altitude() / 10.0, self.clearance() / 10.0],
            [gx, gy],
            [self.m_gas / TH.gas_mass(), tk["frac"]],
            [1.0 - self.t / self.max_steps],
            # The roll state. Without these the task is not even
            # Markov: the policy cannot know whether it still owes a
            # turn, and no amount of training fixes a missing input.
            [self.roll_progress(),
             float(self.went_inverted),
             float(self.rolled())],
            self.u_act,
        ]).astype(np.float64)

    def step(self, action: np.ndarray):
        mujoco = self._mj
        u_cmd = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        self._hist.append(u_cmd)
        u_delayed = self._hist.pop(0)

        done = False
        for _ in range(SUBSTEPS):
            self.u_act += (u_delayed - self.u_act) * (PHYS_DT / VALVE_TAU)
            self.u_act = np.clip(self.u_act, 0.0, 1.0)
            tk = tank_state(self.m_gas, self.beta, self.world.name)
            self.tank = tk
            self.data.ctrl[self.aid] = self.u_act * tk["frac"]
            dm = float(np.sum(self.u_act * self.at)) * tk["G"] * PHYS_DT
            dm = min(dm, self.m_gas)
            self.m_gas -= dm
            self.gas_used += dm
            self._apply_mass()

            mujoco.mj_step(self.model, self.data)

            # Roll bookkeeping at PHYSICS rate, not control rate: at
            # 4 rad/s the car turns 11 degrees per control step, and
            # sampling the inversion test that coarsely would miss it.
            w = self.omega()
            self.roll_acc += float(w[0]) * PHYS_DT
            self.peak_wyz = max(self.peak_wyz,
                                float(max(abs(w[1]), abs(w[2]))))
            if self.up_dot() < INVERT_COS:
                self.went_inverted = True
            if self.rolled() and self.roll_done_step is None:
                self.roll_done_step = self.t

            n_wheel, hull = self._touching()
            if hull and self.altitude() < 2.5:
                self.outcome = "crash"
                done = True
                break
            if n_wheel >= 3 and abs(self.vel()[2]) < 1.5:
                self._settled += 1
                if self._settled >= 5:
                    self.outcome = "landed"
                    done = True
                    break
            else:
                self._settled = 0

        self.t += 1
        rew = self._reward()
        if not done and self.t >= self.max_steps:
            self.outcome = "timeout"
            done = True
        if done:
            rew += self._terminal()
        return self._obs(), rew, done, {}

    # -- reward ------------------------------------------------------
    def _potential(self) -> float:
        """
        Potential for shaping. NON-NEGATIVE, and that is the point.

        Potential-based shaping adds F = gamma*Phi(s') - Phi(s), which
        provably leaves the optimal policy unchanged (Ng, Harada &
        Russell 1999). The theorem is about the OPTIMAL POLICY; it
        says nothing about how pleasant the intermediate rewards are
        to collect, and with gamma < 1 the sign of Phi decides that.

        An agent that holds station collects

            F = gamma*Phi - Phi = (gamma - 1) * Phi

        every step. At gamma = 0.995 that is -0.005 * Phi. So:

            Phi < 0  ->  F > 0  ->  the agent is PAID to do nothing
            Phi > 0  ->  F < 0  ->  dawdling costs, finishing pays

        The first version used Phi = -2*altitude + 3*up + 14*progress,
        which sits near -15 on launch. Holding station therefore paid
        +0.075 per step -- about +60 over an eight second episode, for
        free, with no risk. The policy found it in 500k steps: roll
        completion fell from 21% to 12% while timeouts climbed to 86%.
        The shaping term added to make the goal findable was funding
        the hover.

        Shifting Phi to be non-negative fixes it. A constant offset
        cannot change the optimal policy either, but it flips the sign
        of the per-step leak from a subsidy into a clock.
        """
        if not self.require_roll:
            # Lab 1's potential, verbatim, negative sign and all. It
            # is the configuration that reached 89%, and reproducing
            # it is the point of this stage.
            return -2.5 * min(self.altitude(), 8.0) + 3.0 * self.up_dot()
        alt_term = self.alt_w * (12.0 - min(self.altitude(), 12.0))
        up_term = 2.0 * (self.up_dot() + 1.0)                  # 0 .. 4
        # 35, not 12. The roll potential has to make PARTIAL
        # progress worth something on its own, because a warm started
        # policy already knows how to land and will not gamble a
        # known reward on a manoeuvre it has never attempted. At 12
        # the gradient out of "never roll" was too shallow to climb:
        # reveal landed 94.8% of the time without once trying.
        roll_term = 35.0 * self.roll_progress()
        return alt_term + up_term + roll_term                  # 0 .. 34

    def _reward(self) -> float:
        dt = SUBSTEPS * PHYS_DT
        w = self.omega()
        r = 0.0
        # Uprightness is NOT rewarded per step, in either phase.
        #
        # The first version paid 2.0 * up_dot * dt once the roll was
        # done, exactly as lab 1 did. Lab 1 got away with it because
        # its episodes ENDED at ~150 steps; this one runs to 800, so
        # a policy that rolled and then hovered upright collected up
        # to +80 for doing nothing -- the same as landing perfectly,
        # at none of the risk. It found that immediately and sat at
        # 95% timeout.
        #
        # Uprightness is already in the shaping potential, where it
        # is banked once rather than per step. That is the right
        # place for it.
        if not self.require_roll:
            r += 2.0 * self.up_dot() * dt      # lab 1's dense upright term
            r -= 0.05 * float(w @ w) * dt
        elif self.rolled():
            r -= 0.05 * float(w @ w) * dt
        else:
            # Off-axis rates are still discouraged: a barrel roll is
            # rotation about ONE axis.
            r -= 0.05 * float(w[1] ** 2 + w[2] ** 2) * dt
        r -= 0.004 * float(np.sum(self.u_act)) * dt
        r += 0.05 * dt
        phi = self._potential()
        r += 0.995 * phi - self._prev_phi
        self._prev_phi = phi
        return r

    def _terminal(self) -> float:
        v = self.vel()
        sink = abs(min(v[2], 0.0))
        tilt = self.tilt_deg()
        spin = float(np.linalg.norm(self.omega()))
        rolled = self.rolled() or (not self.require_roll)

        if self.outcome == "crash":
            # A crash after completing the roll is still a crash, but
            # it got further, and the score says so. Flattening them
            # would erase the only gradient between "rolled and fell"
            # and "never tried".
            return -60.0 + (12.0 if rolled else 0.0)
        if self.outcome == "timeout":
            return -40.0 + 6.0 * self.up_dot() + (10.0 if rolled else 0.0)

        self.touchdown = dict(sink=sink, tilt=tilt, spin=spin)
        gentle = sink <= SINK_OK and tilt <= TILT_OK and spin <= SPIN_OK

        if not rolled:
            # Touched down without the roll. ONE price, whether the
            # arrival was neat or not.
            #
            # The first version charged -20 for a gentle no-roll
            # landing and only -5 for a hard one, so landing BADLY
            # scored better than landing well -- and 44.5% of
            # episodes became deliberate hard landings. Partial
            # credit for execution only makes sense once the task
            # itself has been attempted.
            self.outcome = "landed_no_roll"
            return -20.0
        if gentle:
            self.outcome = "rolled_and_landed"
            return (80.0 + 10.0 * (1.0 - sink / SINK_OK)
                    + 10.0 * (1.0 - tilt / TILT_OK))
        self.outcome = "landed_hard"
        return -5.0

    def success(self) -> bool:
        return self.outcome == "rolled_and_landed"


def random_policy(rng, env) -> np.ndarray:
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
        print("=" * 78)
        print("  CAN THE TANK AFFORD A BARREL ROLL?")
        print("=" * 78)
        print(f"  {'rig':<10} {'pure couple':>12} {'free':>10} {'purity':>7} "
              f"{'min 360':>9} {'hover s':>8}")
        for nm in ("minimal", "sixlift", "narrow", "reveal", "fore10",
                   "wide", "dense"):
            d = TH.audit(nm, TH.rig(nm))
            t = d["min_roll_time_s"]
            print(f"  {nm:<10} {d['roll_pure_Nm']:>11,.0f}N "
                  f"{d['roll_free_Nm']:>9,.0f}N {d['purity']:>6.0%} "
                  f"{(f'{t:.2f}s' if math.isfinite(t) else 'never'):>9} "
                  f"{d['hover_s']:>7.1f}s")
        print("\n  `min 360` is the fastest possible turn using a PURE")
        print("  couple: 2*sqrt(pi*I/tau). A rig with no pure couple shows")
        print("  'never' -- it can still roll, but only by also throwing")
        print("  itself somewhere, and that is the lab's whole question.")
        print(f"\n  flight time is ~{MAX_SECONDS:.0f} s max; the launch buys "
              f"about 3.5-4.5 s of air.")
        return 0

    env = RollEnv(a.rig, a.preset, flat=a.flat)
    print("=" * 78)
    print(f"  ROLL ENV — rig {a.rig} ({env.act_dim} nozzles), "
          f"terrain {env.preset}")
    print("=" * 78)
    print(f"  obs {env.obs_dim}   act {env.act_dim}   "
          f"{CTRL_HZ:.0f} Hz, max {MAX_SECONDS:.0f} s")

    rng = np.random.default_rng(0)
    tally: dict[str, int] = {}
    rolls = 0
    for ep in range(a.episodes):
        env.reset(ep)
        while True:
            _o, _r, d, _ = env.step(random_policy(rng, env))
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        rolls += int(env.rolled())
    print(f"\n  {a.episodes} episodes under a RANDOM policy:")
    for k in sorted(tally):
        print(f"    {k:<20} {tally[k]:>3}  ({tally[k] / a.episodes:.0%})")
    print(f"    completed the roll  {rolls:>3}  ({rolls / a.episodes:.0%})")
    print("\n  A random policy should essentially never roll AND land. If")
    print("  it does, the task is too easy and nothing trained on it means")
    print("  anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
