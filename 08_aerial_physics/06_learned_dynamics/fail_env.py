#!/usr/bin/env python3
"""
Land a cold-gas car after a thruster breaks in mid-air.

    uv run fail_env.py --budget            # which failures are survivable?
    uv run fail_env.py --episodes 12       # random-policy sanity run
    uv run fail_env.py --mode open --rig reveal

THE TASK
--------
Lab 1's task -- a tumbling car, six seconds, arrive on the wheels --
with one nozzle breaking at a random moment between 0.5 s and 3.0 s.
Everything else is lab 1's: the launch envelope, the clock, the
landing gate.

TWO FAILURE MODES, AND THEY ARE NOT SYMMETRIC
----------------------------------------------
  shut   the nozzle stops responding. Authority is lost.
  open   the nozzle latches wide open and cannot be closed. Authority
         is lost, a force and torque arrive that nobody asked for,
         and propellant drains until the tank is dry.

`fault.py` answers which of those are survivable AS GEOMETRY, before
anything is trained: with nozzle k gone, is a hover still inside the
reachable wrench polytope? On `reveal` that is 4 of 10 fatal shut and
6 of 10 fatal open.

THE POLICY IS NOT TOLD, DELIBERATELY
-------------------------------------
There is no fault flag in the observation. The vehicle finds out the
way a real one does: it commands a wrench, gets a different one, and
has to notice from the dynamics. A `--tell` arm exists to price what
a perfect fault-detection system would be worth, which is a separate
question from whether the policy can cope without one.

WHAT THE LAB MEASURES
---------------------
The gap between the geometry and the flight. A failure the polytope
says is survivable still has to be CAUGHT -- the transient tumble
starts at altitude and the ground does not wait. "Reachable is not
recoverable" is this lab's version of lab 1's "full rank is not
reachability".
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

# The discount, and it is a property of the TASK rather than a free
# knob -- see lab 3. At 0.995 the horizon 1/(1-gamma) is 200 steps
# against a 600-step episode, so a terminal penalty is discounted to
# 0.05 and the policy postpones rather than avoids.
GAMMA = 0.999
# A constant charge per airborne step. Lab 3's rule: if every episode
# times out, charge for time rather than raising the terminal.
TIME_COST = 0.10

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


def tank_state(m_gas: float, beta: float = BETA_WALL,
               world: str = "earth") -> dict:
    """
    The complete propellant state, from the reference equation of state.

    Returns tank pressure and temperature, chamber pressure after the
    regulator, effective exhaust velocity, throat mass flux, nozzle
    exit temperature, and the THRUST FRACTION -- current thrust over
    design-point thrust, which is what scales the actuator command.

    Nothing here is a correlation. The first version used an
    isothermal tank, a compressibility of 1.95 picked by hand and a
    constant exhaust velocity of 659 m/s. All three were wrong in the
    same direction as each other and in the opposite direction to the
    error they were covering for.
    """
    import gas

    import thrusters as TH

    m_gas = max(m_gas, 0.0)
    rho = m_gas / TH.TANK_V
    rho0 = TH.gas_mass() / TH.TANK_V
    T = gas.blowdown_T(rho, rho0, TH.T_TANK, beta) if rho > 1e-6 else 80.0
    p = gas.tank_p(rho, T) if rho > 1e-6 else 0.0
    pc = min(TH.P_REG, p / TH.REG_HEADROOM)
    h = gas.tank_h(rho, T) if rho > 1e-6 else 0.0
    if pc <= gas._tab()["pc"][0] or m_gas <= 0.05:
        return dict(p=p, T=T, pc=0.0, ce=1.0, G=0.0, Te=0.0, frac=0.0, h=h)
    ce = gas.nozzle_ce(pc, h, world)
    G = gas.nozzle_G(pc, h, world)
    Te = gas.nozzle_Te(pc, h, world)
    G0, ce0 = TH.design_point()
    return dict(p=p, T=T, pc=pc, ce=ce, G=G, Te=Te, h=h,
                frac=float(np.clip((G * ce) / (G0 * ce0), 0.0, 1.5)))


class FailEnv:
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
                 integrator: str = "RK4", world: str = "earth",
                 toss: float = 1.0,
                 fail_p: float = 1.0, fail_mode: str = "mix",
                 fail_at: tuple[float, float] = (0.5, 3.0),
                 tell: bool = False, gamma: float = GAMMA,
                 **rig_kw) -> None:
        import mujoco

        self._mj = mujoco
        self.rig_name = rig_name
        self.preset = "flat" if flat else preset
        self.roll_task = roll
        self.beta = beta
        self.world = TH.WORLDS[world]
        self.toss = toss
        # FAULT INJECTION.
        #   fail_p    chance an episode contains a failure at all. The
        #             healthy arm trains at 0.0, the robust arm at the
        #             default, and both are SCORED on the same draws.
        #   fail_mode shut / open / mix.
        #   fail_at   the window, in seconds, the break happens in. It
        #             never fires at t=0: a vehicle that was always
        #             born broken would let the policy fold the fault
        #             into its launch reflex instead of detecting it.
        #   tell      put the fault in the observation. Off by default
        #             -- the vehicle finds out the way a real one does,
        #             by commanding a wrench and getting another.
        if fail_mode not in ("shut", "open", "mix"):
            raise ValueError(f"fail_mode must be shut/open/mix, got {fail_mode!r}")
        self.fail_p = float(fail_p)
        self.fail_mode = fail_mode
        self.fail_at = fail_at
        self.tell = bool(tell)
        # Must equal the trainer's gamma or the shaping stops
        # being potential-based. Lab 3 shipped two copies of
        # this number with nothing making them agree.
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        self.model, self.rig, self.surf = CAR.build(
            rig_name, preset, seed=seed, flat=flat, start_z=6.0,
            integrator=integrator, world=self.world, **rig_kw)
        self.data = mujoco.MjData(self.model)
        self.aid = CAR.resolve(self.model, self.rig)
        self.fmax = np.array([t.fmax for t in self.rig])
        self.at = np.array([t.throat_area for t in self.rig])
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
        # Everything hanging off the chassis (four wheels), so the
        # subtree mass can be kept correct as gas leaves.
        self._wheel_mass = float(self.model.body_mass.sum()
                                 - self.model.body_mass[self.chassis])
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
        """
        Mass and inertia shrink as gas leaves: 297 kg of 2047 is 15%,
        and roll inertia moves more than mass does because the tanks
        sit outboard.

        `body_subtreemass` MUST be maintained alongside `body_mass`.
        MuJoCo derives `subtree_com` by dividing accumulated mass
        moments by `body_subtreemass`, and the composite-rigid-body
        dynamics are expressed about that point -- so leaving it at
        its compile-time value does not merely mis-report the centre
        of mass, it SIMULATES THE WRONG VEHICLE.

        This shipped. The audit caught it by measuring angular
        momentum in a ballistic tumble: 3.1e-02 drift against RK4's
        true 4.2e-07, a factor of 74,000, with the subtree mass stuck
        at the dry 1750 kg while the chassis carried 1959.
        """
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

        self.m_gas = TH.gas_mass()
        self.tank = tank_state(self.m_gas, self.beta,
                               self.world.name)
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

        # Draw this episode's failure. Drawn in reset so it is a
        # property of the SEED: the healthy and robust arms, scored
        # on the same seeds, meet exactly the same breakages.
        self.fault = None
        if r.uniform() < self.fail_p:
            mode = (self.fail_mode if self.fail_mode != "mix"
                    else ("shut" if r.uniform() < 0.5 else "open"))
            self.fault = {
                "nozzle": int(r.integers(0, self.act_dim)),
                "mode": mode,
                "at": int(r.uniform(*self.fail_at) * CTRL_HZ),
            }
        self.fault_active = False
        self._prev_phi = self._potential()
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self) -> np.ndarray:
        R = self.rotmat()
        tk = tank_state(self.m_gas, self.beta, self.world.name)
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
            [self.m_gas / TH.gas_mass(), tk["frac"]],   # 2
            [1.0 - self.t / MAX_STEPS],           # 1
            self.u_act,                           # act_dim
            # The fault channel, OFF by default. One-hot over the
            # nozzles plus a sign for the mode: -1 shut, +1 open, 0
            # healthy. Its width does not depend on `tell`, so the
            # told and untold arms share an observation width and a
            # policy can be loaded into either -- which is what makes
            # the comparison a comparison rather than two networks.
            self._fault_obs(),                    # act_dim + 1
        ]).astype(np.float64)

    def _fault_obs(self) -> np.ndarray:
        """What a perfect fault-detection system would report."""
        v = np.zeros(self.act_dim + 1)
        if self.tell and self.fault is not None and self.fault_active:
            v[self.fault["nozzle"]] = 1.0
            v[-1] = -1.0 if self.fault["mode"] == "shut" else 1.0
        return v

    def step(self, action: np.ndarray):
        mujoco = self._mj
        u_cmd = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        self._hist.append(u_cmd)
        u_delayed = self._hist.pop(0)

        rew = 0.0
        done = False
        for _ in range(SUBSTEPS):
            # Valve lag, then the real-gas chain, then the command.
            #
            # MuJoCo's actuator gear is the DESIGN-POINT thrust, so
            # the command is scaled by how far current conditions
            # have fallen from it. That ratio is not a fudge: it is
            # (G_t c_e) now over (G_t c_e) at a full tank, and on
            # Earth it reaches 0.4 before the tank is empty.
            self.u_act += (u_delayed - self.u_act) * (PHYS_DT / VALVE_TAU)
            self.u_act = np.clip(self.u_act, 0.0, 1.0)
            tk = tank_state(self.m_gas, self.beta, self.world.name)
            self.tank = tk
            # THE FAULT IS APPLIED HERE, to the REALISED control,
            # after the valve lag and after the thrust scaling.
            #
            # It deliberately does not touch `self.u_act`, which is
            # what the policy commanded and what the observation
            # reports. The gap between the two IS the fault, and
            # leaving it visible in the observation would hand the
            # policy a fault detector it has not earned.
            eff = self.u_act * tk["frac"]
            if self.fault is not None and self.t >= self.fault["at"]:
                self.fault_active = True
                k = self.fault["nozzle"]
                eff = eff.copy()
                eff[k] = 0.0 if self.fault["mode"] == "shut" else tk["frac"]
            self.data.ctrl[self.aid] = eff

            # Mass flow is throat flux x throat area, per nozzle. It
            # is NOT thrust/v_e with a constant v_e -- that was the
            # old model, and it understated consumption by a fifth
            # late in the burn when c_e has collapsed.
            # Propellant follows the REALISED opening, not the
            # command. A latched nozzle keeps spending whether or
            # not the policy asks it to -- that continuous drain is
            # half of what makes stuck-open worse than stuck-shut,
            # and billing the command would hide it entirely.
            u_flow = self.u_act
            if self.fault is not None and self.t >= self.fault["at"]:
                u_flow = u_flow.copy()
                k = self.fault["nozzle"]
                u_flow[k] = 0.0 if self.fault["mode"] == "shut" else 1.0
            dm = float(np.sum(u_flow * self.at)) * tk["G"] * PHYS_DT
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
        # NON-NEGATIVE AND SMALL, which lab 1's shape was not.
        #
        # Lab 1 shipped Phi = -2.5*alt + 3*up and reached 89%, but it
        # spent most of training on a hover plateau. Labs 2 and 3
        # found out why: with gamma below 1 an agent occupying a
        # state collects (gamma - 1) * Phi every step, so a NEGATIVE
        # potential pays it to stall and a LARGE one charges it rent
        # for being anywhere good. Phi here tops out at 5, a toll of
        # about 3 per episode against a landing worth about 80.
        alt = 3.0 * (1.0 - min(self.altitude(), 8.0) / 8.0)   # 0 .. 3
        up = 1.0 * (self.up_dot() + 1.0)                      # 0 .. 2
        return alt + up

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
        w = self.omega()
        r = 0.0
        r -= 0.05 * float(w @ w) * dt           # stop tumbling
        r -= 0.004 * float(np.sum(self.u_act)) * dt   # gas costs
        # A CONSTANT COST PER AIRBORNE STEP, replacing lab 1's alive
        # bonus and uprightness bonus. Lab 1 paid +2*up*dt for being
        # upright, which is +62 over six seconds for hovering neatly,
        # and the policy took it. Lab 3 found the general form: if
        # every episode times out, charge for time rather than
        # raising the terminal.
        r -= TIME_COST
        phi = self._potential()
        r += self.gamma * phi - self._prev_phi
        self._prev_phi = phi
        return r

    def _terminal(self) -> float:
        v = self.vel()
        sink = abs(min(v[2], 0.0))
        tilt = self.tilt_deg()
        spin = float(np.linalg.norm(self.omega()))
        if self.outcome == "crash":
            # GRADED BY IMPACT SPEED. A flat penalty scores a 10 m/s
            # dive and a 2.6 m/s arrival alike, so slowing down is
            # not discoverable -- and on a failure the geometry says
            # is unsurvivable, arriving slowly is the ONLY
            # improvement left.
            return -60.0 - 4.0 * min(sink, 8.0)
        if self.outcome == "timeout":
            # THE WORST OUTCOME, and this lab had to flip it back.
            #
            # Lab 4 put the crash at the bottom of the ladder, on
            # the honest argument that destroying the vehicle is
            # worse than running out of clock. Ported here, it
            # taught the policy never to try: crashes fell 97% ->
            # 3% and timeouts rose 3% -> 97%, with landings never
            # above 1% across 4M steps.
            #
            # The reason is that the two tasks differ in what
            # INACTION reaches. On the gate course, doing nothing
            # earns no gates and the gate bonus pulls the policy
            # out of a hover. In a pure landing task, doing nothing
            # IS a hover, and the only route to a landing runs
            # through the risk of a crash. So the rule is not
            # "crash worst" or "timeout worst" -- it is that the
            # outcome reachable by doing nothing must be the worst
            # one available, or a policy that cannot yet do the
            # task will do nothing instead.
            return -130.0
        self.touchdown = dict(sink=sink, tilt=tilt, spin=spin)
        if sink <= SINK_OK and tilt <= TILT_OK and spin <= SPIN_OK:
            self.outcome = "landed_ok"
            return (60.0 + 10.0 * (1.0 - sink / SINK_OK)
                    + 10.0 * (1.0 - tilt / TILT_OK))
        self.outcome = "landed_hard"
        return -25.0

    def hover_action(self) -> np.ndarray:
        """
        The valve opening that holds the vehicle up.

        Asked of the ENV rather than computed in the trainer, because
        it must follow the world: lab 4 left this defaulting to Earth
        and every Mars episode began commanding 2.9x the thrust
        needed to hover, which read as an ordinary failure to learn.
        """
        from ppo import hover_bias
        return hover_bias(self.rig, self.world)

    def landed_ok(self) -> bool:
        return self.outcome == "landed_ok"

    # Named `success` as well, because evaluate.py and the
    # renderers are shared across this topic and all of them
    # ask the env what counts as a win.
    def success(self) -> bool:
        return self.outcome == "landed_ok"


def random_policy(rng: np.random.Generator, env: FailEnv) -> np.ndarray:
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
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    a, _ = ap.parse_known_args()

    if a.budget:
        w = TH.WORLDS[a.world]
        m0 = TH.gas_mass()
        print("=" * 78)
        print(f"  THE TANK — what it can and cannot buy, on {w.name}")
        print("=" * 78)
        print(f"  {'gas kg':>7} {'tank bar':>9} {'T K':>6} {'p_c bar':>8} "
              f"{'c_e':>6} {'Isp':>6} {'T exit':>7} {'thrust':>8}")
        for frac in (1.0, 0.8, 0.6, 0.45, 0.3, 0.18, 0.1, 0.05, 0.02):
            d = tank_state(m0 * frac, BETA_WALL, w.name)
            print(f"  {m0 * frac:>7.1f} {d['p'] / 1e5:>9.0f} {d['T']:>6.0f} "
                  f"{d['pc'] / 1e5:>8.1f} {d['ce']:>6.0f} "
                  f"{d['ce'] / TH.G0:>6.1f} {d['Te']:>7.1f} "
                  f"{d['frac']:>7.0%}")
        imp = TH.total_impulse(world=w)
        print(f"\n  total impulse {imp / 1000:.1f} kN·s along the real "
              f"blowdown; a constant-c_e")
        print(f"  figure would claim {m0 * TH.design_point(w)[1] / 1000:.1f} "
              f"kN·s. Hover: {TH.hover_seconds(w):.1f} s.")
        print(f"  'thrust' is the fraction of design-point thrust a "
              f"fully-open valve gives.")
        return 0

    env = FailEnv(a.rig, a.preset, flat=a.flat, world=a.world)
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
