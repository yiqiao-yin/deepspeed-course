#!/usr/bin/env python3
"""
One landing task, two ways of seeing it: the teacher's and the real one.

    uv run teach_env.py --sense oracle --episodes 10
    uv run teach_env.py --sense onboard --episodes 10
    uv run teach_env.py --gap                 # what the student cannot see

THE TASK IS LAB 3's, DELIBERATELY
----------------------------------
A target is marked on hazard terrain, the car is launched a few
metres off it, and it has to come down on the mark -- except 57%
of the time the mark is a boulder or a scarp face and it has to
put down nearby instead. Reward, terrain, rig and horizon are lab
3's, unchanged, because this lab varies exactly ONE thing and a
re-tuned reward would have made the comparison meaningless.

What lab 3 found, and left unexplained: giving PPO a 5x5 height
patch made the landing WORSE than flying blind. This lab is the
follow-up, and its claim is that the patch was never the problem.

THE THREE WAYS OF SEEING
-------------------------
Every arm gets attitude, body rates, velocity, where the mark is,
tank state and the clock. They differ in ONE channel:

    oracle    the exact height above ground and the exact clearance
              of the lowest corner. Instantaneous, noise-free,
              independent of attitude, correct even where the
              vehicle's own body would block a sensor. PHYSICALLY
              IMPOSSIBLE -- a simulator looks it up. The teacher's.
    onboard   144 rangefinder returns from a belly-mounted depth
              camera: occluded, perspective-distorted, bolted to a
              tumbling body, range-limited and noisy. The same two
              numbers ARE in there, recoverable to about 0.7 m, but
              only after the student learns to read them. Real
              hardware. The student's.
    deaf      neither. A lander that cannot tell how high it is.
              The floor, and it is meant to be bad.

THE STUDENT IS NOT BEING STRAWMANNED
-------------------------------------
Losing the altimeter is not losing a sensor, it is losing an
ANSWER. 144 ranging beams are strictly better hardware than the
single-beam radar altimeter lab 3 handed every arm; what the
student gives up is the pre-computed, exact, always-valid scalar.
The gap between having a measurement and having the quantity is
this lab's whole subject. `--gap` measures both directions of it:
the teacher's two numbers come back at R^2 0.83 from the beams,
and lab 3's 5x5 height patch does not come back at all.

NO ARM GETS A HEIGHT PATCH, AND THAT IS A MEASURED DECISION
------------------------------------------------------------
This lab's first draft made the teacher PPO-with-a-5x5-height-patch.
That is exactly lab 3's `patch` arm, which lab 3 measured at 0% on
target at 4M steps and 8% at 10M, against `blind`'s 56%. The first
teacher run reproduced it: 430k steps, 0% on target, 93% crashes.
A teacher that cannot fly cannot teach, and the arms were re-cut.

The deeper reason is lab 3's divert-budget sweep, where the best
use of PERFECT terrain knowledge turned out to be ignoring it. The
patch is 25 inputs a policy must learn to discount on a task that
does not reward diverting. Privilege is only worth distilling if it
is worth having.

"""


from __future__ import annotations

import argparse
import math

import numpy as np

import car as CAR
import terrain
import rangefinder as RF
import thrusters as TH

CTRL_HZ = 100.0
PHYS_DT = 0.002
SUBSTEPS = int(round(1.0 / CTRL_HZ / PHYS_DT))
MAX_SECONDS = 6.0

# The discount, and it is a PROPERTY OF THE TASK here rather than a
# free knob. The episode is 600 steps; at gamma = 0.995 the horizon
# 1 / (1 - gamma) is 200 steps, so the end of the episode is
# discounted by 0.995 ** 600 = 0.05 and a terminal penalty is almost
# invisible from the start. This lab measured the consequence: with
# the timeout penalty at -90 -- worse than a crash -- the policy
# STILL learned to fly away and run out the clock, because -90
# arriving at step 600 is worth -4.4 at step 0 while a crash at step
# 100 is worth -36. At 0.999 the same horizon is 1000 steps,
# 0.999 ** 600 = 0.55, and the outcome is something the policy can
# actually see.
GAMMA = 0.999

# A CONSTANT COST PER AIRBORNE STEP, and it is what finally killed
# the flee attractor. Three fixes before it did not:
#
#   the shaping scale    Phi was 31 and the (gamma - 1) * Phi toll
#                        outweighed the prize. Fixed; still fled.
#   the timeout penalty  raised to -90, WORSE than a crash. Still
#                        fled, on gentle terrain as well as hard.
#   the discount         0.995 -> 0.999. Still fled.
#
# The reason none of them worked is that the choice was never
# between the SIZES of the terminals, it was between their TIMING.
# At gamma = 0.999 a crash at step 100 is worth -60 * 0.905 = -54.3
# and a timeout at step 600 is worth -90 * 0.549 = -49.4, so the
# larger penalty is the cheaper one purely because it arrives later.
# The policy was not avoiding a bad outcome, it was postponing one,
# and every episode ran the clock out.
#
# A per-step cost makes postponement itself expensive, so the
# ordering stops depending on when the episode happens to end. This
# is the standard remedy for the procrastination failure mode and
# the thing to reach for whenever every episode times out.
# 0.15, and the value matters more than it looks.
#
# At 0.15 the flee attractor dies immediately: timeouts fall to 1%
# and the miss distance drops 11.3 m -> 3.5 m, inside the scoring
# circle. The policy then crashed on 98% of episodes, which looked
# like the time cost buying a DIVE -- so it was lowered to 0.05.
# That was the wrong diagnosis and the measurement says so: at 0.05
# the miss started climbing again (12 m -> 20 m by 250k steps) and
# the timeouts came back. The cost was holding the policy together,
# not pushing it into the ground.
#
# The dive had a different cause, fixed separately: a FLAT crash
# penalty, which scored a 10 m/s arrival and a 2.6 m/s arrival
# identically and so made slowing down undiscoverable. Grading that
# penalty by impact speed is what gives the approach a gradient.
# Two symptoms, two causes, and swapping one fix for the other
# would have lost both.
TIME_COST = 0.15
MAX_STEPS = int(MAX_SECONDS * CTRL_HZ)

VALVE_TAU = 0.025
DELAY_S = 0.010
BETA_WALL = 0.30

SINK_OK = 2.5                 # m/s
TILT_OK = 25.0                # degrees
SPIN_OK = 1.2                 # rad/s
ON_TARGET_M = 4.0             # 96% of targets have survivable ground inside this

# The three arms. `oracle` is the teacher's and is impossible;
# `onboard` is the student's and is hardware; `deaf` is the control
# that says how much of the score is available with no terrain
# knowledge at all. Lab 3's names (blind/slope/patch) are gone on
# purpose -- `patch` and `oracle` are the same numbers but a
# different CLAIM, and reusing the word would invite the reader to
# treat lab 3's measurement as this lab's baseline.
SENSES = ("deaf", "oracle", "onboard")
PATCH_N = 5                   # oracle: 5 x 5 samples
PATCH_SPAN = 8.0              # oracle: metres across


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


def survivable(surf, x: float, y: float, max_slope: float = 12.0,
               clear: float = 2.6) -> bool:
    """Could the car be put down here and stay up? Geometry only."""
    if surf.slope_deg_at(x, y) > max_slope:
        return False
    for r in surf.rocks:
        if math.hypot(x - r.x, y - r.y) < clear:
            return False
    return True


class TeachEnv:
    """One launch, one marked spot, one landing attempt."""

    def __init__(self, rig_name: str = "default", preset: str = "badlands",
                 seed: int = 0, sense: str = "oracle", beta: float = BETA_WALL,
                 integrator: str = "RK4", world: str = "earth",
                 flat: bool = False, allow_flat: bool = False,
                 gamma: float = GAMMA, **rig_kw) -> None:
        import mujoco

        if sense not in SENSES:
            raise ValueError(f"sense must be one of {SENSES}, got {sense!r}")
        self._mj = mujoco
        self.rig_name = rig_name
        self.preset = "flat" if flat else preset
        self.sense = sense
        # The shaping term is gamma * Phi(s') - Phi(s), so the env's
        # gamma MUST equal the trainer's or the shaping stops being
        # potential-based and starts changing the optimal policy.
        # This was a literal 0.995 in the reward while the trainer
        # took --gamma as a flag: two numbers that had to agree with
        # nothing making them.
        self.gamma = float(gamma)
        self.beta = beta
        self.world = TH.WORLDS[world]
        self.rng = np.random.default_rng(seed)

        self.model, self.rig, self.surf = CAR.build(
            rig_name, preset, seed=seed, flat=flat, start_z=8.0,
            integrator=integrator, world=self.world, **rig_kw)
        if self.surf is None and not allow_flat:
            # Flat ground has no survivable-spot decision in it, so a
            # flat TRAINING run would measure nothing. verify.py passes
            # allow_flat because the conservation audit is about the
            # car and its thrusters, not about the task.
            raise ValueError("the target task needs terrain; --flat is "
                             "meaningless here (verify.py passes "
                             "allow_flat=True for the physics audit)")
        self.data = mujoco.MjData(self.model)
        self.aid = CAR.resolve(self.model, self.rig)
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

        # The rangefinder is built for EVERY arm, not only `onboard`.
        # `distill.py` flies a student and asks the teacher what it
        # would have done, and `render_*.py` draws the beams for a
        # teacher rollout; both need the sensor present on an arm
        # that does not feed it to the policy. It costs nothing
        # until `scan()` is called.
        self.rf = RF.RangeFinder(self.model, mujoco, hull_body=self.chassis)

        self.reset(seed)
        self.obs_dim = len(self._obs())
        self.max_steps = MAX_STEPS

    # -- state -------------------------------------------------------
    def pos(self):
        return self.data.qpos[:3].copy()

    def rotmat(self):
        return _quat_to_mat(self.data.qpos[3:7].copy())

    def vel(self):
        return self.data.qvel[:3].copy()

    def omega(self):
        return self.data.qvel[3:6].copy()

    def up_dot(self) -> float:
        return float(self.rotmat()[2, 2])

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.up_dot(), -1, 1))))

    def altitude(self) -> float:
        p = self.pos()
        return float(p[2] - self._h(p[0], p[1]))

    def clearance(self) -> float:
        R, p = self.rotmat(), self.pos()
        lo = math.inf
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    w = p + R @ np.array([sx * 2.05, sy * 0.94, sz * 0.30])
                    lo = min(lo, w[2] - self._h(w[0], w[1]))
        return float(lo)

    def ground_grad(self, d: float = 1.0) -> np.ndarray:
        """
        Local ground gradient under the car, as (dz/dx, dz/dy).

        The `slope` observation and the HUD both call this, so there
        is one definition of "ground slope" rather than two that can
        drift apart.
        """
        p = self.pos()
        return np.array([(self._h(p[0] + d, p[1]) - self._h(p[0] - d, p[1]))
                         / (2 * d),
                         (self._h(p[0], p[1] + d) - self._h(p[0], p[1] - d))
                         / (2 * d)])

    def _h(self, x: float, y: float) -> float:
        """Ground height. Flat ground is a plane at zero, which
        only verify.py ever asks for -- training refuses it."""
        return self.surf.height_at(x, y) if self.surf else 0.0

    def miss(self) -> float:
        """Horizontal distance from the marked spot."""
        p = self.pos()
        return float(math.hypot(p[0] - self.target[0], p[1] - self.target[1]))

    def on_target(self) -> bool:
        return self.miss() <= ON_TARGET_M

    # -- sensing -----------------------------------------------------
    def oracle_state(self) -> np.ndarray:
        """
        The teacher's terrain channel: two exact numbers.

        Kept as its own public method rather than folded into
        `_obs` because `distill.py` needs to ask the environment
        for the TEACHER's view while the STUDENT is the one
        flying -- that relabelling is the whole of DAgger.
        """
        return np.array([self.altitude() / 8.0, self.clearance() / 8.0])

    def onboard_scan(self, clean: bool = False) -> np.ndarray:
        """The student's terrain channel: one rangefinder frame."""
        return self.rf.scan(self.data, self.rng, clean=clean)

    def _terrain_obs(self) -> np.ndarray:
        """
        What this arm is allowed to know about the ground.

        oracle    the exact height above ground and the exact
                  clearance of the lowest corner, read straight out
                  of the surface. Instantaneous, noise-free,
                  unaffected by attitude, and correct even where the
                  vehicle's own body would block a sensor. Nothing
                  measures this; a simulator looks it up.
        onboard   144 rangefinder returns, and nothing else. The
                  same two numbers are IN there -- a probe recovers
                  altitude at R^2 0.83 and clearance at 0.84 from
                  the beams plus attitude -- but only to about 0.7 m,
                  and only after the student learns to read them.
        deaf      neither. A lander that cannot tell how high it is.

        WHY THE STUDENT IS NOT BEING STRAWMANNED BY LOSING THE
        ALTIMETER. It is not losing the sensor, it is losing the
        ANSWER. 144 ranging beams are strictly better hardware than
        the single-beam radar altimeter lab 3 gave every arm; what
        the student gives up is a pre-computed, exact, always-valid
        scalar. That gap -- between having the measurement and
        having the quantity -- is the entire subject of this lab.

        WHY THERE IS NO HEIGHT PATCH IN ANY ARM. Lab 3 measured one:
        a 5x5 grid over 8 m, added to this same task, took PPO from
        56% to 0% at 4M steps. It is 25 extra inputs the policy must
        learn to ignore, on a task whose own divert-budget sweep
        showed the best use of perfect terrain knowledge is not to
        use it. Handing that to the teacher would have produced a
        teacher that cannot fly, and this lab's first draft did
        exactly that before the arms were re-cut.
        """
        if self.sense == "deaf":
            return np.zeros(0)
        if self.sense == "oracle":
            return self.oracle_state()
        return self.onboard_scan()

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

        # A target somewhere on the hazard field, and a launch a few
        # metres off it. Starting ON the target would make the task a
        # pure descent; starting far away would make it a fuel
        # problem. Four metres is the distance at which 96% of
        # targets have survivable ground in reach.
        ang = r.uniform(0, 2 * math.pi)
        rad = r.uniform(12.0, 22.0)
        self.target = (rad * math.cos(ang), rad * math.sin(ang))
        dx, dy = r.uniform(-3.5, 3.5, 2)
        x0, y0 = self.target[0] + dx, self.target[1] + dy
        gz = self._h(x0, y0)
        self.data.qpos[:3] = [x0, y0, gz + 5.0 + 2.0 * r.uniform(-1, 1)]

        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang2 = math.radians(r.uniform(0, 40))
        self.data.qpos[3:7] = [math.cos(ang2 / 2),
                               *(math.sin(ang2 / 2) * axis)]
        self.data.qvel[:3] = [r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5),
                              r.uniform(0.5, 4.0)]
        sp = r.normal(size=3) * np.array([1.0, 0.45, 0.35])
        self.data.qvel[3:6] = sp / max(np.linalg.norm(sp), 1e-9) \
            * r.uniform(0.5, 2.5)

        self.m_gas = TH.gas_mass()
        self.tank = tank_state(self.m_gas, self.beta, self.world.name)
        self._apply_mass()
        self.u_act = np.zeros(self.act_dim)
        self._hist = [np.zeros(self.act_dim) for _ in range(self._delay_n)]
        self.t = 0
        self.gas_used = 0.0
        self.outcome = ""
        self._settled = 0
        self.touchdown = None
        mujoco.mj_forward(self.model, self.data)
        self._prev_phi = self._potential()
        return self._obs()

    def _obs(self, terrain_channel: np.ndarray | None = None) -> np.ndarray:
        """
        The observation vector, with a pluggable terrain channel.

        `terrain_channel=None` means "whatever this arm is set to",
        which is what `step()` and `reset()` use. Passing one
        explicitly is what `obs_for()` does, and the reason the
        parameter exists at all: DAgger has to ask, about a state
        the STUDENT drove into, what the TEACHER would have seen.
        Without that the relabelling cannot be done and the method
        degenerates into behaviour cloning -- silently, and with
        every array still the right shape.
        """
        R = self.rotmat()
        p = self.pos()
        tk = self.tank
        tc = self._terrain_obs() if terrain_channel is None else terrain_channel
        return np.concatenate([
            R[:, 0], R[:, 1],                        # 6 attitude
            [R[2, 2]],                               # 1 uprightness
            self.omega() / 5.0,                      # 3
            self.vel() / 10.0,                       # 3
            # THE TERRAIN CHANNEL SITS HERE, AT INDEX 12, and the
            # position is load-bearing. Lab 3 put altitude and
            # clearance exactly here and gave them to every arm.
            # This lab makes them the teacher's privilege, so with
            # the channel in the same place the `oracle` observation
            # is BYTE-IDENTICAL to lab 3's `blind` and lab 3's 56%
            # must reproduce. That is the control on the whole
            # comparison.
            #
            # It was first placed after the clock instead -- the
            # same 41 numbers, a different order. That version ran,
            # trained, and scored 4% and 2% on two seeds against
            # lab 3's 56% and 56%, with the SAME learning-curve
            # shape a million steps late and still climbing at the
            # cutoff. A permutation of the inputs is a different
            # random initialisation, and on a task whose seed spread
            # lab 3 measured at 40% against 10% that is enough to
            # look like a broken environment. Nothing errored and
            # every array was the right shape.
            tc,                                      # 0, 2 or 144
            # Where the mark is, in the car's own frame. Without this
            # the task is not Markov.
            (R.T @ np.array([self.target[0] - p[0],
                             self.target[1] - p[1], 0.0]))[:2] / 8.0,
            [min(self.miss(), 12.0) / 12.0],
            [self.m_gas / TH.gas_mass(), tk["frac"]],
            [1.0 - self.t / MAX_STEPS],
            self.u_act,
        ]).astype(np.float64)

    def obs_for(self, sense: str) -> np.ndarray:
        """
        What this same state looks like under a DIFFERENT arm.

        The whole of teacher-student learning is here. The student
        flies, so the trajectory distribution is the student's; the
        labels come from the teacher, so they have to be computed
        from the teacher's view of states the teacher never chose.
        `obs_for("oracle")` is that view.

        Note it does NOT change `self.sense` -- calling it has no
        effect on what `step()` returns next. A version that mutated
        the arm would make the dataset depend on the order the two
        views were requested in, which is the kind of bug that
        trains fine and is wrong.
        """
        if sense not in SENSES:
            raise ValueError(f"sense must be one of {SENSES}, got {sense!r}")
        if sense == "deaf":
            return self._obs(np.zeros(0))
        if sense == "oracle":
            return self._obs(self.oracle_state())
        return self._obs(self.onboard_scan())

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

            n_wheel, hull = self._touching()
            if hull and self.altitude() < 2.0:
                self.outcome = "crash"
                done = True
                break
            # TWO wheels, not three.
            #
            # Labs 1 and 2 landed on a flattened pad where a settled
            # car always reports four contacts. On boulders and
            # slopes it rests on two or three, and requiring three
            # made 72% of episodes "time out" while sitting
            # stationary and upright on the ground. The gate is meant
            # to mean "down and settled", and the wheel count was a
            # proxy for that which stopped being true when the ground
            # stopped being flat. Hull contact is still a crash, so
            # this cannot pass a car resting on its body.
            if n_wheel >= 2 and float(np.linalg.norm(self.vel())) < 1.2:
                self._settled += 1
                if self._settled >= 5:
                    self.outcome = "landed"
                    done = True
                    break
            else:
                self._settled = 0

        self.t += 1
        rew = self._reward()
        if not done and self.t >= MAX_STEPS:
            self.outcome = "timeout"
            done = True
        if done:
            rew += self._terminal()
        return self._obs(), rew, done, {}

    # -- reward ------------------------------------------------------
    def _potential(self) -> float:
        """
        Non-negative AND SMALL, and the second half is the one this
        lab had to learn.

        Potential-based shaping adds gamma*Phi(s') - Phi(s). That
        leaves the optimal policy unchanged in the undiscounted
        case, but at gamma < 1 an agent occupying a state collects
        (gamma - 1) * Phi every step it stays there. The sign and
        the SCALE of Phi are therefore both load-bearing:

          Phi NEGATIVE  -> the leak is a subsidy, and the agent is
                           paid to do nothing. Lab 2 shipped
                           Phi ~ -15, worth +60 per episode, and the
                           policy stopped attempting the manoeuvre.
          Phi LARGE     -> the leak is a toll on being in GOOD
                           states, and the agent runs away from
                           them. This lab shipped Phi_max ~ 31,
                           a toll of 124 per episode against a
                           terminal prize of ~100, and the policy
                           learned to flee: crashes fell 95% -> 0%
                           while the miss distance climbed from
                           12 m to 86 m and every episode timed out.

        So Phi is capped near 6, making the toll about 24 per
        episode against the same ~100 prize. The shaping still
        points the right way and no longer outweighs the thing it
        is supposed to be pointing at.
        """
        # No saturation inside the arena. The first version clipped
        # the distance term at 12 m, so beyond 12 m the shaping was
        # perfectly flat -- the agent fled to 86 m across ground
        # that gave it no reason to come back.
        near = 3.0 * max(0.0, 1.0 - self.miss() / 40.0)      # 0 .. 3
        alt = 2.0 * (1.0 - min(self.altitude(), 10.0) / 10.0)  # 0 .. 2
        up = 0.5 * (self.up_dot() + 1.0)                     # 0 .. 1
        return near + alt + up

    def _reward(self) -> float:
        dt = SUBSTEPS * PHYS_DT
        w = self.omega()
        r = 0.0
        r -= 0.05 * float(w @ w) * dt
        r -= 0.004 * float(np.sum(self.u_act)) * dt
        # Finishing is worth something and dithering is not.
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
        miss = self.miss()

        if self.outcome == "crash":
            # GRADED BY IMPACT SPEED, not a flat number. A flat -60
            # scores a 10 m/s dive and a 2.6 m/s arrival that just
            # missed the gate identically, so there is no gradient
            # from "crashing" towards "landing" -- the policy has no
            # way to discover that slowing down is progress. Same
            # reasoning as the graded on-target reward below.
            return -40.0 - 4.0 * min(sink, 8.0)
        if self.outcome == "timeout":
            # The WORST outcome, below even a crash, and no upright
            # bonus. A lander that never lands has failed harder
            # than one that tried and broke, and the first version
            # -- timeout -40 plus 6 * up_dot, so -34 for floating
            # neatly -- made running away strictly safer than
            # approaching. 100% of episodes ended here.
            return -90.0

        self.touchdown = dict(sink=sink, tilt=tilt, spin=spin, miss=miss)
        gentle = sink <= SINK_OK and tilt <= TILT_OK and spin <= SPIN_OK
        if not gentle:
            self.outcome = "landed_hard"
            return -5.0
        if miss <= ON_TARGET_M:
            self.outcome = "on_target"
            # Graded inside the circle so there is a gradient to the
            # centre, not a step the policy can stop improving on.
            return 70.0 + 20.0 * (1.0 - miss / ON_TARGET_M) \
                + 10.0 * (1.0 - tilt / TILT_OK)
        self.outcome = "landed_off"
        # Survived, missed the mark. Positive -- staying alive IS
        # worth something and the alternative is a crash -- but far
        # below an on-target landing, and decaying with the miss.
        return max(5.0, 25.0 - 3.0 * (miss - ON_TARGET_M))

    def success(self) -> bool:
        return self.outcome == "on_target"


def random_policy(rng, env) -> np.ndarray:
    return rng.uniform(0.0, 0.6, env.act_dim)



def _information_gap(rig: str, preset: str, n: int) -> int:
    """
    Can the student recover what the teacher is simply told?

    The teacher's privilege is two exact numbers: height above
    ground and lowest-corner clearance. The student has 144
    rangefinder beams. Whether those beams DETERMINE the two
    numbers decides whether this lab is a perception problem with
    an answer or a guessing game, so it is measured before anything
    is trained.

    Two probes and an aliasing test, and the pairing is the point:

    RECOVERABLE   scan + attitude -> the teacher's two numbers.
                  Expect high, because a rangefinder is what you
                  would build to measure exactly this. A large
                  train/test GAP would mean memorisation instead,
                  so both are printed -- a single test number
                  cannot distinguish "the information is absent"
                  from "the probe was undertrained", and this lab's
                  first draft was nearly published on that
                  confusion.

    NOT RECOVERABLE  scan + attitude -> a 5x5 world-aligned height
                  patch, the privilege lab 3 gave its `patch` arm.
                  Expect near zero WITH a near-perfect training
                  fit, which is what memorisation looks like and
                  what "the information is genuinely gone" means.
                  No arm in this lab gets that patch, and this is
                  the measurement saying why it would not help.

    ALIASING      how often two states the teacher distinguishes
                  collapse to the same scan.

    All on CLEAN scans, noise and dropout off. That is deliberately
    generous to the student: it isolates sensor GEOMETRY from the
    error model, so a gap measured here is a floor.
    """
    env = TeachEnv(rig, preset, sense="oracle", seed=11)
    rng = np.random.default_rng(11)
    scans, oracles, patches, props = [], [], [], []
    ep = 0
    # ONE sample every STRIDE steps, across many episodes. The first
    # version took 60 consecutive steps per episode, which at 100 Hz
    # is 0.6 s of flight: every sample was a near-duplicate of the
    # one before it, so "nearest neighbour" found the adjacent
    # timestep in BOTH spaces and the aliasing ratio came out at
    # exactly 1.00. It read like a clean null. It was a sampling
    # artifact, and the tell was that the teacher-space distances
    # were four thousandths of a unit.
    STRIDE = 20
    offs = np.linspace(-PATCH_SPAN / 2, PATCH_SPAN / 2, PATCH_N)
    while len(scans) < n and ep < 6000:
        env.reset(1000 + ep)
        ep += 1
        for k in range(MAX_STEPS):
            _o, _r, done, _ = env.step(rng.uniform(0.0, 0.55, env.act_dim))
            if k % STRIDE == 0 and env.altitude() > 0.5:
                pp = env.pos()
                here = env._h(pp[0], pp[1])
                scans.append(env.onboard_scan(clean=True))
                oracles.append(env.oracle_state())
                patches.append(np.array(
                    [(env._h(pp[0] + dx, pp[1] + dy) - here)
                     for dy in offs for dx in offs]) / 2.0)
                props.append(np.concatenate([
                    env.rotmat()[:, 0], env.rotmat()[:, 1], [env.up_dot()],
                    env.omega() / 5.0, env.vel() / 10.0]))
            if done or len(scans) >= n:
                break

    S = np.asarray(scans)
    X = np.concatenate([S, np.asarray(props)], axis=1)
    targets = {"the teacher's 2 numbers (alt, clearance)":
               np.asarray(oracles),
               "a 5x5 world-aligned height patch (lab 3's `patch`)":
               np.asarray(patches)}

    print("=" * 78)
    print("  THE INFORMATION GAP — what the rangefinder can and cannot give")
    print("=" * 78)
    print(f"  {len(X):,} states from {ep} episodes, clean scans "
          f"(no noise, no dropout)")
    print(f"  probe input: {S.shape[1]} beams + {X.shape[1] - S.shape[1]} "
          f"attitude/rate/velocity")

    perm = rng.permutation(len(X))
    X = X[perm]
    m = len(X) // 2
    mu, sd = X[:m].mean(0), X[:m].std(0) + 1e-9

    try:
        import torch
        import torch.nn as nn
        has_torch = True
    except ImportError:
        has_torch = False
        print("\n  (torch not installed — probes skipped)")

    for label, Yall in targets.items():
        Y = Yall[perm]
        Ytr, Yte = Y[:m], Y[m:]
        print(f"\n  TARGET: {label}")
        if not has_torch:
            continue
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(),
                            nn.Linear(256, 256), nn.SiLU(),
                            nn.Linear(256, Y.shape[1]))
        xt = torch.as_tensor((X[:m] - mu) / sd, dtype=torch.float32)
        yt = torch.as_tensor(Ytr, dtype=torch.float32)
        xv = torch.as_tensor((X[m:] - mu) / sd, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(3000):
            j = torch.randint(0, len(xt), (256,))
            opt.zero_grad()
            ((net(xt[j]) - yt[j]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ptr, pte = net(xt).numpy(), net(xv).numpy()

        def r2(pred, T):
            return 1.0 - float(((T - pred) ** 2).sum()) / float(
                ((T - T.mean(0)) ** 2).sum())

        a, b = r2(ptr, Ytr), r2(pte, Yte)
        print(f"    MLP 256x256     train R^2 {a:+.3f}    test R^2 {b:+.3f}")
        if b > 0.6:
            print(f"    -> RECOVERABLE. The student can estimate this, and "
                  f"the lab has something to teach.")
        elif a - b > 0.5:
            print(f"    -> NOT RECOVERABLE. The probe FITS the training set "
                  f"({a:+.3f}) and does not\n       generalise — "
                  f"memorisation, which is what an absent signal looks "
                  f"like.")
        else:
            print(f"    -> weak, and not obviously memorisation either.")

    k = min(600, len(S))
    Ss = S[perm][:k]
    Os = np.asarray(oracles)[perm][:k]
    Ss = (Ss - Ss.mean(0)) / (Ss.std(0) + 1e-9)
    ds = ((Ss[:, None, :] - Ss[None, :, :]) ** 2).sum(-1)
    dy = ((Os[:, None, :] - Os[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(ds, np.inf)
    np.fill_diagonal(dy, np.inf)
    at_nn = np.sqrt(dy[np.arange(k), ds.argmin(1)])
    own_nn = np.sqrt(dy.min(1))
    ratio = float(np.median(at_nn / np.maximum(own_nn, 1e-9)))
    print(f"\n  ALIASING         nearest neighbour in scan space, {k} states")
    print(f"    teacher distance to the scan-nearest state    "
          f"{np.median(at_nn):.4f}  (median)")
    print(f"    teacher distance to the teacher-nearest state "
          f"{np.median(own_nn):.4f}")
    print(f"    ratio {ratio:.2f}x")

    print("\n" + "=" * 78)
    print("  The student is not solving a noisier version of the teacher's")
    print("  problem. It is solving a different one — but one whose answer")
    print("  is in its sensor, which is what makes distillation worth a try.")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--sense", default="oracle", choices=SENSES)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--decidable", action="store_true")
    ap.add_argument("--gap", action="store_true",
                    help="measure what the student's sensor cannot recover")
    ap.add_argument("--gap-samples", type=int, default=4000)
    a, _ = ap.parse_known_args()

    if a.gap:
        return _information_gap(a.rig, a.preset, a.gap_samples)

    if a.decidable:
        print("=" * 78)
        print("  IS THERE A DECISION IN THIS TASK?")
        print("=" * 78)
        for preset in ("rough", "hazard", "badlands"):
            s = terrain.generate(0, preset)
            rng = np.random.default_rng(3)
            tgt = reach = {r: 0 for r in (0, 2, 4, 6)}
            tgt_ok = n = 0
            reach = {0: 0, 2: 0, 4: 0, 6: 0}
            for _ in range(600):
                ang = rng.uniform(0, 2 * math.pi)
                rr = rng.uniform(12, 24)
                tx, ty = rr * math.cos(ang), rr * math.sin(ang)
                n += 1
                tgt_ok += survivable(s, tx, ty)
                for R in (0, 2, 4, 6):
                    hit = any(survivable(s, tx + R * math.cos(t),
                                         ty + R * math.sin(t))
                              for t in np.linspace(0, 2 * math.pi,
                                                   max(1, R * 8)))
                    if hit:
                        reach[R] += 1
            print(f"\n  {preset}:")
            print(f"    target itself survivable   {tgt_ok / n:>5.0%}")
            for R in (0, 2, 4, 6):
                print(f"    good ground within {R} m    {reach[R] / n:>5.0%}")
        print("\n  A target that is usually bad with good ground usually close")
        print("  is what makes this a choice. On `rough` it is not: per-spot")
        print("  success there varies by sd 0.13 against a 0.14 noise floor,")
        print("  so every spot is as good as every other. See the docstring.")
        return 0

    env = TeachEnv(a.rig, a.preset, sense=a.sense)
    print("=" * 78)
    print(f"  TEACH ENV — rig {a.rig}, terrain {a.preset}, sense {a.sense}")
    print("=" * 78)
    print(f"  obs {env.obs_dim}   act {env.act_dim}   "
          f"{CTRL_HZ:.0f} Hz, max {MAX_SECONDS:.0f} s, "
          f"on-target radius {ON_TARGET_M:.0f} m")

    rng = np.random.default_rng(0)
    tally: dict[str, int] = {}
    miss = []
    for ep in range(a.episodes):
        env.reset(ep)
        while True:
            _o, _r, d, _ = env.step(random_policy(rng, env))
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        miss.append(env.miss())
    print(f"\n  {a.episodes} episodes under a RANDOM policy:")
    for k in sorted(tally):
        print(f"    {k:<14} {tally[k]:>3}  ({tally[k] / a.episodes:.0%})")
    print(f"    mean miss    {np.mean(miss):.1f} m")
    print("\n  A random policy should essentially never land on the mark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
