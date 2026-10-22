#!/usr/bin/env python3
"""
Fly a cold-gas car through a gate course on Mars, then land it.

    uv run flight_env.py --budget              # can the tank afford it?
    uv run flight_env.py --world mars          # random-policy sanity run
    uv run flight_env.py --world earth --episodes 20

THE COURSE
----------
Three gates on 12-18 m legs, each 4-10 m above the ground under it,
with the heading bending by up to 0.6 rad per leg. About 45 m of
flying inside a 12 second clock, and then the car has to come down
on its wheels. Passing a gate means coming within 4 m of its centre
in three dimensions, in order; gates cannot be taken out of
sequence and each counts once.

WHY THIS IS A MARS LAB AND NOT AN EARTH ONE
--------------------------------------------
The same vehicle, the same tank and the same course:

    world    g       thrust    L/W     hover    margin
    earth    9.81    1.000     2.40     6.9 s    0.58x
    mars     3.72    1.086     6.87    20.4 s    1.70x

Mars is not merely easier. The margin is the tank's hover endurance
against the episode clock, and on Earth it is BELOW ONE -- the
vehicle cannot hold itself up for the length of the course even at
perfect efficiency. The course is unaffordable there as arithmetic,
before any policy is trained.

Two things move between the worlds and only one of them is gravity.
Weight falls 62%, and thrust RISES 8.6%, because a nozzle expanding
against 610 Pa of Martian atmosphere pushes harder than the same
hardware against 101 kPa. The ratio that decides the task is
lift-to-weight, and it moves 2.87x.
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
# 9 s, and the number is MEASURED rather than chosen.
#
# The hover endurance is 20.4 s on Mars, so a 12 s clock looked
# comfortable. Real flying is not hovering: accelerating and braking
# at each gate, and the thrust fade as the tank cools, together cost
# about 1.65x the hover rate. At 12 s the scheduled controller
# finished the tank -- 290 kg of 297 -- and still timed out. At 9 s
# the course is affordable with a reserve, which is the condition
# for the task to be about flying rather than about running dry.
MAX_SECONDS = 9.0

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
# 0.10, scaled to this lab's longer episode. Lab 3 used 0.15 over
# 600 steps; this course runs 900, so the same rate bleeds 1.5x as
# much and made ending the episode early worth more than flying it.
TIME_COST = 0.10

# PAID THE INSTANT A GATE IS CROSSED, not at the end of the episode.
#
# Gate credit started in the terminal, and that was the single
# biggest reason the policy would not fly. A gate flown at step 150
# was paid at step 900: discounted by 0.999^750 to 47% of its
# value, with 750 steps of unrelated reward in between for the
# advantage estimator to spread it across. Flying a leg cost 15 in
# time cost and returned about 20 of heavily delayed credit, so the
# gradient pointing at the course was thin enough that the policy
# converged to hovering from four different reward designs and two
# action spaces.
#
# Paying at capture makes the signal immediate and local. It stays
# strictly ONE-OFF -- `gate_i` only increases and each gate
# increments it exactly once -- so it cannot become the repeatable
# bonus that taught topic 7's robot to stand on the box collecting
# it instead of crossing it.
GATE_BONUS = 40.0
MAX_STEPS = int(MAX_SECONDS * CTRL_HZ)

VALVE_TAU = 0.025
DELAY_S = 0.010
BETA_WALL = 0.30

SINK_OK = 2.5                 # m/s
TILT_OK = 25.0                # degrees
SPIN_OK = 1.2                 # rad/s
GATE_R = 4.0                  # m, how close counts as flying the gate
N_GATES = 3                   # gates before the landing
# The course is ~45 m of flying. On Mars that is affordable and on
# Earth it is not, which is the whole point -- see the README.

# No perception arms here. Lab 3 measured terrain input and found it
# cost more than it bought; this lab varies the WORLD, not the
# observation, so the policy gets the same proprioceptive set either
# way and nothing about the ground beyond its own altimeter.


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


class WrenchAllocator:
    """
    Turn a demanded body wrench into valve openings, cheaply.

    The SAME objective classical.py solves -- weighted tracking,
    ridge, and a linear gas cost -- at a fraction of the iterations,
    because the training loop runs it millions of times.

    The gas term is not an optimisation nicety, it is the whole
    difference between a working allocator and a wasteful one. A
    plain least-squares fit tracks the demanded wrench perfectly
    well by opening opposing nozzles against each other: the net
    force is right and the propellant is gone. Dropped from the
    first version here, it cost 295 kg of 297 against the exact
    solver's 150 kg for identical commands, and turned a 71%
    controller into a 42% one.
    """

    # Newton-metres and newtons are not interchangeable units, so
    # the residual is weighted before it is squared.
    W_DIAG = np.array([0.5, 0.5, 1.0, 3.0, 3.0, 3.0]) / \
        np.array([5e4, 5e4, 5e4, 1.5e4, 1.5e4, 1.5e4])
    RIDGE = 2e-3
    GAS_COST = 4e-3

    def __init__(self, B: np.ndarray, iters: int = 30) -> None:
        self.B = B
        self.iters = iters
        self.W = np.diag(self.W_DIAG)
        self.WB = self.W @ B
        H = 2.0 * (self.WB.T @ self.WB)
        self.L = float(np.linalg.eigvalsh(H).max()) + 2.0 * self.RIDGE
        self.u = np.zeros(B.shape[1])

    def reset(self) -> None:
        self.u[:] = 0.0

    def solve(self, w: np.ndarray, avail: float = 1.0) -> np.ndarray:
        a = max(avail, 1e-3)
        Wb = self.W @ w
        x = self.u.copy()
        y = x.copy()
        t = 1.0
        L = (a * a) * self.L
        for _ in range(self.iters):
            g = (2.0 * a * self.WB.T @ (a * (self.WB @ y) - Wb)
                 + 2.0 * self.RIDGE * y + self.GAS_COST)
            xn = np.clip(y - g / L, 0.0, 1.0)
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            # O'Donoghue-Candes restart: if the momentum points
            # uphill, drop it.
            if float((y - xn) @ (xn - x)) > 0.0:
                tn, yn = 1.0, xn
            else:
                yn = xn + ((t - 1.0) / tn) * (xn - x)
            x, y, t = xn, yn, tn
        self.u = x
        return x


def survivable(surf, x: float, y: float, max_slope: float = 12.0,
               clear: float = 2.6) -> bool:
    """Could the car be put down here and stay up? Geometry only."""
    if surf.slope_deg_at(x, y) > max_slope:
        return False
    for r in surf.rocks:
        if math.hypot(x - r.x, y - r.y) < clear:
            return False
    return True


class FlightEnv:
    """One launch, three gates, one landing."""

    def __init__(self, rig_name: str = "dense", preset: str = "rough",
                 seed: int = 0, beta: float = BETA_WALL,
                 # 4-8 m legs: about 18 m of course. Sized to what
                 # the vehicle and the learner can actually do --
                 # see the README's course-length sweep, where the
                 # scheduled controller holds 3/3 gates well past
                 # this and PPO does not.
                 leg: tuple[float, float] = (4.0, 8.0),
                 gate_h: tuple[float, float] = (3.0, 8.0),
                 seconds: float = MAX_SECONDS,
                 n_gates: int = N_GATES,
                 action: str = "residual",
                 integrator: str = "RK4", world: str = "earth",
                 flat: bool = False, allow_flat: bool = False,
                 gamma: float = GAMMA, **rig_kw) -> None:
        import mujoco

        self._mj = mujoco
        self.rig_name = rig_name
        self.preset = "flat" if flat else preset
        # The shaping term is gamma * Phi(s') - Phi(s), so the env's
        # gamma MUST equal the trainer's or the shaping stops being
        # potential-based and starts changing the optimal policy.
        # This was a literal 0.995 in the reward while the trainer
        # took --gamma as a flag: two numbers that had to agree with
        # nothing making them.
        self.gamma = float(gamma)
        self.beta = beta
        self.leg = leg
        self.gate_h = gate_h
        # The clock is per-env so a course sweep can hold it
        # fixed while the course grows, which is what makes
        # PROPELLANT the binding constraint rather than time.
        self.seconds = float(seconds)
        # Gate COUNT is per-env so the curriculum can grow the
        # course. The observation width does not change with it:
        # `gates_left` is normalised by the module constant, so a
        # 1-gate warm start loads into a 3-gate run.
        self.n_gates = int(n_gates)
        # WRENCH, not valves, and this is the lab's one departure
        # from labs 1 to 3.
        #
        # Those labs hand the policy 20 unidirectional valves and
        # make it discover thrust allocation end to end. That is the
        # hardest framing and the only one that can beat the
        # allocator, and on a single-objective task it works. It
        # does not work here: across four reward designs and 15M
        # steps the valve-level policy learned to survive and then
        # to hover, reaching 0.5 of 3 gates and no completions,
        # because coordinating 20 valves into SUSTAINED DIRECTED
        # flight across three waypoints is a far larger exploration
        # problem than arresting a tumble.
        #
        # So the policy commands a 6-vector body wrench and a convex
        # solver turns it into valve openings. The policy still
        # learns the whole guidance and attitude problem; it no
        # longer has to rediscover an allocation a solver computes
        # exactly. `--action valves` keeps the old framing for
        # anyone who wants to reproduce the negative result.
        if action not in ("wrench", "valves", "residual"):
            raise ValueError(
                f"action must be wrench, valves or residual, got {action!r}")
        self.action = action
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
        self.n_valves = len(self.rig)
        self.act_dim = 6 if self.action in ("wrench", "residual") \
            else self.n_valves
        if self.action in ("wrench", "residual"):
            B = TH.effectiveness(self.rig)
            self.alloc = WrenchAllocator(B)
            # THE COMMAND SCALE IS A CONTROLLER'S RANGE, NOT THE
            # VEHICLE'S LIMIT, and getting this wrong cost more
            # training time than any reward bug here.
            #
            # Scaling the action box to everything the rig can
            # produce sounds right and is not: full scale is 6.3x
            # weight vertically and 24,600 N.m of torque, which is
            # 11 m/s^2 of lateral acceleration. A Gaussian policy
            # over a box that wide spends almost every sample at a
            # corner, so the car is thrown around violently at
            # every step and fine control is not reachable from the
            # initialisation. The policy hovered because hovering
            # was the only thing it could do accurately.
            #
            # These are the authorities a human would design for:
            # about a gravity of lateral acceleration and enough
            # vertical to climb briskly. The rig's real limits
            # still apply -- the allocator clips to them -- this
            # only sets what one unit of policy output MEANS.
            reach = TH.reach(B)
            weight = TH.wet_mass() * self.world.g
            cap = np.array([1.2 * weight, 1.2 * weight, 2.0 * weight,
                            0.25 * max(abs(reach[3, 0]), abs(reach[3, 1])),
                            0.25 * max(abs(reach[4, 0]), abs(reach[4, 1])),
                            0.25 * max(abs(reach[5, 0]), abs(reach[5, 1]))])
            full = np.array(
                [max(abs(reach[k, 0]), abs(reach[k, 1])) for k in range(6)])
            self.w_scale = np.maximum(np.minimum(cap, full), 1e-6)
            # The residual is deliberately SMALLER than the full
            # command range. It is a correction to a working law,
            # not a second controller fighting it, and a residual
            # free to cancel the base wrench entirely would just be
            # the from-scratch problem wearing a disguise.
            self.r_scale = 0.4 * self.w_scale

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

        # ORDER MATTERS: max_steps and a first reset() must both
        # happen before obs_dim is measured, because _obs() reads
        # the clock and the tank state that reset() populates.
        self.max_steps = int(self.seconds * CTRL_HZ)
        self.reset(seed)
        self.obs_dim = len(self._obs())

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

    def base_wrench(self, wn: float = 4.0, zeta: float = 0.9,
                    kz: float = 2.2, v_cruise: float = 9.0,
                    kv: float = 1.2) -> np.ndarray:
        """
        The scheduled guidance law, as a body wrench.

        ONE definition, used by three callers: classical.py allocates
        it with the exact FISTA solver, the residual action mode adds
        a learned correction to it, and the tests score it. A second
        copy inside the controller would be two laws that have to
        agree with nothing making them.

        Cruise steers at the next gate with a velocity SETPOINT --
        a position gain big enough to cross 7 m quickly overshoots a
        4 m gate, and one gentle enough to arrive neatly never gets
        there inside the clock. Once the gates are flown it reverts
        to lab 3's descent law over the last gate's ground point.
        """
        import thrusters as TH

        R = self.rotmat()
        mass = self.m_dry_body + self.m_gas
        p, v = self.pos(), self.vel()

        if not self.flew_course():
            to_gate = self.gate() - p
            dist = float(np.linalg.norm(to_gate))
            dirn = to_gate / max(dist, 1e-9)
            # Never below 2 m/s, or the car hovers at the gate edge
            # paying the time cost without crossing it.
            speed = min(v_cruise, max(2.0, 0.9 * dist))
            acc = kv * (dirn * speed - v)
            f_w = mass * (acc + np.array([0.0, 0.0, self.world.g]))
            level = False
        else:
            gx, gy, _gz = self.gates[-1]
            ex, ey = gx - p[0], gy - p[1]
            alt = self.altitude()
            v_target = -np.clip(1.0 * max(alt - 0.9, 0.0) + 0.8, 0.0, 2.1)
            az = kz * (v_target - v[2])
            ax = 1.1 * np.clip(ex, -3.0, 3.0) - 1.9 * v[0]
            ay = 1.1 * np.clip(ey, -3.0, 3.0) - 1.9 * v[1]
            f_w = np.array([mass * ax, mass * ay,
                            mass * (self.world.g + az)])
            # Level for the landing: a tilted touchdown fails the
            # attitude gate.
            level = True

        # TILT TO TRANSLATE. The lift nozzles are the strong ones and
        # the side nozzles are trim; holding the car level all the
        # way round put every metre of translation on the weak axis
        # and burned 265 kg of 297 while timing out at 1.94 gates.
        if level or float(np.linalg.norm(f_w)) < 1e-6:
            b3_d = np.array([0.0, 0.0, 1.0])
        else:
            b3_d = f_w / float(np.linalg.norm(f_w))
            max_tilt = math.radians(35.0)
            if math.acos(float(np.clip(b3_d[2], -1.0, 1.0))) > max_tilt:
                flat = b3_d[:2] / max(np.linalg.norm(b3_d[:2]), 1e-9)
                b3_d = np.array([flat[0] * math.sin(max_tilt),
                                 flat[1] * math.sin(max_tilt),
                                 math.cos(max_tilt)])

        bz = R[:, 2]
        axis_w = np.cross(bz, b3_d)
        sn = float(np.linalg.norm(axis_w))
        cs = float(np.clip(bz @ b3_d, -1.0, 1.0))
        err_b = R.T @ ((axis_w / sn * math.atan2(sn, cs))
                       if sn > 1e-9 else np.zeros(3))
        Kp, Kd = wn ** 2, 2.0 * zeta * wn
        tau = TH.inertia() * (Kp * err_b - Kd * self.omega())
        return np.concatenate([R.T @ f_w, tau])

    def hover_action(self) -> np.ndarray:
        """
        The action that holds the vehicle up, in whichever space the
        policy is commanding.

        The learner initialises its output bias here so episode one
        is a hover rather than a dive or a climb. It has to follow
        the world: on Mars the same hardware needs 2.9x less of it.
        """
        import thrusters as TH

        if self.action == "valves":
            from ppo import hover_bias
            return hover_bias(self.rig, self.world)
        if self.action == "residual":
            # Zero residual: start AS the scheduled controller.
            return np.full(6, 0.5)
        w = np.zeros(6)
        w[2] = TH.wet_mass() * self.world.g
        return np.clip(0.5 + 0.5 * w / self.w_scale, 0.0, 1.0)

    def gate(self) -> np.ndarray:
        """The gate currently being flown, as a 3-vector."""
        k = min(self.gate_i, len(self.gates) - 1)
        return np.asarray(self.gates[k], dtype=float)

    def gate_dist(self) -> float:
        """3D distance to the gate currently being flown."""
        return float(np.linalg.norm(self.pos() - self.gate()))

    def _next_gate(self) -> np.ndarray:
        """The gate after the current one, or the current one at the end."""
        k = min(self.gate_i + 1, len(self.gates) - 1)
        return np.asarray(self.gates[k], dtype=float)

    def gates_left(self) -> int:
        return max(0, len(self.gates) - self.gate_i)

    def flew_course(self) -> bool:
        return self.gate_i >= len(self.gates)

    def miss(self) -> float:
        """Horizontal distance to the LANDING point: the last gate."""
        p = self.pos()
        g = self.gates[-1]
        return float(math.hypot(p[0] - g[0], p[1] - g[1]))

    def at_gate(self) -> bool:
        return self.gate_dist() <= GATE_R

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

        # A COURSE: three gates in a gently bending line, then a
        # landing at the last one. Each gate sits above the ground
        # under it, so the car has to climb and turn rather than
        # fly one straight ramp.
        #
        # LEG LENGTH IS SAMPLED PER EPISODE, and the range is the
        # curriculum. A two-stage version of this -- a 3 m single
        # gate, then warm start onto the 9 m course -- collapsed:
        # stage A reached 0.77 gates and stage B fell to 0.01,
        # because a policy that has only ever flown to a point 3 m
        # away is outside everything it knows at 9 m and the warm
        # start carried nothing. Sampling the range every episode
        # removes the stage boundary: every batch holds draws the
        # policy can already fly and draws it cannot yet, so there
        # is always a gradient and never a cliff.
        #
        # The heading is allowed to bend by up to 0.6 rad per leg.
        # Straighter than that and the course is a single burn with
        # no flying in it; much more and the turn radius exceeds what
        # the vehicle can do inside a leg, which makes the task a
        # test of the generator rather than of the policy.
        x0, y0 = 0.0, 0.0
        head = r.uniform(0.0, 2 * math.pi)
        self.gates = []
        gx, gy = x0, y0
        for _k in range(self.n_gates):
            head += r.uniform(-0.6, 0.6)
            leg = r.uniform(*self.leg)
            gx += leg * math.cos(head)
            gy += leg * math.sin(head)
            self.gates.append((gx, gy,
                               self._h(gx, gy) + r.uniform(*self.gate_h)))
        self.gate_i = 0

        gz = self._h(x0, y0)
        self.data.qpos[:3] = [x0, y0, gz + 3.0 + 1.0 * r.uniform(-1, 1)]

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
        self.u_act = np.zeros(self.n_valves)
        self._hist = [np.zeros(self.n_valves) for _ in range(self._delay_n)]
        if self.action in ("wrench", "residual"):
            self.alloc.reset()
        self.t = 0
        self.gas_used = 0.0
        self.outcome = ""
        self.gate_times: list[float] = []
        self._gate_pay = 0.0
        self._settled = 0
        self.touchdown = None
        mujoco.mj_forward(self.model, self.data)
        self._prev_phi = self._potential()
        return self._obs()

    def _obs(self) -> np.ndarray:
        R = self.rotmat()
        p = self.pos()
        tk = self.tank
        return np.concatenate([
            R[:, 0], R[:, 1],                        # 6 attitude
            [R[2, 2]],                               # 1 uprightness
            self.omega() / 5.0,                      # 3
            self.vel() / 10.0,                       # 3
            [self.altitude() / 8.0, self.clearance() / 8.0],
            # Where the NEXT gate is, in the car's own frame, in
            # three dimensions -- the course climbs, so a flat
            # bearing would not be enough. Without the gate index
            # the task is not Markov: the same position means
            # different things depending on how much course is left.
            (R.T @ (self.gate() - p)) / 12.0,        # 3
            [min(self.gate_dist(), 20.0) / 20.0],    # 1
            [self.gates_left() / N_GATES],           # 1
            # The gate AFTER the next one, so the policy can set up
            # a turn instead of arriving at each gate stopped.
            (R.T @ (self._next_gate() - p)) / 20.0,  # 3
            [self.m_gas / TH.gas_mass(), tk["frac"]],
            [1.0 - self.t / self.max_steps],
            self.u_act,
        ]).astype(np.float64)

    def step(self, action: np.ndarray):
        mujoco = self._mj
        act = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        if self.action == "wrench":
            # The policy works in [0, 1] like every other lab here,
            # so a wrench command is mapped off centre: 0.5 is zero
            # force on that axis.
            w = (2.0 * act - 1.0) * self.w_scale
            u_cmd = self.alloc.solve(w, max(self.tank["frac"], 1e-3))
        elif self.action == "residual":
            # RESIDUAL: the guidance law flies, the policy corrects.
            #
            # From scratch, end-to-end RL converged at 1.14 of 3
            # gates across 7.4M steps and two action spaces -- the
            # curve went flat, not slowly up. Chaining three
            # waypoints is a harder exploration problem than this
            # budget solves, and that negative is reported rather
            # than hidden.
            #
            # So the policy starts from a controller that already
            # flies the course and learns a correction to it. At
            # action 0.5 on every axis the residual is exactly
            # zero, which means episode one IS the scheduled
            # controller: the learner can only move away from a
            # working policy deliberately, and any improvement it
            # shows is improvement over a real baseline rather than
            # over nothing.
            resid = (2.0 * act - 1.0) * self.r_scale
            w = self.base_wrench() + resid
            u_cmd = self.alloc.solve(w, max(self.tank["frac"], 1e-3))
        else:
            u_cmd = act
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

            # Gate capture is checked EVERY SUBSTEP, not once per
            # control step. At 12 m/s the car covers 0.12 m per
            # control step but 0.024 m per substep, and a 4 m gate
            # tested at 100 Hz can still be flown clean through at
            # an angle without any single sample landing inside it.
            # Sampling a continuous trajectory at the control rate
            # is how a gate silently becomes impossible.
            while (self.gate_i < len(self.gates)
                   and self.gate_dist() <= GATE_R):
                self.gate_i += 1
                self.gate_times.append(self.t / CTRL_HZ)
                self._gate_pay += GATE_BONUS

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
        rew = self._reward() + self._gate_pay
        self._gate_pay = 0.0
        if not done and self.t >= self.max_steps:
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
        # PROGRESS THROUGH THE COURSE is part of the state, so it
        # belongs in the potential. Phi is a function of the full
        # state -- which includes the gate index -- so the shaping
        # stays potential-based and the telescoping guarantee holds.
        # A term that depended on the PATH taken rather than the
        # state would quietly stop being potential-based.
        done_term = 6.0 * self.gate_i                        # 0 .. 18
        # No saturation inside the arena. Lab 3 clipped its distance
        # term at 12 m, so beyond 12 m the shaping was perfectly
        # flat and the policy fled to 86 m across ground that gave
        # it no reason to come back. A leg here is up to 18 m, so
        # the scale is set from the course rather than guessed.
        # The WEIGHT here is set against the time cost, not picked.
        # At 0.15 per step the car pays to exist; closing on a gate
        # at 10 m/s moves it 0.1 m per control step, so a weight of
        # w gives w * 0.1 / 25 per step of visible progress. The
        # first version used w = 2, worth 0.008 against that 0.15,
        # and the policy flew 0.02 gates in 740k steps because
        # approaching a gate was indistinguishable from hovering.
        up = 0.5 * (self.up_dot() + 1.0)                      # 0 .. 1

        if self.flew_course():
            # ONCE THE GATES ARE FLOWN THE JOB CHANGES, and so must
            # the potential. The first version kept pointing at the
            # last gate -- which sits 3 to 8 m ABOVE the ground --
            # so after flying the course nothing pulled the car
            # down. It passed the gate on half of episodes and then
            # hovered beside it until the clock ran out: 0% course
            # completions with 75% timeouts, which reads like a
            # failure to learn and was a failure to ASK.
            land = 5.0 * max(0.0, 1.0 - self.miss() / 25.0)   # 0 .. 5
            low = 3.0 * (1.0 - min(self.altitude(), 10.0) / 10.0)
            return done_term + land + low + up          # max 18+5+3+1

        near = 5.0 * max(0.0, 1.0 - self.gate_dist() / 25.0)  # 0 .. 5
        return done_term + near + up                      # max 18+5+1

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

        # NO GATE TERM HERE. Gates are paid the instant they are
        # crossed, in step(), and paying them again at the terminal
        # would double the course's value -- making a flown gate
        # worth more to a policy that then crashes than to one that
        # lands.

        if self.outcome == "crash":
            # GRADED BY IMPACT SPEED, and the WORST outcome there
            # is. Two separate lessons sit in this one line.
            #
            # Graded, because a flat penalty scores a 10 m/s dive
            # and a 2.6 m/s arrival identically, so slowing down is
            # not discoverable.
            #
            # Worst, because lab 3 put the timeout below the crash
            # -- a lander that never lands failed harder than one
            # that tried and broke -- and that ordering is wrong
            # here. With a per-step time cost and a cheap terminal
            # one dive away, crashing at step 100 beat running the
            # clock out, so an agent that could not yet fly the
            # course learned to END the episode. Crashes reached
            # 98% and exploration stopped there. Destroying the
            # vehicle is now the bottom of the ladder, which is
            # also the honest engineering ordering, and the
            # cheapest improvement available to a random policy
            # becomes "stop crashing" rather than "crash sooner".
            return -120.0 - 4.0 * min(sink, 8.0)
        if self.outcome == "timeout":
            # Bad, but survivable: the vehicle is intact. Above a
            # crash and far below any landing.
            return -90.0

        self.touchdown = dict(sink=sink, tilt=tilt, spin=spin, miss=miss,
                              gates=self.gate_i)
        gentle = sink <= SINK_OK and tilt <= TILT_OK and spin <= SPIN_OK

        if not gentle:
            self.outcome = "landed_hard"
            return -25.0
        if self.flew_course():
            self.outcome = "flew_course"
            # The whole job: every gate, then down on the wheels.
            # Graded by how neatly it arrived so there is always
            # somewhere left to improve to.
            return (50.0
                    + 10.0 * (1.0 - min(miss, 12.0) / 12.0)
                    + 10.0 * (1.0 - tilt / TILT_OK))
        self.outcome = "landed_short"
        # Put itself down safely without finishing the course. Worth
        # more than breaking and much less than flying it.
        return -5.0

    def success(self) -> bool:
        return self.outcome == "flew_course"


def random_policy(rng, env) -> np.ndarray:
    return rng.uniform(0.0, 0.6, env.act_dim)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="dense", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="rough")
    ap.add_argument("--world", default="mars", choices=sorted(TH.WORLDS))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--budget", action="store_true",
                    help="can the tank afford this course, on each world?")
    a, _ = ap.parse_known_args()

    if a.budget:
        print("=" * 78)
        print("  CAN THE TANK AFFORD THE COURSE?")
        print("=" * 78)
        print(f"  The course is {N_GATES} gates on 12-18 m legs, so about")
        print(f"  45 m of flying, and the clock is {MAX_SECONDS:.0f} s.")
        print("  Supporting the vehicle for that long is the dominant cost --")
        print("  translation is nearly free next to not falling.\n")
        print(f"  {'':8}{'g':>7}{'thrust':>9}{'L/W':>7}"
              f"{'hover s':>9}{'clock':>8}{'margin':>9}")
        for w in ("earth", "mars"):
            W = TH.WORLDS[w]
            au = TH.audit("dense", TH.rig("dense"), world=W)
            hov = au["hover_s"]
            print(f"  {w:<8}{W.g:>7.2f}{TH.thrust_scale(W):>9.3f}"
                  f"{au['lift_to_weight']:>7.2f}{hov:>9.1f}"
                  f"{MAX_SECONDS:>8.0f}{hov / MAX_SECONDS:>9.2f}x")
        print("\n  A margin below 1.0 means the vehicle cannot hold itself up")
        print("  for the length of the episode even at perfect efficiency, so")
        print("  the course is not merely hard there -- it is unaffordable.")
        print("  That is the lab's falsifier and it is arithmetic, not a")
        print("  training result.")
        return 0

    env = FlightEnv(a.rig, a.preset, world=a.world)
    au = TH.audit(a.rig, env.rig, world=TH.WORLDS[a.world])
    print("=" * 78)
    print(f"  FLIGHT ENV — rig {a.rig}, terrain {a.preset}, world {a.world}")
    print("=" * 78)
    print(f"  obs {env.obs_dim}   act {env.act_dim}   "
          f"{CTRL_HZ:.0f} Hz, max {MAX_SECONDS:.0f} s")
    print(f"  {N_GATES} gates, {GATE_R:.0f} m radius   "
          f"L/W {au['lift_to_weight']:.2f}   hover {au['hover_s']:.1f} s")

    rng = np.random.default_rng(0)
    tally: dict[str, int] = {}
    gates = []
    for ep in range(a.episodes):
        env.reset(ep)
        while True:
            _o, _r, d, _ = env.step(random_policy(rng, env))
            if d:
                break
        tally[env.outcome] = tally.get(env.outcome, 0) + 1
        gates.append(env.gate_i)
    print(f"\n  {a.episodes} episodes under a RANDOM policy:")
    for k in sorted(tally):
        print(f"    {k:<14} {tally[k]:>3}  ({tally[k] / a.episodes:.0%})")
    print(f"    mean gates flown  {np.mean(gates):.2f} of {N_GATES}")
    print("\n  A random policy should fly essentially no gates. If it flies")
    print("  one regularly, the first gate is too close to the launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
