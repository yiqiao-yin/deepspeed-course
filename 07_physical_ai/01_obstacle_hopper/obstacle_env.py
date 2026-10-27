#!/usr/bin/env python3
"""
A 3D MuJoCo world with one obstacle of VARYING height, and a hopper.

THE WORLD
---------
Everything here is built from scratch in about sixty lines of MuJoCo XML, so
there is nothing inherited and nothing hidden: a flat plane, one box, and a
three-link hopper -- torso, thigh, shin, foot.

**The box is a different height every episode**, drawn from 0.02 m to
0.17 m, and that is what makes this a learning problem rather than a gait
problem. At the low end a competent walk carries straight over the lip. At
the high end the robot has to notice and climb. One fixed height teaches one
gait; a range forces the policy to look at the obstacle and do something
different depending on what it sees.

The physics is fully three-dimensional -- the geoms are 3D capsules and a 3D
box, contacts are resolved in 3D -- but the robot's root joints constrain it
to the x-z plane. That is deliberate for a first lab. A free-floating root
has to learn balance in two axes before it can learn anything about the
obstacle, which turns a three-minute training run into an overnight one. The
plane constraint is three lines of XML and the obvious thing to relax in a
follow-up lab.

WHAT THE ROBOT CAN SEE
----------------------
Thirteen numbers: five joint positions, six velocities, and the two that
make the task solvable -- the signed distance to the near face of the box,
and **the height of the box**.

Neither is decoration. Standard Gymnasium hoppers deliberately EXCLUDE the
root x-coordinate, because plain locomotion is translation-invariant: a
walking gait is the same wherever you are. This task is not. There is an
obstacle at a known place with an unknown height, and a policy that cannot
perceive it can at best learn one compromise gait that is wrong at both
ends of the range.

`blind_to_height=True` zeroes that one input and changes nothing else. It
exists so the test suite can train the impaired policy and assert it does
measurably worse on tall boxes -- the same construction as
`tests/test_omni_eval.py`, which builds a model that ignores a modality on
purpose and requires the harness to catch it. A claim that the robot "sees"
the obstacle is worth nothing unless blinding it costs something.

THE REWARD
----------
    + forward velocity          go somewhere
    + 1.0 per surviving step    do not fall over
    - 0.001 * sum(torque^2)     do not flail
    + 5.0 once, on clearing     got past the box

The one-off clearing bonus is what makes this an obstacle task rather than a
running task, and it is keyed to getting PAST the box rather than onto it,
so the same goal works at every height. A 0.03 m lip is cleared by walking;
a 0.15 m step has to be climbed. The reward never mentions how.
"""

from __future__ import annotations

import numpy as np

# Geometry shared between the XML and the reward, so the two cannot disagree
# about where the box is -- a class of bug that produces a reward for
# arriving somewhere the obstacle is not.
# Close to the start, deliberately. At 2.2 m the trained policy spent the
# whole episode learning to WALK and fell over around step 276 before ever
# reaching the obstacle -- so the run measured locomotion endurance, not
# obstacle crossing. This lab is about the obstacle. Putting it at 1.2 m
# means the robot meets it within the first couple of seconds, and the
# interesting behaviour is what it does there.
BOX_CENTRE_X = 1.2
# A STEP, not a plateau. The first version was 0.9 m deep, and the trained
# policy stalled on top of it: it reliably climbed on (7/8 at 0.08 m) and
# then had most of a metre of narrow platform to cross before the far edge
# counted as cleared. That measures balance-on-a-ledge, which is a different
# and much harder task than the one this lab is about. 0.3 m deep is one
# stride, so clearing it means stepping up and over.
BOX_HALF_X = 0.15
BOX_FRONT_X = BOX_CENTRE_X - BOX_HALF_X      # 1.75
BOX_BACK_X = BOX_CENTRE_X + BOX_HALF_X       # 2.65

# The obstacle height is drawn fresh every episode, and this range is the
# heart of the task. MEASURED, not guessed: placed gently on the box the
# hopper rests stably from 0.02 m to 0.18 m and topples off at 0.20 m, so
# the range stops below that. Across it the task genuinely changes character
# -- a 0.03 m lip can be walked over with the existing gait, a 0.15 m step
# has to be climbed -- and that is the point. One fixed height teaches a
# policy one gait. A range forces it to LOOK at the obstacle, which is why
# the height is in the observation and why the headline test ablates it.
BOX_HEIGHT_MIN = 0.02
BOX_HEIGHT_MAX = 0.17
BOX_HEIGHT_DEFAULT = 0.10
# Below this a competent walking gait usually carries straight over; above
# it the robot has to change what it is doing. Used for reporting only --
# nothing in the physics knows about this number.
WALKABLE_MAX = 0.06

MODEL_XML = f"""
<mujoco model="obstacle_hopper">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.002"/>
  <!-- Offscreen framebuffer for render.py. MuJoCo defaults to 640x480 and
       raises rather than silently downscaling, so the size has to be
       declared here. Visual only; no effect on the dynamics. -->
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" friction="0.9 0.1 0.1"
          rgba="0.42 0.62 0.82 1" solimp="0.95 0.95 0.01" solref="0.01 1"/>
  </default>
  <worldbody>
    <!-- Lights and cameras are VISUAL ONLY -- MuJoCo does not use them in
         the dynamics, so adding them cannot change a single measurement in
         this lab. They exist so render.py produces something legible. -->
    <light pos="0 -1.5 4" dir="0 0.3 -1" diffuse="0.85 0.85 0.85"
           specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="3 2 3" dir="-0.5 -0.4 -1" diffuse="0.45 0.5 0.6"
           castshadow="false"/>
    <camera name="side" pos="1.0 -3.6 1.0" xyaxes="1 0 0 0 0.35 0.94"/>
    <geom name="floor" type="plane" pos="0 0 0" size="40 2 0.1"
          conaffinity="1" condim="3" rgba="0.16 0.20 0.25 1"/>
    <geom name="step" type="box"
          pos="{BOX_CENTRE_X} 0 {BOX_HEIGHT_DEFAULT / 2}"
          size="{BOX_HALF_X} 0.6 {BOX_HEIGHT_DEFAULT / 2}"
          rgba="0.85 0.55 0.25 1"/>
    <body name="torso" pos="0 0 1.0">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false"
             armature="0" damping="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false"
             armature="0" damping="0"/>
      <!-- NO pitch joint, deliberately. With a free torso rotation this
           becomes the bipedal-balance problem, which is a famously hard
           benchmark and is NOT what this lab teaches. Measured: with pitch
           free, 1M steps of PPO produced a policy that dived forward,
           fell at ~1.0 s, and reached exactly the same x with the obstacle
           REMOVED as with a 0.17 m step -- it was never climbing anything.
           Locking the torso upright isolates the actual lesson: perceive
           the obstacle, change the leg motion. Freeing this joint is the
           obvious next lab. -->
      <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.35" size="0.06"/>
      <body name="thigh" pos="0 0 0">
        <joint name="thigh_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="shin" pos="0 0 -0.42">
          <joint name="shin_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom name="shin_geom" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.04"/>
          <body name="foot" pos="0 0 -0.38">
            <joint name="foot_joint" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot_geom" type="capsule"
                  fromto="-0.1 0 0 0.17 0 0" size="0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="thigh_joint" ctrlrange="-1 1" gear="150" ctrllimited="true"/>
    <motor joint="shin_joint"  ctrlrange="-1 1" gear="150" ctrllimited="true"/>
    <motor joint="foot_joint"  ctrlrange="-1 1" gear="90"  ctrllimited="true"/>
  </actuator>
</mujoco>
"""

OBS_DIM = 11
ACT_DIM = 3


class ObstacleHopper:
    """
    A hopper that must get past a box of unpredictable height.

    `reset(seed)` and `step(action)` follow the Gymnasium 1.x signatures, so
    this drops into any standard loop, but the class deliberately has no
    `gymnasium` import. The environment is MuJoCo plus arithmetic, which
    keeps the dependency surface small and -- more usefully -- puts the
    reward and termination logic in one readable place instead of inherited
    through three base classes.
    """

    def __init__(self, max_steps: int = 600, frame_skip: int = 5,
                 seed: int | None = None,
                 height_range: tuple[float, float] | None = None,
                 fixed_height: float | None = None,
                 blind_to_height: bool = False) -> None:
        import mujoco                                     # heavy; keep local

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.rng = np.random.default_rng(seed)
        self.height_range = height_range or (BOX_HEIGHT_MIN, BOX_HEIGHT_MAX)
        self.fixed_height = fixed_height
        self.blind_to_height = blind_to_height

        self._step_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "step")
        self.box_height = BOX_HEIGHT_DEFAULT
        self.t = 0
        self._cleared_awarded = False
        self._rest_z = self._measure_rest_z()

    def _measure_rest_z(self) -> float:
        """
        What `qpos[1]` reads standing on flat ground. Measured, not assumed.

        `on_box` compared the root-z OFFSET against an ABSOLUTE box height
        and so was always False: this robot rests at -0.152 on the floor,
        reads -0.036 standing on a 0.15 m box, and the threshold was
        +0.070. The reported `climbed 0/10` in this lab was that bug, not
        a fact about the gait, and the page's explanation -- that it hops
        over without resting -- was wrong.

        Nothing else moves. The reward and the headline metric both use
        `cleared()`, which is an x-position test and was never affected,
        so every published number in this lab stands. Found while writing
        lab 2, where the same expression also gated a reward term and did
        real damage.
        """
        import mujoco

        probe = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, probe)
        probe.qpos[0] = -5.0
        mujoco.mj_forward(self.model, probe)
        for _ in range(120):
            mujoco.mj_step(self.model, probe)
        return float(probe.qpos[1])

    # -- the obstacle -------------------------------------------------------

    def _set_box_height(self, h: float) -> None:
        """
        Resize the obstacle in place, without rebuilding the model.

        MuJoCo lets a primitive geom's size be edited on the compiled model,
        which is what makes per-episode randomisation cheap -- no XML string
        formatting and no recompile in the inner loop. Both size and position
        must move together: a box is specified by its half-extent and its
        CENTRE, so changing only the size would sink it into the floor.
        """
        h = float(np.clip(h, 0.0, BOX_HEIGHT_MAX))
        self.box_height = h
        self.model.geom_size[self._step_geom] = [BOX_HALF_X, 0.6, h / 2]
        self.model.geom_pos[self._step_geom] = [BOX_CENTRE_X, 0.0, h / 2]

    # -- observation --------------------------------------------------------

    def _obs(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        height = 0.0 if self.blind_to_height else self.box_height
        return np.concatenate([
            q[1:],                                   # z + 3 joints      (4)
            np.clip(v, -10.0, 10.0),                 # 5 velocities       (5)
            [BOX_FRONT_X - q[0]],                    # distance to box    (1)
            [height],                                # obstacle height    (1)
        ]).astype(np.float64)

    # -- state predicates ---------------------------------------------------

    def torso_height(self) -> float:
        return 1.0 + self.data.qpos[1]

    def on_box(self) -> bool:
        """
        Standing ON the step, rather than beside it or mid-leap over it.

        Height alone is not enough -- a hopper in flight on open floor is
        also high. Requires being horizontally inside the box footprint AND
        raised by most of its height, which together can only mean supported
        by it. Scaled by the CURRENT height, since that now varies.
        """
        x = self.data.qpos[0]
        inside = BOX_FRONT_X <= x <= BOX_BACK_X
        lift = self.data.qpos[1] - self._rest_z      # vs flat ground
        return bool(inside and lift > 0.6 * self.box_height)

    def cleared(self) -> bool:
        """Got past the far face. The goal, at every height."""
        return bool(self.data.qpos[0] > BOX_BACK_X)

    def touching_box(self) -> bool:
        return any(self._step_geom in (c.geom1, c.geom2)
                   for c in self.data.contact[:self.data.ncon])

    def _fallen(self) -> bool:
        """Collapsed. With the torso locked upright, height is the only way."""
        return bool(self.torso_height() < 0.55)

    # -- the loop -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        h = (self.fixed_height if self.fixed_height is not None
             else float(self.rng.uniform(*self.height_range)))
        self._set_box_height(h)

        self._mj.mj_resetData(self.model, self.data)
        # A small random nudge, so a policy cannot win by replaying one
        # trajectory. Deliberately tiny: this is a control problem, not an
        # exploration benchmark.
        self.data.qpos[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nv)
        self._mj.mj_forward(self.model, self.data)
        self.t = 0
        self._cleared_awarded = False
        return self._obs(), {"box_height": self.box_height}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        x_before = self.data.qpos[0]

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        forward = (self.data.qpos[0] - x_before) / self.dt
        reward = forward + 1.0 - 1e-3 * float(np.sum(np.square(action)))

        past = self.cleared()
        if past and not self._cleared_awarded:
            reward += 5.0                      # once, on clearing
            self._cleared_awarded = True

        self.t += 1
        return self._obs(), float(reward), self._fallen(), \
            self.t >= self.max_steps, {
                "x": self.data.qpos[0],
                "box_height": self.box_height,
                "on_box": self.on_box(),
                "cleared": past,
                "touching_box": self.touching_box(),
                "height": self.torso_height(),
            }


def rollout(env: ObstacleHopper, policy, seed: int = 0) -> dict:
    """
    Run one episode and report what happened, not just the return.

    `cleared` and `max_x` are here because the scalar return hides the thing
    the lab is about: a policy can score well by standing still and
    collecting the alive bonus, and a return alone cannot tell that apart
    from one that got over the obstacle.
    """
    obs, info = env.reset(seed=seed)
    total, steps, max_x = 0.0, 0, -np.inf
    touched = cleared = climbed = False
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        steps += 1
        max_x = max(max_x, info["x"])
        touched |= info["touching_box"]
        cleared |= info["cleared"]
        climbed |= info["on_box"]
        if term or trunc:
            break
    return {"return": total, "steps": steps, "max_x": float(max_x),
            "box_height": env.box_height, "touched_box": touched,
            "climbed": climbed, "cleared": cleared}


def random_policy(rng: np.random.Generator):
    """The baseline every result on this lab is quoted against."""
    return lambda obs: rng.uniform(-1.0, 1.0, ACT_DIM)


def main() -> None:
    """Show the world exists and the baseline is weak, without training."""
    rng = np.random.default_rng(0)
    print("=" * 72)
    print("Obstacle hopper -- the world, and the random baseline")
    print("=" * 72)
    print(f"obs {OBS_DIM}  act {ACT_DIM}  box x=[{BOX_FRONT_X}, {BOX_BACK_X}]  "
          f"height ~ U({BOX_HEIGHT_MIN}, {BOX_HEIGHT_MAX}) m")
    print()
    for lo, hi, name in ((BOX_HEIGHT_MIN, WALKABLE_MAX, "low  (walk over)"),
                         (0.12, BOX_HEIGHT_MAX, "high (must climb)")):
        env = ObstacleHopper(height_range=(lo, hi))
        rs = [rollout(env, random_policy(rng), seed=s) for s in range(10)]
        print(f"BASELINE, random policy, {name}  [{lo:.2f}, {hi:.2f}] m")
        print(f"  mean return {np.mean([r['return'] for r in rs]):7.2f}   "
              f"mean max x {np.mean([r['max_x'] for r in rs]):6.3f} m   "
              f"cleared {sum(r['cleared'] for r in rs)}/10")
    print()
    print("Every number this lab publishes is quoted against that baseline.")


if __name__ == "__main__":
    main()
