#!/usr/bin/env python3
"""
A staircase, and a robot that is one of four shapes.

The environment is identical across all four cells of the 2x2 -- same
staircase, same reward, same termination, same episode length. Only the
robot changes. That is what makes the comparison a comparison, and it is
the thing most easily broken by a well-meaning tweak: adjust the reward to
help the free-torso arm along and the experiment stops measuring anything.

TERMINATION, AND A DISTINCTION WORTH KEEPING
--------------------------------------------
There are two ways to fail and they are not the same:

    COLLAPSE   the torso drops below 0.55 m. Available to every robot --
               the legs buckle. Measured passively: 3.0-3.5 s for the
               locked cells.
    TIPPING    the torso rotates past 1.2 rad. Only possible when the
               pitch joint exists. Measured passively: 2.4-2.8 s.

A locked torso cannot tip. It can still collapse, which is why the locked
cells are easier rather than trivial, and why `info` reports which of the
two ended the episode instead of a single `done`.

THE REWARD
----------
    + forward velocity
    + 1.0 per surviving step
    - 0.001 * sum(torque^2)
    + 3.0 once per NEW step climbed

The per-step bonus is what makes a staircase different from one box.
Paying only on reaching the top rewards a single lucky lunge; paying per
tread rewards the repeatable behaviour, which is the point of using three
steps instead of one.
"""

from __future__ import annotations

import numpy as np

from morphology import (N_STAIRS, RISE_MAX, RISE_MIN, STAIR_END_X, STAIR_X0,
                        act_dim, build_xml, obs_dim, stair_tops)


class BipedStairs:
    """Gymnasium-shaped, no Gymnasium dependency. `reset` / `step`."""

    def __init__(self, legs: int = 2, locked_torso: bool = False,
                 max_steps: int = 700, frame_skip: int = 5,
                 seed: int | None = None,
                 fixed_rise: float | None = None) -> None:
        import mujoco

        self._mj = mujoco
        self.legs = legs
        self.locked_torso = locked_torso
        self.fixed_rise = fixed_rise
        self.rise = fixed_rise if fixed_rise is not None else RISE_MIN
        self.obs_dim = obs_dim(legs, locked_torso)
        self.act_dim = act_dim(legs)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_string(
            build_xml(legs, locked_torso, self.rise))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * frame_skip
        self._feet = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in (("footL_geom", "footR_geom") if legs == 2
                      else ("footC_geom",))]
        if any(g < 0 for g in self._feet):
            raise RuntimeError("foot geoms are unnamed; climb detection "
                               "cannot work")
        self._stair_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"stair{i}") for i in range(N_STAIRS)]
        self.t = 0
        self._climbed = 0
        self._rest_z = self._measure_rest_z()

    def _measure_rest_z(self) -> float:
        """
        What `qpos[1]` reads when the robot is standing on flat ground.

        Measured once at construction rather than hardcoded, because it
        depends on leg geometry and would silently rot if a link length
        changed.

        This exists because of a real bug. `steps_climbed` compared the
        root-z OFFSET against an ABSOLUTE tread height: standing on the
        0.24 m top tread reads +0.045, not +0.24, because the robot rests
        at -0.195 on flat floor. The threshold was +0.124, so a robot
        standing squarely on the top step counted as having climbed
        nothing -- and a trained policy that walked 14.5 m past the whole
        staircase reported `climbed 0.00`.

        The same bug is in lab 1's `on_box`, where it was invisible
        because that lab's headline metric was x-position rather than
        height.
        """
        import mujoco

        probe = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, probe)
        probe.qpos[0] = -5.0                     # far from any staircase
        mujoco.mj_forward(self.model, probe)
        for _ in range(120):
            mujoco.mj_step(self.model, probe)
        self._foot_rest_z = min(float(probe.geom_xpos[g][2])
                                for g in self._feet)
        return float(probe.qpos[1])

    # -- the staircase ------------------------------------------------------

    def _set_rise(self, rise: float) -> None:
        """
        Resize every tread in place, no model rebuild.

        Each box is specified by half-extent AND centre, so both must move
        together -- changing only the size sinks the step into the floor.
        """
        self.rise = float(rise)
        for gid, (x0, x1, top) in zip(self._stair_geoms,
                                      stair_tops(self.rise)):
            self.model.geom_size[gid] = [(x1 - x0) / 2, 0.6, top / 2]
            self.model.geom_pos[gid] = [(x0 + x1) / 2, 0.0, top / 2]

    def steps_climbed(self) -> int:
        """
        How many treads are behind the robot AND beneath it.

        Both conditions matter. Horizontal progress alone counts a robot
        that slid along the floor past the staircase; height alone counts
        one mid-leap. Requiring the torso to be raised by most of the
        tread it has passed is the combination that can only mean it is
        standing on the thing.
        """
        x = float(self.data.qpos[0])
        # FOOT LIFT above its own resting height on flat ground. This
        # took three attempts and each wrong answer was quietly plausible:
        #
        #   1. torso qpos[1] vs an absolute tread height -- ignored that
        #      the robot rests at -0.196, so a policy standing on the top
        #      step counted as climbing nothing.
        #   2. torso lift above resting -- a robot on a tread with BENT
        #      legs sits lower than one standing straight on the floor.
        #      Measured crouched on a 0.07 m tread: +0.032 against a 0.042
        #      threshold, missed.
        #   3. raw foot z -- the foot capsule has a 0.045 m radius, so a
        #      foot flat on the ground already clears the first tread's
        #      threshold and a robot that never left the floor read 1/3.
        #
        # Measuring the foot against ITS OWN flat-ground height is immune
        # to all three.
        foot_lift = (min(float(self.data.geom_xpos[g][2])
                         for g in self._feet) - self._foot_rest_z)
        n = 0
        for x0, x1, top in stair_tops(self.rise):
            if x >= x0 and foot_lift > 0.6 * top:
                n += 1
        return n

    # -- observation --------------------------------------------------------

    def _obs(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        climbed = self.steps_climbed()
        next_top = stair_tops(self.rise)[min(climbed, N_STAIRS - 1)][2]
        return np.concatenate([
            q[1:],                                   # everything but x
            np.clip(v, -10.0, 10.0),
            [STAIR_X0 - q[0]],                       # distance to the stairs
            [self.rise],                             # how steep they are
            [climbed / N_STAIRS],                    # how far up already
            [next_top],                              # the next tread's height
        ]).astype(np.float64)

    # -- state --------------------------------------------------------------

    def torso_height(self) -> float:
        return 1.0 + self.data.qpos[1]

    def pitch(self) -> float:
        return 0.0 if self.locked_torso else float(self.data.qpos[2])

    def collapsed(self) -> bool:
        return self.torso_height() < 0.55

    def tipped(self) -> bool:
        return (not self.locked_torso) and abs(self.pitch()) > 1.2

    def at_top(self) -> bool:
        return self.steps_climbed() >= N_STAIRS

    # -- the loop -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._set_rise(self.fixed_rise if self.fixed_rise is not None
                       else self.rng.uniform(RISE_MIN, RISE_MAX))
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nv)
        self._mj.mj_forward(self.model, self.data)
        self.t = 0
        self._climbed = 0
        return self._obs(), {"rise": self.rise}

    def step(self, action: np.ndarray) -> tuple:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        x_before = self.data.qpos[0]

        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        forward = (self.data.qpos[0] - x_before) / self.dt
        reward = forward + 1.0 - 1e-3 * float(np.sum(np.square(action)))

        climbed = self.steps_climbed()
        if climbed > self._climbed:
            reward += 3.0 * (climbed - self._climbed)   # once per NEW tread
            self._climbed = climbed

        self.t += 1
        collapsed, tipped = self.collapsed(), self.tipped()
        return self._obs(), float(reward), bool(collapsed or tipped), \
            self.t >= self.max_steps, {
                "x": float(self.data.qpos[0]),
                "rise": self.rise,
                "climbed": climbed,
                "at_top": self.at_top(),
                "height": self.torso_height(),
                "pitch": self.pitch(),
                "ended_by": ("tipped" if tipped else
                             "collapsed" if collapsed else ""),
            }


def rollout(env: BipedStairs, policy, seed: int = 0) -> dict:
    """One episode, reporting behaviour rather than only the return."""
    obs, _ = env.reset(seed=seed)
    total, steps, max_x, best = 0.0, 0, -np.inf, 0
    ended = ""
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        steps += 1
        max_x = max(max_x, info["x"])
        best = max(best, info["climbed"])
        ended = info["ended_by"]
        if term or trunc:
            break
    return {"return": total, "steps": steps, "max_x": float(max_x),
            "climbed": best, "at_top": best >= N_STAIRS,
            "rise": env.rise, "ended_by": ended or "time"}


def random_policy(rng: np.random.Generator, n: int):
    return lambda obs: rng.uniform(-1.0, 1.0, n)


def main() -> None:
    """The random baseline for every cell, before anything is trained."""
    from morphology import CELLS

    rng = np.random.default_rng(0)
    print("=" * 74)
    print("  Staircase — the random baseline for all four robots")
    print("=" * 74)
    print(f"  {N_STAIRS} steps, rise ~U({RISE_MIN}, {RISE_MAX}) m, "
          f"top at x={STAIR_END_X:.2f}")
    print()
    print(f"  {'cell':<14}{'return':>9}{'max x':>8}{'steps up':>10}"
          f"{'reached top':>13}   ended by")
    for name, legs, locked in CELLS:
        env = BipedStairs(legs=legs, locked_torso=locked)
        rs = [rollout(env, random_policy(rng, env.act_dim), seed=s)
              for s in range(10)]
        ends = {}
        for r in rs:
            ends[r["ended_by"]] = ends.get(r["ended_by"], 0) + 1
        print(f"  {name:<14}{np.mean([r['return'] for r in rs]):>9.1f}"
              f"{np.mean([r['max_x'] for r in rs]):>8.2f}"
              f"{np.mean([r['climbed'] for r in rs]):>10.1f}"
              f"{sum(r['at_top'] for r in rs):>9}/10   "
              f"{', '.join(f'{k} x{v}' for k, v in sorted(ends.items()))}")
    print()
    print("  Every trained number in this lab is quoted against these.")


if __name__ == "__main__":
    main()
