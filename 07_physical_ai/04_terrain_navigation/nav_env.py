#!/usr/bin/env python3
"""
The navigation task: get from A to B across terrain you must read.

    uv run nav_env.py              # random baseline on flat and on terrain

THE OBSERVATION, AND WHY IT IS SPLIT THREE WAYS
-----------------------------------------------
    proprio (17)   joints, velocities, and the robot's own heading
    goal (3)       range and bearing to B, in the ROBOT's frame
    terrain        one of three channels, and the experiment is which:

        blind        nothing. Must bump into a ridge to learn it exists.
        privileged   a local height patch, handed over. An oracle: a
                     real robot has no such array.
        depth        a 64x64 depth image from the head camera.

Goal is given in the robot's own frame -- range and the sine/cosine of
relative bearing -- rather than as world coordinates. A world-frame goal
invites the policy to memorise positions in a fixed arena; a relative
one is the same problem wherever you put it, and transfers to a map it
has never seen. Labs 1 and 3 both excluded absolute position for the
same reason.

WHY THE REWARD IS PROGRESS-TOWARD-GOAL RATHER THAN FORWARD VELOCITY
-------------------------------------------------------------------
Every previous lab in this category paid for forward velocity, which on
a corridor is the same thing as making progress. Here it is not: running
north at full speed when the goal is east is worth nothing, and a reward
that pays for it would produce a robot that sprints in a straight line
and ignores B.

So the reward is the REDUCTION IN DISTANCE TO B, which is the quantity
the task is actually about. It is also what makes the detour decision
expressible: climbing a ridge and walking round it both cost distance,
and the policy is paid the difference.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from robot import hfield_png, model_xml
from world import ARENA, CELL, N, STEP_MAX, generate, solve, straight_line

# PROPRIO is DERIVED from the compiled model, never typed. It is
# len(qpos[2:]) + len(qvel), and lab 3 shipped the same quantity as a
# literal: it went stale the moment the model changed, declared an
# obs_dim one short of reality, and killed three training runs on a
# broadcast error while the arms that did not touch it sailed through.
# A constant that encodes the length of another thing will eventually
# disagree with it.
GOAL_FEATS = 3
# The privileged channel: SIX numbers, not 121.
#
# The first version handed the policy an 11x11 raw height patch. That
# is 121 extra inputs on top of 21 useful ones, fed to a 2x64 MLP, and
# on mostly-flat ground almost all of them are near-zero. All three
# seeds sat at EXACTLY 0% arrival for the whole run while the blind arm
# reached 17% -- a dead-flat zero across seeds is a broken arm, not a
# finding, and reporting "information hurts" from it would have been
# the third time in this session a design mistake could pass as a
# result.
#
# These six answer the question the robot actually faces: is the thing
# ahead climbable, and if not, which way is clearer? That is also the
# shape a real perception stack emits -- a few summary statistics, not
# a raw elevation array.
PRIV_FEATS = 6
PROBE_RANGE = 3.0              # metres ahead the oracle can see
PATCH = 11                     # retained for the raw-patch variant
GOAL_RADIUS = 0.8              # within this of B counts as arrived
# A slight crouch, and the torso height that puts the feet exactly on
# the ground at that crouch -- both MEASURED, not guessed. The first
# version started the torso 0.2 m above its standing height on straight
# legs, so every episode opened with a drop onto an unstable pose that
# folded within 90 steps. The policy then spent its whole budget
# failing to stand and never reached the task.
START_BEND = 0.3
START_DZ = -0.294
DEPTH_NEAR, DEPTH_FAR = 0.5, 6.0


class NavWorld:
    """Gymnasium-shaped. `reset` / `step`, no Gymnasium dependency."""

    def __init__(self, *, mode: str = "blind", flat: bool = False,
                 depth_res: int = 64, max_steps: int = 2000,
                 frame_skip: int = 5, seed: int | None = None,
                 goal_range: float | None = None) -> None:
        import mujoco

        assert mode in ("blind", "privileged", "depth"), mode
        self._mj = mujoco
        self.mode = mode
        self.flat = flat
        self.depth_res = depth_res
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        # Curriculum. `goal_range` caps how far B may be placed; None
        # means the map's own endpoints. A 15 m goal inside a 12 s
        # episode needs 1.23 m/s sustained from the first step, so a
        # robot still learning to stand can never once reach it and
        # never sees the arrival reward at all.
        self.goal_range = goal_range
        self.rng = np.random.default_rng(seed)
        self._tmp = Path(tempfile.mkdtemp())
        self._renderer = None

        self.map = None
        self._build(generate(0))
        self.act_dim = int(self.model.nu)
        from robot import STAND
        lo = self.model.actuator_ctrlrange[:6, 0]
        hi = self.model.actuator_ctrlrange[:6, 1]
        self._centre = np.array([STAND["hip"], STAND["knee"], STAND["ankle"]] * 2)
        # Half-range on each side of standing, clipped to what the joint
        # can actually reach, so the action never commands the impossible.
        self._span = np.minimum(self._centre - lo, hi - self._centre) * 0.9
        self.proprio = int(self.model.nq - 2) + int(self.model.nv)
        extra = (PRIV_FEATS if mode == "privileged" else 0)
        self.obs_dim = self.proprio + GOAL_FEATS + extra
        # Assert rather than trust: the observation actually produced
        # must match what was advertised, every time the model builds.
        assert len(self._obs()) == self.obs_dim, (
            f"declared {self.obs_dim}, produced {len(self._obs())}")

    # -- the world ----------------------------------------------------------

    def _build(self, m) -> None:
        self.map = m
        heights = np.zeros_like(m.heights) if self.flat else m.heights
        self._heights = heights
        hfield_png(heights, self._tmp / "terrain.png")
        self.model = self._mj.MjModel.from_xml_string(
            model_xml("terrain.png", m.start, m.goal),
            {"terrain.png": (self._tmp / "terrain.png").read_bytes()})
        self.data = self._mj.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip
        self._renderer = None
        _, self.route_len = solve(m)
        self.straight = straight_line(m)

    # -- observation --------------------------------------------------------

    def pos(self) -> np.ndarray:
        return np.array([self.data.qpos[0], self.data.qpos[1]])

    def yaw(self) -> float:
        return float(self.data.qpos[3])

    def to_goal(self) -> float:
        return float(np.linalg.norm(np.array(self.map.goal) - self.pos()))

    def _goal_feats(self) -> np.ndarray:
        """Range and relative bearing -- the robot's frame, not the world's."""
        d = np.array(self.map.goal) - self.pos()
        rng_ = float(np.linalg.norm(d))
        bearing = float(np.arctan2(d[1], d[0])) - self.yaw()
        return np.array([min(rng_ / 20.0, 1.5),
                         np.sin(bearing), np.cos(bearing)])

    def height_patch(self) -> np.ndarray:
        """
        An 11x11 window of ground height ahead of the robot, relative to
        the height it is standing on. The oracle channel.

        Relative rather than absolute because the useful quantity is the
        RISE the next step must clear, which is what `STEP_MAX` is
        measured against. Absolute elevation would make the policy
        relearn the threshold at every altitude.
        """
        x, y = self.pos()
        yaw = self.yaw()
        here = self._h_at(x, y)
        out = np.zeros((PATCH, PATCH), dtype=np.float32)
        for a in range(PATCH):
            for b in range(PATCH):
                fwd = (a - PATCH // 2) * 0.30 + 0.45
                lat = (b - PATCH // 2) * 0.30
                px = x + fwd * np.cos(yaw) - lat * np.sin(yaw)
                py = y + fwd * np.sin(yaw) + lat * np.cos(yaw)
                out[a, b] = self._h_at(px, py) - here
        return np.clip(out, -1.0, 1.0).ravel()

    def terrain_feats(self) -> np.ndarray:
        """
        Six numbers: the worst rise in three directions, and whether it
        is climbable.

        Rises are measured RELATIVE to the ground the robot stands on,
        because the decision is about the step it must clear, not about
        altitude -- the top of a climbable ridge is perfectly walkable
        once you are on it.
        """
        x, y = self.pos()
        yaw = self.yaw()
        here = self._h_at(x, y)
        out = []
        for lobe in (0.0, +0.7, -0.7):             # ahead, left, right
            worst = 0.0
            for r in np.linspace(0.6, PROBE_RANGE, 9):
                px = x + r * np.cos(yaw + lobe)
                py = y + r * np.sin(yaw + lobe)
                worst = max(worst, self._h_at(px, py) - here)
            out.append(min(worst, 1.0))
            # The classification itself, which is the whole lab: is the
            # obstacle in this direction one the gait can clear?
            out.append(1.0 if worst > STEP_MAX else 0.0)
        return np.array(out, dtype=np.float64)

    def _h_at(self, x: float, y: float) -> float:
        i = int(np.clip((x + ARENA / 2) / CELL, 0, N - 1))
        j = int(np.clip((y + ARENA / 2) / CELL, 0, N - 1))
        return float(self._heights[j, i])

    def depth(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(
                self.model, height=self.depth_res, width=self.depth_res)
            self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="eye")
        z = np.clip(self._renderer.render(), DEPTH_NEAR, DEPTH_FAR)
        return ((z - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)).astype(np.float32)

    def _obs(self) -> np.ndarray:
        parts = [self.data.qpos[2:], np.clip(self.data.qvel, -10.0, 10.0),
                 self._goal_feats()]
        if self.mode == "privileged":
            parts.append(self.terrain_feats())
        return np.concatenate(parts).astype(np.float64)

    # -- state --------------------------------------------------------------

    def torso_height(self) -> float:
        return float(self.data.qpos[2]) + 1.0

    def clearance(self) -> float:
        return self.torso_height() - self._h_at(*self.pos())

    # Measured, not chosen. A healthy robot settles at clearance ~0.77
    # but its startup wobble troughs at 0.49, while a genuine collapse
    # reaches 0.06-0.09. The first version used 0.55, which sat INSIDE
    # the healthy oscillation and terminated episodes on a standing
    # robot -- so every run began with the policy being punished for
    # the one thing it had got right.
    FALL_CLEARANCE = 0.40

    def fallen(self) -> bool:
        """Sunk relative to the ground it is over -- not an absolute height.

        Absolute would make every climbable ridge read as a fall the
        moment the robot stood on one, which is how lab 3 nearly made
        `down` unlearnable by construction.
        """
        return bool(self.clearance() < self.FALL_CLEARANCE)

    def arrived(self) -> bool:
        return self.to_goal() < GOAL_RADIUS

    # -- the loop -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Only solvable maps. An unsolvable one scores every policy at
        # zero and reads as a hard task rather than a broken one.
        for k in range(40):
            m = generate(int(self.rng.integers(0, 100_000)))
            if solve(m)[0] is not None:
                break
        if self.goal_range is not None:
            m = self._pull_goal_closer(m, self.goal_range)
        self._build(m)
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = m.start
        self.data.qpos[2] = START_DZ + self._h_at(*m.start)
        self.data.qpos[3] = float(self.rng.uniform(-np.pi, np.pi))
        b = START_BEND
        self.data.qpos[4:] = [-b, -1.8 * b, 0.5 * b, -b, -1.8 * b, 0.5 * b]
        self.data.qpos[4:] += self.rng.uniform(-2e-2, 2e-2,
                                               self.model.nq - 4)
        self._mj.mj_forward(self.model, self.data)
        self.t = 0
        self.travelled = 0.0
        self._arrived = False
        self._travelled_at_arrival = None
        self._prev_d = self.to_goal()
        return self._obs(), {"seed": m.seed, "route": self.route_len}

    def _pull_goal_closer(self, m, radius: float):
        """Move B along the oracle's own route until it is `radius` away."""
        path, _ = solve(m)
        if not path:
            return m
        sx, sy = m.start
        for j, i in path:
            x = (i + 0.5) * CELL - ARENA / 2
            y = (j + 0.5) * CELL - ARENA / 2
            if np.hypot(x - sx, y - sy) >= radius:
                m.goal = (float(x), float(y))
                break
        return m

    def step(self, action: np.ndarray) -> tuple:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        p0 = self.pos()
        # Map [-1, 1] onto joint targets CENTRED ON THE STANDING POSE,
        # so a zero action holds the robot up instead of dropping it.
        self.data.ctrl[:6] = self._centre + action[:6] * self._span
        self.data.ctrl[6] = action[6]
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        self.travelled += float(np.linalg.norm(self.pos() - p0))
        d = self.to_goal()
        # Paid for CLOSING THE GAP, not for moving. See the module
        # docstring: forward velocity is the wrong currency here.
        # +1.5 alive, not +0.5. Lab 1 measured what happens when this
        # term is too small: "the policy learns to fall over
        # immediately", because falling ends the episode and ends the
        # accumulating control cost. The first version of this lab
        # halved lab 3's bonus and reproduced exactly that.
        reward = (self._prev_d - d) * 10.0 + 1.5
        reward -= 1e-3 * float(np.sum(np.square(action)))
        self._prev_d = d

        # ARRIVING MUST NOT END THE EPISODE.
        #
        # It used to, and that made success the worst available outcome:
        # standing still for 2000 steps earned 1.5 x 2000 = 3000, while
        # walking 7 m and arriving at step ~700 earned 1050 + 70 + 50 =
        # 1170. Terminating on success forfeited ~1300 steps of alive
        # bonus, so arriving was worth 2.5x LESS than loitering and the
        # policy -- correctly, given what it was paid -- learned not to
        # arrive. 75% of episodes ended in a fall after 1.4 m of travel.
        #
        # Lab 1 documents the mirror of this: "the one-off bonus must be
        # one-off. Pay it every tick and standing on the box forever
        # beats crossing it." The same arithmetic, pointed the other way.
        #
        # Now the robot keeps its alive bonus after arriving, the goal
        # bonus is paid ONCE, and `arrived` records that it ever got
        # there.
        if self.arrived() and not self._arrived:
            self._arrived = True
            # Freeze the odometer AT ARRIVAL.
            #
            # The episode deliberately continues after reaching B (see
            # above), so `travelled` keeps accumulating while the robot
            # mills around the goal for the remaining ~1300 steps. On
            # one measured episode it walked 7.4 m to B against a 7.2 m
            # optimal route -- 97% efficient -- and then another 7.2 m
            # afterwards, so the published efficiency read 49%.
            #
            # Every efficiency number in this lab was therefore about
            # half what it should have been. The task is "get to B",
            # and the distance that answers it is the distance walked
            # BY the time it got there.
            self._travelled_at_arrival = self.travelled
            reward += 100.0
        self.t += 1
        fell = self.fallen()
        return self._obs(), float(reward), bool(fell), \
            self.t >= self.max_steps, {
                "x": float(self.pos()[0]), "y": float(self.pos()[1]),
                "to_goal": d, "arrived": self._arrived, "fell": fell,
                "travelled": self.travelled, "route": self.route_len,
                "travelled_to_goal": (self._travelled_at_arrival
                                      if self._arrived else None),
            }


def rollout(env: NavWorld, policy, seed: int = 0) -> dict:
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        steps += 1
        if term or trunc:
            break
    eff = (info["route"] / info["travelled"]) if info["travelled"] > 0.1 else 0.0
    return {"return": total, "steps": steps, "arrived": info["arrived"],
            "fell": info["fell"], "to_goal": info["to_goal"],
            "travelled": info["travelled"], "efficiency": min(eff, 1.0)}


def random_policy(rng, n: int = 7):
    return lambda obs: rng.uniform(-1.0, 1.0, n)


def main() -> int:
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("  Random baseline, before anything is trained")
    print("=" * 74)
    print(f"  {'world':<12}{'return':>9}{'arrived':>10}{'fell':>8}"
          f"{'left to go':>12}{'walked':>9}")
    for flat in (True, False):
        env = NavWorld(mode="blind", flat=flat)
        rs = [rollout(env, random_policy(rng), seed=s) for s in range(8)]
        print(f"  {'flat' if flat else 'terrain':<12}"
              f"{np.mean([r['return'] for r in rs]):>9.1f}"
              f"{sum(r['arrived'] for r in rs):>8}/8"
              f"{sum(r['fell'] for r in rs):>6}/8"
              f"{np.mean([r['to_goal'] for r in rs]):>11.1f} m"
              f"{np.mean([r['travelled'] for r in rs]):>8.1f} m")
    print()
    print("  'arrived 0/8' is the point. If a random policy reaches B, the")
    print("  goal radius is too generous and there is nothing to learn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
