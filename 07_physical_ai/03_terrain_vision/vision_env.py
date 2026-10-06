#!/usr/bin/env python3
"""
Four terrains, one approach, and an optional depth camera.

THE POINT OF THE DESIGN
-----------------------
Past an identical approach slab the ground does one of four things: stays
flat, climbs three steps, drops three steps, or stops for a gap. The robot
settles at the same height on all four, so **from proprioception they are
indistinguishable** -- and they demand incompatible responses:

    flat        normal gait
    up          lift the foot high BEFORE contact
    down        lower the foot and absorb a drop
    hop         commit to a leap

`down` is why this lab exists. You cannot feel for a descent: by the time
the swinging foot finds nothing there, the robot is already falling. Labs
1 and 2 were both solvable reactively, and both information ablations
came back null. This task is built so that cannot happen.

TEACHER AND STUDENT
-------------------
`privileged=True` hands the policy a terrain descriptor -- the one-hot
type and the geometry. That is the TEACHER, and it is cheap because it
needs no rendering.

`privileged=False` withholds it. A student in that mode has only
proprioception, and the whole experiment is whether depth images let it
recover what the teacher was told.
"""

from __future__ import annotations

import numpy as np

from terrain import (EVENT_X, KINDS, N_PATCHES, N_STONES, PATCH_FRICTION,
                     PATCH_LEN, PATCH_SPACING, PLATEAU_Z, RUN,
                     STONE_GAP_MAX, STONE_GAP_MIN, STONE_TOP, world)

# Terrain randomisation ranges. Deliberately narrow for a first pass: the
# question is whether vision is needed at all, not whether the policy
# generalises across every geometry.
RISE_MIN, RISE_MAX = 0.06, 0.11
GAP_MIN, GAP_MAX = 0.30, 0.50

DEPTH_NEAR, DEPTH_FAR = 0.8, 3.5
_REST_Z: float | None = None

PROPRIO = 15            # qpos[1:] (7) + qvel (8); torso is locked, no pitch
# DERIVED, not written down. This was the literal 6 and went stale the
# moment `hop` was dropped from KINDS: the declared obs_dim said 21 while
# the observation was 20, and three training runs died on the mismatch
# while the three blind runs -- which never touch this constant -- sailed
# through. A constant that encodes the length of another constant will
# eventually disagree with it.
TERRAIN_FEATS = len(KINDS) + 2        # one-hot per terrain + rise + gap


class TerrainWorld:
    """Gymnasium-shaped. `reset` / `step`, no Gymnasium dependency."""

    def __init__(self, kind: str | None = None, *, privileged: bool = True,
                 depth: bool = False, depth_res: int = 64,
                 max_steps: int = 600, frame_skip: int = 5,
                 seed: int | None = None,
                 fixed_rise: float | None = None,
                 fixed_gap: float | None = None,
                 fovy: float = 45.0,
                 kinds: tuple[str, ...] | None = None,
                 sensor: str = "depth") -> None:
        import mujoco

        self._mj = mujoco
        self.fixed_kind = kind
        self.privileged = privileged
        self.want_depth = depth
        self.depth_res = depth_res
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.fixed_rise, self.fixed_gap = fixed_rise, fixed_gap
        # Field of view, in degrees. PART 2 sweeps this: if the camera
        # is doing real work, a WIDER one should do more of it, and a
        # dose-response curve is much harder to explain away than a
        # single on/off comparison.
        self.fovy = fovy
        # Which terrains this world samples from, and therefore how
        # wide the privileged one-hot is. Defaults to the published
        # three so `obs_dim` stays 20 and existing checkpoints load.
        self.kinds = tuple(kinds) if kinds else KINDS
        # "depth" or "rgb". The friction-patch terrain is the first one
        # here where this MATTERS rather than being a rendering detail:
        # a slippery patch is perfectly flat, so depth cannot see it at
        # all and acts as a negative control for "more inputs help".
        assert sensor in ("depth", "rgb"), sensor
        self.sensor = sensor
        self._rgb_renderer = None
        self.rng = np.random.default_rng(seed)

        self.kind = kind or "flat"
        self.rise = fixed_rise or 0.09
        self.gap = fixed_gap or 0.40
        self._build()

        self.obs_dim = PROPRIO + ((len(self.kinds) + 2) if privileged else 0)
        self.act_dim = 6
        self._renderer = None
        self.t = 0
        self._crossed = False

    # -- the world ----------------------------------------------------------

    def _build(self) -> None:
        """
        Rebuild the model. Unlike labs 1-2 this cannot be an in-place geom
        resize, because the four terrains have DIFFERENT geom counts -- a
        gap has no steps to resize. Rebuilding costs ~2 ms, which is
        nothing next to an episode, and it is the honest way to express
        four genuinely different worlds.
        """
        self.model = self._mj.MjModel.from_xml_string(
            world(self.kind, self.rise, self.gap, fovy=self.fovy))
        self.data = self._mj.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip
        self._feet = [self._mj.mj_name2id(self.model,
                                          self._mj.mjtObj.mjOBJ_GEOM, n)
                      for n in ("footL_geom", "footR_geom")]
        self._renderer = None          # the model changed; drop the old one
        self._rgb_renderer = None
        # The approach slab is identical in all four terrains, so the
        # resting height is too. Measuring it per reset cost 150 physics
        # steps every episode across 16 envs for an answer that never
        # changes; cached on the class instead.
        global _REST_Z
        if _REST_Z is None:
            _REST_Z = self._measure_rest_z()
        self._rest_z = _REST_Z

    def _measure_rest_z(self) -> float:
        """Torso qpos[1] standing on the approach slab. Measured, not assumed.

        Labs 1 and 2 both shipped detectors that compared an offset
        against an absolute height; this is the calibration that makes
        every height question here relative to where the robot actually
        stands.
        """
        probe = self._mj.MjData(self.model)
        self._mj.mj_resetData(self.model, probe)
        probe.qpos[0] = -0.8                    # on the approach, before the event
        self._mj.mj_forward(self.model, probe)
        for _ in range(150):
            self._mj.mj_step(self.model, probe)
        return float(probe.qpos[1])

    # -- observation --------------------------------------------------------

    def _terrain_feats(self) -> np.ndarray:
        """
        The oracle's extra channel. Two numbers, whatever the terrain.

        Keeping the WIDTH fixed across terrain sets is deliberate: the
        one constant in this lab that encoded the length of another
        constant went stale and killed three training runs silently.
        On `patches` the two numbers are distance-to-the-next-slippery
        -patch and its friction, which is exactly what a camera would
        have to infer and what proprioception cannot know in advance.
        """
        one_hot = np.zeros(len(self.kinds))
        one_hot[self.kinds.index(self.kind)] = 1.0
        if self.kind == "patches":
            return np.concatenate([one_hot,
                                   [self.dist_to_patch()], [PATCH_FRICTION]])
        return np.concatenate([one_hot, [self.rise], [self.gap]])

    def dist_to_patch(self) -> float:
        """
        Metres to the leading edge of the next slippery patch.

        Negative while standing on one, and clipped to 3 m so the
        number stays in a sane range before the running normaliser
        sees it. This is the privileged signal, and the whole question
        is whether a camera can replace it.
        """
        x = float(self.data.qpos[0])
        span = PATCH_SPACING + PATCH_LEN
        for i in range(N_PATCHES):
            start = EVENT_X + i * span + PATCH_SPACING
            if x < start:
                return float(min(start - x, 3.0))
            if x < start + PATCH_LEN:
                return float(x - start - PATCH_LEN)      # on it: negative
        return 3.0

    def _obs(self) -> np.ndarray:
        parts = [self.data.qpos[1:], np.clip(self.data.qvel, -10.0, 10.0)]
        if self.privileged:
            parts.append(self._terrain_feats())
        return np.concatenate(parts).astype(np.float64)

    def depth(self) -> np.ndarray:
        """
        Egocentric depth from the head camera, clipped to a FIXED window.

        The window is fixed on purpose. Normalising each frame to its own
        min/max was tried first and destroyed the signal -- the absolute
        distance IS the information (mean depth 1.86 m looking up a
        staircase against 3.03 m looking down one), and per-frame scaling
        throws exactly that away.
        """
        if self._renderer is None:
            self._renderer = self._mj.Renderer(
                self.model, height=self.depth_res, width=self.depth_res)
            self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="eye")
        z = np.clip(self._renderer.render(), DEPTH_NEAR, DEPTH_FAR)
        return ((z - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)).astype(np.float32)

    def rgb(self) -> np.ndarray:
        """
        Egocentric colour, 3xHxW in [0, 1].

        Channels-first so it drops straight into a conv encoder beside
        the depth path. MuJoCo renders colour by DEFAULT and depth only
        when asked, so these are two separate Renderer objects -- the
        same object cannot do both.
        """
        if self._rgb_renderer is None:
            self._rgb_renderer = self._mj.Renderer(
                self.model, height=self.depth_res, width=self.depth_res)
        self._rgb_renderer.update_scene(self.data, camera="eye")
        img = self._rgb_renderer.render().astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def observe_image(self) -> np.ndarray:
        """Whichever sensor this world was built with."""
        return self.rgb() if self.sensor == "rgb" else self.depth()

    # -- state --------------------------------------------------------------

    def torso_height(self) -> float:
        return PLATEAU_Z + 1.0 + self.data.qpos[1]

    def pitch(self) -> float:
        """
        Zero: this robot has no pitch joint.

        The torso is locked upright, which lab 2 measured as the stronger
        configuration. This returned `qpos[2]` at first, copied from a
        body that DID have a `rooty` hinge -- here qpos[2] is `hipL`,
        whose range is -2.62..0.35 rad, so `abs(pitch) > 1.2` fired on an
        ordinary deep hip flexion and reported a crouching robot as
        tipped over. It would have made all four terrains far harder than
        intended, before a single training step.
        """
        return 0.0

    def ground_here(self) -> float:
        """Height of the surface the robot is over, from the terrain spec."""
        from terrain import ground_height
        return ground_height(self.kind, float(self.data.qpos[0]),
                             self.rise, self.gap)

    def fallen(self) -> bool:
        """
        Below the ground it should be standing on, or tipped over.

        Relative to the LOCAL ground, not an absolute height -- otherwise
        descending three steps would register as falling, which is the
        obvious way to make `down` unlearnable by accident.
        """
        # Two conditions, because hollow stairs broke the first one. In a
        # void `ground_here()` returns -3.0, so "clearance" reads ~4.7 and
        # a robot plummeting between treads was scored as perfectly fine
        # until it had fallen three metres.
        #
        # The absolute floor catches that: the lowest a legitimately
        # standing robot sits is 1.38 m (bottom of the steepest descent),
        # so anything under 0.60 m is in a hole.
        if self.torso_height() < 0.60:
            return True
        ground = self.ground_here()
        if ground < -1.0:          # over a void: only the absolute test applies
            return False
        return bool(self.torso_height() - ground < 0.55)

    def progress(self) -> float:
        return float(self.data.qpos[0])

    def past_event(self) -> bool:
        if self.kind == "patches":
            # Must cross EVERY patch. The first version used the generic
            # x > EVENT_X + 1.3, which sits 1.3 m BEFORE the first
            # slippery slab -- so both arms scored 100% without ever
            # touching the hazard the lab is about.
            field = N_PATCHES * (PATCH_SPACING + PATCH_LEN)
            return self.progress() > EVENT_X + field
        if self.kind == "stones":
            # Clearing ONE stone is luck; the claim is about placing
            # every foot. Require the far side of the last stone.
            field = N_STONES * (STONE_TOP + self.gap)
            return self.progress() > EVENT_X + field
        return self.progress() > EVENT_X + 1.3

    # -- the loop -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.kind = self.fixed_kind or str(self.rng.choice(self.kinds))
        self.rise = self.fixed_rise or float(self.rng.uniform(RISE_MIN, RISE_MAX))
        lo, hi = ((STONE_GAP_MIN, STONE_GAP_MAX) if self.kind == "stones"
                  else (GAP_MIN, GAP_MAX))
        self.gap = self.fixed_gap or float(self.rng.uniform(lo, hi))
        self._build()

        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nq)
        self.data.qvel[:] += self.rng.uniform(-5e-3, 5e-3, self.model.nv)
        self._mj.mj_forward(self.model, self.data)
        self.t = 0
        self._crossed = False
        return self._obs(), {"kind": self.kind, "rise": self.rise,
                             "gap": self.gap}

    def step(self, action: np.ndarray) -> tuple:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        x0 = self.data.qpos[0]
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            self._mj.mj_step(self.model, self.data)

        forward = (self.data.qpos[0] - x0) / self.dt
        reward = forward + 1.0 - 1e-3 * float(np.sum(np.square(action)))

        # A one-off bonus for committing across the gap.
        #
        # Forward velocity already pays for crossing, but it pays the same
        # for inching up to the edge -- and the +1.0 alive bonus pays 600
        # for standing still all episode. That makes caution competitive
        # with a risky leap, which is exactly the wrong incentive on the
        # one terrain where commitment is the whole task. Paid ONCE, so
        # loitering on the far side earns nothing extra.
        if self.kind == "hop" and not self._crossed:
            if self.data.qpos[0] > EVENT_X + self.gap + 0.10:
                reward += 10.0
                self._crossed = True

        self.t += 1
        fell = self.fallen()
        return self._obs(), float(reward), bool(fell), \
            self.t >= self.max_steps, {
                "x": self.progress(), "kind": self.kind,
                "past_event": self.past_event(), "fell": fell,
                "clearance": self.torso_height() - self.ground_here(),
                "crossed": self._crossed,
            }


def rollout(env: TerrainWorld, policy, seed: int = 0) -> dict:
    obs, info = env.reset(seed=seed)
    total, steps, max_x = 0.0, 0, -np.inf
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        steps += 1
        max_x = max(max_x, info["x"])
        if term or trunc:
            break
    return {"return": total, "steps": steps, "max_x": float(max_x),
            "kind": info["kind"], "past_event": info["past_event"],
            "fell": info["fell"]}


def random_policy(rng: np.random.Generator, n: int = 6):
    return lambda obs: rng.uniform(-1.0, 1.0, n)


def main() -> None:
    """Random baselines per terrain, before anything is trained."""
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("  Four terrains — the random baseline")
    print("=" * 74)
    print(f"  {'terrain':<8}{'return':>9}{'max x':>8}{'past event':>13}"
          f"{'fell':>7}")
    for kind in KINDS:
        env = TerrainWorld(kind=kind)
        rs = [rollout(env, random_policy(rng), seed=s) for s in range(10)]
        print(f"  {kind:<8}{np.mean([r['return'] for r in rs]):>9.1f}"
              f"{np.mean([r['max_x'] for r in rs]):>8.2f}"
              f"{sum(r['past_event'] for r in rs):>10}/10"
              f"{sum(r['fell'] for r in rs):>5}/10")
    print()
    print(f"  the terrain changes at x={EVENT_X}; 'past event' means the")
    print(f"  robot got {1.3} m beyond it, which needs the right response.")


if __name__ == "__main__":
    main()
