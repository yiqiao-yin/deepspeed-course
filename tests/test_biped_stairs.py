#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.9", "numpy>=1.26", "mujoco>=3.2"]
# ///
"""
The 2x2 must differ in exactly two things, and the staircase must be real.

This lab compares four robots. A comparison is only a comparison if the
cells are identical apart from the switch being studied, and that is the
property most easily destroyed by a well-meaning edit -- nudge a gear
ratio to help the free-torso arm along and the experiment silently stops
measuring anything.

So the checks here are mostly about SAMENESS. Link lengths, gear ratios,
friction, the staircase and the reward must be byte-identical across all
four cells; only the pitch joint and the number of leg chains may vary.

The PPO limits are re-asserted here rather than inherited from
`tests/test_obstacle_hopper.py`. `ppo.py` is copied between the two labs
per the repository's no-shared-module rule, and a copy that is never
checked independently is a copy that can drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "07_physical_ai" / "02_biped_stairs"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(Path(__file__).parent))

from _srcload import Results  # noqa: E402
from morphology import (CELLS, N_STAIRS, RISE_MAX, RISE_MIN,  # noqa: E402
                        STAIR_X0, act_dim, build_xml, obs_dim, stair_tops)
from ppo import ActorCritic, clipped_policy_loss, compute_gae  # noqa: E402
from stairs_env import BipedStairs, random_policy, rollout  # noqa: E402


# ---------------------------------------------------------------------------
# The 2x2 is a controlled comparison
# ---------------------------------------------------------------------------

def test_cells_differ_in_exactly_two_things(r: Results) -> None:
    """
    Only the pitch joint and the leg count may vary between cells.

    Everything a physicist would call a parameter -- link lengths, gear
    ratios, friction, timestep, the staircase -- must be identical, or the
    2x2 is measuring an uncontrolled mixture.
    """
    xmls = {name: build_xml(legs, locked) for name, legs, locked in CELLS}

    for probe, what in [('gear="120"', "hip and knee gearing"),
                        ('gear="70"', "ankle gearing"),
                        ('friction="0.9 0.1 0.1"', "friction"),
                        ('timestep="0.002"', "timestep"),
                        ('fromto="0 0 0 0 0 -0.40"', "thigh length"),
                        ('fromto="0 0 0 0 0 -0.36"', "shin length")]:
        r.check(all(probe in x for x in xmls.values()),
                f"every cell shares {what}",
                "a cell differs in something other than the two switches, "
                "so the comparison is uncontrolled")

    # The staircase must be the same geometry in all four.
    stairs = {n: re.findall(r'<geom name="stair\d"[^/]*/>', x)
              for n, x in xmls.items()}
    first = stairs[CELLS[0][0]]
    r.check(all(v == first for v in stairs.values()),
            f"every cell gets the identical staircase ({len(first)} steps)",
            "the robots are not climbing the same thing")

    # And the switches must actually switch.
    r.check(all(('rooty' in xmls[n]) != locked for n, _, locked in CELLS),
            "locked cells omit the pitch joint, free cells have it",
            "the torso switch does nothing")
    chains = {n: xmls[n].count('type="hinge" axis="0 -1 0" range="-150 20"')
              for n, _, _ in CELLS}
    r.check(all(chains[n] == legs for n, legs, _ in CELLS),
            "leg count matches the number of hip joints emitted",
            f"{chains}")


def test_dimensions_are_derived_not_guessed(r: Results) -> None:
    """obs_dim and act_dim must match what MuJoCo actually builds."""
    import mujoco

    for name, legs, locked in CELLS:
        m = mujoco.MjModel.from_xml_string(build_xml(legs, locked))
        env = BipedStairs(legs=legs, locked_torso=locked)
        o, _ = env.reset(seed=0)
        r.check(o.shape == (obs_dim(legs, locked),),
                f"{name}: observation is {obs_dim(legs, locked)} numbers",
                f"got {o.shape[0]}")
        r.check(m.nu == act_dim(legs),
                f"{name}: {act_dim(legs)} actuators",
                f"MuJoCo built {m.nu}")


# ---------------------------------------------------------------------------
# The physics
# ---------------------------------------------------------------------------

def test_a_locked_torso_cannot_tip(r: Results) -> None:
    """
    The constraint under study must actually constrain.

    If a locked cell could still tip, the two columns of the 2x2 would be
    measuring the same thing and any difference between them would be
    noise with a story attached.
    """
    locked = BipedStairs(legs=2, locked_torso=True)
    locked.reset(seed=0)
    pitches = []
    for _ in range(300):
        locked.step(np.random.uniform(-1, 1, locked.act_dim))
        pitches.append(abs(locked.pitch()))
    r.check(max(pitches) == 0.0,
            "a locked torso never rotates, under any action",
            f"max |pitch| was {max(pitches):.3f} rad")
    r.check(not locked.tipped(), "...so it can never be marked as tipped")

    free = BipedStairs(legs=2, locked_torso=False)
    free.reset(seed=0)
    seen = 0.0
    for _ in range(300):
        free.step(np.random.uniform(-1, 1, free.act_dim))
        seen = max(seen, abs(free.pitch()))
    r.check(seen > 0.05,
            "a free torso does rotate (the test is not vacuous)",
            f"max |pitch| only {seen:.3f} rad — the switch does nothing")


def test_the_staircase_is_real(r: Results) -> None:
    """
    Each tread must hold the robot up, not just be drawn.

    Lab 1 shipped a 100% clear rate against an obstacle the robot never
    touched. The same check, applied per tread: standing over step i must
    put the torso roughly i*rise higher than standing on open floor.

    The probe is short on purpose -- an unactuated robot collapses in
    about two seconds, and at 250 steps lab 1's version of this measured
    nothing but its own setup.
    """
    def settle(x: float, lift: float, rise: float) -> float:
        env = BipedStairs(legs=2, locked_torso=True, fixed_rise=rise)
        env.reset(seed=0)
        env.data.qpos[0] = x
        env.data.qpos[1] = lift
        env._mj.mj_forward(env.model, env.data)
        for _ in range(100):
            env.step(np.zeros(env.act_dim))
        return env.torso_height()

    rise = 0.08
    floor = settle(-2.0, 0.0, rise)
    for i, (x0, x1, top) in enumerate(stair_tops(rise)):
        h = settle((x0 + x1) / 2, top + 0.002, rise)
        r.check(h - floor > 0.5 * top,
                f"tread {i + 1} holds the robot {top:.2f} m up "
                f"(measured {h - floor:+.3f} m)",
                "this tread is decorative, and every climb counted on it "
                "is meaningless")


def test_climbing_needs_height_and_position(r: Results) -> None:
    """
    `steps_climbed` must require both, or it counts the wrong things.

    Position alone counts a robot that slid along the floor past the
    staircase. Height alone counts one mid-leap on open ground. Only the
    conjunction can mean "standing on it".
    """
    env = BipedStairs(legs=2, locked_torso=True, fixed_rise=0.08)
    env.reset(seed=0)

    # "On the floor" is qpos[1] == _rest_z, NOT zero. This test first
    # used 0.0 and failed, because zero is 0.196 m ABOVE resting height
    # for this robot -- the same offset-versus-absolute confusion that
    # produced the bug being tested, reproduced inside the test for it.
    env.data.qpos[0] = 3.0           # well past the stairs...
    env.data.qpos[1] = env._rest_z   # ...and actually on the floor
    env._mj.mj_forward(env.model, env.data)
    r.check(env.steps_climbed() == 0,
            "past the staircase but on the floor counts as zero climbed",
            "horizontal progress alone is being counted as climbing")

    env.data.qpos[0] = -1.0                    # nowhere near...
    env.data.qpos[1] = env._rest_z + 0.30      # ...but high up
    env._mj.mj_forward(env.model, env.data)
    r.check(env.steps_climbed() == 0,
            "high up but before the staircase counts as zero climbed",
            "height alone is being counted as climbing")

    x0, x1, top = stair_tops(0.08)[2]
    env.data.qpos[0] = (x0 + x1) / 2
    env.data.qpos[1] = env._rest_z + top
    env._mj.mj_forward(env.model, env.data)
    r.check(env.steps_climbed() == 3,
            "on the top tread counts all three",
            "the positive case fails, so the two negatives prove nothing")


def test_the_climb_detector_is_exact(r: Results) -> None:
    """
    Place the robot on a known tread; the count must be that tread.

    This is the check the lab needed from the start. The detector took
    FIVE attempts and every wrong version produced a clean, consistent,
    publishable table:

      1. torso height vs an absolute tread height -- ignored the -0.196
         resting offset, so a policy that walked 14.5 m past the whole
         staircase reported 0 climbed;
      2. torso lift above resting -- a robot on a tread with bent legs
         sits lower than one standing straight on the floor;
      3. raw foot height -- the foot capsule's 0.045 m radius already
         clears tread one, so a robot on the floor read 1;
      4. foot lift but TORSO x for the footprint -- the foot runs -0.08
         to +0.14 of the ankle, so a robot squarely on the top step was
         credited with the one below;
      5. `lift > 0.6 * top` per tread -- standing on tread 2 of a 0.06 m
         staircase lifts 0.12, which clears 60% of tread 3's 0.18.

    The working version divides the supporting foot's lift by the rise.
    Three of those five also corrupted the reward, so they cost whole
    training sweeps rather than just a wrong printout.
    """
    for rise in (RISE_MIN, 0.06, RISE_MAX):
        for i, (x0, x1, top) in enumerate(stair_tops(rise)):
            env = BipedStairs(legs=2, locked_torso=True, fixed_rise=rise)
            env.reset(seed=0)
            env.data.qpos[0] = (x0 + x1) / 2
            env.data.qpos[1] = env._rest_z + top
            env._mj.mj_forward(env.model, env.data)
            for _ in range(60):
                env.step(np.zeros(env.act_dim))
            got = env.steps_climbed()
            r.check(got == i + 1,
                    f"rise {rise:.2f}, placed on tread {i + 1} -> counts "
                    f"{i + 1}",
                    f"counted {got}; the detector is off by "
                    f"{got - (i + 1):+d} and the reward pays on it")

    for x, where in ((9.0, "far past the staircase"), (-2.0, "before it")):
        env = BipedStairs(legs=2, locked_torso=True, fixed_rise=0.06)
        env.reset(seed=0)
        env.data.qpos[0] = x
        env.data.qpos[1] = env._rest_z
        env._mj.mj_forward(env.model, env.data)
        for _ in range(40):
            env.step(np.zeros(env.act_dim))
        r.check(env.steps_climbed() == 0,
                f"standing on open floor {where} counts zero",
                f"counted {env.steps_climbed()} — a walking gait's swing "
                f"foot is being read as a climb")


def test_rise_randomises_and_is_observable(r: Results) -> None:
    env = BipedStairs(legs=1, locked_torso=True)
    rises = {round(env.reset(seed=s)[1]["rise"], 5) for s in range(20)}
    r.check(len(rises) >= 15,
            f"the staircase rise is redrawn every episode "
            f"({len(rises)}/20 distinct)")
    r.check(all(RISE_MIN - 1e-9 <= x <= RISE_MAX + 1e-9 for x in rises),
            f"...inside the measured-safe range [{RISE_MIN}, {RISE_MAX}] m")

    env2 = BipedStairs(legs=1, locked_torso=True, fixed_rise=0.07)
    o, _ = env2.reset(seed=0)
    r.check(abs(o[-3] - 0.07) < 1e-9,
            "the rise appears in the observation",
            f"expected 0.07 at index -3, got {o[-3]:.4f}")


def test_determinism(r: Results) -> None:
    def run(seed: int):
        e = BipedStairs(legs=2, locked_torso=False, seed=seed)
        o, info = e.reset(seed=seed)
        tr = [o.copy()]
        for i in range(40):
            tr.append(e.step(np.full(e.act_dim, 0.1 * (i % 3 - 1)))[0])
        return np.array(tr), info["rise"]

    a, ra = run(5)
    b, rb = run(5)
    c, _ = run(6)
    r.check(np.array_equal(a, b) and ra == rb,
            "the same seed reproduces the episode exactly")
    r.check(not np.allclose(a, c, atol=1e-6),
            "a different seed gives a different episode")


def test_random_baseline_never_summits(r: Results) -> None:
    """If chance could climb the stairs there would be nothing to learn."""
    rng = np.random.default_rng(0)
    for name, legs, locked in CELLS:
        env = BipedStairs(legs=legs, locked_torso=locked)
        rs = [rollout(env, random_policy(rng, env.act_dim), seed=s)
              for s in range(8)]
        tops = sum(x["at_top"] for x in rs)
        r.check(tops == 0,
                f"{name}: a random policy never reaches the top ({tops}/8)",
                "the task is trivial and every trained number is noise")


# ---------------------------------------------------------------------------
# PPO, re-asserted rather than inherited
# ---------------------------------------------------------------------------

def test_gae_limits_hold_in_this_copy(r: Results) -> None:
    """
    ppo.py is COPIED from lab 1. A copy nobody checks is a copy that drifts.
    """
    torch.manual_seed(0)
    T, N, g = 7, 2, 0.98
    rew, val = torch.randn(T, N), torch.randn(T, N)
    dones, last = torch.zeros(T, N), torch.randn(N)

    adv1, _ = compute_gae(rew, val, dones, last, g, lam=1.0)
    mc, run = torch.zeros(T, N), last.clone()
    for t in reversed(range(T)):
        run = rew[t] + g * run
        mc[t] = run
    r.check(torch.allclose(adv1, mc - val, atol=1e-5),
            "GAE at lambda=1 is the Monte-Carlo return minus the baseline")

    adv0, _ = compute_gae(rew, val, dones, last, g, lam=0.0)
    td = torch.stack([rew[t] + g * (last if t == T - 1 else val[t + 1]) - val[t]
                      for t in range(T)])
    r.check(torch.allclose(adv0, td, atol=1e-6),
            "GAE at lambda=0 is the one-step TD residual")

    adv = torch.ones(4)
    lp = torch.full((4,), 0.7, requires_grad=True)
    clipped_policy_loss(lp, torch.zeros(4), adv, clip=0.2)[0].backward()
    r.check(torch.allclose(lp.grad, torch.zeros(4), atol=1e-8),
            "the clip still zeroes the gradient past its ceiling")


def test_policies_stay_too_small_to_shard(r: Results) -> None:
    """Pins the launcher=python justification for every cell."""
    for name, legs, locked in CELLS:
        n = ActorCritic(obs_dim(legs, locked), act_dim(legs)).n_params()
        r.check(n < 1_000_000,
                f"{name}: {n:,} parameters — nothing for ZeRO to shard",
                "revisit the DeepSpeed note in the README")


def main() -> int:
    r = Results("Biped stairs: a controlled 2x2 on a real staircase")
    test_cells_differ_in_exactly_two_things(r)
    test_dimensions_are_derived_not_guessed(r)
    test_a_locked_torso_cannot_tip(r)
    test_the_staircase_is_real(r)
    test_climbing_needs_height_and_position(r)
    test_the_climb_detector_is_exact(r)
    test_rise_randomises_and_is_observable(r)
    test_determinism(r)
    test_random_baseline_never_summits(r)
    test_gae_limits_hold_in_this_copy(r)
    test_policies_stay_too_small_to_shard(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
