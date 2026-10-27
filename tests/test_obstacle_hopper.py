#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.9", "numpy>=1.26", "mujoco>=3.2"]
# ///
"""
Physics and PPO properties for 07_physical_ai/01_obstacle_hopper.

No training here, and no GPU. Everything below is either exact arithmetic
or a one-second physics probe, which is the only kind of check worth having
on a reinforcement learning lab: a wrong GAE, a reversed clip, or a
decorative obstacle all produce runs that train to *something* and look
entirely normal.

Two of these exist because the lab shipped them broken first.

THE OBSTACLE WAS DECORATIVE
---------------------------
An early version trained to a 100% clear rate at every obstacle height,
which looked like success. Running the same policy with the obstacle
REMOVED gave an identical distance, identical episode length, and
`climbed` was 0/8 throughout -- the robot was diving forward and falling
over at 1 second, and the box never impeded it at all. The clear rate was
real and measured exactly nothing.

`test_the_obstacle_actually_obstructs` is the check that would have caught
it, and it is the direct analogue of `tests/test_omni_eval.py`, which
builds a model that ignores a modality on purpose and requires the harness
to notice.

A SINGLE RUN PROVES NOTHING
---------------------------
The headline experiment was meant to show that a policy blind to the
obstacle height does worse. Across three seeds per arm the means were 955
(seeing) and 987 (blind), while individual seeds ranged 692 to 1119. The
seed spread is an order of magnitude larger than the effect. One run per
arm would have supported either conclusion depending on which seed was
drawn -- and did, twice, in opposite directions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "07_physical_ai" / "01_obstacle_hopper"))
sys.path.insert(0, str(Path(__file__).parent))

from _srcload import Results  # noqa: E402
from obstacle_env import (ACT_DIM, BOX_BACK_X, BOX_FRONT_X,  # noqa: E402
                          BOX_HEIGHT_MAX, BOX_HEIGHT_MIN, OBS_DIM,
                          ObstacleHopper, random_policy, rollout)
from ppo import ActorCritic, clipped_policy_loss, compute_gae  # noqa: E402


# ---------------------------------------------------------------------------
# PPO arithmetic — exact, no simulator
# ---------------------------------------------------------------------------

def test_gae_limits(r: Results) -> None:
    """
    GAE has two exact limits. They are derived, so they must hold exactly.

    lambda=1 is the discounted Monte-Carlo return minus the baseline;
    lambda=0 is the one-step TD residual. An off-by-one in the backward
    recursion, a dropped done-mask, or a flipped sign all still yield
    plausible advantages and a run that learns something -- none of them
    survive these two.
    """
    torch.manual_seed(0)
    T, N, gamma = 8, 3, 0.97
    rewards, values = torch.randn(T, N), torch.randn(T, N)
    dones, last = torch.zeros(T, N), torch.randn(N)

    adv1, ret1 = compute_gae(rewards, values, dones, last, gamma, lam=1.0)
    mc, run = torch.zeros(T, N), last.clone()
    for t in reversed(range(T)):
        run = rewards[t] + gamma * run
        mc[t] = run
    r.check(torch.allclose(adv1, mc - values, atol=1e-5),
            "GAE at lambda=1 is the Monte-Carlo return minus the baseline",
            f"max error {(adv1 - (mc - values)).abs().max():.2e}")

    adv0, _ = compute_gae(rewards, values, dones, last, gamma, lam=0.0)
    td = torch.zeros(T, N)
    for t in range(T):
        nxt = last if t == T - 1 else values[t + 1]
        td[t] = rewards[t] + gamma * nxt - values[t]
    r.check(torch.allclose(adv0, td, atol=1e-6),
            "GAE at lambda=0 is the one-step TD residual",
            f"max error {(adv0 - td).abs().max():.2e}")

    r.check(torch.allclose(ret1, adv1 + values, atol=1e-6),
            "value targets are advantages plus the baseline")


def test_done_mask_stops_bleed_across_episodes(r: Results) -> None:
    """
    A terminal step must cut the recursion. Dropping the mask is silent.

    With a done at t, the advantage at t may not depend on anything after
    it. The check perturbs only the future and requires the past to be
    unchanged -- which fails loudly if either `(1 - done)` factor is lost,
    and fails in no other way a human would notice.
    """
    T, N = 6, 1
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.zeros(T, N)
    dones[2] = 1.0
    a, _ = compute_gae(rewards, values, dones, torch.zeros(N), 0.99, 0.95)

    rewards2 = rewards.clone()
    rewards2[4] = 99.0                      # change the far future only
    b, _ = compute_gae(rewards2, values, dones, torch.zeros(N), 0.99, 0.95)

    r.check(torch.allclose(a[:3], b[:3], atol=1e-6),
            "a terminal step blocks future rewards from leaking backwards",
            "advantages before the done changed when only t=4 moved — the "
            "(1 - done) mask is missing or misapplied")
    r.check(not torch.allclose(a[4:], b[4:], atol=1e-6),
            "...while the future itself did change (the test is not vacuous)")


def test_clipping_is_pessimistic(r: Results) -> None:
    """
    PPO takes the MIN of clipped and unclipped. `max` still descends.

    Beyond the clip boundary, in the direction that would improve the
    objective, the gradient must be exactly zero -- that is the entire trust
    region. A sign slip or a `max` leaves a loss that still goes down while
    the constraint has quietly been removed.
    """
    adv = torch.ones(4)
    logp_old = torch.zeros(4)
    logp = torch.full((4,), 0.7, requires_grad=True)   # ratio ~2.0, clipped
    loss, stats = clipped_policy_loss(logp, logp_old, adv, clip=0.2)
    loss.backward()
    r.check(torch.allclose(logp.grad, torch.zeros(4), atol=1e-8),
            "a positive advantage past the clip ceiling has zero gradient",
            f"grad {logp.grad}")

    logp2 = torch.full((4,), 0.01, requires_grad=True)  # inside the region
    clipped_policy_loss(logp2, logp_old, adv, clip=0.2)[0].backward()
    r.check(logp2.grad.abs().sum() > 0,
            "...while inside the trust region the gradient flows",
            "no gradient anywhere means the check above proves nothing")

    r.check(stats["clip_fraction"] == 1.0,
            "clip_fraction reports the ratios that were actually clipped")


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

def test_the_obstacle_actually_obstructs(r: Results) -> None:
    """
    The box must be a real collision surface, not scenery.

    This is the check the lab needed and did not have. It tests the physics
    directly rather than through a policy: a robot resting over the box must
    sit higher than one resting on open floor, by the height of the box. A
    passive shove test was tried first and could not discriminate, because
    friction stops a limp robot at the same place with or without an
    obstacle.
    """
    FLOOR_REST = 0.849          # torso height resting on open ground

    def settle(x: float, h: float, on_top: bool) -> float:
        """
        Place the robot at x and let it settle under gravity.

        `on_top` lifts it clear of the box surface first. Two earlier
        versions of this helper got that wrong in opposite directions: the
        first left the foot 0.15 m INSIDE the box, so MuJoCo ejected it;
        the second lifted BOTH placements, so the box and the floor cases
        settled identically and the difference was exactly zero. A test
        that reports 0.000 is usually measuring its own setup.
        """
        env = ObstacleHopper(fixed_height=h)
        env.reset(seed=0)
        env.data.qpos[0] = x
        if on_top:
            env.data.qpos[1] = h - 0.151 + 0.002
        env._mj.mj_forward(env.model, env.data)
        for _ in range(100):
            env.step(np.zeros(ACT_DIM))
        return env.torso_height()

    # Only the heights at which a PASSIVE robot rests stably. At 0.17 m it
    # slides off within a second of being placed -- a moving policy clears
    # that height fine, but a static probe cannot measure it, so the tall
    # case is checked by contact instead of by height. Scoping the probe to
    # what it can actually measure beats loosening the tolerance until it
    # passes.
    for h in (0.05, 0.10):
        lift = (settle((BOX_FRONT_X + BOX_BACK_X) / 2, h, on_top=True)
                - settle(-2.0, h, on_top=False))
        r.check(lift > 0.6 * h,
                f"a {h:.2f} m box holds the robot {h:.2f} m higher "
                f"(measured {lift:+.3f} m)",
                "the box is not supporting anything — it is decorative, "
                "and every clear-rate measured against it is meaningless")

    # The tall case: the step must at least be something the robot collides
    # with. Weaker than the lift check, and stated as such.
    tall = ObstacleHopper(fixed_height=BOX_HEIGHT_MAX)
    tall.reset(seed=0)
    tall.data.qpos[0] = (BOX_FRONT_X + BOX_BACK_X) / 2
    tall.data.qpos[1] = BOX_HEIGHT_MAX - 0.151 + 0.002
    tall._mj.mj_forward(tall.model, tall.data)
    touched = any(tall.step(np.zeros(ACT_DIM)) is not None
                  and tall.touching_box() for _ in range(40))
    r.check(touched,
            f"the tallest step ({BOX_HEIGHT_MAX} m) registers real contact",
            "nothing collides with it at all")

    # And the obstacle must be absent when it should be: a sanity check on
    # the control used elsewhere in the lab.
    away = ObstacleHopper(fixed_height=0.10)
    away.reset(seed=0)
    away.data.qpos[0] = -2.0
    away._mj.mj_forward(away.model, away.data)
    for _ in range(40):
        away.step(np.zeros(ACT_DIM))
    r.check(not away.touching_box(),
            "...and a robot nowhere near the box does not touch it",
            "touching_box() is reporting contact that cannot exist, so "
            "every use of it in the lab is suspect")


def test_height_randomises_and_is_observable(r: Results) -> None:
    """The obstacle must vary, and the policy must be able to see it vary."""
    env = ObstacleHopper(seed=3)
    heights = {round(env.reset(seed=s)[1]["box_height"], 4) for s in range(20)}
    r.check(len(heights) >= 15,
            f"the obstacle height is redrawn every episode ({len(heights)}/20 "
            f"distinct)",
            "a fixed obstacle teaches one gait and the lab's premise is gone")
    r.check(all(BOX_HEIGHT_MIN - 1e-9 <= h <= BOX_HEIGHT_MAX + 1e-9
                for h in heights),
            f"...and stays inside the measured-stable range "
            f"[{BOX_HEIGHT_MIN}, {BOX_HEIGHT_MAX}] m")

    seeing = ObstacleHopper(fixed_height=0.13)
    blind = ObstacleHopper(fixed_height=0.13, blind_to_height=True)
    o_s, _ = seeing.reset(seed=1)
    o_b, _ = blind.reset(seed=1)
    r.check(o_s.shape == (OBS_DIM,) and o_b.shape == (OBS_DIM,),
            f"observation is {OBS_DIM} numbers in both arms")
    r.check(abs(o_s[-1] - 0.13) < 1e-9 and abs(o_b[-1]) < 1e-9,
            "blind_to_height zeroes exactly the height input",
            f"seeing {o_s[-1]:.4f}, blind {o_b[-1]:.4f}")
    r.check(np.allclose(o_s[:-1], o_b[:-1], atol=1e-12),
            "...and changes nothing else — a clean ablation",
            "if the two arms differ anywhere but the height input, the "
            "comparison measures something other than perception")


def test_determinism(r: Results) -> None:
    """Same seed, same episode. Otherwise no result here is reproducible."""
    def run(seed: int) -> tuple:
        env = ObstacleHopper(seed=seed)
        obs, info = env.reset(seed=seed)
        trace = [obs.copy()]
        for i in range(40):
            trace.append(env.step(np.full(ACT_DIM, 0.1 * (i % 3 - 1)))[0])
        return np.array(trace), info["box_height"]

    a, ha = run(7)
    b, hb = run(7)
    c, hc = run(8)
    r.check(np.array_equal(a, b) and ha == hb,
            "the same seed reproduces the episode exactly")
    r.check(not np.allclose(a, c, atol=1e-6),
            "a different seed gives a different episode",
            "if seeds do not matter, the seed-variance finding is an artifact")


def test_random_baseline_is_weak(r: Results) -> None:
    """
    The control every number in this lab is quoted against.

    If a random policy already cleared the obstacle there would be nothing
    to learn and the whole lab would be measuring noise. This is the
    `04_video_text/05_video_eval` lesson -- an eval harness there once
    scored a random baseline at 100%, and only running the baseline
    revealed it.
    """
    rng = np.random.default_rng(0)
    env = ObstacleHopper()
    rs = [rollout(env, random_policy(rng), seed=s) for s in range(12)]
    cleared = sum(x["cleared"] for x in rs)
    reach = float(np.mean([x["max_x"] for x in rs]))
    r.check(cleared == 0,
            f"a random policy never clears the step ({cleared}/12)",
            "the task is trivial and the trained numbers mean nothing")
    r.check(reach < BOX_FRONT_X,
            f"...and does not even reach it (mean max x {reach:.2f} m "
            f"vs front face {BOX_FRONT_X:.2f} m)")


def test_the_page_quotes_the_real_source(r: Results) -> None:
    """
    The book page shows code. It must be the code that runs.

    The page was audited from the perspective of a strong software
    engineer with no RL background and failed: 293 lines, one bash block,
    zero XML and zero Python. It explained what was learned without
    showing how any of it was built. The fix was to quote the actual
    reward function, observation and world XML.

    Quoted code rots the instant someone edits the source, and a tutorial
    that shows a reward function the lab does not use is worse than one
    that shows none. These are exact substring checks in both directions.
    """
    page = (REPO / "docusaurus-docs" / "docs" / "tutorials" / "physical"
            / "obstacle-hopper.md").read_text()
    env_src = (REPO / "07_physical_ai" / "01_obstacle_hopper"
               / "obstacle_env.py").read_text()

    for snippet, what in [
        ("reward = forward + 1.0 - 1e-3 * float(np.sum(np.square(action)))",
         "the reward function"),
        ("[BOX_FRONT_X - q[0]],", "the box-relative distance observation"),
        ("np.clip(v, -10.0, 10.0),", "the velocity clip"),
    ]:
        r.check(snippet in page and snippet in env_src,
                f"the page quotes {what} verbatim from obstacle_env.py",
                "page and source disagree — the tutorial is showing code "
                "that does not run")

    # And the builder-facing material must still be there at all.
    for marker, what in [("```xml", "the world as XML"),
                         ("```python", "Python the reader can copy"),
                         ("env.step(action)", "the environment interface")]:
        r.check(marker in page, f"the page shows {what}",
                "a reader who wants to build their own world has nothing "
                "to copy")


def test_the_page_can_actually_be_followed(r: Results) -> None:
    """
    Every command the book page gives a reader must work.

    Audited by cold-cloning the repo and running the page top to bottom.
    Three things were wrong, and the third was fatal:

      * the page opened with `cd 07_physical_ai/01_obstacle_hopper` and
        never said what to clone, or that `uv` is needed at all;
      * `render.py` was absent from the commands even though the page
        shows its output;
      * **the documented happy path crashed.** Train once, then run
        make_figures.py exactly as written, and it died with an
        IndexError because it assumed a `blind*` run existed.

    These check the parts a static reader cannot: that the scripts named
    exist, and that the numbers the page prints match what the code
    prints.
    """
    page = (REPO / "docusaurus-docs" / "docs" / "tutorials" / "physical"
            / "obstacle-hopper.md").read_text()
    lab = REPO / "07_physical_ai" / "01_obstacle_hopper"

    r.check("git clone" in page,
            "the page says how to get the repo",
            "it opens with `cd <folder>` into a repo the reader does not "
            "have")
    r.check("astral.sh/uv/install.sh" in page,
            "...and how to install uv, which is not a standard tool")

    # Every `uv run <script>.py` must name a file that exists.
    import re
    scripts = set(re.findall(r"uv run (?:--no-project python )?"
                             r"([A-Za-z0-9_./]+\.py)", page))
    missing = [x for x in scripts
               if not (lab / x).exists() and not (lab / x).resolve().exists()]
    r.check(not missing,
            f"every script the page tells you to run exists ({len(scripts)})",
            f"missing: {missing}")
    r.check("render.py" in scripts,
            "the page documents render.py",
            "the page shows rendered animations but never says what makes "
            "them")

    # The parameter count is quoted on the page, printed by ppo.py and
    # asserted in this file. All three must agree -- ppo.py's demo once
    # hardcoded the old observation size and printed 10,375 while the page
    # said 10,119, which a reader meets on the second command they run.
    n = ActorCritic(OBS_DIM, ACT_DIM).n_params()
    r.check(f"{n:,}" in page,
            f"the page's parameter count matches the model ({n:,})",
            "the page and the code disagree about the size of the policy")
    # Ask the AST, not the text. The first version of this check searched
    # ppo.py for the string "ActorCritic(13, 3)" and failed on the COMMENT
    # a few lines above that explains why the hardcoding was removed --
    # the exact "a substring is not a fact about the program" trap this
    # repository has a rule about, reproduced inside the check written to
    # enforce it.
    import ast
    hardcoded = [
        node for node in ast.walk(ast.parse((lab / "ppo.py").read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "ActorCritic"
        and node.args and isinstance(node.args[0], ast.Constant)
    ]
    r.check(not hardcoded,
            "ppo.py builds the model from OBS_DIM, not a literal",
            "a hardcoded observation size will print a parameter count "
            "that contradicts the page the moment the observation changes")


def test_make_figures_survives_a_partial_run_set(r: Results) -> None:
    """
    A reader with one trained arm must get a message, not a traceback.

    This is the crash the cold-start audit found. The page says train,
    then make figures; with only the `seeing` arm present that path raised
    IndexError out of `load(names[0])`.
    """
    src = (REPO / "07_physical_ai" / "01_obstacle_hopper"
           / "make_figures.py").read_text()
    r.check("if missing:" in src and "SKIPPED" in src,
            "make_figures reports a missing arm instead of crashing",
            "fig_seed_spread indexes runs_for('blind')[0] unguarded")
    r.check("if not names:" in src,
            "...and the other figures skip an absent arm too")


def test_policy_is_too_small_to_shard(r: Results) -> None:
    """
    Pins the reason this lab carries launcher="python".

    If the policy ever grows to a size where ZeRO is meaningful, this check
    fails and someone has to revisit the claim rather than leaving a stale
    justification in the README.
    """
    n = ActorCritic(OBS_DIM, ACT_DIM).n_params()
    r.check(n < 1_000_000,
            f"the policy is {n:,} parameters — nothing for ZeRO to shard",
            "the launcher=python justification no longer holds; re-read the "
            "DeepSpeed note in the README")


def main() -> int:
    r = Results("Obstacle hopper: physics and PPO properties")
    test_gae_limits(r)
    test_done_mask_stops_bleed_across_episodes(r)
    test_clipping_is_pessimistic(r)
    test_the_obstacle_actually_obstructs(r)
    test_height_randomises_and_is_observable(r)
    test_determinism(r)
    test_random_baseline_is_weak(r)
    test_the_page_quotes_the_real_source(r)
    test_the_page_can_actually_be_followed(r)
    test_make_figures_survives_a_partial_run_set(r)
    test_policy_is_too_small_to_shard(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
