#!/usr/bin/env python3
"""
How good must perception be before it helps a rover cross 40 m?

    uv run traverse.py --check                 # the planner's properties, CPU
    uv run traverse.py --sweep                 # the accuracy sweep
    uv run traverse.py --router oracle --n 12

THE QUESTION
------------
Lab 7 built a model that reads terrain and scored 52.9% -- better than
blind (p=0.006) and no better than a one-line heuristic. The obvious
next move is to put it in a robot's control loop, and that would be
premature: at 52.9% the likely finding is "coin-flip guidance changes
nothing", which measures the model rather than the question.

So this lab makes perception accuracy the INDEPENDENT VARIABLE. A
synthetic router is correct with probability p, swept from 0.5 to
1.0, and the traverse is measured at each level. That turns lab 7's
52.9% from a blocker into a point on a curve, and asks the question
worth asking: **where is the threshold at which seeing starts to
pay?**

WHY THE ROUTE AND NOT THE GAIT
-------------------------------
Lab 5 established that this robot walks about 6 m of sloped ground
before falling, and that asking for 16-30 m produced six
configurations at exactly 0%. Making the goal farther does not
produce a longer traverse, it produces a longer failure.

So a long traverse is DECOMPOSED: the planner lays waypoints along a
route and the policy walks one short leg at a time. No retraining is
needed, because the policy's goal feature is range and bearing --
retargeting it is just moving the goal. Perception chooses the ROUTE
at landform scale; the learned policy handles the metres.

WHAT EACH ERROR COSTS, AND WHY THEY ARE NOT THE SAME
-----------------------------------------------------
The router decides, per landform, OVER or AROUND. Those are lab 7's
two confusion cells, and here they finally have different prices:

  * a wrong AROUND  -- the planner detours round passable ground.
    Costs DISTANCE.
  * a wrong OVER    -- the planner routes through a scarp. The robot
    walks into it, stalls and falls. Costs an INTERVENTION.

An accuracy figure prices those identically. A traverse does not,
which is the whole reason to measure the traverse.

THE METRIC
----------
Arrival is useless here: at a 63% fall rate per episode nothing
completes, and every arm reads 0%. Mars rover teams track **mean
distance between interventions**, which stays informative when
nothing finishes, so that is the headline alongside distance covered
and route efficiency.

A fall is therefore not the end of an episode. The rover is set
upright where it fell, an intervention is counted, and it continues
-- which is what actually happens on Mars.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"

# Lab 5's generator and environment are REPLAYED here, not copied:
# this lab measures a new question about the same world, and a
# divergent copy of `mars.py` would let the two labs quietly disagree
# about what "traversable" means. The path goes on sys.path at import
# so every entry point -- including `--check`, which touches no robot
# -- can reach it.
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))

STEP_MAX = 0.12          # lab 2's measured gait limit, inherited
WAYPOINT_SPACING = 4.0   # metres between waypoints along the route
WAYPOINT_RADIUS = 1.2    # close enough to count as reached
BLOCKED_AT = 0.25        # lab 7's threshold, same measurement
# The rootz JOINT OFFSET at which the rover stands, not its
# clearance. Those differ by exactly 1.0 and conflating them was a
# real bug here: `torso_height()` is `qpos[rootz] + 1.0`, so setting
# rootz to the healthy clearance (0.78) put the robot at clearance
# 1.78 -- a metre in the air, legs dangling. It never fell (clearance
# was far ABOVE the 0.40 fall threshold, not below it), so nothing
# flagged it; it simply wandered, reached 0 of 7 waypoints and was
# scored as "the terrain is too hard". Lab 5 names this constant
# START_DZ and this is the same number.
START_DZ = -0.294


# ---------------------------------------------------------------------------
# Routers. Each answers one question: is this landform crossable?
# ---------------------------------------------------------------------------
@dataclass
class Router:
    """Base: answers from ground truth. The 100% arm."""
    name: str = "oracle"

    def verdict(self, truth_blocked: bool, rng) -> bool:
        """Return the BELIEF that the landform is blocked."""
        return truth_blocked


@dataclass
class NoisyRouter(Router):
    """
    Correct with probability `p`, wrong otherwise.

    The sweep variable. Note this is symmetric noise -- it degrades
    both classes equally -- which is deliberately NOT what lab 7's
    real model does (that one is biased toward AROUND). The symmetric
    version isolates "how much accuracy is needed" from "which way
    the errors lean"; the biased case is a separate arm.
    """
    p: float = 0.75

    def verdict(self, truth_blocked: bool, rng) -> bool:
        return truth_blocked if rng.random() < self.p else not truth_blocked


@dataclass
class BiasedRouter(Router):
    """
    Lab 7's actual failure mode: says AROUND too often.

    Parameterised by the two per-class accuracies measured there --
    71-72% on blockers, 35-46% on crossable ground. A single accuracy
    number cannot express that, and the traverse may care which way
    the errors lean even when the overall rate is identical.
    """
    p_blocked: float = 0.72      # correct when the truth IS blocked
    p_open: float = 0.40         # correct when the truth is crossable

    def verdict(self, truth_blocked: bool, rng) -> bool:
        p = self.p_blocked if truth_blocked else self.p_open
        return truth_blocked if rng.random() < p else not truth_blocked


@dataclass
class BlindRouter(Router):
    """
    No perception. Believes every landform is crossable.

    This is the honest no-information arm for a TRAVERSE, and it is
    not the same as lab 7's blind arm. A model forced to answer a
    question guesses the prior; a robot with no perception simply
    walks at whatever is in front of it and finds out. That is
    exactly what lab 5's robot did, and it is the behaviour this lab
    is trying to beat.
    """

    def verdict(self, truth_blocked: bool, rng) -> bool:
        return False


# ---------------------------------------------------------------------------
# Planning under a BELIEF, which is the point: the plan is computed
# from what the router thinks, and executed against what is true.
# ---------------------------------------------------------------------------
def belief_mask(s, believed_blocked: list[bool]) -> np.ndarray:
    """
    Traversable-according-to-belief.

    Starts from the real step-limited mask -- the robot's gait is a
    fact, not a belief -- and then additionally blocks the discs of
    landforms the router called impassable. A router that is wrong
    about a landform does not change the terrain; it changes the map
    the planner is given.
    """
    import mars

    ok = mars.traversable(s, STEP_MAX)
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - e / 2
    wy = (ys + 0.5) * cell - e / 2
    for (kind, cx, cy, rad, _h), blocked in zip(s.landforms, believed_blocked):
        if blocked:
            ok &= np.hypot(wx - cx, wy - cy) > rad * 1.15
    return ok


def plan(s, start, goal, ok: np.ndarray):
    """
    8-connected Dijkstra over a given mask. Returns (path_xy, length).

    Written out rather than imported from lab 5's `solve_path`,
    which computes its own mask internally and cannot be given a
    belief. Duplicated by repository convention anyway -- labs do not
    share modules.
    """
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    to_ij = lambda p: (int(np.clip((p[1] + e / 2) / cell, 0, n - 1)),
                       int(np.clip((p[0] + e / 2) / cell, 0, n - 1)))
    a, b = to_ij(start), to_ij(goal)
    if not ok[a] or not ok[b]:
        return None, float("inf")
    R2 = math.sqrt(2.0)
    dist = {a: 0.0}
    prev = {a: None}
    pq = [(0.0, a)]
    while pq:
        dd, cur = heapq.heappop(pq)
        if cur == b:
            cells, c = [], b
            while c is not None:
                cells.append(c)
                c = prev[c]
            cells.reverse()
            xy = [((i + 0.5) * cell - e / 2, (j + 0.5) * cell - e / 2)
                  for j, i in cells]
            return xy, dd * cell
        if dd > dist.get(cur, math.inf):
            continue
        j, i = cur
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < n and 0 <= ni < n) or not ok[nj, ni]:
                continue
            w = R2 if (dj and di) else 1.0
            nd = dd + w
            if nd < dist.get((nj, ni), math.inf):
                dist[(nj, ni)] = nd
                prev[(nj, ni)] = cur
                heapq.heappush(pq, (nd, (nj, ni)))
    return None, float("inf")


def waypoints(path_xy, spacing: float = WAYPOINT_SPACING):
    """Thin a dense cell path down to legs the policy can actually walk."""
    if not path_xy:
        return []
    out = [path_xy[0]]
    acc = 0.0
    for a, b in zip(path_xy, path_xy[1:]):
        acc += math.dist(a, b)
        if acc >= spacing:
            out.append(b)
            acc = 0.0
    if out[-1] != path_xy[-1]:
        out.append(path_xy[-1])
    return out[1:]          # the first point is where the robot already is


def truth_blocked(s) -> list[bool]:
    """Ground truth per landform, by lab 7's edge-ring measurement."""
    import mars

    ok = mars.traversable(s, STEP_MAX)
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - e / 2
    wy = (ys + 0.5) * cell - e / 2
    out = []
    for kind, cx, cy, rad, _h in s.landforms:
        d = np.hypot(wx - cx, wy - cy)
        ring = (d > rad * 0.75) & (d < rad * 1.25)
        out.append(bool(ring.sum() >= 8 and (~ok[ring]).mean() >= BLOCKED_AT))
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@dataclass
class Result:
    router: str
    seed: int
    arrived: bool = False
    interventions: int = 0
    distance: float = 0.0
    planned_len: float = float("inf")
    oracle_len: float = float("inf")
    steps: int = 0
    stalled: bool = False
    wp_total: int = 0
    wp_reached: int = 0

    @property
    def mdbi(self) -> float:
        """Mean distance between interventions — the rover metric."""
        return self.distance / max(self.interventions, 1)

    @property
    def efficiency(self) -> float:
        """
        Route efficiency, DEFINED ONLY ON ARRIVAL.

        oracle_len / distance is nonsense for a run that stopped
        early: a rover that stalls after 2 m of a 30 m route scores
        an "efficiency" of 15. The first version of this file
        reported exactly that -- 4.85 for an arm that arrived 0% of
        the time -- which is a plausible-looking number measuring
        nothing. Lab 5's renderer freezes efficiency at arrival for
        the same reason.
        """
        if not self.arrived or not math.isfinite(self.oracle_len) \
                or self.distance < 0.1:
            return float("nan")
        return self.oracle_len / self.distance


def run_traverse(env, net, norm, s, start, goal, router, rng,
                 max_steps: int = 12000, stall_steps: int = 600) -> Result:
    """
    Walk a planned route leg by leg, recovering from falls.

    A fall does NOT end the run. The rover is stood back up where it
    fell, an intervention is counted, and it continues -- which is
    what happens on Mars, and the only way a 40 m traverse is
    measurable at all with a policy that falls every ~6 m.
    """
    import torch

    truth = truth_blocked(s)
    belief = [router.verdict(t, rng) for t in truth]
    ok_belief = belief_mask(s, belief)
    path, planned_len = plan(s, start, goal, ok_belief)

    import mars
    _, oracle_len = plan(s, start, goal, mars.traversable(s, STEP_MAX))

    r = Result(router=router.name, seed=env._seed if hasattr(env, "_seed") else 0,
               planned_len=planned_len, oracle_len=oracle_len)
    if path is None:
        # The belief made the goal unreachable. A real outcome -- a
        # router that calls everything impassable walls itself in --
        # and recorded as such rather than silently scored 0 m.
        r.stalled = True
        return r

    wps = waypoints(path)
    r.wp_total = len(wps)
    obs = env._obs()
    prev = env.pos().copy()
    since_progress, best_d = 0, float("inf")

    for wp in wps:
        env.map.goal = (float(wp[0]), float(wp[1]))
        best_d, since_progress = float("inf"), 0
        while r.steps < max_steps:
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, _term, _trunc, _info = env.step(act)
            r.steps += 1
            now = env.pos()
            r.distance += float(np.linalg.norm(now - prev))
            prev = now.copy()

            d = env.to_goal()
            if d < best_d - 0.05:
                best_d, since_progress = d, 0
            else:
                since_progress += 1

            if env.fallen():
                recover(env)
                r.interventions += 1
                obs = env._obs()

            if d < WAYPOINT_RADIUS:
                r.wp_reached += 1
                break
            if since_progress > stall_steps:
                # Wedged against something the plan said was passable.
                # This is what a wrong OVER costs, and it ends the run.
                r.stalled = True
                return r
        if r.steps >= max_steps:
            break

    r.arrived = bool(np.linalg.norm(np.array(goal) - env.pos()) < 1.5)
    return r


def recover(env) -> None:
    """
    Stand the rover back up where it fell. One intervention.

    Position and heading are kept; height, tilt and all velocities are
    reset. This is a deliberately GENEROUS model of a recovery -- a
    real rover would not always get up -- and the lab says so rather
    than implying the robot is more robust than it is.
    """
    q = env.q
    env.data.qpos[q["rootz"]] = env._h_at(*env.pos()) + START_DZ
    for j in ("rootpitch", "rootroll"):
        if j in q:
            env.data.qpos[q[j]] = 0.0
    env.data.qvel[:] = 0.0
    env._mj.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# Tasks: LONG routes, which is the point of this lab
# ---------------------------------------------------------------------------
def make_long_task(seed: int, scale: str = "course",
                   want: tuple = (25.0, 50.0), tries: int = 60,
                   relief: float = 1.0):
    """
    A long, solvable A->B on a generated surface.

    SCALE IS `course`, NOT `rover`, and that was measured rather than
    assumed. Over 20 surfaces in the 25-50 m band:

        rover  (60 m arena)  37.7 m routes, 0.7 of 15 landforms blocked
        course (34 m arena)  29.7 m routes, 2.5 of 10 landforms blocked

    `rover` gives the longer route and almost nothing to decide -- a
    perception sweep there would measure noise, because 0.7 blockers
    per surface means most arms plan the identical path. That is the
    geometry lesson both navigation labs had to learn separately: what
    makes terrain a DECISION is the ratio of route to arena, not the
    absolute distance. `course` is also lab 7's scale, so "blocked"
    means the same thing in both labs.

    `want` is far beyond anything lab 5 asked for, and deliberately:
    lab 5 measured that 16-30 m end-to-end gave six configurations at
    exactly 0%, which is why this lab decomposes the route instead of
    handing it to the policy whole.

    Endpoints must be STANDABLE, not merely traversable. Lab 5 lost a
    whole sweep to that distinction -- spawned on a tilt or straddling
    a boulder, the robot topples before it can act, and all nine arms
    read 0% with 33-75% fall rates.
    """
    import mars

    s = mars.generate(seed, scale=scale)
    # Relief is applied HERE, before anything is planned or measured,
    # so the planner, the labels and MuJoCo all see one array.
    if relief != 1.0:
        s.heights = s.heights * relief
    ok = mars.traversable(s, STEP_MAX)
    rng = np.random.default_rng(seed)
    e = s.extent
    for _ in range(tries):
        a = (rng.uniform(-e / 2 + 3, e / 2 - 3), rng.uniform(-e / 2 + 3, e / 2 - 3))
        b = (rng.uniform(-e / 2 + 3, e / 2 - 3), rng.uniform(-e / 2 + 3, e / 2 - 3))
        if not (mars.standable(s, *a) and mars.standable(s, *b)):
            continue
        _, L = plan(s, a, b, ok)
        if math.isfinite(L) and want[0] <= L <= want[1]:
            return s, a, b, L
    return None


def load_policy(run: str):
    """Lab 5's trained policy, replayed. Nothing is trained in this lab."""
    import torch

    sys.path.insert(0, str(LAB5))
    from ppo import ActorCritic, RunningNorm

    meta = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
    ck = torch.load(LAB5 / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta.get("act_dim", 9),
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm, meta


def build_env(meta):
    """
    The environment, with relief_scale FIXED AT 1.0.

    Relief is applied to the surface BEFORE planning instead (see
    `make_long_task`), so that the heights the planner reasons about
    and the heights MuJoCo collides against are the same array.
    Letting the env scale them would mean planning on one surface and
    walking on another.
    """
    from mars_env import NavWorld

    return NavWorld(mode=meta["mode"], flat=False, seed=0,
                    goal_range=meta.get("goal_range"),
                    relief_scale=1.0,
                    upright_w=meta.get("upright_w", 0.0))


def place(env, s, start) -> None:
    """
    Build the physics on OUR surface and stand the rover at A.

    THE BUG THIS EXISTS TO PREVENT, which this lab shipped once:
    `reset()` draws a task from lab 5's fixed pool and calls `_build`
    on it, so assigning `env.map = s` afterwards changes only a Python
    reference. The robot then walks pool terrain while the planner
    reasons about a different surface entirely -- and it still
    produces plausible numbers (an "oracle" arm arriving 100% on flat
    ground). Lab 4 shipped the same class of error with a renderer
    that filmed a task nobody measured.

    So the surface is handed to `_build` directly, and the result is
    ASSERTED to match what we planned on.
    """
    env._build(s)
    env._mj.mj_resetData(env.model, env.data)

    # The frames must agree. Compared after removing the constant
    # offset, because `_build` shifts the field so its minimum sits at
    # z=0 and that offset cannot change traversability.
    mine = s.heights - float(s.heights.min())
    theirs = env._heights - float(env._heights.min())
    assert np.allclose(mine, theirs, atol=1e-6), (
        "the physics terrain is not the surface we planned on: "
        f"max |diff| {np.abs(mine - theirs).max():.4f} m")

    env.data.qpos[env.q["rootx"]] = start[0]
    env.data.qpos[env.q["rooty"]] = start[1]
    env.data.qpos[env.q["rootyaw"]] = 0.0
    b = 0.35
    bend = {"hiproll": 0.0, "hip": -b, "knee": -1.8 * b, "ankle": 0.5 * b}
    for adr, nm in zip(env.leg_q, env.leg_names):
        key = next(k for k in ("hiproll", "hip", "knee", "ankle")
                   if nm.startswith(k))
        env.data.qpos[adr] = bend[key]
    recover(env)
    env.t = 0
    env._prev_action = np.zeros(env.act_dim)
    env.travelled = 0.0
    env._arrived = False
    env._travelled_at_arrival = None
    env._prev_d = env.to_goal()


ROUTERS = {
    "oracle": lambda: Router("oracle"),
    "blind": lambda: BlindRouter("blind"),
    "biased": lambda: BiasedRouter("biased(lab7)"),
    **{f"p{int(p*100)}": (lambda p=p: NoisyRouter(f"p={p:.2f}", p))
       for p in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)},
}


def check() -> int:
    """
    Does the machinery behave, before any robot is involved?

    Every check here guards a way this lab could produce a clean
    curve that means nothing.
    """
    import mars

    fails = []

    def ck(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 74)
    print("  LAB 8 — the planner and the routers, before any robot")
    print("=" * 74)

    task = make_long_task(3)
    ck("a long task exists", task is not None)
    if task is None:
        return 1
    s, a, b, L = task
    ck("the route is actually long", 25.0 <= L <= 50.0, f"{L:.1f} m")

    truth = truth_blocked(s)
    ck("the surface has blockers and open landforms",
       any(truth) and not all(truth),
       f"{sum(truth)} blocked of {len(truth)}")

    # An oracle plan must never be longer than a belief-constrained one:
    # extra blocked discs can only remove options.
    ok_true = mars.traversable(s, STEP_MAX)
    _, oracle_len = plan(s, a, b, ok_true)
    _, all_blocked = plan(s, a, b, belief_mask(s, [True] * len(truth)))
    ck("believing everything impassable never helps",
       not math.isfinite(all_blocked) or all_blocked >= oracle_len - 1e-6,
       f"oracle {oracle_len:.1f} m vs paranoid {all_blocked:.1f} m")

    # The router contract, watched in both directions.
    rng = np.random.default_rng(0)
    ck("the oracle router is always right",
       all(Router().verdict(t, rng) == t for t in (True, False)))
    ck("the blind router believes nothing is blocked",
       BlindRouter().verdict(True, rng) is False
       and BlindRouter().verdict(False, rng) is False)
    n_trial = 4000
    nr = NoisyRouter("p", 0.75)
    hits = sum(nr.verdict(True, rng) is True for _ in range(n_trial)) / n_trial
    ck("a p=0.75 router is right about 75% of the time",
       0.72 < hits < 0.78, f"measured {hits:.3f}")
    # p=0.5 must be genuinely uninformative, not accidentally biased.
    nr5 = NoisyRouter("p", 0.50)
    h5 = sum(nr5.verdict(True, rng) is True for _ in range(n_trial)) / n_trial
    ck("a p=0.50 router carries no information",
       0.47 < h5 < 0.53, f"measured {h5:.3f}")

    # The biased router must reproduce lab 7's lopsided behaviour.
    br = BiasedRouter("b")
    on_blocked = sum(br.verdict(True, rng) for _ in range(n_trial)) / n_trial
    on_open = sum(not br.verdict(False, rng) for _ in range(n_trial)) / n_trial
    ck("the biased router is lopsided, like lab 7's model",
       on_blocked > on_open + 0.2,
       f"{on_blocked:.2f} on blockers vs {on_open:.2f} on open ground")

    # Waypoints must decompose into legs the policy can actually walk.
    path, _ = plan(s, a, b, ok_true)
    wps = waypoints(path)
    legs = [math.dist(p, q) for p, q in zip([a] + wps, wps)]
    ck("waypoints decompose the route into short legs",
       wps and max(legs) <= WAYPOINT_SPACING * 1.6,
       f"{len(wps)} legs, longest {max(legs):.1f} m")
    ck("the last waypoint is the goal",
       math.dist(wps[-1], b) < 1.0, f"{math.dist(wps[-1], b):.2f} m off")

    print("=" * 74)
    if fails:
        print(f"  {len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("  the machinery is sound")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="planner and router properties; no robot, no GPU")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--router", default="oracle", choices=sorted(ROUTERS))
    ap.add_argument("--run", default="v3_flat_blind",
                    help="which lab 5 policy to replay")
    ap.add_argument("--n", type=int, default=12, help="surfaces per arm")
    ap.add_argument("--seed0", type=int, default=3)
    ap.add_argument("--scale", default="course")
    ap.add_argument("--relief", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--json", default="runs/sweep.json")
    a, _ = ap.parse_known_args()

    if a.check:
        return check()

    net, norm, meta = load_policy(a.run)
    arms = (["oracle", "p90", "p80", "p70", "p60", "p50", "biased", "blind"]
            if a.sweep else [a.router])

    print("=" * 74)
    print(f"  LONG TRAVERSE — {a.scale}, relief {a.relief:g}, "
          f"policy {a.run}")
    print("=" * 74)
    print(f"  {'arm':<14} {'arrived':>8} {'dist':>8} {'MDBI':>8} "
          f"{'interv':>7} {'eff':>6} {'stalled':>8}")

    out = {"policy": a.run, "scale": a.scale, "relief": a.relief,
           "n": a.n, "arms": {}}
    for name in arms:
        router = ROUTERS[name]()
        rows = []
        for k in range(a.n):
            seed = a.seed0 + k
            task = make_long_task(seed, a.scale, relief=a.relief)
            if task is None:
                continue
            s, A, B, L = task
            s.start, s.goal = A, B
            env = build_env(meta)
            place(env, s, A)
            rng = np.random.default_rng(10_000 + seed)
            rows.append(run_traverse(env, net, norm, s, A, B, router, rng,
                                     max_steps=a.max_steps))
        if not rows:
            print(f"  {name:<14}  no solvable tasks")
            continue
        arr = np.mean([r.arrived for r in rows])
        dist = np.mean([r.distance for r in rows])
        # POOLED, not the mean of per-run ratios. A run that happens
        # to fall once over 9 m reports MDBI 9 and drags the average
        # up; total distance over total interventions is what a rover
        # team means by the term.
        tot_d = float(sum(r.distance for r in rows))
        tot_i = int(sum(r.interventions for r in rows))
        mdbi = tot_d / max(tot_i, 1)
        iv = np.mean([r.interventions for r in rows])
        effs = [r.efficiency for r in rows if not math.isnan(r.efficiency)]
        eff = float(np.mean(effs)) if effs else float("nan")
        st = np.mean([r.stalled for r in rows])
        print(f"  {name:<14} {arr:>7.0%} {dist:>7.1f}m {mdbi:>7.1f}m "
              f"{iv:>7.1f} {eff:>6.2f} {st:>7.0%}")
        out["arms"][name] = {
            "arrived": float(arr), "distance": float(dist),
            "mdbi": float(mdbi), "interventions": float(iv),
            "efficiency": float(eff), "stalled": float(st),
            "total_distance": tot_d, "total_interventions": tot_i,
            "n": len(rows),
            "per_seed": [r.__dict__ for r in rows],
        }

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
