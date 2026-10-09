#!/usr/bin/env python3
"""
A 20 x 20 m terrain grid, with obstacles of two kinds and a route between.

    uv run world.py              # generate, solve, and describe one map
    uv run world.py --show 5     # five maps as ASCII, with their routes

THE POINT OF THE DESIGN
-----------------------
Labs 1-3 were PLANAR: the robot had `rootx` and `rootz` and nothing else,
so "the terrain" was a profile along a line and every obstacle had to be
dealt with head-on. There was never a choice to make.

Here the ground is a height field over a square, and obstacles come in
two kinds that look similar from a distance and demand opposite
responses:

    STEPS      rise <= STEP_MAX. Climbable. Going around costs distance.
    WALLS      rise >  STEP_MAX. Not climbable at any gait. Going around
               is the ONLY option, and trying to climb wastes the episode.

That distinction is the whole lab. A robot that treats everything as
climbable gets stuck on walls; one that treats everything as impassable
takes the long way round every step and is slow. Telling them apart is
what a camera is for, and unlike labs 1-3 there is a cheap, unambiguous
measure of success: did it reach B, and how far did it walk to get there.

WHY A HEIGHT FIELD RATHER THAN BOXES
------------------------------------
MuJoCo's `hfield` is one elevation array sampled over a rectangle. It
gives a continuous surface with no seams for a foot to catch on, it
costs one texture rather than hundreds of geoms, and -- the part that
matters for the experiment -- the SAME array is what the planner reads
and what the depth camera photographs. There is no second description
of the world that can drift from the first.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

# Metres. The arena is square and the robot starts somewhere in it.
ARENA = 20.0
# Height-field resolution. 0.1 m per cell: fine enough that a 0.12 m step
# is several cells wide, coarse enough that the array stays small.
CELL = 0.10
N = int(ARENA / CELL)                       # 200 x 200 samples

# The threshold the whole lab turns on. Lab 2 measured a two-legged robot
# climbing 0.04-0.10 m treads reliably and failing above that, so the
# climbable/impassable boundary is placed where that lab's evidence puts
# it rather than where it would be convenient.
STEP_MAX = 0.12
WALL_MIN = 0.45                             # unambiguously not climbable

MAX_H = 1.2                                 # hfield elevations scale to this

# Climbable ridges are drawn in one colour and impassable ones in
# another. This is PURELY for the reader: the height field is a single
# greyscale elevation array and carries no colour at all, the policy
# observes heights rather than pixels, and a depth camera could not see
# a colour even if one existed. Lab 3 had to be careful here -- colour
# -coding terrain there would have taught its CNN "orange means up" --
# but nothing in this lab can read it.
STEP_RGBA = "0.33 0.49 0.38 1"              # climbable
WALL_RGBA = "0.46 0.26 0.26 1"              # not climbable


@dataclass
class Obstacle:
    """
    An elongated RIDGE, not a mound.

    Round mounds were tried first and made the climbable half of this
    lab vacuous: a route can slip past a 2 m mound for nothing, so
    refusing to climb cost a median of 0.00 m over 109 maps and the
    policy never had a decision to make. A ridge BLOCKS the line it
    crosses, which is what turns "is this climbable?" into a question
    worth answering -- climb it and save the detour, or walk to its end.
    """
    kind: str                               # "step" or "wall"
    cx: float
    cy: float
    length: float                           # along its own axis
    width: float                            # across it
    angle: float                            # radians
    height: float


@dataclass
class Map:
    heights: np.ndarray                     # (N, N) metres
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: list[Obstacle] = field(default_factory=list)
    seed: int = 0

    def h_at(self, x: float, y: float) -> float:
        """Elevation at a world point, nearest sample."""
        i = int(np.clip((x + ARENA / 2) / CELL, 0, N - 1))
        j = int(np.clip((y + ARENA / 2) / CELL, 0, N - 1))
        return float(self.heights[j, i])


def _spine(cx: float, cy: float, length: float, angle: float,
           n: int = 12) -> np.ndarray:
    """Points along a ridge's centre line, for overlap testing."""
    t = np.linspace(-length / 2, length / 2, n)
    return np.stack([cx + t * np.cos(angle), cy + t * np.sin(angle)], axis=1)


def generate(seed: int, n_steps: int = 8, n_walls: int = 6) -> Map:
    """
    One random arena: long climbable ridges, long impassable ones.

    Every constant here was chosen by SWEEPING it against the property
    the lab needs, not by taste, because the first two attempts produced
    a task with no decision in it:

      - round mounds (r = 1.2-2.2 m): refusing to climb cost a median
        of 0.00 m over 109 maps. A route slips past a mound for free.
      - short ridges (5-8 m): still 0.00 m. Walking round the end of a
        6 m ridge in a 20 m arena is nearly free.
      - LONG ridges (12-17 m): refusing to climb costs a median of
        3.40 m on a ~15 m route, and 96% of maps stay solvable.

    Length is the lever. A ridge that nearly spans the arena cannot be
    skirted cheaply, so "is this one climbable?" becomes a question
    worth answering. Ends are left open so going around stays POSSIBLE
    -- on 94% of maps both routes exist, which keeps it a choice rather
    than a forced climb.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros((N, N), dtype=np.float32)
    ys, xs = np.mgrid[0:N, 0:N]
    wx = (xs + 0.5) * CELL - ARENA / 2
    wy = (ys + 0.5) * CELL - ARENA / 2

    obstacles: list[Obstacle] = []

    def place(kind: str, height: float, length: float, width: float) -> None:
        """
        Reject only on ACTUAL overlap, sampled along both ridges.

        The first version compared centre distance against the sum of
        LENGTHS, as though the ridges were discs. Two 7 m ridges then
        had to sit 8 m apart whatever their orientation, so only about
        five ever fitted and asking for twenty-two placed 5.2 -- a
        density sweep that varied nothing. Ridges are thin; two
        parallel ones can sit two metres apart.
        """
        for _ in range(200):
            cx, cy = rng.uniform(-ARENA / 2 + 2.5, ARENA / 2 - 2.5, size=2)
            ang = float(rng.uniform(0, np.pi))
            pts = _spine(cx, cy, length, ang)
            clear = True
            for o in obstacles:
                opts = _spine(o.cx, o.cy, o.length, o.angle)
                d2 = ((pts[:, None, 0] - opts[None, :, 0]) ** 2
                      + (pts[:, None, 1] - opts[None, :, 1]) ** 2).min()
                if d2 < ((width + o.width) / 2 + 1.1) ** 2:
                    clear = False
                    break
            if clear:
                obstacles.append(Obstacle(kind, cx, cy, length, width, ang,
                                          height))
                return

    for _ in range(n_steps):
        place("step", float(rng.uniform(0.06, STEP_MAX)),
              float(rng.uniform(12.0, 17.0)), float(rng.uniform(1.0, 1.8)))
    for _ in range(n_walls):
        place("wall", float(rng.uniform(WALL_MIN, 0.9)),
              float(rng.uniform(6.0, 10.0)), float(rng.uniform(0.8, 1.4)))

    for o in obstacles:
        # Rotate the sample grid into the ridge's own frame.
        ca, sa = np.cos(-o.angle), np.sin(-o.angle)
        dx, dy = wx - o.cx, wy - o.cy
        u = dx * ca - dy * sa                   # along the ridge
        v = dx * sa + dy * ca                   # across it
        inside = np.abs(u) < o.length / 2
        if o.kind == "step":
            # A ramp up each side and a flat top: genuinely walkable,
            # not a cliff that happens to be short.
            prof = np.clip((o.width / 2 - np.abs(v)) / 0.6, 0.0, 1.0)
            h = np.maximum(h, inside * prof * o.height)
        else:
            prof = (np.abs(v) < o.width / 2).astype(np.float32)
            h = np.maximum(h, inside * prof * o.height)

    start, goal = _endpoints(rng, h)
    return Map(heights=h, start=start, goal=goal, obstacles=obstacles,
               seed=seed)


def _endpoints(rng, h: np.ndarray) -> tuple[tuple, tuple]:
    """Two flat, well-separated points. A start inside a wall is not a task."""
    def flat_point():
        for _ in range(500):
            x, y = rng.uniform(-ARENA / 2 + 1.5, ARENA / 2 - 1.5, size=2)
            i = int((x + ARENA / 2) / CELL)
            j = int((y + ARENA / 2) / CELL)
            if h[j, i] < 0.02:
                return float(x), float(y)
        return 0.0, 0.0

    a = flat_point()
    for _ in range(200):
        b = flat_point()
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > 9.0 ** 2:
            return a, b
    return a, b


def traversable(h: np.ndarray) -> np.ndarray:
    """
    Cells a legged robot could stand on, given the local slope.

    A cell is blocked when the step UP to any 4-neighbour exceeds
    STEP_MAX. That is a statement about the gait, not about absolute
    height: the top of a climbable mound is perfectly walkable once you
    are on it, and only the rise between cells can stop you.
    """
    ok = np.ones_like(h, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
        ok &= (nb - h) <= STEP_MAX
    return ok


def solve(m: Map) -> tuple[list[tuple[int, int]] | None, float]:
    """
    Shortest traversable route from A to B, by breadth-first search.

    This is the ORACLE, and it exists for two reasons. It certifies that
    a map is solvable at all -- an unsolvable map would score every
    policy at zero and look like a hard task rather than a broken one,
    which this repository has shipped twice. And its path length is the
    denominator for efficiency: a policy that reaches B having walked
    three times the necessary distance has not solved the task in the
    way the lab cares about.
    """
    from collections import deque

    ok = traversable(m.heights)
    to_ij = lambda p: (int((p[1] + ARENA / 2) / CELL),
                       int((p[0] + ARENA / 2) / CELL))
    s, g = to_ij(m.start), to_ij(m.goal)
    if not ok[s] or not ok[g]:
        return None, 0.0

    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            break
        j, i = cur
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (j + dj, i + di)
            if (0 <= nxt[0] < N and 0 <= nxt[1] < N
                    and nxt not in prev and ok[nxt]):
                prev[nxt] = cur
                q.append(nxt)
    if g not in prev:
        return None, 0.0

    path, cur = [], g
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, len(path) * CELL


def geodesic(m: Map) -> float:
    """
    The shortest traversable DISTANCE from A to B. The denominator for
    path efficiency, and deliberately not `solve()`.

    `solve()` is a 4-connected breadth-first search. It is the right
    tool for its two jobs -- certifying a map is solvable, and walking B
    back along a route to set the goal range -- and it must not change,
    because the goal placement it produces defines the task that every
    trained policy was trained on.

    But a 4-connected path CANNOT move diagonally. It staircases, so a
    straight diagonal of length L comes back as L * sqrt(2). Using that
    as the optimum inflates the denominator by up to 41%, and the
    efficiencies this lab computed went over 100% because of it -- 133%,
    136%, 144% on seeds 20016, 20011, 20047, against a sqrt(2) ceiling
    of 141%. A policy cannot beat the optimum; the optimum was wrong.

    Worse, `evaluate.py` wrapped the ratio in `min(..., 1.0)`, which
    turned every one of those into "exactly 100%" -- the clamp converted
    a broken denominator into a plausible number and hid the evidence.
    That is the failure mode this repository keeps re-learning: the
    quantity looked fine precisely where it was most wrong.

    So: Dijkstra, eight neighbours, sqrt(2) for the diagonals. A
    diagonal move is only legal when BOTH orthogonal cells beside it are
    traversable, otherwise a route could squeeze through the corner
    where two walls touch -- a gap of zero width that a robot with a
    body cannot use.
    """
    import heapq

    ok = traversable(m.heights)
    to_ij = lambda p: (int((p[1] + ARENA / 2) / CELL),
                       int((p[0] + ARENA / 2) / CELL))
    s, g = to_ij(m.start), to_ij(m.goal)
    if not ok[s] or not ok[g]:
        return 0.0

    R2 = math.sqrt(2.0)
    dist = {s: 0.0}
    pq = [(0.0, s)]
    while pq:
        d, cur = heapq.heappop(pq)
        if cur == g:
            return d * CELL
        if d > dist.get(cur, math.inf):
            continue
        j, i = cur
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < N and 0 <= ni < N) or not ok[nj, ni]:
                continue
            if dj and di and not (ok[j, ni] and ok[nj, i]):
                continue                      # no corner-squeezing
            w = R2 if (dj and di) else 1.0
            nd = d + w
            if nd < dist.get((nj, ni), math.inf):
                dist[(nj, ni)] = nd
                heapq.heappush(pq, (nd, (nj, ni)))
    return 0.0


def straight_line(m: Map) -> float:
    return float(np.hypot(m.goal[0] - m.start[0], m.goal[1] - m.start[1]))


def ascii_map(m: Map, path=None) -> str:
    """A coarse picture, for reading a map without opening a renderer."""
    step = N // 40
    rows = []
    pathset = {(j // step, i // step) for j, i in (path or [])}
    for jj in range(40):
        row = ""
        for ii in range(40):
            j, i = jj * step, ii * step
            hh = m.heights[j, i]
            if (jj, ii) in pathset:
                row += "."
            elif hh >= WALL_MIN:
                row += "#"
            elif hh > 0.02:
                row += "^"
            else:
                row += " "
        rows.append(row)
    sj, si = (int((m.start[1] + ARENA/2)/CELL)//step,
              int((m.start[0] + ARENA/2)/CELL)//step)
    gj, gi = (int((m.goal[1] + ARENA/2)/CELL)//step,
              int((m.goal[0] + ARENA/2)/CELL)//step)
    rows[sj] = rows[sj][:si] + "A" + rows[sj][si+1:]
    rows[gj] = rows[gj][:gi] + "B" + rows[gj][gi+1:]
    return "\n".join("  |" + r + "|" for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=0,
                    help="generate N maps and report how many are solvable")
    a, _ = ap.parse_known_args()

    if a.check:
        solvable = detours = 0
        ratios = []
        for s in range(a.check):
            m = generate(s)
            p, length = solve(m)
            if p:
                solvable += 1
                r = length / max(straight_line(m), 1e-6)
                ratios.append(r)
                if r > 1.15:
                    detours += 1
        print(f"  {solvable}/{a.check} maps solvable")
        print(f"  {detours}/{solvable} need a real detour (>15% over straight line)")
        print(f"  path/straight ratio: median {np.median(ratios):.2f}  "
              f"max {max(ratios):.2f}")
        print()
        print("  A map where the straight line already works teaches nothing —")
        print("  the robot could ignore the terrain entirely and still score.")
        return 0

    for s in range(a.seed, a.seed + a.show):
        m = generate(s)
        p, length = solve(m)
        print("=" * 46)
        print(f"  seed {s}   A{tuple(round(v,1) for v in m.start)} -> "
              f"B{tuple(round(v,1) for v in m.goal)}")
        print(f"  straight line {straight_line(m):.1f} m   "
              f"route {length:.1f} m" if p else "  NO ROUTE")
        print(f"  {sum(o.kind=='step' for o in m.obstacles)} climbable, "
              f"{sum(o.kind=='wall' for o in m.obstacles)} impassable")
        print("=" * 46)
        print(ascii_map(m, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
