#!/usr/bin/env python3
"""
A Mars-like surface, generated rather than downloaded.

    uv run mars.py --check 40        # do the measured properties hold?
    uv run mars.py --ascii           # look at one map without a renderer
    uv run mars.py --scale regional  # the planetary features
    uv run mars.py --scale rover     # the one a robot could walk on

PART 1 OF LAB 5, AND IT CONTAINS NO ROBOT
------------------------------------------
This lab is split because terrain and policy are different problems and
mixing them is how you end up unable to say which one broke. Lab 4
learned that the expensive way: four of its bugs were in the world, the
renderer or the metric rather than in any policy, and each one produced
a plausible number first. So Part 1 builds the ground and measures it,
with nothing standing on it. No PPO, no reward, no arrival rate.

WHY THERE ARE TWO SCALES, AND WHY THAT IS NOT A COP-OUT
--------------------------------------------------------
The famous Martian landforms are planetary. Olympus Mons is ~600 km
across and ~22 km tall; Valles Marineris runs ~4,000 km and is up to
7 km deep. A robot arena is tens of metres. You cannot put either in
one, and a 2 m bump labelled "Olympus Mons" would be a lie told with
geometry.

So the generator has two presets and they answer different questions:

    regional   20 km across. The shield volcano, the canyon system,
               the cratered highlands and the dichotomy boundary are
               all present at something near their real proportions.
               This is for LOOKING at. Nothing walks here.

    rover      60 m across. What a surface robot actually meets:
               boulder fields, aeolian ripples, small craters, the
               floor of a dry channel, and the regional slope of a
               shield volcano's flank reduced to what it is locally --
               a gentle, almost imperceptible grade. This is the one
               Part 2 will put a robot on.

The honest statement is that the rover map is a *local sample of a
Martian-looking surface*, not a scale model of Mars. Both are
generated from the same code with different constants, and the
`--check` properties are asserted at both scales so that one cannot
quietly stop being Mars-like while the other still is.

WHAT IS AND IS NOT MODELLED
----------------------------
Modelled, because it changes the shape of the ground:

    * the crustal dichotomy -- smooth low northern plains against
      high, rough, heavily cratered southern highlands
    * impact craters with raised rims, bowl interiors and ejecta, drawn
      from a power-law size distribution (small ones vastly outnumber
      large ones, which is what a real crater count looks like)
    * a shield volcano: very low slope, broad, with a summit caldera
    * a rift canyon system with stepped walls
    * dendritic dry channels that run downhill and join
    * aeolian ripples and dune fields
    * scattered rock, as angular PLATY SLABS rather than boulders --
      which is what rover imagery actually shows
    * patches of exposed, fractured bedrock pavement between the dust
    * crater DEGRADATION: sharp young bowls superimposed on soft,
      half-filled old ones, skewed older in the southern highlands
    * swarms of parallel graben (*fossae*), sharing one orientation

NOT modelled, and listed so nobody assumes otherwise:

    * polar ice caps. A square patch of surface has no pole in it. The
      caps are a global feature and placing a white blob in the corner
      of a 20 km tile would be decoration, not terrain.
    * atmosphere, dust suspension, lighting physics. The renderer
      picks a rusty palette because iron oxide is why Mars is red, but
      nothing here simulates scattering.
    * real elevation data. MOLA exists and this is not it. Everything
      is procedural, seeded and reproducible.
    * regolith mechanics. The ground is rigid. Sinkage, slip and dust
      behaviour are a different lab and a much harder one.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Scale presets.
#
# Two sets of constants, one generator. Every length below is in METRES
# so the two presets can be compared without unit arithmetic in your
# head -- the regional map is simply 20,000 m across instead of 60.
# ---------------------------------------------------------------------------
SCALE = os.environ.get("MARS_SCALE", "rover")

PRESETS = {
    "rover": dict(
        extent=60.0,          # metres across
        cells=320,            # height-field samples per side
        relief=3.0,           # metres from lowest to highest, roughly
        n_craters=26,
        crater_r=(0.35, 4.0),      # metres
        n_boulders=140,
        boulder_r=(0.06, 0.45),
        ripple_wavelength=1.8,
        ripple_amp=0.045,
        channel_width=6.0,
        volcano=False,        # a 600 km shield has no summit at 60 m
        canyon=False,
        flank_grade=0.035,    # the volcano's flank, seen locally: 3.5%
        n_outcrop=9,          # patches of exposed fractured bedrock
        outcrop_thick=0.16,   # metres proud of the soil
        n_graben=0,
        # The two affordance classes, and the whole point of the rover
        # map. Sized so the dunes stay under the step limit and the
        # mesas are unambiguously over it -- then MEASURED, because a
        # constant chosen by eye is not evidence.
        n_hills=3,            # round rises, climbable
        hill_r=(3.5, 7.5),
        hill_h=(0.35, 0.95),
        # Long smooth dune ridges -- the climbable barrier. Length as
        # a FRACTION of the arena, so the task survives a change of
        # scale instead of silently emptying out, which is the bug
        # lab 5's big-world sweep caught in lab 4's geometry.
        n_dunes=5,
        dune_len=(0.40, 0.65),       # x extent
        dune_w=(5.0, 9.0),
        dune_h=(0.40, 0.80),
        n_mesas=2,            # cliff-edged buttes: go around these
        mesa_r=(2.2, 4.0),
        mesa_h=(0.55, 1.15),
        # ELONGATED scarps. Length is the lever: round buttes cost
        # nothing to skirt, so the task needs ridges that nearly span
        # the arena, with open ends so a route always exists.
        n_scarps=5,
        scarp_len=(0.55, 0.88),      # also a fraction of the arena
        scarp_w=(1.6, 3.0),
        scarp_h=(0.5, 1.0),
    ),
    "regional": dict(
        extent=20_000.0,
        cells=420,
        relief=6_500.0,
        n_craters=240,
        crater_r=(60.0, 1_400.0),
        n_boulders=0,         # a 40 cm rock is below one cell here
        boulder_r=(0.0, 0.0),
        ripple_wavelength=900.0,
        ripple_amp=12.0,
        channel_width=700.0,
        volcano=True,
        canyon=True,
        flank_grade=0.0,
        n_outcrop=0,          # below one cell at 48 m per cell
        outcrop_thick=0.0,
        n_graben=11,          # a fossae swarm, as at Sirenum
        n_hills=0, hill_r=(0, 0), hill_h=(0, 0),
        n_dunes=0, dune_len=(0, 0), dune_w=(0, 0), dune_h=(0, 0),
        n_mesas=0, mesa_r=(0, 0), mesa_h=(0, 0),
        n_scarps=0, scarp_len=(0, 0), scarp_w=(0, 0), scarp_h=(0, 0),
    ),
}


def preset(name: str | None = None) -> dict:
    name = name or SCALE
    if name not in PRESETS:
        raise SystemExit(f"--scale must be one of {sorted(PRESETS)}, got {name!r}")
    return PRESETS[name]


@dataclass
class Rock:
    """
    A surface rock, and it is a SLAB rather than a boulder.

    Rover imagery of Meridiani, Gusev and Jezero shows the same thing
    almost everywhere: the loose rock is platy and angular, lying
    roughly flat, broken out of bedding. Spheres are what you get when
    you reach for the nearest primitive, and a field of them reads as
    gravel on a driveway rather than as Mars. Half-extents here are
    deliberately anisotropic -- `rz` is a fraction of `rx`/`ry` -- and
    each slab carries a small tilt, because they sit on uneven ground
    and on each other.
    """

    x: float
    y: float
    z: float          # centre height of the slab
    rx: float         # half-extents, metres
    ry: float
    rz: float
    yaw: float
    pitch: float      # small, from resting on uneven ground
    roll: float
    buried: float     # fraction of thickness below the surface

    @property
    def r(self) -> float:
        """Nominal size, for the size-frequency statistics."""
        return float(max(self.rx, self.ry))


@dataclass
class Surface:
    heights: np.ndarray            # (cells, cells), metres
    rocks: list[Rock] = field(default_factory=list)
    craters: list[tuple] = field(default_factory=list)   # (x, y, r, age)
    landforms: list[tuple] = field(default_factory=list)  # (kind,x,y,r,h)
    scale: str = "rover"
    seed: int = 0

    @property
    def extent(self) -> float:
        return preset(self.scale)["extent"]

    def height_at(self, x: float, y: float) -> float:
        """Bilinear sample, in world metres."""
        n = self.heights.shape[0]
        e = self.extent
        u = (x + e / 2) / e * (n - 1)
        v = (y + e / 2) / e * (n - 1)
        i0, j0 = int(np.clip(u, 0, n - 2)), int(np.clip(v, 0, n - 2))
        fu, fv = u - i0, v - j0
        h = self.heights
        return float(
            h[j0, i0] * (1 - fu) * (1 - fv) + h[j0, i0 + 1] * fu * (1 - fv)
            + h[j0 + 1, i0] * (1 - fu) * fv + h[j0 + 1, i0 + 1] * fu * fv)


# ---------------------------------------------------------------------------
# Noise. Value noise with octaves -- "fractional Brownian motion".
#
# Written out rather than imported because the whole point of this file
# is that a reader can see where every feature comes from, and because
# a dependency on a noise library would be a dependency on its seeding
# behaviour, which is exactly the kind of thing that makes a "seeded,
# reproducible" claim quietly false.
# ---------------------------------------------------------------------------
def _value_noise(rng: np.random.Generator, n: int, freq: int) -> np.ndarray:
    """One octave: random values on a coarse grid, smoothly interpolated."""
    g = rng.random((freq + 1, freq + 1))
    # Smoothstep interpolation; linear leaves visible creases along the
    # lattice, which read as rectangular artefacts rather than terrain.
    ys, xs = np.mgrid[0:n, 0:n] / (n - 1) * freq
    i0, j0 = np.clip(xs.astype(int), 0, freq - 1), np.clip(ys.astype(int), 0, freq - 1)
    fx, fy = xs - i0, ys - j0
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    return ((g[j0, i0] * (1 - sx) + g[j0, i0 + 1] * sx) * (1 - sy)
            + (g[j0 + 1, i0] * (1 - sx) + g[j0 + 1, i0 + 1] * sx) * sy)


def fbm(rng: np.random.Generator, n: int, octaves: int = 6,
        lacunarity: float = 2.0, gain: float = 0.5,
        base_freq: int = 2) -> np.ndarray:
    """Sum octaves of value noise. Returns roughly [0, 1]."""
    out = np.zeros((n, n))
    amp, freq, norm = 1.0, base_freq, 0.0
    for _ in range(octaves):
        out += amp * _value_noise(rng, n, int(freq))
        norm += amp
        amp *= gain
        freq *= lacunarity
    return out / max(norm, 1e-9)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def crater_radii(rng: np.random.Generator, n: int,
                 lo: float, hi: float, slope: float = 2.0) -> np.ndarray:
    """
    Crater sizes from a POWER LAW, not a uniform draw.

    Real crater populations follow N(>D) proportional to D**-b with b
    around 2: small craters hugely outnumber large ones. Drawing sizes
    uniformly gives a surface evenly covered in medium bowls, which is
    the single most recognisable way a synthetic planet looks fake.

    Inverse-transform sampled, so the exponent is honoured exactly
    rather than approximated by rejection.
    """
    u = rng.random(n)
    a, b = lo ** (-slope), hi ** (-slope)
    return (a + u * (b - a)) ** (-1.0 / slope)


def add_crater(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
               cx: float, cy: float, r: float, age: float = 0.0,
               depth_ratio: float = 0.18, rim_ratio: float = 0.045) -> None:
    """
    One impact crater: a bowl, a raised rim, ejecta -- and an AGE.

    Depth is a FRACTION of diameter rather than a constant, because
    that is how craters scale: a 10 m crater is not as deep as a 1 km
    one. Fresh simple craters sit near depth/diameter ~ 0.2.

    `age` in [0, 1] is the part that makes a cratered plain look real.
    Orbital images of the southern highlands (e.g. Sirenum Fossae) show
    sharp young bowls superimposed on soft, half-filled, rimless old
    ones -- the surface is a palimpsest. Drawing every crater fresh
    gives a uniform golf-ball texture that no real terrain has.
    Degradation here shallows the floor, flattens the rim and widens
    the whole profile, which is what infilling and mass wasting do.
    """
    d = np.hypot(wx - cx, wy - cy)
    soft = 1.0 - 0.85 * age              # old craters are shallow
    widen = 1.0 + 0.35 * age             # and blurred outward
    rr = r * widen
    inside = d < rr
    bowl = -depth_ratio * soft * 2 * r * (1.0 - (d / max(rr, 1e-9)) ** 2)
    rim = (rim_ratio * (1.0 - 0.9 * age) * 2 * r
           * np.exp(-((d - rr) / max(0.42 * rr, 1e-9)) ** 2))
    h += np.where(inside, bowl, 0.0) + rim


def add_shield_volcano(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                       cx: float, cy: float, radius: float,
                       height: float) -> None:
    """
    A shield volcano: enormous, and far flatter than anyone draws it.

    Olympus Mons averages about a 5% grade -- you could drive up it and
    not notice. Drawing it as a cone is the second most recognisable
    way a synthetic planet looks fake. The profile here is a smooth
    dome with a summit caldera punched into it.
    """
    d = np.hypot(wx - cx, wy - cy)
    flank = height * np.exp(-(d / (0.52 * radius)) ** 2)
    h += flank
    # Summit caldera: a flat-floored collapse, not a volcanic cone's pit.
    cal_r = 0.13 * radius
    cal = (d < cal_r)
    if cal.any():
        rim_h = float(np.max(h[cal]))
        h[cal] = np.minimum(h[cal], rim_h - 0.085 * height)


def add_canyon(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
               x0: float, y0: float, x1: float, y1: float,
               width: float, depth: float, rng=None) -> None:
    """
    A rift canyon with STEPPED walls.

    Valles Marineris is a tectonic rift that was then widened by
    landslides and collapse, so its walls are terraced rather than
    smooth. Three steps is enough to read as terracing at this
    resolution.
    """
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((wx - x0) * vx + (wy - y0) * vy) / max(L2, 1e-9), 0.0, 1.0)
    px, py = x0 + t * vx, y0 + t * vy
    d = np.hypot(wx - px, wy - py)
    # Width tapers towards the ends rather than stopping square.
    w = width * (0.45 + 0.55 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5)
    prof = np.clip(1.0 - d / np.maximum(w, 1e-9), 0.0, 1.0)
    stepped = np.ceil(prof * 3.0) / 3.0          # three terraces
    h -= depth * stepped * prof ** 0.5


def add_channels(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                 rng: np.random.Generator, extent: float, width: float,
                 depth: float, n_trunks: int = 2) -> None:
    """
    Dry riverbeds: they run DOWNHILL and they JOIN.

    A channel network drawn as random squiggles is the third tell. The
    diagnostic property of a real one is that tributaries merge into
    trunks and the whole thing is monotonic in elevation, because water
    had to flow somewhere. Each trunk here walks downslope by sampling
    the local gradient, and tributaries are launched onto it.
    """
    n = h.shape[0]
    step = extent / n * 2.0
    # Accumulate the deepest cut per cell, then apply ONCE.
    #
    # The first version subtracted `depth` at every one of 150 walk
    # steps, and consecutive steps overlap almost completely -- so a
    # channel asked for 0.15 m of depth was carved to tens of metres
    # and the rover map came back with 46 m of relief inside a 60 m
    # box. It still looked like terrain in ASCII, which is exactly why
    # the relief is measured against the preset rather than eyeballed.
    cut_depth = np.zeros_like(h)

    def walk(x, y, w, steps):
        for _ in range(steps):
            # local downhill direction
            i = int(np.clip((x + extent / 2) / extent * (n - 1), 1, n - 2))
            j = int(np.clip((y + extent / 2) / extent * (n - 1), 1, n - 2))
            gx = h[j, i + 1] - h[j, i - 1]
            gy = h[j + 1, i] - h[j - 1, i]
            g = math.hypot(gx, gy)
            if g < 1e-12:
                ang = rng.uniform(0, 2 * math.pi)
                dx, dy = math.cos(ang), math.sin(ang)
            else:
                dx, dy = -gx / g, -gy / g
            # meander: water does not run in straight lines
            a = math.atan2(dy, dx) + rng.normal(0, 0.28)
            dx, dy = math.cos(a), math.sin(a)
            x, y = x + dx * step, y + dy * step
            if abs(x) > extent / 2 or abs(y) > extent / 2:
                return
            d = np.hypot(wx - x, wy - y)
            cut = np.clip(1.0 - d / w, 0.0, 1.0)
            np.maximum(cut_depth, depth * cut ** 0.6, out=cut_depth)

    for _ in range(n_trunks):
        x0 = rng.uniform(-extent / 2 * 0.8, extent / 2 * 0.8)
        y0 = rng.uniform(-extent / 2 * 0.8, extent / 2 * 0.8)
        walk(x0, y0, width, 150)
        for _ in range(3):                       # tributaries onto it
            walk(x0 + rng.normal(0, extent * 0.06),
                 y0 + rng.normal(0, extent * 0.06), width * 0.45, 60)
    h -= cut_depth


# ---------------------------------------------------------------------------
# THE AFFORDANCE CLASSES
#
# A pretty surface is not a task. Lab 4 established that the thing
# which makes terrain a LEARNING problem is a choice with a cost: some
# obstacles you should climb, some you must go around, and telling
# them apart has to pay. Mars supplies both shapes naturally, so the
# two classes here are landforms rather than inventions:
#
#   climbable   aeolian dunes, degraded crater rims, low ramparts --
#               anything the wind or a billion years has rounded off.
#               Rise per step stays under STEP_MAX, so a legged robot
#               walks over it and detouring costs distance.
#
#   blocked     mesas, buttes and fresh scarps. Mars is full of
#               flat-topped remnants with near-vertical sides, because
#               a resistant caprock protects softer rock beneath until
#               the edge collapses. The top is perfectly walkable; the
#               EDGE is a wall. That is a much more interesting
#               obstacle than a tall mound, because the robot cannot
#               tell from height alone -- it has to read the edge.
#
# Both are measured by `traversable()` below rather than trusted, and
# `--check` reports whether refusing to climb actually costs anything.
# ---------------------------------------------------------------------------
STEP_MAX = 0.12          # metres a two-legged robot can step up (lab 2)


def add_dune_ridge(h, wx, wy, cx, cy, length, width, height, angle):
    """
    A long, smooth dune ridge: the CLIMBABLE barrier.

    Both classes have to be long, not just the blocking one. Round
    hills measured at zero cost -- walking around a 5 m dune is free,
    so "should I climb this?" was never a question worth asking. A
    transverse dune ridge spanning most of the arena makes it one, and
    it is what a Martian dune field actually looks like from above:
    long parallel crests combed by one prevailing wind.

    The cross-section is a raised cosine, so the steepest rise per
    cell stays well under STEP_MAX by construction -- and is then
    measured, because "by construction" is how lab 4 shipped two
    obstacle shapes that cost nothing.
    """
    ca, sa = math.cos(-angle), math.sin(-angle)
    dx, dy = wx - cx, wy - cy
    u = dx * ca - dy * sa
    v = dx * sa + dy * ca
    along = np.clip(1.0 - (np.abs(u) / (length / 2)) ** 4, 0.0, 1.0)
    across = np.clip(1.0 - np.abs(v) / (width / 2), 0.0, 1.0)
    h += height * along * (0.5 - 0.5 * np.cos(np.pi * across))


def add_smooth_hill(h, wx, wy, cx, cy, radius, height):
    """
    A rounded rise the robot SHOULD climb.

    Gaussian, so the steepest slope is height/(radius*sqrt(e)) and can
    be kept below the step limit by construction. A cone or a
    paraboloid both end in a discontinuity at the base, which reads as
    a kerb and would block a robot at the one point it is supposed to
    be able to walk on.
    """
    d = np.hypot(wx - cx, wy - cy)
    h += height * np.exp(-(d / (0.5 * radius)) ** 2)


def add_scarp(h, wx, wy, cx, cy, length, width, height, angle, rng=None):
    """
    An ELONGATED cliff-edged ridge. The obstacle that actually blocks.

    Round buttes turned out to cost nothing: a 4 m mesa in a 60 m
    arena is free to walk around, and the measured cost of refusing to
    climb came back at 0.00 m on four seeds out of six -- a beautiful
    surface with no decision in it, which is the exact failure lab 4
    shipped twice before length was identified as the lever.

    So the blocking class is a ridge: long enough that skirting it
    costs real distance, with ends left open so going around stays
    POSSIBLE. On Mars this is an escarpment or an inverted channel
    ridge -- a resistant layer standing above eroded surroundings --
    and it has the same flat top and sharp shoulder as a butte.
    """
    ca, sa = math.cos(-angle), math.sin(-angle)
    dx, dy = wx - cx, wy - cy
    u = dx * ca - dy * sa                   # along the ridge
    v = dx * sa + dy * ca                   # across it
    if rng is not None:
        v = v + 0.18 * width * np.sin(u / max(length * 0.08, 1e-9)
                                      + rng.uniform(0, 6.3))
    along = np.clip(1.0 - (np.abs(u) / (length / 2)) ** 6, 0.0, 1.0)
    top = np.clip((1.0 - np.abs(v) / (width / 2)) * 14.0, 0.0, 1.0)
    talus = np.clip(1.25 - np.abs(v) / (width / 2), 0.0, 1.0) ** 2 * 0.16
    h += height * np.maximum(top, talus) * along


def add_mesa(h, wx, wy, cx, cy, radius, height, rng=None):
    """
    A flat-topped butte with a CLIFF edge -- the go-around obstacle.

    The profile is deliberately almost a step function: flat top, near
    vertical wall, talus apron at the foot. An irregular outline keeps
    it from reading as a cylinder, and the apron matters because real
    scarps shed debris and because a perfectly vertical wall meeting
    flat ground is the one case a height-field renderer draws worst.
    """
    ang = np.arctan2(wy - cy, wx - cx)
    if rng is not None:
        wob = (1.0 + 0.22 * np.sin(3 * ang + rng.uniform(0, 6.3))
               + 0.12 * np.sin(5 * ang + rng.uniform(0, 6.3)))
    else:
        wob = 1.0
    d = np.hypot(wx - cx, wy - cy) / np.maximum(radius * wob, 1e-9)
    # Hard shoulder: 1 inside, falling to 0 over a few per cent of the
    # radius. That steepness IS the obstacle.
    top = np.clip((1.0 - d) * 14.0, 0.0, 1.0)
    talus = np.clip((1.25 - d) * 1.1, 0.0, 1.0) ** 2 * 0.16
    h += height * np.maximum(top, talus)


def traversable(s, step_max: float = STEP_MAX) -> np.ndarray:
    """
    Cells a legged robot could stand on, given the local rise.

    Blocked when the step UP to any 4-neighbour exceeds `step_max`.
    This is a statement about the gait, not about absolute height: the
    top of a mesa is as walkable as the plain once you are on it, and
    only the rise between adjacent cells can stop you. That is exactly
    why the mesa is the interesting obstacle -- height alone does not
    predict whether you can be there.
    """
    h = s.heights.astype(float)
    ok = np.ones_like(h, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
        ok &= (nb - h) <= step_max
    return ok


def geodesic(s, start, goal, step_max: float = STEP_MAX) -> float:
    """
    Shortest walkable distance between two points, 8-connected.

    Dijkstra with sqrt(2) diagonals, and a diagonal is only legal when
    both orthogonal neighbours are walkable -- otherwise a route can
    squeeze through the corner where two cliffs touch, a gap of zero
    width that a robot with a body cannot use.

    Lab 4 used a 4-connected breadth-first search for this and it
    overstated the true distance by up to sqrt(2), which silently
    corrupted every efficiency number that lab published. Eight
    neighbours from the start here.
    """
    import heapq

    ok = traversable(s, step_max)
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    to_ij = lambda p: (int(np.clip((p[1] + e / 2) / cell, 0, n - 1)),
                       int(np.clip((p[0] + e / 2) / cell, 0, n - 1)))
    a, b = to_ij(start), to_ij(goal)
    if not ok[a] or not ok[b]:
        return float("inf")
    R2 = math.sqrt(2.0)
    dist = {a: 0.0}
    pq = [(0.0, a)]
    while pq:
        dd, cur = heapq.heappop(pq)
        if cur == b:
            return dd * cell
        if dd > dist.get(cur, math.inf):
            continue
        j, i = cur
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nj, ni = j + dj, i + di
            if not (0 <= nj < n and 0 <= ni < n) or not ok[nj, ni]:
                continue
            if dj and di and not (ok[j, ni] and ok[nj, i]):
                continue
            nd = dd + (R2 if (dj and di) else 1.0)
            if nd < dist.get((nj, ni), math.inf):
                dist[(nj, ni)] = nd
                heapq.heappush(pq, (nd, (nj, ni)))
    return float("inf")


def add_outcrop(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                rng: np.random.Generator, extent: float,
                n_patches: int, thickness: float) -> None:
    """
    Patches of exposed, fractured bedrock pavement.

    This is the texture that dominates rover imagery and that a
    noise-plus-rocks surface completely lacks: flat slabs of bedrock
    standing a few centimetres proud of the soil, broken into polygonal
    plates by joints. The ground is not a smooth blanket with stones on
    it -- it is partly stone, with blankets of dust between.

    Built as low plateaux with a hard edge and a fracture pattern cut
    into the top, because the fractures are what make it read as rock
    rather than as a step.
    """
    n = h.shape[0]
    for _ in range(n_patches):
        cx = rng.uniform(-extent / 2 * 0.9, extent / 2 * 0.9)
        cy = rng.uniform(-extent / 2 * 0.9, extent / 2 * 0.9)
        rad = extent * rng.uniform(0.03, 0.11)
        # Irregular outline: a circle warped by low-frequency noise, so
        # the patch has lobes and embayments instead of being a disc.
        ang = np.arctan2(wy - cy, wx - cx)
        wobble = (1.0 + 0.30 * np.sin(3 * ang + rng.uniform(0, 6.3))
                  + 0.18 * np.sin(7 * ang + rng.uniform(0, 6.3)))
        d = np.hypot(wx - cx, wy - cy) / np.maximum(rad * wobble, 1e-9)
        slab = np.clip((1.0 - d) * 6.0, 0.0, 1.0)      # hard-ish edge
        # Polygonal jointing across the top of the slab.
        fx, fy = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
        k = 2 * math.pi / (extent * rng.uniform(0.012, 0.03))
        joints = (np.abs(np.sin(k * wx + fx)) ** 12
                  + np.abs(np.sin(k * wy + fy)) ** 12)
        h += slab * thickness * (1.0 - 0.55 * np.clip(joints, 0, 1))


def add_graben_swarm(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                     rng: np.random.Generator, extent: float,
                     n: int, depth: float) -> None:
    """
    A set of parallel linear troughs -- *fossae*.

    Sirenum Fossae, Cerberus Fossae, Memnonia: Mars is covered in these
    and they come in SWARMS sharing one orientation, because they are
    extensional fractures from a common stress field. One lonely rift
    is a canyon; a dozen parallel ones is tectonics, and the parallelism
    is the part that reads as real.
    """
    trend = rng.uniform(0, math.pi)
    ca, sa = math.cos(trend), math.sin(trend)
    across = -wx * sa + wy * ca
    along = wx * ca + wy * sa
    for _ in range(n):
        off = rng.uniform(-extent * 0.42, extent * 0.42)
        w = extent * rng.uniform(0.004, 0.012)
        # Graben die out along their length rather than ending square.
        span = rng.uniform(0.35, 0.95) * extent
        centre = rng.uniform(-extent * 0.2, extent * 0.2)
        taper = np.clip(1.0 - np.abs(along - centre) / (span / 2), 0.0, 1.0)
        prof = np.clip(1.0 - np.abs(across - off) / w, 0.0, 1.0)
        h -= depth * prof ** 0.45 * taper ** 0.6


def add_ripples(h: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                wavelength: float, amp: float, angle: float) -> None:
    """
    Aeolian ripples: one dominant wind direction, slightly irregular.

    Transverse to the wind, which is why dune fields look combed.
    """
    u = wx * math.cos(angle) + wy * math.sin(angle)
    v = -wx * math.sin(angle) + wy * math.cos(angle)
    # The gentle bend along-crest keeps them from looking like corduroy.
    phase = 2 * math.pi * (u + 0.08 * v * np.sin(v / (wavelength * 7))) / wavelength
    h += amp * (0.65 * np.sin(phase) + 0.35 * np.sin(2.3 * phase + 1.1))


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------
def generate(seed: int = 0, scale: str | None = None) -> Surface:
    """One Mars-like surface. Deterministic in `seed`."""
    P = preset(scale)
    name = scale or SCALE
    rng = np.random.default_rng(seed)
    n, e = P["cells"], P["extent"]
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) / n * e - e / 2
    wy = (ys + 0.5) / n * e - e / 2

    # --- base relief -------------------------------------------------------
    h = fbm(rng, n, octaves=7, gain=0.52) * P["relief"] * 0.35

    # --- the crustal dichotomy, REGIONAL ONLY ------------------------------
    #
    # This is a hemispheric feature. A 60 m patch of ground does not
    # straddle the boundary between the northern plains and the
    # southern highlands -- it is in one or the other. Applying it at
    # rover scale produced a "highlands vs plains" difference of 0.7 m
    # across the arena, which is not the dichotomy, it is a slope.
    # Mars is two different planets stuck together: the north is low,
    # smooth volcanic plain and the south is high, rough, ancient and
    # cratered. This is the single most important fact about its
    # topography and it is one line of code, so there is no excuse for
    # leaving it out.
    if name == "regional":
        boundary = 0.5 * (1.0 + np.tanh(wy / (e * 0.06)))  # smooth, not a cliff
        h += (1.0 - boundary) * P["relief"] * 0.22         # highlands lifted
        rough = fbm(rng, n, octaves=8, gain=0.60, base_freq=4)
        h += (1.0 - boundary) * rough * P["relief"] * 0.09  # and roughened

    # --- craters, power-law sized, concentrated in the old south ----------
    craters = []
    radii = crater_radii(rng, P["n_craters"], *P["crater_r"])
    for r in radii:
        # The southern highlands are ancient and therefore saturated;
        # the northern plains were resurfaced and have far fewer.
        for _ in range(12):
            cx = rng.uniform(-e / 2, e / 2)
            cy = rng.uniform(-e / 2, e / 2)
            # Old southern highlands are saturated; the resurfaced
            # north is not. At rover scale there is no north and south,
            # so craters fall uniformly.
            p_keep = 1.0 if name == "rover" else (0.22 if cy > 0 else 1.0)
            if rng.random() < p_keep:
                break
        # Age, so the plain is a palimpsest rather than a golf ball.
        # Skewed towards OLD in the southern highlands, which is the
        # whole reason that half of the planet looks ancient.
        age = float(rng.beta(1.4, 1.6) if (name == "rover" or cy > 0)
                    else rng.beta(2.4, 1.1))
        add_crater(h, wx, wy, cx, cy, float(r), age=age)
        craters.append((float(cx), float(cy), float(r), age))

    # --- the big two, regional only ---------------------------------------
    if P["volcano"]:
        add_shield_volcano(h, wx, wy, -e * 0.26, e * 0.22,
                           radius=e * 0.30, height=P["relief"] * 0.55)
    if P["canyon"]:
        add_canyon(h, wx, wy, -e * 0.45, -e * 0.16, e * 0.46, -e * 0.30,
                   width=e * 0.045, depth=P["relief"] * 0.30)

    # --- the two affordance classes ---------------------------------------
    landforms = []
    for _ in range(P["n_hills"]):
        cx = rng.uniform(-e / 2 * 0.82, e / 2 * 0.82)
        cy = rng.uniform(-e / 2 * 0.82, e / 2 * 0.82)
        rad = float(rng.uniform(*P["hill_r"]))
        ht = float(rng.uniform(*P["hill_h"]))
        add_smooth_hill(h, wx, wy, cx, cy, rad, ht)
        landforms.append(("hill", float(cx), float(cy), rad, ht))
    for _ in range(P["n_mesas"]):
        cx = rng.uniform(-e / 2 * 0.82, e / 2 * 0.82)
        cy = rng.uniform(-e / 2 * 0.82, e / 2 * 0.82)
        rad = float(rng.uniform(*P["mesa_r"]))
        ht = float(rng.uniform(*P["mesa_h"]))
        add_mesa(h, wx, wy, cx, cy, rad, ht, rng)
        landforms.append(("mesa", float(cx), float(cy), rad, ht))
    for _ in range(P["n_dunes"]):
        cx = rng.uniform(-e / 2 * 0.5, e / 2 * 0.5)
        cy = rng.uniform(-e / 2 * 0.5, e / 2 * 0.5)
        L = float(rng.uniform(*P["dune_len"])) * e
        W = float(rng.uniform(*P["dune_w"]))
        ht = float(rng.uniform(*P["dune_h"]))
        ang = float(rng.uniform(0, math.pi))
        add_dune_ridge(h, wx, wy, cx, cy, L, W, ht, ang)
        landforms.append(("dune", float(cx), float(cy), W / 2, ht))
    for _ in range(P["n_scarps"]):
        cx = rng.uniform(-e / 2 * 0.55, e / 2 * 0.55)
        cy = rng.uniform(-e / 2 * 0.55, e / 2 * 0.55)
        L = float(rng.uniform(*P["scarp_len"])) * e
        W = float(rng.uniform(*P["scarp_w"]))
        ht = float(rng.uniform(*P["scarp_h"]))
        ang = float(rng.uniform(0, math.pi))
        add_scarp(h, wx, wy, cx, cy, L, W, ht, ang, rng)
        landforms.append(("scarp", float(cx), float(cy), W, ht))

    # --- tectonics: parallel fractures, not one lonely rift ---------------
    if P["n_graben"]:
        add_graben_swarm(h, wx, wy, rng, e, P["n_graben"],
                         depth=P["relief"] * 0.035)

    # --- exposed bedrock, the texture rover images are full of ------------
    if P["n_outcrop"]:
        add_outcrop(h, wx, wy, rng, e, P["n_outcrop"], P["outcrop_thick"])

    # --- water, long gone --------------------------------------------------
    add_channels(h, wx, wy, rng, e, P["channel_width"],
                 depth=P["relief"] * (0.012 if name == "regional" else 0.05))

    # --- wind, still going -------------------------------------------------
    add_ripples(h, wx, wy, P["ripple_wavelength"], P["ripple_amp"],
                angle=rng.uniform(0, math.pi))

    # --- the flank of something enormous, seen from very close -------------
    # At rover scale the honest expression of "you are on a shield
    # volcano" is not a mountain in view; it is a constant, boring grade.
    if P["flank_grade"]:
        h += wx * P["flank_grade"]

    h = h.astype(np.float32)

    # --- boulders ----------------------------------------------------------
    rocks: list[Rock] = []
    surf = Surface(heights=h, rocks=rocks, craters=craters,
                   landforms=landforms, scale=name, seed=seed)
    if P["n_boulders"]:
        lo, hi = P["boulder_r"]
        for _ in range(P["n_boulders"]):
            x = rng.uniform(-e / 2 * 0.95, e / 2 * 0.95)
            y = rng.uniform(-e / 2 * 0.95, e / 2 * 0.95)
            # Power law again: pebbles outnumber boulders, as on any
            # real debris field.
            a = float(crater_radii(rng, 1, lo, hi, slope=2.4)[0])
            b = a * float(rng.uniform(0.55, 1.0))      # not square in plan
            # PLATY. Thickness is a small fraction of the long axis,
            # which is what rover imagery shows nearly everywhere.
            c = a * float(rng.uniform(0.18, 0.42))
            buried = float(rng.uniform(0.15, 0.45))
            z = surf.height_at(x, y) + c * (1.0 - 2.0 * buried)
            rocks.append(Rock(
                x=x, y=y, z=z, rx=a, ry=b, rz=c,
                yaw=float(rng.uniform(0, math.pi)),
                # Small tilts only: a slab resting on uneven ground, not
                # one thrown there.
                pitch=float(rng.normal(0, 0.10)),
                roll=float(rng.normal(0, 0.10)),
                buried=buried))
    return surf


# ---------------------------------------------------------------------------
# Measurements. The lab's actual content: claims that can fail.
# ---------------------------------------------------------------------------
def slope_degrees(s: Surface) -> np.ndarray:
    """Per-cell slope magnitude in degrees."""
    cell = s.extent / s.heights.shape[0]
    gy, gx = np.gradient(s.heights.astype(float), cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def dichotomy(s: Surface) -> dict:
    """North vs south: elevation, roughness and crater count."""
    h = s.heights
    n = h.shape[0]
    north, south = h[n // 2:, :], h[: n // 2, :]
    sl = slope_degrees(s)
    cn = sum(1 for _, cy, *_ in s.craters if cy > 0)
    return dict(
        north_mean=float(north.mean()), south_mean=float(south.mean()),
        north_rough=float(sl[n // 2:, :].mean()),
        south_rough=float(sl[: n // 2, :].mean()),
        craters_north=cn, craters_south=len(s.craters) - cn)


def crater_power_law(s: Surface) -> float:
    """
    Fit the exponent of N(>D) ~ D**-b. Should come back near 2.

    Measured rather than assumed: the sampler is supposed to produce
    this, and a fit is the only way to know it did.
    """
    r = np.array([c[2] for c in s.craters])
    if len(r) < 8:
        return float("nan")
    r = np.sort(r)
    counts = np.arange(len(r), 0, -1)
    lo, hi = int(len(r) * 0.05), int(len(r) * 0.95)
    x, y = np.log(r[lo:hi]), np.log(counts[lo:hi])
    return float(-np.polyfit(x, y, 1)[0])


def rocks_sit_on_surface(s: Surface) -> tuple[float, float]:
    """
    Worst floating rock and worst buried one, in metres.

    A rock whose centre is not at `surface + r(1 - buried)` is either
    hovering or swallowed. Both look fine in a still frame from above
    and absurd the moment the camera drops to ground level -- and a
    floating rock becomes an invisible collision for the robot in
    Part 2.
    """
    worst_float = worst_sink = 0.0
    for k in s.rocks:
        want = s.height_at(k.x, k.y) + k.rz * (1.0 - 2.0 * k.buried)
        err = k.z - want
        worst_float = max(worst_float, err)
        worst_sink = min(worst_sink, err)
    return worst_float, abs(worst_sink)


def traversable_fraction(s: Surface, max_slope_deg: float = 15.0) -> float:
    """
    How much of this surface could a legged robot stand on?

    Part 2 puts a robot here, and a beautiful surface it cannot cross
    is a broken lab that looks like a hard one -- which this repository
    has shipped twice. Measuring it now, before any policy exists, is
    the whole reason Part 1 is separate.
    """
    return float((slope_degrees(s) <= max_slope_deg).mean())


def classes_differ(s: Surface) -> dict:
    """
    Do the two landform classes actually afford different things?

    The constants say hills are climbable and mesas are not. That is a
    hypothesis about geometry, and this measures it: sample the
    traversability mask over each landform's footprint and report how
    much of it a robot could stand on.

    Asserting the PROPERTY rather than the parameters is the lesson
    lab 4 paid for twice -- it shipped obstacle shapes that looked
    right and cost nothing to avoid, so the "learn to climb" half of
    the task was decoration while every test passed.
    """
    # Measure the EDGE RING, not the footprint.
    #
    # The first version averaged traversability over each landform's
    # bounding box and reported mesas as 86% walkable -- which is true
    # and meaningless, because the box is mostly the flat top (walkable
    # by design) and the surrounding plain. The obstacle is the
    # shoulder. What distinguishes the classes is whether a ring at the
    # landform's own radius contains blocked cells.
    ok = traversable(s)
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - e / 2
    wy = (ys + 0.5) * cell - e / 2
    out = {}
    for kind, cx, cy, rad, _h in s.landforms:
        d = np.hypot(wx - cx, wy - cy)
        ring = (d > rad * 0.75) & (d < rad * 1.25)
        if ring.sum() < 8:
            continue
        out.setdefault(kind, []).append(float((~ok[ring]).mean()))
    return {f"{k}_edge_blocked": float(np.mean(v)) for k, v in out.items()}


def climbing_pays(s: Surface, n_pairs: int = 10, rng=None) -> dict:
    """
    What does refusing to climb COST, in metres? Returns the count too.

    Two bugs lived in the first version of this function and the second
    one is the instructive one.

    It blocked every cell above the median elevation, which was 38% of
    the arena -- that walls the whole map off, so every route in the
    counterfactual world came back infinite. Those pairs were then
    dropped, `gains` ended up empty, and the function returned `0.0`.

    **Zero and no-data are different facts and they were being reported
    with the same number.** Eight seeds in a row read "+0.00 m", which
    looked exactly like "this terrain contains no decision" and was
    actually "this measurement produced nothing". A silent zero is the
    worst possible failure value: it is plausible, it is in range, and
    it points at the wrong culprit.

    So: block only the CLIMBABLE landforms, which is what refusing to
    climb actually means; and report `n_usable` alongside the median so
    an empty measurement cannot pass as a finding.
    """
    rng = rng or np.random.default_rng(s.seed)
    e = s.extent
    n = s.heights.shape[0]
    cell = e / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - e / 2
    wy = (ys + 0.5) * cell - e / 2

    # The counterfactual: the robot refuses to go over anything it
    # could have climbed. Only dunes and hills -- scarps and mesas were
    # never climbable, so walling them again changes nothing.
    blocked = Surface(heights=s.heights.copy(), scale=s.scale, seed=s.seed)
    mask = np.zeros((n, n), dtype=bool)
    for kind, cx, cy, rad, _h in s.landforms:
        if kind in ("dune", "hill"):
            mask |= np.hypot(wx - cx, wy - cy) < max(rad, cell * 2)
    if not mask.any():
        return dict(median_gain=float("nan"), n_usable=0, n_pairs=n_pairs)
    blocked.heights[mask] += 9.0

    gains, usable = [], 0
    for _ in range(n_pairs):
        for _ in range(60):
            a = (float(rng.uniform(-e / 2 * 0.9, e / 2 * 0.9)),
                 float(rng.uniform(-e / 2 * 0.9, e / 2 * 0.9)))
            b = (float(rng.uniform(-e / 2 * 0.9, e / 2 * 0.9)),
                 float(rng.uniform(-e / 2 * 0.9, e / 2 * 0.9)))
            if math.hypot(a[0] - b[0], a[1] - b[1]) > e * 0.45:
                break
        d0 = geodesic(s, a, b)
        d1 = geodesic(blocked, a, b)
        if math.isfinite(d0) and math.isfinite(d1):
            usable += 1
            gains.append(d1 - d0)
    return dict(
        median_gain=float(np.median(gains)) if gains else float("nan"),
        n_usable=usable, n_pairs=n_pairs)


def summary(s: Surface) -> dict:
    d = dichotomy(s)
    f, b = rocks_sit_on_surface(s)
    sl = slope_degrees(s)
    return dict(
        scale=s.scale, seed=s.seed, extent_m=s.extent,
        cells=s.heights.shape[0],
        relief_m=float(s.heights.max() - s.heights.min()),
        slope_mean=float(sl.mean()), slope_p95=float(np.percentile(sl, 95)),
        traversable_15deg=traversable_fraction(s),
        crater_exponent=crater_power_law(s),
        **(classes_differ(s) if s.landforms else {}),
        n_craters=len(s.craters), n_rocks=len(s.rocks),
        rock_float_m=f, rock_sink_m=b, **d)


# ---------------------------------------------------------------------------
# Looking at it without a renderer
# ---------------------------------------------------------------------------
def ascii_map(s: Surface, w: int = 64) -> str:
    h = s.heights
    n = h.shape[0]
    step = max(1, n // w)
    ramp = " .:-=+*#%@"
    lo, hi = float(h.min()), float(h.max())
    rows = []
    for jj in range(w):
        row = ""
        for ii in range(w):
            j, i = min(jj * step, n - 1), min(ii * step, n - 1)
            t = (h[j, i] - lo) / max(hi - lo, 1e-9)
            row += ramp[min(int(t * len(ramp)), len(ramp) - 1)]
        rows.append(row)
    return "\n".join(reversed(rows))        # north at the top


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=sorted(PRESETS), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, metavar="N",
                    help="measure the properties over N seeds")
    ap.add_argument("--ascii", action="store_true")
    a, _ = ap.parse_known_args()
    name = a.scale or SCALE

    if a.check:
        print("=" * 74)
        print(f"  MARS SURFACE — {a.check} seeds at both scales")
        print("=" * 74)
        for sc in sorted(PRESETS):
            rows = [summary(generate(s, sc)) for s in range(a.check)]
            g = lambda k: np.array([r[k] for r in rows])
            print(f"\n  [{sc}]  {PRESETS[sc]['extent']:,.0f} m across, "
                  f"{PRESETS[sc]['cells']} cells")
            print(f"    relief                 {g('relief_m').mean():10,.1f} m")
            print(f"    slope, mean / p95      {g('slope_mean').mean():6.1f} deg"
                  f" / {g('slope_p95').mean():.1f} deg")
            print(f"    traversable (<=15 deg) {g('traversable_15deg').mean():10.1%}")
            print(f"    crater exponent b      {np.nanmean(g('crater_exponent')):10.2f}"
                  f"   (real populations ~2)")
            print(f"    craters  south / north {g('craters_south').mean():6.0f}"
                  f" / {g('craters_north').mean():.0f}")
            print(f"    highlands vs plains    "
                  f"{(g('south_mean') - g('north_mean')).mean():+10,.1f} m")
            print(f"    roughness  S / N       {g('south_rough').mean():6.2f}"
                  f" / {g('north_rough').mean():.2f} deg")
            if g('n_rocks').max():
                print(f"    rocks                  {g('n_rocks').mean():10.0f}"
                      f"   worst float {g('rock_float_m').max():.4f} m,"
                      f" worst sink {g('rock_sink_m').max():.4f} m")
        print("\n  Nothing here is a robot. Part 2 adds one.")
        return 0

    s = generate(a.seed, name)
    d = summary(s)
    print(f"  {name} surface, seed {a.seed}: {d['extent_m']:,.0f} m across, "
          f"{d['relief_m']:,.1f} m of relief")
    print(f"  slope mean {d['slope_mean']:.1f} deg, "
          f"{d['traversable_15deg']:.0%} walkable at 15 deg, "
          f"{d['n_craters']} craters, {d['n_rocks']} rocks")
    if a.ascii:
        print()
        print(ascii_map(s))
        print("\n  (north at the top; the southern highlands are the bright half)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
