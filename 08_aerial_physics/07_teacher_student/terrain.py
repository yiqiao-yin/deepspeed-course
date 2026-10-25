#!/usr/bin/env python3
"""
The surface the car has to land on. No robot, no thrusters.

    uv run terrain.py                  # stats for the default pad
    uv run terrain.py --preset rough --png /tmp/t.png

WHY A SEPARATE FILE WITH NOTHING IN IT BUT GROUND
--------------------------------------------------
Topic 7's lab 5 split terrain from policy on purpose, because the lab
before it published a wrong result whose faults were in its world,
its renderer and its metric rather than in anything a policy learned.
Terrain and control are different problems; mixing them leaves you
unable to say which one broke. So this file contains no vehicle.

It is also deliberately NOT imported from topic 7. This course
duplicates rather than shares: a reader must be able to open one
folder and run it without the other twenty-four existing.

THREE PRESETS, AND THEY ARE ADDED RATHER THAN SUBSTITUTED
----------------------------------------------------------
    pad        flat. The control. If a policy cannot land here the
               failure is the controller, not the ground.
    rolling    gentle relief. Landing attitude now has to match a
               slope the car did not choose.
    rough      ridges and scattered boulders -- somewhere a 2-tonne
               car genuinely should not be put down.

A published number describes the surface it was measured on, so a
preset's parameters never change once a result quotes it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

# One cell is 25 cm. The car is 4.34 m long, so it spans ~17 cells --
# enough that a landing can straddle a feature rather than sitting on
# a single sample, which is the resolution at which "rough" starts to
# mean anything to a vehicle this size.
CELL = 0.25
N = 257                               # grid is N x N; 64 m across
EXTENT = (N - 1) * CELL


@dataclass
class Rock:
    x: float
    y: float
    rx: float
    ry: float
    rz: float
    yaw: float


@dataclass
class Surface:
    heights: np.ndarray               # (N, N) metres
    extent: float
    relief: float
    preset: str
    rocks: list[Rock] = field(default_factory=list)

    def height_at(self, x: float, y: float) -> float:
        """Bilinear sample. Used by the env to place the car."""
        n = self.heights.shape[0]
        u = (x + self.extent / 2) / self.extent * (n - 1)
        v = (y + self.extent / 2) / self.extent * (n - 1)
        u = float(np.clip(u, 0, n - 1 - 1e-6))
        v = float(np.clip(v, 0, n - 1 - 1e-6))
        i, j = int(u), int(v)
        fu, fv = u - i, v - j
        h = self.heights
        return float((1 - fu) * (1 - fv) * h[j, i]
                     + fu * (1 - fv) * h[j, i + 1]
                     + (1 - fu) * fv * h[j + 1, i]
                     + fu * fv * h[j + 1, i + 1])

    def slope_deg_at(self, x: float, y: float) -> float:
        """Local slope, which is what a landing gear actually cares
        about -- height alone says nothing about whether the car will
        tip over once it is down."""
        d = CELL * 2
        dzdx = (self.height_at(x + d, y) - self.height_at(x - d, y)) / (2 * d)
        dzdy = (self.height_at(x, y + d) - self.height_at(x, y - d)) / (2 * d)
        return math.degrees(math.atan(math.hypot(dzdx, dzdy)))


PRESETS: dict[str, dict] = {
    "pad":     dict(relief=0.00, octaves=0, n_rocks=0, ridge=0.0),
    "rolling": dict(relief=0.90, octaves=4, n_rocks=0, ridge=0.0),
    "rough":   dict(relief=2.40, octaves=5, n_rocks=26, ridge=0.55),
    # DECISIVE hazards, not merely bumpy ground.
    #
    # `rough` turned out not to be a landing DECISION: measured over
    # ten spots with twelve drops each, per-spot success varied by
    # sd 0.13 against the 0.14 expected from binomial noise alone.
    # Every spot was about as good as every other, so terrain
    # knowledge had nothing to offer and a perception experiment on
    # it would have produced a clean null by construction.
    #
    # `hazard` adds features a car cannot land on at all -- boulders
    # taller than its ground clearance and scarps past the angle it
    # can stand on -- separated by genuinely flat pans. Spot choice
    # then matters, which `uv run target_env.py --decidable`
    # re-measures rather than assumes. (That flag lives on
    # target_env because it needs `survivable`, which needs the
    # vehicle's clearance and stance -- not a property of the
    # surface alone.)
    "hazard":  dict(relief=1.10, octaves=3, n_rocks=0, ridge=0.0,
                    boulders=34, scarps=5, pans=7),
    # BADLANDS: hazards larger than a guidance error.
    #
    # `hazard` made spots differ (per-spot success 8% to 83%) but the
    # lab still could not use that, because a controller's touchdown
    # scatter is 1-3 m and the hazards were about the same size. A
    # blind controller aiming at bad ground landed on GOOD ground 52%
    # of the time by accident -- as often as one that aimed at good
    # ground on purpose. The decision existed and was unexploitable.
    #
    # Here the terrain is built from a low-frequency mask instead, so
    # landable pans and impassable badlands come in 10-20 m patches.
    # A 2 m miss then keeps you in the same region, and choosing the
    # region is the whole task.
    # Few and LARGE. For a circular bad patch of radius R a
    # typical interior point is only R/3 from the edge, so 6 m
    # fields leave a 2 m escape -- inside the touchdown scatter.
    # Three fields of 9-12 m put the median escape past 4 m.
    "badlands": dict(relief=0.35, octaves=2, n_rocks=0, ridge=0.0,
                     badlands=True, boulder_clusters=3),
}


def _value_noise(rng: np.random.Generator, n: int, octaves: int) -> np.ndarray:
    """
    Fractal value noise by bilinear upsampling of coarse grids.

    Not Perlin: Perlin needs gradients and a permutation table and
    buys nothing visible at this resolution. What matters is that the
    spectrum falls off, so the surface has both long swells and local
    texture rather than one scale of bumps.
    """
    out = np.zeros((n, n), dtype=np.float64)
    if octaves <= 0:
        return out
    amp, total = 1.0, 0.0
    for o in range(octaves):
        g = max(2, 2 ** (o + 2))
        coarse = rng.normal(0.0, 1.0, (g, g))
        ii = np.linspace(0, g - 1, n)
        i0 = np.floor(ii).astype(int)
        i1 = np.minimum(i0 + 1, g - 1)
        f = (ii - i0)[None, :]
        rowlo = coarse[:, i0] * (1 - f) + coarse[:, i1] * f
        f2 = (ii - i0)[:, None]
        grid = rowlo[i0, :] * (1 - f2) + rowlo[i1, :] * f2
        out += amp * grid
        total += amp
        amp *= 0.5
    return out / max(total, 1e-9)


def generate(seed: int = 0, preset: str = "rolling") -> Surface:
    """A surface. Deterministic in `seed`, so a run is reproducible."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    p = PRESETS[preset]
    rng = np.random.default_rng(seed)
    rocks_pending: list = []
    h = _value_noise(rng, N, p["octaves"])

    if p["ridge"] > 0:
        # A couple of ridges, so "rough" has structure a car can be
        # wrong about rather than only noise it averages over.
        xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
        X, Y = np.meshgrid(xs, xs)
        for _ in range(2):
            cx, cy = rng.uniform(-EXTENT / 4, EXTENT / 4, 2)
            ang = rng.uniform(0, math.pi)
            d = (X - cx) * math.sin(ang) - (Y - cy) * math.cos(ang)
            h += p["ridge"] * np.exp(-(d / rng.uniform(3.0, 6.0)) ** 2)

    if h.std() > 1e-9:
        h = (h - h.mean()) / h.std()
    h = h * (p["relief"] / 3.0)        # 3 sigma spans the stated relief

    # --- badlands: large, coherent go / no-go regions ---------------
    if p.get("badlands"):
        xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
        X, Y = np.meshgrid(xs, xs)
        # Two octaves only -> features about 16 m across, comfortably
        # larger than a 1-3 m touchdown scatter.
        mask = _value_noise(rng, N, 2)
        mask = (mask - mask.mean()) / (mask.std() + 1e-9)
        bad = np.clip((mask - 0.25) / 0.45, 0.0, 1.0)      # smooth 0..1
        # Inside a bad region the ground is violently broken; outside
        # it is a flat pan.
        # The bad region is a packed BOULDER FIELD, not a hill.
        #
        # A smooth dome is only steep on its flanks; its summit is
        # flat and perfectly landable, so the middle of every "bad"
        # region was safe. High-frequency rubble fails the same way:
        # between any two bumps there is a sub-12-degree spot. Both
        # left the nearest safe ground a median 1.5 m from a bad
        # target -- inside a 2 m touchdown scatter, so the decision
        # could not be exploited.
        #
        # Coverage is what makes a region uniformly unlandable. The
        # car needs 2.6 m of clearance from a boulder, so boulders on
        # a 3.4 m lattice leave no clear point anywhere inside the
        # field.
        h = h + bad * 1.2
        for _ in range(p.get("boulder_clusters", 0)):
            for _try in range(300):
                cx, cy = rng.uniform(-EXTENT / 2 + 9, EXTENT / 2 - 9, 2)
                i = int((cx + EXTENT / 2) / CELL)
                j = int((cy + EXTENT / 2) / CELL)
                if math.hypot(cx, cy) > 11.0 and bad[j, i] > 0.5:
                    break
            rad = rng.uniform(9.0, 12.0)
            step = 3.4
            k = int(rad / step) + 1
            for a in range(-k, k + 1):
                for b in range(-k, k + 1):
                    bx = cx + step * (a + 0.5 * (b % 2))
                    by = cy + step * b * 0.87
                    if math.hypot(bx - cx, by - cy) > rad:
                        continue
                    if math.hypot(bx, by) < 10.0:
                        continue
                    rocks_pending.append(Rock(
                        x=float(bx + rng.normal(0, 0.25)),
                        y=float(by + rng.normal(0, 0.25)),
                        rx=float(rng.uniform(1.0, 1.6)),
                        ry=float(rng.uniform(1.0, 1.6)),
                        rz=float(rng.uniform(0.6, 1.1)),
                        yaw=float(rng.uniform(0, math.pi))))

    # --- decisive hazards -------------------------------------------
    if p.get("scarps"):
        xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
        X, Y = np.meshgrid(xs, xs)
        for _ in range(p["scarps"]):
            cx, cy = rng.uniform(-EXTENT / 2 + 8, EXTENT / 2 - 8, 2)
            ang = rng.uniform(0, math.pi)
            d = (X - cx) * math.sin(ang) - (Y - cy) * math.cos(ang)
            # A step, not a swell: tanh over ~1.2 m gives a face
            # steeper than the car can stand on.
            h += rng.uniform(1.8, 3.2) * np.tanh(d / 1.2) * \
                np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / 11.0 ** 2))
    if p.get("pans"):
        xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
        X, Y = np.meshgrid(xs, xs)
        for _ in range(p["pans"]):
            cx, cy = rng.uniform(-EXTENT / 2 + 9, EXTENT / 2 - 9, 2)
            w = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / 4.5 ** 2))
            h = h * (1 - w) + float(h[int((cy + EXTENT / 2) / CELL),
                                      int((cx + EXTENT / 2) / CELL)]) * w

    # The landing zone is a disc at the origin, flattened with a
    # smooth blend. Without it the start pose depends on the seed and
    # "the policy got a flat spot" becomes a confound.
    xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    blend = np.clip((r - 4.0) / 5.0, 0.0, 1.0)
    h = h * blend + h[N // 2, N // 2] * (1 - blend) * 0.0

    h -= h.min()
    rocks = list(rocks_pending)
    for _ in range(p.get("boulders", 0)):
        while True:
            x, y = rng.uniform(-EXTENT / 2 + 5, EXTENT / 2 - 5, 2)
            if math.hypot(x, y) > 9.0:
                break
        # Taller than the car's 0.55 m ground clearance, so landing
        # ON one is not a landing.
        rocks.append(Rock(x=float(x), y=float(y),
                          rx=float(rng.uniform(0.9, 1.9)),
                          ry=float(rng.uniform(0.9, 1.9)),
                          rz=float(rng.uniform(0.55, 1.10)),
                          yaw=float(rng.uniform(0, math.pi))))
    for _ in range(p["n_rocks"]):
        while True:
            x, y = rng.uniform(-EXTENT / 2 + 4, EXTENT / 2 - 4, 2)
            if math.hypot(x, y) > 9.0:      # never inside the pad
                break
        rocks.append(Rock(x=float(x), y=float(y),
                          rx=float(rng.uniform(0.25, 0.75)),
                          ry=float(rng.uniform(0.25, 0.75)),
                          rz=float(rng.uniform(0.12, 0.40)),
                          yaw=float(rng.uniform(0, math.pi))))

    return Surface(heights=h, extent=EXTENT,
                   relief=max(float(h.max()), 1e-3),
                   preset=preset, rocks=rocks)


def hfield_png(h: np.ndarray, path) -> None:
    """
    MuJoCo loads a height field from a greyscale PNG and rescales it
    so white equals the `size` z component. The METRES therefore live
    in the model's `hfield` size, NOT in the image -- topic 7 shipped
    a test that asserted on the image data and could never fail,
    because MuJoCo normalises it to [0, 1] at compile time.
    """
    from PIL import Image

    a = h - h.min()
    a = a / max(float(a.max()), 1e-9)
    Image.fromarray((a * 65535).astype(np.uint16)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="rolling", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default="")
    a, _ = ap.parse_known_args()

    s = generate(a.seed, a.preset)
    print("=" * 74)
    print(f"  SURFACE — preset {a.preset}, seed {a.seed}")
    print("=" * 74)
    print(f"  grid        {s.heights.shape[0]} x {s.heights.shape[0]} "
          f"at {CELL} m — {s.extent:.1f} m across")
    print(f"  relief      {s.relief:.2f} m "
          f"(p5 {np.percentile(s.heights, 5):.2f}, "
          f"p95 {np.percentile(s.heights, 95):.2f})")
    print(f"  rocks       {len(s.rocks)}")
    print(f"  pad slope   {s.slope_deg_at(0, 0):.2f}° at the origin")
    far = [s.slope_deg_at(x, 0.0) for x in (10, 14, 18, 22)]
    print(f"  off-pad     {', '.join(f'{v:.1f}°' for v in far)} "
          f"at x = 10, 14, 18, 22 m")
    if a.png:
        hfield_png(s.heights, a.png)
        print(f"\n  wrote {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
