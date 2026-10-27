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

    # The landing zone is a disc at the origin, flattened with a
    # smooth blend. Without it the start pose depends on the seed and
    # "the policy got a flat spot" becomes a confound.
    xs = np.linspace(-EXTENT / 2, EXTENT / 2, N)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    blend = np.clip((r - 4.0) / 5.0, 0.0, 1.0)
    h = h * blend + h[N // 2, N // 2] * (1 - blend) * 0.0

    h -= h.min()
    rocks = []
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
