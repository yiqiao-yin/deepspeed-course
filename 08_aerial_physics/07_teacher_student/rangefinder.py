#!/usr/bin/env python3
"""
The sensor the student is allowed to have: a ray-cast depth camera.

    uv run rangefinder.py                 # geometry self-check, no MuJoCo
    uv run rangefinder.py --scene         # cast against real terrain
    uv run rangefinder.py --cost          # what a scan costs vs physics

WHY RAYS AND NOT AN OPENGL CAMERA
----------------------------------
The obvious way to give a policy sight is to render an image. That
was measured first and rejected, and the measurement is worth
keeping because it is a trap:

    offscreen depth, 32x32, hazard terrain    147 ms / frame
    offscreen depth, 64x64, hazard terrain    146 ms / frame
    offscreen depth, 32x32, FLAT ground       109 ms / frame
    a 12x12 ray scan                            0.74 ms / scan

The resolution column is the tell. A renderer whose cost does not
move when the pixel count quadruples is not spending its time on
pixels, and a flat plane costing 109 ms proves it is not spending
it on the terrain mesh either. It is per-frame overhead in the GL
stack -- on this box OpenGL reaches the GPU through WSL's D3D12
translation layer, and forcing the discrete card instead of the
integrated one changed nothing (147 ms either way). Topic 7 lab 6
hit the same class of bug from the other side: a physics backend
that silently fell back to CPU and reported throughput that looked
like GPU throughput. Ask the stack what it is, and time the thing
you intend to run.

A renderer also needs a GL context per process, which is exactly
what the 16-worker pool in `vecenv.py` cannot give it.

So the sensor is `mj_ray`, and this is not a consolation prize. A
scanning rangefinder IS the hardware small landers actually carry --
it is what gives you range in shadow, which passive stereo cannot --
and ray casting is a faithful model of one rather than an
approximation of a camera.

WHAT THE STUDENT THEREFORE SUFFERS, AND THE TEACHER DOES NOT
-------------------------------------------------------------
Everything that makes perception hard survives the change, which
is the point:

    occlusion     a beam stops at the first thing it hits, so the
                  far side of a boulder is unknown
    perspective   the ground sample spacing grows with altitude, so
                  the same 144 numbers mean different things at 7 m
                  and at 2 m
    attitude      the frustum is bolted to the body. Roll the car
                  and the whole scan swings; the teacher's height
                  patch is world-aligned and does not
    range limit   past MAX_RANGE a beam returns "nothing", which is
                  not the same as "flat"
    noise         2% of range, plus 1% of beams dropping out

The last one matters more than it looks. Without dropout the
student can detect "no return" as an exact constant and use it as
a clean feature; with it, the same value sometimes means a missing
pixel. Sensors that fail cleanly are the ones policies learn to
exploit.

THE FRUSTUM
-----------
A square pyramid about the body's own down-axis, +/- HALF_FOV in
each of the body x and y directions, sampled on a BEAM_N x BEAM_N
grid. At 6 m altitude it covers roughly a 10 m square of ground,
which is deliberately close to the 8 m square the teacher's height
patch covers -- the arms differ in HOW they see the same region,
not in how much of it they are shown. An unmatched footprint would
have made the comparison a field-of-view experiment.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

BEAM_N = 12                   # BEAM_N x BEAM_N beams
HALF_FOV = math.radians(40.0)  # half-angle of the square frustum
MAX_RANGE = 25.0              # metres; beyond this a beam returns nothing
NOISE_FRAC = 0.02             # proportional range noise, 1 sigma
DROPOUT_P = 0.01              # chance a beam returns nothing anyway

# What a beam reports when it hits nothing. It is 1.0 -- the far end
# of the normalised scale -- and NOT a sentinel like -1. A policy
# reading a 1.0 sees "very far away", which is the honest reading of
# a missing return and keeps the input on one continuous scale. A
# -1 would put a discontinuity in the middle of the sensor's range
# for a network to trip over.
NO_RETURN = 1.0


def beam_directions(n: int = BEAM_N,
                    half_fov: float = HALF_FOV) -> np.ndarray:
    """
    Unit directions of every beam, in the BODY frame.

    The optical axis is body -z (straight down when upright). A
    pixel at tangent offsets (tx, ty) points along (tx, ty, -1),
    normalised -- the pinhole model, so the angular spacing is NOT
    uniform and the corner beams sit further out than a naive
    linear-in-angle fan would put them. That is what a real depth
    camera does and it is what makes the sample spacing on flat
    ground grow toward the edges.

    Returns `(n * n, 3)`, row-major with x varying fastest, so a
    scan reshapes to `(n, n)` as an image with +x to the right.

    The first version wrote `np.tan(np.linspace(-fov, +fov, n))`,
    which spaces the ANGLES uniformly and builds a fan rather than
    a pinhole. It produced unit vectors pointing in sensible
    directions and a plausible-looking depth image; what it got
    wrong was the ground sample spacing, silently, in the direction
    that makes perception look easier than it is. The geometry
    check below is what caught it, which is the whole argument for
    testing a property rather than a shape.
    """
    if n < 2:
        raise ValueError(f"need at least a 2x2 grid, got {n}")
    e = math.tan(half_fov)
    t = np.linspace(-e, e, n)
    ty, tx = np.meshgrid(t, t, indexing="ij")
    d = np.stack([tx.ravel(), ty.ravel(), -np.ones(n * n)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def footprint_m(altitude: float, half_fov: float = HALF_FOV) -> float:
    """Width of ground the frustum covers, flat ground, pointing down."""
    return 2.0 * altitude * math.tan(half_fov)


class RangeFinder:
    """
    A body-mounted depth camera, cast with `mj_ray`.

    Construction resolves the vehicle's own geoms so a self-hit can
    be DETECTED rather than silently read as ground at 30 cm. See
    `scan()` -- that check is the reason this is a class and not a
    function.
    """

    def __init__(self, model, mujoco, n: int = BEAM_N,
                 half_fov: float = HALF_FOV, max_range: float = MAX_RANGE,
                 noise: float = NOISE_FRAC, dropout: float = DROPOUT_P,
                 mount: tuple[float, float, float] = (0.0, 0.0, -0.35),
                 hull_body: int = 1):
        self._mj = mujoco
        self.model = model
        self.n = int(n)
        self.max_range = float(max_range)
        self.noise = float(noise)
        self.dropout = float(dropout)
        self.mount = np.asarray(mount, dtype=np.float64)
        # The body the sensor is BOLTED TO is excluded from the cast.
        # A real sensor cannot see its own mounting plate, and
        # including it means every beam returns 4 mm -- measured,
        # and the reason this argument exists. The WHEELS are
        # separate bodies and stay visible on purpose: a belly
        # camera really is occluded by its own gear, and that is
        # part of what the student has to cope with.
        self.hull_body = int(hull_body)
        self._exclude = int(hull_body)
        self.dirs = beam_directions(self.n, half_fov)
        self.n_beams = self.dirs.shape[0]
        # Every geom's owning body, so a return can be classified as
        # "the world" (ground, rocks) or "the vehicle".
        self.geom_body = np.asarray(model.geom_bodyid).copy()
        self._gid = np.zeros(1, dtype=np.int32)
        self.self_hits = 0
        self.total_beams = 0

    def scan(self, data, rng: np.random.Generator | None = None,
             clean: bool = False) -> np.ndarray:
        """
        One frame. Returns `(n_beams,)` normalised ranges in [0, 1].

        `clean=True` disables noise and dropout, which is what the
        figures and the self-hit audit use -- never training.
        """
        mujoco = self._mj
        R = _rotmat(data.qpos[3:7])
        origin = data.qpos[:3] + R @ self.mount
        world_dirs = self.dirs @ R.T
        out = np.empty(self.n_beams, dtype=np.float64)
        selfhit = 0
        for i in range(self.n_beams):
            dist = mujoco.mj_ray(self.model, data, origin, world_dirs[i],
                                 None, 1, self._exclude, self._gid)
            g = int(self._gid[0])
            if dist < 0.0 or dist > self.max_range or g < 0:
                out[i] = NO_RETURN
                continue
            if self.geom_body[g] != 0:
                # The beam hit the car. Not fatal -- report it as a
                # return at its true range, because that is what the
                # sensor would do -- but COUNT it, so `audit()` can
                # fail loudly if the mount is looking at the vehicle
                # instead of the ground.
                selfhit += 1
            out[i] = dist / self.max_range
        self.self_hits += selfhit
        self.total_beams += self.n_beams
        if clean:
            return out
        if rng is not None:
            out = out * (1.0 + self.noise * rng.standard_normal(self.n_beams))
            out = np.where(rng.random(self.n_beams) < self.dropout,
                           NO_RETURN, out)
        return np.clip(out, 0.0, 1.0)

    def self_hit_rate(self) -> float:
        return self.self_hits / max(self.total_beams, 1)

    def audit(self, limit: float = 0.05) -> None:
        """
        Raise if the sensor is mostly looking at its own vehicle.

        A mount buried inside the hull returns a dense field of
        30 cm readings. It is a perfectly valid array of the right
        shape and dtype, it varies a little as the suspension moves,
        and a policy trained on it learns nothing and reports no
        error -- the exact `silent-wrong` shape this course keeps
        getting caught by. So the check is an assertion, not a note
        in the README.
        """
        r = self.self_hit_rate()
        if r > limit:
            raise RuntimeError(
                f"{r:.1%} of beams hit the vehicle (limit {limit:.0%}) over "
                f"{self.total_beams:,} beams. The sensor mount at "
                f"{tuple(self.mount)} is looking at the car, not the "
                f"ground, and a policy trained on it would see a constant.")


def _rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _geometry_checks() -> int:
    """Properties of the frustum that hold without any simulator."""
    bad = 0

    def ck(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            bad += 1

    d = beam_directions()
    ck("every beam is a unit vector",
       np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12))
    ck("every beam points DOWN in the body frame",
       bool((d[:, 2] < 0).all()),
       f"max z component {d[:, 2].max():+.3f}")
    centre = d[(BEAM_N * BEAM_N) // 2 + BEAM_N // 2]
    ck("the middle beam is the optical axis",
       abs(centre[0]) < 0.2 and abs(centre[1]) < 0.2,
       f"({centre[0]:+.2f}, {centre[1]:+.2f}, {centre[2]:+.2f})")
    # A pinhole grid is uniform in TANGENT and therefore NOT uniform
    # in angle. If it were uniform in angle it would be a fan, the
    # ground sample spacing would be wrong, and nothing downstream
    # could tell -- which is exactly what the first version shipped.
    grid = d.reshape(BEAM_N, BEAM_N, 3)
    tx = grid[:, :, 0] / (-grid[:, :, 2])          # recovered tangents
    ck("the grid is uniform in TANGENT — a pinhole",
       bool(np.allclose(np.diff(tx, axis=1),
                        np.diff(tx, axis=1)[0, 0], atol=1e-12)),
       f"step {np.diff(tx, axis=1)[0, 0]:.4f} everywhere")
    ang = np.degrees(np.arctan(tx[0]))
    gaps = np.diff(ang)
    half = len(gaps) // 2
    ck("...so the ANGLES tighten toward the edge, they do not fan",
       bool((np.diff(gaps[:half]) > 0).all() and gaps[0] < gaps[half]),
       f"{gaps[0]:.2f} deg at the edge rising to {gaps[half]:.2f} "
       f"deg at the axis")
    fp = footprint_m(6.0)
    ck("the footprint at 6 m is close to the teacher's 8 m patch",
       6.0 < fp < 14.0, f"{fp:.1f} m across at 6 m altitude")
    ck("a scan reshapes to a square image",
       d.shape[0] == BEAM_N * BEAM_N)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="store_true",
                    help="cast against real terrain and print the image")
    ap.add_argument("--cost", action="store_true",
                    help="time a scan against a physics step")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--seed", type=int, default=0)
    a, _ = ap.parse_known_args()

    print("=" * 72)
    print(f"  RANGEFINDER — {BEAM_N}x{BEAM_N} beams, "
          f"{math.degrees(HALF_FOV) * 2:.0f} deg square frustum, "
          f"{MAX_RANGE:.0f} m range")
    print("=" * 72)
    bad = _geometry_checks()

    if a.scene or a.cost:
        import time

        import mujoco

        import car as CAR
        import thrusters as TH

        model, rig, surf = CAR.build("default", a.preset, seed=a.seed,
                                     start_z=8.0, world=TH.WORLDS["earth"])
        data = mujoco.MjData(model)
        for _ in range(40):
            mujoco.mj_step(model, data)
        rf = RangeFinder(model, mujoco)
        rng = np.random.default_rng(a.seed)

        if a.scene:
            img = rf.scan(data, clean=True).reshape(BEAM_N, BEAM_N)
            print(f"\n  altitude {data.qpos[2] - surf.height_at(*data.qpos[:2]):.2f} m"
                  f"   footprint ~{footprint_m(data.qpos[2]):.1f} m")
            ramp = " .:-=+*#%@"
            print("  the scan, near = dark:")
            for row in img:
                print("    " + "".join(
                    ramp[min(int(v * (len(ramp) - 1)), len(ramp) - 1)]
                    for v in row))
            print(f"  range  min {img.min() * MAX_RANGE:5.2f} m   "
                  f"max {img.max() * MAX_RANGE:5.2f} m   "
                  f"no-return {(img >= NO_RETURN - 1e-9).mean():.1%}")
            print(f"  self-hit rate {rf.self_hit_rate():.2%}")
            rf.audit()
            print("  PASS  the sensor is looking at the ground, not the car")

        if a.cost:
            for _ in range(20):
                rf.scan(data, rng)
            t0 = time.time()
            N = 400
            for _ in range(N):
                rf.scan(data, rng)
            ms = (time.time() - t0) / N * 1e3
            t0 = time.time()
            for _ in range(2000):
                mujoco.mj_step(model, data)
            step_ms = (time.time() - t0) / 2000 * 1e3 * 5   # 5 substeps
            print(f"\n  one scan          {ms:7.3f} ms")
            print(f"  one control step  {step_ms:7.3f} ms  (5 physics "
                  f"substeps, no sensing)")
            print(f"  sensing makes a step {1 + ms / step_ms:.2f}x more "
                  f"expensive")
            print(f"  an OpenGL depth frame, measured on this box: "
                  f"~147 ms — {147 / ms:,.0f}x this")

    print("\n" + "=" * 72)
    print(f"  {'FAILED' if bad else 'all geometry checks passed'}")
    print("=" * 72)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
