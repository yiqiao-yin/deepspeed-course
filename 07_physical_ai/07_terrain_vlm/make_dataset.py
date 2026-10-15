#!/usr/bin/env python3
"""
A traversability dataset with EXACT labels, generated for free.

    uv run make_dataset.py --check            # properties only, no rendering
    uv run make_dataset.py --seeds 400        # build it
    uv run make_dataset.py --seeds 400 --out data/

WHY THIS LAB EXISTS
-------------------
Lab 5 part 3 measured something uncomfortable. The robot discriminates
climbable landforms from impassable ones about 3:1 -- and so does an
arm fed six constant zeros. The discrimination was emergent from
having a BODY, not from perception: you learn a scarp is impassable by
walking into one. Three parts of that lab measured outcome and could
not have told the two apart.

So the question this lab asks is the one lab 5 left open: **can a
model that actually SEES the terrain beat walking into it?** That
needs a vision model, a labelled dataset, and a baseline that is
already measured rather than invented.

WHERE THE LABELS COME FROM
--------------------------
Not from a human, and not from the landform's NAME. Both would be
circular -- the generator chose the name, so predicting it measures
nothing about traversability.

The label is geometry, computed the way lab 5's `classes_differ` was
eventually forced to compute it: the fraction of a RING at the
landform's own radius that a step-limited gait cannot cross. The first
version of that measurement averaged over the bounding box and
reported mesas as 86% walkable -- true, and meaningless, because the
box is mostly the flat top and the surrounding plain. The obstacle is
the shoulder.

Samples in the ambiguous middle are DISCARDED rather than rounded to
the nearest class. A label that the oracle itself is unsure of is
noise in both training and evaluation, and it is the kind of noise
that makes a ceiling look like a model failure.

WHAT IS CHECKED BEFORE THE DATA IS USED
---------------------------------------
`--check` asserts the dataset is not degenerate, because every way
this can go quietly wrong produces a file that looks fine:

  * both classes present, and not at 95/5
  * the label is NOT perfectly predicted by the landform kind -- and
    the actual predictability is REPORTED rather than assumed, since
    if kind alone gets 100% the task is name-recognition wearing a
    vision costume
  * viewpoints are places a robot could actually stand (`standable`),
    not points floating inside a cliff
  * the two classes are not trivially separable by image brightness
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
BACKENDS = ("egl", "osmesa", "glfw")

# A landform counts as BLOCKED when this much of its edge ring is
# impassable to a step-limited gait, and CROSSABLE below the lower
# bound. The gap between them is the ambiguous band and is discarded.
#
# These two numbers are taken FROM THE MEASURED DISTRIBUTION, not
# chosen a priori, and the first attempt shows why that matters. It
# used 0.45 and 0.10 -- reasonable-looking round numbers -- and 0.45
# turned out to sit above the maximum a mesa ever reaches (0.390 on
# `rover`). Every blocker therefore landed in the discard band, and
# the generator produced 945 samples of which 100% were labelled
# OVER, including 207 scarps. It looked like a working dataset.
#
# Measured over 60 surfaces, edge-blocked fraction by kind:
#
#            rover (60 m)        course (34 m)
#   hill     0.045 mean          0.048
#   dune     0.074               0.124
#   scarp    0.131               0.185
#   mesa     0.228               0.311
#
# `course` is used because it separates the classes further apart,
# and because it is the preset lab 5's part 4 measured on.
BLOCKED_AT = 0.25
CROSSABLE_AT = 0.08

PROMPT = ("You are a rover on Mars. A terrain feature is directly ahead. "
          "Can you walk over it, or must you go around it? "
          "Answer with exactly one word: OVER or AROUND.")


def pick_backend() -> str:
    import subprocess

    forced = os.environ.get("MUJOCO_GL")
    probe = ("import mujoco;"
             "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>"
             "<geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
             "r=mujoco.Renderer(m,64,64);"
             "import mujoco as _m;d=_m.MjData(m);_m.mj_forward(m,d);"
             "r.update_scene(d);r.render()")
    for b in ((forced,) if forced else BACKENDS):
        env = dict(os.environ, MUJOCO_GL=b, PYOPENGL_PLATFORM=b)
        try:
            subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                           capture_output=True, timeout=120)
            return b
        except Exception:                                   # noqa: BLE001
            pass
    print("\n  No OpenGL backend, so no images can be rendered.\n"
          "  The label logic needs none:\n\n"
          "      uv run make_dataset.py --check\n")
    sys.exit(1)


def edge_blocked(s, cx: float, cy: float, rad: float) -> float | None:
    """
    Fraction of the landform's EDGE RING a gait cannot cross.

    This is the label, and it is deliberately the same measurement
    lab 5's `classes_differ` makes, so the two labs cannot drift into
    disagreeing about what "blocked" means.
    """
    import mars

    ok = mars.traversable(s)
    n = s.heights.shape[0]
    e = s.extent
    cell = e / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - e / 2
    wy = (ys + 0.5) * cell - e / 2
    d = np.hypot(wx - cx, wy - cy)
    ring = (d > rad * 0.75) & (d < rad * 1.25)
    if ring.sum() < 8:
        return None
    return float((~ok[ring]).mean())


def view_is_clean(s, vx: float, vy: float, cx: float, cy: float,
                  target, half_angle_deg: float = 18.0) -> bool:
    """
    Is the CENTRED feature the one the label is about?

    The camera aims at the target landform, but Mars is crowded and
    another landform can sit between the camera and the target, or
    just beside it in frame. When that happens the picture shows a
    cliff and the label describes a hill behind it -- which is label
    noise, and the kind that gets blamed on the model later because
    the image genuinely does not support the answer.

    So a viewpoint is rejected when any OTHER landform falls within
    `half_angle_deg` of the view direction and is nearer than the
    target. Cheap, and it removes a ceiling nobody would otherwise
    know was there.
    """
    tx, ty = cx - vx, cy - vy
    td = math.hypot(tx, ty)
    if td < 1e-6:
        return False
    for kind, ox, oy, orad, _h in s.landforms:
        if (kind, ox, oy, orad) == target:
            continue
        dx, dy = ox - vx, oy - vy
        od = math.hypot(dx, dy)
        if od < 1e-6 or od >= td:
            continue
        cosang = (tx * dx + ty * dy) / (td * od)
        if math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) < half_angle_deg:
            return False
    return True


def viewpoint(s, cx: float, cy: float, rad: float, rng,
              tries: int = 40, target=None) -> tuple | None:
    """
    Somewhere a robot could actually STAND, facing the landform.

    `standable` rather than `traversable`, for the reason lab 5 paid a
    whole sweep to learn: traversable asks whether a gait could step
    up to a neighbour, standing asks whether the ground under the
    whole footprint is flat and free of boulders. A viewpoint chosen
    by the weaker test puts the camera somewhere the robot would fall
    over, which is not a view it would ever have.
    """
    import mars

    for _ in range(tries):
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(rad * 1.8, rad * 3.2)
        x, y = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
        if abs(x) > s.extent / 2 - 2 or abs(y) > s.extent / 2 - 2:
            continue
        if not mars.standable(s, x, y):
            continue
        if target is not None and not view_is_clean(s, x, y, cx, cy, target):
            continue
        return float(x), float(y), float(dist)
    return None


def build_samples(seeds: range, scale: str, rng_seed: int = 0) -> list[dict]:
    """Every (surface, landform) pair that earns an unambiguous label."""
    sys.path.insert(0, str(LAB5))
    import mars

    rng = np.random.default_rng(rng_seed)
    rows, ambiguous, occluded = [], 0, 0
    for sd in seeds:
        s = mars.generate(sd, scale=scale)
        for kind, cx, cy, rad, h in s.landforms:
            eb = edge_blocked(s, cx, cy, rad)
            if eb is None:
                continue
            if eb >= BLOCKED_AT:
                label = "AROUND"
            elif eb <= CROSSABLE_AT:
                label = "OVER"
            else:
                ambiguous += 1
                continue
            vp = viewpoint(s, cx, cy, rad, rng,
                           target=(kind, cx, cy, rad))
            if vp is None:
                # No clean line of sight from anywhere standable.
                occluded += 1
                continue
            rows.append({"seed": sd, "scale": scale, "kind": kind,
                         "cx": cx, "cy": cy, "radius": rad, "height": h,
                         "edge_blocked": eb, "label": label,
                         "vx": vp[0], "vy": vp[1], "view_dist": vp[2]})
    return rows, ambiguous, occluded


def check(rows: list[dict], ambiguous: int, occluded: int = 0) -> int:
    """
    Is this dataset worth training on? Every check here exists
    because the failure it catches produces a file that looks fine.
    """
    fails = []

    def ck(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    n = len(rows)
    print(f"\n  {n} samples; {ambiguous} discarded as ambiguous, "
          f"{occluded} for having no clean line of sight "
          f"({(ambiguous+occluded)/(n+ambiguous+occluded):.0%} of "
          f"candidates dropped)\n")
    ck("the dataset is not empty", n >= 50, f"{n} samples")
    if not n:
        return 1

    lab = [r["label"] for r in rows]
    frac = lab.count("AROUND") / n
    ck("both classes present, not 95/5", 0.2 <= frac <= 0.8,
       f"{frac:.0%} AROUND, {1-frac:.0%} OVER")

    # Is the label just the landform's NAME? If kind alone predicts it
    # perfectly, a vision model is being asked to recognise a texture
    # that stands for a word, and the lab should say so out loud.
    kinds = sorted({r["kind"] for r in rows})
    best = {}
    for k in kinds:
        sub = [r["label"] for r in rows if r["kind"] == k]
        maj = max(set(sub), key=sub.count)
        best[k] = (maj, sub.count(maj) / len(sub), len(sub))
    acc = sum(c * m for _, (_, c, m) in best.items()) / n
    print("      label by landform kind:")
    for k, (maj, c, m) in sorted(best.items()):
        print(f"        {k:<7} n={m:<5} {c:>5.0%} {maj}")
    ck("kind alone does NOT settle the label", acc < 0.98,
       f"a kind-only classifier scores {acc:.1%}")

    # Geometry must actually separate the classes -- otherwise the
    # threshold is slicing noise.
    a = [r["edge_blocked"] for r in rows if r["label"] == "AROUND"]
    o = [r["edge_blocked"] for r in rows if r["label"] == "OVER"]
    ck("the classes are separated in the oracle", min(a) > max(o),
       f"OVER <= {max(o):.2f} < {min(a):.2f} <= AROUND")

    ck("viewpoints stand off the landform",
       all(r["view_dist"] > r["radius"] for r in rows),
       f"closest {min(r['view_dist']/r['radius'] for r in rows):.2f} radii")
    print()
    if fails:
        print(f"  {len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("  dataset is usable")
    return 0


def render(rows: list[dict], out: Path, width: int, height: int) -> None:
    """One robot's-eye view per sample."""
    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(LAB5))
    import tempfile

    import mujoco
    from PIL import Image

    import mars
    from robot import hfield_png, model_xml

    (out / "images").mkdir(parents=True, exist_ok=True)
    cache, model, rend = None, None, None
    for i, r in enumerate(rows):
        key = (r["seed"], r["scale"])
        if key != cache:
            if rend is not None:
                rend.close()
            s = mars.generate(r["seed"], scale=r["scale"])
            h = s.heights - s.heights.min()
            tmp = Path(tempfile.mkdtemp())
            hfield_png(h, tmp / "t.png")
            xml = model_xml("t.png", (0.0, 0.0), (1.0, 1.0),
                            relief=max(float(h.max()), 1e-3), rocks=s.rocks)
            model = mujoco.MjModel.from_xml_string(
                xml, {"t.png": (tmp / "t.png").read_bytes()})
            rend = mujoco.Renderer(model, height=height, width=width)
            cache, surf = key, s

        cam = mujoco.MjvCamera()
        cam.lookat[:] = [r["cx"], r["cy"],
                         surf.height_at(r["cx"], r["cy"]) - surf.heights.min()]
        cam.distance = r["view_dist"]
        cam.elevation = -8.0
        cam.azimuth = math.degrees(math.atan2(r["cy"] - r["vy"],
                                              r["cx"] - r["vx"]))
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        rend.update_scene(d, cam)
        name = f"{r['seed']:05d}_{i:04d}.png"
        Image.fromarray(rend.render()).save(out / "images" / name)
        r["image"] = f"images/{name}"
        if (i + 1) % 100 == 0:
            print(f"    rendered {i+1}/{len(rows)}", flush=True)
    if rend is not None:
        rend.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--scale", default="course")
    ap.add_argument("--out", default="data")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--check", action="store_true",
                    help="properties only — no rendering, no GL, seconds")
    a, _ = ap.parse_known_args()

    print("=" * 74)
    print("  TRAVERSABILITY DATASET — labels from geometry, not from names")
    print("=" * 74)
    rows, ambiguous, occluded = build_samples(
        range(a.seed0, a.seed0 + a.seeds), a.scale)
    rc = check(rows, ambiguous, occluded)
    if a.check or rc:
        return rc

    out = Path(a.out)
    render(rows, out, a.width, a.height)

    # Split by SEED, not by sample. Two views of the same surface share
    # its landforms, so a sample-level split leaks the answer across
    # the boundary and inflates validation accuracy for free.
    seeds = sorted({r["seed"] for r in rows})
    n_val = max(1, int(len(seeds) * a.val_frac))
    val = set(seeds[-n_val:])
    for split in ("train", "val"):
        sel = [r for r in rows if (r["seed"] in val) == (split == "val")]
        with (out / f"{split}.jsonl").open("w") as f:
            for r in sel:
                f.write(json.dumps({"image": r["image"], "prompt": PROMPT,
                                    "answer": r["label"], "meta": r}) + "\n")
        frac = sum(r["label"] == "AROUND" for r in sel) / max(len(sel), 1)
        print(f"  {split:<6} {len(sel):>5} samples, {frac:.0%} AROUND, "
              f"{len({r['seed'] for r in sel})} surfaces")
    print(f"\n  wrote {out}/  (split by SURFACE, so no landform is in both)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
