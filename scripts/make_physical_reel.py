#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0"]
# ///
"""
The seven-lab reel for the Physical AI overview page.

    uv run scripts/make_physical_reel.py
    uv run scripts/make_physical_reel.py --seconds 3

An AUTHORING TOOL, not a lab: it carries PEP 723 inline metadata and
deliberately no `pyproject.toml`, because a pyproject would oblige an
entry in `clawdeck.yaml` (CI enforces it) and a learner should never
be offered "run the figure generator" as an exercise.

WHAT IT BUILDS
--------------
The overview page is the entry point to seven labs and had no image
on it at all -- a page that says "these labs produce animations" and
shows none. This stitches one clip from each lab, in order, with a
caption naming what that lab measured.

It COMPOSES EXISTING GIFs rather than re-rendering anything. Every
frame here has already been published on the lab's own page, so the
reel cannot drift from the labs: if a lab's animation is regenerated,
rerunning this picks up the new one. Nothing is simulated, trained or
measured here.

Clips are sampled across each source's full length rather than taken
from the start, because the first second of a rollout is a robot
standing still.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docusaurus-docs" / "static" / "img" / "physical"

# (source gif, lab label, what that lab measured)
LABS = [
    ("hopper-after.gif", "1 · Obstacle Hopper",
     "the one-off bonus must be ONE-OFF"),
    ("stairs-showcase.gif", "2 · Biped Stairs",
     "0.04-0.10 m treads climb; above that they do not"),
    ("terrain-up.gif", "3 · Terrain Vision",
     "the camera helps on ascent and nowhere else"),
    ("nav-route.gif", "4 · Terrain Navigation",
     "climb it, or walk around it"),
    ("mars-p4-iso.gif", "5 · Mars Terrain",
     "a planet, and a policy that crosses 1 episode in 7"),
    ("mjx-swarm.gif", "6 · Mars on GPU Physics",
     "GPU physics 1.50x a CPU pool — bar was 5x"),
    ("vlm-confusion.gif", "7 · Terrain VLM",
     "sight helps (p=0.006) and still loses to touch"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.5,
                    help="screen time per lab")
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--out", default="physical-reel")
    a, _ = ap.parse_known_args()

    from PIL import Image, ImageDraw

    per = max(2, int(a.seconds * 1000 / a.frame_ms))
    frames = []
    missing = []

    for src, label, measured in LABS:
        path = IMG / src
        if not path.exists():
            # Loudly, and listed again at the end. A reel that silently
            # skips a lab still looks complete.
            print(f"  MISSING {src} — lab '{label}' will not appear")
            missing.append(src)
            continue
        im = Image.open(path)
        n = getattr(im, "n_frames", 1)
        # Sample ACROSS the clip: the first second of a rollout is a
        # robot standing still, which makes a poor advertisement.
        picks = [int(i * (n - 1) / max(per - 1, 1)) for i in range(per)]
        for k in picks:
            im.seek(k)
            fr = im.convert("RGB").resize((a.width, a.height))
            d = ImageDraw.Draw(fr, "RGBA")
            # Caption at the BOTTOM: every source lab draws its own
            # HUD in the top-left, and a banner there sat on top of it.
            d.rectangle([0, a.height - 44, a.width, a.height],
                        fill=(10, 16, 22, 232))
            d.text((14, a.height - 38), label, fill=(236, 242, 248, 255))
            d.text((14, a.height - 20), measured, fill=(150, 170, 186, 255))
            frames.append(fr)
        print(f"  {label:<26} {per:>3} frames from {src} ({n} available)")

    if not frames:
        print("\n  no source animations found — nothing to build")
        return 1

    out = IMG / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(out, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {out.name}  ({len(frames)} frames, "
          f"{out.stat().st_size/1024/1024:.1f} MB, "
          f"{len(LABS) - len(missing)}/{len(LABS)} labs)")
    if missing:
        print(f"  INCOMPLETE — missing: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
