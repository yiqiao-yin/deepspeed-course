#!/usr/bin/env python3
"""
The body zoo: every shape the search built, turning on the spot.

    uv run render_zoo.py                    # all searched bodies
    uv run render_zoo.py --bodies bs_s3,bs_m4

WHY A TURNTABLE AND NOT MORE WALKING
-------------------------------------
The ranked grid shows the bodies WORKING, which is the result. It is
a poor way to see what they ARE: tiles are small, the camera chases,
and a 10-legged mammal and an 8-legged sprawl look much alike at
300 px while scrambling over a ridge.

This is the other half. Each body stands still on flat ground under
a slowly orbiting camera, so the shape is legible: where the hips
are, how far the feet reach, how low the chassis rides. Nothing is
being measured here -- it is a catalogue, and the page says so.

Flat ground, deliberately: terrain is the subject of every other
animation in this topic and here it would only hide the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("egl", "osmesa", "glfw")
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))

INK = (236, 242, 248, 255)
GREY = (150, 170, 186, 255)


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
            print(f"  [render] MUJOCO_GL={b}")
            return b
        except Exception:                                   # noqa: BLE001
            pass
    print("\n  No OpenGL backend; the zoo is pictures only.\n")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=None,
                    help="comma list of run names; default every bs_*")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-w", type=int, default=300)
    ap.add_argument("--tile-h", type=int, default=240)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--out", default="body-zoo")
    ap.add_argument("--frame-ms", type=int, default=80)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from mars_env import NavWorld
    from robot import make_layout

    if a.bodies:
        runs = [b.strip() for b in a.bodies.split(",") if b.strip()]
    else:
        runs = sorted(p.name for p in (LAB5 / "runs").glob("bs_*")
                      if (p / "summary.json").exists())
    if not runs:
        print("  no bs_* runs; train some first")
        return 1
    print(f"  {len(runs)} bodies")

    strips = []
    for run in runs:
        meta = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
        spec = meta.get("body_spec") or {}
        layout = make_layout(**spec) if spec else None
        # FLAT ground: the shape is the subject here, not the terrain.
        env = NavWorld(mode=meta["mode"], flat=True, seed=0,
                       morphology=meta.get("morphology", "biped"),
                       integrator=meta.get("integrator"), layout=layout)
        env.reset(seed=1)
        zero = np.zeros(env.act_dim)
        for _ in range(a.settle):           # let it find its stance
            env.step(zero)
        r = mujoco.Renderer(env.model, height=a.tile_h, width=a.tile_w)
        cam = mujoco.MjvCamera()
        px, py = env.pos()
        shots = []
        for k in range(a.frames):
            cam.distance, cam.elevation = 2.4, -18.0
            cam.azimuth = 360.0 * k / a.frames
            cam.lookat[:] = [px, py, env._h_at(px, py) + 0.20]
            r.update_scene(env.data, cam)
            img = Image.fromarray(r.render())
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([0, 0, a.tile_w, 38], fill=(10, 16, 22, 228))
            d.text((7, 3), f"{spec.get('n_legs','?')} legs  "
                           f"{spec.get('posture','')}", fill=INK)
            d.text((7, 20), f"{env.act_dim} actuators   stance "
                            f"{env.stance_height():.2f} m", fill=GREY)
            shots.append(img)
        r.close()
        strips.append(shots)
        print(f"    {run:<10} {spec.get('n_legs','?')} legs "
              f"{spec.get('posture',''):<7} {env.act_dim:>3} act", flush=True)

    cols = min(a.cols, len(strips))
    rows_n = math.ceil(len(strips) / cols)
    W, H = a.tile_w, a.tile_h
    frames = []
    for k in range(a.frames):
        sheet = Image.new("RGB", (cols * W, rows_n * H), (8, 12, 16))
        for i, shots in enumerate(strips):
            sheet.paste(shots[k], ((i % cols) * W, (i // cols) * H))
        frames.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
