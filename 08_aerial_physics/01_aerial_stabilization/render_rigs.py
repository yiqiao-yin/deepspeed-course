#!/usr/bin/env python3
"""
The rig zoo: every nozzle layout, turning, with its authority printed.

    uv run render_rigs.py
    uv run render_rigs.py --rigs reveal,waisted --out aerial-pair

WHAT THE PICTURE IS FOR
------------------------
The sweep's result is that neither nozzle COUNT nor full RANK
predicts whether a rig can land, and a table states that without
showing it. Side by side, `reveal` (10 nozzles, rank 6) and `dense8`
(18 nozzles, rank 6) look like healthy rigs and neither can produce
any backward force at all: every horizontal nozzle pushes the same
way. `fore10` has twelve and can.

So the caption is the REACH -- what the rig cannot produce at any
command -- not the rank. Captioning rank would print the same green
"6/6" on the rig that lands 75% and the one that lands 98%, which is
exactly the mistake this lab made.

Each tile is the real compiled model with its real nozzle bells, lit
and rotating on the spot. Rendered one at a time and composited --
concurrent GL contexts corrupt the heap under GLFW.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"
BACKENDS = ("egl", "osmesa", "glfw")
INK = (236, 242, 248)
DIM = (150, 170, 186)
GOOD = (108, 206, 148)
BAD = (228, 108, 108)


def pick_backend() -> str:
    forced = os.environ.get("MUJOCO_GL")
    probe = ("import mujoco;"
             "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>"
             "<geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
             "r=mujoco.Renderer(m,64,64);d=mujoco.MjData(m);"
             "mujoco.mj_forward(m,d);r.update_scene(d);r.render()")
    for b in ((forced,) if forced else BACKENDS):
        env = dict(os.environ, MUJOCO_GL=b, PYOPENGL_PLATFORM=b)
        try:
            subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                           capture_output=True, timeout=120)
            print(f"  [render] MUJOCO_GL={b}")
            return b
        except Exception:                                   # noqa: BLE001
            pass
    print("\n  No OpenGL backend. `uv run thrusters.py --all` needs none.\n")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rigs",
                    default="minimal,waisted,sixlift,narrow,reveal,wide,dense")
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--tile-w", type=int, default=330)
    ap.add_argument("--tile-h", type=int, default=270)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--fire", action="store_true", default=True,
                    help="pulse each rig's maximum pure roll couple")
    ap.add_argument("--out", default="aerial-rigs")
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))

    import mujoco
    from PIL import Image, ImageDraw

    import car as CAR
    import thrusters as TH
    from render_flight import add_plumes

    names = [r.strip() for r in a.rigs.split(",") if r.strip()]
    W, H = a.tile_w, a.tile_h
    strips = []

    for nm in names:
        rig = TH.rig(nm)
        d_aud = TH.audit(nm, rig)
        model, _r, _s = CAR.build(nm, flat=True, start_z=0.9)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # The commands that produce this rig's biggest PURE roll
        # couple. For a rank-deficient rig there is none, and the
        # picture shows nothing firing -- which is the point.
        B = TH.effectiveness(rig)
        _tau, u = TH.max_pure_couple(B, 0)
        u = np.zeros(len(rig)) if u is None else u

        class _Fake:
            """The plume drawer wants an env; it needs four accessors."""
            def __init__(self, model, data, rig):
                self.rig, self._d = rig, data
                self.act_dim = len(rig)
                self.u_act = np.zeros(len(rig))

            def rotmat(self):
                q = self._d.qpos[3:7]
                w, x, y, z = q
                return np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)]])

            def pos(self):
                return self._d.qpos[:3].copy()

        fake = _Fake(model, data, rig)
        r = mujoco.Renderer(model, height=H, width=W)
        cam = mujoco.MjvCamera()
        cam.distance = 7.4
        cam.elevation = -17.0
        cam.lookat[:] = [0, 0, 0.5]

        shots = []
        for k in range(a.frames):
            cam.azimuth = 360.0 * k / a.frames
            fake.u_act = u * (0.5 + 0.5 * math.sin(2 * math.pi * k /
                                                   a.frames * 2)) \
                if a.fire else np.zeros(len(rig))
            r.update_scene(data, cam)
            add_plumes(r.scene, fake, mujoco)
            img = Image.fromarray(r.render()).convert("RGB")
            dr = ImageDraw.Draw(img, "RGBA")
            dr.rectangle([0, 0, W, 34], fill=(8, 12, 18, 220))
            dr.text((8, 4), f"{nm}", fill=INK)
            dr.text((8, 19), f"{d_aud['n_thrusters']} nozzles", fill=DIM)
            dr.rectangle([0, H - 52, W, H], fill=(8, 12, 18, 225))
            # The caption is the REACH, not the rank. `reveal` is
            # rank 6 and cannot brake; captioning rank would put the
            # same green "6/6" on the rig that lands 75% and the one
            # that lands 98%, which is the error this lab made.
            dead = TH.dead_axes(TH.effectiveness(rig))
            can_brake = "-Fx" not in dead
            bcol = GOOD if can_brake else BAD
            dr.text((8, H - 49), f"rank {d_aud['rank']}/6", fill=DIM)
            dr.text((78, H - 49),
                    "brake: YES" if can_brake else "brake: NO",
                    fill=bcol)
            dr.text((8, H - 35),
                    f"pure roll {d_aud['roll_pure_Nm']:>7,.0f} N·m",
                    fill=INK if d_aud["roll_pure_Nm"] > 0 else BAD)
            dr.text((8, H - 21),
                    "cannot produce: " + (", ".join(dead) if dead
                                          else "nothing"),
                    fill=BAD if dead else GOOD)
            shots.append(img)
        r.close()
        strips.append(shots)
        print(f"    {nm:<10} {d_aud['n_thrusters']:>2} nozzles, "
              f"rank {d_aud['rank']}/6, {len(shots)} frames", flush=True)

    cols = min(a.cols, len(strips))
    rows = (len(strips) + cols - 1) // cols
    frames = []
    for k in range(a.frames):
        sheet = Image.new("RGB", (cols * W, rows * H), (8, 12, 16))
        for i, sh in enumerate(strips):
            sheet.paste(sh[k], ((i % cols) * W, (i // cols) * H))
        frames.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=128,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
