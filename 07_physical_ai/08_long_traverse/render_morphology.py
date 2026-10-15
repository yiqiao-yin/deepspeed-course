#!/usr/bin/env python3
"""
Five bodies, one hillside: what a leg count actually buys.

    uv run render_morphology.py
    uv run render_morphology.py --bodies biped,spider8 --seconds 14

WHAT THE PICTURE IS FOR
-----------------------
The comparison table says the biped falls 41.7% of the time and the
quadruped 8.3%. That is the right number and it does not show you
WHY, because the why is a shape: where the feet are relative to the
body, and how much the torso has to be actively caught.

So every panel is the same Mars terrain, the same seed, the same
task, the same training budget -- and a different body, each running
its own policy. The only variable is the machine.

Watch for the thing the numbers cannot carry: the biped is
continuously catching itself and the spider simply is not. A
sprawling body is statically stable, so staying upright is not a
behaviour it has to produce.

WHAT IT DOES NOT SHOW
---------------------
Speed and efficiency, which is where the sprawlers are expected to
LOSE. A spider's feet sit far from its centre of mass, so its joints
resist large moments and its body sways side to side; that is why
sprawling posture is energetically expensive in real animals too.
The clip shows stability, the table shows the price, and neither is
the whole result on its own.

Rendered one body at a time and composited. Five live MuJoCo
renderers means five OpenGL contexts, which corrupts the heap under
GLFW -- lab 6 found that the hard way.
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
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("egl", "osmesa", "glfw")

if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))

INK = (236, 242, 248, 255)
GREY = (150, 170, 186, 255)
GREEN = (92, 196, 141, 255)
RED = (226, 110, 110, 255)


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
    print("\n  No OpenGL backend. The numbers need none:\n"
          "      uv run compare_bodies.py\n")
    sys.exit(1)


def load(run: str):
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
    ck = torch.load(LAB5 / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta["act_dim"],
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return net, norm, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies",
                    default="biped,quadruped,hexapod,spider6,spider8")
    ap.add_argument("--prefix", default="morph_")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--tile-w", type=int, default=340)
    ap.add_argument("--tile-h", type=int, default=250)
    ap.add_argument("--view", default="chase", choices=("chase", "iso"))
    ap.add_argument("--out", default="morphology")
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    from mars_env import NavWorld

    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()]
    missing = [b for b in bodies
               if not (LAB5 / "runs" / f"{a.prefix}{b}" / "summary.json").exists()]
    if missing:
        # Loud, not a silently shorter reel.
        print(f"\n  no trained policy for: {', '.join(missing)}\n"
              f"  train them first:\n"
              f"      uv run train_mars.py --morphology <body> "
              f"--name {a.prefix}<body>\n")
        bodies = [b for b in bodies if b not in missing]
        if not bodies:
            return 1

    strips, stats = [], []
    for body in bodies:
        net, norm, meta = load(f"{a.prefix}{body}")
        env = NavWorld(mode=meta["mode"], flat=meta.get("flat", False),
                       seed=0, relief_scale=meta.get("relief_scale", 1.0),
                       upright_w=meta.get("upright_w", 0.0),
                       morphology=body)
        obs, _ = env.reset(seed=a.seed)
        r = mujoco.Renderer(env.model, height=a.tile_h, width=a.tile_w)
        cam = mujoco.MjvCamera()
        shots, falls, travelled = [], 0, 0.0
        prev = env.pos().copy()
        for t in range(a.steps):
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(act)
            now = env.pos()
            travelled += float(np.linalg.norm(now - prev))
            prev = now.copy()
            if env.fallen():
                falls += 1
            if t % a.every == 0:
                px, py = env.pos()
                if a.view == "chase":
                    cam.distance, cam.elevation = 4.6, -17.0
                    cam.azimuth = math.degrees(env.yaw()) + 180.0
                    cam.lookat[:] = [px, py, env._h_at(px, py) + 0.35]
                else:
                    cam.distance, cam.elevation, cam.azimuth = 5.5, -26.0, 130.0
                    cam.lookat[:] = [px, py, env._h_at(px, py)]
                r.update_scene(env.data, cam)
                img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img, "RGBA")
                d.rectangle([0, 0, a.tile_w, 40], fill=(10, 16, 22, 225))
                d.text((8, 4), f"{body}   {env.act_dim} actuators", fill=INK)
                d.text((8, 21),
                       f"stance {env.stance_height():.2f} m   "
                       f"tilt {env.tilt_deg():>3.0f}°", fill=GREY)
                d.rectangle([0, a.tile_h - 20, a.tile_w, a.tile_h],
                            fill=(10, 16, 22, 225))
                col = RED if env.fallen() else GREEN
                d.text((8, a.tile_h - 17),
                       f"walked {travelled:>5.1f} m", fill=INK)
                d.text((a.tile_w - 96, a.tile_h - 17),
                       "DOWN" if env.fallen() else "upright", fill=col)
                shots.append(img)
            if term or trunc:
                break
        r.close()
        strips.append(shots)
        stats.append((body, env.act_dim, travelled, falls / max(t, 1)))
        print(f"    {body:<10} {env.act_dim:>3} act   {len(shots):>4} frames   "
              f"walked {travelled:>5.1f} m   down {falls/max(t,1):>5.1%} of steps")

    cols = len(strips)
    span = max(len(s) for s in strips)
    W, H = a.tile_w, a.tile_h
    frames = []
    for k in range(span):
        sheet = Image.new("RGB", (cols * W, H), (8, 12, 16))
        for i, shots in enumerate(strips):
            sheet.paste(shots[min(k, len(shots) - 1)], (i * W, 0))
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
