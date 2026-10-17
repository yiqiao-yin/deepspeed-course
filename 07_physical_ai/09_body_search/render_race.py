#!/usr/bin/env python3
"""
The race: every body from one start, trails drawn, overhead.

    uv run render_race.py
    uv run render_race.py --bodies bs_m2,bs_m4,bs_s8

WHY A RACE
----------
The headline of this lab is a DISTANCE: the best body walks several
times further than the worst. A table states that and a chase-camera
grid hides it, because every tile keeps its own robot centred and a
robot going nowhere looks much like a robot making progress.

Overhead with a trail, the difference is the picture. Same start,
same heading, same terrain, same budget — the only variable is the
machine, and the lines diverge immediately.

Each body is simulated separately and their trails are drawn on one
map, so this is a composite, not a physical race. Nothing interacts;
nobody is blocking anybody. The clock is shared (frame k is step k
for every body), which is what makes the comparison fair.

The trail is where the robot HAS BEEN, so a body that thrashes in
place draws a knot and a body that travels draws a line. That is
the distinction the number `walked` cannot show: two robots can
cover the same metres with one of them going somewhere.
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

# Distinct, colour-blind-safe-ish, and readable on orange terrain.
TRAIL = [(255, 255, 255), (120, 200, 255), (255, 210, 90), (150, 255, 170),
         (255, 140, 200), (170, 150, 255), (90, 240, 230), (255, 160, 110),
         (200, 255, 120), (240, 120, 120)]
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
    print("\n  No OpenGL backend; the race needs one for its backdrop.\n")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", default="runs/ranking.json")
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--top", type=int, default=6,
                    help="how many bodies, best and worst included")
    ap.add_argument("--seed", type=int, default=9003)
    ap.add_argument("--steps", type=int, default=4200)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", default="body-race")
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm
    from robot import make_layout

    if a.bodies:
        runs = [b.strip() for b in a.bodies.split(",") if b.strip()]
    else:
        p = Path(a.ranking)
        if not p.exists():
            print(f"\n  no {p}; run rank_bodies.py first\n")
            return 1
        rank = json.loads(p.read_text())          # sorted by distance
        # Best few AND the worst, so the picture spans the result
        # rather than showing a tight pack at the top.
        keep = rank[:max(1, a.top - 1)] + rank[-1:]
        runs, seen = [], set()
        for r in keep:
            if r["run"] not in seen:
                runs.append(r["run"])
                seen.add(r["run"])
    print(f"  {len(runs)} bodies racing from one start")

    tracks, labels, backdrop, extent = [], [], None, None
    for idx, run in enumerate(runs):
        meta = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
        spec = meta.get("body_spec") or {}
        layout = make_layout(**spec) if spec else None
        ck = torch.load(LAB5 / "runs" / run / "policy.pt", weights_only=False)
        net = ActorCritic(meta["obs_dim"], meta["act_dim"],
                          hidden=meta.get("hidden", 64))
        net.load_state_dict(ck["model"])
        net.eval()
        norm = RunningNorm(meta["obs_dim"])
        norm.load_state_dict(ck["norm"])
        env = NavWorld(mode=meta["mode"], flat=meta.get("flat", False),
                       seed=0, relief_scale=meta.get("relief_scale", 1.0),
                       upright_w=meta.get("upright_w", 0.0),
                       morphology=meta.get("morphology", "biped"),
                       integrator=meta.get("integrator"), layout=layout)
        obs, _ = env.reset(seed=a.seed)
        if backdrop is None:
            # One overhead plate of the terrain, shot once and reused
            # so every trail is drawn on the SAME map.
            r = mujoco.Renderer(env.model, height=a.height, width=a.width)
            cam = mujoco.MjvCamera()
            cam.orthographic = 1
            cam.distance = env.map.extent * 0.62
            cam.elevation, cam.azimuth = -90.0, 90.0
            cam.lookat[:] = [0, 0, 0]
            r.update_scene(env.data, cam)
            backdrop = Image.fromarray(r.render()).convert("RGB")
            extent = cam.distance
            r.close()
        pts = []
        for t in range(a.steps):
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, _i = env.step(act)
            if t % a.every == 0:
                pts.append(tuple(env.pos()))
            if term or trunc:
                break
        tracks.append(pts)
        d = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
        labels.append((f"{spec.get('n_legs','?')} legs "
                       f"{spec.get('posture','')}", d))
        print(f"    {run:<10} {len(pts):>4} points, {d:>5.1f} m", flush=True)

    W, H = a.width, a.height

    def to_px(p):
        # Orthographic overhead: world metres -> pixels.
        sx = W / (2 * extent)
        return (W / 2 + p[0] * sx, H / 2 - p[1] * sx)

    span = max(len(t) for t in tracks)
    frames = []
    for k in range(span):
        sheet = backdrop.copy()
        d = ImageDraw.Draw(sheet, "RGBA")
        for i, pts in enumerate(tracks):
            col = TRAIL[i % len(TRAIL)]
            upto = pts[:min(k + 1, len(pts))]
            if len(upto) > 1:
                d.line([to_px(p) for p in upto], fill=col + (235,), width=3)
            if upto:
                x, y = to_px(upto[-1])
                d.ellipse([x - 5, y - 5, x + 5, y + 5],
                          fill=col + (255,), outline=(10, 16, 22, 255))
        # Legend, ordered as given (best first).
        d.rectangle([0, 0, 232, 20 + 18 * len(labels)], fill=(10, 16, 22, 215))
        d.text((9, 5), f"step {min(k, span-1) * a.every:>4}", fill=INK)
        for i, (name, dist) in enumerate(labels):
            col = TRAIL[i % len(TRAIL)]
            run_d = sum(math.dist(p, q) for p, q in
                        zip(tracks[i][:k + 1], tracks[i][1:k + 1]))
            d.rectangle([9, 24 + 18 * i, 21, 34 + 18 * i], fill=col + (255,))
            d.text((27, 23 + 18 * i), f"{name:<16} {run_d:>5.1f} m",
                   fill=INK)
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
