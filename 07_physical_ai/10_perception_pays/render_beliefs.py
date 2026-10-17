#!/usr/bin/env python3
"""
Belief vs reality: three routers, one map, the routes they plan.

    uv run render_beliefs.py                 # the three-panel comparison
    uv run render_beliefs.py --seed 7

WHAT THE PICTURE SHOWS
----------------------
The sweep's headline is a non-result: perfect perception ties no
perception at 25% arrival, and the realistic biased router is WORSE
than blind. A table states that; it does not show why.

Three panels, same terrain, same start and goal, same body. Each
plans under a different BELIEF and the planned route is drawn before
the robot moves:

    oracle   believes the truth          -- the careful route
    biased   lab 7's real failure shape  -- cries wolf, detours 1.27x
    blind    believes nothing blocks     -- the straight line

Then the robot walks whichever route its belief produced, and the
trail shows how far it actually got. The point is the gap between
the two: the oracle's careful 1.09x avoidance buys nothing because
the robot cannot execute a route well enough for route quality to
matter, and the biased router's extra 27% of detour is pure loss.

Rendered one panel at a time and composited -- concurrent MuJoCo
renderers mean concurrent GL contexts, which corrupts the heap under
GLFW (lab 6 found that the hard way).
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
LAB8 = HERE.parent / "08_long_traverse"
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("egl", "osmesa", "glfw")
for _p in (LAB5, LAB8, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

INK = (236, 242, 248, 255)
GREY = (150, 170, 186, 255)
GREEN = (92, 196, 141, 255)
AMBER = (224, 170, 94, 255)
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
          "      uv run perception_sweep.py\n")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="bs_m2")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=380)
    ap.add_argument("--out", default="perception-beliefs")
    ap.add_argument("--frame-ms", type=int, default=80)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    import perception_sweep as P
    import traverse as T
    from mars_env import NavWorld
    from robot import make_layout

    T.WAYPOINT_SPACING = P.WAYPOINT_GAP
    T.WAYPOINT_RADIUS = max(0.8, P.WAYPOINT_GAP * 0.4)

    task = P.belief_sensitive_task(a.seed, P.ROUTE_BAND)
    if task is None:
        print(f"  no belief-sensitive task at seed {a.seed}")
        return 1
    surf, A, B, L, gap = task
    surf.start, surf.goal = A, B
    print(f"  seed {a.seed}: {L:.1f} m route, belief gap {gap:.1f} m")

    meta = json.loads((LAB5 / "runs" / a.body / "summary.json").read_text())
    layout = make_layout(**meta["body_spec"])
    net, norm, _m = T.load_policy(a.body)
    truth = T.truth_blocked(surf)

    panels = [("oracle", T.Router("oracle"), GREEN),
              ("biased (lab 7)", T.BiasedRouter("b"), AMBER),
              ("blind", T.BlindRouter("blind"), RED)]

    strips = []
    for label, router, col in panels:
        rng = np.random.default_rng(50_000 + a.seed)
        belief = [router.verdict(t, rng) for t in truth]
        path, planned = T.plan(surf, A, B, T.belief_mask(surf, belief))
        _p, oracle_len = T.plan(surf, A, B, T.belief_mask(surf, truth))

        env = NavWorld(mode=meta["mode"], flat=False, seed=0,
                       relief_scale=meta.get("relief_scale", 1.0),
                       upright_w=meta.get("upright_w", 0.0),
                       integrator=meta.get("integrator"), layout=layout)
        T.place(env, surf, A)

        r = mujoco.Renderer(env.model, height=a.height, width=a.width)
        cam = mujoco.MjvCamera()
        cam.orthographic = 1
        cam.distance = surf.extent * 0.55
        cam.elevation, cam.azimuth = -90.0, 90.0
        cam.lookat[:] = [0, 0, 0]
        r.update_scene(env.data, cam)
        plate = Image.fromarray(r.render()).convert("RGB")
        W, H = a.width, a.height
        sx = W / (2 * cam.distance)

        def px(p):
            return (W / 2 + p[0] * sx, H / 2 - p[1] * sx)

        wps = T.waypoints(path) if path else []
        trail, shots = [], []
        obs = env._obs()
        reached = 0
        wrong_over = sum(1 for t, b in zip(truth, belief) if t and not b)
        wrong_around = sum(1 for t, b in zip(truth, belief) if b and not t)

        done = False
        for wp in wps:
            if done:
                break
            env.map.goal = (float(wp[0]), float(wp[1]))
            best, since = float("inf"), 0
            while True:
                with torch.no_grad():
                    x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                    act = net.distribution(x).mean.squeeze(0).numpy()
                obs, _, _t, _tr, _i = env.step(act)
                if env.fallen():
                    T.recover(env)
                    obs = env._obs()
                p = env.pos()
                if len(trail) % 1 == 0:
                    trail.append(tuple(p))
                d = env.to_goal()
                if d < best - 0.05:
                    best, since = d, 0
                else:
                    since += 1
                if len(trail) % a.every == 0:
                    img = plate.copy()
                    dr = ImageDraw.Draw(img, "RGBA")
                    if path:
                        dr.line([px(q) for q in path], fill=col[:3] + (160,),
                                width=2)
                    for q in wps:
                        qx, qy = px(q)
                        dr.ellipse([qx-3, qy-3, qx+3, qy+3],
                                   outline=col[:3] + (220,))
                    if len(trail) > 1:
                        dr.line([px(q) for q in trail],
                                fill=(255, 255, 255, 240), width=3)
                    ax, ay = px(A)
                    bx, by = px(B)
                    dr.ellipse([ax-5, ay-5, ax+5, ay+5], fill=(120, 200, 255, 255))
                    dr.ellipse([bx-6, by-6, bx+6, by+6], outline=(120, 255, 170, 255), width=3)
                    dr.rectangle([0, 0, W, 54], fill=(10, 16, 22, 230))
                    dr.text((8, 4), label, fill=col)
                    dr.text((8, 20), f"planned {planned:.1f} m  "
                                     f"({planned/max(oracle_len,1e-9):.2f}x oracle)",
                            fill=GREY)
                    dr.text((8, 36), f"wrong OVER {wrong_over}   "
                                     f"wrong AROUND {wrong_around}", fill=GREY)
                    dr.rectangle([0, H-20, W, H], fill=(10, 16, 22, 230))
                    dr.text((8, H-17), f"waypoints {reached}/{len(wps)}",
                            fill=INK)
                    shots.append(img)
                if d < T.WAYPOINT_RADIUS:
                    reached += 1
                    break
                if since > 600 or len(trail) > a.steps:
                    done = True
                    break
        r.close()
        strips.append(shots)
        print(f"    {label:<16} planned {planned:>5.1f} m, "
              f"reached {reached}/{len(wps)}, {len(shots)} frames", flush=True)

    span = max(len(s) for s in strips)
    W, H = a.width, a.height
    frames = []
    for k in range(span):
        sheet = Image.new("RGB", (3 * W, H), (8, 12, 16))
        for i, shots in enumerate(strips):
            if shots:
                sheet.paste(shots[min(k, len(shots) - 1)], (i * W, 0))
        frames.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    path_out = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path_out, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {path_out.name}  ({len(frames)} frames, "
          f"{path_out.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
