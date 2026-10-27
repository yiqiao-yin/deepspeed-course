#!/usr/bin/env python3
"""
Film a trained policy. Part 2's animations.

    uv run render_run.py --run curr_flat --flat --out mars-walk-flat
    uv run render_run.py --run curr_flat --out mars-walk-rough

Every parameter of the task comes from the RUN'S OWN `summary.json` --
observation mode, goal range, terrain relief scale. Lab 4 shipped a
renderer that rebuilt its world from defaults instead, and filmed
goals no policy had ever been trained to reach while the published
table reported something else entirely. A clip that disagrees with the
number beside it is not a rendering quirk.
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
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("egl", "osmesa", "glfw")


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
            print(f"  [render] using MUJOCO_GL={b}")
            return b
        except Exception:                                   # noqa: BLE001
            print(f"  [render] MUJOCO_GL={b} unavailable")
    print("\n  No OpenGL backend. The measurements do not need one:\n")
    print("      uv run mars.py --check 20\n")
    sys.exit(1)


def hud(frame, env, info, mode, step, relief_scale, flat):
    """
    The panel. Every row is something a reader can check against the
    picture, which is the point: a HUD that cannot be falsified by
    looking at the frame is decoration.
    """
    from PIL import Image, ImageDraw

    import numpy as _np

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    gx, gy = env.map.goal
    bearing = math.atan2(gy - env.pos()[1], gx - env.pos()[0])
    off = math.degrees((bearing - env.yaw() + math.pi) % (2 * math.pi) - math.pi)
    speed = float(_np.hypot(env.data.qvel[env.q["rootx"]],
                            env.data.qvel[env.q["rooty"]]))
    pitch = math.degrees(abs(float(env.data.qpos[env.q["rootpitch"]]))) \
        if "rootpitch" in env.q else 0.0
    roll = math.degrees(abs(float(env.data.qpos[env.q["rootroll"]]))) \
        if "rootroll" in env.q else 0.0
    # Efficiency, frozen at arrival and against the GEODESIC -- the same
    # definition the published table uses. A clip that scores itself
    # differently from the page is how lab 4 shipped a figure that
    # disagreed with its own caption.
    den = info.get("travelled_to_goal") or info["travelled"]
    eff = (env.geodesic_len / den) if (info["arrived"] and den > 0.1) else None
    rows = [("mode", mode),
            ("ground", "flat" if flat else
             f"Mars {_slope_deg(env):.0f}° mean slope"),
            ("step", f"{step} / {env.max_steps}"),
            ("to goal", f"{info['to_goal']:.1f} m"),
            ("shortest path", f"{env.geodesic_len:.1f} m"),
            ("walked", f"{info['travelled']:.1f} m"),
            ("efficiency", f"{eff:.0%}" if eff else "--"),
            ("speed", f"{speed:.2f} m/s"),
            ("heading err", f"{off:+.0f}°"),
            ("torso pitch/roll", f"{pitch:.0f}° / {roll:.0f}°"),
            ("clearance", f"{env.clearance():.2f} m")]
    w, h = 248, 26 + 19 * len(rows)
    d.rounded_rectangle([12, 12, 12 + w, 12 + h], 8,
                        fill=(14, 22, 30, 225), outline=(60, 80, 96, 255))
    y = 23
    for k, v in rows:
        d.text((24, y), k, fill=(150, 170, 186, 255))
        d.text((12 + w - 14 - d.textlength(str(v)), y), str(v),
               fill=(236, 242, 248, 255))
        y += 19
    tag = ("ARRIVED" if info["arrived"] else
           "FELL" if info["fell"] else None)
    if tag:
        col = (92, 196, 141, 255) if info["arrived"] else (226, 110, 110, 255)
        tw = d.textlength(tag) + 26
        d.rounded_rectangle([frame.shape[1] - tw - 16, frame.shape[0] - 46,
                             frame.shape[1] - 16, frame.shape[0] - 16], 7,
                            fill=(14, 22, 30, 225), outline=col)
        d.text((frame.shape[1] - tw + 1, frame.shape[0] - 39), tag, fill=col)
    return np.array(img)


def _slope_deg(env) -> float:
    """Mean slope of the surface actually loaded, not a constant."""
    import numpy as _np

    h = env._heights.astype(float)
    cell = env.map.extent / h.shape[0]
    gy, gx = _np.gradient(h, cell)
    return float(_np.degrees(_np.arctan(_np.hypot(gx, gy))).mean())


def _terrain_panel(d, frame, env):
    """
    What the privileged channel sees: worst rise ahead / left / right,
    and whether the gait can clear it. Green under the 0.12 m step
    limit, red over -- which is the whole climb-or-go-around decision,
    drawn.

    Shown for every arm, because the question "did it go over or
    around" is about BEHAVIOUR and applies to a blind policy too. The
    panel is labelled as an observer's readout, not as the robot's
    input, so a blind run cannot be misread as having seen it.
    """
    feats = env.terrain_feats()
    H = frame.shape[0]
    x0, y0, w = 12, H - 104, 248
    d.rounded_rectangle([x0, y0, x0 + w, y0 + 88], 8,
                        fill=(14, 22, 30, 225), outline=(60, 80, 96, 255))
    d.text((x0 + 12, y0 + 8), "ground ahead (observer readout)",
           fill=(150, 170, 186, 255))
    for k, (lab, rise) in enumerate(zip(("left", "ahead", "right"),
                                        (feats[2], feats[0], feats[4]))):
        bx = x0 + 12 + k * 78
        ok = rise <= STEP_LIMIT
        col = (92, 196, 141, 255) if ok else (226, 110, 110, 255)
        d.rounded_rectangle([bx, y0 + 28, bx + 70, y0 + 56], 5,
                            outline=col, fill=(20, 30, 40, 200))
        d.text((bx + 12, y0 + 35), f"{rise:+.2f} m", fill=col)
        d.text((bx + 20, y0 + 62), lab, fill=(150, 170, 186, 255))


STEP_LIMIT = 0.12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="basename, no extension")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--relief-scale", type=float, default=None,
                    help="default: whatever the run was trained on")
    ap.add_argument("--seed", type=int, default=401)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--stop-after", type=int, default=250)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=440)
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--view", default="chase", choices=("chase", "iso", "orbit"))
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))
    import mujoco
    import torch
    from PIL import Image

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / a.run / "summary.json").read_text())
    scale = (a.relief_scale if a.relief_scale is not None
             else meta.get("relief_scale", 1.0))
    ck = torch.load(HERE / "runs" / a.run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta.get("act_dim", 7),
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    env = NavWorld(mode=meta["mode"], flat=a.flat, seed=0,
                   goal_range=meta.get("goal_range"), relief_scale=scale)
    obs, _ = env.reset(seed=a.seed)
    r = mujoco.Renderer(env.model, height=a.height, width=a.width)
    cam = mujoco.MjvCamera()
    frames, after = [], None
    step = 0
    while True:
        with torch.no_grad():
            t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
            act = net.distribution(t).mean.squeeze(0).numpy()
        obs, _, term, trunc, info = env.step(act)
        step += 1
        if step % a.every == 0 or term:
            x, y = env.pos()
            if a.view == "chase":
                cam.distance, cam.elevation = 5.0, -14.0
                cam.azimuth = math.degrees(env.yaw()) + 180.0
                cam.lookat[:] = [x, y, env._h_at(x, y) + 0.7]
            elif a.view == "iso":
                cam.distance, cam.elevation, cam.azimuth = 14.0, -34.0, 130.0
                cam.lookat[:] = [x, y, env._h_at(x, y)]
            else:
                cam.distance, cam.elevation = 18.0, -30.0
                cam.azimuth = (step * 0.6) % 360
                cam.lookat[:] = [x, y, env._h_at(x, y)]
            r.update_scene(env.data, cam)
            fr = r.render()
            from PIL import Image as _I, ImageDraw as _D
            _img = _I.fromarray(hud(fr, env, info, meta["mode"], step,
                                    scale, a.flat))
            _terrain_panel(_D.Draw(_img, "RGBA"), fr, env)
            frames.append(np.array(_img))
        # Stop filming shortly after arrival; the episode runs on.
        if info["arrived"] and after is None:
            after = 0
        elif after is not None:
            after += 1
            if after >= a.stop_after:
                break
        if term or trunc:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=a.colors,
                                      dither=Image.Dither.NONE)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KB)  "
          f"{'ARRIVED' if info['arrived'] else 'fell' if info['fell'] else 'timed out'}"
          f", walked {info['travelled']:.1f} m of a {env.route_len:.1f} m route")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
