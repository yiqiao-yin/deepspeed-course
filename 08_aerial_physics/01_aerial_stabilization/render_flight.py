#!/usr/bin/env python3
"""
Film a landing, with a HUD showing what the vehicle actually knows.

    uv run render_flight.py runs/r_reveal
    uv run render_flight.py runs/r_reveal --episodes 4 --out flight
    uv run render_flight.py --classical --rig reveal    # the baseline

WHAT IS DRAWN, AND WHY IT CAN BE TRUSTED
-----------------------------------------
Every frame is MuJoCo rendering the same `MjData` the policy just
acted on. There is no separate animation path, no interpolation of a
log and no second model. A renderer that is not the simulator can
show a car doing something the physics never did, and this course
has paid for that once already.

The HUD is drawn on top in PIL from the SAME step's state: altitude,
tilt, body rates, gas left, tank pressure, and the per-nozzle valve
openings as a bar strip. The nozzle strip is the interesting one --
it is the policy's actual output, and watching which nozzles fire
during a recovery is the only way to see what it learned.

PLUMES
------
Each firing nozzle gets a translucent cone scaled by its command,
drawn as a MuJoCo scene geom rather than a sprite. It is cosmetic
and it is honest: the cone length is the valve opening, so a nozzle
that is not firing does not draw one.

RENDERED SEQUENTIALLY
---------------------
One renderer at a time, composited afterwards. Concurrent MuJoCo
renderers mean concurrent GL contexts, which corrupts the heap under
GLFW -- topic 7 found that with sixteen of them and a crash that
looked like a MuJoCo bug.
"""

from __future__ import annotations

import argparse
import json
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
WARN = (226, 176, 92)
BAD = (228, 108, 108)
JET = (126, 198, 248)


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
    print("\n  No OpenGL backend available. The measurements need none:\n"
          "      uv run evaluate.py <run>\n")
    sys.exit(1)


def add_plumes(scene, env, mujoco) -> None:
    """One translucent cone per firing nozzle, length = valve opening."""
    R, p = env.rotmat(), env.pos()
    for k, t in enumerate(env.rig):
        u = float(env.u_act[k])
        if u < 0.04 or scene.ngeom >= scene.maxgeom:
            continue
        d_b = np.asarray(t.direction)
        base_w = p + R @ np.asarray(t.pos)
        dir_w = R @ (-d_b)                     # gas leaves this way
        # Plume length: short enough to read as exhaust rather
        # than as a pole. The first version used 0.25 + 2.3u and
        # a full-open roof nozzle drew a 2.5 m grey column that
        # dominated the frame.
        L = 0.15 + 0.62 * u
        mid = base_w + dir_w * (L * 0.5)
        # Orient a capsule along dir_w via the rotation taking +z to it.
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, dir_w)
        c = float(np.dot(z, dir_w))
        if np.linalg.norm(v) < 1e-8:
            M = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            M = np.eye(3) + vx + vx @ vx / (1 + c)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            # A plume is about four times as long as it is wide. At
        # 1:16 it rendered as a knitting needle.
        np.array([0.068 + 0.105 * u, L * 0.5, 0.0]),
            mid.astype(np.float64), M.flatten().astype(np.float64),
            np.array([0.72, 0.88, 1.0, 0.13 + 0.34 * u], dtype=np.float32))
        scene.ngeom += 1


def hud(img, env, step, n_steps, label: str):
    """Draw the instrument panel. Same state, same instant."""
    from PIL import ImageDraw

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    pad = 8

    # Top strip: who and when.
    d.rectangle([0, 0, W, 40], fill=(8, 12, 18, 215))
    d.text((pad, 5), label, fill=INK)
    d.text((pad, 21), f"t {step / 100:5.2f} s", fill=DIM)

    tilt = env.tilt_deg()
    alt = env.altitude()
    w = env.omega()
    p_bar, t_k, avail = __import__("aerial_env").tank_state(env.m_gas,
                                                            env.beta)
    gas_f = env.m_gas / __import__("thrusters").M_GAS_0

    tilt_col = GOOD if tilt < 25 else (WARN if tilt < 60 else BAD)
    rows = [("ALT", f"{alt:6.2f} m", INK),
            ("TILT", f"{tilt:6.1f} deg", tilt_col),
            ("SINK", f"{-env.vel()[2]:6.2f} m/s", INK),
            ("SPIN", f"{np.linalg.norm(w):6.2f} rad/s", INK),
            ("GAS", f"{env.m_gas:6.1f} kg", GOOD if gas_f > .3 else WARN),
            ("TANK", f"{p_bar / 1e5:6.0f} bar", INK),
            ("TEMP", f"{t_k - 273.15:6.0f} °C", INK)]
    bw = 128
    d.rectangle([W - bw - pad * 2, 48, W, 48 + 18 * len(rows) + 10],
                fill=(8, 12, 18, 205))
    for i, (k, v, col) in enumerate(rows):
        y = 54 + 18 * i
        d.text((W - bw - pad, y), k, fill=DIM)
        d.text((W - bw + 40, y), v, fill=col)

    # Horizon bubble: tilt, drawn rather than numbered.
    cx, cy, rr = 56, H - 58, 30
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(8, 12, 18, 210),
              outline=(70, 90, 110, 255))
    bz = env.rotmat()[:, 2]
    d.line([cx - rr * 0.8, cy, cx + rr * 0.8, cy], fill=(70, 90, 110, 255))
    d.line([cx, cy, cx + rr * 0.85 * float(bz[1]) * -1,
            cy - rr * 0.85 * float(bz[2])], fill=tilt_col, width=3)
    d.text((cx - 16, cy + rr + 4), "UPRIGHT", fill=DIM)

    # Nozzle strip: the policy's actual output, one bar per nozzle.
    n = env.act_dim
    sw, sh, x0 = 12, 46, 110
    d.rectangle([x0 - 6, H - sh - 30, x0 + n * (sw + 3) + 6, H - 8],
                fill=(8, 12, 18, 205))
    d.text((x0 - 2, H - sh - 27), "NOZZLES", fill=DIM)
    for k in range(n):
        u = float(env.u_act[k])
        x = x0 + k * (sw + 3)
        d.rectangle([x, H - sh + 2, x + sw, H - 14],
                    outline=(60, 78, 96, 255))
        h = int((sh - 16) * min(u, 1.0))
        if h > 0:
            role = env.rig[k].role
            col = JET if role == "lift" else (
                (214, 160, 240) if role == "down" else (240, 200, 120))
            d.rectangle([x + 1, H - 14 - h, x + sw - 1, H - 15], fill=col)
    d.text((x0 + n * (sw + 3) + 12, H - sh + 2),
           f"{avail:.0%} avail", fill=DIM)

    # Progress bar.
    d.rectangle([0, H - 5, int(W * step / max(n_steps, 1)), H],
                fill=(90, 150, 210, 230))
    return img


def film(run_dir, rig_name, preset, flat, episodes, seed0, size,
         classical, every, max_frames):
    import mujoco
    import torch
    from PIL import Image

    from aerial_env import AerialEnv

    env = AerialEnv(rig_name, preset, flat=flat)
    if classical:
        from classical import PDController
        net = norm = None
        label_kind = "classical PD + allocator"
    else:
        from evaluate import load_policy
        net, norm = load_policy(run_dir)
        label_kind = f"PPO  {Path(run_dir).name}"

    W, H = size
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 9.5
    cam.elevation = -16.0
    frames = []
    results = []

    for ep in range(episodes):
        o = env.reset(seed0 + ep)
        ctl = PDController(env) if classical else None
        k = 0
        shots = []
        while True:
            if classical:
                a = ctl.act()
            else:
                with torch.no_grad():
                    x = torch.as_tensor(norm(o), dtype=torch.float32
                                        ).unsqueeze(0)
                    a = net.distribution(x).mean.squeeze(0).numpy()
            o, _rw, done, _ = env.step(np.clip(a, 0.0, 1.0))
            k += 1
            if k % every == 0 or done:
                p = env.pos()
                cam.lookat[:] = [p[0], p[1], max(p[2] * 0.6, 0.4)]
                cam.azimuth = 128.0 + 14.0 * math.sin(k / 150.0)
                r.update_scene(env.data, cam)
                add_plumes(r.scene, env, mujoco)
                img = Image.fromarray(r.render()).convert("RGB")
                shots.append(hud(img, env, k, env.max_steps,
                                 f"{label_kind}   |   throw {ep + 1}"))
            if done:
                break
        # Hold the last frame so the outcome is readable.
        tail = 10 if env.landed_ok() else 6
        shots.extend([shots[-1]] * tail)
        frames.extend(shots)
        results.append(env.outcome)
        print(f"    throw {ep + 1}: {env.outcome:<13} "
              f"{k / 100:.2f} s, gas {env.gas_used:.0f} kg, "
              f"{len(shots)} frames", flush=True)
        if len(frames) >= max_frames:
            print(f"    (frame cap {max_frames} reached after "
                  f"{ep + 1} throws — capping is reported, not silent)")
            break
    r.close()
    return frames, results


def save_gif(frames, path, ms, colors):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=ms, loop=0, optimize=True)
    print(f"\n  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--rig", default="reveal")
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--frame-ms", type=int, default=50)
    ap.add_argument("--colors", type=int, default=160)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--out", default="aerial-flight")
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))

    rig, preset, flat = a.rig, a.preset, a.flat
    if a.run:
        meta = json.loads((Path(a.run) / "summary.json").read_text())
        rig = meta["rig"]
        flat = meta.get("flat", False)
        preset = "rolling" if meta["preset"] == "flat" else meta["preset"]

    frames, res = film(a.run, rig, preset, flat, a.episodes, a.seed0,
                       (a.width, a.height), a.classical, a.every,
                       a.max_frames)
    save_gif(frames, OUT / f"{a.out}.gif", a.frame_ms, a.colors)
    ok = sum(r == "landed_ok" for r in res)
    print(f"  {ok}/{len(res)} of the filmed throws landed cleanly "
          f"— this is a FILM of a sample, not the measurement "
          f"(use evaluate.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
