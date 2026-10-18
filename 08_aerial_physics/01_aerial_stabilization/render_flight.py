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
        L = 0.12 + 0.46 * u
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
            np.array([0.72, 0.88, 1.0, 0.10 + 0.22 * u], dtype=np.float32))
        scene.ngeom += 1


def _arrow(scene, mujoco, origin, vec, rgba, width=0.075):
    """One arrow in the 3D scene, oriented along `vec`."""
    L = float(np.linalg.norm(vec))
    if L < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    d = np.asarray(vec) / L
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-8:
        M = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        M = np.eye(3) + vx + vx @ vx / (1 + c)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([width, width, L]),
                        np.asarray(origin, dtype=np.float64),
                        M.flatten().astype(np.float64),
                        np.asarray(rgba, dtype=np.float32))
    # EMISSIVE, or the scene lighting washes every arrow to the same
    # white and a four-colour legend describes one colour.
    g.emission = 0.65
    g.specular = 0.0
    g.shininess = 0.0
    scene.ngeom += 1


def add_vectors(scene, env, mujoco) -> None:
    """
    The physics, drawn to scale: net thrust, weight, angular
    momentum, velocity.

    Every arrow is a real quantity with a stated scale, not a
    decoration. Thrust and weight share one scale so the picture
    answers "is it climbing?" by inspection; H and v have their own.
    """
    import thrusters as TH

    # Arrows start a little above the body so they are not buried
    # inside the hull from every camera angle.
    p = env.pos() + np.array([0.0, 0.0, 0.45])
    R = env.rotmat()
    W = env.model.body_mass.sum() * env.world.g
    FN = 2.6 / max(W, 1.0)                     # metres per newton
    # Net thrust in world, summed from the nozzles actually firing.
    f_b = np.zeros(3)
    for k, t in enumerate(env.rig):
        f_b += np.asarray(t.direction) * (env.u_act[k] * t.fmax
                                          * env.tank["frac"])
    _arrow(scene, mujoco, p, (R @ f_b) * FN, (0.00, 0.62, 1.00, 1.0))
    _arrow(scene, mujoco, p, np.array([0, 0, -W]) * FN,
           (1.00, 0.30, 0.12, 1.0), width=0.060)
    mujoco.mj_subtreeVel(env.model, env.data)
    H = env.data.subtree_angmom[env.chassis]
    _arrow(scene, mujoco, p, H * (2.2 / 3000.0), (1.00, 0.00, 0.85, 1.0),
           width=0.065)
    _arrow(scene, mujoco, p, env.vel() * (2.2 / 12.0),
           (0.15, 1.00, 0.20, 1.0), width=0.060)


def hud(img, env, step, n_steps, label: str, trace=None):
    """Draw the instrument panel. Same state, same instant."""
    from PIL import ImageDraw

    import gas

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    pad = 8
    TH = __import__("thrusters")
    tk = env.tank

    d.rectangle([0, 0, W, 40], fill=(8, 12, 18, 220))
    d.text((pad, 5), label, fill=INK)
    d.text((pad, 21), f"t {step / 100:5.2f} s   "
                      f"{env.world.name}  g {env.world.g:.2f}", fill=DIM)

    tilt = env.tilt_deg()
    w = env.omega()
    rho = env.m_gas / TH.TANK_V
    sup = gas.superheat(rho, tk["T"]) if rho > 1e-6 else None
    tilt_col = GOOD if tilt < 25 else (WARN if tilt < 60 else BAD)

    # The state panel. Grouped: attitude, then propellant, then the
    # two numbers that say whether the thruster is still a thruster.
    rows = [("ATTITUDE", "", DIM),
            ("tilt", f"{tilt:7.1f} deg", tilt_col),
            ("|omega|", f"{np.linalg.norm(w):7.2f} rad/s", INK),
            ("alt", f"{env.altitude():7.2f} m", INK),
            ("sink", f"{-env.vel()[2]:7.2f} m/s", INK),
            ("", "", DIM),
            ("PROPELLANT", "", DIM),
            ("gas", f"{env.m_gas:7.1f} kg", INK),
            ("tank p", f"{tk['p'] / 1e5:7.0f} bar", INK),
            ("tank T", f"{tk['T']:7.1f} K", INK),
            ("Z", f"{gas.compressibility(rho, tk['T']):7.3f}"
                  if rho > 1e-6 else "      -", INK),
            ("superheat",
             "  supercrit" if sup is None else f"{sup:7.1f} K",
             INK if sup is None else (BAD if sup < 8 else INK)),
            ("", "", DIM),
            ("NOZZLE", "", DIM),
            ("p_c", f"{tk['pc'] / 1e5:7.1f} bar", INK),
            ("Isp", f"{tk['ce'] / TH.G0:7.1f} s", INK),
            ("T exit", f"{tk['Te']:7.1f} K",
             BAD if tk["Te"] < gas.T_TRIPLE + 5 else INK),
            ("thrust", f"{tk['frac']:7.0%}",
             WARN if tk["frac"] < 0.6 else INK)]
    bw = 168
    d.rectangle([W - bw - pad, 48, W, 48 + 15 * len(rows) + 10],
                fill=(8, 12, 18, 215))
    for i, (k, v, col) in enumerate(rows):
        y = 53 + 15 * i
        if v == "":
            d.text((W - bw - pad + 4, y), k, fill=(110, 140, 170))
            continue
        d.text((W - bw - pad + 10, y), k, fill=DIM)
        d.text((W - bw + 58, y), v, fill=col)

    # Impulse budget: how much of the tank this episode has spent.
    used = env.gas_used
    tot = TH.gas_mass()
    bx, by, bwid = pad, H - 100, 250
    d.rectangle([bx, by, bx + bwid, by + 14], outline=(70, 90, 110, 255))
    d.rectangle([bx + 1, by + 1, bx + 1 + int((bwid - 2) * used / tot),
                 by + 13], fill=(226, 150, 90, 240))
    d.rectangle([bx - 2, by - 18, bx + bwid + 2, by + 16],
                fill=(10, 14, 20, 0))
    d.text((bx + 2, by - 15), f"PROPELLANT  {used:.0f} / {tot:.0f} kg",
           fill=DIM)

    # Live trace: altitude and |omega| so far, the two things the
    # policy is actually trading against each other.
    if trace:
        tx, ty, tw, th = pad, H - 78, 250, 68
        d.rectangle([tx, ty, tx + tw, ty + th], fill=(12, 17, 24, 225))
        n = len(trace)
        amax = max(1.0, max(a for a, _ in trace))
        wmax = max(0.5, max(b for _, b in trace))
        for i in range(1, n):
            x0 = tx + tw * (i - 1) / max(n - 1, 1)
            x1 = tx + tw * i / max(n - 1, 1)
            d.line([x0, ty + th - (th - 16) * trace[i - 1][0] / amax,
                    x1, ty + th - (th - 16) * trace[i][0] / amax],
                   fill=(126, 198, 248), width=2)
            d.line([x0, ty + th - (th - 16) * trace[i - 1][1] / wmax,
                    x1, ty + th - (th - 16) * trace[i][1] / wmax],
                   fill=(226, 140, 200), width=2)
        d.text((tx + 3, ty + 2), f"alt  max {amax:.1f} m",
               fill=(126, 198, 248))
        d.text((tx + 124, ty + 2), f"|omega|  max {wmax:.1f}",
               fill=(226, 140, 200))

    # Arrow legend, so the 3D overlay is readable.
    lx = pad + 268
    for i, (nm, col) in enumerate((("net thrust", (0, 158, 255)),
                                   ("weight", (255, 77, 31)),
                                   ("ang. momentum", (255, 0, 217)),
                                   ("velocity", (38, 255, 51)))):
        d.line([lx, H - 76 + 16 * i, lx + 16, H - 76 + 16 * i],
               fill=col, width=3)
        d.text((lx + 22, H - 83 + 16 * i), nm, fill=DIM)

    # Nozzle strip: the policy's actual output.
    n = env.act_dim
    sw, sh, x0 = 11, 44, lx + 132
    d.rectangle([x0 - 6, H - sh - 26, x0 + n * (sw + 3) + 8, H - 10],
                fill=(8, 12, 18, 215))
    d.text((x0 - 2, H - sh - 23), "NOZZLE VALVES", fill=DIM)
    for k in range(n):
        u = float(env.u_act[k])
        x = x0 + k * (sw + 3)
        d.rectangle([x, H - sh + 2, x + sw, H - 16],
                    outline=(60, 78, 96, 255))
        hh = int((sh - 18) * min(u, 1.0))
        if hh > 0:
            role = env.rig[k].role
            col = JET if role == "lift" else (
                (214, 160, 240) if role == "down" else (240, 200, 120))
            d.rectangle([x + 1, H - 16 - hh, x + sw - 1, H - 17], fill=col)

    # Progress bar.
    d.rectangle([0, H - 5, int(W * step / max(n_steps, 1)), H],
                fill=(90, 150, 210, 230))
    return img


def film(run_dir, rig_name, preset, flat, episodes, seed0, size,
         classical, every, max_frames, world="earth", per_throw=None):
    import mujoco
    import torch
    from PIL import Image

    from aerial_env import AerialEnv

    env = AerialEnv(rig_name, preset, flat=flat, world=world)
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
        trace = []
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
            trace.append((max(env.altitude(), 0.0),
                          float(np.linalg.norm(env.omega()))))
            if k % every == 0 or done:
                p = env.pos()
                # Track the car properly and pull back as it climbs.
                # At 0.6 * height with a fixed 9.5 m distance the car
                # left the top of the frame above about 7 m, which is
                # inside the normal throw.
                alt = max(env.altitude(), 0.0)
                cam.lookat[:] = [p[0], p[1], max(p[2] * 0.85, 0.5)]
                cam.distance = 9.5 + 0.45 * max(alt - 2.0, 0.0)
                cam.azimuth = 128.0 + 14.0 * math.sin(k / 150.0)
                r.update_scene(env.data, cam)
                add_plumes(r.scene, env, mujoco)
                add_vectors(r.scene, env, mujoco)
                img = Image.fromarray(r.render()).convert("RGB")
                shots.append(hud(img, env, k, env.max_steps,
                                 f"{label_kind}   |   throw {ep + 1}",
                                 trace=trace))
            if done:
                break
        # Hold the last frame so the outcome is readable.
        tail = 10 if env.landed_ok() else 6
        shots.extend([shots[-1]] * tail)
        frames.extend(shots)
        if per_throw is not None:
            per_throw.append((shots, env.outcome, k / 100.0, env.gas_used))
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
    ap.add_argument("--world", default="earth",
                    choices=("earth", "mars"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--frame-ms", type=int, default=50)
    ap.add_argument("--colors", type=int, default=160)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--out", default="aerial-flight")
    ap.add_argument("--split", action="store_true",
                    help="ALSO write one short GIF per throw. The combined "
                         "film is 197 frames and 40 MB, which is the right "
                         "artefact for studying a policy and the wrong one "
                         "for a reader skimming a page.")
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))

    rig, preset, flat = a.rig, a.preset, a.flat
    if a.run:
        meta = json.loads((Path(a.run) / "summary.json").read_text())
        rig = meta["rig"]
        flat = meta.get("flat", False)
        preset = "rolling" if meta["preset"] == "flat" else meta["preset"]

    clips: list = [] if a.split else None
    frames, res = film(a.run, rig, preset, flat, a.episodes, a.seed0,
                       (a.width, a.height), a.classical, a.every,
                       a.max_frames, a.world, clips)
    save_gif(frames, OUT / f"{a.out}.gif", a.frame_ms, a.colors)
    if clips:
        print()
        for i, (shots, outcome, secs, gas) in enumerate(clips, 1):
            save_gif(shots, OUT / f"{a.out}-{i}.gif", a.frame_ms, a.colors)
            print(f"      throw {i}: {outcome}, {secs:.2f} s, "
                  f"{gas:.0f} kg of gas")
    ok = sum(r == "landed_ok" for r in res)
    print(f"  {ok}/{len(res)} of the filmed throws landed cleanly "
          f"— this is a FILM of a sample, not the measurement "
          f"(use evaluate.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
