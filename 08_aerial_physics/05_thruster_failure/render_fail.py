#!/usr/bin/env python3
"""
Film a thruster failure, and whether the car catches it.

    uv run render_fail.py runs/robust_s0 --episodes 4
    uv run render_fail.py runs/healthy_s0 --seeds 900003 --out healthy-loses
    uv run render_fail.py --classical --out baseline

WHAT IS ON SCREEN
-----------------
Everything labs 1 to 4 showed -- to-scale arrows for net thrust,
weight, angular momentum and velocity, plus tank pressure,
temperature, compressibility, live Isp and nozzle exit temperature
-- and three things this lab needs:

  FAULT PANEL    which nozzle broke, in which mode, and at what
                 second. Before the break it counts down; after it,
                 the banner stays lit for the rest of the flight.
  NOZZLE STRIP   the policy's commanded opening per nozzle, with the
                 broken one outlined in red and drawn at what it is
                 ACTUALLY doing. On a latched nozzle those two
                 disagree permanently, and the gap is the fault.
  ATTITUDE BALL  tilt from upright, which is what a recovery is
                 actually about. Green inside the 25 degree landing
                 gate, red outside it.

The policy is not told any of this -- the fault panel is for the
viewer. The car has to infer the break from the dynamics.
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
    import fail_env as TE
    from fail_env import TILT_OK

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    pad = 8
    TH = __import__("thrusters")
    tk = env.tank

    fl = env.fault
    broken = bool(fl) and env.fault_active
    on = env.outcome == "landed_ok"
    tilt_now = env.tilt_deg()
    if env.outcome in ("landed_ok", "landed_hard"):
        phase = {"landed_ok": "LANDED", "landed_hard": "DOWN HARD"}[env.outcome]
    elif env.outcome == "crash":
        phase = "CRASHED"
    elif broken:
        # ASCII only: the default PIL bitmap font has no em-dash and
        # renders it as a hollow box, right in the most important
        # words on the HUD.
        phase = f"FAULT - nozzle {fl['nozzle']} stuck {fl['mode'].upper()}"
    elif fl:
        phase = f"NOMINAL - fault in {(fl['at'] - env.t) / 100.0:.1f} s"
    else:
        phase = "NOMINAL"
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
    rows = [("FAULT", "", DIM),
            ("nozzle", ("    none" if not fl
                        else f"{fl['nozzle']:>8}"),
             BAD if broken else DIM),
            ("mode", ("    none" if not fl else f"{fl['mode']:>8}"),
             BAD if broken else DIM),
            ("at", ("       —" if not fl
                    else f"{fl['at'] / 100.0:7.2f} s"),
             BAD if broken else DIM),
            ("", "", DIM),
            ("ATTITUDE", "", DIM),
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

    # ATTITUDE BALL. The task, so it is the largest instrument.
    # A recovery is about tilt: the landing gate is 25 degrees and
    # a latched nozzle drives the car past that in under a second,
    # so the ring is drawn AT the gate and the needle either sits
    # inside it or does not.
    mx, my, mr = 68, H - 86, 54
    d.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(8, 12, 18, 232),
              outline=(70, 90, 110, 255))
    gate_r = mr * (TILT_OK / 90.0) * 2.2
    d.ellipse([mx - gate_r, my - gate_r, mx + gate_r, my + gate_r],
              outline=GOOD if tilt_now <= TILT_OK else (110, 140, 170), width=2)

    # The body up-axis projected onto the horizontal plane: direction
    # is which way it has fallen, length is how far.
    R = env.rotmat()
    bz = R[:, 2]
    lean = math.hypot(float(bz[0]), float(bz[1]))
    ang = math.atan2(float(bz[1]), float(bz[0]))
    rad = min(mr - 6, mr * lean * 2.2)
    nx, ny = mx + rad * math.cos(ang), my - rad * math.sin(ang)
    col = GOOD if tilt_now <= TILT_OK else (WARN if tilt_now < 60 else BAD)
    d.line([mx, my, nx, ny], fill=col, width=3)
    d.ellipse([nx - 5, ny - 5, nx + 5, ny + 5], fill=col)
    d.line([mx - 5, my, mx + 5, my], fill=(90, 110, 130), width=1)
    d.line([mx, my - 5, mx, my + 5], fill=(90, 110, 130), width=1)

    if trace and len(trace) > 1:
        pts = []
        for _a, _m, tx_, ty_ in trace[-400:]:
            rr2 = min(mr - 6, mr * tx_ * 2.2)
            pts.append((mx + rr2 * math.cos(ty_), my - rr2 * math.sin(ty_)))
        if len(pts) > 1:
            d.line(pts, fill=(126, 198, 248), width=2)

    d.text((mx - 26, my + mr - 16), f"tilt {tilt_now:5.1f}\u00b0", fill=col)
    d.text((mx - 34, my + mr + 4),
           f"ATTITUDE  gate {TILT_OK:.0f}\u00b0", fill=DIM)

    # PHASE, top strip, on the RIGHT: the run label on the left is
    # long enough to reach the middle. Kept inside this block on
    # purpose -- in lab 3 it sat between two sections and a
    # slice-based edit deleted it silently.
    d.text((W - 330, 5), f"PHASE  {phase}",
           fill=BAD if (broken or phase == "CRASHED") else (GOOD if on else JET))
    d.text((W - 330, 21),
           (f"tilt {tilt_now:.0f}\u00b0   sink {max(0.0, -env.vel()[2]):.1f} m/s"
            f"   gas {env.gas_used:.0f} kg"),
           fill=GOOD if on else DIM)

    # Impulse budget: how much of the tank this episode has spent.
    used = env.gas_used
    tot = TH.gas_mass()
    bx, by, bwid = pad + 124, H - 124, 208
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
        tx, ty, tw, th = pad + 124, H - 100, 208, 64
        d.rectangle([tx, ty, tx + tw, ty + th], fill=(12, 17, 24, 225))
        n = len(trace)
        amax = max(1.0, max(a for a, _m, _x, _y in trace))
        wmax = max(0.5, max(_m for _a, _m, _x, _y in trace))
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
        d.text((tx + 108, ty + 2), f"tilt  max {wmax:.0f} deg",
               fill=(226, 140, 200))

    # Arrow legend, so the 3D overlay is readable.
    lx = pad + 344
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
    d.text((x0 - 2, H - sh - 23),
           "NOZZLE VALVES" + (f"   #{fl['nozzle']} STUCK {fl['mode'].upper()}"
                              if broken else ""),
           fill=BAD if broken else DIM)
    for k in range(n):
        u = float(env.u_act[k])
        dead = broken and k == fl["nozzle"]
        if dead:
            # Draw what the nozzle is ACTUALLY doing, not what was
            # asked of it. On a latched nozzle those disagree for
            # the rest of the flight, and that disagreement is the
            # whole fault.
            u = 0.0 if fl["mode"] == "shut" else 1.0
        x = x0 + k * (sw + 3)
        d.rectangle([x, H - sh + 2, x + sw, H - 16],
                    outline=(214, 70, 70, 255) if dead else (60, 78, 96, 255),
                    width=2 if dead else 1)
        hh = int((sh - 18) * min(u, 1.0))
        if hh > 0:
            role = env.rig[k].role
            col = ((228, 92, 92) if dead else
                   (JET if role == "lift" else
                    ((214, 160, 240) if role == "down" else
                     (240, 200, 120))))
            d.rectangle([x + 1, H - 16 - hh, x + sw - 1, H - 17], fill=col)

    # Progress bar.
    d.rectangle([0, H - 5, int(W * step / max(n_steps, 1)), H],
                fill=(90, 150, 210, 230))
    return img


def film(run_dir, rig_name, preset, episodes, seed0, size,
         classical, every, max_frames, world="earth", per_throw=None,
         pick_seeds="", flat=False, fail_mode="mix"):
    import mujoco
    import torch
    from PIL import Image

    from fail_env import FailEnv

    env = FailEnv(rig_name, preset, world=world, flat=flat,
                  fail_p=1.0, fail_mode=fail_mode)
    if classical:
        from classical import FlightController
        net = norm = None
        label_kind = "scheduled guidance law"
    else:
        from evaluate import load_policy
        net, norm = load_policy(run_dir)
        label_kind = f"PPO  {Path(run_dir).name}  ({world})"

    W, H = size
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 9.5
    cam.elevation = -16.0
    frames = []
    results = []

    picks = ([int(x) for x in str(pick_seeds).split(',') if x]
             if pick_seeds else [seed0 + i for i in range(episodes)])
    for ep, sd in enumerate(picks):
        o = env.reset(sd)
        ctl = FlightController(env) if classical else None
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
            pxy = env.pos()
            # The ball plots the body up-axis projected flat, so the
            # trace carries that projection rather than world x/y.
            _bz = env.rotmat()[:, 2]
            trace.append((max(env.altitude(), 0.0),
                          float(env.tilt_deg()),
                          float(math.hypot(_bz[0], _bz[1])),
                          float(math.atan2(_bz[1], _bz[0]))))
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
        tail = 10 if env.success() else 6
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
    ap.add_argument("--fail-mode", default="mix",
                    choices=("shut", "open", "mix"))
    ap.add_argument("--world", default="earth",
                    choices=("earth", "mars"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--seeds", default="",
                    help="comma-separated absolute seeds to film, instead "
                         "of seed0..seed0+n. Used to show the manoeuvre "
                         "working; the page says plainly that it is a "
                         "selected sample and points at evaluate.py for "
                         "the rate.")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--frame-ms", type=int, default=50)
    ap.add_argument("--colors", type=int, default=160)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--out", default="fail-flight")
    ap.add_argument("--split", action="store_true",
                    help="ALSO write one short GIF per landing. The "
                         "combined film is tens of MB, which is the right "
                         "artefact for studying a policy and the wrong one "
                         "for a reader skimming a page.")
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))

    rig, preset, world = a.rig, a.preset, a.world
    flat = a.flat
    if a.run:
        # Read the WORLD back off the run rather than trusting the
        # flag. Filming a Mars policy under Earth gravity would
        # produce a plausible-looking film of a controller that
        # cannot fly, with nothing anywhere saying so.
        meta = json.loads((Path(a.run) / "summary.json").read_text())
        rig = meta["rig"]
        preset = meta["preset"]
        world = meta.get("world", world)
        # Read the terrain flag back too. A policy trained flat and
        # filmed on rolling ground loads without error and flies a
        # task it was never trained for.
        flat = meta.get("flat", flat)

    clips: list = [] if a.split else None
    frames, res = film(a.run, rig, preset, a.episodes, a.seed0,
                       (a.width, a.height), a.classical, a.every,
                       a.max_frames, world, clips, a.seeds,
                       flat=flat, fail_mode=a.fail_mode)
    save_gif(frames, OUT / f"{a.out}.gif", a.frame_ms, a.colors)
    if clips:
        print()
        for i, (shots, outcome, secs, gas) in enumerate(clips, 1):
            save_gif(shots, OUT / f"{a.out}-{i}.gif", a.frame_ms, a.colors)
            print(f"      throw {i}: {outcome}, {secs:.2f} s, "
                  f"{gas:.0f} kg of gas")
    ok = sum(r == "landed_ok" for r in res)
    print(f"  {ok}/{len(res)} of the filmed flights landed "
          f"— this is a FILM of a sample, not the measurement "
          f"(use evaluate.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
