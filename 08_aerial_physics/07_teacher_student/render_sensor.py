#!/usr/bin/env python3
"""
Film what the student actually sees: the beams, and the image they make.

    uv run render_sensor.py --beams          # 3D, the frustum on the ground
    uv run render_sensor.py --split          # 3D beside the depth image
    uv run render_sensor.py --all

TWO PANELS, AND THE RIGHT-HAND ONE IS THE POINT
------------------------------------------------
The left panel is the scene a reader already understands: a car
falling toward hazard terrain, with 144 ranging beams drawn from
its belly and a dot where each one strikes. The right panel is
the ONLY thing the student receives -- those 144 numbers, laid
out as a 12x12 image.

Watching them together is the argument. The beams sweep as the
body rolls, so the image swings even when the ground is still;
the sample spacing stretches as the vehicle climbs, so the same
boulder occupies fewer pixels; a rock edge casts a shadow of
no-return cells that look exactly like distant flat ground. None
of that is visible in a table of success rates, and all of it is
what the student has to learn to discount.

COLOUR
------
Near is dark, far is bright, no-return is flat grey -- and grey
is deliberately NOT the far end of the ramp. A missing return and
a distant surface are different facts and a reader should be able
to tell them apart at a glance, even though the policy receives
the same 1.0 for both. That mismatch between what the figure
shows and what the network gets is itself worth seeing.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"
BACKENDS = ("egl", "osmesa", "glfw")

BG = (11, 15, 20)
PANEL = (16, 22, 30)
INK = (236, 242, 248)
DIM = (150, 170, 186)
FAINT = (104, 126, 146)
GOOD = (108, 206, 148)
WARN = (226, 176, 92)
BAD = (228, 108, 108)
JET = (126, 198, 248)
BEAM = (120, 200, 255)


def pick_backend() -> str:
    """
    Find a backend that can actually RENDER, not merely import.

    Probing with `MjModel.from_xml_string` touches no OpenGL at
    all, so it happily accepts an `osmesa` whose mujoco module
    omits `Renderer` entirely -- the failure then surfaces a
    hundred lines later as a confusing AttributeError. Lab 6 hit
    exactly that. Probe the thing you intend to use: build a
    one-pixel renderer and draw with it.

    On this box OpenGL reaches the GPU through WSL's D3D12
    translation layer and a frame costs ~150 ms no matter how few
    pixels it has. That is fine for offline figures and is the
    measured reason the SENSOR is ray-cast rather than rendered --
    see rangefinder.py.
    """
    import importlib
    import sys

    for b in BACKENDS:
        os.environ["MUJOCO_GL"] = b
        for mod in [m for m in list(sys.modules) if m.startswith("mujoco")]:
            del sys.modules[mod]
        try:
            mujoco = importlib.import_module("mujoco")
            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
            rr = mujoco.Renderer(m, height=8, width=8)
            rr.update_scene(mujoco.MjData(m))
            rr.render()
            rr.close()
            print(f"  [render] MUJOCO_GL={b}")
            return b
        except Exception:
            continue
    raise SystemExit(
        "  No OpenGL backend available. Everything this lab MEASURES\n"
        "  needs none:\n"
        "      uv run rangefinder.py --scene     # the scan, as ASCII\n"
        "      uv run teach_env.py --gap\n"
        "      uv run compare.py")


def save_gif(frames, path, ms=70, colors=160, hold_ms=1400):
    """
    Write a GIF that actually animates.

    `optimize=True` MERGES consecutive identical frames and rolls
    their durations together, so a file can report a frame count
    it does not have and look like a still picture. Lab 6 shipped
    three figures that way. The end pause is ONE frame with a long
    duration rather than a run of duplicates, for the same reason,
    and the result is verified by READING THE FILE BACK instead of
    trusting the list handed to save().
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=colors,
                     dither=Image.Dither.NONE) for f in frames]
    durations = [ms] * (len(pil) - 1) + [hold_ms]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=durations, loop=0, optimize=False)
    kept = getattr(Image.open(path), "n_frames", 1)
    if kept != len(frames):
        raise RuntimeError(
            f"{path.name}: asked for {len(frames)} frames, the file has "
            f"{kept}. Consecutive frames were identical enough to be "
            f"merged -- the animation would look frozen.")
    print(f"  {path.name}  ({kept} frames @ {ms} ms + {hold_ms} ms hold, "
          f"{path.stat().st_size / 1024 / 1024:.2f} MB)")


def depth_rgb(v: float) -> tuple[int, int, int]:
    """One depth value to a colour. No-return is grey, not 'far'."""
    import rangefinder as RF

    if v >= RF.NO_RETURN - 1e-6:
        return (86, 92, 100)
    t = max(0.0, min(1.0, v))
    # near -> deep blue, mid -> cyan, far -> pale
    return (int(20 + 200 * t ** 1.6),
            int(60 + 175 * t),
            int(110 + 130 * t ** 0.7))


def draw_scan(d, x0: int, y0: int, cell: int, scan: np.ndarray, n: int,
              title: str = "what the student receives") -> None:
    """The 12x12 depth image, as a panel."""
    import rangefinder as RF

    img = scan.reshape(n, n)
    d.rectangle([x0 - 10, y0 - 26, x0 + n * cell + 10, y0 + n * cell + 34],
                fill=PANEL)
    d.text((x0, y0 - 20), title, fill=DIM)
    for r in range(n):
        for c in range(n):
            d.rectangle([x0 + c * cell, y0 + r * cell,
                         x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                        fill=depth_rgb(float(img[r, c])))
    nr = float((img >= RF.NO_RETURN - 1e-6).mean())
    d.text((x0, y0 + n * cell + 8),
           f"near {img.min() * RF.MAX_RANGE:5.1f} m   "
           f"far {img.max() * RF.MAX_RANGE:5.1f} m   "
           f"no-return {nr:.0%}", fill=FAINT)


def add_beams(scene, mujoco, origin, hits, mask, every: int = 5) -> None:
    """
    Draw the frustum: a ray per beam, a dot where it lands.

    Built with the same construction lab 5 uses for its thrust
    vectors, and that is not laziness. The first version placed a
    CAPSULE at the midpoint with a half-length in `size[2]`, which
    is what the MuJoCo docs describe and which drew NOTHING --
    every beam silently absent while the landing dots rendered
    fine from the same loop. An arrow anchored at its base with a
    full length in `size[2]`, specular and shininess zeroed, is
    the construction that is known to work in this repository.

    Only every `every`-th beam is drawn. All 144 is an opaque cone
    that hides the terrain it is supposed to be measuring, and the
    figure is about what the beams FIND, not how many there are.
    """
    for i in range(0, len(hits), every):
        if scene.ngeom >= scene.maxgeom - 2:
            return
        vec = hits[i] - origin
        L = float(np.linalg.norm(vec))
        if L < 1e-6:
            continue
        d = vec / L
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, d)
        c = float(np.dot(z, d))
        if np.linalg.norm(v) < 1e-8:
            M = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            M = np.eye(3) + vx + vx @ vx / (1 + c)
        col = ((0.34, 0.80, 1.0, 0.55) if mask[i]
               else (0.60, 0.62, 0.66, 0.20))
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([0.035, 0.035, L]),
            np.asarray(origin, dtype=np.float64),
            M.flatten().astype(np.float64),
            np.asarray(col, dtype=np.float32))
        # EMISSIVE, or the scene lighting washes the beams to the
        # same white as the vehicle and the figure loses its subject.
        g.emission = 0.55
        g.specular = 0.0
        g.shininess = 0.0
        scene.ngeom += 1
        if not mask[i] or scene.ngeom >= scene.maxgeom - 1:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.16, 0.16, 0.16]),
            np.asarray(hits[i], dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray((0.25, 0.85, 1.0, 1.0), dtype=np.float32))
        g.emission = 0.45
        g.specular = 0.0
        g.shininess = 0.0
        scene.ngeom += 1


def beam_hits(env, clean_scan: np.ndarray):
    """World-space landing point of every beam, and which returned."""
    import rangefinder as RF

    R = env.rotmat()
    origin = env.pos() + R @ env.rf.mount
    dirs = env.rf.dirs @ R.T
    rng = clean_scan * RF.MAX_RANGE
    mask = clean_scan < RF.NO_RETURN - 1e-6
    pts = origin[None, :] + dirs * rng[:, None]
    return origin, pts, mask


def hud(img, env, step, n_steps, label, extra=""):
    from PIL import ImageDraw

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 44], fill=(8, 12, 18, 225))
    d.text((12, 6), label, fill=INK)
    d.text((12, 24), f"t {step / 100:5.2f} s   step {step:>3}/{n_steps}"
                     f"   alt {env.altitude():5.2f} m   "
                     f"miss {env.miss():5.2f} m", fill=DIM)
    if extra:
        d.text((W - 330, 6), extra, fill=WARN)
    tilt = env.tilt_deg()
    col = GOOD if tilt < 15 else (WARN if tilt < 25 else BAD)
    d.text((W - 330, 24), f"tilt {tilt:5.1f} deg   "
                          f"sink {max(0.0, -env.vel()[2]):4.1f} m/s", fill=col)
    bw = int((W - 24) * step / max(n_steps, 1))
    d.rectangle([12, H - 5, 12 + bw, H - 3], fill=col)
    return img


def film(split: bool, episodes: int, seed0: int, horizon: int,
         width: int, height: int, every: int, out_name: str,
         preset: str, rig: str) -> None:
    import mujoco
    from PIL import Image, ImageDraw

    import rangefinder as RF
    from teach_env import TeachEnv

    env = TeachEnv(rig, preset, sense="onboard", seed=0)
    n = env.rf.n
    cell = 17
    panel_w = n * cell + 44 if split else 0
    scene_w = width - panel_w

    r = mujoco.Renderer(env.model, height=height, width=scene_w)
    cam = mujoco.MjvCamera()
    cam.elevation = -26.0

    frames = []
    for ep in range(episodes):
        obs = env.reset(seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        hov = env.hover_action() if hasattr(env, "hover_action") else None
        for k in range(horizon):
            clean = env.onboard_scan(clean=True)
            noisy = env.onboard_scan()
            if k % every == 0:
                origin, pts, mask = beam_hits(env, clean)
                p = env.pos()
                cam.lookat[:] = [p[0], p[1], max(p[2] * 0.75, 0.4)]
                cam.distance = 13.0 + 1.1 * max(env.altitude() - 2.0, 0.0)
                cam.azimuth = 132.0 + 16.0 * math.sin(k / 90.0)
                r.update_scene(env.data, cam)
                add_beams(r.scene, mujoco, origin, pts, mask, every=5)
                scene = Image.fromarray(r.render()).convert("RGB")

                img = Image.new("RGB", (width, height), BG)
                img.paste(scene, (0, 0))
                if split:
                    d = ImageDraw.Draw(img)
                    draw_scan(d, scene_w + 22, 86, cell, noisy, n)
                    d.text((scene_w + 22, 86 + n * cell + 52),
                           "left: 144 beams in the world", fill=JET)
                    d.text((scene_w + 22, 86 + n * cell + 68),
                           "right: the only thing the", fill=DIM)
                    d.text((scene_w + 22, 86 + n * cell + 82),
                           "student is given", fill=DIM)
                frames.append(hud(
                    img, env, k + 1, horizon,
                    f"the student's view   flight {ep + 1} of {episodes}",
                    f"{n}x{n} = {n * n} beams"))
            act = np.clip(rng.uniform(0.25, 0.65, env.act_dim), 0.0, 1.0)
            obs, _rw, done, _ = env.step(act)
            if done:
                break
        print(f"    flight {ep + 1}: {env.outcome}, "
              f"{len(frames)} frames so far")
    r.close()
    save_gif(frames, OUT / f"{out_name}.gif", ms=75)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beams", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=820_000)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--every", type=int, default=3)
    a, _ = ap.parse_known_args()
    if not (a.beams or a.split or a.all):
        print("  nothing asked for; try --all")
        return 0

    pick_backend()
    import sys

    sys.path.insert(0, str(HERE))

    if a.all or a.beams:
        film(False, a.episodes, a.seed0, a.horizon, 860, 500, a.every,
             "ts-beams", a.preset, a.rig)
    if a.all or a.split:
        film(True, a.episodes, a.seed0, a.horizon, 1120, 500, a.every,
             "ts-sensor-split", a.preset, a.rig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
