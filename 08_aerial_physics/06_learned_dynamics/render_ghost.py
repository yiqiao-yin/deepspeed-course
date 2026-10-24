#!/usr/bin/env python3
"""
Fly the car and its own prediction side by side, and watch them part.

    uv run render_ghost.py --size small
    uv run render_ghost.py --size xxl --episodes 2 --out dyn-ghost-xxl

WHAT IS ON SCREEN
-----------------
One MuJoCo scene, two vehicles. The solid car is the real physics.
The translucent GHOST is where the dynamics model thinks the car is,
having been given the true starting state and then nothing but the
valve commands -- it is running open-loop on its own predictions,
exactly as the 50-step metric measures.

They start on top of each other. The error table says the model is
excellent one step out and useless fifty steps out; this is that
sentence as a picture.

HOW THE GHOST POSE IS RECOVERED
--------------------------------
The model predicts the 41-number OBSERVATION, not MuJoCo state, so
the pose has to be read back out of it:

    attitude   observation 0:6 are the first two columns of the
               rotation matrix; the third is their cross product.
    altitude   observation 13, scaled by 6.
    position   NOT in the observation at all. It is integrated from
               the predicted velocity (observation 10:13, scaled by
               10) starting from the true launch point.

That integration is itself a compounding sum, so the ghost's
horizontal drift accumulates faster than any single channel of the
error table suggests. That is honest rather than unfair: a planner
asking this model "where will I be" would integrate exactly the
same way.
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

INK = (236, 242, 248)
DIM = (150, 170, 186)
GOOD = (108, 206, 148)
WARN = (226, 176, 92)
BAD = (228, 108, 108)
JET = (126, 198, 248)
GHOST = (226, 140, 200)


def pick_backend() -> str:
    """
    Find a backend that can actually RENDER, not merely import.

    The first version probed with `MjModel.from_xml_string`, which
    touches no OpenGL at all. It therefore accepted `osmesa` on a box
    where OSMesa does not work -- mujoco imports fine, omits
    `Renderer` entirely, and the failure surfaces 150 lines later as
    "module 'mujoco' has no attribute 'Renderer'". Probe the thing
    you are going to use: build a one-pixel renderer and draw.
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
        "  No OpenGL backend available. The measurements need none:\n"
        "      uv run ladder.py")


def pose_from_obs(o: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation matrix and altitude, read back out of the observation."""
    c0 = np.asarray(o[0:3], dtype=float)
    c1 = np.asarray(o[3:6], dtype=float)
    c0 /= max(np.linalg.norm(c0), 1e-9)
    c1 = c1 - c0 * float(c0 @ c1)
    c1 /= max(np.linalg.norm(c1), 1e-9)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1), float(o[13]) * 6.0


def add_ghost(scene, mujoco, pos, R, rgba=(0.89, 0.55, 0.78, 0.40)):
    """A translucent body at the model's predicted pose."""
    if scene.ngeom >= scene.maxgeom - 4:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_BOX,
        np.array([2.3, 1.0, 0.55]),
        np.asarray(pos, dtype=np.float64),
        R.flatten().astype(np.float64),
        np.asarray(rgba, dtype=np.float32))
    g.emission = 0.45
    g.specular = 0.0
    scene.ngeom += 1
    # A short mast so the ghost's attitude is readable even when the
    # box is edge-on to the camera.
    g = scene.geoms[scene.ngeom]
    up = R[:, 2]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.10, 0.10, 1.8]),
        np.asarray(pos, dtype=np.float64),
        R.flatten().astype(np.float64),
        np.asarray((0.89, 0.55, 0.78, 0.85), dtype=np.float32))
    g.emission = 0.8
    scene.ngeom += 1


def hud(img, step, horizon, drift, alt_t, alt_p, label):
    from PIL import ImageDraw

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 42], fill=(8, 12, 18, 225))
    d.text((10, 5), label, fill=INK)
    d.text((10, 22), f"t {step / 100:5.2f} s   step {step:>3} of {horizon}",
           fill=DIM)
    col = GOOD if drift < 0.5 else (WARN if drift < 2.0 else BAD)
    d.text((W - 300, 5), f"DRIFT  {drift:6.3f}", fill=col)
    d.text((W - 300, 22),
           f"altitude  real {alt_t:5.2f} m   model {alt_p:5.2f} m", fill=DIM)
    d.rectangle([0, H - 26, W, H], fill=(8, 12, 18, 215))
    d.text((10, H - 20), "solid = real physics", fill=JET)
    d.text((220, H - 20), "translucent = the model's prediction",
           fill=GHOST)
    bw = int((W - 20) * step / max(horizon, 1))
    d.rectangle([10, H - 4, 10 + bw, H - 2], fill=col)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="mlp")
    ap.add_argument("--size", default="small")
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=500_000)
    ap.add_argument("--width", type=int, default=820)
    ap.add_argument("--height", type=int, default=470)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", default="")
    a, _ = ap.parse_known_args()

    pick_backend()
    import sys

    sys.path.insert(0, str(HERE))
    import mujoco
    import torch
    from PIL import Image

    import dynamics as DY
    from fail_env import FailEnv
    from render_figs import save_gif

    ck = torch.load(HERE / "runs" / f"{a.kind}_{a.size}" / "model.pt",
                    map_location="cpu", weights_only=False)
    model = DY.build(ck["kind"], ck["size"], ck["obs_dim"], ck["act_dim"])
    model.load_state_dict(ck["model"])
    model.eval()

    env = FailEnv("reveal", "rolling", world="earth", fail_p=0.5)
    r = mujoco.Renderer(env.model, height=a.height, width=a.width)
    cam = mujoco.MjvCamera()
    cam.elevation = -16.0

    frames = []
    for ep in range(a.episodes):
        o = env.reset(a.seed0 + ep)
        p0 = env.pos().copy()
        ghost_xy = p0[:2].copy()
        pred = torch.as_tensor(o[None, :].astype(np.float32))
        st = None
        rng = np.random.default_rng(a.seed0 + ep)
        hov = env.hover_action()
        drift = 0.0
        for k in range(a.horizon):
            u = np.clip(hov + 0.22 * rng.normal(size=env.act_dim), 0.0, 1.0)
            with torch.no_grad():
                pred, st = model(pred,
                                 torch.as_tensor(u[None, :].astype(np.float32)),
                                 st)
            po = pred.numpy()[0]
            # Position is not observed; integrate the predicted
            # velocity, which is what a planner would have to do.
            ghost_xy = ghost_xy + po[10:13][:2] * 10.0 * 0.01
            Rg, alt_p = pose_from_obs(po)

            o, _rw, done, _ = env.step(u)
            drift = float(np.linalg.norm(po - o))

            if k % a.every == 0 or done:
                p = env.pos()
                cam.lookat[:] = [0.5 * (p[0] + ghost_xy[0]),
                                 0.5 * (p[1] + ghost_xy[1]),
                                 max(p[2] * 0.8, 0.5)]
                spread = float(np.linalg.norm(p[:2] - ghost_xy))
                cam.distance = 11.0 + 0.8 * spread + 0.4 * max(
                    env.altitude() - 2.0, 0.0)
                cam.azimuth = 128.0 + 12.0 * math.sin(k / 120.0)
                r.update_scene(env.data, cam)
                gz = env.ground_z() + alt_p
                add_ghost(r.scene, mujoco,
                          np.array([ghost_xy[0], ghost_xy[1], gz]), Rg)
                img = Image.fromarray(r.render()).convert("RGB")
                frames.append(hud(img, k + 1, a.horizon, drift,
                                  env.altitude(), alt_p,
                                  f"{a.kind} {a.size}  "
                                  f"({ck['obs_dim']}-d state)   flight "
                                  f"{ep + 1}"))
            if done:
                break
        print(f"    flight {ep + 1}: {env.outcome}, final drift {drift:.3f}")
    r.close()
    name = a.out or f"dyn-ghost-{a.size}"
    save_gif(frames, OUT / f"{name}.gif", ms=60, colors=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
