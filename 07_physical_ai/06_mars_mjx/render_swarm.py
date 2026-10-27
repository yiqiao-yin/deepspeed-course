#!/usr/bin/env python3
"""
What parallel simulation looks like: many Mars rollouts at once.

    uv run render_swarm.py --tiles 16
    uv run render_swarm.py --tiles 9 --view iso

WHY A MONTAGE AND NOT ONE CAMERA
---------------------------------
This lab is about running many environments at once, and a single
tracking camera cannot show that. Sixteen tiles can: sixteen
independent episodes on sixteen different Mars maps, each with its own
robot, its own route and its own outcome, drawn into one frame on a
shared clock.

Every tile is a REAL rollout of the policy trained in lab 5 — not a
replay, not a scripted gait. Some arrive. Most do not, which is the
honest state of the task and the reason the grid is worth watching:
you can see the failure modes beside each other rather than one at a
time.

WHAT THE PICTURE DOES AND DOES NOT CLAIM
----------------------------------------
The episodes are simulated ONE AT A TIME and composited afterwards,
not stepped concurrently. Frame k of every tile is step k of that
episode, so the grid is synchronised in simulation time and a reader
comparing tiles is comparing like with like — but nothing here is
evidence of parallel throughput, and the picture is not offered as
such. The numbers for that are in `bench_backends.py`, and they say
the best GPU backend reaches 1.54x the CPU worker pool on this model
— well under the 5x the lab set as the bar for renting a card.

Rendering sequentially is also the only way this runs at all:
sixteen live MuJoCo renderers means sixteen OpenGL contexts, which
corrupts the heap under GLFW. One context, reused, is both safer and
indistinguishable in the output.
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
    print("\n  No OpenGL backend. The benchmark needs none:\n")
    print("      uv run probe_mjx.py\n")
    sys.exit(1)


def finished(state: dict) -> bool:
    return state["arrived"] or state["fell"] or state["timeout"]


def tile_label(d, w, h, idx, info, arrived, fell):
    """A small per-tile caption: which episode, and how it ended."""
    from PIL import ImageDraw

    col = ((92, 196, 141, 255) if arrived else
           (226, 110, 110, 255) if fell else (150, 170, 186, 255))
    d.rectangle([0, h - 18, w, h], fill=(10, 16, 22, 210))
    d.text((6, h - 15), f"#{idx:02d}  {info['to_goal']:.1f} m to go", fill=col)
    if arrived:
        d.text((w - 58, h - 15), "ARRIVED", fill=col)
    elif fell:
        d.text((w - 34, h - 15), "FELL", fill=col)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v3_flat_blind",
                    help="a lab 5 checkpoint")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--tile-w", type=int, default=240)
    ap.add_argument("--tile-h", type=int, default=180)
    ap.add_argument("--view", default="iso", choices=("iso", "chase", "top"))
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--every", type=int, default=9)
    ap.add_argument("--seed0", type=int, default=40_000)
    ap.add_argument("--out", default="mjx-swarm")
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--rough", action="store_true",
                    help="film on Mars relief even if the run trained flat")
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(LAB5))
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    from mars_env import NavWorld
    from ppo import ActorCritic, RunningNorm

    meta = json.loads((LAB5 / "runs" / a.run / "summary.json").read_text())
    ck = torch.load(LAB5 / "runs" / a.run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], meta.get("act_dim", 9),
                      hidden=meta.get("hidden", 64))
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    flat = meta.get("flat", False) and not a.rough
    n = a.tiles
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    print(f"  {n} episodes, {cols}x{rows} grid, "
          f"{cols*a.tile_w}x{rows*a.tile_h} px, "
          f"{'flat' if flat else 'Mars relief'}")

    cam = mujoco.MjvCamera()
    tiles, done = [], []
    for i in range(n):
        # One env per tile, differing ONLY in the reset seed -- that is
        # what selects the map and the A-B pair. Varying the
        # constructor seed instead would also vary the task pool, and
        # the tiles would no longer be the same experiment.
        e = NavWorld(mode=meta["mode"], flat=flat, seed=0,
                     goal_range=meta.get("goal_range"),
                     relief_scale=meta.get("relief_scale", 1.0),
                     upright_w=meta.get("upright_w", 0.0))
        obs, _ = e.reset(seed=a.seed0 + i)
        state = {"arrived": False, "fell": False, "timeout": False,
                 "info": {"to_goal": e.to_goal()}}

        # One renderer, built and released inside this episode. Sixteen
        # live contexts corrupt the heap under GLFW; see the docstring.
        r = mujoco.Renderer(e.model, height=a.tile_h, width=a.tile_w)
        shots = []
        for t in range(a.steps):
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = e.step(act)
            state["info"] = info
            # `fell` comes from the environment, not inferred from
            # `term`: an episode also terminates on arrival, and
            # reading termination as a fall would paint arriving
            # tiles red.
            if info["arrived"]:
                state["arrived"] = True
            elif info["fell"]:
                state["fell"] = True
            elif trunc:
                state["timeout"] = True

            if t % a.every == 0:
                px, py = e.pos()
                if a.view == "iso":
                    cam.orthographic = 0
                    cam.distance, cam.elevation, cam.azimuth = 6.5, -26.0, 130.0
                    cam.lookat[:] = [px, py, e._h_at(px, py)]
                elif a.view == "chase":
                    cam.orthographic = 0
                    cam.distance, cam.elevation = 4.5, -14.0
                    cam.azimuth = math.degrees(e.yaw()) + 180.0
                    cam.lookat[:] = [px, py, e._h_at(px, py) + 0.7]
                else:
                    cam.orthographic = 1
                    cam.distance, cam.elevation, cam.azimuth = 18.0, -90.0, 90.0
                    cam.lookat[:] = [px, py, 0]
                r.update_scene(e.data, cam)
                img = Image.fromarray(r.render())
                tile_label(ImageDraw.Draw(img, "RGBA"), a.tile_w, a.tile_h,
                           i, info, state["arrived"], state["fell"])
                shots.append(img)
            if finished(state):
                break
        r.close()
        tiles.append(shots)
        done.append(state)
        end = ("arrived" if state["arrived"] else
               "fell" if state["fell"] else "still walking")
        print(f"    tile {i:>2}  {len(shots):>4} frames  {end}", flush=True)

    # Composite. Frame k of the sheet is step k of every episode; a
    # tile that ended early holds its last frame, so the grid stays
    # synchronised in SIMULATION time rather than in list index.
    span = max(len(s) for s in tiles)
    frames = []
    for k in range(span):
        sheet = Image.new("RGB", (cols * a.tile_w, rows * a.tile_h),
                          (8, 12, 16))
        for i, shots in enumerate(tiles):
            sheet.paste(shots[min(k, len(shots) - 1)],
                        ((i % cols) * a.tile_w, (i // cols) * a.tile_h))
        frames.append(np.array(sheet))

    arrived = sum(x["arrived"] for x in done)
    fell = sum(x["fell"] for x in done)
    print(f"  outcome: {arrived} arrived, {fell} fell, "
          f"{n - arrived - fell} still walking at the cutoff")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=a.colors,
                                      dither=Image.Dither.NONE)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
