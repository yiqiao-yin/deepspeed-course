#!/usr/bin/env python3
"""
Every body the search trained, walking the same hillside.

    uv run render_search.py                  # the grid, ranked
    uv run render_search.py --cols 5 --seconds 12
    uv run render_search.py --solo bs_m4     # one body, large

Reads `runs/ranking.json`, so run `rank_bodies.py` first.

WHY A RANKED GRID
-----------------
A table says the best body walks four times further than the worst.
A grid says WHY in a way a table cannot: tiles are ordered by
distance walked, so the eye travels from the thing that falls over
to the thing that does not, and the shapes change as it goes.

Every tile is the same terrain, same seed, same task, same training
budget. The only variable is the machine, and the ordering is the
result.

MEDIAN RUNS, NOT SELECTED ONES
-------------------------------
Each body is filmed on the seed closest to ITS OWN median distance,
with the range printed on the tile. Filming each body's best run
would make a better reel and overstate the spread the table reports
-- and the spread is the whole finding.

Rendered one body at a time and composited: concurrent MuJoCo
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
          "      uv run rank_bodies.py\n")
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


def make_env(meta):
    from mars_env import NavWorld

    layout = None
    if meta.get("body_spec"):
        from robot import make_layout
        layout = make_layout(**meta["body_spec"])
    return NavWorld(mode=meta["mode"], flat=meta.get("flat", False), seed=0,
                    relief_scale=meta.get("relief_scale", 1.0),
                    upright_w=meta.get("upright_w", 0.0),
                    morphology=meta.get("morphology", "biped"),
                    integrator=meta.get("integrator"), layout=layout)


def median_seed(net, norm, meta, seeds) -> tuple[int, float, float, float]:
    """The seed nearest this body's own median distance."""
    import torch

    env = make_env(meta)
    got = []
    for sd in seeds:
        obs, _ = env.reset(seed=sd)
        prev = env.pos().copy()
        d = 0.0
        while True:
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                a = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, _i = env.step(a)
            now = env.pos()
            d += float(np.linalg.norm(now - prev))
            prev = now.copy()
            if term or trunc:
                break
        got.append((sd, d))
    ds = [d for _s, d in got]
    med = float(np.median(ds))
    sd_best = min(got, key=lambda t: abs(t[1] - med))
    return sd_best[0], sd_best[1], min(ds), max(ds)


def film(run, seed, tile_w, tile_h, steps, every):
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    net, norm, meta = load(run)
    env = make_env(meta)
    obs, _ = env.reset(seed=seed)
    r = mujoco.Renderer(env.model, height=tile_h, width=tile_w)
    cam = mujoco.MjvCamera()
    legs = len(meta["body_spec"]["legs"]) if False else None
    spec = meta.get("body_spec") or {}
    label = (f"{spec.get('n_legs', '?')} legs {spec.get('posture', '')}"
             if spec else meta.get("morphology", run))
    shots, travelled = [], 0.0
    prev = env.pos().copy()
    for t in range(steps):
        with torch.no_grad():
            x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
            a = net.distribution(x).mean.squeeze(0).numpy()
        obs, _, term, trunc, _i = env.step(a)
        now = env.pos()
        travelled += float(np.linalg.norm(now - prev))
        prev = now.copy()
        if t % every == 0:
            px, py = env.pos()
            cam.distance, cam.elevation = 4.6, -17.0
            cam.azimuth = math.degrees(env.yaw()) + 180.0
            cam.lookat[:] = [px, py, env._h_at(px, py) + 0.35]
            r.update_scene(env.data, cam)
            img = Image.fromarray(r.render())
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([0, 0, tile_w, 38], fill=(10, 16, 22, 228))
            d.text((7, 3), f"{run}   {label}", fill=INK)
            d.text((7, 20), f"{env.act_dim} actuators", fill=GREY)
            d.rectangle([0, tile_h - 19, tile_w, tile_h],
                        fill=(10, 16, 22, 228))
            d.text((7, tile_h - 16), f"{travelled:>5.1f} m", fill=INK)
            d.text((tile_w - 62, tile_h - 16),
                   "DOWN" if env.fallen() else "up",
                   fill=RED if env.fallen() else GREEN)
            shots.append(img)
        if term or trunc:
            break
    r.close()
    return shots, travelled


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", default="runs/ranking.json")
    ap.add_argument("--solo", default=None)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-w", type=int, default=300)
    ap.add_argument("--tile-h", type=int, default=220)
    ap.add_argument("--steps", type=int, default=4200)
    ap.add_argument("--every", type=int, default=34)
    ap.add_argument("--probe-seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--out", default="body-search")
    ap.add_argument("--frame-ms", type=int, default=75)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    from PIL import Image

    if a.solo:
        runs = [a.solo]
    else:
        p = Path(a.ranking)
        if not p.exists():
            print(f"\n  no {p}. Rank first:\n      uv run rank_bodies.py\n")
            return 1
        rank = json.loads(p.read_text())
        runs = [r["run"] for r in rank]      # already sorted by distance
    print(f"  {len(runs)} bodies, ranked by distance walked")

    seeds = list(range(a.seed0, a.seed0 + a.probe_seeds))
    strips = []
    for run in runs:
        net, norm, meta = load(run)
        sd, dist, lo, hi = median_seed(net, norm, meta, seeds)
        shots, walked = film(run, sd, a.tile_w, a.tile_h, a.steps, a.every)
        strips.append(shots)
        print(f"    {run:<16} seed {sd} (median {dist:5.1f} m of "
              f"{lo:.1f}-{hi:.1f})  {len(shots):>4} frames", flush=True)

    cols = min(a.cols, len(strips))
    rows_n = math.ceil(len(strips) / cols)
    span = max(len(s) for s in strips)
    W, H = a.tile_w, a.tile_h
    frames = []
    for k in range(span):
        sheet = Image.new("RGB", (cols * W, rows_n * H), (8, 12, 16))
        for i, shots in enumerate(strips):
            sheet.paste(shots[min(k, len(shots) - 1)],
                        ((i % cols) * W, (i // cols) * H))
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
