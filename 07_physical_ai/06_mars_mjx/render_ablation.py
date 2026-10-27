#!/usr/bin/env python3
"""
The 8.6x, drawn: the same robot on a flat height field and on Mars.

    uv run render_ablation.py
    uv run render_ablation.py --steps 1200 --out mjx-ablation

WHAT THIS ANIMATION IS FOR
--------------------------
`bench_backends.py` reports something counterintuitive: Mars terrain
generates several times the contacts of flat ground, and costs almost
no GPU throughput for it -- 1.10x, against a 3.2% run-to-run spread.
Rocks and height-field resolution cost nothing measurable at all.

So this films both sides of that comparison. Left: a height field of
zeros. Right: the Mars course. Same robot, same policy, same seed,
same collider type -- the ONLY difference is whether the surface has
relief, which is exactly the variable the ablation isolates.

The readout under each panel is the live contact count, and it is
there to make the surprise visible rather than to prove the
conclusion. A height field is diced into triangles on contact, so a
robot on a slope generates many more contact pairs than the same
robot on a flat one, and you can watch the two numbers separate. The
lesson is that this large a difference in contacts buys so little
difference in throughput: the terrain is not where the time goes, and
an eyeball estimate from this animation alone would get it wrong.

NOTE ON THE CONTACT NUMBERS
---------------------------
These are measured while the policy WALKS, over a full episode. The
benchmark's contact column is measured differently -- a robot
settling under gravity for 300 steps -- so the two ratios are not the
same quantity and do not have to agree. Each is scoped where it is
reported; neither is the other's replication.

WHAT IT DOES NOT SHOW
---------------------
Nothing here is a timing measurement. Both panels are stepped on the
CPU and the frames are composited; the rendering cost has no
relationship to the physics cost. Contact counts ARE measured, live,
from `data.ncon`. For the throughput numbers see `bench_backends.py`.
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
    print("\n  No OpenGL backend. The numbers need none:\n")
    print("      uv run bench_backends.py --ablate-only\n")
    sys.exit(1)


def panel(img, title, subtitle, ncon, mean_ncon, accent):
    """Caption one side. Contact count is measured, not asserted."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, 0, w, 30], fill=(10, 16, 22, 220))
    d.text((12, 4), title, fill=accent)
    d.text((12, 17), subtitle, fill=(150, 170, 186, 255))
    d.rectangle([0, h - 34, w, h], fill=(10, 16, 22, 220))
    d.text((12, h - 29), "contacts this step", fill=(150, 170, 186, 255))
    d.text((12, h - 16), f"mean so far", fill=(150, 170, 186, 255))
    for row, val in ((h - 29, f"{ncon}"), (h - 16, f"{mean_ncon:.1f}")):
        d.text((w - 14 - d.textlength(val), row), val, fill=accent)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v3_flat_blind")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=40_003)
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--out", default="mjx-ablation")
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=64)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(LAB5))
    import mujoco
    import torch
    from PIL import Image

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

    # The two sides. `flat=True` is lab 5's flat world -- a height
    # field of zeros, NOT a plane geom, so both panels run the same
    # collider and the only variable is relief.
    sides = [
        ("FLAT", "height field of zeros", True, (120, 190, 240, 255)),
        ("MARS", "the course, same collider", False, (240, 150, 110, 255)),
    ]

    shots, stats = [], []
    for title, sub, flat, accent in sides:
        e = NavWorld(mode=meta["mode"], flat=flat, seed=0,
                     goal_range=meta.get("goal_range"),
                     relief_scale=meta.get("relief_scale", 1.0),
                     upright_w=meta.get("upright_w", 0.0))
        obs, _ = e.reset(seed=a.seed)
        r = mujoco.Renderer(e.model, height=a.height, width=a.width)
        cam = mujoco.MjvCamera()
        cam.orthographic = 0
        frames, ncons = [], []
        for t in range(a.steps):
            with torch.no_grad():
                x = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
                act = net.distribution(x).mean.squeeze(0).numpy()
            obs, _, term, trunc, info = e.step(act)
            ncons.append(int(e.data.ncon))
            if t % a.every == 0:
                px, py = e.pos()
                cam.distance, cam.elevation = 5.5, -20.0
                cam.azimuth = math.degrees(e.yaw()) + 180.0
                cam.lookat[:] = [px, py, e._h_at(px, py) + 0.6]
                r.update_scene(e.data, cam)
                frames.append(panel(Image.fromarray(r.render()), title, sub,
                                    ncons[-1], float(np.mean(ncons)), accent))
            if term or trunc:
                break
        r.close()
        shots.append(frames)
        stats.append((title, float(np.mean(ncons)), max(ncons), len(ncons)))
        print(f"    {title:<5} {len(ncons):>5} steps   mean contacts "
              f"{np.mean(ncons):>5.1f}   peak {max(ncons)}")

    lo, hi = stats[0][1], stats[1][1]
    print(f"\n  Mars generates {hi/max(lo, 1e-9):.1f}x the contacts of a flat "
          f"field of the same\n  resolution, under the same collider — while "
          f"costing only 1.10x the GPU\n  throughput (`bench_backends.py`). "
          f"The contacts are real; the slowdown\n  everyone expects from them "
          f"is not.")

    span = max(len(s) for s in shots)
    out = []
    for k in range(span):
        sheet = Image.new("RGB", (2 * a.width, a.height), (8, 12, 16))
        for i, fr in enumerate(shots):
            sheet.paste(fr[min(k, len(fr) - 1)], (i * a.width, 0))
        out.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in out]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {path.name}  ({len(out)} frames, "
          f"{path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
