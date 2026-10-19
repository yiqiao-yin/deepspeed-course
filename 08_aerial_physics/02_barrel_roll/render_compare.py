#!/usr/bin/env python3
"""
Two rigs, one identical throw, side by side.

    uv run render_compare.py --left reveal --right minimal --classical
    uv run render_compare.py --left runs/reveal_s0 --right runs/minimal_s0
    uv run render_compare.py --seeds 3 --out roll-compare

WHAT THIS PICTURE IS FOR
-------------------------
Lab 2's question is whether a rig's COUPLE PURITY -- the share of its
roll torque it can produce without also shoving itself somewhere --
predicts whether it can roll and then land. A table of success rates
states the answer. It does not show it.

Side by side on the SAME launch, with the same initial attitude,
tumble and velocity, the difference is visible rather than inferred:
both cars turn over, and only one of them is still in a position to
land afterwards.

The throws are identical because both environments are reset with the
same seed, and the launch is a pure function of that seed. The HUD on
each panel carries its own roll dial and propellant bar, so a viewer
can see which one is paying more to achieve the same turn.

RENDERED SEQUENTIALLY
---------------------
One renderer at a time, composited afterwards. Concurrent MuJoCo
renderers mean concurrent GL contexts, which corrupts the heap under
GLFW -- topic 7 found that with sixteen of them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="reveal")
    ap.add_argument("--right", default="minimal")
    ap.add_argument("--classical", action="store_true",
                    help="use the scheduled controller on both sides")
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=500)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--colors", type=int, default=80)
    ap.add_argument("--out", default="roll-compare")
    a, _ = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    from render_roll import pick_backend

    os.environ["MUJOCO_GL"] = pick_backend()

    import mujoco
    import torch
    from PIL import Image, ImageDraw

    from render_roll import add_plumes, add_vectors, hud
    from roll_env import RollEnv

    def side(spec: str):
        """A rig name means the scheduled controller; a run dir means PPO."""
        if Path(spec).exists() and (Path(spec) / "policy.pt").exists():
            import json

            from evaluate import load_policy
            meta = json.loads((Path(spec) / "summary.json").read_text())
            net, norm = load_policy(spec)
            return dict(rig=meta["rig"], net=net, norm=norm,
                        label=f"PPO  {Path(spec).name}")
        return dict(rig=spec, net=None, norm=None,
                    label=f"scheduled  {spec}")

    L, R = side(a.left), side(a.right)
    W, H = a.width, a.height
    frames = []
    summary = []

    for ep in range(a.seeds):
        seed = a.seed0 + ep
        strips = []
        for s in (L, R):
            env = RollEnv(s["rig"], a.preset)
            o = env.reset(seed)
            ctl = None
            if s["net"] is None:
                from classical import RollController
                ctl = RollController(env)
            r = mujoco.Renderer(env.model, height=H, width=W)
            cam = mujoco.MjvCamera()
            cam.elevation = -14.0
            shots, trace, k = [], [], 0
            while True:
                if ctl is not None:
                    act = ctl.act()
                else:
                    with torch.no_grad():
                        x = torch.as_tensor(s["norm"](o),
                                            dtype=torch.float32).unsqueeze(0)
                        act = s["net"].distribution(x).mean.squeeze(0).numpy()
                o, _rw, done, _ = env.step(np.clip(act, 0.0, 1.0))
                k += 1
                trace.append((max(env.altitude(), 0.0),
                              float(np.linalg.norm(env.omega()))))
                if k % a.every == 0 or done:
                    p = env.pos()
                    alt = max(env.altitude(), 0.0)
                    cam.lookat[:] = [p[0], p[1], max(p[2] * 0.85, 0.5)]
                    cam.distance = 11.0 + 0.45 * max(alt - 2.0, 0.0)
                    cam.azimuth = 128.0
                    r.update_scene(env.data, cam)
                    add_plumes(r.scene, env, mujoco)
                    add_vectors(r.scene, env, mujoco)
                    img = Image.fromarray(r.render()).convert("RGB")
                    shots.append(hud(img, env, k, env.max_steps,
                                     f"{s['label']}  |  throw {ep + 1}",
                                     trace=trace))
                if done:
                    break
            r.close()
            shots.extend([shots[-1]] * 12)
            strips.append(shots)
            summary.append((seed, s["label"], env.outcome, env.rolled(),
                            env.gas_used))
            print(f"    seed {seed} {s['label']:<24} {env.outcome:<18} "
                  f"rolled {int(env.rolled())} gas {env.gas_used:>5.0f} kg",
                  flush=True)

        n = max(len(x) for x in strips)
        for k in range(n):
            sheet = Image.new("RGB", (2 * W, H), (8, 12, 16))
            for i, sh in enumerate(strips):
                sheet.paste(sh[min(k, len(sh) - 1)], (i * W, 0))
            d = ImageDraw.Draw(sheet)
            d.line([W, 0, W, H], fill=(60, 80, 100), width=2)
            frames.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\n  Same seed means the same launch: identical attitude, tumble")
    print("  and velocity on both sides. The ONLY difference is which")
    print("  half of the manoeuvre each policy was taught first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
