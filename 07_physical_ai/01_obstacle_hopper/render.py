#!/usr/bin/env python3
"""
Render the world and the trained policy to PNGs and GIFs.

    uv run render.py              # everything
    uv run render.py --stills     # just the world
    uv run render.py --gif        # just the before/after animation

OPTIONAL BY DESIGN
------------------
Nothing in the lab depends on this script. Training uses state
observations -- joint angles and velocities -- so it never touches a
graphics stack, which is why `train_ppo.py` and the test suite run
anywhere including a bare container. Rendering is the one part that needs
OpenGL, and OpenGL is the one part that reliably breaks on a rented box.

So this script is separate, and it **fails with an explanation rather than
a traceback**. If no backend works you lose the pictures and nothing else.

MuJoCo offers three backends and which one exists is a property of the
machine, not of the code: `egl` (headless, GPU), `osmesa` (headless, CPU),
`glfw` (needs a display). The script tries each and reports which worked.
On the box these figures were made, `egl` and `osmesa` both failed to
initialise and `glfw` succeeded -- the opposite of what a headless-server
guide would predict, which is why it probes instead of assuming.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("glfw", "egl", "osmesa")


def pick_backend() -> str:
    """
    Find a working OpenGL backend, or explain what to do about it.

    Each attempt happens in a subprocess because MuJoCo binds its GL
    backend once per process at first use -- trying `egl` and then falling
    back to `glfw` in the same interpreter does not work, and the second
    attempt fails in a way that looks like a bug in the second backend.
    """
    import subprocess

    forced = os.environ.get("MUJOCO_GL")
    order = (forced,) if forced else BACKENDS

    probe = ("import mujoco;"
             "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>"
             "<geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
             "r=mujoco.Renderer(m,64,64);"
             "import mujoco as _m;d=_m.MjData(m);_m.mj_forward(m,d);"
             "r.update_scene(d);r.render()")

    for backend in order:
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                           capture_output=True, timeout=120)
            print(f"  [render] using MUJOCO_GL={backend}")
            return backend
        except Exception:                                      # noqa: BLE001
            print(f"  [render] MUJOCO_GL={backend} unavailable")

    print()
    print("=" * 72)
    print("  No working OpenGL backend — cannot render, but nothing else")
    print("  in this lab is affected.")
    print("=" * 72)
    print("\n  Training, evaluation and all 22 property checks use state")
    print("  observations and never open a graphics context:")
    print("\n      uv run train_ppo.py")
    print("      uv run ../../tests/test_obstacle_hopper.py")
    print("\n  To get pictures, install one headless backend:")
    print("      apt-get install -y libegl1 libgl1-mesa-dri   # then egl")
    print("      apt-get install -y libosmesa6                # then osmesa")
    print("  and re-run. Override the probe with MUJOCO_GL=<backend>.\n")
    sys.exit(1)


def frames_for(policy, env, seed: int, every: int, width: int, height: int,
               max_frames: int) -> list:
    """Roll out one episode, keeping every Nth frame, camera tracking x."""
    import mujoco
    import numpy as np

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8

    obs, _ = env.reset(seed=seed)
    out, t = [], 0
    while len(out) < max_frames:
        obs, _, term, trunc, _ = env.step(policy(obs))
        if t % every == 0:
            # Track the robot, but never scroll backwards -- a camera that
            # jitters back and forth makes a fall look like a cut.
            x = float(env.data.qpos[0])
            cam.lookat[:] = [max(0.95, x + 0.35), 0.0, 0.55]
            renderer.update_scene(env.data, camera=cam)
            out.append(renderer.render().copy())
        t += 1
        if term or trunc:
            break
    renderer.close()
    return out


def label(frame, text: str, colour=(230, 240, 246)):
    """Stamp a caption using PIL if present, silently skip if not."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((14, 10), text, fill=colour)
    import numpy as np
    return np.asarray(img)


def load_policy(run: str):
    """The trained actor, as a deterministic function of the observation."""
    import numpy as np
    import torch

    from obstacle_env import ACT_DIM, OBS_DIM
    from ppo import ActorCritic, RunningNorm

    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(OBS_DIM, ACT_DIM)
    net.load_state_dict(ck["model"])
    norm = RunningNorm(OBS_DIM)
    norm.load_state_dict(ck["norm"])

    def policy(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).numpy()
    return policy


def do_stills(args) -> None:
    """One strip showing the same world at three obstacle heights."""
    import imageio.v3 as iio
    import mujoco
    import numpy as np

    from obstacle_env import ObstacleHopper

    panels = []
    for h in (0.03, 0.10, 0.17):
        env = ObstacleHopper(fixed_height=h)
        env.reset(seed=0)
        r = mujoco.Renderer(env.model, height=380, width=600)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0.95, 0, 0.55]
        cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8
        r.update_scene(env.data, camera=cam)
        panels.append(label(r.render(), f"step height {h:.2f} m"))
        r.close()

    iio.imwrite(OUT / "hopper-world.png", np.hstack(panels))
    print("  hopper-world.png")


def do_gif(args) -> None:
    """
    Before and after, stacked, from the same seed and the same obstacle.

    The 'before' is the random policy -- the lab's actual measured
    baseline, not a hand-picked bad run. It collapses in about a second,
    which is what the termination condition is for.
    """
    import imageio.v3 as iio
    import numpy as np

    from obstacle_env import ObstacleHopper, random_policy

    rng = np.random.default_rng(0)
    runs = sorted(p.name for p in (HERE / "runs").iterdir()
                  if p.is_dir() and p.name.startswith("seeing"))
    if not runs:
        print("  no trained policy in runs/ — train first:\n"
              "      uv run train_ppo.py --name seeing_s0", file=sys.stderr)
        sys.exit(1)

    import json
    best = max(runs, key=lambda n: json.loads(
        (HERE / "runs" / n / "summary.json").read_text())["final"]["return"])
    print(f"  using {best} (best final return of {len(runs)} runs)")

    h = args.height
    before = frames_for(lambda o: random_policy(rng)(o),
                        ObstacleHopper(fixed_height=h), 3, args.every,
                        args.width, args.height_px, args.max_frames)
    after = frames_for(load_policy(best), ObstacleHopper(fixed_height=h),
                       3, args.every, args.width, args.height_px,
                       args.max_frames)

    # Pad the shorter clip by holding its last frame, so the two panels stay
    # in step. The random policy terminates early -- that IS the result, and
    # freezing on the collapse shows it rather than hiding it.
    n = max(len(before), len(after))
    before += [before[-1]] * (n - len(before))
    after += [after[-1]] * (n - len(after))

    frames = [np.vstack([label(b, f"BEFORE  random policy  (step {h:.2f} m)"),
                         label(a, "AFTER  600k steps of PPO")])
              for b, a in zip(before, after)]
    # Palette-quantise. At full colour this GIF was 4.1 MB, which is a slow
    # page on a phone for an animation whose content is two blue shapes on
    # a dark floor. 64 colours is visually indistinguishable here and costs
    # about a quarter of the bytes.
    from PIL import Image
    pil = [Image.fromarray(f).convert(
        "P", palette=Image.ADAPTIVE, colors=args.colors) for f in frames]
    pil[0].save(OUT / "hopper-before-after.gif", save_all=True,
                append_images=pil[1:], duration=args.frame_ms, loop=0,
                optimize=True)
    print(f"  hopper-before-after.gif ({len(frames)} frames)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--height", type=float, default=0.10,
                    help="obstacle height for the animation, metres")
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height-px", type=int, default=300)
    ap.add_argument("--every", type=int, default=4,
                    help="keep every Nth simulation step")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--frame-ms", type=int, default=80)
    ap.add_argument("--colors", type=int, default=64,
                    help="GIF palette size; 64 is visually "
                         "lossless here and a quarter the bytes")
    args, _ = ap.parse_known_args()
    if not (args.stills or args.gif):
        args.stills = args.gif = True

    os.environ["MUJOCO_GL"] = pick_backend()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    print(f"  writing to {OUT}")

    if args.stills:
        do_stills(args)
    if args.gif:
        do_gif(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
