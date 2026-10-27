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
from pathlib import Path as pathlib_Path

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


def frames_for(policy, env, *, seed: int, every: int, width: int,
               height: int, max_frames: int, title: str, subtitle: str
               ) -> list:
    """
    Roll out one episode and return HUD-annotated frames.

    Stats are read from the live simulation each frame, so the panel can
    never disagree with what the robot is doing.
    """
    import mujoco
    import numpy as np

    from obstacle_env import BOX_BACK_X, BOX_FRONT_X

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8

    obs, _ = env.reset(seed=seed)
    out, t, total, cleared = [], 0, 0.0, False
    alive = True

    while len(out) < max_frames:
        obs, rew, term, trunc, info = env.step(policy(obs))
        total += rew
        cleared |= info["cleared"]
        alive = not term

        if t % every == 0:
            # Track forward only -- a camera that scrolls back makes a
            # fall look like an edit.
            #
            # The offset keeps the robot just RIGHT of centre. The first
            # version centred 0.35 m ahead of it, which put the robot
            # directly behind the stats panel in the top-left for most of
            # every clip -- the HUD was hiding the thing it described.
            x = float(env.data.qpos[0])
            cam.lookat[:] = [max(0.20, x + 0.05), 0.0, 0.62]
            renderer.update_scene(env.data, camera=cam)

            gap = BOX_FRONT_X - x
            out.append(hud(
                renderer.render(),
                title=title, subtitle=subtitle,
                stats=[("step", f"{t}"),
                       ("distance", f"{x:+.2f} m"),
                       ("to the step", f"{gap:+.2f} m" if gap > 0 else "past"),
                       ("torso height", f"{env.torso_height():.2f} m"),
                       ("step height", f"{env.box_height:.2f} m"),
                       ("return", f"{total:.0f}")],
                status=("CLEARED" if cleared else
                        "UPRIGHT" if alive else "FALLEN"),
                status_ok=alive,
                progress=max(0.0, x / BOX_BACK_X),
                bar_label="progress to the far edge of the step"))
        t += 1
        if term or trunc:
            break

    # Hold the final frame so a short, failed episode is readable rather
    # than a flicker. The freeze IS the result for the untrained arm.
    if out:
        out += [out[-1]] * 8
    renderer.close()
    return out


_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    """DejaVu, which ships inside matplotlib, so there is nothing to install."""
    key = (size, bold)
    if key not in _FONTS:
        from PIL import ImageFont
        try:
            import matplotlib
            d = (pathlib_Path(matplotlib.__file__).parent / "mpl-data"
                 / "fonts" / "ttf")
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            _FONTS[key] = ImageFont.truetype(str(d / name), size)
        except Exception:                                      # noqa: BLE001
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


# Palette, matching the course's own.
INK = (233, 240, 246)
MUTED = (141, 163, 181)
ACCENT = (227, 160, 90)
GOOD = (92, 196, 141)
BAD = (226, 110, 110)
BLUE = (99, 163, 208)


def hud(frame, *, title: str, subtitle: str, stats: list, status: str,
        status_ok: bool, progress: float, bar_label: str):
    """
    Draw a stats panel over a rendered frame.

    An animation of a robot moving is pleasant and says very little. The
    same animation with the numbers the training loop is actually
    optimising -- distance, torso height, cumulative reward, whether the
    episode is still alive -- shows WHY it moves that way, and makes the
    failure case legible instead of just sad.

    Everything drawn here is read from the live `MjData`; nothing is
    pre-baked or smoothed.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Panel. Semi-transparent so the scene stays visible behind it.
    pw, ph = 232, 34 + 20 * len(stats) + 46
    d.rounded_rectangle([12, 12, 12 + pw, 12 + ph], 8,
                        fill=(8, 24, 42, 205), outline=(45, 90, 134, 255))

    d.text((26, 22), title, font=_font(15, True), fill=INK)
    d.text((26, 41), subtitle, font=_font(11), fill=MUTED)

    y = 64
    for k, v in stats:
        d.text((26, y), k, font=_font(11), fill=MUTED)
        d.text((12 + pw - 16, y), v, font=_font(11, True), fill=INK,
               anchor="ra")
        y += 20

    # Progress toward clearing the obstacle.
    y += 4
    d.text((26, y), bar_label, font=_font(10), fill=MUTED)
    y += 15
    x0, x1 = 26, 12 + pw - 16
    d.rounded_rectangle([x0, y, x1, y + 7], 3, fill=(20, 44, 68, 255))
    if progress > 0:
        d.rounded_rectangle([x0, y, x0 + (x1 - x0) * min(progress, 1.0),
                             y + 7], 3, fill=BLUE if progress < 1 else GOOD)

    # Status chip, bottom-right.
    col = GOOD if status_ok else BAD
    tw = d.textlength(status, font=_font(12, True))
    d.rounded_rectangle([W - tw - 40, H - 42, W - 16, H - 16], 6,
                        fill=(8, 24, 42, 205), outline=col + (255,))
    d.text((W - 28 - tw / 2, H - 29), status, font=_font(12, True),
           fill=col, anchor="mm")

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


def _save_gif(frames, path, frame_ms: int, colors: int) -> None:
    """
    Palette-quantise and write. At full colour the first of these was
    4.1 MB, which is a slow page on a phone for two blue shapes on a dark
    floor; 64 colours is visually indistinguishable here.
    """
    from PIL import Image

    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=colors) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=frame_ms, loop=0, optimize=True)
    kb = path.stat().st_size / 1024
    print(f"  {path.name}  ({len(frames)} frames, {kb:.0f} KB)")


def do_gif(args) -> None:
    """
    Three animations: untrained alone, trained alone, and the comparison.

    The two solo clips exist because the stacked version makes both panels
    small, and the interesting detail -- the HUD numbers, what the leg is
    actually doing -- is lost. The comparison answers "is it better?"; the
    solo clips answer "what is it doing?".

    The untrained arm is the RANDOM POLICY, which is the lab's measured
    baseline rather than a hand-picked bad run. It collapses in about a
    second and the clip freezes on the collapse, because that is the
    result.
    """
    import json

    import numpy as np

    from obstacle_env import ObstacleHopper, random_policy

    runs = sorted(p.name for p in (HERE / "runs").iterdir()
                  if p.is_dir() and p.name.startswith("seeing")
                  and (p / "summary.json").exists())
    if not runs:
        print("  no trained policy in runs/ — train first:\n"
              "      uv run train_ppo.py --name seeing_s0", file=sys.stderr)
        sys.exit(1)
    best = max(runs, key=lambda n: json.loads(
        (HERE / "runs" / n / "summary.json").read_text())["final"]["return"])
    summ = json.loads((HERE / "runs" / best / "summary.json").read_text())
    print(f"  using {best} (best of {len(runs)} seeds, "
          f"final return {summ['final']['return']:.0f})")

    h = args.height
    rng = np.random.default_rng(0)
    common = dict(seed=3, every=args.every, width=args.width,
                  height=args.height_px, max_frames=args.max_frames)

    before = frames_for(
        lambda o: random_policy(rng)(o), ObstacleHopper(fixed_height=h),
        title="BEFORE TRAINING",
        subtitle=f"random policy · baseline return "
                 f"{summ['baseline_return']:.0f}", **common)
    _save_gif(before, OUT / "hopper-before.gif", args.frame_ms, args.colors)

    after = frames_for(
        load_policy(best), ObstacleHopper(fixed_height=h),
        title="AFTER TRAINING",
        subtitle=f"PPO · {summ['total_steps'] // 1000}k steps · return "
                 f"{summ['final']['return']:.0f}", **common)
    _save_gif(after, OUT / "hopper-after.gif", args.frame_ms, args.colors)

    # No stacked side-by-side clip. One existed and was dropped: at half
    # scale the HUD numbers -- the entire reason the panel is there -- were
    # unreadable, and it pushed the page past 3.5 MB of animation. Two
    # full-size clips shown one after another read better and cost half as
    # much.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--height", type=float, default=0.10,
                    help="obstacle height for the animation, metres")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height-px", type=int, default=330)
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
