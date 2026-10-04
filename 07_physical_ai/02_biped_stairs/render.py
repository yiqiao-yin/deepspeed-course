#!/usr/bin/env python3
"""
Render the staircase and the two interesting policies, with a HUD.

The comparison worth animating is `2leg_locked` against `2leg_free`: one
scores nearly three times the return and climbs almost nothing, the other
scores less and reaches the top. Seeing them side by side is the fastest
way to understand why the headline metric and the task disagree.

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

    from morphology import N_STAIRS, STAIR_END_X, STAIR_X0

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8

    obs, _ = env.reset(seed=seed)
    out, t, total = [], 0, 0.0
    alive = True

    while len(out) < max_frames:
        obs, rew, term, trunc, info = env.step(policy(obs))
        total += rew
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

            gap = STAIR_X0 - x
            out.append(hud(
                renderer.render(),
                title=title, subtitle=subtitle,
                stats=[("step", f"{t}"),
                       ("distance", f"{x:+.2f} m"),
                       ("to the step", f"{gap:+.2f} m" if gap > 0 else "past"),
                       ("torso height", f"{env.torso_height():.2f} m"),
                       ("treads climbed", f"{env.steps_climbed()}/{N_STAIRS}"),
                       ("rise / step", f"{env.rise:.2f} m"),
                       ("return", f"{total:.0f}")],
                status=("AT THE TOP" if env.at_top() else
                        "UPRIGHT" if alive else
                        "TIPPED" if env.tipped() else "FALLEN"),
                status_ok=alive,
                progress=max(0.0, x / STAIR_END_X),
                bar_label="progress along the staircase"))
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


def label(frame, text: str):
    """A plain caption for the still frames, where no HUD is wanted."""
    from PIL import Image, ImageDraw
    import numpy as np

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([12, 10, 20 + int(d.textlength(text, _font(13, True))),
                         38], 6, fill=(8, 24, 42, 205),
                        outline=(45, 90, 134, 255))
    d.text((18, 17), text, font=_font(13, True), fill=INK)
    return np.asarray(img)


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
    """The trained actor for a run directory, as obs -> action."""
    import json

    import numpy as np
    import torch

    from morphology import act_dim, obs_dim
    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(obs_dim(meta["legs"], meta["locked_torso"]),
                      act_dim(meta["legs"]))
    net.load_state_dict(ck["model"])
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    def policy(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).numpy()
    return policy, meta


def best_run(cell: str) -> str:
    """
    The seed with the highest SUMMIT RATE, not the highest return.

    Picking by return would select the policy that runs fastest on the
    flat, which for `2leg_locked` is exactly the one that never climbs --
    and the animation exists to show climbing. Choosing the metric that
    matches what the picture is for is the same decision this lab is
    about.
    """
    import json

    runs = sorted(p.name for p in (HERE / "runs").iterdir()
                  if p.is_dir() and p.name.startswith(cell)
                  and (p / "summary.json").exists())
    if not runs:
        print(f"  no trained run for {cell} — train it first:\n"
              f"      uv run train_ppo.py --cell {cell} --seed 0",
              file=sys.stderr)
        sys.exit(1)
    return max(runs, key=lambda n: json.loads(
        (HERE / "runs" / n / "summary.json").read_text())["final"]["at_top"])


def do_stills(args) -> None:
    """The staircase at three steepnesses, and all four robots."""
    import imageio.v3 as iio
    import mujoco
    import numpy as np

    from morphology import CELLS, RISE_MAX, RISE_MIN
    from stairs_env import BipedStairs

    def shot(env, caption: str, w=560, h=360):
        r = mujoco.Renderer(env.model, height=h, width=w)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [1.35, 0, 0.62]
        cam.distance, cam.azimuth, cam.elevation = 3.6, 90, -9
        r.update_scene(env.data, camera=cam)
        out = label(r.render(), caption)
        r.close()
        return out

    panels = []
    for rise in (RISE_MIN, 0.07, RISE_MAX):
        e = BipedStairs(legs=2, locked_torso=False, fixed_rise=rise)
        e.reset(seed=0)
        panels.append(shot(e, f"rise {rise:.2f} m per step"))
    iio.imwrite(OUT / "stairs-world.png", np.hstack(panels))
    print("  stairs-world.png")

    panels = []
    for name, legs, locked in CELLS:
        e = BipedStairs(legs=legs, locked_torso=locked, fixed_rise=0.08)
        e.reset(seed=0)
        panels.append(shot(e, name.replace("_", "  "), w=420, h=300))
    iio.imwrite(OUT / "stairs-morphologies.png",
                np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    print("  stairs-morphologies.png")


def do_gif(args) -> None:
    """One clip per interesting cell, each with the live HUD."""
    from stairs_env import BipedStairs

    for cell, title in (("2leg_locked", "TORSO LOCKED"),
                        ("2leg_free", "TORSO FREE")):
        run = best_run(cell)
        policy, meta = load_policy(run)
        env = BipedStairs(legs=meta["legs"], locked_torso=meta["locked_torso"],
                          fixed_rise=args.rise)
        f = meta["final"]
        frames = frames_for(
            policy, env, seed=4, every=args.every, width=args.width,
            height=args.height_px, max_frames=args.max_frames,
            title=title,
            subtitle=f"{meta['legs']} legs · return {f['return']:.0f} · "
                     f"summit {f['at_top']:.0%}")
        _save_gif(frames, OUT / f"stairs-{cell}.gif", args.frame_ms,
                  args.colors)


def _save_gif(frames, path, frame_ms: int, colors: int) -> None:
    """Palette-quantise; full colour costs four times the bytes here."""
    from PIL import Image

    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=colors) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--rise", type=float, default=0.07)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height-px", type=int, default=340)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--frame-ms", type=int, default=80)
    ap.add_argument("--colors", type=int, default=48)
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
