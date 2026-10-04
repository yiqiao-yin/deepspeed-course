#!/usr/bin/env python3
"""
Render the terrains, the policies, and what the robot actually sees.

    uv run render.py --clip blind      # a trained policy with NO camera
    uv run render.py --clip vision     # the same task, with depth
    uv run render.py --clip tour       # one long run, all three terrains
    uv run render.py --all

WHY THERE IS A DEPTH INSET
--------------------------
A robot succeeding on camera is not evidence that it looked. Both earlier
labs in this category shipped an information ablation that came back
null, and in each case the animation was just as convincing before the
ablation as after it.

So every vision clip carries the 64x64 depth image the policy is
actually driving on, drawn from the same `env.depth()` call that feeds
the network -- not a prettier second render. Watch the staircase darken
as the robot approaches it. That, plus the blank-image ablation in
`train_student.py`, is the evidence; the animation alone is not.

WHY `blind` IS THE HONEST "BEFORE"
----------------------------------
An under-trained policy falls over, which looks bad but shows nothing:
every policy falls over early, with or without a camera. `blind` is a
FULLY trained policy -- same algorithm, same steps, same reward -- that
differs from the vision student in one respect only. When it misses the
first tread it is not because it is under-trained. It is because it
cannot see.

`--clip blind --undertrained` renders the other thing as well, from a
2-epoch checkpoint, for anyone who wants the ordinary before/after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pathlib import Path as pathlib_Path

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("glfw", "egl", "osmesa")
_FONTS: dict = {}


def pick_backend() -> str:
    """
    Find a working OpenGL backend, or explain what to do about it.

    Each attempt happens in a subprocess because MuJoCo binds its GL
    backend once per process at first use -- trying `egl` and then
    falling back to `glfw` in the same interpreter does not work, and
    the second attempt fails in a way that looks like a bug in the
    second backend.
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

    print("\n" + "=" * 72)
    print("  No working OpenGL backend — cannot render, but nothing else")
    print("  in this lab is affected.")
    print("=" * 72)
    print("\n  Training, collection, the student and every check run on")
    print("  state observations and depth buffers obtained through the")
    print("  same context, so they need a backend too — but the ANALYSIS")
    print("  in tests/test_terrain_vision.py does not:")
    print("\n      uv run ../../tests/test_terrain_vision.py")
    print("\n  To get pictures, install one headless backend:")
    print("      apt-get install -y libegl1 libgl1-mesa-dri   # then egl")
    print("      apt-get install -y libosmesa6                # then osmesa")
    print("  and re-run. Override the probe with MUJOCO_GL=<backend>.\n")
    sys.exit(1)


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


INK = (233, 240, 246)
MUTED = (141, 163, 181)
GOOD = (92, 196, 141)
BAD = (226, 110, 110)
BLUE = (99, 163, 208)
PANEL = (8, 24, 42, 205)
EDGE = (45, 90, 134, 255)


# ---------------------------------------------------------------- overlays

def overlay(frame, *, title: str, subtitle: str, stats: list, status: str,
            status_ok: bool, progress: float, bar_label: str, depth=None):
    """
    Stats panel, progress bar, status chip, and the depth inset.

    Everything is read from the live `MjData` and from the same
    `env.depth()` the policy consumed on this step. Nothing is pre-baked,
    smoothed, or re-rendered at a nicer angle.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    pw, ph = 236, 34 + 20 * len(stats) + 46
    d.rounded_rectangle([12, 12, 12 + pw, 12 + ph], 8, fill=PANEL,
                        outline=EDGE)
    d.text((26, 22), title, font=_font(15, True), fill=INK)
    d.text((26, 41), subtitle, font=_font(11), fill=MUTED)

    y = 64
    for k, v in stats:
        d.text((26, y), k, font=_font(11), fill=MUTED)
        d.text((12 + pw - 16, y), v, font=_font(11, True), fill=INK,
               anchor="ra")
        y += 20

    y += 4
    d.text((26, y), bar_label, font=_font(10), fill=MUTED)
    y += 15
    x0, x1 = 26, 12 + pw - 16
    d.rounded_rectangle([x0, y, x1, y + 7], 3, fill=(20, 44, 68, 255))
    if progress > 0:
        d.rounded_rectangle([x0, y, x0 + (x1 - x0) * min(progress, 1.0),
                             y + 7], 3, fill=BLUE if progress < 1 else GOOD)

    col = GOOD if status_ok else BAD
    tw = d.textlength(status, font=_font(12, True))
    d.rounded_rectangle([W - tw - 40, H - 42, W - 16, H - 16], 6,
                        fill=PANEL, outline=col + (255,))
    d.text((W - 28 - tw / 2, H - 29), status, font=_font(12, True), fill=col,
           anchor="mm")

    if depth is not None:
        img = _inset(img, depth, W)
    return np.asarray(img)


def _inset(img, depth, W: int):
    """
    The 64x64 depth frame, upscaled, exactly as the network received it.

    Near is bright and far is dark, which is the inverse of the stored
    array -- `depth()` returns 0 at 0.8 m and 1 at 3.5 m. Inverting for
    display makes an approaching step brighten as it gets closer, which
    reads correctly; the array itself is untouched.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    side, pad = 144, 14
    x0, y0 = W - side - pad, pad
    vis = np.clip(1.0 - depth, 0.0, 1.0)
    rgb = np.stack([(vis * 210 + 10).astype(np.uint8),
                    (vis * 228 + 14).astype(np.uint8),
                    (vis * 246 + 24).astype(np.uint8)], axis=-1)
    tile = Image.fromarray(rgb).resize((side, side), Image.NEAREST)
    img.paste(tile, (x0, y0))

    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([x0 - 1, y0 - 1, x0 + side, y0 + side], outline=EDGE)
    d.rounded_rectangle([x0, y0 + side + 5, x0 + side, y0 + side + 24], 4,
                        fill=PANEL, outline=EDGE)
    d.text((x0 + side / 2, y0 + side + 14), "64×64 depth — what it sees",
           font=_font(9), fill=MUTED, anchor="mm")
    return img


def card(W: int, H: int, title: str, body: str, n: int = 10):
    """A short title card between terrains in the tour."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (6, 14, 24))
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 16), title, font=_font(30, True), fill=INK,
           anchor="mm")
    d.text((W // 2, H // 2 + 22), body, font=_font(14), fill=MUTED,
           anchor="mm")
    return [np.asarray(img)] * n


# ---------------------------------------------------------------- policies

def load_ppo(run: str):
    """A PPO policy from `runs/<run>` — used for the blind arm."""
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], 6)
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    def act(obs, _depth):
        with torch.no_grad():
            t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).numpy()

    return act, meta


def load_student(run: str = "student"):
    """The vision student. Takes the depth frame as well as proprioception."""
    import torch
    import torch.nn as nn

    from train_student import build

    ck = torch.load(HERE / "runs" / run / "student.pt", weights_only=False)
    net = build(torch, nn, ck["proprio_dim"], 6, ck["width"])
    net.load_state_dict(ck["model"])
    net.eval()

    def act(obs, depth):
        with torch.no_grad():
            return net(
                torch.as_tensor(depth).unsqueeze(0),
                torch.as_tensor(obs[:ck["proprio_dim"]],
                                dtype=torch.float32).unsqueeze(0),
            ).squeeze(0).numpy()

    return act, {}


def best_blind() -> str:
    cands = sorted(p.name for p in (HERE / "runs").iterdir()
                   if p.name.startswith("v3_blind_")
                   and (p / "summary.json").exists())
    if not cands:
        print("  no blind run found. Train one:\n"
              "      uv run train_teacher.py --no-privileged --name v3_blind_s0",
              file=sys.stderr)
        sys.exit(1)
    return max(cands, key=lambda n: json.loads(
        (HERE / "runs" / n / "summary.json").read_text())["final"]["past_all"])


# ---------------------------------------------------------------- rollouts

def episode(act, kind: str, *, seed: int, title: str, subtitle: str,
            width: int, height: int, every: int, show_depth: bool,
            needs_depth: bool):
    """
    One episode, returned as HUD'd frames.

    `needs_depth` and `show_depth` are separate on purpose: the blind
    clip renders no inset because the policy genuinely has no camera,
    and drawing one would imply it had.
    """
    import mujoco
    import numpy as np

    from terrain import EVENT_X
    from vision_env import TerrainWorld

    env = TerrainWorld(kind=kind, privileged=False,
                       depth=needs_depth or show_depth, seed=0)
    obs, _ = env.reset(seed=seed)

    r = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    frames, total, t, far = [], 0.0, 0, -9e9

    while True:
        img = env.depth() if (needs_depth or show_depth) else None
        a = act(obs, img)
        obs, rew, term, trunc, info = env.step(a)
        total += rew
        t += 1
        far = max(far, info["x"])

        if t % every == 0 or term:
            # Three-quarter, not side-on. At azimuth 90 the camera looks
            # straight down the head's wide axis, so the BD-1 head reads
            # edge-on as a small box and the robot looks like the
            # stick figure it was two labs ago. 108 shows the head's
            # width while keeping enough of the terrain's side profile
            # to see what the treads are doing.
            cam.lookat[:] = [info["x"] + 0.18, 0, env.torso_height() - 0.42]
            cam.distance, cam.elevation, cam.azimuth = 3.2, -9, 108
            r.update_scene(env.data, cam)
            alive = not info["fell"]
            frames.append(overlay(
                r.render(), title=title, subtitle=subtitle,
                stats=[("terrain", kind),
                       ("distance", f"{info['x']:+.2f} m"),
                       ("torso height", f"{env.torso_height():.2f} m"),
                       ("return", f"{total:.0f}")],
                status="CLEARED" if info["past_event"]
                       else ("WALKING" if alive else "FELL"),
                status_ok=alive,
                progress=(info["x"] + 1.0) / (EVENT_X + 1.3 + 1.0),
                bar_label=f"progress to x = {EVENT_X + 1.3:.1f} m",
                depth=img if show_depth else None))
        if term or trunc:
            break

    return frames, {"past": bool(info["past_event"]), "x": float(far),
                    "ret": total, "steps": t}


# ------------------------------------------------------------------- clips

def clip_blind(a) -> None:
    """A fully trained policy that cannot see, meeting a staircase."""
    run = best_blind()
    act, meta = load_ppo(run)
    print(f"  blind policy: {run} (past_all {meta['final']['past_all']:.0%})")
    frames = []
    for kind in ("up", "down"):
        f, res = episode(act, kind, seed=a.seed,
                         title="No camera", subtitle=f"{run} · proprioception only",
                         width=a.width, height=a.height, every=a.every,
                         show_depth=False, needs_depth=False)
        print(f"    {kind:<5} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}")
        frames += f + card(a.width, a.height, "", "", 6)
    _save(frames[:-6], OUT / "terrain-blind.gif", a)


def clip_vision(a) -> None:
    """The student, same terrains, with the depth inset."""
    act, _ = load_student(a.student)
    frames = []
    for kind in ("up", "down"):
        f, res = episode(act, kind, seed=a.seed,
                         title="With a camera",
                         subtitle="vision student · 64×64 depth + proprioception",
                         width=a.width, height=a.height, every=a.every,
                         show_depth=True, needs_depth=True)
        print(f"    {kind:<5} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}")
        frames += f + card(a.width, a.height, "", "", 6)
    _save(frames[:-6], OUT / "terrain-vision.gif", a)


def clip_tour(a) -> None:
    """The long one: every terrain, back to back, camera view throughout."""
    act, _ = load_student(a.student)
    frames = card(a.width, a.height, "One policy, three terrains",
                  "flat · upstairs · downstairs — it is told which only by its camera", 14)
    for kind, blurb in (("flat", "nothing to do but walk"),
                        ("up", "three treads up — lift before contact"),
                        ("down", "three treads down — you cannot feel a descent")):
        frames += card(a.width, a.height, kind.upper(), blurb, 10)
        f, res = episode(act, kind, seed=a.seed,
                         title=f"Terrain: {kind}",
                         subtitle="vision student · the same weights on all three",
                         width=a.width, height=a.height, every=a.every,
                         show_depth=True, needs_depth=True)
        print(f"    {kind:<5} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}  "
              f"({res['steps']} steps)")
        frames += f
    _save(frames, OUT / "terrain-tour.gif", a)


def clip_untrained(a) -> None:
    """The ordinary before/after: a 2-epoch student on the same terrain."""
    if not (HERE / "runs" / "student_poor" / "student.pt").exists():
        print("  no under-trained checkpoint. Make one:\n"
              "      uv run train_student.py --epochs 2 --tag student_poor",
              file=sys.stderr)
        return
    act, _ = load_student("student_poor")
    frames = []
    for kind in ("up", "down"):
        f, res = episode(act, kind, seed=a.seed,
                         title="Under-trained",
                         subtitle="same network, 2 epochs instead of 40",
                         width=a.width, height=a.height, every=a.every,
                         show_depth=True, needs_depth=True)
        print(f"    {kind:<5} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}")
        frames += f + card(a.width, a.height, "", "", 6)
    _save(frames[:-6], OUT / "terrain-untrained.gif", a)


def _save(frames, path, a) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=a.colors,
                                      dither=Image.Dither.NONE)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KB)")


CLIPS = {"blind": clip_blind, "vision": clip_vision, "tour": clip_tour,
         "untrained": clip_untrained}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=sorted(CLIPS), default="tour")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--student", default="student")
    ap.add_argument("--seed", type=int, default=8002,
                    help="8002 clears all three terrains. The student is "
                         "79%% on `up`, so some seeds genuinely fail -- "
                         "the per-terrain rates in train_student.py are the "
                         "claim, and this clip is an illustration of it")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=430)
    ap.add_argument("--every", type=int, default=6,
                    help="render every Nth control step. At 5 the clip runs "
                         "near real time (the control step is 10 ms) and a "
                         "full episode costs ~120 frames instead of 600 -- "
                         "the difference between a 2 MB asset and a 9 MB one")
    ap.add_argument("--frame-ms", type=int, default=55)
    ap.add_argument("--colors", type=int, default=64)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))
    print(f"  writing to {OUT}")
    for name in (sorted(CLIPS) if a.all else [a.clip]):
        print(f"\n  [{name}]")
        CLIPS[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
