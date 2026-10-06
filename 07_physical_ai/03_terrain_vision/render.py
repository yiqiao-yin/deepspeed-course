#!/usr/bin/env python3
"""
Render the terrains, the policies, and what the robot actually sees.

    uv run render.py --all             # all six clips
    uv run render.py --clip compare    # blind vs vision, the controlled pair
    uv run render.py --clip heldout    # a staircase steeper than it trained on

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

WHICH "BLIND" ARM THE COMPARISON USES
-------------------------------------
`--clip compare` runs the blind BEHAVIOUR-CLONED student, not the blind
PPO policy the first version of this lab animated. That matters: the
PPO arm differs from the vision student in the camera AND in how it was
trained, so animating the two side by side invited exactly the wrong
conclusion -- and the published numbers drew it. The BC arm differs in
one thing.

One clip per scenario, because the terrains behave differently and an
average over them hides the result: `flat` and `down` are cleared with
no camera at all, and `up` is the only place the camera is worth
anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pathlib import Path as pathlib_Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from terrain import PATCH_KINDS, STONE_KINDS  # noqa: E402
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


def load_student(run: str = "student_s2"):
    """
    A behaviour-cloned student, with or without a camera.

    `use_depth` has to come from the CHECKPOINT. Building the camera
    network by default and loading blind weights into it fails loudly
    here (a 15-vs-271 shape mismatch), which is the good case -- but the
    same omission in a scoring path would silently compare the wrong
    architecture.
    """
    import torch
    import torch.nn as nn

    from train_student import build

    ck = torch.load(HERE / "runs" / run / "student.pt", weights_only=False)
    use_depth = ck.get("use_depth", True)
    net = build(torch, nn, ck["proprio_dim"], 6, ck["width"],
                use_depth=use_depth)
    net.load_state_dict(ck["model"])
    net.eval()

    def act(obs, depth):
        import numpy as np
        with torch.no_grad():
            img = depth if depth is not None else np.zeros((64, 64), np.float32)
            return net(
                torch.as_tensor(img).unsqueeze(0),
                torch.as_tensor(obs[:ck["proprio_dim"]],
                                dtype=torch.float32).unsqueeze(0),
            ).squeeze(0).numpy()

    return act, use_depth


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
            needs_depth: bool, rise: float | None = None,
            kinds=None, extra=None):
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

    env = TerrainWorld(kind=kind, privileged=(kinds is not None and extra),
                       depth=needs_depth or show_depth, seed=0,
                       fixed_rise=rise, kinds=kinds)
    obs, _ = env.reset(seed=seed)

    # Read the goal from the ENVIRONMENT rather than restating it. The
    # bar first showed "progress to x = 2.5 m" on a patches clip whose
    # field does not end until 6.3 m -- a caption that disagrees with
    # the success criterion it is drawn next to.
    from terrain import N_PATCHES, N_STONES, PATCH_LEN, PATCH_SPACING, STONE_TOP
    if kind == "patches":
        goal = EVENT_X + N_PATCHES * (PATCH_SPACING + PATCH_LEN)
    elif kind == "stones":
        goal = EVENT_X + N_STONES * (STONE_TOP + env.gap)
    else:
        goal = EVENT_X + 1.3

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
            # Framed for the BD-1 head, which stands ~0.9 m above the
            # torso origin. The earlier 3.2 m / -0.42 framing was set
            # for a head a third the height and cut the new one off at
            # the top of every frame.
            cam.lookat[:] = [info["x"] + 0.05, 0, env.torso_height() + 0.02]
            cam.distance, cam.elevation, cam.azimuth = 4.4, -7, 108
            r.update_scene(env.data, cam)
            alive = not info["fell"]
            frames.append(overlay(
                r.render(), title=title, subtitle=subtitle,
                stats=[("terrain", kind if rise is None
                                   else f"{kind}  rise {rise:.2f} m"),
                       *( [("surface", "SLIPPERY" if on_slip(env) else "grip")]
                          if kind == "patches" else [] ),
                       ("distance", f"{info['x']:+.2f} m"),
                       ("torso height", f"{env.torso_height():.2f} m"),
                       ("return", f"{total:.0f}")],
                status="CLEARED" if info["past_event"]
                       else ("WALKING" if alive else "FELL"),
                status_ok=alive,
                progress=(info["x"] + 1.0) / (goal + 1.0),
                bar_label=f"progress to x = {goal:.1f} m",
                depth=img if show_depth else None))
        if term or trunc:
            break

    return frames, {"past": bool(info["past_event"]), "x": float(far),
                    "ret": total, "steps": t}


# ------------------------------------------------------------------- clips
#
# One clip per scenario, and the comparison clip uses the RIGHT control.
#
# The first version of this lab animated a blind PPO policy against the
# vision student and let the viewer conclude the camera was the
# difference. It was not a controlled comparison -- those two arms
# differ in the camera AND in how they were trained. `compare` now runs
# the blind BEHAVIOUR-CLONED student, which differs from the vision
# student in exactly one thing.


def on_slip(env) -> bool:
    """Is a foot touching a low-friction patch RIGHT NOW?

    Read from the live contact list rather than from x-position, so the
    HUD cannot disagree with the physics -- which is precisely how the
    patch managed to be inert and look fine for three attempts.
    """
    import mujoco
    for c in range(env.data.ncon):
        for g in (env.data.contact[c].geom1, env.data.contact[c].geom2):
            n = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if n and n.startswith("slip"):
                return True
    return False


def _teacher(run: str):
    """A PPO policy, wrapped to the (obs, image) signature episode() uses."""
    act, meta = load_ppo(run)
    return (lambda obs, img: act(obs, img)), meta


def clip_patches(a) -> None:
    """
    Crossing ground with 18-25x less grip -- and the null beside it.

    Both arms converge to 100% here, so this pair is not a before/after.
    It is what a measured NULL looks like: the policy that was told
    where the slippery ground is, and the one that was not, doing the
    same thing.
    """
    frames = card(a.width, a.height, "Ground with 18–25x less grip",
                  "flat, so a depth camera cannot see it at all", 16)
    for run, label, blurb in (
            ("pp3_priv",  "TOLD where the ice is", "privileged — reaches 100% by 0.75M steps"),
            ("pp3_blind", "NOT told",              "blind — reaches 100% by 1.75M steps")):
        act, _ = _teacher(run)
        meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
        frames += card(a.width, a.height, label, blurb, 12)
        f, res = episode(act, "patches", seed=a.seed, title=label, subtitle=blurb,
                         width=a.width, height=a.height, every=a.every,
                         show_depth=False, needs_depth=False,
                         kinds=PATCH_KINDS, extra=(meta["obs_dim"] == 19))
        print(f"    {label:<24} x={res['x']:+.2f}  "
              f"{'crossed' if res['past'] else 'did NOT cross'}")
        frames += f
    _save(frames, OUT / "terrain-patches.gif", a)


def clip_stones(a) -> None:
    """Sparse footholds -- the terrain built to need vision, that did not."""
    act, _ = _teacher(a.stones_run)
    meta = json.loads((HERE / "runs" / a.stones_run / "summary.json").read_text())
    frames = card(a.width, a.height, "Stepping stones",
                  "nothing to feel between them — and blind solves it anyway", 14)
    f, res = episode(act, "stones", seed=a.seed, title="Stepping stones",
                     subtitle="privileged and blind both converge here",
                     width=a.width, height=a.height, every=a.every,
                     show_depth=False, needs_depth=False,
                     kinds=STONE_KINDS, extra=(meta["obs_dim"] == 21))
    print(f"    stones x={res['x']:+.2f}  "
          f"{'crossed' if res['past'] else 'did NOT cross'}")
    _save(frames + f, OUT / "terrain-stones.gif", a)


def _student(a, blind: bool):
    run = a.blind_student if blind else a.student
    act, use_depth = load_student(run)
    # Catch a mislabelled run rather than animating the wrong arm: a
    # "blind" clip driven by a camera network would look identical and
    # be a lie.
    assert use_depth is not blind, (
        f"{run} has use_depth={use_depth} but was asked for blind={blind}")
    return act, run


def clip_terrain(a, kind: str, blurb: str) -> None:
    """One terrain, the vision student, with the depth inset."""
    act, run = _student(a, blind=False)
    f, res = episode(act, kind, seed=a.seed,
                     title=f"Terrain: {kind}", subtitle=blurb,
                     width=a.width, height=a.height, every=a.every,
                     show_depth=True, needs_depth=True)
    print(f"    {kind:<5} reached x={res['x']:+.2f}  "
          f"{'cleared' if res['past'] else 'did NOT clear'}  ({run})")
    _save(f, OUT / f"terrain-{kind}.gif", a)


def clip_flat(a) -> None:
    clip_terrain(a, "flat", "solved without a camera too — this is the control")


def clip_up(a) -> None:
    clip_terrain(a, "up", "the ONLY terrain where the camera measurably helps")


def clip_down(a) -> None:
    clip_terrain(a, "down", "100% without a camera — the thesis was backwards here")


def clip_compare(a) -> None:
    """
    The controlled comparison: same teacher, same objective, one camera.

    Both arms here are behaviour-cloned students. The only difference is
    whether the network receives the depth image.
    """
    # A SELECTED episode, and the clip says so. On this blind
    # checkpoint -- seed 0, the strongest of the three at 71% -- the
    # blind student clears 12 of these 16 episode seeds. 8005 is one of
    # the four where it does not. Picking an episode that shows the
    # effect is fine; picking one and implying it is typical is not, so
    # the rates are on the title card and in the caption.
    frames = card(a.width, a.height, "Same teacher. Same training.",
                  "The only difference is the camera.", 16)
    frames += card(a.width, a.height, "One selected episode",
                   "across seeds: no camera 45.8%, with camera 88.9%", 14)
    for blind, label, blurb in (
            (True,  "NO camera", "blind student — clears 45.8% of episodes"),
            (False, "WITH camera", "vision student — clears 88.9%")):
        act, run = _student(a, blind=blind)
        frames += card(a.width, a.height, label, blurb, 12)
        f, res = episode(act, "up", seed=a.compare_seed, title=label,
                         subtitle=blurb,
                         width=a.width, height=a.height, every=a.every,
                         show_depth=not blind, needs_depth=not blind)
        print(f"    {label:<12} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}  ({run})")
        frames += f
    _save(frames, OUT / "terrain-compare.gif", a)


def clip_heldout(a) -> None:
    """
    A staircase steeper than any it trained on.

    Training drew the rise from [0.06, 0.11] m. This is 0.13 m, where
    the measured clear rate is 0%. The point of the clip is that the
    failure is not a stumble -- the gait simply does not reach.
    """
    act, run = _student(a, blind=False)
    frames = card(a.width, a.height, "Never trained on this",
                  "rise 0.13 m — outside the training range of 0.06–0.11", 16)
    f, res = episode(act, "up", seed=a.seed, rise=0.13,
                     title="Unseen geometry",
                     subtitle="rise 0.13 m — measured clear rate here is 0%",
                     width=a.width, height=a.height, every=a.every,
                     show_depth=True, needs_depth=True)
    print(f"    heldout 0.13 reached x={res['x']:+.2f}  "
          f"{'cleared' if res['past'] else 'did NOT clear'}  ({run})")
    _save(frames + f, OUT / "terrain-heldout.gif", a)


def clip_tour(a) -> None:
    """All three terrains, back to back, on one set of weights."""
    act, run = _student(a, blind=False)
    frames = card(a.width, a.height, "One policy, three terrains",
                  "it is told which only by its camera", 14)
    for kind, blurb in (("flat", "nothing to do but walk"),
                        ("up", "three treads up — lift before contact"),
                        ("down", "three treads down")):
        frames += card(a.width, a.height, kind.upper(), blurb, 10)
        f, res = episode(act, kind, seed=a.seed,
                         title=f"Terrain: {kind}",
                         subtitle="the same weights on all three",
                         width=a.width, height=a.height, every=a.every,
                         show_depth=True, needs_depth=True)
        print(f"    {kind:<5} reached x={res['x']:+.2f}  "
              f"{'cleared' if res['past'] else 'did NOT clear'}")
        frames += f
    _save(frames, OUT / "terrain-tour.gif", a)


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


CLIPS = {"patches": clip_patches, "stones": clip_stones,
         "flat": clip_flat, "up": clip_up, "down": clip_down,
         "compare": clip_compare, "heldout": clip_heldout, "tour": clip_tour}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=sorted(CLIPS), default="tour")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stones-run", default="stf_privileged_s2",
                    help="which trained stones policy to animate")
    ap.add_argument("--student", default="student_s2",
                    help="the vision arm to animate")
    ap.add_argument("--blind-student", default="student_blind",
                    help="the CONTROL arm: a behaviour-cloned student "
                         "with no camera. Not the blind PPO policy, "
                         "which differs in two things at once.")
    ap.add_argument("--compare-seed", type=int, default=8005,
                    help="episode seed for the comparison clip. Selected so "
                         "the measured difference is visible; the blind arm "
                         "clears 12 of 16 nearby seeds, so this is one of "
                         "the four it misses, and the clip labels it as "
                         "selected rather than typical.")
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
