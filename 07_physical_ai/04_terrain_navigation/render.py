#!/usr/bin/env python3
"""
Animate the route, not just the robot.

    uv run render.py --all
    uv run render.py --clip route       # overhead, with the path drawn
    uv run render.py --clip compare     # blind vs privileged, same map

WHY OVERHEAD
------------
Labs 1-3 filmed from the side because the task was a profile along a
line: everything that mattered happened in the vertical plane. Here the
decision is WHERE TO GO, which is invisible from the side -- a robot
walking round a wall and a robot walking into it look identical from a
tracking side camera for the first two seconds.

So the main clips are overhead, with three things drawn on top of the
frame that are not in the simulation:

    the oracle's route       what the shortest traversable path was
    the robot's track        where it actually went
    the obstacle legend      which ridges were climbable

That overlay is the only way to see the thing being measured. It is
drawn from `world.solve()` and from the recorded positions, so it
cannot show a route the planner did not produce or a track the robot
did not walk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path
from pathlib import Path as pathlib_Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("glfw", "egl", "osmesa")
_FONTS: dict = {}

INK = (233, 240, 246)
MUTED = (141, 163, 181)
GOOD = (92, 196, 141)
BAD = (226, 110, 110)
BLUE = (99, 163, 208)
ORANGE = (227, 160, 90)
PANEL = (8, 24, 42, 205)
EDGE = (45, 90, 134, 255)


def pick_backend() -> str:
    """Probe backends in subprocesses; MuJoCo binds GL once per process."""
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
    print("\n  No working OpenGL backend — cannot render. Nothing else in")
    print("  this lab is affected; training and the checks never open a")
    print("  graphics context:\n")
    print("      uv run ../../tests/test_terrain_navigation.py\n")
    sys.exit(1)


def _font(size: int, bold: bool = False):
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


def load_policy(run: str):
    """A trained navigator, with the observation mode it was trained in."""
    import torch

    from ppo import ActorCritic, RunningNorm

    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    ck = torch.load(HERE / "runs" / run / "policy.pt", weights_only=False)
    net = ActorCritic(meta["obs_dim"], 7)
    net.load_state_dict(ck["model"])
    net.eval()
    norm = RunningNorm(meta["obs_dim"])
    norm.load_state_dict(ck["norm"])

    def act(obs):
        with torch.no_grad():
            t = torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).numpy()

    # The MODE has to come from the checkpoint. Driving a privileged
    # policy with a blind observation would run, produce a plausible
    # animation, and be a lie -- the same class of error as lab 3's
    # loader ignoring `use_depth`.
    return act, meta["mode"]


def episode(run: str, seed: int, width: int, height: int, every: int,
            overhead: bool = True):
    """One episode, filmed, with the route and the track drawn on."""
    import mujoco
    import numpy as np

    from nav_env import NavWorld
    from world import ARENA, CELL, solve

    act, mode = load_policy(run)
    env = NavWorld(mode=mode, flat=False, seed=0)
    obs, _ = env.reset(seed=seed)
    path, route_len = solve(env.map)
    route = [((i + 0.5) * CELL - ARENA / 2, (j + 0.5) * CELL - ARENA / 2)
             for j, i in (path or [])]

    r = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    frames, track = [], []
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        track.append((info["x"], info["y"]))
        if len(track) % every == 0 or term:
            if overhead:
                # ORTHOGRAPHIC, straight down. The overlay projects world
                # metres to pixels with a linear map, and under a
                # PERSPECTIVE camera that map is simply wrong -- every
                # route line, track and obstacle outline lands slightly
                # off the terrain it describes, which looks like a
                # rendering quirk and is actually the figure disagreeing
                # with the data. Orthographic makes the linear map exact.
                cam.orthographic = 1
                cam.lookat[:] = [0, 0, 0]
                cam.distance = ORTHO_SPAN
                cam.elevation, cam.azimuth = -90.0, 90.0
            else:
                cam.lookat[:] = [info["x"], info["y"], 1.0]
                cam.distance, cam.elevation, cam.azimuth = 5.0, -14, 120
            r.update_scene(env.data, cam)
            if overhead:
                # The oracle's route, then where the robot actually went,
                # then the goal -- all as scene geometry.
                for x, y in route[::6]:
                    add_marker(r.scene, (x, y, env._h_at(x, y) + 0.12),
                               (0.10, 0.95, 0.45, 1.0), 0.11)
                for x, y in track[::4]:
                    add_marker(r.scene, (x, y, env._h_at(x, y) + 0.30),
                               (1.0, 0.42, 0.10, 1.0), 0.14)
                sx, sy = env.map.start
                add_marker(r.scene, (sx, sy, env._h_at(sx, sy) + 0.5),
                           (0.25, 0.60, 1.0, 1.0), 0.32)
            frames.append(overlay(r.render(), env, info, route, track,
                                  mode=mode, overhead=overhead))
        if term or trunc:
            break
    return frames, {"arrived": info["arrived"], "fell": info["fell"],
                    "travelled": info["travelled"], "route": route_len,
                    "to_goal": info["to_goal"], "mode": mode}


ORTHO_SPAN = 21.0


def add_marker(scene, pos, rgba, size=0.14):
    """
    Put a sphere into the MuJoCo scene at a world position.

    The route, the track and the obstacle outlines are drawn as real
    GEOMETRY rather than painted onto the finished frame, and that is a
    correctness decision rather than an aesthetic one.

    Painting them required projecting world metres to pixels by hand,
    which means reimplementing MuJoCo's camera -- and the hand version
    was wrong by 21%: the rendered terrain spanned 591 px where the
    projection predicted 487. Every route line and obstacle outline sat
    a fifth of a frame away from the thing it described, which reads as
    a rendering quirk and is actually the figure disagreeing with the
    data.

    Rendered as geometry, the overlay goes through the same camera as
    the terrain. It cannot disagree, at any camera angle, ever.
    """
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), np.array(pos, dtype=float),
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    # EMISSIVE. A scene geom is lit like anything else, so under the
    # dim lighting this overhead view needs, a green route marker and
    # an orange track marker both render as the same washed yellow --
    # the two things the figure exists to distinguish. Emission makes
    # them show their own colour regardless of the lights.
    # 0.35, not 0.85. Emission adds WHITE, so a high value drove a
    # green marker and an orange one to the same pale yellow -- the two
    # series the figure exists to tell apart. Enough to survive dim
    # lighting, not enough to bleach the hue.
    g.emission = 0.35
    g.specular = 0.0
    g.shininess = 0.0
    scene.ngeom += 1


def overlay(frame, env, info, route, track, *, mode, overhead):
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    eff = (info["route"] / info["travelled"]) if info["travelled"] > 0.1 else 0
    stats = [("mode", mode),
             ("to goal", f"{info['to_goal']:.1f} m"),
             ("walked", f"{info['travelled']:.1f} m"),
             ("best route", f"{info['route']:.1f} m"),
             ("efficiency", f"{min(eff,1):.0%}")]
    pw, ph = 228, 30 + 20 * len(stats)
    d.rounded_rectangle([12, 12, 12 + pw, 12 + ph], 8, fill=PANEL,
                        outline=EDGE)
    y = 24
    for k, v in stats:
        d.text((26, y), k, font=_font(11), fill=MUTED)
        d.text((12 + pw - 16, y), str(v), font=_font(11, True), fill=INK,
               anchor="ra")
        y += 20

    if overhead:
        legend = (((26, 242, 115), "oracle's shortest route"),
                  ((255, 107, 26), "where the robot went"),
                  ((64, 153, 255), "start"))
        lh = 19 * len(legend) + 12
        d.rounded_rectangle([12, H - lh - 12, 232, H - 12], 8, fill=PANEL,
                            outline=EDGE)
        for i, (col, lab) in enumerate(legend):
            yy = H - lh - 2 + i * 19
            d.line([(24, yy + 6), (50, yy + 6)], fill=col, width=3)
            d.text((58, yy), lab, font=_font(11), fill=MUTED)

    chip = ("ARRIVED" if info["arrived"] else
            ("FELL" if info["fell"] else "WALKING"))
    col = GOOD if info["arrived"] else (BAD if info["fell"] else BLUE)
    tw = d.textlength(chip, font=_font(12, True))
    d.rounded_rectangle([W - tw - 40, H - 42, W - 16, H - 16], 6, fill=PANEL,
                        outline=col + (255,))
    d.text((W - 28 - tw / 2, H - 29), chip, font=_font(12, True), fill=col,
           anchor="mm")
    return np.asarray(img)


def card(W, H, title, body, n=12):
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (6, 14, 24))
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 14), title, font=_font(26, True), fill=INK,
           anchor="mm")
    d.text((W // 2, H // 2 + 20), body, font=_font(13), fill=MUTED,
           anchor="mm")
    return [np.asarray(img)] * n


def _save(frames, path, a):
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


def clip_route(a) -> None:
    """Overhead: the oracle's route, and the one the robot actually took."""
    f, res = episode(a.run, a.seed, a.width, a.height, a.every, overhead=True)
    print(f"    {res['mode']:<11} walked {res['travelled']:.1f} m vs route "
          f"{res['route']:.1f} m  {'ARRIVED' if res['arrived'] else 'did not'}")
    _save(f, OUT / "nav-route.gif", a)


def clip_ground(a) -> None:
    """The same run from behind the robot, where the terrain is legible."""
    f, res = episode(a.run, a.seed, a.width, a.height, a.every, overhead=False)
    print(f"    ground view: walked {res['travelled']:.1f} m")
    _save(f, OUT / "nav-ground.gif", a)


def clip_compare(a) -> None:
    """Blind against privileged on the SAME map — the controlled pair."""
    frames = card(a.width, a.height, "Same map. Same goal.",
                  "One of them can see the terrain.", 16)
    for run, label in ((a.blind_run, "BLIND — proprioception only"),
                       (a.run, "PRIVILEGED — sees the ground ahead")):
        frames += card(a.width, a.height, label.split(" — ")[0],
                       label.split(" — ")[1], 12)
        f, res = episode(run, a.seed, a.width, a.height, a.every,
                         overhead=True)
        print(f"    {label:<34} walked {res['travelled']:5.1f} m  "
              f"{'ARRIVED' if res['arrived'] else 'did not arrive'}")
        frames += f
    _save(frames, OUT / "nav-compare.gif", a)


def clip_fail(a) -> None:
    """
    An episode it does NOT solve.

    Arrival is ~12% even for the best arm, so a reel of successes would
    misrepresent the lab by a factor of eight. This is the common case.
    """
    f, res = episode(a.run, a.fail_seed, a.width, a.height, a.every,
                     overhead=True)
    print(f"    failure: walked {res['travelled']:.1f} m, "
          f"{res['to_goal']:.1f} m still to go, "
          f"{'fell' if res['fell'] else 'timed out'}")
    _save(f, OUT / "nav-fail.gif", a)


CLIPS = {"route": clip_route, "ground": clip_ground,
         "compare": clip_compare, "fail": clip_fail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=sorted(CLIPS), default="route")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run", default="nav_privileged_s0")
    ap.add_argument("--blind-run", default="nav_blind_s0")
    ap.add_argument("--seed", type=int, default=20021)
    ap.add_argument("--fail-seed", type=int, default=20001,
                    help="an episode the policy does NOT solve")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--frame-ms", type=int, default=55)
    ap.add_argument("--colors", type=int, default=64)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    print(f"  writing to {OUT}")
    for name in (sorted(CLIPS) if a.all else [a.clip]):
        print(f"\n  [{name}]")
        CLIPS[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
