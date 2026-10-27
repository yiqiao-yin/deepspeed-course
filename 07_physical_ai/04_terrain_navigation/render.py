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


def adopt_world(run: str) -> str:
    """
    Select the WORLD the run was trained in, before `world` is imported.

    `world.PRESET` is read from the environment at import time and
    `ARENA`/`N` are imported by value all over this lab, so the choice
    has to be made before the first import and cannot be changed after.

    Doing this automatically, rather than leaving it to whoever types
    the command, is the direct lesson of the `goal_range` bug: the
    renderer built its environment without the training task's
    parameters and filmed goals the policy had never been trained to
    reach, while the published table said something else entirely. The
    run records its world; nothing downstream should have to be told.
    """
    import json
    import os
    from pathlib import Path

    f = Path(__file__).parent / "runs" / run / "summary.json"
    want = "standard"
    if f.exists():
        want = json.loads(f.read_text()).get("world", "standard")
    have = os.environ.get("NAV_WORLD")
    if have and have != want:
        raise SystemExit(
            f"\n  {run} was trained in the '{want}' world but "
            f"NAV_WORLD={have} is set.\n"
            f"  Refusing to score or film a policy in a world it never "
            f"saw -- unset NAV_WORLD and let the run choose.\n")
    os.environ["NAV_WORLD"] = want
    return want


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


# Camera presets. `overhead` is orthographic so the arena reads as a
# map; the rest are perspective, because relief is what they exist to
# show and an orthographic 3/4 view flattens exactly that.
#
# All of them draw the route and track correctly, which is only true
# because those are scene GEOMETRY rather than pixels painted on. The
# hand projection this replaced was valid for one camera and one angle;
# every view below would have needed its own version of it, and each
# would have been wrong in its own way.
# WORLD-RELATIVE distances for the wide shots.
#
# These were absolute metres tuned for a 20 m arena. In the 40 m world
# an overhead span of 21 m frames a quarter of the map and crops the
# route out of its own clip -- the camera silently showing a different
# place than the one the robot is crossing. The tracking shots
# (`chase`, `shoulder`) stay absolute because they frame the ROBOT,
# whose size does not change with the arena.
# Stored as a FRACTION of the arena and resolved at use time, not at
# import time. `VIEWS` is built when this module loads, which happens
# before `main()` calls `adopt_world()` -- so reading `world.ARENA`
# here would bake in the default 20 m arena and film the big world
# through the small world's camera.
VIEWS = {
    "overhead": dict(ortho=True,  elev=-90.0, azim=90.0,  dist_frac=1.05,
                     track=False, label="overhead, orthographic"),
    "iso":      dict(ortho=False, elev=-42.0, azim=125.0, dist_frac=1.35,
                     track=False, label="isometric"),
    "chase":    dict(ortho=False, elev=-16.0, azim=None,  dist=5.5,
                     track=True,  label="chase"),
    "shoulder": dict(ortho=False, elev=-24.0, azim=None,  dist=3.2,
                     track=True,  label="over the shoulder"),
    "orbit":    dict(ortho=False, elev=-34.0, azim="spin", dist_frac=1.30,
                     track=False, label="orbit"),
}


def episode(run: str, seed: int, width: int, height: int, every: int,
            overhead: bool = True, view: str = "overhead",
            stop_after: int | None = None):
    """One episode, filmed, with the route and the track drawn on."""
    import mujoco
    import numpy as np

    from nav_env import NavWorld
    from world import ARENA, CELL, solve

    act, mode = load_policy(run)
    # The goal range the run was TRAINED on, read from its own summary.
    # Hardcoding it here -- or omitting it, as this did -- films a
    # different task than the one the published numbers describe.
    meta = json.loads((HERE / "runs" / run / "summary.json").read_text())
    env = NavWorld(mode=mode, flat=False, seed=0,
                   goal_range=meta.get("goal_range"))
    obs, _ = env.reset(seed=seed)
    path, route_len = solve(env.map)
    route = [((i + 0.5) * CELL - ARENA / 2, (j + 0.5) * CELL - ARENA / 2)
             for j, i in (path or [])]

    r = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    frames, track = [], []
    # Stop filming shortly after the robot gets there.
    #
    # The episode deliberately CONTINUES after arrival -- ending it
    # there once made success the worst outcome, since the policy
    # forfeited the remaining alive bonus and correctly learned not to
    # arrive. That is right for training and wrong for a clip: in the
    # 40 m world the robot reaches B at step 1512 of 6000 and then
    # jitters at the flag for 4,500 steps, so three quarters of the
    # animation is a stationary robot and "walked 70.4 m" is mostly
    # milling. Truncating the FILM changes nothing about the episode
    # or any number measured from it.
    after = None
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        track.append((info["x"], info["y"]))
        if info["arrived"] and after is None:
            after = 0
        elif after is not None:
            after += 1
            if stop_after is not None and after >= stop_after:
                term = True
        if len(track) % every == 0 or term:
            V = VIEWS[view]
            dist = view_dist(view)
            cam.orthographic = 1 if V["ortho"] else 0
            cam.distance = dist
            cam.elevation = V["elev"]
            if V["track"]:
                cam.lookat[:] = [info["x"], info["y"],
                                 env._h_at(info["x"], info["y"]) + 0.9]
                # Behind the robot, looking the way it faces -- which is
                # the whole point of a chase view and needs the live yaw
                # rather than a fixed angle.
                cam.azimuth = np.degrees(env.yaw()) + 180.0
            else:
                cam.lookat[:] = [0, 0, 0]
                cam.azimuth = (V["azim"] if V["azim"] != "spin"
                               else (len(track) * 0.35) % 360.0)
            r.update_scene(env.data, cam)
            # Markers scale with camera distance. A sphere sized for a
            # 21 m overhead shot is the size of the robot on a 5.5 m
            # chase cam, and swallows the thing it is annotating.
            k = dist / 21.0
            for x, y in route[::6]:
                add_marker(r.scene, (x, y, env._h_at(x, y) + 0.12),
                           (0.10, 0.95, 0.45, 1.0), 0.11 * k)
            for x, y in track[::4]:
                add_marker(r.scene, (x, y, env._h_at(x, y) + 0.30 * k),
                           (1.0, 0.42, 0.10, 1.0), 0.14 * k)
            sx, sy = env.map.start
            # Start marker A. Saturated on purpose: the GIF is
            # quantised to 64 colours, and the original (0.25,0.60,1.0)
            # came out the other side as a muddy (86,127,137) that read
            # as terrain rather than as a marker. Colour chosen to
            # survive quantisation and to sit far from BOTH the green
            # route and the orange track.
            add_marker(r.scene, (sx, sy, env._h_at(sx, sy) + 0.5),
                       (0.15, 0.80, 1.0, 1.0), 0.32 * k)
            frames.append(overlay(r.render(), env, info, route, track,
                                  mode=mode, overhead=overhead, env_=env,
                                  view=view))
        if term or trunc:
            break
    return frames, {"arrived": info["arrived"], "fell": info["fell"],
                    "travelled": info["travelled"], "route": route_len,
                    "to_goal": info["to_goal"], "mode": mode}


def view_dist(view: str) -> float:
    """Camera distance in metres: absolute for tracking shots, arena-relative for wide ones."""
    from world import ARENA

    V = VIEWS[view]
    return V["dist"] if "dist" in V else ARENA * V["dist_frac"]


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


def overlay(frame, env, info, route, track, *, mode, overhead,
            env_=None, view='overhead'):
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # The SAME definition the published table uses, or the clip and the
    # page contradict each other. Two things were wrong here and both
    # flattered the policy in one direction and punished it in the
    # other:
    #
    #   * the denominator was `travelled`, which keeps counting after
    #     the robot reaches B -- and the episode runs on for ~1300 more
    #     steps, so a 97%-optimal route read as 49%;
    #   * the numerator was `route`, the 4-connected BFS length, which
    #     cannot move diagonally and overstates the optimum by up to
    #     sqrt(2).
    #
    # Frozen at arrival, against the geodesic.
    den = info.get("travelled_to_goal") or info["travelled"]
    eff = (env.geodesic_len / den) if den > 0.1 else 0
    import numpy as _np

    # Heading, and how far off the goal bearing it is. A navigating
    # robot can be walking beautifully in the wrong direction, and the
    # distance-to-goal row alone does not show that.
    gx, gy = env.map.goal
    bearing = _np.arctan2(gy - env.pos()[1], gx - env.pos()[0])
    off = _np.degrees((bearing - env.yaw() + _np.pi) % (2 * _np.pi) - _np.pi)
    speed = float(_np.hypot(env.data.qvel[0], env.data.qvel[1]))

    stats = [("mode", mode),
             ("view", VIEWS[view]["label"]),
             ("step", f"{env.t} / {env.max_steps}"),
             ("to goal", f"{info['to_goal']:.1f} m"),
             ("heading error", f"{off:+.0f}\u00b0"),
             ("speed", f"{speed:.2f} m/s"),
             ("walked", f"{info['travelled']:.1f} m"),
             ("shortest path", f"{env.geodesic_len:.1f} m"),
             ("efficiency", f"{eff:.0%}" if info["arrived"] else "--"),
             ("clearance", f"{env.clearance():.2f} m")]
    pw, ph = 244, 30 + 20 * len(stats)
    d.rounded_rectangle([12, 12, 12 + pw, 12 + ph], 8, fill=PANEL,
                        outline=EDGE)
    y = 24
    for k, v in stats:
        d.text((26, y), k, font=_font(11), fill=MUTED)
        d.text((12 + pw - 16, y), str(v), font=_font(11, True), fill=INK,
               anchor="ra")
        y += 20

    # WHAT THE ROBOT KNOWS about the ground around it: the same three
    # probes the privileged policy receives, drawn as a little compass.
    # On the blind arm this panel is shown greyed, because the whole
    # point is that it does NOT have these numbers -- a HUD that looked
    # identical for both arms would quietly imply they see the same
    # thing.
    from world import STEP_MAX
    feats = env.terrain_feats() if env.mode == "privileged" else None
    bx, by, bw = 12, 12 + ph + 10, 244
    d.rounded_rectangle([bx, by, bx + bw, by + 96], 8, fill=PANEL,
                        outline=EDGE)
    d.text((bx + 14, by + 8),
           "ground ahead" if feats is not None else "ground ahead — NOT OBSERVED",
           font=_font(10, True), fill=INK if feats is not None else MUTED)
    for i, (lab, lobe) in enumerate((("left", 1), ("ahead", 0), ("right", 2))):
        cx = bx + 46 + i * 76
        if feats is None:
            d.rounded_rectangle([cx - 30, by + 30, cx + 30, by + 58], 5,
                                outline=(70, 86, 104, 255))
            d.text((cx, by + 44), "?", font=_font(13, True),
                   fill=(90, 106, 124), anchor="mm")
        else:
            rise, blocked = float(feats[lobe * 2]), bool(feats[lobe * 2 + 1])
            col = BAD if blocked else GOOD
            d.rounded_rectangle([cx - 30, by + 30, cx + 30, by + 58], 5,
                                fill=col + (55,), outline=col + (255,))
            d.text((cx, by + 44), f"{rise:.2f} m", font=_font(11, True),
                   fill=col, anchor="mm")
        d.text((cx, by + 68), lab, font=_font(10), fill=MUTED, anchor="mm")
    if feats is not None:
        # Below the row, not beside the title -- at 244 px wide the two
        # labels collided.
        d.text((bx + bw / 2, by + 80), f"red when the rise exceeds "
               f"{STEP_MAX:.2f} m", font=_font(9), fill=MUTED, anchor="mm")

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
    """
    Write the GIF, tagging the filename with the world it came from.

    Without this every big-world clip would land on the small world's
    filename and silently replace it: `nav-iso.gif` rendered at 40 m
    would overwrite the 20 m one the page already shows, and the page
    would keep its caption. Two different experiments cannot share an
    output path.
    """
    from PIL import Image

    from world import PRESET

    if PRESET != "standard":
        path = path.with_name(f"{path.stem}-{PRESET}{path.suffix}")
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
    f, res = episode(a.run, a.seed, a.width, a.height, a.every,
                     overhead=True, stop_after=a.stop_after)
    print(f"    {res['mode']:<11} walked {res['travelled']:.1f} m vs route "
          f"{res['route']:.1f} m  {'ARRIVED' if res['arrived'] else 'did not'}")
    _save(f, OUT / "nav-route.gif", a)


def clip_route_b(a) -> None:
    """
    A SECOND map, because one map is an anecdote.

    `nav-route-b.gif` was referenced by the tutorial page while nothing
    in this file produced it: it had been rendered once by hand and
    then orphaned. The whole lab was re-rendered after the
    coordinate-frame fix and that one asset silently survived from the
    broken world, still on the page beside eight corrected clips. An
    asset the page shows must be generated by the command the page
    documents, or it rots exactly this way.
    """
    f, res = episode(a.run, a.seed_b, a.width, a.height, a.every,
                     overhead=True, stop_after=a.stop_after)
    print(f"    second map  walked {res['travelled']:.1f} m vs route "
          f"{res['route']:.1f} m  {'ARRIVED' if res['arrived'] else 'did not'}")
    _save(f, OUT / "nav-route-b.gif", a)


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


def _angled(a, view: str, out: str) -> None:
    f, res = episode(a.run, a.seed, a.width, a.height, a.every,
                     overhead=(view == "overhead"), view=view,
                     stop_after=a.stop_after)
    print(f"    {VIEWS[view]['label']:<22} walked {res['travelled']:5.1f} m  "
          f"{'ARRIVED' if res['arrived'] else 'did not arrive'}")
    _save(f, OUT / out, a)


def clip_iso(a) -> None:
    """A 3/4 view: relief AND the route, which neither other view gives."""
    _angled(a, "iso", "nav-iso.gif")


def clip_chase(a) -> None:
    """Behind the robot, turning with it."""
    _angled(a, "chase", "nav-chase.gif")


def clip_shoulder(a) -> None:
    """Close in, where the ridges are at eye level."""
    _angled(a, "shoulder", "nav-shoulder.gif")


def clip_orbit(a) -> None:
    """A slow orbit: the arena as a three-dimensional place."""
    _angled(a, "orbit", "nav-orbit.gif")


CLIPS = {"route": clip_route, "route-b": clip_route_b,
         "ground": clip_ground,
         "compare": clip_compare, "fail": clip_fail,
         "iso": clip_iso, "chase": clip_chase,
         "shoulder": clip_shoulder, "orbit": clip_orbit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=sorted(CLIPS), default="route")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run", default="nav5_privileged_s5")
    ap.add_argument("--blind-run", default="nav5_blind_s5")
    # Both episode seeds were chosen by ROLLING OUT every seed in
    # 20000-20060 against the shipped checkpoint, not by guessing, and
    # they were re-chosen after the coordinate-frame fix because the old
    # defaults described episodes in a world where the robot spawned at
    # twice its start coordinates.
    #
    # 20002: the privileged policy arrives, 88% efficient, on a map
    # whose route is 1.52x the straight line -- so the detour is the
    # story rather than a decoration. The blind policy on the SAME map
    # finishes 5.8 m short, which is what makes the side-by-side worth
    # filming.
    ap.add_argument("--seed", type=int, default=20002)
    # 20005: stuck, not fallen (`fell=False`, 4.1 m short) on the most
    # extreme detour in the window, 1.82x. A clip of a robot falling
    # over teaches nothing about navigation; a clip of one that walks
    # competently into the wrong side of a wall teaches the lab.
    # 20053: a second arriving episode on a different map (route 1.49x
    # the straight line, 74% efficient). One map is an anecdote.
    ap.add_argument("--seed-b", type=int, default=20053)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop FILMING this many steps after arrival; the "
                         "episode itself is unchanged")
    ap.add_argument("--fail-seed", type=int, default=20005,
                    help="an episode the policy does NOT solve")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--frame-ms", type=int, default=55)
    ap.add_argument("--colors", type=int, default=64)
    a, _ = ap.parse_known_args()

    # Before anything imports `world`.
    adopt_world(a.run)
    os.environ["MUJOCO_GL"] = pick_backend()
    print(f"  writing to {OUT}")
    for name in (sorted(CLIPS) if a.all else [a.clip]):
        print(f"\n  [{name}]")
        CLIPS[name](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
