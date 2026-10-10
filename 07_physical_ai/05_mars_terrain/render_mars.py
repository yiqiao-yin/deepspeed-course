#!/usr/bin/env python3
"""
Animate the surface. No robot — Part 1 is terrain only.

    uv run render_mars.py --all
    uv run render_mars.py --clip rover-orbit

THE PALETTE IS A CLAIM, SO IT IS SOURCED
-----------------------------------------
Mars is red because the regolith is rich in iron oxide, and the sky is
**butterscotch, not blue**: fine suspended dust scatters forward and
reddens transmitted light, so surface imagery from Pathfinder onward
shows a salmon-tan sky. Rendering a blue sky over red ground is the
single most common way a "Mars" picture announces that nobody checked.

Rocks are rendered DARKER and GREYER than the soil. Martian float rock
is basaltic; the red is largely a thin dust coating, and freshly broken
or wind-scoured faces read grey-brown. A scene where the rocks are the
same colour as the dust looks like a sandcastle.

None of this is physically simulated — there is no scattering model
here. It is a palette chosen to match published surface imagery, and
it is a presentation choice rather than a result.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"
BACKENDS = ("egl", "osmesa", "glfw")

# Surface colours, sampled from published rover imagery rather than
# invented: dust is a light rusty orange, bedrock a darker red-brown,
# float rock grey-brown basalt.
SKY_HI = "0.62 0.44 0.32"          # butterscotch, darker aloft
SKY_LO = "0.86 0.67 0.49"          # brighter near the horizon
DUST = "0.76 0.44 0.26 1"
ROCK = "0.42 0.34 0.29 1"


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
    print("  this lab is affected; the generator and its checks never open")
    print("  a graphics context:\n")
    print("      uv run mars.py --check 40\n")
    sys.exit(1)


def hfield_png(heights: np.ndarray, path: Path) -> None:
    """
    Write the elevation array as the 8-bit PNG MuJoCo's hfield reads.

    Normalised to its own range, because MuJoCo scales the PNG by the
    `size` attribute in the XML. Writing absolute metres here and
    scaling there would encode the relief twice.
    """
    from PIL import Image

    h = heights.astype(np.float64)
    lo, hi = float(h.min()), float(h.max())
    img = np.zeros_like(h) if hi - lo < 1e-9 else (h - lo) / (hi - lo)
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)


def model_xml(surface, png_name: str, sun_az: float = 50.0,
              sun_el: float = 32.0) -> str:
    """One static scene: the height field, the rocks, a sun."""
    e = surface.extent
    relief = float(surface.heights.max() - surface.heights.min())
    relief = max(relief, 1e-3)
    az, el = math.radians(sun_az), math.radians(sun_el)
    sx = math.cos(el) * math.cos(az)
    sy = math.cos(el) * math.sin(az)
    sz = -math.sin(el)

    rocks = []
    for k in surface.rocks:
        # Rocks are BOXES, not spheres -- platy and angular, as rover
        # imagery shows. Tilted slightly because they rest on uneven
        # ground. See `mars.Rock`.
        cy_, sy_ = math.cos(k.yaw / 2), math.sin(k.yaw / 2)
        rocks.append(
            f'<geom type="box" size="{k.rx:.4f} {k.ry:.4f} {k.rz:.4f}" '
            f'pos="{k.x:.4f} {k.y:.4f} {k.z:.4f}" '
            f'euler="{k.roll:.4f} {k.pitch:.4f} {k.yaw:.4f}" '
            f'material="rock" contype="0" conaffinity="0"/>')
    rock_xml = "\n      ".join(rocks)

    return f"""
<mujoco model="mars_surface">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <!-- Dim ambient, strong directional: the Martian atmosphere is
         thin, so shadows are hard-edged compared with Earth's. -->
    <headlight ambient="0.26 0.21 0.18" diffuse="0.10 0.08 0.07"
               specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="{SKY_HI}" rgb2="{SKY_LO}" width="512" height="512"/>
    <hfield name="surface" file="{png_name}"
            size="{e/2} {e/2} {relief} {max(e*0.004, 0.05)}"/>
    <material name="dust" rgba="{DUST}" specular="0.02" shininess="0.02"/>
    <material name="rock" rgba="{ROCK}" specular="0.08" shininess="0.1"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="1.05 0.86 0.70"
           specular="0.05 0.04 0.03"
           pos="0 0 {e}" dir="{sx:.4f} {sy:.4f} {sz:.4f}" castshadow="true"/>
    <geom name="surface" type="hfield" hfield="surface" pos="0 0 0"
          material="dust" contype="0" conaffinity="0"/>
    <body name="rocks" pos="0 0 0">
      {rock_xml}
    </body>
  </worldbody>
</mujoco>
"""


def build(surface, tmp: Path, sun_az: float = 50.0, sun_el: float = 32.0):
    import mujoco

    hfield_png(surface.heights, tmp / "mars.png")
    return mujoco.MjModel.from_xml_string(
        model_xml(surface, "mars.png", sun_az, sun_el),
        {"mars.png": (tmp / "mars.png").read_bytes()})


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
def _frames(surface, cam_fn, n_frames: int, w: int, h: int,
            sun_fn=None) -> list:
    """Render n_frames, moving the camera (and optionally the sun)."""
    import mujoco
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    out = []
    model = build(surface, tmp)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if sun_fn is not None:
            # The sun is baked into the model, so a moving sun means
            # rebuilding it. Slow, and worth it for one clip.
            az, el = sun_fn(t)
            model = build(surface, tmp, az, el)
            data = mujoco.MjData(model)
            r = mujoco.Renderer(model, height=h, width=w)
        cam_fn(cam, t, surface)
        mujoco.mj_forward(model, data)
        r.update_scene(data, cam)
        out.append(r.render())
    return out


def _save(frames, path: Path, frame_ms: int, colors: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                      colors=colors,
                                      dither=Image.Dither.NONE)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KB)")


def clip_regional_orbit(a, mars):
    """The planetary view: volcano, canyon, fossae, cratered highlands."""
    s = mars.generate(a.seed, "regional")
    e = s.extent

    def cam(c, t, surf):
        c.distance = e * 1.15
        c.elevation = -38.0
        c.azimuth = 360.0 * t
        c.lookat[:] = [0, 0, 0]

    f = _frames(s, cam, a.frames, a.width, a.height)
    _save(f, OUT / "mars-regional-orbit.gif", a.frame_ms, a.colors)


def clip_regional_top(a, mars):
    """Straight down: the crater size distribution and the dichotomy."""
    s = mars.generate(a.seed, "regional")
    e = s.extent

    def cam(c, t, surf):
        c.orthographic = 1
        c.distance = e * 1.02
        c.elevation = -90.0
        c.azimuth = 90.0 + 8.0 * math.sin(2 * math.pi * t)
        c.lookat[:] = [0, 0, 0]

    f = _frames(s, cam, a.frames, a.width, a.height)
    _save(f, OUT / "mars-regional-top.gif", a.frame_ms, a.colors)


def clip_rover_orbit(a, mars):
    """The 60 m patch a robot would be dropped into."""
    s = mars.generate(a.seed, "rover")
    e = s.extent

    def cam(c, t, surf):
        c.distance = e * 0.95
        c.elevation = -32.0
        c.azimuth = 360.0 * t
        c.lookat[:] = [0, 0, 0]

    f = _frames(s, cam, a.frames, a.width, a.height)
    _save(f, OUT / "mars-rover-orbit.gif", a.frame_ms, a.colors)


def clip_rover_ground(a, mars):
    """
    Eye level, as a rover sees it. The one that shows the slabs.

    Camera height is 0.6 m -- roughly a small rover's mast -- because
    the whole point of this view is the horizon, and a camera at 5 m
    makes a boulder field look like gravel.
    """
    s = mars.generate(a.seed, "rover")
    e = s.extent

    def cam(c, t, surf):
        ang = 2 * math.pi * t
        rad = e * 0.33
        x, y = rad * math.cos(ang), rad * math.sin(ang)
        c.distance = 4.0
        c.elevation = -6.0
        c.azimuth = math.degrees(ang) + 180.0
        c.lookat[:] = [x, y, surf.height_at(x, y) + 0.6]

    f = _frames(s, cam, a.frames, a.width, a.height)
    _save(f, OUT / "mars-rover-ground.gif", a.frame_ms, a.colors)


def clip_sunrise(a, mars):
    """
    A sun sweep. Relief is invisible at noon and obvious at low sun.

    This is not decoration: shadow length is how a human reads terrain,
    and it is also what a vision-based policy in a later lab would key
    on. Watching the ripples and crater rims appear as the sun drops is
    the clearest demonstration that the surface has structure at all.
    """
    s = mars.generate(a.seed, "rover")
    e = s.extent

    def cam(c, t, surf):
        c.distance = e * 0.8
        c.elevation = -24.0
        c.azimuth = 35.0
        c.lookat[:] = [0, 0, 0]

    def sun(t):
        # 4 deg (dawn) up to 58 deg and back down.
        el = 4.0 + 54.0 * math.sin(math.pi * t)
        return 20.0 + 140.0 * t, el

    f = _frames(s, cam, max(a.frames // 2, 24), a.width, a.height, sun_fn=sun)
    _save(f, OUT / "mars-sunrise.gif", a.frame_ms, a.colors)


def clip_crater(a, mars):
    """Close on the largest crater: rim, bowl, ejecta."""
    s = mars.generate(a.seed, "rover")
    if not s.craters:
        print("  (no craters on this seed)")
        return
    cx, cy, r, *_ = max(s.craters, key=lambda c: c[2])

    def cam(c, t, surf):
        c.distance = r * 6.0
        c.elevation = -28.0
        c.azimuth = 360.0 * t
        c.lookat[:] = [cx, cy, surf.height_at(cx, cy)]

    f = _frames(s, cam, a.frames, a.width, a.height)
    _save(f, OUT / "mars-crater.gif", a.frame_ms, a.colors)


CLIPS = {
    "regional-orbit": clip_regional_orbit,
    "regional-top": clip_regional_top,
    "rover-orbit": clip_rover_orbit,
    "rover-ground": clip_rover_ground,
    "sunrise": clip_sunrise,
    "crater": clip_crater,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", choices=sorted(CLIPS), default="rover-orbit")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=500)
    ap.add_argument("--frame-ms", type=int, default=60)
    ap.add_argument("--colors", type=int, default=96)
    a, _ = ap.parse_known_args()

    os.environ["MUJOCO_GL"] = pick_backend()
    sys.path.insert(0, str(HERE))
    import mars

    print(f"  writing to {OUT}")
    for name in (sorted(CLIPS) if a.all else [a.clip]):
        print(f"\n  [{name}]")
        CLIPS[name](a, mars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
