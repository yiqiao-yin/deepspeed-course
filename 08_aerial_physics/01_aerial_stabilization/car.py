#!/usr/bin/env python3
"""
The vehicle: a lofted roadster body with a soft-coded thruster rig.

    uv run car.py                          # write the XML and compile it
    uv run car.py --rig wide --xml /tmp/car.xml
    uv run car.py --check                  # the names resolve, the rig matches

WHY MUJOCO AND NOT A WEB 3D ENGINE
-----------------------------------
The obvious way to make this look good is three.js. It is the wrong
tool HERE, for one reason: a renderer that is not the simulator can
show a car doing something the physics never did, and this course
has already paid for that mistake -- topic 7 published a result whose
faults were in its renderer rather than in anything a policy learned.

So the picture is drawn from the same model the policy is trained
against. Every geom below is either a collider or a visual hung on
the same body, and the GIFs are frames of the actual state. The
cosmetics budget goes into the MODEL instead: a 24-slice lofted hull,
separate glass, wheels with rims and brake discs, and a real nozzle
bell at every thruster.

THE PART THAT MATTERS FOR TRAINING
-----------------------------------
Each thruster is a `<site>` plus a `<general>` actuator with SITE
transmission, so `gear = (0, 0, F, 0, 0, 0)` applies F newtons along
the site's own z axis -- exactly a nozzle. The site's `zaxis` is the
thrust direction, which means the rig geometry from `thrusters.py` is
the only place a direction is written down.

Actuators are then resolved BY NAME. Topic 7 shipped `ctrl[6] = yaw`
three times, and each time the index silently addressed something
else after the body changed; the robot stood perfectly upright at a
0% fall rate and never moved. MuJoCo makes this trap easy to fall
into because actuators declared inside a body can come back UNNAMED
from `mj_id2name`, so this file names every actuator explicitly and
`resolve()` raises if any lookup misses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import thrusters as TH

# ---------------------------------------------------------------------------
# Body profile. Half-width, floor and beltline against x, in metres.
# Keys are interpolated to make the loft slices; the shape is a
# roadster's: narrow nose, shoulders over the rear axle, tucked tail.
# ---------------------------------------------------------------------------
W_KEYS = [(-2.17, 0.62), (-2.12, 0.86), (-1.90, 0.97), (-1.40, 1.045),
          (-0.95, 0.97), (-0.30, 0.905), (0.50, 0.915), (1.35, 1.00),
          (1.85, 0.94), (2.08, 0.82), (2.17, 0.58)]
FLOOR_KEYS = [(-2.17, -0.12), (-2.10, -0.22), (-1.95, -0.27), (-1.60, -0.30),
              (1.70, -0.30), (2.00, -0.27), (2.12, -0.22), (2.17, -0.15)]
BELT_KEYS = [(-2.17, 0.10), (-2.12, 0.22), (-1.85, 0.235), (-1.40, 0.25),
             (-1.00, 0.20), (-0.40, 0.14), (0.30, 0.10), (0.90, 0.07),
             (1.35, 0.075), (1.80, 0.015), (2.05, -0.06), (2.17, -0.14)]
CANOPY_KEYS = [(-1.70, 0.0), (-1.35, 0.085), (-0.95, 0.22), (-0.55, 0.33),
               (-0.15, 0.37), (0.15, 0.335), (0.45, 0.23), (0.75, 0.09),
               (0.98, 0.0)]

N_SLICES = 24
WHEEL_X, WHEEL_Y = 1.35, 0.85
WHEEL_R, WHEEL_W = 0.35, 0.16
RIDE_Z = -0.20                      # wheel centre height in body frame


def _interp(keys, x: float) -> float:
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    return float(np.interp(x, xs, ys))


def _materials() -> str:
    """
    Physically-ish shaded materials. MuJoCo's renderer supports
    specular, shininess and reflectance, which is enough for the car
    to read as a car rather than as a pile of primitives.
    """
    return """
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.52 0.36 0.28" rgb2="0.06 0.07 0.10" width="256" height="256"/>
    <!-- Regolith, not a checkerboard. `builtin="checker"` at
         texrepeat 48 tiled the whole arena in squares and the Mars
         surface read as a bathroom floor. `flat` + random marks is
         speckle, which is what dust actually looks like, and it has
         no grid for the eye to lock onto. -->
    <texture name="regolith" type="2d" builtin="flat" width="512" height="512"
             rgb1="0.42 0.27 0.20" rgb2="0.42 0.27 0.20"
             mark="random" markrgb="0.54 0.38 0.29" random="0.28"/>
    <material name="ground" texture="regolith" texrepeat="14 14"
              specular="0.05" shininess="0.03" reflectance="0.0"/>
    <material name="rock"   rgba="0.33 0.22 0.17 1" specular="0.12" shininess="0.1"/>
    <material name="paint"  rgba="0.868 0.886 0.902 1" specular="0.85"
              shininess="0.92" reflectance="0.22"/>
    <material name="trim"   rgba="0.071 0.078 0.094 1" specular="0.35" shininess="0.4"/>
    <material name="glass"  rgba="0.165 0.231 0.310 0.80" specular="1.0"
              shininess="0.98" reflectance="0.40"/>
    <material name="roof"   rgba="0.090 0.102 0.125 1" specular="0.9"
              shininess="0.95" reflectance="0.30"/>
    <material name="tyre"   rgba="0.078 0.082 0.086 1" specular="0.05" shininess="0.05"/>
    <material name="rim"    rgba="0.831 0.850 0.871 1" specular="0.8" shininess="0.75"/>
    <material name="disc"   rgba="0.353 0.373 0.392 1" specular="0.6" shininess="0.5"/>
    <material name="brake"  rgba="0.878 0.509 0.102 1" specular="0.4" shininess="0.4"/>
    <material name="bell"   rgba="0.706 0.737 0.769 1" specular="0.9" shininess="0.85"/>
    <material name="throat" rgba="0.055 0.059 0.067 1" specular="0.2" shininess="0.2"/>
    <material name="lamp"   rgba="0.62 0.84 1.0 1" emission="0.9"/>
    <material name="tail"   rgba="0.90 0.09 0.07 1" emission="0.75"/>
"""


def _hull_visual() -> str:
    """
    The loft. `N_SLICES` thin boxes along x, each sized to the width
    and height profile at its station, plus a canopy of ellipsoids.

    Cosine spacing puts more slices at the nose and tail, where the
    section changes fastest -- uniform spacing makes the nose look
    chopped and wastes slices in the flat middle.
    """
    out = []
    x0, x1 = -TH.HALF_L, TH.HALF_L
    for i in range(N_SLICES):
        f = 0.5 * (1 - math.cos(math.pi * i / (N_SLICES - 1)))
        xa = x0 + (x1 - x0) * f
        fb = 0.5 * (1 - math.cos(math.pi * (i + 1) / (N_SLICES - 1)))
        xb = x0 + (x1 - x0) * min(fb, 1.0)
        if xb - xa < 1e-4:
            continue
        xm = 0.5 * (xa + xb)
        hw = _interp(W_KEYS, xm)
        zf = _interp(FLOOR_KEYS, xm)
        zb = _interp(BELT_KEYS, xm)
        hz = max(0.5 * (zb - zf), 0.01)
        out.append(
            f'      <geom type="box" material="paint" contype="0" conaffinity="0" '
            f'group="1" size="{0.5 * (xb - xa) + 0.013:.4f} {hw:.4f} {hz:.4f}" '
            f'pos="{xm:.4f} 0 {0.5 * (zf + zb):.4f}"/>')
        # Sill: a dark band along the bottom edge, which is what makes
        # the body read as having a floor rather than floating.
        out.append(
            f'      <geom type="box" material="trim" contype="0" conaffinity="0" '
            f'group="1" size="{0.5 * (xb - xa) + 0.013:.4f} {hw * 0.985:.4f} 0.016" '
            f'pos="{xm:.4f} 0 {zf + 0.016:.4f}"/>')

    # Canopy: BOX slices, not ellipsoids.
    #
    # Two attempts at this rendered as a caterpillar. Ellipsoids
    # waist between their centres -- each is widest at its own
    # midpoint, so however much they overlap you still see a scallop
    # at every join, and fifteen of them in a row read as a string of
    # dark beads sitting on the deck. Boxes tile exactly, and the
    # hull below is boxes too, so the greenhouse now matches the body
    # it sits on instead of fighting it.
    n_can, x_a, x_b = 18, -1.74, 1.02
    for i in range(n_can):
        xa = x_a + (x_b - x_a) * i / n_can
        xb = x_a + (x_b - x_a) * (i + 1) / n_can
        xm = 0.5 * (xa + xb)
        ch = _interp(CANOPY_KEYS, xm)
        if ch <= 0.008:
            continue
        cw = float(np.interp(xm, [-1.74, -0.95, -0.20, 0.45, 1.02],
                             [0.80, 0.66, 0.60, 0.64, 0.80]))
        zb = _interp(BELT_KEYS, xm)
        mat = "roof" if (-1.00 <= xm <= 0.02) else "glass"
        out.append(
            f'      <geom type="box" material="{mat}" contype="0" '
            f'conaffinity="0" group="1" '
            f'size="{0.5 * (xb - xa) + 0.010:.4f} {cw:.4f} {ch * 0.5:.4f}" '
            f'pos="{xm:.4f} 0 {zb + ch * 0.5:.4f}"/>')
        # Shoulder fillet: a narrower cap on top of each slice, so the
        # greenhouse tapers instead of ending in a slab edge.
        out.append(
            f'      <geom type="box" material="{mat}" contype="0" '
            f'conaffinity="0" group="1" '
            f'size="{0.5 * (xb - xa) + 0.010:.4f} {cw * 0.80:.4f} '
            f'{ch * 0.16:.4f}" pos="{xm:.4f} 0 {zb + ch * 1.06:.4f}"/>')
    # A-pillar, so the glass has somewhere to end.
    out.append('      <geom type="box" material="trim" contype="0" '
               'conaffinity="0" group="1" size="0.022 0.61 0.014" '
               'pos="0.99 0 0.17" euler="0 -28 0"/>')

    # Lights, splitter, diffuser strakes.
    out.append('      <geom type="box" material="lamp" contype="0" '
               'conaffinity="0" group="1" size="0.03 0.40 0.035" '
               'pos="2.10 0 0.02"/>')
    out.append('      <geom type="box" material="tail" contype="0" '
               'conaffinity="0" group="1" size="0.025 0.80 0.035" '
               'pos="-2.14 0 0.10"/>')
    out.append('      <geom type="box" material="trim" contype="0" '
               'conaffinity="0" group="1" size="0.13 0.92 0.016" '
               'pos="2.02 0 -0.29"/>')
    for k in range(5):
        y = -0.56 + 1.12 * k / 4
        out.append(f'      <geom type="box" material="trim" contype="0" '
                   f'conaffinity="0" group="1" size="0.20 0.022 0.075" '
                   f'pos="-1.92 {y:.3f} -0.235"/>')
    return "\n".join(out)


def _wheel(name: str, x: float, y: float) -> str:
    """A wheel body: tyre, rim barrel, dish, brake disc and caliper.

    The hinge is about body y, so the wheel spins on touchdown and
    the GIF shows it. The caliper is on the PARENT body on purpose --
    a caliper that rotates with the wheel is the classic tell that a
    model was assembled without looking at it.
    """
    return f"""
      <body name="{name}" pos="{x:.4f} {y:.4f} {RIDE_Z:.4f}">
        <joint name="{name}_spin" type="hinge" axis="0 1 0" damping="0.6"/>
        <geom name="{name}_tyre" type="cylinder" material="tyre"
              size="{WHEEL_R:.4f} {WHEEL_W / 2:.4f}" quat="0.7071068 0.7071068 0 0"
              condim="3" friction="1.1 0.02 0.001" mass="22"/>
        <geom type="cylinder" material="rim" contype="0" conaffinity="0" group="1"
              size="{WHEEL_R * 0.66:.4f} {WHEEL_W / 2 + 0.002:.4f}" mass="0"
              quat="0.7071068 0.7071068 0 0"/>
        <geom type="cylinder" material="trim" contype="0" conaffinity="0" group="1"
              size="{WHEEL_R * 0.60:.4f} {WHEEL_W / 2 + 0.006:.4f}" mass="0"
              quat="0.7071068 0.7071068 0 0"/>
        <geom type="cylinder" material="disc" contype="0" conaffinity="0" group="1"
              size="{WHEEL_R * 0.52:.4f} {WHEEL_W / 2 + 0.010:.4f}" mass="0"
              quat="0.7071068 0.7071068 0 0"/>
        {_spokes()}
      </body>
      <geom type="box" material="brake" contype="0" conaffinity="0" group="1"
            size="0.055 0.030 0.085"
            pos="{x - 0.19:.4f} {y:.4f} {RIDE_Z + 0.19:.4f}"/>"""


def _spokes(n: int = 10) -> str:
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(
            f'        <geom type="box" material="rim" contype="0" '
            f'conaffinity="0" group="1" size="0.022 {WHEEL_W / 2 + 0.012:.4f} '
            f'{WHEEL_R * 0.5:.4f}" mass="0" pos="0 0 0" '
            f'euler="0 {math.degrees(a):.2f} 0"/>')
    return "\n".join(out)


def _nozzle(t: TH.Thruster, fmax_ref: float) -> str:
    """
    A bell, a throat plug and the site the actuator pushes on.

    Bell size scales with the square root of thrust, because throat
    area does -- a 5 kN nozzle next to a 20 kN nozzle should look
    like one, and a reader who can see the difference can see when a
    rig is lopsided.
    """
    d = np.asarray(t.direction)
    k = math.sqrt(max(t.fmax, 1.0) / max(fmax_ref, 1.0))
    exit_r = 0.098 * k
    length = 0.085 * k
    # The bell sits just outside the skin, pointing the way the gas
    # leaves (i.e. opposite the force).
    base = np.asarray(t.pos)
    mid = base - d * (length * 0.5)
    zax = f'{-d[0]:.6f} {-d[1]:.6f} {-d[2]:.6f}'
    return f"""
      <geom type="cylinder" material="bell" contype="0" conaffinity="0" group="1"
            size="{exit_r:.4f} {length / 2:.4f}"
            pos="{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f}" zaxis="{zax}"/>
      <geom type="cylinder" material="throat" contype="0" conaffinity="0" group="1"
            size="{exit_r * 0.45:.4f} {length * 0.30:.4f}"
            pos="{base[0]:.4f} {base[1]:.4f} {base[2]:.4f}" zaxis="{zax}"/>
      <site name="{t.name}" pos="{base[0]:.4f} {base[1]:.4f} {base[2]:.4f}"
            zaxis="{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}" size="0.012" group="3"/>"""


def model_xml(rig_: list[TH.Thruster],
              hfield_png: str | None = None,
              relief: float = 1.0,
              extent: float = 64.0,
              rocks=(),
              start_z: float = 1.0,
              integrator: str = "implicitfast") -> str:
    """
    One MuJoCo model: ground, the car, and one actuator per thruster.

    `integrator` defaults to `implicitfast` rather than `RK4`. Topic 7
    measured that swap at 3.9x on a contact-rich body, which beat what
    eight rented GPUs would have bought -- and it is the documented
    recommendation for contact-rich models, with RK4 reserved for
    energy-conserving systems. The default is written here once so no
    caller has to remember it.
    """
    fmax_ref = max(t.fmax for t in rig_)
    noz = "\n".join(_nozzle(t, fmax_ref) for t in rig_)
    wheels = "\n".join(
        _wheel(f"wh{sx}{sy}", x * TH.HALF_L / TH.HALF_L, y)
        for sx, x in (("F", WHEEL_X), ("R", -WHEEL_X))
        for sy, y in (("L", WHEEL_Y), ("R", -WHEEL_Y)))

    if hfield_png:
        ground = (f'<geom name="ground" type="hfield" hfield="terrain" '
                  f'material="ground" condim="3" friction="1.0 0.02 0.001"/>')
        assets = (f'<hfield name="terrain" file="{hfield_png}" '
                  f'size="{extent / 2:.3f} {extent / 2:.3f} '
                  f'{max(relief, 1e-3):.4f} 0.5"/>')
    else:
        ground = ('<geom name="ground" type="plane" size="60 60 0.5" '
                  'material="ground" condim="3" friction="1.0 0.02 0.001"/>')
        assets = ""

    rock_xml = "\n      ".join(
        f'<geom type="box" material="rock" size="{r.rx:.3f} {r.ry:.3f} {r.rz:.3f}" '
        f'pos="{r.x:.3f} {r.y:.3f} {r.rz * 0.6:.3f}" '
        f'euler="0 0 {math.degrees(r.yaw):.2f}" condim="3"/>'
        for r in rocks)

    # One actuator per nozzle, NAMED, gear along the site's own z.
    acts = "\n".join(
        f'    <general name="{t.name}" site="{t.name}" '
        f'gear="0 0 {t.fmax:.4f} 0 0 0" ctrllimited="true" ctrlrange="0 1"/>'
        for t in rig_)

    return f"""<mujoco model="aerial_roadster">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="{integrator}" gravity="0 0 -9.80665"
          density="0.02" viscosity="0"/>
  <!-- density 0.02 kg/m^3: thin Martian air. It is not zero, so the
       car is not in vacuum, but it is 1/60th of Earth's and nothing
       aerodynamic saves a bad landing. -->
  <visual>
    <headlight ambient="0.42 0.40 0.39" diffuse="0.62 0.58 0.55"
               specular="0.35 0.35 0.35"/>
    <quality shadowsize="4096" offsamples="8"/>
    <!-- The offscreen framebuffer defaults to 640x480 and the renderer
         REFUSES a larger image rather than quietly downscaling, which
         is the right behaviour and is why this clause has to be here
         at all. The flight GIFs are 1280-wide. -->
    <global offwidth="1600" offheight="1000"/>
    <map znear="0.05" zfar="60" shadowclip="12"/>
  </visual>

  <asset>{_materials()}    {assets}
  </asset>

  <worldbody>
    <light name="sun" directional="true" diffuse="0.85 0.80 0.74"
           specular="0.30 0.30 0.30" pos="12 -16 24" dir="-0.4 0.55 -0.78"
           castshadow="true"/>
    <light name="fill" directional="true" diffuse="0.22 0.24 0.30"
           pos="-14 12 16" dir="0.45 -0.4 -0.8" castshadow="false"/>
    {ground}
      {rock_xml}

    <body name="chassis" pos="0 0 {start_z:.4f}">
      <freejoint name="root"/>
      <!-- Mass and inertia are SET, not inferred from the visual
           geoms: the loft is cosmetic and its slices would otherwise
           invent an inertia nobody chose. -->
      <inertial pos="0 0 0" mass="{TH.M_DRY - 4 * 22:.2f}"
                diaginertia="{TH.I_DRY[0]:.1f} {TH.I_DRY[1]:.1f} {TH.I_DRY[2]:.1f}"/>
      <!-- Colliders: two boxes, deliberately coarse. A 24-slice
           collision hull would cost a fortune per step and change no
           outcome -- what matters is whether the car lands on its
           wheels or on its roof. -->
      <geom name="hull_low" type="box" size="2.05 0.94 0.17" pos="0 0 -0.12"
            rgba="1 0 0 0" condim="3" friction="0.7 0.02 0.001" mass="0"/>
      <geom name="hull_top" type="box" size="1.30 0.76 0.22" pos="-0.35 0 0.22"
            rgba="1 0 0 0" condim="3" friction="0.7 0.02 0.001" mass="0"/>
{_hull_visual()}
{noz}
{wheels}
      <site name="cg" pos="0 0 0" size="0.03" group="3"/>
    </body>
  </worldbody>

  <actuator>
{acts}
  </actuator>
</mujoco>
"""


def build(rig_name: str = "default", preset: str = "rolling", seed: int = 0,
          flat: bool = False, start_z: float = 1.0,
          integrator: str = "implicitfast", tmpdir=None, **rig_kw):
    """Compile a model. Returns `(model, rig, surface)`."""
    import tempfile

    import mujoco

    import terrain

    r = TH.rig(rig_name, **rig_kw)
    if flat:
        xml = model_xml(r, None, start_z=start_z, integrator=integrator)
        return mujoco.MjModel.from_xml_string(xml), r, None

    s = terrain.generate(seed, preset)
    tmp = Path(tmpdir or tempfile.mkdtemp())
    png = tmp / "terrain.png"
    terrain.hfield_png(s.heights, png)
    xml = model_xml(r, "terrain.png", relief=s.relief, extent=s.extent,
                    rocks=s.rocks, start_z=start_z, integrator=integrator)
    m = mujoco.MjModel.from_xml_string(xml, {"terrain.png": png.read_bytes()})
    return m, r, s


def resolve(model, rig_: list[TH.Thruster]) -> np.ndarray:
    """
    Actuator ids for the rig's thrusters, BY NAME, in rig order.

    Raises if a name misses. That is the entire point: a name filter
    that silently returns an empty list makes `ctrl[[]] = []` a no-op,
    which is how topic 7 trained an uncontrollable robot three times
    and blamed the observation width.
    """
    import mujoco

    ids = []
    for t in rig_:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, t.name)
        if i < 0:
            raise KeyError(f"actuator {t.name!r} not found in the model — "
                           f"the rig and the XML have diverged")
        ids.append(i)
    out = np.asarray(ids, dtype=np.int32)
    if len(set(ids)) != len(ids):
        raise ValueError("two thrusters resolved to the same actuator")
    if model.nu != len(rig_):
        raise ValueError(f"model has {model.nu} actuators for "
                         f"{len(rig_)} thrusters")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="default", choices=sorted(TH.RIGS))
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--xml", default="")
    ap.add_argument("--check", action="store_true")
    a, _ = ap.parse_known_args()

    import mujoco

    m, r, s = build(a.rig, a.preset, flat=a.flat)
    ids = resolve(m, r)

    print("=" * 74)
    print(f"  MODEL — rig {a.rig}, terrain {'flat' if a.flat else a.preset}")
    print("=" * 74)
    print(f"  bodies {m.nbody}   geoms {m.ngeom}   actuators {m.nu}   "
          f"dof {m.nv}")
    print(f"  mass   {m.body_mass.sum():.1f} kg "
          f"(target {TH.M_DRY + TH.M_GAS_0:.0f} with gas)")
    print(f"  names  resolved {len(ids)}/{len(r)} by name, ids {ids.tolist()}")

    # The check that actually bites: push one nozzle and confirm the
    # wrench MuJoCo produces equals the column `thrusters.py` claims.
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    B = TH.effectiveness(r)
    worst, worst_name = 0.0, ""
    for k, t in enumerate(r):
        d.ctrl[:] = 0
        d.ctrl[ids[k]] = 1.0
        mujoco.mj_forward(m, d)
        f = np.zeros(6)
        mujoco.mj_objectVelocity  # noqa: B018  (keep the import honest)
        # qfrc_actuator is in generalised coordinates; for a free
        # joint the first six are force and torque about the body
        # frame origin, which is the CG here.
        f[:] = d.qfrc_actuator[:6]
        # MuJoCo reports the free-joint force in WORLD; at identity
        # attitude that is the body frame too.
        err = float(np.abs(f - B[:, k]).max())
        if err > worst:
            worst, worst_name = err, t.name
    scale = float(np.abs(B).max())
    print(f"  wrench max |MuJoCo - thrusters.py| = {worst:.4f} N "
          f"on {worst_name}, against a {scale:,.0f} N scale "
          f"({worst / scale:.2e} relative)")

    if a.xml:
        Path(a.xml).write_text(model_xml(r, None))
        print(f"\n  wrote {a.xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
