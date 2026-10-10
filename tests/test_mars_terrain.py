#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""
A Mars-like surface must be Mars-like in ways that can FAIL.

`07_physical_ai/05_mars_terrain` generates terrain. Terrain is the
easiest thing in this repository to get wrong without noticing,
because the only check most people run is "does it look right", and a
height field almost always looks right.

So the properties asserted here are the ones a picture cannot show:

  1. THE SURFACE MUST BE THE SIZE IT CLAIMS. A channel carver that
     subtracted its depth at each of 150 overlapping steps produced
     46 m of relief inside a 60 m box -- a cliff-world -- and the
     ASCII map still looked like terrain.

  2. CRATERS MUST FOLLOW A POWER LAW. Uniformly sized craters are the
     single most recognisable way a synthetic planet looks fake, and
     the fitted exponent is the only way to know the sampler works.

  3. THE TWO AFFORDANCE CLASSES MUST DIFFER. A "cliff" the traversal
     rule treats as walkable is not a cliff. This is lab 4's lesson
     restated: assert the PROPERTY, not the parameter.

  4. THE ROVER SURFACE MUST BE CROSSABLE. A beautiful terrain a robot
     cannot walk on is a broken lab that looks like a hard one, and
     Part 2 is going to put a robot here.

  5. ROCKS MUST SIT ON THE GROUND. A floating rock is an invisible
     collision; a sunken one is a hole. Both look fine from above.

  6. A MEASUREMENT MUST NOT RETURN A PLAUSIBLE NUMBER WHEN IT HAS NO
     DATA. `climbing_pays` once returned 0.0 whenever every route in
     its counterfactual was unreachable, which reads exactly like
     "this terrain contains no decision" and meant "nothing was
     measured". It now reports its usable-sample count.

Shape assertions would pass on every one of these.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "07_physical_ai" / "05_mars_terrain"

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def test_scale_is_honest(mars) -> None:
    """Relief must match the preset, at both scales."""
    for name in ("rover", "regional"):
        P = mars.PRESETS[name]
        rel = [float(np.ptp(mars.generate(s, name).heights)) for s in range(6)]
        want = P["relief"]
        # Within a factor of two of the requested relief. Loose on
        # purpose -- craters and a volcano legitimately add to it --
        # but tight enough to catch an accumulating carve, which
        # overshot by 15x.
        ok = 0.4 * want <= float(np.mean(rel)) <= 2.0 * want
        check(f"relief matches the preset ({name})", ok,
              f"asked {want:,.0f} m, got {np.mean(rel):,.1f} m")

    # And the two scales must actually differ by the stated factor.
    r = mars.PRESETS["regional"]["extent"] / mars.PRESETS["rover"]["extent"]
    check("the two scales are far apart", r > 100,
          f"regional / rover = {r:,.0f}x")


def test_craters_follow_a_power_law(mars) -> None:
    """
    N(>D) ~ D**-b with b near 2, measured by fitting.

    And the counterexample is kept: a UNIFORM sample must be rejected
    by the same test, otherwise the test proves nothing.
    """
    bs = [mars.crater_power_law(mars.generate(s, "regional"))
          for s in range(5)]
    b = float(np.nanmean(bs))
    check("crater sizes follow a power law", 1.5 < b < 2.6,
          f"fitted exponent b = {b:.2f} (real populations ~2)")

    # The counterexample. Uniform radii must NOT pass.
    rng = np.random.default_rng(0)
    fake = mars.Surface(heights=np.zeros((8, 8)), scale="regional")
    fake.craters = [(0.0, 0.0, float(r), 0.0)
                    for r in rng.uniform(60, 1400, 240)]
    b_fake = mars.crater_power_law(fake)
    check("and a UNIFORM sample is rejected", not (1.5 < b_fake < 2.6),
          f"uniform radii fit b = {b_fake:.2f}, outside the accepted band")


def test_affordance_classes_differ(mars) -> None:
    """
    Smooth landforms must be climbable and cliffs must not.

    The first version of this measurement averaged traversability over
    each landform's bounding box and reported mesas as 86% walkable --
    true, and meaningless, because the box is mostly the flat top and
    the surrounding plain. What distinguishes the classes is the EDGE.
    """
    smooth, cliff = [], []
    for s in range(8):
        surf = mars.generate(s, "rover")
        c = mars.classes_differ(surf)
        for k, v in c.items():
            kind = k.replace("_edge_blocked", "")
            (smooth if kind in ("hill", "dune") else cliff).append(v)
    sm, cl = float(np.mean(smooth)), float(np.mean(cliff))
    check("smooth landforms are walkable at the edge", sm < 0.10,
          f"{sm:.1%} of the ring blocked")
    check("cliff-edged landforms are NOT", cl > 2.0 * sm,
          f"{cl:.1%} blocked, {cl / max(sm, 1e-9):.1f}x the smooth class")


def test_rover_surface_is_crossable(mars) -> None:
    """Part 2 puts a robot here. It has to be able to walk."""
    fr = [float(mars.traversable(mars.generate(s, "rover")).mean())
          for s in range(8)]
    check("most of the rover surface is walkable",
          float(np.mean(fr)) > 0.85, f"{np.mean(fr):.1%} of cells")
    check("but it is not a car park", float(np.mean(fr)) < 0.995,
          f"{1 - np.mean(fr):.1%} blocked — there are obstacles in it")


def test_rocks_sit_on_the_ground(mars) -> None:
    """Not hovering, not swallowed."""
    worst_f = worst_s = 0.0
    n = 0
    for s in range(6):
        surf = mars.generate(s, "rover")
        f, b = mars.rocks_sit_on_surface(surf)
        worst_f, worst_s, n = max(worst_f, f), max(worst_s, b), n + len(surf.rocks)
    check("no rock floats", worst_f < 1e-6, f"worst {worst_f:.2e} m over {n} rocks")
    check("no rock is swallowed", worst_s < 1e-6, f"worst {worst_s:.2e} m")

    # Rocks are SLABS, not spheres: thickness well under the long axis.
    surf = mars.generate(0, "rover")
    ratio = float(np.mean([k.rz / max(k.rx, 1e-9) for k in surf.rocks]))
    check("rocks are platy, not spherical", ratio < 0.55,
          f"mean thickness/length = {ratio:.2f} "
          f"(a sphere would be 1.00)")


def test_no_data_is_not_reported_as_zero(mars) -> None:
    """
    `climbing_pays` must distinguish "costs nothing" from "measured
    nothing".

    It did not. Blocking every cell above the median walled off the
    arena, every route came back infinite, every pair was dropped, and
    the function returned 0.0 -- which looked exactly like a terrain
    with no decision in it. Eight seeds in a row read +0.00 m and the
    real answer was that nothing had been measured at all.
    """
    out = [mars.climbing_pays(mars.generate(s, "rover")) for s in range(8)]
    check("climbing_pays reports its sample size",
          all("n_usable" in r for r in out))
    check("and returns NaN rather than 0.0 when it has none",
          all(math.isnan(r["median_gain"]) for r in out if r["n_usable"] == 0),
          "a silent zero is indistinguishable from a real null")
    usable = [r for r in out if r["n_usable"] > 0]
    check("most seeds yield a usable measurement",
          len(usable) >= 6, f"{len(usable)}/8 seeds")


def test_determinism(mars) -> None:
    """Same seed, same planet. Otherwise nothing above is reproducible."""
    a = mars.generate(5, "rover")
    b = mars.generate(5, "rover")
    check("the surface is deterministic in its seed",
          np.array_equal(a.heights, b.heights)
          and len(a.rocks) == len(b.rocks))
    c = mars.generate(6, "rover")
    check("and different seeds differ",
          not np.array_equal(a.heights, c.heights))


def main() -> int:
    sys.path.insert(0, str(LAB))
    import mars

    print("=" * 74)
    print("  A generated planet has to be checkable, not just plausible")
    print("=" * 74)
    test_scale_is_honest(mars)
    test_craters_follow_a_power_law(mars)
    test_affordance_classes_differ(mars)
    test_rover_surface_is_crossable(mars)
    test_rocks_sit_on_the_ground(mars)
    test_no_data_is_not_reported_as_zero(mars)
    test_determinism(mars)

    print()
    if _fails:
        print(f"  {len(_fails)} FAILED: {', '.join(_fails)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
