#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0"]
# ///
"""
Lab 9 — a body is parameters, and the generator must prove it.

`make_layout` claims to generalise the five hand-written bodies in
`LEG_LAYOUTS`. That claim is cheap to make and easy to get subtly
wrong: a generator that produces the right NUMBER of legs in roughly
the right places looks correct and quietly drops features.

It already did. The first version named a two-pair body's rows
"front" and "mid" instead of front and hind, and it dropped the
hexapod's widened middle pair — a real feature (insects splay
mid-legs so the swing phase misses the front and hind legs), not a
typo in the table.

So the test is exact reproduction: every named layout must come back
from parameters, leg for leg, to four decimals. Anything less and
the "generalisation" is a second implementation that happens to
agree on the easy cases.

No GPU, no training, no downloads.

    uv run tests/test_bodyspec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAB5 = REPO / "07_physical_ai" / "05_mars_terrain"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def norm(legs):
    return [tuple(round(v, 4) if isinstance(v, float) else v for v in t)
            for t in legs]


def main() -> int:
    sys.path.insert(0, str(LAB5))
    import mujoco

    import robot

    print("=" * 74)
    print("  LAB 9 — a body is parameters")
    print("=" * 74)

    # ---------------------------------------------------------------
    # Exact reproduction of every hand-written body.
    # ---------------------------------------------------------------
    cases = {
        "biped": dict(n_legs=2, posture="mammal", leg_scale=1.0,
                      spread=0.0, width=0.07),
        "quadruped": dict(n_legs=4, posture="mammal", leg_scale=0.58,
                          spread=0.26, width=0.17),
        "hexapod": dict(n_legs=6, posture="mammal", leg_scale=0.52,
                        spread=0.30, width=0.18, mid_widen=0.03),
        "spider6": dict(n_legs=6, posture="sprawl"),
        "spider8": dict(n_legs=8, posture="sprawl"),
    }
    for name, kw in cases.items():
        got = robot.make_layout(**kw)
        want = robot.LEG_LAYOUTS[name]
        check(f"{name} reproduces from parameters",
              norm(got["legs"]) == norm(want["legs"]),
              f"{len(got['legs'])} legs")

    # ---------------------------------------------------------------
    # Bodies nobody typed out must build and be physically sane.
    # ---------------------------------------------------------------
    import numpy as np
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    robot.hfield_png(np.zeros((64, 64)), tmp / "t.png")
    png = {"t.png": (tmp / "t.png").read_bytes()}

    novel = [(8, "mammal"), (10, "mammal"), (3, "sprawl"), (5, "sprawl"),
             (12, "sprawl")]
    for n, posture in novel:
        layout = robot.make_layout(n, posture)
        xml = robot.model_xml("t.png", (0, 0), (1, 1), relief=1.0,
                              rocks=[], layout=layout)
        m = mujoco.MjModel.from_xml_string(xml, png)
        # Three joints per sprawl leg, four per mammal leg, plus yaw.
        per = 3 if posture == "sprawl" else 4
        check(f"{posture} with {n} legs builds and actuates",
              m.nu == n * per + 1, f"nu={m.nu}, expected {n*per+1}")

    # ---------------------------------------------------------------
    # It must REFUSE nonsense loudly rather than build a lopsided
    # robot. A generator that silently accepts 3 mammal legs produces
    # a body with two on one side and one on the other, which would
    # train, walk badly, and be scored as a finding.
    # ---------------------------------------------------------------
    for kw, why in (
            (dict(n_legs=1), "fewer than two legs"),
            (dict(n_legs=3, posture="mammal"), "odd count for a mammal"),
            (dict(n_legs=4, posture="banana"), "unknown posture")):
        try:
            robot.make_layout(**kw)
            check(f"refuses {why}", False, "it was accepted")
        except (ValueError, KeyError):
            check(f"refuses {why}", True)

    # ---------------------------------------------------------------
    # The mechanism behind "more legs is more stable", stated
    # correctly.
    #
    # It is NOT a wider base. `spread` is a parameter, so extra legs
    # add rows BETWEEN the extremes and the fore-aft extent is
    # identical at 4, 6 and 8 legs (0.52 m each). The first version
    # of this check asserted the base "does not shrink" and passed on
    # three equal numbers -- true, unfalsifiable, and testing the
    # wrong thing.
    #
    # What actually grows is SUPPORT POINTS: how many feet remain on
    # the ground while one swings. That is what lets a hexapod hold a
    # stable tripod and a biped hold nothing.
    # ---------------------------------------------------------------
    extents, supports = [], []
    for n in (4, 6, 8):
        legs = robot.make_layout(n, "mammal", spread=0.26, width=0.17)["legs"]
        xs = [x for _t, x, _y in legs]
        extents.append(round(max(xs) - min(xs), 4))
        supports.append(len(legs) - 1)          # feet down while one swings
    check("extra legs add support points, not a longer body",
          supports == [3, 5, 7] and len(set(extents)) == 1,
          f"support points {supports}, fore-aft extent constant at "
          f"{extents[0]} m")
    check("a sprawl's foot circle is wider than its hip circle",
          (robot.COXA + robot.FEMUR) > robot.COXA * 2,
          f"feet at {robot.COXA + robot.FEMUR:.2f} m vs hips at "
          f"{robot.COXA:.2f} m — this is what 'sprawling' means")

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
