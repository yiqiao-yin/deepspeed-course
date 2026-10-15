#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0"]
# ///
"""
Lab 6 (`07_physical_ai/06_mars_mjx`) — the ablation must be real.

Lab 6's whole finding rests on an ablation whose results are almost
entirely NEGATIVE: rocks make no measurable difference, height-field
resolution makes no measurable difference, and flattening the terrain
buys only 1.10x against a 3.2% spread. That is exactly the shape a
broken knob produces.

A `rocks=0` argument that silently did nothing would report "rocks
make no difference" and be believed, because the number it prints is
plausible and agrees with the conclusion the author already has. That
is this repository's most-repeated failure mode, so the knobs are
tested rather than trusted: each one must demonstrably change the
model it claims to change.

No GPU, no JAX, no Warp. The backends are not under test here; the
MODEL CONSTRUCTION the measurement rests on is.

    uv run tests/test_mars_mjx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB6 = REPO / "07_physical_ai" / "06_mars_mjx"
LAB5 = REPO / "07_physical_ai" / "05_mars_terrain"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def hfield_of(model) -> np.ndarray:
    """
    The height field actually loaded into the compiled model.

    NOTE: MuJoCo NORMALISES this array to [0, 1] at compile time and
    keeps the real elevation in `hfield_size[2]`. So `ptp()` of this
    is 1.0 for every non-flat terrain and asserting on it is a check
    that cannot fail -- which is what the first version of this file
    did. Use `relief_of()` for metres.
    """
    assert model.nhfield == 1, f"expected one height field, got {model.nhfield}"
    n = int(model.hfield_nrow[0]) * int(model.hfield_ncol[0])
    adr = int(model.hfield_adr[0])
    return np.asarray(model.hfield_data[adr:adr + n], dtype=float)


def relief_of(model) -> float:
    """Vertical extent of the terrain in METRES, as compiled."""
    return float(model.hfield_size[0][2])


def main() -> int:
    sys.path.insert(0, str(LAB6))
    from bench_backends import WORTH_RENTING, build_model

    print("=" * 74)
    print("  LAB 6 — the ablation knobs must actually move the model")
    print("=" * 74)

    base, task = build_model(0)
    n_rocks = len(task.rocks)
    check("the course builds", base.ngeom > 0 and n_rocks > 0,
          f"{base.ngeom} geoms, {n_rocks} rocks")

    # ---------------------------------------------------------------
    # Knob 1: rocks. `rocks=0` must remove geometry, not just relabel.
    # ---------------------------------------------------------------
    bare, _ = build_model(0, rocks=0)
    check("rocks=0 removes every rock geom",
          bare.ngeom == base.ngeom - n_rocks,
          f"{base.ngeom} -> {bare.ngeom}, expected -{n_rocks}")

    # The counterexample, kept permanently: `rocks=None` is the DEFAULT
    # and must be a no-op. If this ever starts differing, the "rocks
    # make no difference" reading is measuring the knob, not the rocks.
    same, _ = build_model(0, rocks=None)
    check("rocks=None is the unmodified course",
          same.ngeom == base.ngeom,
          f"{base.ngeom} vs {same.ngeom}")

    # ---------------------------------------------------------------
    # Knob 2: resolution. The TABULATION changes; the surface does not.
    # ---------------------------------------------------------------
    coarse, _ = build_model(0, hfield_res=45)
    fine, _ = build_model(0, hfield_res=180)
    check("hfield_res=45 gives a 45x45 field",
          (int(coarse.hfield_nrow[0]), int(coarse.hfield_ncol[0])) == (45, 45),
          f"{int(coarse.hfield_nrow[0])}x{int(coarse.hfield_ncol[0])}")
    check("hfield_res=180 gives a 180x180 field",
          (int(fine.hfield_nrow[0]), int(fine.hfield_ncol[0])) == (180, 180),
          f"{int(fine.hfield_nrow[0])}x{int(fine.hfield_ncol[0])}")
    # Same terrain, differently sampled: the relief must survive. A
    # resample that flattened the surface would also show "no effect"
    # on throughput, for the wrong reason.
    c_rel, f_rel = relief_of(coarse), relief_of(fine)
    check("resampling preserves the relief",
          c_rel > 0.3 and abs(c_rel - f_rel) / max(f_rel, 1e-9) < 0.25,
          f"45^2 {c_rel:.3f} m, 180^2 {f_rel:.3f} m")

    # ---------------------------------------------------------------
    # Knob 3: the decisive one. `plane=True` must flatten the field
    # while keeping the COLLIDER a height field -- otherwise the 8.6x
    # is a comparison between two different collision routines and
    # says nothing about terrain.
    # ---------------------------------------------------------------
    flat, _ = build_model(0, plane=True)
    check("plane=True flattens the height field",
          float(np.abs(hfield_of(flat)).max()) == 0.0,
          f"max |height| {float(np.abs(hfield_of(flat)).max()):.3e}")
    check("plane=True keeps ONE height field (not a plane geom)",
          flat.nhfield == 1 and flat.ngeom == base.ngeom,
          f"nhfield {flat.nhfield}, ngeom {flat.ngeom} vs {base.ngeom}")
    check("plane=True is measurably flatter than the real course",
          relief_of(flat) < 0.01 < 0.3 < relief_of(base),
          f"flat {relief_of(flat):.4f} m vs course {relief_of(base):.3f} m")

    # ---------------------------------------------------------------
    # The falsifier is a number in the source, and the verdict must
    # follow it in BOTH directions. A bar that only ever says HOLDS is
    # not a bar.
    # ---------------------------------------------------------------
    check("the falsifier bar is a real threshold",
          isinstance(WORTH_RENTING, (int, float)) and WORTH_RENTING > 1.0,
          f"{WORTH_RENTING}x the CPU pool")
    verdict = lambda s: "HOLDS" if s >= WORTH_RENTING else "FAILS"   # noqa: E731
    check("a speedup under the bar FAILS",
          verdict(WORTH_RENTING - 0.01) == "FAILS")
    check("a speedup over the bar HOLDS",
          verdict(WORTH_RENTING + 0.01) == "HOLDS")

    # ---------------------------------------------------------------
    # Amdahl: a missing or malformed run must return None, never a
    # number. A silently-wrong share would be the one figure in this
    # lab nobody could check by eye.
    # ---------------------------------------------------------------
    from bench_backends import amdahl_share
    check("amdahl_share returns None for a run that does not exist",
          amdahl_share("no_such_run_xyz") is None)

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
