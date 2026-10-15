#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2"]
# ///
"""
Lab 7 (`07_physical_ai/07_terrain_vlm`) — the label, the scorer, the bar.

Three things in this lab can be quietly wrong while every number
still looks plausible, and each has a precedent in this repository:

  * the LABEL could be circular (predicting the generator's own name
    for a landform rather than its geometry)
  * the SCORER could match by substring, counting "I would not go
    AROUND" as the answer AROUND -- `05_video_speech` shipped exactly
    that, matching "not Paris" against "Paris"
  * the CONTROL could be mis-specified, so a model that ignores the
    image looks like one that reads terrain

No GPU, no model download, no images. The vision model is not under
test here; the machinery that decides whether it succeeded is.

    uv run tests/test_terrain_vlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB7 = REPO / "07_physical_ai" / "07_terrain_vlm"
LAB5 = REPO / "07_physical_ai" / "05_mars_terrain"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    # ORDER MATTERS, and getting it wrong is silent. Lab 5 also has an
    # `evaluate.py`; with lab 5 ahead on the path, `from evaluate
    # import parse` imports lab 5's module and this suite tests the
    # wrong lab's scorer. Lab 7 goes first; `mars` still resolves from
    # lab 5 because lab 7 has no module of that name.
    sys.path.insert(0, str(LAB5))
    sys.path.insert(0, str(LAB7))
    import make_dataset as md
    from evaluate import VALID, parse, wilson
    from verify_arch import footprint

    print("=" * 74)
    print("  LAB 7 — the label, the scorer, and the bar")
    print("=" * 74)

    # ---------------------------------------------------------------
    # The scorer. This is the one with a shipped precedent.
    # ---------------------------------------------------------------
    check("a bare answer parses", parse("OVER") == "OVER")
    check("punctuation and case are tolerated",
          parse("  around. ") == "AROUND")
    # THE KEPT COUNTEREXAMPLE. A substring scorer returns AROUND here
    # and scores a refusal as a correct answer.
    check("a sentence CONTAINING the word does not count",
          parse("I would not go AROUND that") is None,
          "substring matching would have returned AROUND")
    check("prose is unparseable, not wrong-but-scored",
          parse("The terrain appears rocky") is None)
    check("empty output is unparseable", parse("") is None)
    check("only the two labels are valid", set(VALID) == {"OVER", "AROUND"})

    # ---------------------------------------------------------------
    # The interval. Small n, so the normal approximation is not safe.
    # ---------------------------------------------------------------
    lo, hi = wilson(50, 100)
    check("the CI brackets the estimate", lo < 0.5 < hi, f"[{lo:.3f},{hi:.3f}]")
    n_lo, n_hi = wilson(5, 10)
    check("a smaller sample gives a wider interval",
          (n_hi - n_lo) > (hi - lo),
          f"n=10 width {n_hi-n_lo:.2f} vs n=100 width {hi-lo:.2f}")
    check("a perfect score still has an interval",
          wilson(10, 10)[0] < 1.0,
          f"10/10 lower bound {wilson(10,10)[0]:.3f}, not 1.0")

    # ---------------------------------------------------------------
    # The label thresholds. An ambiguous band must EXIST, or samples
    # the oracle cannot classify get rounded into training data.
    # ---------------------------------------------------------------
    check("the thresholds leave an ambiguous band",
          md.CROSSABLE_AT < md.BLOCKED_AT,
          f"OVER <= {md.CROSSABLE_AT}, AROUND >= {md.BLOCKED_AT}")

    # ---------------------------------------------------------------
    # The label is GEOMETRY. Build two surfaces by hand -- one flat,
    # one with a wall -- and check the measurement separates them.
    # Asserting on the real generator would only show that today's
    # seeds behave; this shows the measurement itself works.
    # ---------------------------------------------------------------
    import mars

    flat = mars.generate(0, scale="course")
    flat.heights = np.zeros_like(flat.heights)
    eb_flat = md.edge_blocked(flat, 0.0, 0.0, 3.0)
    check("flat ground measures as unblocked", eb_flat == 0.0,
          f"{eb_flat:.3f}")

    wall = mars.generate(0, scale="course")
    h = np.zeros_like(wall.heights)
    n = h.shape[0]
    # A step far taller than STEP_MAX, right where the ring samples.
    cell = wall.extent / n
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * cell - wall.extent / 2
    wy = (ys + 0.5) * cell - wall.extent / 2
    h[np.hypot(wx, wy) < 3.0] = 5.0
    wall.heights = h
    eb_wall = md.edge_blocked(wall, 0.0, 0.0, 3.0)
    check("a 5 m wall measures as impassable, not crossable",
          eb_wall > md.CROSSABLE_AT, f"{eb_wall:.3f} > {md.CROSSABLE_AT}")
    check("the measurement separates wall from flat",
          eb_wall - eb_flat > 0.05, f"{eb_flat:.3f} vs {eb_wall:.3f}")

    # A PROPERTY OF THE METRIC, pinned because it is counterintuitive
    # and because the first version of this test asserted the opposite.
    #
    # `edge_blocked` is the fraction of a RING that a gait cannot
    # cross, so it measures how WIDE the impassable zone is, not how
    # tall the obstacle is. A perfectly vertical 5 m wall blocks only
    # the single ring of cells at its discontinuity and scores 0.12,
    # while a real mesa -- whose shoulder is graded over several
    # cells -- scores 0.31. Taller is not higher-scoring; broader is.
    #
    # That is the right behaviour for this lab (a sheer step and a
    # wide scree slope are both impassable, and the wide one is harder
    # to get around) but it means the metric must never be described
    # as measuring height.
    check("the metric measures WIDTH of blockage, not height",
          eb_wall < 0.25,
          f"a sheer 5 m wall scores {eb_wall:.3f}, below a real mesa's ~0.31")

    # ---------------------------------------------------------------
    # The occlusion filter, watched in BOTH directions. A filter that
    # never rejects is not a filter.
    # ---------------------------------------------------------------
    s = mars.generate(0, scale="course")
    s.landforms = [("mesa", 10.0, 0.0, 2.0, 1.0), ("hill", 5.0, 0.0, 1.5, 0.5)]
    target = ("mesa", 10.0, 0.0, 2.0)
    check("a nearer landform in the line of sight is rejected",
          not md.view_is_clean(s, 0.0, 0.0, 10.0, 0.0, target),
          "the hill at 5 m sits between camera and target")
    check("a clear line of sight is accepted",
          md.view_is_clean(s, 0.0, 9.0, 10.0, 0.0, target),
          "viewed from the side, the hill is out of the cone")

    # ---------------------------------------------------------------
    # The capacity argument lab 7 is built on. If LoRA needed ZeRO,
    # the lab's whole justification for using it would be wrong.
    # ---------------------------------------------------------------
    p = 8.03e9
    full_1 = footprint(p, p, 1)
    full_8 = footprint(p, p, 8)
    lora_1 = footprint(p, p * 0.002, 1)
    check("full fine-tuning does NOT fit one 24 GB card", full_1 > 24,
          f"{full_1:.0f} GB")
    check("ZeRO-3 across 8 brings it under 24 GB", full_8 < 24,
          f"{full_8:.0f} GB")
    check("LoRA already fits one card, so ZeRO would be decoration",
          lora_1 < 24, f"{lora_1:.0f} GB")
    check("sharding divides, as ZeRO-3 claims",
          abs(full_1 / 8 - full_8) < 1e-6)

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
