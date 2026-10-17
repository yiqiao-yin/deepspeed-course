#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0"]
# ///
"""
Lab 10 — the sweep is only a measurement if the task is decidable.

Lab 10 sweeps router accuracy and reports that perception does not
pay. That is a believable conclusion AND the exact shape a broken
experiment produces, so the parts that could manufacture it are the
parts worth testing:

  1. THE TASK MUST CONTAIN A DECISION. On a randomly chosen A-B
     pair the oracle, the blind and the paranoid beliefs planned the
     IDENTICAL route — the pair never passed near a blocker, so what
     the router believed could not change anything. Every arm then
     scores the same by construction and the sweep reads as a clean
     null. `belief_sensitive_task` exists to reject those, and the
     counterexample is kept here permanently: an unselected pair is
     shown to be undecidable often, so the selector is doing work.

  2. THE ROUTERS MUST ACTUALLY DIFFER. A blind router that returns
     ground truth, or a noisy one whose `p` does nothing, would also
     produce a flat sweep. Each arm is checked against the belief it
     claims to hold, over enough draws for the rate to be meaningful.

  3. THE TWO ERRORS ARE NOT THE SAME ERROR. Wrong-AROUND costs
     distance; wrong-OVER costs the run. The biased arm must lean
     toward AROUND, which is the whole reason it is not just another
     accuracy number.

Properties, not shapes. No GPU, no training, no downloads, no robot.

    uv run tests/test_perception_pays.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB5 = REPO / "07_physical_ai" / "05_mars_terrain"
LAB8 = REPO / "07_physical_ai" / "08_long_traverse"
LAB10 = REPO / "07_physical_ai" / "10_perception_pays"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    for p in (LAB5, LAB8, LAB10):
        sys.path.insert(0, str(p))

    import mars
    import traverse as T

    import perception_sweep as P

    print("=" * 74)
    print("  LAB 10 — does perception pay?")
    print("=" * 74)

    # ------------------------------------------------------------------
    # 1. The routers hold the beliefs they advertise.
    # ------------------------------------------------------------------
    print("\n  Routers")
    rng = np.random.default_rng(0)
    N = 4000

    orc = T.Router("oracle")
    check("the oracle is right on both classes",
          all(orc.verdict(t, rng) == t for t in (True, False) for _ in range(50)))

    blind = T.BlindRouter("blind")
    check("the blind router believes NOTHING blocks, truth regardless",
          not any(blind.verdict(t, rng) for t in (True, False) for _ in range(50)))

    # A noisy router whose p does nothing would flatten the whole sweep.
    rates = {}
    for p in (0.9, 0.7, 0.5):
        r = T.NoisyRouter(f"p={p}", p)
        hit = sum(r.verdict(t, rng) == t
                  for t in (True, False) for _ in range(N // 2))
        rates[p] = hit / N
    check("noisy accuracy tracks p (±0.03)",
          all(abs(rates[p] - p) < 0.03 for p in rates),
          ", ".join(f"p={p}→{rates[p]:.3f}" for p in rates))
    check("and the arms are ORDERED by it — p90 > p70 > p50",
          rates[0.9] > rates[0.7] > rates[0.5])

    # ------------------------------------------------------------------
    # 2. The two errors are different errors, and the biased arm leans.
    # ------------------------------------------------------------------
    print("\n  The biased arm leans, which is why it is not an accuracy")
    b = T.BiasedRouter("biased")
    on_blocked = sum(b.verdict(True, rng) for _ in range(N)) / N
    on_open = sum(b.verdict(False, rng) for _ in range(N)) / N
    check("correct on blockers at the measured ~0.72",
          abs(on_blocked - b.p_blocked) < 0.03, f"{on_blocked:.3f}")
    check("cries wolf on open ground more often than not",
          on_open > 0.5, f"says BLOCKED on {on_open:.1%} of crossable ground")
    check("so its errors lean AROUND, not OVER",
          on_open > (1.0 - on_blocked),
          f"wrong-AROUND {on_open:.2f} vs wrong-OVER {1 - on_blocked:.2f}")

    # ------------------------------------------------------------------
    # 3. Belief must be able to move the route — the counterexample
    #    that forced `belief_sensitive_task` to exist, kept permanently.
    # ------------------------------------------------------------------
    print("\n  Is the task decidable? (the check that caught the null)")
    s = mars.generate(3, scale="traverse")
    ok = mars.traversable(s, T.STEP_MAX)
    free = T.belief_mask(s, [False] * len(s.landforms))
    para = T.belief_mask(s, [True] * len(s.landforms))
    # `belief_mask` returns PASSABLE cells, so paranoia leaves fewer.
    check("believing everything blocked leaves strictly less passable ground",
          int(para.sum()) < int(free.sum()),
          f"{int(free.sum())} → {int(para.sum())} passable cells")
    check("and still leaves somewhere to walk",
          int(ok.sum()) > 0 and int(para.sum()) > 0)

    # An UNSELECTED pair: the failure mode. Sample random standable
    # pairs in the band and count how many are undecidable — belief
    # cannot change their route. If that number were zero the selector
    # would be decoration.
    e = s.extent
    g = np.random.default_rng(3)
    flat, dec = 0, 0
    for _ in range(140):
        A = (g.uniform(-e / 2 + 3, e / 2 - 3), g.uniform(-e / 2 + 3, e / 2 - 3))
        B = (g.uniform(-e / 2 + 3, e / 2 - 3), g.uniform(-e / 2 + 3, e / 2 - 3))
        if not (mars.standable(s, *A) and mars.standable(s, *B)):
            continue
        d = float(np.hypot(B[0] - A[0], B[1] - A[1]))
        if not (P.ROUTE_BAND[0] <= d <= P.ROUTE_BAND[1]):
            continue
        _p0, l0 = T.plan(s, A, B, free)
        _p1, l1 = T.plan(s, A, B, para)
        if not (np.isfinite(l0) and np.isfinite(l1)):
            continue
        dec += 1
        if l1 - l0 < 1.5:
            flat += 1
    check("unselected pairs ARE often undecidable — the selector works",
          dec > 0 and flat > 0,
          f"{flat}/{dec} random pairs in band move the route by < 1.5 m")

    # And the selected one is not.
    task = P.belief_sensitive_task(3, P.ROUTE_BAND)
    check("belief_sensitive_task finds a task at all", task is not None)
    if task is not None:
        _s, A, B, L, gap = task
        check("the selected task's route is inside the sized band",
              P.ROUTE_BAND[0] <= L <= P.ROUTE_BAND[1], f"{L:.1f} m")
        check("and belief moves its route by at least the threshold",
              gap >= 1.5, f"gap {gap:.1f} m")

    # ------------------------------------------------------------------
    # 4. Wilson intervals — the sweep's whole conclusion is "these
    #    overlap", so an interval that cannot be wrong proves nothing.
    # ------------------------------------------------------------------
    print("\n  The interval the conclusion rests on")
    lo, hi = P.wilson(11, 33)
    check("contains the point estimate", lo < 11 / 33 < hi, f"[{lo:.3f},{hi:.3f}]")
    check("a small n gives a WIDE interval", (hi - lo) > 0.2, f"{hi - lo:.3f}")
    wide = P.wilson(11, 33)
    tight = P.wilson(1100, 3300)
    check("and it narrows with n — otherwise it is a constant, not a CI",
          (tight[1] - tight[0]) < (wide[1] - wide[0]) / 4,
          f"{wide[1] - wide[0]:.3f} → {tight[1] - tight[0]:.3f}")
    check("degenerate n=0 does not raise", P.wilson(0, 0) == (0.0, 0.0))

    # ------------------------------------------------------------------
    # 5. The arm table is the experiment's design. A missing control
    #    is the one defect that cannot be seen in the output.
    # ------------------------------------------------------------------
    print("\n  The arms")
    a = P.arms()
    check("a perfect ceiling arm exists", isinstance(a.get("oracle"), T.Router))
    check("a no-information control exists",
          isinstance(a.get("blind"), T.BlindRouter))
    check("lab 7's real, asymmetric failure is its own arm",
          isinstance(a.get("biased"), T.BiasedRouter))
    check("and the swept middle is monotone in p",
          [a[k].p for k in ("p90", "p80", "p70", "p60")]
          == sorted([a[k].p for k in ("p90", "p80", "p70", "p60")],
                    reverse=True))

    print("\n" + "=" * 74)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 74)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
