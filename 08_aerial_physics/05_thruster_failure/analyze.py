#!/usr/bin/env python3
"""
Did the geometry predict the flight?

    uv run analyze.py runs/healthy_s0 runs/robust_s0
    uv run analyze.py runs/robust_s0 --json cmp.json

WHAT THIS ANSWERS
-----------------
`fault.py` asks a question about shape: with nozzle k broken, is a
hover still inside the reachable wrench polytope? That is cheap, it
runs before training, and it sorts every failure into "feasible" and
"infeasible".

This file checks that prediction against what actually flew, and the
interesting part is that it fails in BOTH directions:

  reachable but not recovered   the polytope allows a hover and the
                                vehicle still hit the ground. The
                                transient has to be caught, and the
                                ground does not wait.
  unreachable but landed        the polytope forbids a hover and the
                                vehicle landed anyway. LANDING IS NOT
                                HOVERING. A vehicle that cannot hold
                                station indefinitely can still spend
                                its remaining authority on the way
                                down and arrive inside the gate.

The second one is a correction to this lab's own first framing, which
treated hover feasibility as the ceiling on landing. It is not a
ceiling; it is a different question that happens to be easy to ask.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def fatal_sets(rig: str, world: str) -> dict[str, set[int]]:
    """Which nozzles are fatal in each mode, from geometry alone."""
    import fault

    d = fault.survey(rig, world)
    return {"shut": set(d["shut_fatal"]), "open": set(d["open_fatal"])}


def split(rows: list[dict], fatal: dict[str, set[int]]) -> dict:
    """Landing rate by (mode, whether geometry called it fatal)."""
    out = {}
    for mode in ("shut", "open"):
        for lab, want in (("feasible", False), ("infeasible", True)):
            xs = [r for r in rows
                  if r["mode"] == mode
                  and ((r["nozzle"] in fatal[mode]) is want)]
            out[(mode, lab)] = (sum(r["ok"] for r in xs), len(xs))
    xs = [r for r in rows if r["mode"] == "none"]
    out[("none", "—")] = (sum(r["ok"] for r in xs), len(xs))
    return out


def report(name: str, rows: list[dict], fatal: dict[str, set[int]]) -> dict:
    g = split(rows, fatal)
    print(f"\n  {name}")
    print(f"    {'failure':<12}{'geometry':<12}{'landed':>12}{'95% CI':>16}")
    for key in (("none", "—"), ("shut", "feasible"), ("shut", "infeasible"),
                ("open", "feasible"), ("open", "infeasible")):
        k, n = g[key]
        if n == 0:
            continue
        lo, hi = wilson(k, n)
        mode = "healthy" if key[0] == "none" else f"stuck {key[0]}"
        print(f"    {mode:<12}{key[1]:<12}{f'{k}/{n}':>7} {k / n:>4.0%}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}")
    tot_k = sum(r["ok"] for r in rows)
    print(f"    {'ALL':<24}{f'{tot_k}/{len(rows)}':>7} "
          f"{tot_k / max(len(rows), 1):>4.0%}")
    return {str(k): v for k, v in g.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    import sys
    sys.path.insert(0, str(HERE))

    print("=" * 74)
    print("  DID THE GEOMETRY PREDICT THE FLIGHT?")
    print("=" * 74)
    print("  'feasible' means fault.py found a hover still reachable with")
    print("  that nozzle broken. It is a statement about shape, made before")
    print("  training, and it is NOT a ceiling on landing -- see below.")

    out = {}
    for rd in a.runs:
        meta = json.loads((Path(rd) / "summary.json").read_text())
        rows = meta["eval"].get("rows", [])
        if not rows:
            print(f"\n  {Path(rd).name}: no per-episode rows recorded")
            continue
        fatal = fatal_sets(meta["rig"], meta.get("world", "earth"))
        out[Path(rd).name] = report(Path(rd).name, rows, fatal)

    print("\n  Two ways the prediction misses, and both are real:")
    print("    feasible + crashed   the polytope allowed a hover and the")
    print("                         vehicle still could not catch the")
    print("                         transient before the ground.")
    print("    infeasible + landed  the polytope forbade a hover and the")
    print("                         vehicle landed regardless. Landing is")
    print("                         not hovering: authority spent on the")
    print("                         way down does not have to be held.")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
