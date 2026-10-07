#!/usr/bin/env python3
"""
Every figure, read from `runs/*.json`. Nothing here is typed by hand.

    uv run make_figures.py

The arrival and efficiency numbers come from `evaluate.py`, not from the
training log. That distinction is load-bearing: the in-training
evaluation uses 12 episodes, which quantises to 8.3% and reported two
arms differing by 4.4 points as EXACTLY equal. A figure drawn from it
would have shown no effect where there is one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
DEMO_SEED = 23
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"

DEEP, PANEL = "#08182a", "#0e1b26"
FG, MUTED, GRID = "#e9f0f6", "#8ea3b5", "#1d2f3e"
BLIND, PRIV, ORACLE = "#e26e6e", "#63a3d0", "#5cc48d"

sys.path.insert(0, str(HERE))


def style(ax, title="", xlabel="", ylabel="") -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9, axis="y")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def legend(ax):
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(FG)


def evaluations() -> list[dict]:
    """Both sweeps, from evaluate.py. Six seeds per arm."""
    rows = []
    for f in ("evaluation.json", "evaluation3.json"):
        p = RUNS / f
        if p.exists():
            rows += json.loads(p.read_text())
    return rows


def fig_arrival(plt) -> None:
    """
    Arrival, paired by seed, both sweeps.

    Paired because the two arms were trained from the same seed and
    scored on the same episode seeds, so the comparison that means
    something is within-pair, not between-pools.
    """
    import numpy as np

    rows = evaluations()
    b = [r["arrived"] for r in rows if r["mode"] == "blind"]
    p = [r["arrived"] for r in rows if r["mode"] == "privileged"]
    n = min(len(b), len(p))
    b, p = np.array(b[:n]), np.array(p[:n])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.patch.set_facecolor(DEEP)
    xs = np.arange(n)
    ax.bar(xs - 0.2, b, 0.4, color=BLIND, edgecolor=GRID, label="blind")
    ax.bar(xs + 0.2, p, 0.4, color=PRIV, edgecolor=GRID,
           label="privileged (sees the ground ahead)")
    for i, (bb, pp) in enumerate(zip(b, p)):
        ax.annotate(f"{pp-bb:+.1%}", (i, max(bb, pp) + 0.008),
                    ha="center", color=PRIV if pp > bb else BLIND, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i%3}\n{'sweep 2' if i<3 else 'sweep 3'}"
                        for i in range(n)], fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, f"Arrival, 120 episodes per bar — privileged ahead on "
              f"{int((p>b).sum())} of {n} seeds", "",
          "episodes reaching B")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-arrival.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-arrival.png")


def fig_efficiency(plt) -> None:
    """Path efficiency on maps BOTH arms solved — a paired, continuous measure."""
    import numpy as np

    f = RUNS / "efficiency.json"
    if not f.exists():
        print("  (no runs/efficiency.json — skipping)")
        return
    d = json.loads(f.read_text())
    labels = [r["pair"] for r in d]
    b = np.array([r["blind"] for r in d])
    p = np.array([r["privileged"] for r in d])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.patch.set_facecolor(DEEP)
    xs = np.arange(len(d))
    ax.bar(xs - 0.2, b, 0.4, color=BLIND, edgecolor=GRID, label="blind")
    ax.bar(xs + 0.2, p, 0.4, color=PRIV, edgecolor=GRID, label="privileged")
    ax.axhline(1.0, color=ORACLE, ls="--", lw=1.4,
               label="oracle's route (efficiency 1.0)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Path efficiency on maps BOTH arms solved", "",
          "oracle route / distance walked")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-efficiency.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-efficiency.png")


def fig_world(plt) -> None:
    """One arena, its two obstacle classes, and the oracle's route."""
    import numpy as np

    from world import ARENA, CELL, STEP_MAX, generate, solve

    # A map whose shortest route genuinely detours. Seed 7's route runs
    # straight from A to B, which illustrates nothing about routing.
    m = generate(DEMO_SEED)
    path, length = solve(m)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    fig.patch.set_facecolor(DEEP)
    # vmax at 0.30, not the 1.2 m ceiling. Climbable ridges are
    # 0.06-0.12 m, so on a scale that reaches 1.2 they are black and
    # the figure shows only the walls -- hiding one of the two classes
    # it exists to contrast.
    ax.imshow(m.heights, origin="lower", cmap="bone", vmin=0, vmax=0.30,
              extent=[-ARENA/2, ARENA/2, -ARENA/2, ARENA/2])
    clip = lambda v: float(np.clip(v, -ARENA/2, ARENA/2))
    for o in m.obstacles:
        col = ORACLE if o.kind == "step" else BLIND
        # Clipped to the arena: a 17 m ridge centred within +-7.5 m of a
        # +-10 m square runs off the edge, where the height field simply
        # stops. An unclipped line draws terrain that is not there.
        ax.plot([clip(o.cx - o.length/2*np.cos(o.angle)),
                 clip(o.cx + o.length/2*np.cos(o.angle))],
                [clip(o.cy - o.length/2*np.sin(o.angle)),
                 clip(o.cy + o.length/2*np.sin(o.angle))],
                color=col, lw=2.6, alpha=0.9, label=None)
    if path:
        xs = [(i + .5)*CELL - ARENA/2 for _, i in path]
        ys = [(j + .5)*CELL - ARENA/2 for j, _ in path]
        ax.plot(xs, ys, color="#ffb04a", lw=2.6, label=f"route {length:.1f} m")
    ax.plot(*m.start, "o", color="#4099ff", ms=11, label="A")
    ax.plot(*m.goal, "*", color=ORACLE, ms=18, label="B")
    ax.plot([], [], color=ORACLE, lw=2.4, label=f"climbable (rise <= {STEP_MAX} m)")
    ax.plot([], [], color=BLIND, lw=2.4, label="impassable")
    ax.set_xticks([]); ax.set_yticks([])
    style(ax, "One arena: long ridges, two classes, one route")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-world.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-world.png")


def fig_curves(plt) -> None:
    """Every training curve, both arms, both sweeps."""
    import csv

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor(DEEP)
    seen = set()
    for f in sorted(RUNS.glob("nav[23]_*_s*/curve.csv")):
        mode = "blind" if "blind" in f.parent.name else "privileged"
        col = BLIND if mode == "blind" else PRIV
        rows = list(csv.DictReader(f.open()))
        ax.plot([float(r["step"])/1e6 for r in rows],
                [float(r["arrived"]) for r in rows],
                color=col, lw=1.2, alpha=0.75,
                label=mode if mode not in seen else None)
        seen.add(mode)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Training curves — every run peaks mid-training, then sags",
          "million environment steps", "arrival (12-episode eval)")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-curves.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-curves.png")


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing. Run: uv sync", file=sys.stderr)
        return 1
    if not evaluations():
        print("  no runs/evaluation*.json — run evaluate.py first",
              file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  writing to {OUT}")
    fig_world(plt)
    fig_arrival(plt)
    fig_efficiency(plt)
    fig_curves(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
