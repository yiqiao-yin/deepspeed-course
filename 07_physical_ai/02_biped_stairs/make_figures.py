#!/usr/bin/env python3
"""
Plot the 2x2. Every number is read from `runs/*/summary.json`.

    uv run make_figures.py

Nothing here can draw a result a training run did not produce, which
matters more on a 2x2 than on a single curve: the whole point is the
comparison, and a hand-adjusted bar would be a fabricated comparison.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"

DEEP, PANEL = "#08182a", "#0e1b26"
FG, MUTED, GRID = "#e9f0f6", "#8ea3b5", "#1d2f3e"
LOCKED, FREE, BASE, GOOD = "#63a3d0", "#e3a05a", "#6d8498", "#5cc48d"

sys.path.insert(0, str(HERE))
from morphology import CELLS, N_STAIRS  # noqa: E402


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


def load(cell: str) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(RUNS.glob(f"{cell}_s*/summary.json"))]


def legend(ax):
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(FG)


def fig_grid(plt) -> None:
    """
    The 2x2 on the two metrics that disagree.

    Return and treads-climbed are plotted side by side because the lab's
    point is that they rank the robots differently, and a single-panel
    figure would have to pick one and hide the other.
    """
    import numpy as np

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor(DEEP)
    labels = ["1 leg", "2 legs"]
    xs = np.arange(2)
    w = 0.36

    for k, (torso, colour) in enumerate((("locked", LOCKED),
                                         ("free", FREE))):
        rets, tread, err_r, err_t = [], [], [], []
        for legs in (1, 2):
            js = load(f"{legs}leg_{torso}")
            r = [j["final"]["return"] for j in js]
            c = [j["final"]["climbed"] for j in js]
            rets.append(np.mean(r)); err_r.append(np.std(r))
            tread.append(np.mean(c)); err_t.append(np.std(c))
        off = (k - 0.5) * w
        a1.bar(xs + off, rets, w, yerr=err_r, color=colour, edgecolor=GRID,
               ecolor=MUTED, capsize=4, label=f"torso {torso}")
        a2.bar(xs + off, tread, w, yerr=err_t, color=colour, edgecolor=GRID,
               ecolor=MUTED, capsize=4, label=f"torso {torso}")

    base = np.mean([j["baseline_return"] for c, _, _ in CELLS
                    for j in load(c)])
    a1.axhline(base, color=BASE, ls="--", lw=1.4,
               label=f"random baseline ({base:.0f})")
    a2.axhline(N_STAIRS, color=GOOD, ls="--", lw=1.4, label="all 3 treads")

    for a, t, y in ((a1, "Episode return", "return"),
                    (a2, "Treads actually climbed", f"of {N_STAIRS}")):
        a.set_xticks(xs); a.set_xticklabels(labels)
        style(a, t, "", y)
        legend(a)
    fig.suptitle("The two metrics rank the robots differently",
                 color=FG, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "stairs-2x2.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  stairs-2x2.png")


def fig_interaction(plt) -> None:
    """
    The interaction plot: non-parallel lines mean the switches interact.

    This is the figure that carries the finding. If freeing the torso had
    one effect regardless of leg count the two lines would be parallel.
    They cross.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(DEEP)

    for torso, colour in (("locked", LOCKED), ("free", FREE)):
        ys = [np.mean([j["final"]["climbed"] for j in load(f"{n}leg_{torso}")])
              for n in (1, 2)]
        ax.plot([0, 1], ys, color=colour, lw=2.4, marker="o", markersize=9,
                label=f"torso {torso}")
        for x, y in zip((0, 1), ys):
            ax.annotate(f"{y:.2f}", (x, y), color=colour, fontsize=10,
                        xytext=(0, 10), textcoords="offset points",
                        ha="center")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["1 leg", "2 legs"])
    ax.set_xlim(-0.25, 1.25)
    ax.axhline(N_STAIRS, color=GOOD, ls="--", lw=1.4, label="all 3 treads")
    style(ax, "Freeing the torso helps one leg and costs two",
          "", f"treads climbed, of {N_STAIRS}")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "stairs-interaction.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  stairs-interaction.png")


def fig_curves(plt) -> None:
    """Learning curves, every seed, coloured by cell."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(DEEP)
    styles = {"1leg_locked": (LOCKED, ":"), "1leg_free": (FREE, ":"),
              "2leg_locked": (LOCKED, "-"), "2leg_free": (FREE, "-")}

    for cell, (colour, ls) in styles.items():
        for i, d in enumerate(sorted(RUNS.glob(f"{cell}_s*/curve.csv"))):
            rows = list(csv.DictReader(d.open()))
            ax.plot([float(r["step"]) / 1e3 for r in rows],
                    [float(r["climbed"]) for r in rows],
                    color=colour, ls=ls, lw=1.5, alpha=0.85,
                    label=cell.replace("_", "  ") if i == 0 else None)

    ax.axhline(N_STAIRS, color=GOOD, ls="--", lw=1.3, label="all 3 treads")
    style(ax, "Learning to climb — every seed",
          "thousand environment steps", f"treads climbed, of {N_STAIRS}")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "stairs-curves.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  stairs-curves.png")


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing. Run: uv sync", file=sys.stderr)
        return 1

    missing = [c for c, _, _ in CELLS if not load(c)]
    if missing:
        print(f"  no runs for {missing}. Train the full 2x2 first:\n"
              f"      uv run train_ppo.py --sweep --seeds 3", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  writing to {OUT}")
    fig_grid(plt)
    fig_interaction(plt)
    fig_curves(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
