#!/usr/bin/env python3
"""
Plot the three-arm result. Every number is read from `runs/*/summary.json`.

    uv run make_figures.py

Nothing here can draw a result a run did not produce. That matters more
than usual in this lab: the whole claim is a DIFFERENCE between arms,
and a hand-adjusted bar would be a fabricated difference rather than a
cosmetic liberty.
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
PRIV, BLIND, VISION, GOOD = "#63a3d0", "#e26e6e", "#5cc48d", "#6d8498"

sys.path.insert(0, str(HERE))
from terrain import KINDS  # noqa: E402


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


def load(prefix: str) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(RUNS.glob(f"{prefix}_s*/summary.json"))]


def student() -> dict:
    return json.loads((RUNS / "student" / "summary.json").read_text())


def fig_arms(plt) -> None:
    """
    The figure the lab exists for: three arms, per terrain.

    Per terrain and never averaged. `flat` is solvable blind, so a mean
    over all three would dilute the only two columns that carry the
    result -- and would have reported a modest overall gap instead of
    the 71-point one on ascent.
    """
    import numpy as np

    priv, blind, st = load("v3_priv"), load("v3_blind"), student()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(DEEP)

    xs = np.arange(len(KINDS))
    w = 0.26
    for k, (name, colour, vals, err) in enumerate((
        ("privileged — told the terrain", PRIV,
         [np.mean([j["final"][f"past_{t}"] for j in priv]) for t in KINDS],
         [np.std([j["final"][f"past_{t}"] for j in priv]) for t in KINDS]),
        ("blind — proprioception only", BLIND,
         [np.mean([j["final"][f"past_{t}"] for j in blind]) for t in KINDS],
         [np.std([j["final"][f"past_{t}"] for j in blind]) for t in KINDS]),
        ("vision — 64×64 depth", VISION,
         [st["rollout"]["vision"][t] for t in KINDS], None),
    )):
        ax.bar(xs + (k - 1) * w, vals, w, yerr=err, color=colour,
               edgecolor=GRID, ecolor=MUTED, capsize=4, label=name)

    ax.set_xticks(xs)
    ax.set_xticklabels(["flat", "upstairs", "downstairs"])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "What the camera buys, terrain by terrain", "",
          "episodes clearing the terrain")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-arms.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-arms.png")


def fig_ablation(plt) -> None:
    """
    The honesty check, not the result: blank the camera and re-measure.

    A policy that scores well may still be ignoring its input. This is
    the panel that distinguishes "the student can see" from "the student
    is good at walking".
    """
    import numpy as np

    st = student()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    fig.patch.set_facecolor(DEEP)

    xs = np.arange(len(KINDS))
    w = 0.34
    ax.bar(xs - w / 2, [st["rollout"]["vision"][t] for t in KINDS], w,
           color=VISION, edgecolor=GRID, label="camera working")
    ax.bar(xs + w / 2, [st["rollout"]["blank"][t] for t in KINDS], w,
           color=BLIND, edgecolor=GRID, label="camera blanked")

    ax.set_xticks(xs)
    ax.set_xticklabels(["flat", "upstairs", "downstairs"])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Feed the same policy a blank image", "",
          "episodes clearing the terrain")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-ablation.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-ablation.png")


def fig_curves(plt) -> None:
    """
    Learning curves per terrain, every seed.

    Worth plotting per terrain rather than as one return curve, because
    the interesting thing here is the ORDER the terrains are solved in:
    flat, then descent, then ascent. Nothing in the reward says that.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    fig.patch.set_facecolor(DEEP)

    for ax, terrain in zip(axes, KINDS):
        for prefix, colour, name in (("v3_priv", PRIV, "privileged"),
                                     ("v3_blind", BLIND, "blind")):
            for i, d in enumerate(sorted(RUNS.glob(f"{prefix}_s*/curve.csv"))):
                rows = list(csv.DictReader(d.open()))
                ax.plot([float(r["step"]) / 1e3 for r in rows],
                        [float(r[f"past_{terrain}"]) for r in rows],
                        color=colour, lw=1.4, alpha=0.8,
                        label=name if i == 0 else None)
        ax.set_ylim(-0.04, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax, terrain, "thousand environment steps",
              "cleared" if terrain == KINDS[0] else "")
    legend(axes[0])
    fig.suptitle("Flat is solved first, then descent, then ascent — "
                 "nothing in the reward says so", color=FG, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-curves.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-curves.png")


def fig_depth(plt) -> None:
    """
    What the three terrains look like to the camera, from the dataset.

    Drawn from `data/bc.npz`, i.e. from frames the student actually
    trained on, and averaged over every frame of each terrain. The mean
    images differ visibly, which is the cheapest possible evidence that
    the task is not blind-equivalent.
    """
    import numpy as np

    f = HERE / "data" / "bc.npz"
    if not f.exists():
        print("  (no data/bc.npz — skipping the depth panel)")
        return
    d = np.load(f)
    depth, terrain = d["depth"], d["terrain"]

    fig, axes = plt.subplots(1, len(KINDS), figsize=(10.2, 3.7))
    fig.patch.set_facecolor(DEEP)
    for ax, (i, k) in zip(axes, enumerate(KINDS)):
        m = depth[terrain == i].mean(axis=0)
        ax.imshow(1.0 - m, cmap="bone", vmin=0, vmax=1)
        ax.set_title(f"{k}   (mean of {int((terrain == i).sum()):,} frames)",
                     color=FG, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle("The mean depth image, per terrain — bright is near",
                 color=FG, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-depth.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-depth.png")


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing. Run: uv sync", file=sys.stderr)
        return 1

    missing = [p for p in ("v3_priv", "v3_blind") if not load(p)]
    if missing or not (RUNS / "student" / "summary.json").exists():
        print("  missing runs. Produce them first:\n"
              "      uv run train_teacher.py --name v3_priv_s0\n"
              "      uv run train_teacher.py --no-privileged --name v3_blind_s0\n"
              "      uv run collect.py && uv run train_student.py",
              file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  writing to {OUT}")
    fig_arms(plt)
    fig_ablation(plt)
    fig_curves(plt)
    fig_depth(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
