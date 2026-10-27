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


def results() -> dict:
    import json
    return json.loads((RUNS / "results.json").read_text())


def fig_arms(plt) -> None:
    """
    Four arms, because two was not enough to say anything.

    The lab originally compared a VISION student against a BLIND PPO
    policy and attributed the whole gap to the camera. Those two differ
    in the camera AND in how they were trained, so the comparison could
    not separate them. The two middle bars are the arms that were
    missing.
    """
    import numpy as np

    r = results()["arms"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.patch.set_facecolor(DEEP)

    def bc(depth):
        return [a["rollout"]["vision"] for n, a in r.items()
                if n.startswith("student") and a.get("use_depth") is depth]

    ARMS = [
        ("blind\nPPO", [s for s in r["blind_ppo"]["seeds"]], BLIND),
        ("blind\nBC", bc(False), "#e8a33d"),
        ("vision\nBC", bc(True), VISION),
        ("privileged\nPPO (oracle)", [s for s in r["privileged_ppo"]["seeds"]], PRIV),
    ]
    xs = np.arange(len(KINDS))
    w = 0.2
    for i, (name, seeds, colour) in enumerate(ARMS):
        m = [np.mean([s[k] for s in seeds]) for k in KINDS]
        e = [np.std([s[k] for s in seeds], ddof=1) for k in KINDS]
        ax.bar(xs + (i - 1.5) * w, m, w, yerr=e, color=colour, edgecolor=GRID,
               ecolor=MUTED, capsize=3, label=name.replace("\n", " "))

    ax.set_xticks(xs)
    ax.set_xticklabels(["flat", "upstairs", "downstairs"])
    ax.set_ylim(0, 1.14)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.annotate("only this column moves", xy=(1, 1.06), ha="center",
                color=MUTED, fontsize=9)
    style(ax, "Four arms, three seeds each — flat and downstairs are "
              "solved WITHOUT a camera", "", "episodes clearing the terrain")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-arms.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-arms.png")


def fig_decomposition(plt) -> None:
    """Where the published 12.5% -> 89% on `up` actually came from."""
    import numpy as np

    r = results()["arms"]
    ppo = np.mean([s["up"] for s in r["blind_ppo"]["seeds"]])
    bcb = np.mean([a["rollout"]["vision"]["up"] for n, a in r.items()
                   if n.startswith("student") and not a.get("use_depth")])
    bcv = np.mean([a["rollout"]["vision"]["up"] for n, a in r.items()
                   if n.startswith("student") and a.get("use_depth")])

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    fig.patch.set_facecolor(DEEP)
    ax.barh([0], [ppo], color=BLIND, edgecolor=GRID, label="blind PPO baseline")
    ax.barh([0], [bcb - ppo], left=[ppo], color="#e8a33d", edgecolor=GRID,
            label=f"+{100*(bcb-ppo):.0f} from DISTILLATION (no camera)")
    ax.barh([0], [bcv - bcb], left=[bcb], color=VISION, edgecolor=GRID,
            label=f"+{100*(bcv-bcb):.0f} from the CAMERA")
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Upstairs: the headline was two effects, not one", "", "")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-decomposition.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-decomposition.png")


def fig_heldout(plt) -> None:
    """Does it work on stairs steeper than it ever trained on? Upstairs, no."""
    r = results()["heldout"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_facecolor(DEEP)

    for terrain, colour in (("up", VISION), ("down", PRIV)):
        xs = sorted(float(k) for k in r[terrain])
        ys = [r[terrain][f"{x:.2f}"]["rate"] for x in xs]
        ax.plot(xs, ys, color=colour, lw=2.2, marker="o", markersize=7,
                label=terrain)
    ax.axvspan(0.06, 0.11, color="#1e5f8f", alpha=0.18)
    ax.annotate("trained here", xy=(0.085, 1.04), ha="center", color=MUTED,
                fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Outside the training range, only UPSTAIRS breaks",
          "stair rise (m)", "episodes clearing the terrain")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "terrain-heldout.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  terrain-heldout.png")


def fig_ablation(plt) -> None:
    """
    The honesty check, not the result: blank the camera and re-measure.

    A policy that scores well may still be ignoring its input. This is
    the panel that distinguishes "the student can see" from "the student
    is good at walking".
    """
    import numpy as np

    r = results()["arms"]
    st = r["student_s2"]["rollout"]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    fig.patch.set_facecolor(DEEP)

    xs = np.arange(len(KINDS))
    w = 0.26
    ax.bar(xs - w, [st["vision"][t] for t in KINDS], w,
           color=VISION, edgecolor=GRID, label="real depth")
    ax.bar(xs, [st["mean"][t] for t in KINDS], w,
           color="#e8a33d", edgecolor=GRID,
           label="training-set MEAN image (in-distribution control)")
    ax.bar(xs + w, [st["blank"][t] for t in KINDS], w,
           color=BLIND, edgecolor=GRID,
           label="all zeros (OUT-of-distribution — a weak control)")

    ax.set_xticks(xs)
    ax.set_xticklabels(["flat", "upstairs", "downstairs"])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "The zeros control destroys even FLAT, which needs no camera",
          "", "episodes clearing the terrain")
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
    fig_decomposition(plt)
    fig_heldout(plt)
    fig_ablation(plt)
    fig_curves(plt)
    fig_depth(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
