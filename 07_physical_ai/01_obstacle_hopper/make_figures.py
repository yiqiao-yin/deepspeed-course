#!/usr/bin/env python3
"""
Plot what the runs actually produced. Nothing here invents a number.

    uv run make_figures.py

Every figure is read from `runs/*/curve.csv` and `runs/*/summary.json`,
written by `train_ppo.py`. There is no path through this script that can
draw a curve a training run did not produce -- which matters more than it
sounds, because a hand-tuned "illustrative" learning curve is the figure
equivalent of a fabricated measurement, and it is the easiest thing in the
world to produce by accident.

Figures land in `docusaurus-docs/static/img/physical/` so the book page can
reference them directly.
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
SEEING, BLIND, BASE, ACCENT = "#63a3d0", "#e3a05a", "#6d8498", "#5cc48d"


def style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def load(name: str) -> tuple[list[dict], dict]:
    d = RUNS / name
    rows = list(csv.DictReader((d / "curve.csv").open()))
    for r in rows:
        for k, v in r.items():
            r[k] = float(v)
    return rows, json.loads((d / "summary.json").read_text())


def runs_for(prefix: str) -> list[str]:
    return sorted(p.name for p in RUNS.iterdir()
                  if p.is_dir() and p.name.startswith(prefix)
                  and (p / "curve.csv").exists())


def fig_learning(plt) -> None:
    """Every seed, both arms, with the baseline as a floor."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor(DEEP)

    base = None
    for arm, colour, label in (("seeing", SEEING, "sees obstacle height"),
                               ("blind", BLIND, "blind to height")):
        names = runs_for(arm)
        for i, n in enumerate(names):
            rows, summ = load(n)
            base = summ["baseline_return"]
            x = [r["step"] / 1e3 for r in rows]
            a1.plot(x, [r["return"] for r in rows], color=colour, lw=1.6,
                    alpha=0.85, label=label if i == 0 else None)
            a2.plot(x, [r["max_x"] for r in rows], color=colour, lw=1.6,
                    alpha=0.85, label=label if i == 0 else None)

    if base is not None:
        a1.axhline(base, color=BASE, ls="--", lw=1.4,
                   label=f"random baseline ({base:.0f})")
    a2.axhline(1.35, color=ACCENT, ls="--", lw=1.4, label="far edge of the step")

    style(a1, "Episode return", "thousand environment steps", "return")
    style(a2, "How far it gets", "thousand environment steps", "max x (m)")
    for a in (a1, a2):
        leg = a.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
        for t in leg.get_texts():
            t.set_color(FG)
    fig.suptitle("Three seeds per arm — the spread is the finding",
                 color=FG, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "hopper-learning-curves.png", dpi=150,
                facecolor=DEEP)
    plt.close(fig)
    print("  hopper-learning-curves.png")


def fig_seed_spread(plt) -> None:
    """The bar chart that kills the single-run conclusion."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_facecolor(DEEP)

    arms = [("seeing", SEEING, "sees height"), ("blind", BLIND, "blind")]
    width, base = 0.35, None
    for k, (arm, colour, label) in enumerate(arms):
        names = runs_for(arm)
        vals = [load(n)[1]["final"]["return"] for n in names]
        base = load(names[0])[1]["baseline_return"]
        xs = np.arange(len(vals)) + (k - 0.5) * width
        ax.bar(xs, vals, width, color=colour, label=label, edgecolor=GRID)
        m = float(np.mean(vals))
        ax.hlines(m, -0.6, len(vals) - 0.4, color=colour, ls=":", lw=1.6)
        ax.text(len(vals) - 0.42, m, f" mean {m:.0f}", color=colour,
                va="center", fontsize=9)

    if base is not None:
        ax.axhline(base, color=BASE, ls="--", lw=1.4,
                   label=f"random baseline ({base:.0f})")
    ax.set_xticks(range(len(runs_for("seeing"))))
    ax.set_xticklabels([f"seed {i}" for i in range(len(runs_for("seeing")))])
    style(ax, "Seed variance swamps the ablation",
          "", "final episode return")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "hopper-seed-spread.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  hopper-seed-spread.png")


def fig_height_sweep(plt) -> None:
    """Behaviour against obstacle height, for the best seed of each arm."""
    import numpy as np
    import torch

    sys.path.insert(0, str(HERE))
    from obstacle_env import (ACT_DIM, OBS_DIM, BOX_BACK_X, ObstacleHopper,
                              rollout)
    from ppo import ActorCritic, RunningNorm

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.patch.set_facecolor(DEEP)
    heights = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17]

    for arm, colour, label in (("seeing", SEEING, "sees height"),
                               ("blind", BLIND, "blind to height")):
        names = runs_for(arm)
        best = max(names, key=lambda n: load(n)[1]["final"]["return"])
        ck = torch.load(RUNS / best / "policy.pt", weights_only=False)
        net = ActorCritic(OBS_DIM, ACT_DIM)
        net.load_state_dict(ck["model"])
        norm = RunningNorm(OBS_DIM)
        norm.load_state_dict(ck["norm"])
        blind = arm == "blind"

        def policy(o):
            with torch.no_grad():
                t = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
                return net.distribution(t).mean.squeeze(0).numpy()

        means, lo, hi = [], [], []
        for h in heights:
            env = ObstacleHopper(fixed_height=h, blind_to_height=blind)
            xs = [rollout(env, policy, seed=900 + s)["max_x"] for s in range(8)]
            means.append(np.mean(xs))
            lo.append(np.min(xs))
            hi.append(np.max(xs))
        ax.plot(heights, means, color=colour, lw=2.0, marker="o", label=label)
        ax.fill_between(heights, lo, hi, color=colour, alpha=0.16)

    ax.axhline(BOX_BACK_X, color=ACCENT, ls="--", lw=1.4,
               label="far edge of the step")
    style(ax, "Distance travelled vs obstacle height  (best seed, 8 episodes)",
          "obstacle height (m)", "max x reached (m)")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "hopper-height-sweep.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  hopper-height-sweep.png")


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")                      # headless, always
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is missing. Run: uv sync", file=sys.stderr)
        return 1

    if not RUNS.exists() or not runs_for("seeing"):
        print("No runs found. Train first:\n"
              "    uv run train_ppo.py --name seeing_s0\n"
              "    uv run train_ppo.py --blind-to-height --name blind_s0",
              file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUT}")
    fig_learning(plt)
    fig_seed_spread(plt)
    fig_height_sweep(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
