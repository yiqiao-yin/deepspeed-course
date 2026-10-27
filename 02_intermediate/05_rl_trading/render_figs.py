#!/usr/bin/env python3
"""
Three figures: the equity curves, the drawdowns, and the cost curve.

    uv run render_figs.py

WHY EQUITY CURVES AND NOT A BAR CHART OF RETURNS
--------------------------------------------------
This lab's finding is about the PATH. Buy-and-hold ends higher and
gets there through a 37% hole; the reward-driven arms end lower
through a 21% one. A bar chart of annualised return shows the part
where buy-and-hold wins and hides the part where it does not,
which is exactly the wrong summary of this result.

The drawdown panel is the figure that carries the claim. Plotted
underwater -- distance below the running peak -- because that is
what an investor actually experiences and what the mean return
cannot show.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

import baselines as B                     # noqa: E402
import data as D                          # noqa: E402
import market as M                        # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "trading"

INK = "#eaf0f6"
GRID = "#223042"
BG = "#0b0f14"
COL = {"buyhold": "#e2b05c", "ppo": "#7ec6f8", "grpo": "#6cce94",
       "grpo_neutral": "#e28cc8", "forecast_rule": "#9aa6b2",
       "rsp_index": "#5a6676"}


def _style(ax, title, ylab):
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylab, color=INK, fontsize=9)
    ax.tick_params(colors="#8fa0b0", labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    for s in ax.spines.values():
        s.set_color(GRID)


def load(runs=HERE / "runs"):
    base, yte, dte, _fc = B.all_baselines()
    series = {k: v["daily"] for k, v in base.items()}
    for d in sorted(runs.glob("*/")):
        p, s = d / "positions.npy", d / "summary.json"
        if not (p.exists() and s.exists()):
            continue
        arm = d.name.rsplit("_s", 1)[0]
        series.setdefault(arm, []) if False else None
        key = arm
        pos = np.load(p)
        daily = M.book(pos, yte, M.DEFAULT_COST)
        series.setdefault(f"_{key}", []).append(daily)
    # median seed per arm
    for k in [k for k in series if k.startswith("_")]:
        arms = series.pop(k)
        anns = [M.stats(x)["ann"] for x in arms]
        series[k[1:]] = arms[int(np.argsort(anns)[len(anns) // 2])]
    return series, yte, dte


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    series, yte, dte = load()
    show = [k for k in ("buyhold", "ppo", "grpo", "grpo_neutral",
                        "forecast_rule") if k in series]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             facecolor=BG, height_ratios=[2, 1])
    for k in show:
        eq = np.cumprod(1 + series[k])
        axes[0].plot(dte, eq, lw=1.6, color=COL.get(k), label=k)
        peak = np.maximum.accumulate(eq)
        axes[1].fill_between(dte, (eq / peak - 1) * 100, 0,
                             color=COL.get(k), alpha=0.28, lw=0)
        axes[1].plot(dte, (eq / peak - 1) * 100, lw=1.0, color=COL.get(k))
    for nm, lo, hi in D.DRAWDOWNS:
        for ax in axes:
            ax.axvspan(np.datetime64(lo), np.datetime64(hi),
                       color="#ffffff", alpha=0.05, lw=0)
        axes[0].text(np.datetime64(lo), axes[0].get_ylim()[1] * 0.98, nm,
                     color="#8fa0b0", fontsize=7, va="top")
    _style(axes[0], "Growth of 1 unit, after 5 bps a side", "equity")
    _style(axes[1], "Underwater — distance below the running peak. "
                    "This panel is the finding.", "drawdown %")
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=8,
                   loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "rl-equity.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  rl-equity.png")

    # --- the cost curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    costs = np.array([0, 1, 2, 5, 10, 25]) / 1e4
    base2, _, _, fc = B.all_baselines()
    books = {"buyhold": np.ones_like(yte),
             "forecast_rule": B.rule(fc),
             "random": B.random_book(yte)}
    for k, pos in books.items():
        ys = [M.stats(M.book(pos, yte, c))["ann"] * 100 for c in costs]
        ax.plot(costs * 1e4, ys, "o-", lw=1.6, color=COL.get(k, "#9aa6b2"),
                label=k)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(5, color="#e26c6c", lw=1, ls="--")
    ax.text(5.2, ax.get_ylim()[1] * 0.9, "the lab's 5 bps",
            color="#e26c6c", fontsize=8)
    _style(ax, "Where an edge dies", "annualised %")
    ax.set_xlabel("cost, basis points per side", color=INK, fontsize=9)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rl-costs.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  rl-costs.png")

    # --- the bias figure -----------------------------------------------
    sec = D.load_prices("sectors")
    sur = D.load_prices("survivors")
    common = sec.index.intersection(sur.index)
    rs = sec.loc[common].pct_change().fillna(0)
    ru = sur.loc[common].pct_change().fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
    for lab, s, c in (("12 hand-picked survivors",
                       ru[D.SURVIVORS].mean(axis=1), "#e26c6c"),
                      ("9 sector ETFs (this lab)",
                       rs[D.SECTORS].mean(axis=1), "#6cce94"),
                      ("RSP, equal-weight index", rs["RSP"], "#9aa6b2")):
        ax.plot(common, np.cumprod(1 + s.to_numpy()), lw=1.7,
                color=c, label=lab)
    ax.set_yscale("log")
    _style(ax, "Survivorship bias is worth about ten points a year",
           "growth of 1, log scale")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=8,
              loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "rl-bias.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  rl-bias.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
