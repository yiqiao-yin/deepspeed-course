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
PADDED = "#c9a227"        # the width control

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


# Every figure reads nav5_* and nothing else.
#
# The nav_, nav2_, nav3_ and nav4_ runs were all trained in a BROKEN
# COORDINATE FRAME: `rootx`/`rooty` are slide joints and the torso body
# was declared at pos="(start)", so reset() writing the start into qpos
# spawned the robot at TWICE its start coordinates -- frequently off the
# height field entirely. Their arrival rates, their terrain probes and
# their published comparison are all measurements of a different world.
# They stay on disk as the record of what was withdrawn; they must never
# re-enter a figure.
SWEEP = "nav5_"


def evaluations(mode: str | None = None) -> list[dict]:
    """The nine corrected-frame seeds per arm, from evaluate.py."""
    rows = []
    for f in ("evaluation5.json", "evaluation_padded.json"):
        p = RUNS / f
        if p.exists():
            rows += json.loads(p.read_text())
    # de-duplicate: a run scored in both files keeps the later row
    seen = {}
    for r in rows:
        if r["run"].startswith(SWEEP):
            seen[r["run"]] = r
    out = sorted(seen.values(), key=lambda r: r["run"])
    return [r for r in out if mode is None or r["mode"] == mode]


def fig_arrival(plt) -> None:
    """
    Arrival, paired by seed, both sweeps.

    Paired because the two arms were trained from the same seed and
    scored on the same episode seeds, so the comparison that means
    something is within-pair, not between-pools.
    """
    import numpy as np

    b = np.array([r["arrived"] for r in evaluations("blind")])
    p = np.array([r["arrived"] for r in evaluations("privileged")])
    n = min(len(b), len(p))
    b, p = b[:n], p[:n]

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
    ax.set_xticklabels([f"s{i}" for i in range(n)], fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.axhline(0.10, color="#8ea3b5", ls=":", lw=1.2,
               label="10% — below this the run did not learn the task")
    style(ax, f"Arrival, 120 episodes per bar — privileged ahead on "
              f"{int((p>b).sum())} of {n} seeds, and COLLAPSED on "
              f"{int((p<0.10).sum())}", "", "episodes reaching B")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-arrival.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-arrival.png")


def fig_efficiency(plt) -> None:
    """
    Path efficiency, paired by seed, from the CORRECTED denominator.

    This figure has been wrong twice and the reasons are different, so
    both are worth stating. First the odometer kept running after the
    robot reached B, so a policy that walked a 97%-optimal route and
    then milled around the goal for 1300 steps was published at 49%.
    Then the denominator itself: `solve()` is a 4-connected search that
    cannot move diagonally, so it overstated the optimum by up to
    sqrt(2) and three episodes scored over 100% -- which `min(..., 1.0)`
    quietly rewrote to exactly 100%.

    What is plotted is `geodesic / distance walked to B`, with no clamp.
    A bar over 1.0 would mean the denominator is wrong again.
    """
    import numpy as np

    b = evaluations("blind")
    p = evaluations("privileged")
    n = min(len(b), len(p))
    # Only seeds where BOTH arms actually arrived often enough for a
    # route to average. An efficiency computed over one lucky episode is
    # not a measurement, and the collapsed privileged seeds contribute
    # exactly that.
    keep = [i for i in range(n)
            if b[i]["arrived"] >= 0.10 and p[i]["arrived"] >= 0.10]
    if not keep:
        print("  (no seed pair where both arms arrived — skipping)")
        return
    be = np.array([b[i]["efficiency"] for i in keep])
    pe = np.array([p[i]["efficiency"] for i in keep])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.patch.set_facecolor(DEEP)
    xs = np.arange(len(keep))
    ax.bar(xs - 0.2, be, 0.4, color=BLIND, edgecolor=GRID, label="blind")
    ax.bar(xs + 0.2, pe, 0.4, color=PRIV, edgecolor=GRID, label="privileged")
    ax.axhline(1.0, color=ORACLE, ls="--", lw=1.4,
               label="the geodesic (efficiency 1.0)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in keep], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, f"Path efficiency on the {len(keep)} seed pairs where both "
              f"arms arrived", "", "geodesic / distance walked to B")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "nav-efficiency.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-efficiency.png")


def fig_control(plt) -> None:
    """
    The finding, and the control that scopes it.

    The privileged arm is bimodal: it matches blind when it trains and
    sits on the floor when it does not. Two explanations fit equally
    well -- six extra DIMENSIONS destabilise PPO at this scale, or those
    particular FEATURES are harmful -- and they are told apart by
    widening the blind observation with six constant zeros. Same input
    width, zero information.
    """
    import numpy as np

    arms = [("blind", BLIND, evaluations("blind")),
            ("privileged\n(+6 terrain features)", PRIV,
             evaluations("privileged")),
            ("padded control\n(+6 constant zeros)", PADDED,
             evaluations("padded"))]
    arms = [(lab, col, rows) for lab, col, rows in arms if rows]
    if len(arms) < 3:
        print("  (padded control not scored yet — skipping nav-control.png)")
        return

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(DEEP)

    # Left: every seed as a dot, so bimodality is visible rather than
    # averaged away. A mean over a bimodal arm describes no run in it.
    for k, (lab, col, rows) in enumerate(arms):
        v = np.array([r["arrived"] for r in rows])
        jit = np.linspace(-0.13, 0.13, len(v))
        ax.scatter(np.full(len(v), k) + jit, v, s=74, color=col,
                   edgecolor=GRID, zorder=3, linewidth=0.8)
        ax.plot([k - 0.26, k + 0.26], [v.mean()] * 2, color=FG, lw=2.0,
                zorder=4)
    ax.axhspan(0, 0.10, color="#3a1620", zorder=0)
    ax.annotate("did not learn the task", (-0.42, 0.045), ha="left",
                va="center", color="#c98a8a", fontsize=9, zorder=5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Every seed, 120 episodes each (bar = mean)", "",
          "episodes reaching B")

    # Right: the thing the arms actually differ in.
    for k, (lab, col, rows) in enumerate(arms):
        v = np.array([r["arrived"] for r in rows])
        frac = float((v < 0.10).mean())
        ax2.bar(k, frac, 0.56, color=col, edgecolor=GRID)
        ax2.annotate(f"{int((v < 0.10).sum())}/{len(v)}",
                     (k, frac + 0.025), ha="center", color=FG, fontsize=10)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels([a[0].split("\n")[0] for a in arms], fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax2, "Seeds that collapsed", "", "fraction under 10%")

    fig.tight_layout()
    fig.savefig(OUT / "nav-control.png", dpi=150, facecolor=DEEP)
    plt.close(fig)
    print("  nav-control.png")


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
    for f in sorted(RUNS.glob(f"{SWEEP}*_s*/curve.csv")):
        nm = f.parent.name
        mode = ("blind" if "blind" in nm
                else "padded" if "padded" in nm else "privileged")
        col = {"blind": BLIND, "privileged": PRIV, "padded": PADDED}[mode]
        rows = list(csv.DictReader(f.open()))
        ax.plot([float(r["step"])/1e6 for r in rows],
                [float(r["arrived"]) for r in rows],
                color=col, lw=1.2, alpha=0.75,
                label=mode if mode not in seen else None)
        seen.add(mode)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Training curves — the privileged arm is BIMODAL: half its "
              "seeds track blind, half never leave the floor",
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
    fig_control(plt)
    fig_curves(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
