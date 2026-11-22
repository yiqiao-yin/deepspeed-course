#!/usr/bin/env python3
"""
Every figure for this lab, drawn from measured numbers -- none hand-typed.

    uv run make_figures.py                      # all figures
    uv run make_figures.py --only reach         # one of them

Deliberately carries NO PEP 723 inline metadata, unlike the figure scripts
in scripts/. Those are standalone authoring tools; this one lives inside a
lab that is already a uv project, and it imports that lab's zeta_core to
draw Z(t) rather than reimplementing the formula in numpy. Inline metadata
would make uv build an isolated environment WITHOUT torch, and the
animation would fail on import -- which is exactly how it first failed.
`uv run` here resolves against the lab's own pyproject, which already
declares torch, matplotlib, numpy and pillow.

Four figures and one animation:

    reach.png        ULP against zero spacing -- where float32 runs out
    zeros.png        zeros found: float64 reference against the float32 arm
    penalty.png      THE HEADLINE: fp64 penalty, matmul vs this workload
    scaling.png      cost per evaluation, and where the GPU starts to pay
    fp32_drowns.gif  Z(t) in both precisions as t climbs

Run on the CPU; it draws, it does not measure. Carries PEP 723 inline
metadata and no pyproject of its own, deliberately: it is an authoring tool
rather than a lab, and a pyproject.toml would oblige an entry in
clawdeck.yaml, which would offer a learner "run the figure generator" as an
exercise. See CLAUDE.md on scripts/.

The measured numbers live in MEASURED below, each tagged with the card it
came from. A number and the configuration it was measured in are one
indivisible fact -- quoting the number alone is a different claim, not a
shorter one -- so nothing here is plotted without its card.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figures"
TAU = 2.0 * math.pi

# ---------------------------------------------------------------- house style
# Matches the Docusaurus site: its background is #000000 and it is dark-mode
# only (colorMode disableSwitch), so a figure on a light ground would glare.
BG = "#000000"
PANEL = "#08182a"
GRID = "#2d5a86"
FG = "#e8f1f8"
MUTED = "#8fb3cc"
C64 = "#63a3d0"        # float64 -- the reference arm
C32 = "#FF7043"        # float32 -- the arm under test
ACCENT = "#1e5f8f"
WARN = "#ffb300"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.alpha": 0.35,
    "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 130,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    "legend.labelcolor": FG,
})


# ------------------------------------------------------------ measured inputs
# Every figure below is drawn from these, and each row names the card. The
# matmul ratios are best-of-3; a single sample moved this measurement by 2.2x.
MEASURED = {
    "RTX 3080 Ti Laptop": {
        "matmul_fp32_tflops": 12.33,
        "matmul_fp64_tflops": 0.364,
        "matmul_ratio": 33.84,
        "zt_penalty_at_1e10": 2.39,
        "note": "consumer: fp64 at 1/64 rate",
    },
    "A100 80GB PCIe": {
        "matmul_fp32_tflops": 18.293,
        "matmul_fp64_tflops": 18.282,
        "matmul_ratio": 1.00,
        "zt_penalty_at_1e10": 1.47,
        "note": "datacentre: fp64 TENSOR cores",
    },
}

# Zero counts, bit-identical on both cards across all eight heights.
HEIGHTS = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
TERMS = [12, 39, 126, 398, 1261, 3989, 12615, 39894]
ZEROS_64 = [416, 402, 400, 399, 400, 400, 401, 400]
ZEROS_32 = [416, 402, 480, 595, 90, 5, 2, 0]


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _save(fig, name):
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out.relative_to(HERE)}")


# --------------------------------------------------------------------- figures
def fig_reach():
    """
    ULP against zero spacing. The crossing is the whole story.

    This is a REACH plot, not an error plot. A float's spacing grows with
    magnitude; the gap between zeros shrinks only logarithmically. Where the
    two lines cross, distinct grid points stop being distinct numbers.
    """
    t = np.logspace(3, 16, 400)
    spacing = TAU / np.log(t / TAU)
    ulp32 = np.array([np.spacing(np.float32(x)) for x in t])
    ulp64 = np.array([np.spacing(np.float64(x)) for x in t])

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.loglog(t, spacing, color=FG, lw=2.6, label="mean gap between zeros  $2\\pi/\\log(t/2\\pi)$")
    ax.loglog(t, ulp32, color=C32, lw=2.2, label="float32 ULP (spacing of representable $t$)")
    ax.loglog(t, ulp64, color=C64, lw=2.2, label="float64 ULP")

    # The crossings, found rather than hard-coded.
    for ulp, colour, label in ((ulp32, C32, "float32"), (ulp64, C64, "float64")):
        idx = np.argmax(ulp > spacing)
        if idx > 0:
            tc = t[idx]
            ax.axvline(tc, color=colour, ls="--", lw=1.2, alpha=0.6)
            ax.annotate(f"{label} runs out\n$t \\approx$ {tc:.0e}",
                        xy=(tc, spacing[idx]), xytext=(tc * 1.6, spacing[idx] * 60),
                        color=colour, fontsize=9.5,
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.1))

    ax.fill_between(t, spacing, ulp32, where=(ulp32 > spacing),
                    color=C32, alpha=0.12)
    _style(ax, "Precision is REACH, not an error bar",
           "height $t$ on the critical line", "size (log scale)")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.set_xlim(1e3, 1e16)
    ax.text(0.99, 0.03,
            "above the white line, one representable $t$ spans several zeros",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED, style="italic")
    _save(fig, "reach.png")


def fig_zeros():
    """
    Zeros found. float64 is the reference; float32 is the arm.

    The point is that the failure is NOT graceful. float32 is exact to 1e4,
    then OVER-counts (spurious sign changes from cancellation), and only
    then collapses. A lab that stopped at 1e4 would have published
    "float32 is fine".
    """
    x = np.arange(len(HEIGHTS))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.8),
                                  gridspec_kw={"height_ratios": [2.1, 1]},
                                  sharex=True)

    w = 0.38
    ax.bar(x - w / 2, ZEROS_64, w, color=C64, label="float64 (reference)")
    ax.bar(x + w / 2, ZEROS_32, w, color=C32, label="float32")
    ax.axhline(400, color=MUTED, ls=":", lw=1.2)
    # Placed right of centre: the tall float32 bars at 1e5-1e6 occupy the
    # left, and the bars from 1e7 on are short, so this band is clear.
    ax.text(4.25, 415, "400 expected per window", color=MUTED, fontsize=9)
    _style(ax, "float32 does not degrade -- it over-counts, then stops counting",
           None, "zeros found in a 400-zero window")
    ax.legend(loc="upper right", fontsize=10)

    diff = np.array(ZEROS_32) - np.array(ZEROS_64)
    cols = [WARN if d > 0 else (C32 if d < 0 else MUTED) for d in diff]
    ax2.bar(x, diff, 0.6, color=cols)
    ax2.axhline(0, color=FG, lw=1)
    for i, d in enumerate(diff):
        if d:
            ax2.annotate(f"{d:+d}", (i, d), ha="center",
                         va="bottom" if d > 0 else "top",
                         fontsize=9, color=WARN if d > 0 else C32)
    _style(ax2, None, "height $t$", "float32 $-$ float64")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$10^{{{int(np.log10(h))}}}$" for h in HEIGHTS])
    ax2.text(0.99, 0.9, "spurious zeros  →  missing zeros", transform=ax2.transAxes,
             ha="right", fontsize=9, color=MUTED, style="italic")
    fig.suptitle("", y=0.99)
    _save(fig, "zeros.png")


def fig_penalty():
    """
    THE HEADLINE. What fp64 costs, on a FLOP-bound workload and on this one.

    Two cards whose fp64 matmul capability differs by 34x. If this workload
    were FLOP-bound, its penalty would differ by 34x too. It differs by 1.6x.
    """
    cards = list(MEASURED)
    mm = [MEASURED[c]["matmul_ratio"] for c in cards]
    zt = [MEASURED[c]["zt_penalty_at_1e10"] for c in cards]

    x = np.arange(len(cards))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    b1 = ax.bar(x - w / 2, mm, w, color=ACCENT, label="matmul  (FLOP-bound control)")
    b2 = ax.bar(x + w / 2, zt, w, color=C32, label="Z(t) at $t=10^{10}$  (this lab)")

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}×",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color=FG)

    ax.axhline(2.0, color=FG, ls="--", lw=1.4, alpha=0.8)
    ax.text(-0.42, 2.18, "2× — float64 moves twice the BYTES",
            ha="left", fontsize=9.5, color=FG)

    ax.set_yscale("log")
    ax.set_ylim(0.62, 70)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{MEASURED[c]['note']}" for c in cards], fontsize=10)
    _style(ax, "A 34× difference in fp64 hardware buys a 1.6× difference here",
           None, "time(float64) / time(float32)   — lower is better")
    ax.legend(loc="upper center", fontsize=10)
    # Below the axes, not inside them: at log scale the 1.00x bar sits right
    # where an in-axes caption would land, and the two overlapped.
    fig.text(0.5, -0.045,
             "the control swings 33.84× → 1.00×;  this workload barely moves",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    _save(fig, "penalty.png")


def fig_scaling():
    """
    Cost per evaluation, and the height below which a GPU is not worth it.

    N(t) = floor(sqrt(t/2pi)) terms per point. The GPU wins once there is
    enough arithmetic to cover a kernel launch, and not before.
    """
    t = np.logspace(3, 12, 300)
    n = np.sqrt(t / TAU)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    ax.loglog(t, n, color=C64, lw=2.6)
    ax.scatter(HEIGHTS, TERMS, s=42, color=WARN, zorder=5,
               label="measured heights")
    ax.annotate(f"{TERMS[-1]:,} terms", xy=(HEIGHTS[-1], TERMS[-1]),
                xytext=(HEIGHTS[-1] / 60, TERMS[-1] * 1.5), color=WARN, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.1))
    _style(ax, "Work per point grows as $\\sqrt{t}$",
           "height $t$", "terms in the main sum  $N(t)$")
    ax.legend(loc="upper left", fontsize=9.5)

    # The float64 penalty against height, on both cards. Best-of-3.
    #
    # The GPU-over-CPU comparison that used to sit here has been REMOVED
    # rather than cleaned up. On a rented pod the single CPU core is
    # contended, and it produced 856x at t=1e3 and 4.21x at t=1e4 -- two
    # orders of magnitude apart on adjacent heights, which measures the
    # neighbours' load, not this workload. Publishing it with a caveat would
    # still have put an uninterpretable number on the page.
    pen_3080 = [0.96, 1.16, 0.97, 1.31, 1.71, 2.09, 2.26, 2.39]
    pen_a100 = [0.99, 1.17, 0.77, 1.15, 1.20, 1.24, 1.33, 1.47]

    ax2.semilogx(HEIGHTS, pen_3080, "o-", color=C32, lw=2.4, ms=6,
                 label="RTX 3080 Ti  (fp64 at 1/64 rate)")
    ax2.semilogx(HEIGHTS, pen_a100, "s-", color=C64, lw=2.4, ms=6,
                 label="A100 80GB  (fp64 tensor cores)")
    ax2.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax2.axhline(2.0, color=FG, ls="--", lw=1.3, alpha=0.75)
    ax2.text(1.4e3, 2.07, "2× — the byte ratio", fontsize=9, color=FG)
    ax2.set_ylim(0.6, 2.6)
    _style(ax2, "The float64 penalty grows with work, then flattens",
           "height $t$", "time(float64) / time(float32)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.text(0.97, 0.06,
             "below $10^6$ there is too little work to\nmeasure anything but launch overhead",
             transform=ax2.transAxes, ha="right", fontsize=8.5,
             color=MUTED, style="italic")
    _save(fig, "scaling.png")


def fig_gif():
    """
    Z(t) in both precisions as t climbs. The last frame is the finding.

    float32 is not a NOISY version of float64 at large t -- it is a flat
    line, because every grid point in the window rounds to the same
    representable t. Seeing that is what identified the mechanism as
    representability rather than accumulated rounding error.
    """
    from PIL import Image

    import sys
    sys.path.insert(0, str(HERE))
    import torch
    from zeta_core import mean_zero_spacing, z_values

    frames_t = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    images = []
    for t0 in frames_t:
        spacing = mean_zero_spacing(t0)
        grid = np.linspace(t0, t0 + 6 * spacing, 1400)
        z64 = z_values(grid, torch.float64, "cpu").numpy()
        z32 = z_values(grid, torch.float32, "cpu").numpy()

        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        off = grid - t0
        ax.axhline(0, color=MUTED, lw=1, alpha=0.6)
        ax.plot(off, z64, color=C64, lw=2.3, label="float64")
        ax.plot(off, z32, color=C32, lw=1.8, alpha=0.95, label="float32")
        ulp = float(np.spacing(np.float32(t0)))
        _style(ax, f"$Z(t)$ near $t = 10^{{{int(np.log10(t0))}}}$",
               f"$t - 10^{{{int(np.log10(t0))}}}$", "$Z(t)$")
        ax.legend(loc="upper right", fontsize=10)
        ax.text(0.015, 0.045,
                f"float32 ULP {ulp:g}   zero gap {spacing:.4f}"
                + ("   ← one representable $t$ spans the window"
                   if ulp > spacing else ""),
                transform=ax.transAxes, fontsize=9,
                color=C32 if ulp > spacing else MUTED)
        tmp = FIGS / "_frame.png"
        FIGS.mkdir(exist_ok=True)
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        images.append(Image.open(tmp).convert("P", palette=Image.ADAPTIVE).copy())

    out = FIGS / "fp32_drowns.gif"
    # Hold the final frame: it is the one worth reading.
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=[1100] * (len(images) - 1) + [3200], loop=0)
    (FIGS / "_frame.png").unlink(missing_ok=True)
    print(f"  wrote {out.relative_to(HERE)}  ({len(images)} frames)")


FIGURES = {"reach": fig_reach, "zeros": fig_zeros, "penalty": fig_penalty,
           "scaling": fig_scaling, "gif": fig_gif}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FIGURES), action="append",
                    help="build just this figure (repeatable)")
    args = ap.parse_args()

    wanted = args.only or list(FIGURES)
    print(f"\n  figures -> {FIGS}\n")
    for name in wanted:
        FIGURES[name]()
    print()


if __name__ == "__main__":
    main()
