#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "pillow", "pyarrow", "huggingface-hub", "torch"]
# ///
"""
Render the animated figures for the 06_protein_folding book pages.

    uv run scripts/make_protein_animations.py                    # ~2 min, CPU
    uv run scripts/make_protein_animations.py --quality high     # 2x the pixels
    uv run scripts/make_protein_animations.py --only trunk-refinement
    uv run scripts/make_protein_animations.py --format png       # stills

What a GPU actually buys here, stated plainly
----------------------------------------------
**Not the rendering.** matplotlib draws through Agg on the CPU, and no flag
makes that faster. `--quality high` doubles the resolution and the frame
count on any machine; it is simply slower, and on a laptop CPU the four
default figures go from about two minutes to about seven.

What a GPU buys is the ability to COMPUTE a far richer input in reasonable
time, and there are two places that is real:

1. `--only coevolution --quality high` sweeps mutual information over a
   **1024-sequence, 128-residue** alignment instead of 160 x 40. That is a
   [128, 128, 20, 20] joint distribution per frame; on `--device cuda` it is
   a few seconds, on CPU it is minutes.
2. `--only trunk-refinement` runs **the course's actual Evoformer** from
   `06_protein_folding/02_evoformer` and animates the predicted contact map
   sharpening as it trains. This is a real forward and backward pass per
   frame. It is the figure that demonstrates the repo is GPU-ready, because
   it is the repo's own model doing the work.

Everything still runs without a GPU. `trunk-refinement` drops to a smaller
protein and fewer steps on CPU and says so on the figure, so nobody mistakes
a reduced render for the real one.

Why this is a script and not a lab
-----------------------------------
It writes documentation assets. It trains nothing, needs no GPU, and must not
be bookable -- so it lives in `scripts/` with PEP 723 inline dependencies
rather than in a topic folder. That placement is load-bearing: **every
directory with a `pyproject.toml` must appear in `clawdeck.yaml`**, enforced
by `tests/test_clawdeck_manifest.py`, so a new project folder here would
either fail CI or put a figure-generator in front of learners as a rentable
GPU lab. `scripts/` has no `pyproject.toml` and is exactly where
`audit_readmes.py` and `check_contract.py` already live.

It **reads** one lab -- `trunk-refinement` imports `EvoformerStack` from
`02_evoformer` -- but modifies nothing. The four topics are verified and
working; a documentation tool is not a reason to edit them, and a figure that
animates the real model is worth more than one that animates a reimplementation
of it that could drift.

A note on "DNA"
---------------
AlphaFold2 -- and every lab in `06_protein_folding/` -- predicts **protein**
structure. The alphabet is 20 amino acids, the geometry is a polypeptide
backbone, and the coevolution signal comes from protein alignments.
AlphaFold3 did extend to nucleic acids and ligands, which is part of why it
gave up the SE(3) guarantee (see `04_structure_module`), but none of that is
implemented here. These figures show proteins.

Output
------
`docusaurus-docs/static/img/protein/*.gif`, sized for a docs page and
committed, because CI builds the site but does not run this script.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docusaurus-docs" / "static" / "img" / "protein"

# The house palette from docusaurus.config.js, so the figures match the
# diagrams on the same page. CONTRIBUTING.md publishes these five.
DEEP = "#08182a"
DARK = "#0a1f33"
BASE = "#16324f"
BRIGHT = "#1e5f8f"
STEEL = "#28527a"
LIGHT = "#9ec9e8"
ACCENT = "#ffb86b"        # one warm colour, used only for "the thing to notice"
FG = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": "#000000",      # custom.css sets the site to black
    "axes.facecolor": DEEP,
    "savefig.facecolor": "#000000",
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": LIGHT,
    "ytick.color": LIGHT,
    "axes.edgecolor": STEEL,
    "font.size": 9,
    "axes.titlesize": 10,
})


# Set by main(). `draft` is what the committed GIFs use; `high` doubles the
# pixels and the frame count for anyone rendering their own.
QUALITY = {"dpi": 90, "scale": 1.0, "frames": 1.0}
DEVICE = "cpu"


def q(n: int) -> int:
    """Scale a frame count by the quality setting."""
    return max(2, int(round(n * QUALITY["frames"])))


def figsize(w: float, h: float) -> tuple[float, float]:
    return (w * QUALITY["scale"], h * QUALITY["scale"])


def usable_device(preferred: str, probe) -> str:
    """
    Return `preferred` if a real forward+backward survives on it, else "cpu".

    Some boxes advertise CUDA and still cannot run every kernel. This one
    does: torch 2.11 routes one outer-product backward through a **Triton**
    kernel, Triton JIT-compiles a small C shim against `libcuda.so.1`, and
    the only `gcc` on PATH here is `zig cc`, whose libc headers collide. The
    GPU is fine; the toolchain is not.

    Probing with a two-step run costs under a second and is the difference
    between a clear message and a stack trace two minutes into rendering.
    The rule is the same one `02_evoformer` applies to
    DS4Sci_EvoformerAttention: an optional accelerator may change the speed,
    never the answer.
    """
    if preferred == "cpu":
        return "cpu"
    try:
        probe(preferred)
        return preferred
    except (NameError, AttributeError, TypeError, ImportError, IndexError):
        # A bug in the probe is not evidence about the device. Re-raise
        # rather than quietly degrading to CPU: the first version of this
        # swallowed a NameError in its own closure and reported "cuda cannot
        # run this figure", which is a fallback manufacturing confidence
        # about hardware it never actually tested.
        raise
    except Exception as exc:                                 # noqa: BLE001
        head = str(exc).strip().splitlines()[0][:90]
        print(f"  [device] {preferred} cannot run this figure's backward "
              f"pass here; falling back to CPU.")
        print(f"  [device] reason: {type(exc).__name__}: {head}")
        return "cpu"


def pick_device(requested: str) -> str:
    """Resolve --device, and say what was chosen rather than guessing."""
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        print("  [device] --device cuda requested but no CUDA device is "
              "visible; falling back to CPU.")
    return "cpu"


# =============================================================================
# Data: a real backbone if the Hub is reachable, a plausible one if not
# =============================================================================


def load_backbone(n_res: int = 96) -> tuple[np.ndarray, str]:
    """
    C-alpha trace of one real CATH chain, or a synthetic fold as a fallback.

    Returns ``(ca [L, 3], provenance)``. The provenance string is drawn on the
    figure: a reader should never have to guess whether they are looking at a
    real protein or something this script invented.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("ajiang2025/cath-4.3-backbone",
                               "data/validation.parquet", repo_type="dataset")
        tbl = pq.read_table(path, columns=["coords", "mask", "length", "name"])
        rows = tbl.to_pylist()
        for row in rows:
            L = int(row["length"])
            if L < n_res:
                continue
            m = np.asarray(row["mask"], dtype=bool)
            if not m[:n_res].all():
                continue
            c = np.asarray(row["coords"], dtype=np.float64).reshape(L, 4, 3)
            return c[:n_res, 1, :], f"CATH {row['name']}"
    except Exception:                                        # noqa: BLE001
        pass

    # Fallback: helices and strands joined at random angles. Same generator
    # idea as 04_structure_module, enough to look like a fold.
    rng = np.random.default_rng(0)
    pts, pos, k = [], np.zeros(3), 0
    while k < n_res:
        helix = rng.random() < 0.6
        seg = min(int(rng.integers(6, 16)), n_res - k)
        local = np.empty((seg, 3))
        for j in range(seg):
            if helix:
                a = math.radians(100.0) * j
                local[j] = [2.3 * math.cos(a), 2.3 * math.sin(a), 1.5 * j]
            else:
                local[j] = [0.8 * ((-1.0) ** j), 0.0, 3.3 * j]
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        s = local @ q.T + pos
        pts.append(s)
        pos = s[-1] + rng.normal(size=3) / np.linalg.norm(rng.normal(size=3)) * 3.8
        k += seg
    return np.concatenate(pts)[:n_res], "synthetic (Hub unreachable)"


def _save(anim, name: str, fps: int, fmt: str, fig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path, dpi=QUALITY["dpi"] + 20, bbox_inches="tight")
    else:
        path = OUT_DIR / f"{name}.gif"
        anim.save(path, writer=PillowWriter(fps=fps), dpi=QUALITY["dpi"])
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(REPO)}  ({kb:.0f} KB)")


# =============================================================================
# 1. A contact map IS a 3D structure, flattened
# =============================================================================


def anim_contact_map(fmt: str) -> None:
    """
    A backbone rotating beside the contact map it produces.

    This is what `02_evoformer` predicts, and the animation exists to make one
    point: the contact map is not a summary of the structure, it is very
    nearly the structure itself, written as a matrix. The off-diagonal blobs
    are the places the chain folds back and touches itself.
    """
    ca, prov = load_backbone(96)
    ca = ca - ca.mean(0)
    n = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    contacts = (d < 8.0).astype(float)
    np.fill_diagonal(contacts, 0.0)

    fig = plt.figure(figsize=figsize(7.2, 3.6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    ax3d.set_facecolor(DEEP)

    colours = plt.cm.viridis(np.linspace(0, 1, n))
    ax2d.imshow(contacts, cmap="bone", origin="lower", interpolation="nearest")
    ax2d.set_title("contact map  |i-j| and distance < 8 A", color=FG)
    ax2d.set_xlabel("residue j")
    ax2d.set_ylabel("residue i")
    for s in ax2d.spines.values():
        s.set_color(STEEL)

    lim = float(np.abs(ca).max()) * 1.05
    marker = ax2d.plot([], [], "o", color=ACCENT, ms=5)[0]

    def frame(t: int):
        ax3d.clear()
        ax3d.set_facecolor(DEEP)
        for i in range(n - 1):
            ax3d.plot(*ca[i:i + 2].T, color=colours[i], lw=2.4)
        # Highlight one residue and its partners, walking the chain.
        i = int(t / max(q(72), 1) * n) % n
        ax3d.scatter(*ca[i], color=ACCENT, s=42, depthshade=False)
        partners = np.where(contacts[i] > 0)[0]
        for j in partners:
            ax3d.plot(*np.stack([ca[i], ca[j]]).T, color=ACCENT,
                      lw=0.8, alpha=0.55)
        marker.set_data(partners, np.full(len(partners), i))

        ax3d.view_init(elev=18, azim=t * 5)
        ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
        # matplotlib's 3D axes reserve a lot of empty cube around the data;
        # without the zoom the molecule renders at about a third of the panel.
        ax3d.set_box_aspect((1, 1, 1), zoom=1.45)
        ax3d.set_axis_off()
        ax3d.set_title(f"backbone  ({prov})", color=FG)
        return ()

    anim = FuncAnimation(fig, frame, frames=q(72), interval=70, blit=False)
    fig.suptitle("the contact map is the structure, flattened",
                 color=FG, y=0.99)
    fig.tight_layout()
    _save(anim, "contact-map", 14, fmt, fig)


# =============================================================================
# 2. The cubic wall
# =============================================================================


def anim_memory_wall(fmt: str) -> None:
    """
    Pair representation (quadratic) against triangle attention logits (cubic),
    as the protein gets longer.

    AlphaFold2's real widths: c_z=128, 4 heads, bf16. The point of animating
    it rather than tabling it is that the gap does not just grow, it grows
    *faster* -- which is what makes it a wall and not a constant.
    """
    lengths = np.arange(64, 1025, 16)
    pair = lengths ** 2 * 128 * 2 / 1e6
    logits = lengths ** 3 * 4 * 2 / 1e6

    fig, (ax, axr) = plt.subplots(1, 2, figsize=figsize(7.2, 3.4),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    for a in (ax, axr):
        a.set_facecolor(DEEP)
        for s in a.spines.values():
            s.set_color(STEEL)

    ax.set_xlim(lengths[0], lengths[-1])
    ax.set_ylim(1, 60_000)          # headroom so the 24 GB line is ON the axes
    ax.set_yscale("log")
    ax.set_xlabel("protein length (residues)")
    ax.set_ylabel("activation, MB per block  (log scale)")
    ax.set_title("pair rep is N^2.  triangle logits are N^3.", color=FG)
    l_pair, = ax.plot([], [], color=LIGHT, lw=2.2, label="pair representation")
    l_log, = ax.plot([], [], color=ACCENT, lw=2.6,
                     label="triangle attention logits")
    ax.axhline(24_000, color=BRIGHT, ls="--", lw=1.1)
    # Precise on purpose: the curves are PER BLOCK and AlphaFold2 has 48 of
    # them, so "one block fits" is not the same as "the model fits". Saying
    # just "a 24 GB card" would invite exactly that confusion.
    # Two lines, right-aligned. The legend box is wider than it looks and
    # eats the left half of this strip; a single long line gets clipped by it
    # whichever way it is anchored.
    ax.text(lengths[0] + 15, 28_000,
            "24 GB — and this is ONE of 48 blocks",
            color=BRIGHT, fontsize=7.5, ha="left")
    # Below the axes, horizontal. Three attempts at placing it INSIDE all
    # collided with the 24 GB annotation -- the legend box is consistently
    # wider than it looks, and guessing at its extent in data coordinates is
    # not a method. Out of the axes, the whole plot area is free.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              facecolor=DARK, edgecolor=STEEL, labelcolor=FG, fontsize=8,
              frameon=False)

    axr.set_xlim(lengths[0], lengths[-1])
    axr.set_ylim(0, (logits / pair).max() * 1.1)
    axr.set_xlabel("protein length")
    axr.set_ylabel("logits / pair rep")
    axr.set_title("and the ratio keeps climbing", color=FG)
    l_ratio, = axr.plot([], [], color=ACCENT, lw=2.2)
    cap = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right",
                  color=FG, fontsize=8)

    def frame(t: int):
        k = max(2, int((t + 1) / max(q(60), 1) * len(lengths)))
        l_pair.set_data(lengths[:k], pair[:k])
        l_log.set_data(lengths[:k], logits[:k])
        l_ratio.set_data(lengths[:k], (logits / pair)[:k])
        cap.set_text(f"N = {lengths[k-1]}    "
                     f"pair {pair[k-1]:,.0f} MB    "
                     f"logits {logits[k-1]:,.0f} MB")
        return l_pair, l_log, l_ratio, cap

    anim = FuncAnimation(fig, frame, frames=q(60), interval=60, blit=False)
    fig.suptitle("ZeRO shards parameters. This is an activation.",
                 color=FG, y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(anim, "memory-wall", 15, fmt, fig)


# =============================================================================
# 3. SE(3): a guarantee versus a learned symmetry
# =============================================================================


def anim_equivariance(fmt: str) -> None:
    """
    Rotate the input. A correct structure module rotates its prediction by
    exactly the same amount.

    IPA does, to float precision, because it only ever compares points inside
    shared local frames. An AlphaFold3-style head using ordinary attention
    over coordinates does not -- it has to learn the symmetry, and learns it
    approximately. Here that is drawn as drift: the orange trace is where the
    prediction *should* be, the dashed one is where the learned head puts it.

    The drift magnitude (5.1e-02 of the molecule's size, falling to 7.8e-03
    with augmentation) is the measurement from `04_structure_module`; this
    figure exaggerates it x6 so it is visible at this size, and says so.
    """
    ca, _ = load_backbone(64)
    ca = ca - ca.mean(0)
    scale = np.linalg.norm(ca, axis=-1).mean()
    rng = np.random.default_rng(3)
    # A fixed, smooth error field: a learned symmetry fails consistently for
    # a given input, not randomly per frame.
    err_dir = rng.normal(size=ca.shape)
    err_dir /= np.linalg.norm(err_dir, axis=-1, keepdims=True)
    EXAGGERATION = 6.0
    drift = err_dir * scale * 5.075e-02 * EXAGGERATION

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(7.2, 3.6),
                                 subplot_kw={"projection": "3d"})
    lim = float(np.abs(ca).max()) * 1.15

    def rot(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def frame(t: int):
        th = t / max(q(48), 1) * 2 * math.pi
        R = rot(th)
        truth = ca @ R.T
        learned = truth + drift @ R.T * (0.5 + 0.5 * math.sin(th * 2))

        for ax, pred, title, col, note in (
            (a1, truth, "IPA  (AlphaFold2)", LIGHT,
             "equivariant BY CONSTRUCTION   error 1e-15"),
            (a2, learned, "attention on coordinates  (AF3-style)", ACCENT,
             "learned, approximate   error 5e-02"),
        ):
            ax.clear()
            ax.set_facecolor(DEEP)
            ax.plot(*truth.T, color=STEEL, lw=1.0, alpha=0.5)
            ax.plot(*pred.T, color=col, lw=2.2,
                    ls="-" if ax is a1 else "--")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1), zoom=1.75)
            ax.set_axis_off()
            ax.set_title(title, color=FG, fontsize=9)
            ax.text2D(0.5, 0.02, note, transform=ax.transAxes, ha="center",
                      color=col, fontsize=7.5)
            ax.view_init(elev=16, azim=25)
        return ()

    anim = FuncAnimation(fig, frame, frames=q(48), interval=70, blit=False)
    fig.suptitle("rotate the input: does the prediction follow exactly?",
                 color=FG, y=0.99)
    fig.text(0.5, 0.015, f"grey = where the prediction should be. "
             f"drift exaggerated {EXAGGERATION:.0f}x to be visible.",
             ha="center", color=LIGHT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(anim, "se3-equivariance", 14, fmt, fig)


# =============================================================================
# 4. Coevolution: why MSAs exist
# =============================================================================


def anim_coevolution(fmt: str) -> None:
    """
    An alignment filling in, and the contact map emerging from it.

    Residues that touch in 3D mutate together, because a change at one must be
    compensated at the other. Nothing else in AlphaFold's input says which
    residues touch. As the alignment deepens, the mutual-information matrix
    sharpens from noise into the planted contacts -- which is the whole reason
    depth matters and a single sequence is not enough.
    """
    rng = np.random.default_rng(1)
    # --quality high makes this genuinely heavy: a [128,128,20,20] joint per
    # frame. That is the part a GPU accelerates -- see --device.
    if QUALITY["scale"] > 1.0:
        n_res, n_seq, n_pairs = 128, 1024, 32
    else:
        n_res, n_seq, n_pairs = 40, 160, 10

    pairs = []
    used = set()
    while len(pairs) < n_pairs:
        i, j = sorted(rng.integers(0, n_res, 2))
        if j - i < 6 or i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((int(i), int(j), int(rng.integers(1, 20))))

    query = rng.integers(0, 20, n_res)
    msa = np.tile(query, (n_seq, 1))
    coupled = {p for i, j, _ in pairs for p in (i, j)}
    for s in range(1, n_seq):
        row = msa[s]
        for p in range(n_res):
            if p not in coupled and rng.random() < 0.5:
                row[p] = rng.integers(0, 20)
        for i, j, shift in pairs:
            if rng.random() < 0.5:
                row[i] = rng.integers(0, 20)
            row[j] = (row[i] + shift) % 20      # j follows i

    truth = np.zeros((n_res, n_res))
    for i, j, _ in pairs:
        truth[i, j] = truth[j, i] = 1.0

    import torch

    msa_t = torch.from_numpy(msa).to(DEVICE)

    def mi_upto(depth: int) -> np.ndarray:
        # torch rather than numpy so --device cuda is a real speed-up: the
        # einsum below is [n_res, n_res, 20, 20] per frame, which at
        # --quality high is 26M elements and the only genuinely hot loop in
        # this script.
        sub = msa_t[:depth]
        oh = torch.zeros(n_res, depth, 20, device=DEVICE)
        idx = torch.arange(depth, device=DEVICE)
        for p in range(n_res):
            oh[p, idx, sub[:, p]] = 1.0
        px = oh.mean(1)
        joint = torch.einsum("isa,jsb->ijab", oh, oh) / depth
        outer = px[:, None, :, None] * px[None, :, None, :]
        term = joint * (torch.log(joint) - torch.log(outer))
        mi = torch.where(joint > 0, term,
                         torch.zeros((), device=DEVICE)).sum(dim=(2, 3))
        mi.fill_diagonal_(0.0)
        mi = mi.cpu().numpy()
        # APC, as the field does -- raw MI is biased upward by column entropy.
        d = max(n_res - 1, 1)
        rm = mi.sum(1) / d
        tm = mi.sum() / (n_res * d)
        if tm > 1e-12:
            mi = mi - np.outer(rm, rm) / tm
        np.fill_diagonal(mi, 0.0)
        return mi

    fig, (am, ai, at) = plt.subplots(1, 3, figsize=figsize(8.4, 3.0))
    for a in (am, ai, at):
        a.set_facecolor(DEEP)
        for s in a.spines.values():
            s.set_color(STEEL)
    im_msa = am.imshow(np.zeros((n_seq, n_res)), cmap="viridis", vmin=0,
                       vmax=19, aspect="auto", interpolation="nearest")
    am.set_title("the alignment, filling in", color=FG)
    am.set_xlabel("residue"); am.set_ylabel("homolog")
    im_mi = ai.imshow(np.zeros((n_res, n_res)), cmap="magma", origin="lower",
                      interpolation="nearest")
    ai.set_title("mutual information (APC)", color=FG)
    at.imshow(truth, cmap="bone", origin="lower", interpolation="nearest")
    at.set_title("the contacts that were planted", color=FG)
    cap = fig.text(0.5, 0.012, "", ha="center", color=ACCENT, fontsize=9)

    depths = np.unique(np.clip(
        np.geomspace(2, n_seq, 36).astype(int), 2, n_seq))

    def frame(t: int):
        depth = int(depths[min(t, len(depths) - 1)])
        shown = np.full((n_seq, n_res), np.nan)
        shown[:depth] = msa[:depth]
        im_msa.set_data(shown)
        mi = mi_upto(depth)
        im_mi.set_data(mi)
        im_mi.set_clim(0, max(mi.max(), 1e-6))
        cap.set_text(f"depth = {depth} sequences")
        return im_msa, im_mi, cap

    anim = FuncAnimation(fig, frame, frames=len(depths), interval=110,
                         blit=False)
    fig.suptitle("coevolution: one sequence says nothing, a population says "
                 "where the contacts are", color=FG, y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(anim, "coevolution", 9, fmt, fig)




# =============================================================================
# 5. The real trunk, learning. The figure that needs a GPU.
# =============================================================================


def anim_trunk_refinement(fmt: str) -> None:
    """
    Run the course's actual Evoformer and watch the contact map sharpen.

    Every other figure here draws a quantity that was computed analytically or
    sampled. This one runs `EvoformerStack` from
    `06_protein_folding/02_evoformer` -- the same class the lab trains -- for a
    few hundred real forward and backward passes, capturing the predicted
    contact map as it goes.

    That makes it the honest demonstration of a GPU-ready repository: the work
    on the card is the repo's own model, not a rendering trick. On this
    machine's RTX 3080 Ti the default sweep is well under a minute; on CPU the
    same sweep takes long enough that the function drops to a smaller protein
    and says so **on the figure**, so a reduced render can never be mistaken
    for the real one.

    Nothing in `02_evoformer` is modified. The import is read-only, and a
    figure that animates the real model is worth more than one animating a
    copy that could drift from it.
    """
    import sys

    import torch

    sys.path.insert(0, str(REPO / "06_protein_folding" / "02_evoformer"))
    from evoformer import EvoformerConfig, EvoformerStack
    from synthetic_msa import SyntheticConfig, SyntheticContactDataset, eval_mask

    # Probe at the size this figure actually uses -- see usable_device.
    def _probe(dev: str) -> None:
        m = EvoformerStack(EvoformerConfig(n_blocks=2)).to(dev)
        msa = torch.randint(0, 20, (1, 96, 64), device=dev)
        m(msa).float().sum().backward()

    device = usable_device(DEVICE, _probe)
    gpu = device == "cuda"
    if gpu:
        n_res, n_seq, n_chains, steps, blocks = 64, 96, 128, 420, 2
    else:
        # Same code path, smaller problem -- and the figure says so.
        n_res, n_seq, n_chains, steps, blocks = 32, 32, 32, 150, 1

    note = (f"RTX-class GPU: {n_res} residues, depth {n_seq}, {steps} steps"
            if gpu else
            f"CPU fallback: {n_res} residues, depth {n_seq}, {steps} steps "
            f"— run with a GPU for the full sweep")

    torch.manual_seed(0)
    cfg = SyntheticConfig(n_res=n_res, n_seq=n_seq,
                          n_contacts=max(4, n_res // 2), coupling=1.0)
    data = SyntheticContactDataset(n_chains, cfg, seed=0)
    scored = torch.from_numpy(eval_mask(cfg)).float().to(device)

    model = EvoformerStack(EvoformerConfig(n_blocks=blocks)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

    # One fixed held-out chain, so every frame shows the same protein and the
    # animation is the model changing rather than the input changing.
    probe_msa, probe_true, _ = data[0]
    probe_msa = probe_msa.unsqueeze(0).to(device)
    probe_true = probe_true.numpy()

    n_frames = q(40)
    every = max(1, steps // n_frames)
    snaps, losses, step = [], [], 0

    model.train()
    while step < steps:
        for msa, contacts, mask in loader:
            if step >= steps:
                break
            msa = msa.to(device)
            contacts = contacts.to(device).float()
            mask = mask.to(device).float()
            logits = model(msa).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, contacts, weight=mask, reduction="sum"
            ) / mask.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % every == 0:
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(probe_msa).float())[0] * scored
                # Multiplied by `scored` on purpose. The near-diagonal band
                # (|i-j| < 6) is EXCLUDED from the loss -- residues adjacent
                # in sequence always touch, so scoring them would let a model
                # win by reading the index difference. The model is therefore
                # free to output anything there, and it does: the first
                # version of this figure showed a bright predicted diagonal
                # against a dark true one, which reads as "the model is
                # wrong" when it is actually "this region was never asked
                # for". Showing an unscored region as a prediction is a lie
                # by omission.
                snaps.append((step, p.cpu().numpy(), loss.item()))
                model.train()
            losses.append(loss.item())
            step += 1

    fig, (ap, at, al) = plt.subplots(
        1, 3, figsize=figsize(8.4, 3.0),
        gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for a in (ap, at, al):
        a.set_facecolor(DEEP)
        for sp in a.spines.values():
            sp.set_color(STEEL)

    im = ap.imshow(snaps[0][1], cmap="magma", origin="lower", vmin=0, vmax=1,
                   interpolation="nearest")
    ap.set_title("what the trunk predicts", color=FG)
    at.imshow(probe_true * scored.cpu().numpy(), cmap="bone", origin="lower",
              interpolation="nearest")
    at.set_title("the true contacts", color=FG)
    for a in (ap, at):
        a.set_xlabel("residue j", fontsize=8)
    ap.set_ylabel("residue i", fontsize=8)

    al.set_xlim(0, steps)
    al.set_ylim(0, max(losses[:20]) * 1.1)
    al.set_xlabel("optimizer step")
    al.set_ylabel("loss")
    al.set_title("and the loss falling", color=FG)
    l_loss, = al.plot([], [], color=ACCENT, lw=1.8)
    cap = fig.text(0.5, 0.015, "", ha="center", color=ACCENT, fontsize=9)

    def frame(t: int):
        st, pred, ls = snaps[min(t, len(snaps) - 1)]
        im.set_data(pred)
        al.set_ylim(0, max(losses[:20]) * 1.1)
        l_loss.set_data(range(st + 1), losses[:st + 1])
        cap.set_text(f"step {st}    loss {ls:.4f}    {note}")
        return im, l_loss, cap

    anim = FuncAnimation(fig, frame, frames=len(snaps), interval=110,
                         blit=False)
    fig.suptitle("the Evoformer learning a contact map, for real "
                 "(long-range pairs only)", color=FG, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(anim, "trunk-refinement", 8, fmt, fig)


# =============================================================================
# 6. Prediction on truth, in the CASP style -- with OUR model's numbers
# =============================================================================


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Optimal rotation aligning P onto Q, both centred. Kabsch 1976.

    Without this, comparing two structures measures their orientation rather
    than their geometry -- the same reason FAPE exists in
    `04_structure_module`.
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])          # reflections are a different molecule
    return Vt.T @ D @ U.T


def gdt_ts(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    """
    GDT-TS, the metric on the CASP figures, and the per-residue deviations.

    GDT_TS = mean over cutoffs {1, 2, 4, 8} A of the percentage of residues
    within that distance after superposition. 100 is perfect.

    **This uses a single global Kabsch superposition.** Official CASP GDT
    (LGA) searches for the superposition that maximises the score at each
    cutoff, which can only ever report a HIGHER number than this. So the
    figure under-reports rather than over-reports, which is the right
    direction for a metric a reader might quote.
    """
    pc = pred - pred.mean(0)
    tc = true - true.mean(0)
    R = kabsch(pc, tc)
    aligned = pc @ R.T
    d = np.linalg.norm(aligned - tc, axis=-1)
    score = float(np.mean([(d < c).mean() for c in (1.0, 2.0, 4.0, 8.0)]) * 100)
    return score, aligned


def _smooth(points: np.ndarray, per_seg: int = 8) -> np.ndarray:
    """Catmull-Rom through the CA trace, so the ribbon reads as a fold."""
    p = np.vstack([points[0], points, points[-1]])
    out = []
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i:i + 4]
        out.append(0.5 * ((2 * p1)
                          + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [points[-1][None]])


def anim_prediction_vs_truth(fmt: str) -> None:
    """
    Prediction superimposed on truth, rotating, with a real GDT-TS.

    This is the figure grammar of DeepMind's CASP14 panel -- green for the
    experimental structure, blue for the computational prediction, a GDT
    number underneath.

    **The content is ours, and that distinction is the whole point.** The
    CASP14 figure shows AlphaFold2 predicting a target it had never seen from
    sequence alone. This shows `04_structure_module` -- a ~50k-parameter
    structure module trained in this repository for a few hundred steps --
    refining a *noised* real backbone. Those are very different problems, and
    a figure that blurred them would be the kind of fabricated result this
    repository has rules against.

    What is honest about it: the structure is a real CATH chain, the
    superposition is a real Kabsch fit, and the GDT-TS is computed the real
    way (and under-reported, see `gdt_ts`). What it demonstrates is that the
    course's own model produces something you can superimpose and score --
    not that it competes with AlphaFold2, which it emphatically does not.
    """
    import sys

    import torch

    sys.path.insert(0, str(REPO / "06_protein_folding" / "04_structure_module"))
    from structure import (StructureConfig, StructureModule, fape_loss,
                           frames_from_backbone)

    # Probe at the REAL size. The first version used n_res=8 and passed
    # happily, then the actual run blew up at n_res=96 -- torch only routes
    # the outer-product backward through Triton above a size threshold, so a
    # small probe exercises a different code path and proves nothing. Same
    # failure in miniature as testing a lab on a card bigger than the one it
    # declares.
    # A real chain, with the three atoms a frame needs.
    n_res = 96

    def _probe(dev: str) -> None:
        m = StructureModule(StructureConfig(n_blocks=2), "ipa").to(dev)
        x = torch.randn(1, n_res, 3, device=dev)
        R, t = frames_from_backbone(x, x + 1, x + 2)
        s0 = torch.zeros(1, n_res, m.cfg.c_s, device=dev)
        z0 = torch.zeros(1, n_res, n_res, m.cfg.c_z, device=dev)
        Rp, tp = m(s0, z0, R, t)
        fape_loss(Rp, tp, R, t).backward()

    device = usable_device(DEVICE, _probe)
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("ajiang2025/cath-4.3-backbone",
                               "data/validation.parquet", repo_type="dataset")
        rows = pq.read_table(path).to_pylist()
        row = next(r for r in rows
                   if int(r["length"]) >= n_res
                   and np.asarray(r["mask"], dtype=bool)[:n_res].all())
        c = np.asarray(row["coords"], dtype=np.float32).reshape(
            int(row["length"]), 4, 3)[:n_res]
        name = str(row["name"])
    except Exception:                                        # noqa: BLE001
        ca, _ = load_backbone(n_res)
        c = np.stack([ca - 1.2, ca, ca + 1.0], axis=1).astype(np.float32)
        c = np.concatenate([c, c[:, :1]], axis=1)
        name = "synthetic (Hub unreachable)"

    nx = torch.from_numpy(c[:, 0])[None].to(device)
    ca_t = torch.from_numpy(c[:, 1])[None].to(device)
    cx = torch.from_numpy(c[:, 2])[None].to(device)

    cfg = StructureConfig(n_blocks=2)
    torch.manual_seed(0)
    model = StructureModule(cfg, "ipa").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # 300 either way. The CPU fallback renders this in ~10 s, so there is no
    # reason to show a reader a worse-converged structure just because their
    # toolchain cannot build a Triton kernel.
    steps = 300
    noise = 2.0
    R_true, t_true = frames_from_backbone(nx, ca_t, cx)

    def blank():
        return (torch.zeros(1, n_res, cfg.c_s, device=device),
                torch.zeros(1, n_res, n_res, cfg.c_z, device=device))

    g = torch.Generator(device="cpu").manual_seed(7)
    fixed = (torch.randn(ca_t.shape, generator=g) * noise).to(device)
    R_in, t_in = frames_from_backbone(nx + fixed, ca_t + fixed, cx + fixed)

    snaps = []
    n_frames = q(36)
    every = max(1, steps // n_frames)
    for step in range(steps + 1):
        if step % every == 0:
            model.eval()
            with torch.no_grad():
                _, tp = model(*blank(), R_in, t_in)
            pred = tp[0].cpu().numpy().astype(np.float64)
            score, aligned = gdt_ts(pred, c[:, 1].astype(np.float64))
            snaps.append((step, aligned, score))
            model.train()
        if step == steps:
            break
        R_p, t_p = model(*blank(), R_in, t_in)
        loss = fape_loss(R_p, t_p, R_true, t_true)
        opt.zero_grad()
        loss.backward()
        opt.step()

    truth_c = c[:, 1].astype(np.float64)
    truth_c = truth_c - truth_c.mean(0)
    truth_s = _smooth(truth_c)
    lim = float(np.abs(truth_c).max()) * 1.05

    fig = plt.figure(figsize=figsize(5.0, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    GREEN, BLUE = "#21d07a", "#4f6bed"

    head = fig.text(0.5, 0.955, "", ha="center", color=FG, fontsize=11)

    def frame(t: int):
        step, pred, score = snaps[min(t, len(snaps) - 1)]
        ax.clear()
        ax.set_facecolor(DEEP)
        ax.plot(*truth_s.T, color=GREEN, lw=2.6, solid_capstyle="round")
        ax.plot(*_smooth(pred).T, color=BLUE, lw=2.6, solid_capstyle="round",
                alpha=0.9)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.75)
        ax.set_axis_off()
        ax.view_init(elev=14, azim=t * (360 / max(len(snaps), 1)))
        # As a figure text, not an axes title: a 3D axes title sits above
        # the cube and gets clipped by tight_layout.
        head.set_text(f"CATH {name}   —   {score:.1f} GDT-TS   (step {step})")
        return ()

    anim = FuncAnimation(fig, frame, frames=len(snaps), interval=110,
                         blit=False)
    fig.text(0.5, 0.095, "●  experimental structure", ha="center",
             color=GREEN, fontsize=9)
    fig.text(0.5, 0.055, "●  this repo's structure module", ha="center",
             color=BLUE, fontsize=9)
    fig.text(0.5, 0.012,
             "refining a noised backbone — NOT AlphaFold2 folding from "
             "sequence", ha="center", color=LIGHT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    _save(anim, "prediction-vs-truth", 8, fmt, fig)


FIGURES = {
    "contact-map": anim_contact_map,
    "memory-wall": anim_memory_wall,
    "se3-equivariance": anim_equivariance,
    "coevolution": anim_coevolution,
    "trunk-refinement": anim_trunk_refinement,
    "prediction-vs-truth": anim_prediction_vs_truth,
}

# Rendered only on request: these two TRAIN a model, so they are far slower
# than the analytic figures and they are the ones that genuinely want a GPU.
DEFAULT_FIGURES = ("contact-map", "memory-wall", "se3-equivariance",
                   "coevolution")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render the 06_protein_folding book figures."
    )
    p.add_argument("--only", choices=sorted(FIGURES), action="append",
                   help="render one figure (repeatable). Default is the four "
                        "analytic ones; trunk-refinement must be asked for "
                        "because it trains a model.")
    p.add_argument("--format", choices=("gif", "png"), default="gif",
                   help="gif animates; png saves the final frame")
    p.add_argument("--quality", choices=("draft", "high"), default="draft",
                   help="high doubles the pixels and the frame count, and "
                        "makes coevolution a much larger alignment. Slower "
                        "everywhere -- rendering is CPU-bound on any machine.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="where the COMPUTE runs. Rendering is always CPU; "
                        "this moves the mutual-information sweep and the "
                        "Evoformer training onto the GPU.")
    args, _ = p.parse_known_args()

    global DEVICE
    if args.quality == "high":
        QUALITY.update(dpi=160, scale=1.35, frames=1.6)
    DEVICE = pick_device(args.device)

    wanted = args.only or list(DEFAULT_FIGURES)
    print("=" * 70)
    print("  RENDERING 06_protein_folding FIGURES")
    print("=" * 70)
    print(f"  output : {OUT_DIR.relative_to(REPO)}")
    print(f"  format : {args.format}   quality: {args.quality}")
    print(f"  compute: {DEVICE}   (rendering is CPU-bound either way)\n")
    for name in wanted:
        print(f"  {name} ...")
        FIGURES[name](args.format)
    print("\n  Done. These are committed -- CI builds the site but does not")
    print("  run this script.")
    print("=" * 70)


if __name__ == "__main__":
    main()
