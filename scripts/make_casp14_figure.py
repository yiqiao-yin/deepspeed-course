#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pymol-open-source", "pillow", "numpy", "gemmi"]
# ///
"""
Recreate DeepMind's CASP14 figure from public data, and say what it is.

    uv run scripts/make_casp14_figure.py                 # PNGs + GIFs
    uv run scripts/make_casp14_figure.py --frames 48     # smoother
    uv run scripts/make_casp14_figure.py --bg dark       # for the book pages

The famous panel from AlphaFold2's CASP14 result shows two targets --
T1037 / 6vr4 at 90.7 GDT and T1049 / 6y4f at 93.3 GDT -- with the
experimental structure in green and the computational prediction in blue.

**Nothing here runs AlphaFold.** That is the point worth understanding: both
halves of that figure are public files.

    experimental   RCSB                     6VR4, 6Y4F
    prediction     CASP14 prediction archive, group 427 = AlphaFold2
                   predictioncenter.org/download_area/CASP14/predictions/regular/

So this script downloads AlphaFold2's *actual submitted coordinates* -- the
same atoms DeepMind rendered -- superimposes them on the deposited crystal
structures, and draws the result. Re-running ColabFold instead would give
*a* prediction, not *the* prediction: different MSAs, no CASP-condition
templates, different seeds, and a GDT near but not equal to the published
numbers.

Why this is a script and not a lab
-----------------------------------
It writes a documentation asset. It trains nothing and must not be bookable,
so it lives in `scripts/` with PEP 723 inline dependencies and no
`pyproject.toml` -- every directory with one of those must be registered in
`clawdeck.yaml`, and a figure generator has no business being a rentable GPU
lab. Same placement and same reasoning as `make_protein_animations.py`.

On the GDT numbers
------------------
90.7 and 93.3 are CASP's official scores, computed with Zemla's LGA, which
searches for the superposition maximising the score at each distance cutoff.
This script reports them as published and *separately* prints its own
GDT-TS from a single global Kabsch fit. Both are labelled, because quoting
the official number as though we had reproduced the official calculation
would be borrowed credibility.

**An earlier version of this script claimed its own score "can only be
lower" than LGA's, and the data disproved it:** it reported 95.3 for T1037
against the official 90.7. The cause was subset selection, not a better
algorithm. `super` runs refinement cycles that discard badly-fitting
residues -- correct for *placing* a model, wrong for *scoring* one -- so it
was averaging over 322 of 404 residues, having silently dropped the ones
that would have pulled the score down. The scoring alignment now runs with
`cycles=0`, which keeps every matched pair. A superposition metric computed
over a subset the metric itself chose is not a measurement; it is a
flattering summary.

With every pair kept, the scores land where they should: 89.2 against 90.7
for T1037, and 78.9 against 93.3 for T1049. T1049's larger gap is the
expected behaviour of a *single* superposition -- AlphaFold2 predicted a
long terminal segment the crystal never resolved, and one global fit has to
compromise between that segment and the well-ordered core, while LGA's
per-cutoff search does not. The gap is a property of the simpler method,
not a disagreement with CASP.

What cannot be reproduced exactly
----------------------------------
The camera. DeepMind's orientation was chosen by hand and is not recoverable
from the published image, so these renders use PyMOL's `orient` plus a
rotation sweep. The structures, the superposition and the colours are
faithful; the viewpoint is ours.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "casp14"
OUT_DIR = REPO / "docusaurus-docs" / "static" / "img" / "protein"

CASP_URL = ("https://predictioncenter.org/download_area/CASP14/"
            "predictions/regular/{target}.tar.gz")
RCSB_URL = "https://files.rcsb.org/download/{pdb}.cif"

# AlphaFold2 competed in CASP14 as group 427. Model 1 is the submitted
# first-ranked prediction, which is what the published figure shows.
AF2_GROUP = 427

TARGETS = [
    # target, pdb, chain, the official LGA GDT_TS, and the caption DeepMind used
    ("T1037", "6VR4", "A", 90.7, "RNA polymerase domain"),
    ("T1049", "6Y4F", "A", 93.3, "adhesin tip"),
]

# DeepMind's panel: a saturated green for experiment, a medium blue for the
# prediction. Matched by eye from the published figure.
GREEN = "0x21c45b"
BLUE = "0x3b4ff0"


# =============================================================================
# Data
# =============================================================================


def fetch(url: str, dest: Path) -> Path:
    """Download once, then reuse. These files do not change."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"    fetching {url}")
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def af2_model(target: str) -> Path:
    """AlphaFold2's submitted model 1 for a CASP14 target."""
    tgz = fetch(CASP_URL.format(target=target), CACHE / f"{target}.tar.gz")
    out = CACHE / target
    want = f"{target}TS{AF2_GROUP}_1"
    path = out / want
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as tf:
            member = next((m for m in tf.getmembers()
                           if Path(m.name).name == want), None)
            if member is None:
                raise RuntimeError(
                    f"{want} not in {tgz.name}. AlphaFold2 was group "
                    f"{AF2_GROUP} in CASP14; if the archive layout changed, "
                    "this is where to look.")
            member.name = want
            tf.extract(member, out)
    return path


# =============================================================================
# Our own GDT-TS, reported alongside the official one and never instead of it
# =============================================================================


def kabsch_gdt(pred: np.ndarray, true: np.ndarray) -> float:
    """
    GDT-TS from a single global superposition.

    Official CASP GDT (LGA) searches for the best superposition at each
    cutoff, so on the SAME residue set it reports at least this much. That
    only holds if the residue set is the same: see the module docstring for
    how scoring a refinement-filtered subset inflated this by five points.
    """
    pc, tc = pred - pred.mean(0), true - true.mean(0)
    U, _, Vt = np.linalg.svd(pc.T @ tc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    dist = np.linalg.norm(pc @ R.T - tc, axis=-1)
    return float(np.mean([(dist < c).mean() for c in (1, 2, 4, 8)]) * 100)


# =============================================================================
# Rendering
# =============================================================================


def render_target(cmd, target: str, pdb: str, chain: str, gdt: float,
                  caption: str, frames: int, bg: str, width: int,
                  height: int) -> tuple[Path, float, float]:
    """Superimpose, render a rotation sweep, return (dir, rmsd, our_gdt)."""
    pred_path = af2_model(target)
    exp_path = fetch(RCSB_URL.format(pdb=pdb), CACHE / f"{pdb}.cif")

    pred, exp = f"pred_{target}", f"exp_{target}"
    cmd.delete("all")
    cmd.load(str(pred_path), pred, format="pdb")
    cmd.load(str(exp_path), "full")
    cmd.remove("full and not polymer")
    cmd.create(exp, f"full and chain {chain}")
    cmd.delete("full")

    # Structure-based superposition. `super` is sequence-independent, which
    # matters because T1037 is a 404-residue DOMAIN inside a 2,166-residue
    # chain and the crystal has unresolved stretches the model predicts
    # anyway -- so matching by sequence finds nothing.
    aln = f"aln_{target}"
    r = cmd.super(pred, exp, object=aln)
    rmsd, n_atoms = r[0], r[1]

    # A SECOND alignment, with outlier rejection switched off, purely for
    # scoring. `super` above runs refinement cycles that discard residues
    # which fit badly -- excellent for placing the model, wrong for scoring
    # it. Using the refined correspondence scored only 322 of T1037's 404
    # residues and returned 95.3 against CASP's official 90.7: a number
    # inflated by throwing away exactly the residues that would have lowered
    # it. cycles=0 keeps every matched pair.
    aln_score = f"alnscore_{target}"
    cmd.super(pred, exp, object=aln_score, cycles=0, transform=0)

    # Our own score, on CA atoms PAIRED BY THE ALIGNMENT.
    #
    # This runs BEFORE the domain restriction below. Creating `exp_dom` and
    # deleting `exp` invalidates every reference the alignment object holds
    # to `exp`, so get_raw_alignment afterwards returns pairs naming an
    # object that no longer exists and the pair count silently drops to zero.
    #
    # Selecting "pred and name CA and aln" / "exp and name CA and aln"
    # separately returns two lists that are not guaranteed to be the same
    # length or in correspondence -- the first version did that and got a
    # length mismatch, so the score came out NaN. get_raw_alignment gives the
    # actual pairing, which is the only thing a superposition score can be
    # computed from.
    raw = cmd.get_raw_alignment(aln_score)
    atoms = {}
    for obj in (pred, exp):
        atoms[obj] = {a.index: (a.coord, a.name)
                      for a in cmd.get_model(obj).atom}
    P, Q = [], []
    for pair in raw:
        d = dict(pair)
        if pred in d and exp in d:
            cp, np_name = atoms[pred].get(d[pred], (None, None))
            cq, nq_name = atoms[exp].get(d[exp], (None, None))
            if cp and cq and np_name == "CA" and nq_name == "CA":
                P.append(cp)
                Q.append(cq)
    our_gdt = (kabsch_gdt(np.asarray(P), np.asarray(Q))
               if len(P) >= 10 else float("nan"))
    n_pairs = len(P)

    # Show only the part of the experimental chain that actually corresponds
    # to the target. Without this, T1037's panel is one small domain floating
    # inside the rest of the polymerase.
    cmd.create(f"{exp}_dom", f"byres ({exp} and {aln})")
    cmd.delete(exp)
    exp = f"{exp}_dom"

    cmd.hide("everything")
    cmd.show_as("cartoon", pred)
    cmd.show_as("cartoon", exp)
    cmd.color(GREEN, exp)
    cmd.color(BLUE, pred)

    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 0)
    cmd.set("specular", 0.2)
    cmd.set("ambient", 0.28)
    cmd.set("direct", 0.55)
    if bg == "dark":
        cmd.bg_color("black")
        cmd.set("ray_opaque_background", 1)
    elif bg == "transparent":
        cmd.set("ray_opaque_background", 0)
    else:
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)

    cmd.orient(f"{pred} or {exp}")
    cmd.zoom(f"{pred} or {exp}", buffer=-1.0)

    frame_dir = CACHE / "frames" / target
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    step = 360.0 / frames
    for i in range(frames):
        cmd.png(str(frame_dir / f"{i:03d}.png"), width=width, height=height,
                dpi=150, ray=1)
        cmd.turn("y", step)
    return frame_dir, rmsd, our_gdt, n_pairs


def to_gif(frames, path: Path, duration: int = 90, colors: int = 96) -> Path:
    """
    Save an animated GIF with an adaptive palette.

    Ray-traced cartoons are flat-shaded, so they survive heavy quantisation
    and balloon without it: the two-panel sweep was 2.3 MB at full colour and
    is a third of that at 96, with no visible difference on a docs page.
    """
    from PIL import Image

    q = [f.quantize(colors=colors, method=Image.MEDIANCUT) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=duration,
              loop=0, optimize=True)
    return path


def label(img, lines, bg: str):
    """Caption a frame. PIL's default face, because no webfont is bundled."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    fg = (255, 255, 255) if bg == "dark" else (20, 28, 36)
    y = img.height - 13 * len(lines) - 8
    for ln in lines:
        w = d.textlength(ln)
        d.text(((img.width - w) / 2, y), ln, fill=fg)
        y += 13
    return img


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recreate DeepMind's CASP14 panel from public data."
    )
    p.add_argument("--frames", type=int, default=36)
    p.add_argument("--bg", choices=("light", "dark", "transparent"),
                   default="dark", help="light matches the published figure; "
                                        "dark suits the book pages")
    p.add_argument("--width", type=int, default=460)
    p.add_argument("--height", type=int, default=460)
    p.add_argument("--keep-pngs", action="store_true",
                   help="also copy the individual frames into the output dir")
    args, _ = p.parse_known_args()

    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("  RECREATING THE CASP14 PANEL FROM PUBLIC DATA")
    print("=" * 74)
    print("  experimental : RCSB")
    print(f"  prediction   : CASP14 archive, group {AF2_GROUP} = AlphaFold2")
    print("  (no AlphaFold is run -- these are the submitted coordinates)\n")

    panels, meta = [], []
    for target, pdb, chain, gdt, caption in TARGETS:
        print(f"  {target} / {pdb.lower()}  ({caption})")
        frame_dir, rmsd, our_gdt, n_pairs = render_target(
            cmd, target, pdb, chain, gdt, caption,
            args.frames, args.bg, args.width, args.height)
        print(f"    superposition RMSD {rmsd:.2f} A")
        print(f"    GDT-TS {gdt} (CASP official, LGA)"
              f"   |   {our_gdt:.1f} (ours, single Kabsch fit over "
              f"{n_pairs} paired CA, every pair kept)")

        frames = sorted(frame_dir.glob("*.png"))
        imgs = [label(Image.open(f).convert("RGB"),
                      [f"{target} / {pdb.lower()}", f"{gdt} GDT", caption],
                      args.bg)
                for f in frames]
        out = to_gif(imgs, OUT_DIR / f"casp14-{target.lower()}.gif")
        print(f"    wrote {out.relative_to(REPO)} "
              f"({out.stat().st_size / 1024:.0f} KB)")

        if args.keep_pngs:
            still_dir = OUT_DIR / f"casp14-{target.lower()}-frames"
            still_dir.mkdir(exist_ok=True)
            for i, im in enumerate(imgs):
                im.save(still_dir / f"{i:03d}.png")
            print(f"    wrote {len(imgs)} PNGs to "
                  f"{still_dir.relative_to(REPO)}")

        # A representative still, for pages that want a static image.
        imgs[0].save(OUT_DIR / f"casp14-{target.lower()}.png")

        # Four views, 90 degrees apart, as one sheet. A single still hides
        # whether the agreement holds all the way round; four shows it does.
        # Built from the RAW frames, not the captioned ones -- pasting the
        # captioned stills repeats "T1049 / 6y4f  93.3 GDT  adhesin tip" four
        # times in one image. Each view gets its rotation instead, and the
        # identification is written once at the bottom.
        k = max(1, len(frames) // 4)
        quad = [Image.open(frames[(j * k) % len(frames)]).convert("RGB")
                for j in range(4)]
        w, h = quad[0].width, quad[0].height
        sheet = Image.new("RGB", (w * 2, h * 2 + 26),
                          (0, 0, 0) if args.bg == "dark" else (255, 255, 255))
        for j, im in enumerate(quad):
            sheet.paste(im, ((j % 2) * w, (j // 2) * h))
        # ASCII separators: PIL's default bitmap face has no em-dash and
        # renders it as a tofu box.
        label(sheet, [f"{target} / {pdb.lower()}  |  {gdt} GDT  |  {caption}"
                      "  |  four views, 90 deg apart"], args.bg)
        for j in range(4):
            label_x = (j % 2) * w + w - 52
            label_y = (j // 2) * h + 8
            from PIL import ImageDraw
            ImageDraw.Draw(sheet).text(
                (label_x, label_y), f"{j * 90} deg",
                fill=(255, 255, 255) if args.bg == "dark" else (20, 28, 36))
        angles_path = OUT_DIR / f"casp14-{target.lower()}-angles.png"
        sheet.save(angles_path)
        print(f"    wrote {angles_path.relative_to(REPO)} "
              f"(4 views, {angles_path.stat().st_size / 1024:.0f} KB)")
        panels.append(imgs)
        meta.append((target, pdb, gdt, caption))

    # The two-panel composite, in DeepMind's layout.
    n = min(len(p_) for p_ in panels)
    combined = []
    for i in range(n):
        row = [p_[i] for p_ in panels]
        w = sum(im.width for im in row)
        h = max(im.height for im in row)
        sheet = Image.new("RGB", (w, h),
                          (0, 0, 0) if args.bg == "dark" else (255, 255, 255))
        x = 0
        for im in row:
            sheet.paste(im, (x, 0))
            x += im.width
        combined.append(sheet)
    out = to_gif(combined, OUT_DIR / "casp14-panel.gif")
    combined[0].save(OUT_DIR / "casp14-panel.png")
    print(f"\n  wrote {out.relative_to(REPO)} "
          f"({out.stat().st_size / 1024:.0f} KB)")
    print("\n  Green = experimental structure. Blue = AlphaFold2's submitted")
    print("  CASP14 prediction. The camera is ours; everything else is theirs.")
    print("=" * 74)


if __name__ == "__main__":
    main()
