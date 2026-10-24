#!/usr/bin/env python3
"""
Figures for the dynamics lab: the scaling curve, and the drift.

    uv run render_figs.py --scaling
    uv run render_figs.py --drift --size small
    uv run render_figs.py --all

TWO FIGURES, AND THE SECOND IS THE ONE THAT ARGUES
---------------------------------------------------
`--scaling` draws 50-step error against parameter count, with the
null model as a dashed line. It is flat, which is the lab's result.

`--drift` is the figure that shows WHY. It runs a model open-loop
from a real starting state and animates its prediction against the
truth, channel by channel, as the horizon grows. One-step error is
tiny and the traces sit on top of each other; by fifty steps the
model's version of the vehicle is somewhere else entirely. The
number in the table is that picture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"

BG = (11, 15, 20)
PANEL = (16, 22, 30)
GRID = (32, 42, 54)
INK = (236, 242, 248)
DIM = (150, 170, 186)
FAINT = (104, 126, 146)
GOOD = (108, 206, 148)
WARN = (226, 176, 92)
BAD = (228, 108, 108)
JET = (126, 198, 248)
SERIES = [(126, 198, 248), (108, 206, 148), (226, 140, 200),
          (240, 200, 120), (170, 150, 240)]

# Channels worth watching. The observation is 41 numbers and most of
# them are the previous action; these are the ones a person would
# recognise as "where is the vehicle and what is it doing".
CHANNELS = [(6, "uprightness", ""), (15, "altitude", " (norm)"),
            (12, "vertical speed", " (norm)"), (7, "roll rate", " (norm)")]


def save_gif(frames, path, ms=70, colors=128, hold_ms=1200):
    """
    Write a GIF that actually animates.

    optimize=True MERGES consecutive identical frames and rolls their
    durations together. A progressive chart reveal over 11 data
    points produces ~11 distinct images in 70 requested frames, so
    the first version of this collapsed 92 frames into 7 at 1610 ms
    each -- a file that loads, reports a frame count, and looks like
    a still picture. Verified by reading the saved file back rather
    than trusting the frame list handed to save().
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=colors,
                     dither=Image.Dither.NONE) for f in frames]
    # A pause at the end is ONE frame with a long duration, not a
    # run of duplicates. Duplicates are exactly what the optimiser
    # merges, and then the frame count no longer describes the file.
    durations = [ms] * (len(pil) - 1) + [hold_ms]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=durations, loop=0, optimize=False)
    back = Image.open(path)
    kept = getattr(back, "n_frames", 1)
    if kept != len(frames):
        raise RuntimeError(
            f"{path.name}: asked for {len(frames)} frames, the file has "
            f"{kept}. Consecutive frames were identical enough to be "
            f"merged -- the animation would look frozen. Verified by "
            f"reading the file back, not by trusting the frame list.")
    print(f"  {path.name}  ({kept} frames @ {ms} ms + {hold_ms} ms hold, "
          f"{path.stat().st_size / 1024 / 1024:.2f} MB)")


def load_runs(lr_filter=1e-3):
    rows = []
    for d in sorted((HERE / "runs").glob("*/")):
        s = d / "summary.json"
        if not s.exists():
            continue
        r = json.loads(s.read_text())
        if "params" not in r:
            continue
        rows.append(r)
    return rows


def fig_scaling(width=880, height=470, frames=70, hold=0):
    """50-step error against parameter count, drawn progressively."""
    from PIL import Image, ImageDraw

    runs = [r for r in load_runs() if abs(r["lr"] - 1e-3) < 1e-12]
    runs.sort(key=lambda r: r["params"])
    if not runs:
        print("  no runs to plot")
        return
    NULL_HOLD = 1.6334

    # LOG on BOTH axes. The errors span 1.1 to 49, so a linear y
    # squashes every MLP rung onto the floor and the figure shows
    # nothing but the GRU blowing up -- which is the side story.
    xs = [math.log10(r["params"]) for r in runs]
    ys = [math.log10(r["rollout_err"]) for r in runs]
    x0, x1 = min(xs) - 0.25, max(xs) + 0.25
    ylo = min(min(ys), math.log10(NULL_HOLD)) - 0.12
    yhi = max(max(ys), math.log10(NULL_HOLD)) + 0.12

    L, R, T, B = 86, width - 170, 70, height - 54
    out = []
    for fi in range(frames + hold):
        t = min(1.0, (fi + 1) / frames)
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([L, T, R, B], fill=PANEL)
        d.text((14, 14), "Does a bigger dynamics model predict better?",
               fill=INK)
        d.text((14, 32), "50-step open-loop error, log scale "
                         "(lower is better)", fill=DIM)

        def px(lx, v):
            ly = math.log10(max(v, 1e-6))
            return (L + (R - L) * (lx - x0) / (x1 - x0),
                    B - (B - T) * (ly - ylo) / (yhi - ylo))

        for v in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            if not (10 ** ylo <= v <= 10 ** yhi):
                continue
            _x, y = px(x0, v)
            d.line([L, y, R, y], fill=GRID)
            lab = f"{v:.0f}"
            d.text((L - 12 - 6 * len(lab), y - 6), lab, fill=FAINT)
        for e in range(int(math.floor(x0)), int(math.ceil(x1)) + 1):
            x, _y = px(e, 0)
            if x < L or x > R:
                continue
            d.line([x, T, x, B], fill=GRID)
            d.text((x - 14, B + 8), f"1e{e}", fill=FAINT)
        d.text(((L + R) // 2 - 40, B + 28), "parameters", fill=DIM)

        _x, yn = px(x0, NULL_HOLD)
        for xx in range(L, R, 10):
            d.line([xx, yn, xx + 5, yn], fill=WARN)
        d.text((L + 10, yn - 16), "predict no change  1.63", fill=WARN)

        # Interpolate ALONG the polyline rather than stepping point to
        # point. With 6 rungs a point-wise reveal changes the picture
        # six times in seventy frames and the rest are duplicates,
        # which is what collapsed the first version to a still.
        for kind, col in (("mlp", SERIES[0]), ("gru", SERIES[1])):
            ks = [r for r in runs if r["kind"] == kind]
            pts = [px(math.log10(r["params"]), r["rollout_err"]) for r in ks]
            if len(pts) < 2:
                continue
            span = (len(pts) - 1) * t
            whole = int(span)
            frac = span - whole
            drawn = pts[:whole + 1]
            if whole < len(pts) - 1:
                x0_, y0_ = pts[whole]
                x1_, y1_ = pts[whole + 1]
                drawn = drawn + [(x0_ + (x1_ - x0_) * frac,
                                  y0_ + (y1_ - y0_) * frac)]
            if len(drawn) > 1:
                d.line(drawn, fill=col, width=3, joint="curve")
            for (cx, cy), r in zip(pts[:whole + 1], ks):
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=col)
            hx, hy = drawn[-1]
            d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=INK)
            d.text((R + 10, hy - 6), kind, fill=col)

        best = min(runs, key=lambda r: r["rollout_err"])
        if t >= 1.0:
            bx, by = px(math.log10(best["params"]), best["rollout_err"])
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], outline=GOOD, width=2)
            d.text((bx - 30, by - 26),
                   f"best: {best['params']:,}", fill=GOOD)
        out.append(img)
    save_gif(out, OUT / "dyn-scaling.gif")


def fig_drift(kind="mlp", size="small", horizon=50, width=880, height=470):
    """The model's prediction against the truth, as the horizon grows."""
    import torch
    from PIL import Image, ImageDraw

    import dynamics as DY

    ck = torch.load(HERE / "runs" / f"{kind}_{size}" / "model.pt",
                    map_location="cpu", weights_only=False)
    model = DY.build(ck["kind"], ck["size"], ck["obs_dim"], ck["act_dim"])
    model.load_state_dict(ck["model"])
    model.eval()

    va = np.load(HERE / "data/val.npz", allow_pickle=True)
    starts, ends = va["ep_start"], va["ep_end"]
    pick = next(i for i, (s, e) in enumerate(zip(starts, ends))
                if e - s > horizon + 5)
    s0 = int(starts[pick])

    truth = va["obs"][s0:s0 + horizon + 1]
    acts = va["act"][s0:s0 + horizon + 1]
    pred = [truth[0]]
    o = torch.as_tensor(truth[0:1])
    st = None
    with torch.no_grad():
        for h in range(horizon):
            o, st = model(o, torch.as_tensor(acts[h:h + 1]), st)
            pred.append(o.numpy()[0])
    pred = np.asarray(pred)

    cols, rows = 2, 2
    pw, ph = (width - 60) // cols, (height - 110) // rows
    out = []
    for f in range(1, horizon + 1):
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        # ASCII only: the default PIL bitmap font draws an em-dash as
        # a hollow box, and it lands in the figure's title.
        d.text((14, 12), f"What the model thinks will happen  -  "
                         f"{kind} {size}, {ck['obs_dim']}-d state", fill=INK)
        err = float(np.linalg.norm(pred[f] - truth[f]))
        d.text((14, 32), f"step {f:>3} of {horizon}     drift {err:6.3f}",
               fill=GOOD if err < 0.5 else (WARN if err < 1.5 else BAD))
        d.text((width - 250, 12), "truth", fill=SERIES[0])
        d.text((width - 190, 12), "model's prediction", fill=SERIES[2])

        for ci, (ch, label, unit) in enumerate(CHANNELS):
            cx = 30 + (ci % cols) * (pw + 14)
            cy = 60 + (ci // cols) * (ph + 26)
            d.rectangle([cx, cy, cx + pw, cy + ph], fill=PANEL)
            seg_t = truth[:f + 1, ch]
            seg_p = pred[:f + 1, ch]
            lo = float(min(seg_t.min(), seg_p.min()))
            hi = float(max(seg_t.max(), seg_p.max()))
            if hi - lo < 1e-6:
                lo, hi = lo - 0.5, hi + 0.5
            pad = (hi - lo) * 0.15
            lo, hi = lo - pad, hi + pad

            def P(i, v):
                return (cx + pw * i / max(horizon, 1),
                        cy + ph - ph * (v - lo) / (hi - lo))

            for arr, col in ((seg_t, SERIES[0]), (seg_p, SERIES[2])):
                pts = [P(i, float(v)) for i, v in enumerate(arr)]
                if len(pts) > 1:
                    d.line(pts, fill=col, width=2)
            d.text((cx + 6, cy + 4), label + unit, fill=DIM)
        out.append(img)
    save_gif(out, OUT / f"dyn-drift-{size}.gif", ms=80)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kind", default="mlp")
    ap.add_argument("--size", default="small")
    a, _ = ap.parse_known_args()
    if a.all or a.scaling:
        fig_scaling()
    if a.all or a.drift:
        fig_drift(a.kind, a.size)
    if not (a.all or a.scaling or a.drift):
        print("  nothing asked for; try --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
