#!/usr/bin/env python3
"""
Learning curves that draw themselves, as GIFs.

    uv run render_curves.py runs/r_reveal
    uv run render_curves.py runs/r_* --out aerial-curves
    uv run render_curves.py runs/r_reveal --metric crash

WHY ANIMATE A CHART
--------------------
A static learning curve shows where a run ENDED. An animated one
shows the shape of getting there, and on this task the shape is the
result: the first reward design drove crashes from 98% to 44% and
converted every one of them into a hover that never landed. The
landing curve stayed flat on the floor while the crash curve fell,
and those two lines moving against each other is the whole story --
a final-value table would have shown "44% crash" and read like
progress.

No matplotlib. The curves are drawn with PIL into the same palette
as the rest of the lab, which keeps the dependency list short and
the figures consistent with the flight GIFs.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"

BG = (10, 14, 20)
PANEL = (16, 22, 30)
GRID = (38, 50, 64)
INK = (232, 238, 246)
DIM = (142, 162, 180)
SERIES = [(108, 206, 148), (126, 186, 248), (228, 136, 120),
          (226, 190, 100), (190, 150, 238), (120, 220, 214),
          (240, 160, 200)]

METRICS = {
    "landed_ok": ("clean landings", "%"),
    "crash":     ("crashes", "%"),
    "timeout":   ("never landed (timeout)", "%"),
    "return":    ("episode return", ""),
}


def load(run: Path) -> dict | None:
    h = run / "history.json"
    if not h.exists():
        return None
    rows = json.loads(h.read_text())
    if not rows:
        return None
    return {"name": run.name, "rows": rows}


def draw(runs, metric, W, H, upto, title, baseline=None):
    from PIL import Image, ImageDraw

    name, unit = METRICS[metric]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    L, R, T, B = 82, W - 150, 68, H - 52
    d.rectangle([L, T, R, B], fill=PANEL)

    xmax = max(r["rows"][-1]["step"] for r in runs)
    vals = [row[metric] for r in runs for row in r["rows"]]
    if baseline is not None:
        vals.append(baseline)
    ymin, ymax = min(vals), max(vals)
    if metric != "return":
        ymin, ymax = 0.0, max(ymax * 1.12, 0.1)
    else:
        pad = (ymax - ymin) * 0.12 + 1e-6
        ymin, ymax = ymin - pad, ymax + pad

    def px(step, v):
        x = L + (R - L) * step / max(xmax, 1)
        y = B - (B - T) * (v - ymin) / max(ymax - ymin, 1e-9)
        return x, y

    # Grid and axes.
    for k in range(6):
        v = ymin + (ymax - ymin) * k / 5
        _x, y = px(0, v)
        d.line([L, y, R, y], fill=GRID)
        lab = f"{v:.0%}" if unit == "%" else f"{v:.0f}"
        d.text((L - 10 - 6 * len(lab), y - 6), lab, fill=DIM)
    for k in range(5):
        st = xmax * k / 4
        x, _y = px(st, ymin)
        d.line([x, T, x, B], fill=GRID)
        d.text((x - 14, B + 8), f"{st / 1e6:.1f}M", fill=DIM)
    # Labels live OUTSIDE the plot box and out of each other's way.
    # The first layout put the subtitle on top of the top y-tick and
    # the step counter on top of the x-axis caption.
    d.text((L, B + 28), "environment steps", fill=DIM)
    d.text((14, 12), title, fill=INK)
    d.text((14, 30), name, fill=DIM)

    if baseline is not None:
        _x, y = px(0, baseline)
        for xx in range(L, R, 10):
            d.line([xx, y, xx + 5, y], fill=(226, 190, 100))
        d.text((R + 8, y - 7), "classical", fill=(226, 190, 100))
        d.text((R + 8, y + 5), f"{baseline:.0%}" if unit == "%"
               else f"{baseline:.0f}", fill=(226, 190, 100))

    for i, r in enumerate(runs):
        col = SERIES[i % len(SERIES)]
        pts = [px(row["step"], row[metric]) for row in r["rows"]
               if row["step"] <= upto]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3, joint="curve")
        if pts:
            x, y = pts[-1]
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
            cur = [row for row in r["rows"] if row["step"] <= upto][-1]
            txt = (f"{cur[metric]:.0%}" if unit == "%"
                   else f"{cur[metric]:.0f}")
            d.text((R + 8, T + 6 + 30 * i), r["name"], fill=col)
            d.text((R + 8, T + 18 + 30 * i), txt, fill=INK)

    d.text((R - 96, 30), f"step {upto:,}", fill=DIM)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default="landed_ok", choices=sorted(METRICS))
    ap.add_argument("--baseline", type=float, default=None,
                    help="classical rate to draw as a reference line")
    ap.add_argument("--title", default="PPO learns to land a cold-gas car")
    ap.add_argument("--width", type=int, default=860)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--hold", type=int, default=22)
    ap.add_argument("--out", default="aerial-curve")
    a, _ = ap.parse_known_args()

    paths = []
    for pat in a.runs:
        paths.extend(sorted(glob.glob(pat)))
    runs = [r for r in (load(Path(p)) for p in paths) if r]
    if not runs:
        print(f"  no runs with a history.json among {paths}")
        return 1
    print(f"  {len(runs)} run(s): {', '.join(r['name'] for r in runs)}")

    xmax = max(r["rows"][-1]["step"] for r in runs)
    frames = [draw(runs, a.metric, a.width, a.height,
                   int(xmax * (k + 1) / a.frames), a.title, a.baseline)
              for k in range(a.frames)]
    frames.extend([frames[-1]] * a.hold)

    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=96,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KB)")
    for r in runs:
        print(f"    {r['name']:<14} final {a.metric} = "
              f"{r['rows'][-1][a.metric]:.3f} at "
              f"{r['rows'][-1]['step']:,} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
