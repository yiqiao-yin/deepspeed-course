#!/usr/bin/env python3
"""
Where the model actually fails: a confusion matrix you can look at.

    uv run render_confusion.py
    uv run render_confusion.py --run blind --eval eval_blind.json

Reads saved predictions, so it needs no GPU and no model.

WHY A CONFUSION MATRIX AND NOT AN ACCURACY
-------------------------------------------
52.9% is one number covering two very different behaviours, and the
per-landform table says which: the sighted model scores 71-72% on
mesas and scarps and 35-46% on dunes and hills. That is not uniform
weakness, it is a BIAS -- the model has learned "Martian terrain
looks impassable" rather than the difference between a smooth
shoulder and a sharp one.

A number cannot show that. Four cells of terrain can. Each quadrant
here cycles through real held-out views belonging to it:

    top-left      truth OVER,   said OVER      correct, crossable
    top-right     truth OVER,   said AROUND    THE EXPENSIVE ERROR
    bottom-left   truth AROUND, said OVER      the dangerous one
    bottom-right  truth AROUND, said AROUND    correct, blocked

The top-right quadrant is the lab's result in picture form: ground
the robot could have walked straight over, refused. Watch how
ordinary that terrain looks.

The two error cells are NOT symmetric in consequence, which is worth
saying out loud even though this lab scores them equally. Refusing
passable ground costs a detour; driving at a scarp because you
called it passable costs the rover. An accuracy figure prices them
the same, and a real mission would not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "physical"

GREEN = (92, 196, 141, 255)
RED = (226, 110, 110, 255)
AMBER = (224, 170, 94, 255)
GREY = (150, 170, 186, 255)
INK = (236, 242, 248, 255)

CELLS = [
    ("OVER", "OVER", "said OVER, was OVER", "correct — crossable", GREEN),
    ("OVER", "AROUND", "said AROUND, was OVER", "refused passable ground", AMBER),
    ("AROUND", "OVER", "said OVER, was AROUND", "walked into a blocker", RED),
    ("AROUND", "AROUND", "said AROUND, was AROUND", "correct — blocked", GREEN),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="lora3b")
    ap.add_argument("--eval", default="eval_val.json")
    ap.add_argument("--output", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--tile-w", type=int, default=336)
    ap.add_argument("--tile-h", type=int, default=252)
    ap.add_argument("--out", default="vlm-confusion")
    ap.add_argument("--frame-ms", type=int, default=750)
    ap.add_argument("--colors", type=int, default=128)
    a, _ = ap.parse_known_args()

    from PIL import Image, ImageDraw

    path = Path(a.output) / a.run / a.eval
    if not path.exists():
        print(f"\n  missing {path}. Run the evaluation first:\n\n"
              f"      uv run evaluate.py --run {a.run}\n")
        return 1
    E = json.loads(path.read_text())
    preds = E["preds"]

    # Bucket the held-out views into the four cells.
    buckets: dict = {}
    for p in preds:
        buckets.setdefault((p["truth"], p["pred"]), []).append(p)
    rng = np.random.default_rng(0)
    for k in buckets:
        rng.shuffle(buckets[k])

    print(f"  {len(preds)} held-out views, {a.run}")
    for truth, pred, label, _why, _c in CELLS:
        n = len(buckets.get((truth, pred), []))
        print(f"    {label:<24} {n:>4}  ({n/len(preds):>5.1%})")
    empty = [c[2] for c in CELLS if not buckets.get((c[0], c[1]))]
    if empty:
        # Not an error -- a model biased hard enough genuinely empties
        # a cell -- but it must be visible rather than silently drawn
        # as a blank square.
        print(f"    EMPTY CELLS: {', '.join(empty)}")

    data = Path(a.data)
    W, H, PAD, TOP = a.tile_w, a.tile_h, 4, 34
    sheet_w = 2 * W + 3 * PAD
    sheet_h = TOP + 2 * (H + 34) + 3 * PAD + 26

    frames = []
    for f in range(a.frames):
        sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 12, 16))
        d = ImageDraw.Draw(sheet, "RGBA")
        d.text((PAD + 6, 8),
               f"Where the model fails — {a.run},  "
               f"{E['accuracy']:.1%} over {E['n']} held-out views", fill=INK)

        for i, (truth, pred, label, why, col) in enumerate(CELLS):
            x = PAD + (i % 2) * (W + PAD)
            y = TOP + (i // 2) * (H + 34 + PAD)
            items = buckets.get((truth, pred), [])
            if items:
                r = items[f % len(items)]
                img = Image.open(data / r["image"]).convert("RGB")
                if img.size != (W, H):
                    img = img.resize((W, H))
                sheet.paste(img, (x, y))
                kind = r["kind"]
            else:
                d.rectangle([x, y, x + W, y + H], fill=(16, 22, 28))
                d.text((x + W // 2 - 56, y + H // 2),
                       "no views in this cell", fill=GREY)
                kind = "—"
            d.rectangle([x, y, x + W, y + H], outline=col, width=3)
            d.rectangle([x, y + H, x + W, y + H + 34], fill=(12, 18, 24, 240))
            d.text((x + 8, y + H + 4), label, fill=col)
            d.text((x + 8, y + H + 19),
                   f"{why}   ·   {len(items)} views   ·   {kind}", fill=GREY)

        d.text((PAD + 6, sheet_h - 20),
               "the two error cells cost different things: a detour, "
               "or the rover", fill=GREY)
        frames.append(sheet)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.out}.gif"
    pil = [x.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for x in frames]
    pil[0].save(out, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"\n  {out.name}  ({len(frames)} frames, "
          f"{out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
