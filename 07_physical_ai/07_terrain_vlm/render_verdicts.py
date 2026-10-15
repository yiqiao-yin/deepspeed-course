#!/usr/bin/env python3
"""
The control, drawn: the same terrain, seen and unseen.

    uv run render_verdicts.py
    uv run render_verdicts.py --sighted runs/lora3b/eval_val.json \
                              --blind runs/blind/eval_blind.json

Reads saved predictions, so it needs no GPU and no model. Run the two
evaluations first.

WHY THIS IS THE ANIMATION WORTH HAVING
--------------------------------------
A bar chart of two accuracies is a true summary and a poor argument.
The claim this lab makes is that SIGHT is what separates the arms, and
the way to show that is to put the same picture in front of both and
let a reader watch one of them use it.

Each frame is one validation view the model has never seen. The left
badge is the sighted arm's verdict, the right is the blind control's
-- same architecture, same training, same token count, a uniform grey
image instead of terrain.

The footer carries TWO pairs of numbers and they are different
quantities. "this reel" is the running tally over the frames actually
shown, which is a small random subset and wanders by several points.
"full set" is the published result over all 429 held-out views. A
reader who watched the reel converge to a number that disagreed with
the page would be right to distrust both, so both are on screen.

Where the blind arm is RIGHT, note that it is guessing a prior rather
than reading anything: it cannot see the frame it is scoring. That is
the point of showing it beside the terrain rather than on its own.
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
GREY = (150, 170, 186, 255)
INK = (236, 242, 248, 255)


def badge(d, x, y, w, title, pred, truth):
    """One arm's verdict on this frame, and whether it was right."""
    ok = pred == truth
    col = GREEN if ok else RED
    d.rounded_rectangle([x, y, x + w, y + 52], 7,
                        fill=(14, 22, 30, 230), outline=col)
    d.text((x + 10, y + 6), title, fill=GREY)
    d.text((x + 10, y + 22), pred or "unparseable", fill=col)
    mark = "correct" if ok else "wrong"
    d.text((x + w - 10 - d.textlength(mark), y + 22), mark, fill=col)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sighted", default="runs/lora3b/eval_val.json")
    ap.add_argument("--blind", default="runs/blind/eval_blind.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--out", default="vlm-verdicts")
    ap.add_argument("--frame-ms", type=int, default=600)
    ap.add_argument("--colors", type=int, default=64)
    a, _ = ap.parse_known_args()

    from PIL import Image, ImageDraw

    for p in (a.sighted, a.blind):
        if not Path(p).exists():
            print(f"\n  missing {p}. Run the evaluations first:\n\n"
                  f"      uv run evaluate.py --run lora3b\n"
                  f"      uv run evaluate.py --run blind --blind\n")
            return 1

    S = json.loads(Path(a.sighted).read_text())
    B = json.loads(Path(a.blind).read_text())
    bmap = {p["image"]: p for p in B["preds"]}

    # Same frames, same order, for both arms -- otherwise the running
    # tallies are over different samples and the comparison is void.
    rows = [p for p in S["preds"] if p["image"] in bmap]
    print(f"  {len(rows)} validation views scored by both arms")

    # Show a mix rather than the first N, which would be one surface:
    # interleave so the reader sees both classes and all landforms.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))[:a.frames]
    rows = [rows[i] for i in idx]

    data = Path(a.data)
    frames, s_ok, b_ok = [], 0, 0
    for i, r in enumerate(rows, 1):
        b = bmap[r["image"]]
        s_ok += r["pred"] == r["truth"]
        b_ok += b["pred"] == b["truth"]

        img = Image.open(data / r["image"]).convert("RGB")
        W, H = img.size
        sheet = Image.new("RGB", (W, H + 112), (8, 12, 16))
        sheet.paste(img, (0, 0))
        d = ImageDraw.Draw(sheet, "RGBA")

        d.rectangle([0, 0, W, 26], fill=(10, 16, 22, 215))
        d.text((10, 6), f"{r['kind']}   ground truth: {r['truth']}", fill=INK)
        d.text((W - 90, 6), f"view {i}/{len(rows)}", fill=GREY)

        half = (W - 30) // 2
        badge(d, 10, H + 10, half, "SIGHTED  (LoRA 3B)", r["pred"], r["truth"])
        badge(d, 20 + half, H + 10, half, "BLIND  (grey image)",
              b["pred"], b["truth"])

        d.text((10, H + 72), f"this reel ({i:>3})   sighted {s_ok/i:5.1%}"
                             f"    blind {b_ok/i:5.1%}", fill=INK)
        d.text((10, H + 88), f"full set  ({S['n']})   sighted "
                             f"{S['accuracy']:5.1%}    blind "
                             f"{B['accuracy']:5.1%}    majority "
                             f"{S['majority']:5.1%}", fill=INK)
        frames.append(sheet)

    print(f"  final: sighted {s_ok/len(rows):.1%}, blind {b_ok/len(rows):.1%}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.out}.gif"
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors,
                     dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=a.frame_ms, loop=0, optimize=True)
    print(f"  {path.name}  ({len(frames)} frames, "
          f"{path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
