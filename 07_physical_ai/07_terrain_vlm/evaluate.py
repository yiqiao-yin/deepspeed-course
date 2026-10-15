#!/usr/bin/env python3
"""
Does seeing the terrain beat walking into it? The arms, side by side.

    uv run evaluate.py --run lora3b
    uv run evaluate.py --zero-shot            # the base model, untrained
    uv run evaluate.py --run blind --blind    # the control

WHAT IS BEING COMPARED
----------------------
Four numbers, and the third is the one that matters:

    majority class    ~52%   predict AROUND every time
    zero-shot         ?      the base VLM, no fine-tuning
    blind (CONTROL)   ?      fine-tuned on a uniform grey image
    lab 5's robot     ~75%   contact-only discrimination, 3:1

A fine-tuned VLM scoring 85% sounds good and means nothing on its
own. It has to beat the BLIND arm -- same pipeline, same token count,
no terrain information -- or the camera contributed nothing and the
model simply learned the prior. That is lab 5's `padded` control, and
the reason lab 5 could say its robot's 3:1 discrimination came from
having a body rather than from perception.

HOW ANSWERS ARE SCORED
----------------------
By exact first word, after stripping. NOT by substring: a scorer that
matched "AROUND" inside "I would not go AROUND" would count a refusal
as a correct answer, and this repository has shipped exactly that bug
in `05_video_speech` -- a spoken-answer scorer that matched "not
Paris" against "Paris".

Unparseable generations are COUNTED AND REPORTED, never silently
mapped to a class. A model that emits prose on 30% of items and is
scored as if it guessed is being flattered by its own evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

VALID = ("OVER", "AROUND")


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Duplicated verbatim from `train_vlm.py` rather than imported, by
    repository convention: there is no shared module here, and a
    reader opening this one file must be able to see what it checks
    and why. (`require_gpu()` appears verbatim in ~34 files on
    purpose.)

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        return

    print("\n" + "=" * 72)
    print("  NO GPU DETECTED — scoring a 3-8B vision model needs one")
    print("=" * 72)
    print("  What you can still run on this machine, right now:\n")
    print("      uv run make_dataset.py --check     # the labels and their"
          " properties")
    print("      uv run verify_arch.py              # where ZeRO-3 becomes"
          " necessary")
    print("      uv run ../../tests/test_terrain_vlm.py\n")
    print("  Already have predictions? The animation needs no GPU:\n")
    print("      uv run render_verdicts.py\n")
    print("  To rent a GPU for the real thing:\n")
    print("      cd ../../runpod && uv run runpod_ctl.py run \\")
    print("          07_physical_ai/07_terrain_vlm --dry-run --collect \\")
    print("          --wait --terminate --yes\n")
    print("  Set ALLOW_CPU=1 to continue anyway (it will be very slow).")
    print("=" * 72 + "\n")
    sys.exit(1)



def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a proportion. Small n here, so not normal-approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def parse(text: str) -> str | None:
    """
    First word, exactly. Returns None rather than guessing.

    `None` is a THIRD outcome, not a wrong answer: conflating "said
    something unparseable" with "said the wrong thing" hides a
    degenerate model behind an ordinary-looking accuracy.
    """
    if not text:
        return None
    w = text.strip().split()
    if not w:
        return None
    first = w[0].strip(".,:;!?\"'*").upper()
    return first if first in VALID else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--run", default=None,
                    help="a run under --output; omit with --zero-shot")
    ap.add_argument("--output", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blind", action="store_true",
                    help="evaluate on a uniform grey image (the control)")
    ap.add_argument("--zero-shot", action="store_true",
                    help="the base model, no adapter")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_known_args()[0]

    if not a.zero_shot and not a.run:
        print("  need --run NAME or --zero-shot")
        return 2

    require_gpu()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    auto = getattr(transformers, "AutoModelForImageTextToText", None) \
        or getattr(transformers, "AutoModelForVision2Seq")

    data = Path(a.data)
    rows = [json.loads(l) for l in (data / f"{a.split}.jsonl").open()]
    if a.limit:
        rows = rows[:a.limit]

    processor = AutoProcessor.from_pretrained(a.model)
    model = auto.from_pretrained(a.model, dtype=torch.bfloat16,
                                 attn_implementation="sdpa").cuda()
    if not a.zero_shot:
        from peft import PeftModel

        adapter = Path(a.output) / a.run
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"  adapter {adapter}")
    model.eval()

    GREY = Image.new("RGB", (448, 336), (128, 128, 128))
    arm = ("zero-shot" if a.zero_shot else a.run) + (" (BLIND)" if a.blind else "")

    bar = "=" * 72
    print(bar)
    print(f"  {arm} — {len(rows)} {a.split} samples")
    print(bar)

    correct, unparsed, n = 0, 0, 0
    by_kind: dict = {}
    preds = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts, images = [], []
        for r in chunk:
            img = GREY if a.blind else Image.open(data / r["image"]).convert("RGB")
            images.append(img)
            texts.append(processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": r["prompt"]}]}],
                tokenize=False, add_generation_prompt=True))
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False)
        gen = processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        for r, g in zip(chunk, gen):
            p = parse(g)
            n += 1
            k = r["meta"]["kind"]
            by_kind.setdefault(k, [0, 0])
            by_kind[k][1] += 1
            if p is None:
                unparsed += 1
            elif p == r["answer"]:
                correct += 1
                by_kind[k][0] += 1
            preds.append({"image": r["image"], "kind": k,
                          "truth": r["answer"], "raw": g.strip()[:40],
                          "pred": p})
        if (i // a.batch) % 5 == 0:
            print(f"    {n}/{len(rows)}  running {correct/max(n,1):.1%}",
                  flush=True)

    acc = correct / n
    lo, hi = wilson(correct, n)
    maj = max(sum(r["answer"] == "AROUND" for r in rows),
              sum(r["answer"] == "OVER" for r in rows)) / n

    print(f"\n  accuracy        {acc:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  majority class  {maj:.1%}")
    print(f"  unparseable     {unparsed} / {n} ({unparsed/n:.1%})")
    if unparsed / n > 0.1:
        print("    NOTE: over a tenth of generations were not OVER/AROUND. "
              "Accuracy\n    below is over ALL items, counting those as "
              "wrong, which is the\n    honest reading but not a "
              "comparable one.")
    print("\n  by landform kind:")
    for k, (c, m) in sorted(by_kind.items()):
        print(f"    {k:<7} {c:>4}/{m:<5} {c/m:>6.1%}")

    # The two bars that matter, named rather than implied.
    print(f"\n  vs majority     {'ABOVE' if lo > maj else 'NOT above'} "
          f"(CI low {lo:.1%} vs {maj:.1%})")
    print(f"  vs lab 5 (75%)  {'ABOVE' if lo > 0.75 else 'NOT above'} "
          f"(CI low {lo:.1%})")

    res = {"arm": arm, "run": a.run, "zero_shot": a.zero_shot,
           "blind": a.blind, "split": a.split, "n": n,
           "accuracy": acc, "ci": [lo, hi], "majority": maj,
           "unparseable": unparsed, "by_kind": by_kind}
    out = Path(a.json) if a.json else (
        Path(a.output) / (a.run or "zeroshot") /
        f"eval_{'blind' if a.blind else a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "preds": preds}, indent=2))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
