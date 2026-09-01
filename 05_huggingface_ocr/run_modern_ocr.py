#!/usr/bin/env python3
"""
Compare modern OCR models: accuracy AND the vision tokens they spend.

    uv run run_modern_ocr.py --list-models             # no GPU needed
    deepspeed --num_gpus=1 run_modern_ocr.py --models got-ocr2,qwen2.5-vl-3b
    python run_modern_ocr.py --models all --max-samples 32

CoreWeave / SLURM:      sbatch submit_job.sh --models all
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 05_huggingface_ocr \\
                            --dry-run --collect --wait --terminate --yes

Why this script exists
----------------------
`train_ds.py` in this folder FINE-TUNES one vision-language model
(Qwen2-VL-2B) for OCR. That teaches the training mechanics, and it says nothing
about which model you should be fine-tuning. The field moved: purpose-built OCR
models now beat general VLMs several times their size, and they differ by more
than an order of magnitude in what a page costs them.

So this is the other half -- inference and measurement across five models,
reporting the two numbers that actually decide a choice:

    CER            character error rate, pooled over the corpus
    tokens/page    how much of the context budget one page consumes

Reporting only the first is how you end up recommending a model that is half a
point better and sixty times more expensive.

The trade this exposes
----------------------
DeepSeek-OCR (arXiv:2510.18234) makes the argument explicitly: a page rendered
as an image and compressed into ~100 vision tokens can be decoded back to text
at ~97% precision, and at 20x compression accuracy falls to ~60%. That is the
same bargain as `08_vtt/02_token_compression` -- shrink what the model looks
at, and pay for it in accuracy -- which is the through-line of this whole
course. Here it is measurable in one table.

Note on GPUs
------------
This is INFERENCE, so it does not use the DeepSpeed launcher's data
parallelism for anything except running independent models on independent
cards. Launching it under `deepspeed --num_gpus=N` works and each rank
evaluates a subset of the model list; plain `python` is equally valid and is
what the examples above mostly show.
"""

import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this the script downloads several GB of model weights and only then
    fails inside transformers with a device error -- after the reader has
    waited. Set ALLOW_CPU=1 to bypass; these models will run on CPU, extremely
    slowly, which is occasionally what you want for a single page.
    """
    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. From this folder:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return
    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU; ALLOW_CPU=1 set, continuing (expect minutes"
              " per page).\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before several GB are downloaded")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  These are 230M-4B parameter vision-language models. They need a GPU.")
    print("\n  No GPU at all? These need none:")
    print("      uv run ocr_metrics.py            # the scoring, with a demo")
    print("      uv run ../tests/test_ocr_metrics.py")
    print("      uv run run_modern_ocr.py --list-models")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 05_huggingface_ocr \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> "argparse.Namespace":
    """
    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N and a strict parser exits 2 before anything runs.
    CONTRIBUTING.md section 3.2.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="got-ocr2",
                   help="Comma-separated keys from --list-models, or 'all'.")
    p.add_argument("--list-models", action="store_true",
                   help="Describe the models and exit. Needs no GPU.")
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "hf"],
                   help="'synthetic' renders text to images locally: exact "
                        "ground truth, no download, reproducible. 'hf' pulls a "
                        "real document dataset (see --dataset).")
    p.add_argument("--dataset", default="naver-clova-ix/cord-v2",
                   help="HuggingFace dataset id when --source hf.")
    p.add_argument("--max-samples", type=int, default=16,
                   help="Pages to evaluate. Keep small: this is O(models x pages) "
                        "forward passes of a multi-billion-parameter model.")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Alias for --max-samples, for consistency with the "
                        "training scripts and the --dry-run path.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Generation cap per page. Too low truncates a real "
                        "page and shows up as a high CER that is the harness's "
                        "fault, not the model's.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


# =============================================================================
# Data
# =============================================================================


def synthetic_pages(n: int, seed: int):
    """
    Render short passages to images. Exact ground truth, no download.

    Honest about what this measures: clean rendered text is EASIER than a
    photographed receipt or a scanned page with skew and JPEG artefacts, so
    error rates here are a floor, not a document-benchmark score. It exists so
    the comparison runs anywhere, deterministically, without a dataset
    dependency -- use --source hf for real documents.
    """
    import random
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    vocab = ("invoice total amount due date customer account number balance "
             "payment received thank you for your business reference order "
             "quantity unit price subtotal tax shipping discount").split()

    pages = []
    for i in range(n):
        n_lines = rng.randint(3, 6)
        lines = [" ".join(rng.choices(vocab, k=rng.randint(4, 8)))
                 for _ in range(n_lines)]
        text = "\n".join(lines)

        img = Image.new("RGB", (640, 40 + 34 * n_lines), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except OSError:
            # No system font: the default bitmap font still renders legibly and
            # keeps this runnable on a bare container.
            font = ImageFont.load_default()
        for row, line in enumerate(lines):
            draw.text((20, 20 + 34 * row), line, fill="black", font=font)
        pages.append((img, text))
    return pages


def hf_pages(dataset_id: str, n: int):
    """Load a real document dataset and extract (image, ground-truth text)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=f"train[:{n}]")
    pages = []
    for row in ds:
        image = row.get("image")
        text = None
        for key in ("text", "ground_truth", "label", "caption"):
            if key in row and isinstance(row[key], str):
                text = row[key]
                break
        if image is None or not text:
            continue
        pages.append((image.convert("RGB"), text))
    if not pages:
        raise SystemExit(
            f"{dataset_id} yielded no (image, text) pairs. Its columns are "
            f"{list(ds.features)}; this loader looks for one of "
            "text/ground_truth/label/caption. Point --dataset at something "
            "with a plain-text column, or use --source synthetic.")
    return pages


# =============================================================================
# Model runners
# =============================================================================


def run_model(key: str, spec: dict, pages, args, torch):
    """
    Load one model, read every page, return (predictions, vision_tokens).

    Each family has a different processor contract, which is the entire reason
    a comparison script like this is more work than it looks. Getting one of
    them subtly wrong -- a missing chat template, the wrong task prompt --
    produces plausible text and a quietly terrible score, so each branch is
    written from that model's own documented usage rather than a shared guess.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, args.dtype)
    hf_id = spec["hf_id"]
    predictions, vision_tokens = [], None

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id, dtype=dtype, trust_remote_code=True,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu").eval()

    for image, _ in pages:
        if key == "got-ocr2":
            # GOT-OCR2 takes the image alone with a format flag; it has no chat
            # template and passing one produces the prompt back as "text".
            inputs = processor(image, return_tensors="pt").to(model.device)
        elif key.startswith("florence"):
            inputs = processor(text="<OCR>", images=image,
                               return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Read all the text in this image. Output only the text."},
            ]}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image],
                               return_tensors="pt").to(model.device)

        if vision_tokens is None:
            # How many tokens this page cost. For the chat models the image
            # placeholder expands inside the processor, so the honest count is
            # the input length rather than anything the config advertises.
            vision_tokens = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        # Strip the prompt: decoding the whole sequence would score the
        # instruction as if the model had read it off the page.
        generated = out[0][inputs["input_ids"].shape[-1]:]
        predictions.append(processor.decode(generated, skip_special_tokens=True))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, (vision_tokens or 0)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ocr_metrics import OCR_MODELS, accuracy_per_token, char_error_rate, corpus_cer

    bar = "=" * 78
    if args.list_models:
        print(bar)
        print("  OCR models this folder can run")
        print(bar)
        for name, spec in OCR_MODELS.items():
            print(f"  {name:<16} {spec['params']:<22} {spec['hf_id']}")
            print(f"                   {spec['note']}")
        print(bar)
        print("  Compare them:  run_modern_ocr.py --models all --max-samples 16")
        print(bar)
        return

    require_gpu()
    import torch

    torch.manual_seed(args.seed)
    n = args.max_steps if args.max_steps > 0 else args.max_samples

    keys = (list(OCR_MODELS) if args.models == "all"
            else [k.strip() for k in args.models.split(",")])
    unknown = [k for k in keys if k not in OCR_MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(OCR_MODELS)}. Run --list-models.")

    pages = (synthetic_pages(n, args.seed) if args.source == "synthetic"
             else hf_pages(args.dataset, n))
    references = [text for _, text in pages]

    print(bar)
    print("  Modern OCR comparison")
    print(bar)
    print(f"  source        {args.source}"
          + (f" ({args.dataset})" if args.source == "hf" else " (rendered locally)"))
    print(f"  pages         {len(pages)}")
    print(f"  models        {', '.join(keys)}")
    print(f"  dtype         {args.dtype}   max_new_tokens {args.max_new_tokens}")
    print(bar)

    rows = []
    for key in keys:
        spec = OCR_MODELS[key]
        print(f"\n  {key} — {spec['hf_id']} ({spec['params']})")
        try:
            predictions, tokens = run_model(key, spec, pages, args, torch)
        except Exception as exc:                       # noqa: BLE001
            # One model failing must not lose the results for the others; a
            # comparison that aborts on the first bad processor contract is
            # worth nothing.
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            rows.append((key, spec["params"], None, None, None))
            continue

        cer = corpus_cer(references, predictions)
        per_page = [char_error_rate(r, p) for r, p in zip(references, predictions)]
        rows.append((key, spec["params"], cer, tokens,
                     accuracy_per_token(cer, max(tokens, 1))))
        print(f"    CER (pooled)  {cer:.4f}")
        print(f"    CER (median)  {sorted(per_page)[len(per_page) // 2]:.4f}")
        print(f"    tokens/page   {tokens}")
        print(f"    sample        {predictions[0][:90]!r}")

    print("\n" + bar)
    print(f"  {'model':<16} {'params':<22} {'CER':>8} {'tok/page':>9} {'acc/100tok':>11}")
    print("  " + "-" * 74)
    for key, params, cer, tokens, apt in rows:
        if cer is None:
            print(f"  {key:<16} {params:<22} {'FAILED':>8}")
        else:
            print(f"  {key:<16} {params:<22} {cer:>8.4f} {tokens:>9} {apt:>11.4f}")
    print(bar)
    print("  Lower CER is better. Higher acc/100tok is better. Read both:")
    print("  a model half a point better that spends ten times the tokens is")
    print("  not better, it is a different point on the same trade.")
    if args.source == "synthetic":
        print("\n  NOTE: rendered text is cleaner than real documents, so these")
        print("  error rates are a FLOOR. Use --source hf for a real corpus.")
    print(bar)


if __name__ == "__main__":
    main()
