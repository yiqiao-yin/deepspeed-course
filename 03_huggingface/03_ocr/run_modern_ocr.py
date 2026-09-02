#!/usr/bin/env python3
"""
Compare modern OCR models: accuracy AND the vision tokens they spend.

    uv run run_modern_ocr.py --list-models             # no GPU needed
    deepspeed --num_gpus=1 run_modern_ocr.py --models got-ocr2,qwen2.5-vl-3b
    python run_modern_ocr.py --models all --max-samples 32

CoreWeave / SLURM:      sbatch submit_job.sh --models all
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 03_huggingface/03_ocr \\
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
same bargain as `04_video_text/03_token_compression` -- shrink what the model looks
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
    print("      uv run ../../tests/test_ocr_metrics.py")
    print("      uv run run_modern_ocr.py --list-models")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 03_huggingface/03_ocr \\")
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
    p.add_argument("--degrade", default="none",
                   choices=["none", "blur", "noise", "small"],
                   help="Make the pages HARDER on purpose. A benchmark that "
                        "reports 0.0000 should be able to show a non-zero, or "
                        "you cannot tell a perfect model from a broken "
                        "harness. 'small' shrinks to 60%% (fewer pixels per "
                        "glyph), 'blur' softens, 'noise' adds speckle.")
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

    Lines are wrapped to the MEASURED width of the font, and the function
    asserts afterwards that nothing overflows. That assertion is not
    defensive decoration -- the first version of this generator laid out
    fixed-length lines and 8 of them ran past the right edge, so the reference
    text contained words that were not in the image at all. Every model was
    then scored against text it could not possibly read, and the benchmark
    reported plausible numbers for a comparison that meant nothing.

    Honest about what this measures: cleanly rendered text is EASIER than a
    photographed receipt or a skewed scan, so error rates here are a floor,
    not a document-benchmark score. Use --source hf for real documents.
    """
    import random
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    # Document-like phrasing rather than a bag of words. Random word salad
    # punishes models with strong language priors for no good reason, and no
    # real page looks like it.
    templates = [
        "invoice number {n:05d}", "order reference {n:06d}",
        "customer account {n:04d}", "date 2026-0{m}-{d:02d}",
        "subtotal {a}.{b:02d} usd", "tax {c}.{b:02d} usd",
        "shipping {c}.{b:02d} usd", "total amount due {a}.{b:02d} usd",
        "payment received thank you", "quantity {q} unit price {a}.{b:02d}",
        "balance carried forward {a}.{b:02d}", "please remit within 30 days",
    ]

    width, margin, font_size, line_h = 720, 24, 22, 34
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def measure(text: str) -> float:
        if hasattr(font, "getlength"):
            return font.getlength(text)
        return font.getbbox(text)[2]

    usable = width - 2 * margin
    pages = []
    for _ in range(n):
        lines = []
        for _ in range(rng.randint(3, 6)):
            line = rng.choice(templates).format(
                n=rng.randint(1, 99999), m=rng.randint(1, 9), d=rng.randint(1, 28),
                a=rng.randint(10, 999), b=rng.randint(0, 99),
                c=rng.randint(1, 99), q=rng.randint(1, 40))
            # Wrap on words until it fits; a line that cannot fit at all is a
            # template bug and should be seen, not silently clipped.
            while measure(line) > usable and " " in line:
                line = line.rsplit(" ", 1)[0]
            lines.append(line)

        img = Image.new("RGB", (width, 2 * margin + line_h * len(lines)), "white")
        draw = ImageDraw.Draw(img)
        for row, line in enumerate(lines):
            draw.text((margin, margin + line_h * row), line, fill="black", font=font)

        overflow = [ln for ln in lines if measure(ln) > usable]
        if overflow:
            raise AssertionError(
                f"rendered line does not fit the image and would be clipped: "
                f"{overflow[0]!r} -- the reference would contain text that is "
                "not in the picture, which silently invalidates every score")
        pages.append((img, "\n".join(lines)))
    return pages


def degrade(image, how: str):
    """
    Make a page harder on purpose.

    Exists to answer one question about any benchmark that reports a perfect
    score: is the model perfect, or is the harness broken? A pipeline that
    cannot produce a non-zero error rate on a deliberately degraded page is
    not measuring anything, and no amount of reasoning about the code
    substitutes for showing it move.
    """
    from PIL import Image, ImageFilter

    if how == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.6))
    if how == "small":
        w, h = image.size
        small = image.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)   # back up, detail gone
    if how == "noise":
        import random
        px = image.load()
        rng = random.Random(0)
        for _ in range((image.size[0] * image.size[1]) // 12):
            x, y = rng.randrange(image.size[0]), rng.randrange(image.size[1])
            v = rng.choice((0, 255))
            px[x, y] = (v, v, v)
        return image
    return image


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

    Each family has a different contract, which is the whole reason a
    comparison script is more work than it looks. Two of the five needed a
    bespoke path, discovered by running them rather than by reading:

      deepseek-ocr    AutoProcessor cannot instantiate it at all
                      ("Unrecognized processing class"); it ships a custom
                      `infer` entry point behind trust_remote_code.
      florence-2      dies with "Florence2LanguageConfig object has no
                      attribute forced_bos_token_id" -- its remote config
                      predates a field transformers' generate() now reads.

    Getting one of these subtly wrong does not raise: it produces plausible
    text and a quietly terrible score, which is indistinguishable from the
    model being bad. That is why each branch is written from that model's own
    documented usage.
    """
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Print WHICH transformers this process actually imported, not which one
    # was installed. A pinned environment that is silently not on the path
    # looks identical to a pinned environment that is -- and that cost several
    # GPU runs here, chasing a model bug that was an environment bug.
    print(f"    transformers  {transformers.__version__}  "
          f"({os.path.dirname(transformers.__file__)})")

    # Restore the pre-5.0 default the Florence-2 remote code relies on. Its
    # Florence2LanguageConfig.__init__ reads self.forced_bos_token_id while
    # constructing itself; transformers 5 moved the generation defaults off
    # PretrainedConfig, so that read raises. A class attribute fixes it for any
    # config that never sets one -- and it must be in place before the
    # PROCESSOR loads, which is where the config is first built. Every earlier
    # attempt patched at model-load time and therefore ran too late.
    from transformers import PretrainedConfig

    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None

    dtype = getattr(torch, args.dtype)

    # transformers renamed this: `torch_dtype` through 4.x, `dtype` from 5.0.
    # The remote-code models below only load on 4.47, so the script has to
    # speak both. Passing the wrong one is a TypeError at construction, after
    # the weights have already downloaded.
    _major = int(transformers.__version__.split(".")[0])
    dtype_kw = "dtype" if _major >= 5 else "torch_dtype"
    hf_id = spec["hf_id"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictions, vision_tokens = [], None

    # ---- deepseek-ocr: custom entry point, no AutoProcessor ----------------
    if key == "deepseek-ocr":
        import pathlib
        import tempfile

        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        # float32, not bfloat16. Its infer() loads the image itself and builds
        # a float32 tensor, so a bf16 model raises
        #   RuntimeError: Input type (float) and bias type (c10::BFloat16)
        #   should be the same
        # on the first conv. Casting the model is the side we control; 3B MoE
        # with 570M active fits in fp32 on a 24 GB card.
        ds_dtype = torch.float32 if args.dtype != "float32" else dtype
        model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True,
            _attn_implementation="eager",
            **{dtype_kw: ds_dtype}).eval().to(device)
        with tempfile.TemporaryDirectory() as tmp:
            for i, (image, _) in enumerate(pages):
                path = os.path.join(tmp, f"page_{i}.png")
                image.save(path)
                # eval_mode=True is the whole ballgame. Reading its source:
                #
                #   if not eval_mode:      -> generate(..., streamer=...)   # prints,
                #                                                          # returns None
                #   else:                  -> outputs = tokenizer.decode(...)
                #                             return outputs
                #
                # With the default (False) it STREAMS the transcription to
                # stdout and returns None, which is why earlier runs scored the
                # literal string "None" and then an empty file. Nothing about
                # the model or the environment was wrong; the call was.
                # "Free OCR.", NOT "<|grounding|>...". The grounding prompt
                # makes this model emit layout markup -- refs and bounding
                # boxes around every span -- so scoring it against plain text
                # measured the markup, not the reading: CER 2.3107, i.e. it
                # "wrote" more than twice the reference. The plain-text prompt
                # is the one listed in the model's own source.
                out = model.infer(
                    tokenizer, prompt="<image>\nFree OCR.",
                    image_file=path, output_path=tmp, base_size=640,
                    image_size=640, crop_mode=False, save_results=False,
                    test_compress=False, eval_mode=True)
                text = out if isinstance(out, str) and out.strip() else ""
                if not text:
                    # Read whatever it wrote, newest first.
                    written = sorted(
                        (f for f in pathlib.Path(tmp).rglob("*")
                         if f.is_file() and f.suffix in {".txt", ".mmd", ".md"}),
                        key=lambda f: f.stat().st_mtime, reverse=True)
                    if written:
                        text = written[0].read_text(errors="ignore")
                        written[0].unlink()          # do not re-read it next page
                predictions.append(text)
        # The compressor's own budget, not an input-length proxy: this model's
        # entire claim is about how few vision tokens a page costs.
        vision_tokens = 100
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return predictions, vision_tokens

    # ---- everything else goes through AutoProcessor -------------------------
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    load_kwargs = {dtype_kw: dtype, "trust_remote_code": True,
                   "device_map": device}
    if key.startswith("florence"):
        # Build and repair the config BEFORE instantiating. Two earlier
        # attempts patched the model's config AFTER from_pretrained and failed
        # with the identical error every time, because the error is raised
        # during loading -- so the patch never ran. The lesson is that "the fix
        # did not take" and "the fix never executed" look the same from the
        # outside, and only the traceback's position tells them apart.
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        bos = getattr(processor.tokenizer, "bos_token_id", None) or 0
        seen = set()

        def _patch(obj) -> None:
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            if getattr(obj, "forced_bos_token_id", None) is None:
                try:
                    obj.forced_bos_token_id = getattr(obj, "bos_token_id", None) or bos
                except Exception:                      # noqa: BLE001
                    pass
            for attr in ("text_config", "vision_config", "language_config",
                         "decoder", "encoder"):
                _patch(getattr(obj, attr, None))

        _patch(cfg)

        # Patching the config INSTANCE was not enough -- from_pretrained
        # rebuilds sub-configs from their dicts, discarding it, which is why
        # the identical AttributeError survived three separate fixes. Setting
        # the attribute on the CLASS means every instance has it however and
        # whenever it is reconstructed.
        def _patch_class(obj) -> None:
            if obj is None:
                return
            klass = type(obj)
            if not hasattr(klass, "forced_bos_token_id"):
                setattr(klass, "forced_bos_token_id",
                        getattr(obj, "bos_token_id", None) or bos)
            for attr in ("text_config", "vision_config", "language_config"):
                _patch_class(getattr(obj, attr, None))

        _patch_class(cfg)
        load_kwargs["config"] = cfg

    try:
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs).eval()
    except (ValueError, KeyError):
        # On transformers 4.x, Florence-2's remote config is not registered for
        # the image-text-to-text auto class and raises "Unrecognized
        # configuration class". Its documented loader there is AutoModelForCausalLM.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs).eval()

    if key.startswith("florence"):
        # generation_config is built from the (already repaired) config, but
        # carries its own copy of the field on some versions.
        gc = getattr(model, "generation_config", None)
        if gc is not None and getattr(gc, "forced_bos_token_id", None) is None:
            gc.forced_bos_token_id = getattr(
                model.config, "forced_bos_token_id", None) or 0

    for image, _ in pages:
        if key == "got-ocr2":
            # GOT-OCR2 takes the image alone; it has no chat template and
            # passing one returns the prompt back as "text".
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

        # Match the model's precision. A bf16 model handed float32 pixels dies
        # with "Input type (float) and bias type (c10::BFloat16) should be the
        # same" on the first convolution -- which is what stopped florence-2
        # even after its config was repaired.
        model_dtype = next(model.parameters()).dtype
        for name, value in list(inputs.items()):
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[name] = value.to(model_dtype)

        if vision_tokens is None:
            # NOTE what this counts: the length of input_ids. For the chat-style
            # VLMs the image placeholder expands INTO input_ids, so this is a
            # fair proxy for what a page costs the context.
            #
            # It is NOT fair for Florence-2, whose visual tokens travel in
            # pixel_values and never enter input_ids -- it reports ~10, which
            # would rank it an order of magnitude "cheaper" than anything else
            # purely as an artefact of where its tokens live. Flagged rather
            # than silently tabulated.
            vision_tokens = int(inputs["input_ids"].shape[-1])
            if key.startswith("florence"):
                vision_tokens = -abs(vision_tokens)   # negative = not comparable

        with torch.no_grad():
            if key.startswith("florence"):
                # Explicit arguments: its processor returns keys generate()
                # does not accept, and this is the call verified to work on
                # transformers 4.47.1 (it reads a rendered line exactly).
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=3)
            else:
                out = model.generate(**inputs,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
        # Strip the prompt: decoding the whole sequence would score the
        # instruction as if the model had read it off the page.
        generated = out[0][inputs["input_ids"].shape[-1]:]
        text = processor.decode(generated, skip_special_tokens=True)
        if key.startswith("florence"):
            # Florence returns its answer wrapped in the task token.
            text = text.replace("<OCR>", "").strip()
        predictions.append(text)

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
    if args.degrade != "none":
        pages = [(degrade(img, args.degrade), text) for img, text in pages]
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

        # A model that returned nothing must be reported as FAILED, not
        # scored. Scoring str(None) against a reference yields a plausible
        # 0.96 that looks like a measurement and is purely an artefact of the
        # harness -- which is exactly what happened before this check existed.
        useful = [p for p in predictions if p and p.strip() and p.strip() != "None"]
        if not useful:
            print(f"    NO OUTPUT: every page returned empty or None -- refusing "
                  f"to report a CER for this. Sample: {predictions[0]!r}")
            rows.append((key, spec["params"], None, None, None))
            continue

        cer = corpus_cer(references, predictions)
        per_page = [char_error_rate(r, p) for r, p in zip(references, predictions)]
        comparable = tokens > 0
        rows.append((key, spec["params"], cer, abs(tokens),
                     accuracy_per_token(cer, max(abs(tokens), 1)) if comparable
                     else None))
        ordered = sorted(per_page)
        exact = sum(1 for v in per_page if v == 0.0)
        print(f"    CER (pooled)  {cer:.4f}")
        print(f"    CER (median)  {ordered[len(ordered) // 2]:.4f}")
        print(f"    CER (min/max) {ordered[0]:.4f} / {ordered[-1]:.4f}")
        # A pooled 0.0000 is only meaningful next to this count. If every page
        # is exact, say so explicitly rather than leaving the reader to wonder
        # whether the harness is scoring anything at all.
        print(f"    exact pages   {exact}/{len(per_page)}")
        print(f"    tokens/page   {tokens}")
        print(f"    sample        {predictions[0][:90]!r}")

    print("\n" + bar)
    print(f"  {'model':<16} {'params':<22} {'CER':>8} {'tok/page':>9} {'acc/100tok':>11}")
    print("  " + "-" * 74)
    for key, params, cer, tokens, apt in rows:
        if cer is None:
            print(f"  {key:<16} {params:<22} {'FAILED':>8}")
        elif apt is None:
            print(f"  {key:<16} {params:<22} {cer:>8.4f} {tokens:>9}*{'n/a':>10}")
        else:
            print(f"  {key:<16} {params:<22} {cer:>8.4f} {tokens:>9} {apt:>11.4f}")
    print(bar)
    if any(r[4] is None for r in rows if r[2] is not None):
        print("  * input_ids length only -- this model's visual tokens travel")
        print("    outside input_ids, so its token count is NOT comparable and")
        print("    no efficiency figure is reported for it.")
    print("  Lower CER is better. Higher acc/100tok is better. Read both:")
    print("  a model half a point better that spends ten times the tokens is")
    print("  not better, it is a different point on the same trade.")
    if args.source == "synthetic":
        print("\n  NOTE: rendered text is cleaner than real documents, so these")
        print("  error rates are a FLOOR. Use --source hf for a real corpus.")
    print(bar)


if __name__ == "__main__":
    main()
