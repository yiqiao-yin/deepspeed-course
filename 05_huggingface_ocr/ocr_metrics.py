#!/usr/bin/env python3
"""
OCR evaluation: character error rate, and the accuracy-per-vision-token trade.

    uv run ocr_metrics.py        # no GPU, no download, no model

Why this file exists separately from the models
-----------------------------------------------
An OCR benchmark is two things: a model, and a way of scoring it. The scoring
is where the quiet mistakes live, because every one of these produces a number
that looks reasonable:

  * computing CER against unnormalised text, so a model that gets every
    character right but uses different whitespace scores 30% error
  * dividing by the length of the PREDICTION instead of the reference, which
    lets a model improve its score by emitting less
  * returning 0.0 for an empty reference, which makes a model that outputs
    nothing look perfect on blank pages
  * averaging per-page rates instead of pooling edits, which weights a
    ten-character caption the same as a thousand-word page

None of those raise. All of them change the ranking. So the metric lives here,
runs on CPU, and is asserted in tests/test_ocr_metrics.py against properties
rather than against a golden number.

The second axis: tokens per page
--------------------------------
Accuracy alone is the wrong way to compare modern OCR models, because they
differ by more than an order of magnitude in how many vision tokens they spend
per page, and those tokens are the cost. DeepSeek-OCR's central claim
(arXiv:2510.18234) is precisely this trade: it reports ~97% decoding precision
while compressing text tokens ~10x into vision tokens, and ~60% at 20x.

That is the same bargain as `08_vtt/02_token_compression` -- shrink what the
model looks at, and pay in accuracy -- which is why this module reports
accuracy AND token budget rather than accuracy alone.

Pure stdlib. No torch, no transformers.
"""

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple


# =============================================================================
# Normalisation
# =============================================================================


def normalize(text: str, *, case: bool = True, whitespace: bool = True,
              punctuation: bool = False, unicode_nfkc: bool = True) -> str:
    """
    Canonicalise text before scoring.

    Every flag here is a judgement about what counts as an error, and the
    defaults are the conservative ones:

    `unicode_nfkc`  -- a model emitting the full-width digit "１" instead of
                       "1" is not making a reading error, it is making an
                       encoding choice. NFKC folds those together.
    `whitespace`    -- collapsing runs of space/newline to one space. Layout
                       is a separate problem from recognition; a model that
                       reads every character correctly but wraps lines
                       differently should not score as badly wrong.
    `case`          -- lowercasing. Defensible either way; ON by default
                       because most OCR benchmarks report case-insensitive CER
                       and comparing against them otherwise is apples to
                       oranges.
    `punctuation`   -- OFF by default. Dropping punctuation flatters a model
                       and hides real errors in tables and formulas, which is
                       exactly where the interesting failures are.
    """
    if unicode_nfkc:
        text = unicodedata.normalize("NFKC", text)
    if case:
        text = text.lower()
    if punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    if whitespace:
        text = " ".join(text.split())
    return text


# =============================================================================
# Edit distance
# =============================================================================


def levenshtein(a: Sequence, b: Sequence) -> int:
    """
    Edit distance with the two-row trick: O(len(a) * len(b)) time, O(min) space.

    A full matrix on a 5,000-character page against a 5,000-character reference
    is 25M cells; two rows is 5,000. That difference decides whether a whole
    benchmark fits in memory.
    """
    if a == b:
        return 0
    if len(a) < len(b):          # keep the inner loop over the shorter one
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (ca != cb)  # substitution
            ))
        previous = current
    return previous[-1]


# =============================================================================
# Rates
# =============================================================================


def char_error_rate(reference: str, prediction: str, **norm) -> float:
    """
    CER = edit distance / length of the REFERENCE.

    Note the denominator. Dividing by the prediction's length would let a model
    lower its error rate by emitting less text, which is the opposite of what
    the metric is for.

    An empty reference with a non-empty prediction returns 1.0, not 0.0 and not
    a ZeroDivisionError: the model hallucinated text onto a blank page, which is
    a total failure, not a perfect score. Empty against empty is 0.0.

    CER is NOT capped at 1.0 -- a model that emits ten times the reference
    length scores above 1.0, and it should. Clipping would hide runaway
    generation, which is the most common failure mode of a VLM asked to read a
    page it cannot parse.
    """
    ref, pred = normalize(reference, **norm), normalize(prediction, **norm)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(ref, pred) / len(ref)


def word_error_rate(reference: str, prediction: str, **norm) -> float:
    """WER: the same, over whitespace-delimited tokens rather than characters."""
    ref = normalize(reference, **norm).split()
    pred = normalize(prediction, **norm).split()
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(ref, pred) / len(ref)


def corpus_cer(references: Sequence[str], predictions: Sequence[str],
               **norm) -> float:
    """
    CER over a corpus, POOLED: total edits / total reference characters.

    Not the mean of per-page rates. Averaging rates weights a ten-character
    caption exactly as much as a thousand-word page, so one short page the
    model happens to fail can move the corpus score by more than a long page it
    reads perfectly. Pooling is what OmniDocBench and the CER literature mean.

    The two disagree substantially on real documents, which is why a benchmark
    that does not say which one it used is not comparable to anything.
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"{len(references)} references but {len(predictions)} predictions "
            "-- a silent zip() here would score only the shorter list and "
            "report a number that looks fine")
    edits = total = 0
    for ref, pred in zip(references, predictions):
        r, p = normalize(ref, **norm), normalize(pred, **norm)
        edits += levenshtein(r, p)
        total += len(r)
    return edits / total if total else (0.0 if not any(predictions) else 1.0)


# =============================================================================
# The cost axis
# =============================================================================


def compression_ratio(text_tokens: int, vision_tokens: int) -> float:
    """
    How many text tokens each vision token stands in for.

    The quantity DeepSeek-OCR reports (arXiv:2510.18234): they observe ~97%
    decoding precision below 10x and ~60% at 20x. Above 1.0 the page is being
    compressed; below 1.0 the model is spending more tokens looking at the page
    than the page contains.
    """
    if vision_tokens <= 0:
        raise ValueError("vision_tokens must be positive")
    return text_tokens / vision_tokens


def accuracy_per_token(cer: float, vision_tokens: int) -> float:
    """
    Character accuracy delivered per 100 vision tokens.

    A blunt instrument, and deliberately so: it exists to stop a comparison
    table that ranks a model spending 6,000 tokens per page above one spending
    100 for half a point of accuracy. Read it alongside CER, never instead.
    """
    if vision_tokens <= 0:
        raise ValueError("vision_tokens must be positive")
    return max(0.0, 1.0 - cer) / (vision_tokens / 100)


# =============================================================================
# The models this folder can run
# =============================================================================
# Every id below was checked against the HuggingFace API before being written
# here -- a model card that 404s is a worse failure than an absent one, because
# the reader assumes their setup is broken.

OCR_MODELS: Dict[str, Dict] = {
    "qwen2-vl-2b": dict(
        hf_id="Qwen/Qwen2-VL-2B-Instruct", params="2.2B",
        note="The folder's existing baseline (train_ds.py fine-tunes this).",
    ),
    "qwen2.5-vl-3b": dict(
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct", params="3.8B",
        note="Direct successor to the baseline; the 72B variant of this family "
             "is the reference point most OCR papers compare against "
             "(arXiv:2502.13923).",
    ),
    "got-ocr2": dict(
        hf_id="stepfun-ai/GOT-OCR-2.0-hf", params="580M",
        note="Purpose-built OCR-2.0 model, transformers-native. The pick when "
             "VRAM is the binding constraint -- 580M against 2-4B.",
    ),
    "deepseek-ocr": dict(
        hf_id="deepseek-ai/DeepSeek-OCR", params="3B MoE (570M active)",
        note="Optical context compression (arXiv:2510.18234): SAM-base + "
             "CLIP-large + a 16x token compressor. Reports beating GOT-OCR2.0 "
             "on OmniDocBench with 100 vision tokens against its 256.",
    ),
    "florence-2-base": dict(
        hf_id="microsoft/Florence-2-base", params="230M",
        note="Not an OCR specialist -- a general vision-language model with an "
             "OCR task prompt. Included as the control: it shows how much of "
             "the score is 'reads text' versus 'is a document model'.",
    ),
}

# Deliberately absent, and why -- so nobody re-adds them without knowing:
#   PaddleOCR-VL  leads OmniDocBench v1.6 (96.33%) but ships as PaddlePaddle,
#                 not transformers. Running it here would mean a second deep
#                 learning framework in a DeepSpeed course.
#   dots.ocr      the id commonly cited for it does not resolve on the Hub.


def _demo() -> None:
    bar = "=" * 78
    print(bar)
    print("  OCR metrics — the scoring, before any model exists")
    print(bar)

    ref = "The quick brown fox jumps over the lazy dog."
    cases = [
        ("perfect",            ref),
        ("one character off",  "The quick brown fox jumps over the lazy dof."),
        ("different spacing",  "The  quick brown\nfox jumps over the  lazy dog."),
        ("different case",     "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG."),
        ("half the page",      "The quick brown fox"),
        ("runaway generation", ref + " " + ref + " " + ref),
        ("empty",              ""),
    ]
    print(f"  reference: {ref!r}\n")
    for label, pred in cases:
        print(f"  {label:<20} CER {char_error_rate(ref, pred):.4f}   "
              f"WER {word_error_rate(ref, pred):.4f}")
    print()
    print("  Note 'different spacing' and 'different case' both score 0.0000 —")
    print("  normalisation decides what counts as an error, before any model")
    print("  is involved. And 'runaway generation' scores ABOVE 1.0, which is")
    print("  the point: a capped metric would hide it.")
    print(bar)

    print("  Pooled vs averaged, on a corpus where they disagree")
    refs = ["a" * 1000, "hi"]
    preds = ["a" * 1000, "XX"]          # long page perfect, tiny page wrong
    pooled = corpus_cer(refs, preds)
    averaged = sum(char_error_rate(r, p) for r, p in zip(refs, preds)) / len(refs)
    print(f"    pooled   (total edits / total chars) : {pooled:.4f}")
    print(f"    averaged (mean of per-page rates)    : {averaged:.4f}")
    print(f"    -> the same predictions, {averaged / max(pooled, 1e-9):.0f}x apart.")
    print("       A two-character page moved the score as much as a 1000-char one.")
    print(bar)

    print("  The cost axis — accuracy is not comparable without it")
    print(f"    {'model':<16} {'vision tokens/page':>18} {'if CER=0.05':>14}")
    for name, tokens in [("deepseek-ocr", 100), ("got-ocr2", 256),
                         ("qwen2.5-vl-3b", 1300), ("MinerU2.0-style", 6000)]:
        print(f"    {name:<16} {tokens:>18} {accuracy_per_token(0.05, tokens):>14.4f}")
    print("    (accuracy delivered per 100 vision tokens — read WITH the CER)")
    print(f"    a 1000-token page at 100 vision tokens = "
          f"{compression_ratio(1000, 100):.0f}x compression")
    print(bar)

    print("  Models this folder can run")
    for name, spec in OCR_MODELS.items():
        print(f"    {name:<16} {spec['params']:<20} {spec['hf_id']}")
    print(bar)


if __name__ == "__main__":
    _demo()
