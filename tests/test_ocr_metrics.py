# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: OCR scoring.

Run:
    uv run tests/test_ocr_metrics.py

Why this suite exists
---------------------
An OCR benchmark is a model plus a metric, and the metric is where the silent
mistakes live. Each of these produces a plausible number and a wrong ranking:

  * dividing by the prediction length rather than the reference -- a model
    improves its score by emitting less
  * returning 0.0 for an empty reference -- a model that outputs nothing scores
    perfectly on blank pages
  * clamping the rate at 1.0 -- runaway generation, the most common VLM failure
    on an unreadable page, becomes invisible
  * averaging per-page rates instead of pooling edits -- a two-character page
    moves the corpus score as much as a thousand-character one
  * zip()-ing mismatched reference and prediction lists -- scores the shorter
    one and reports a fine-looking number

So this asserts PROPERTIES, not golden values. Where a property has a
counterexample, the counterexample is asserted too: it is not enough that
pooling and averaging differ in principle, the test builds a corpus where they
differ by 250x.

Pure stdlib. No GPU, no download, no model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "05_huggingface_ocr"))

from ocr_metrics import (  # noqa: E402
    OCR_MODELS, accuracy_per_token, char_error_rate, compression_ratio,
    corpus_cer, levenshtein, normalize, word_error_rate)


def test_levenshtein(r: Results) -> None:
    r.check(levenshtein("kitten", "sitting") == 3,
            "levenshtein('kitten','sitting') == 3 (the textbook case)")
    r.check(levenshtein("", "") == 0, "empty vs empty is 0")
    r.check(levenshtein("abc", "") == 3, "everything deleted costs its length")
    r.check(levenshtein("abc", "abc") == 0, "identical strings cost nothing")

    # Symmetry: the two-row implementation swaps its arguments internally to
    # keep the inner loop short, so this is a real check of that swap.
    pairs = [("abcdef", "az"), ("", "xyz"), ("aaa", "aaab"), ("page", "gape")]
    r.check(all(levenshtein(a, b) == levenshtein(b, a) for a, b in pairs),
            "distance is symmetric (the row-swap optimisation is correct)")

    # Triangle inequality -- cheap, and catches an off-by-one in the recurrence
    # that symmetry alone would not.
    a, b, c = "receipt", "reciept", "recipe"
    r.check(levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c),
            "distance obeys the triangle inequality")


def test_normalisation(r: Results) -> None:
    r.check(normalize("A  B\n C") == "a b c",
            "whitespace collapses and case folds by default")
    r.check(normalize("ＡＢ１") == "ab1",
            "NFKC folds full-width characters",
            "a model emitting the full-width '1' is making an encoding choice, "
            "not a reading error")
    r.check(normalize("a,b.", punctuation=False) == "a,b.",
            "punctuation is KEPT by default",
            "dropping it flatters the model and hides errors in tables and "
            "formulas, which is where the interesting failures are")
    r.check(normalize("a,b.", punctuation=True) == "ab",
            "punctuation can be dropped explicitly")
    r.check(normalize("A B", case=False) == "A B", "case folding is optional")


def test_cer_denominator(r: Results) -> None:
    """The single most consequential choice in the metric."""
    ref = "the quick brown fox"          # 19 chars
    # A model that emits half the page must score ~0.5, not ~0.0.
    half = char_error_rate(ref, "the quick")
    r.check(0.4 < half < 0.7,
            f"emitting half the page scores ~0.5 CER ({half:.3f})",
            "if this is near zero the denominator is the PREDICTION length, "
            "and a model can improve by saying less")

    # Runaway generation must score ABOVE 1.0, not be clipped.
    runaway = char_error_rate(ref, " ".join([ref] * 4))
    r.check(runaway > 1.0,
            f"runaway generation scores above 1.0 ({runaway:.3f})",
            "clipping at 1.0 hides the most common VLM failure on a page it "
            "cannot parse")

    r.check(char_error_rate(ref, ref) == 0.0, "a perfect read scores 0.0")
    r.check(char_error_rate(ref, "") == 1.0, "empty output scores 1.0")


def test_empty_reference(r: Results) -> None:
    """A blank page is a trap: the natural implementation divides by zero."""
    r.check(char_error_rate("", "") == 0.0,
            "blank page, blank output -> 0.0 (correct)")
    r.check(char_error_rate("", "hallucinated") == 1.0,
            "blank page, invented text -> 1.0",
            "returning 0.0 here makes a model that hallucinates onto empty "
            "pages look perfect; a ZeroDivisionError crashes the benchmark")
    r.check(word_error_rate("", "x") == 1.0, "same for WER")


def test_normalisation_is_applied(r: Results) -> None:
    ref = "The Quick  Brown\nFox"
    r.check(char_error_rate(ref, "the quick brown fox") == 0.0,
            "case and whitespace differences are not errors",
            "if this is non-zero the metric is scoring layout, not recognition")


def test_pooled_vs_averaged(r: Results) -> None:
    """It is not enough that they differ; show HOW MUCH."""
    refs = ["a" * 1000, "hi"]
    preds = ["a" * 1000, "XX"]           # long page perfect, tiny page wrong
    pooled = corpus_cer(refs, preds)
    averaged = sum(char_error_rate(a, b) for a, b in zip(refs, preds)) / 2

    r.check(pooled < 0.01,
            f"pooled CER is dominated by the long page ({pooled:.4f})")
    r.check(averaged > 0.4,
            f"averaged CER is dominated by the two-character page ({averaged:.4f})")
    r.check(averaged / pooled > 100,
            f"the two disagree by {averaged / pooled:.0f}x on the same predictions",
            "a benchmark that does not say which it used is not comparable to "
            "anything")

    r.check(corpus_cer(["abc"], ["abc"]) == 0.0, "a perfect corpus scores 0.0")


def test_mismatched_lengths_raise(r: Results) -> None:
    try:
        corpus_cer(["a", "b"], ["a"])
        raised = False
    except ValueError:
        raised = True
    r.check(raised,
            "mismatched reference/prediction counts raise",
            "a silent zip() would score only the shorter list and report a "
            "number that looks completely fine")


def test_token_economics(r: Results) -> None:
    r.check(compression_ratio(1000, 100) == 10.0,
            "1000 text tokens in 100 vision tokens is 10x compression")

    # The ordering this exists to protect: a slightly worse model that spends a
    # fraction of the tokens must rank higher on the cost axis.
    cheap = accuracy_per_token(cer=0.08, vision_tokens=100)
    dear = accuracy_per_token(cer=0.05, vision_tokens=1300)
    r.check(cheap > dear,
            f"100 tokens at CER 0.08 beats 1300 tokens at CER 0.05 "
            f"({cheap:.3f} vs {dear:.3f}) on the cost axis",
            "this is the whole reason the metric exists -- accuracy alone "
            "ranks a 13x more expensive model first for 3 points")

    r.check(accuracy_per_token(cer=1.5, vision_tokens=100) == 0.0,
            "a model worse than useless scores 0, not negative")

    for bad in (0, -1):
        try:
            compression_ratio(100, bad)
            raised = False
        except ValueError:
            raised = True
        r.check(raised, f"vision_tokens={bad} raises rather than dividing by zero")


def test_model_registry(r: Results) -> None:
    r.check(len(OCR_MODELS) >= 4, f"the registry lists {len(OCR_MODELS)} models")
    for name, spec in OCR_MODELS.items():
        r.check(bool(spec.get("hf_id")) and "/" in spec["hf_id"],
                f"{name}: has a well-formed HuggingFace id")
        r.check(bool(spec.get("note")),
                f"{name}: says why it is in the list",
                "a model in a comparison table without a reason is noise")


def main() -> int:
    r = Results("OCR metrics — scoring, normalisation and token economics")
    test_levenshtein(r)
    test_normalisation(r)
    test_cer_denominator(r)
    test_empty_reference(r)
    test_normalisation_is_applied(r)
    test_pooled_vs_averaged(r)
    test_mismatched_lengths_raise(r)
    test_token_economics(r)
    test_model_registry(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
