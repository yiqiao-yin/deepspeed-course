# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: the ablation harness must actually CATCH a model that ignores
a modality.

Run:
    uv run tests/test_omni_eval.py

Why this suite exists
---------------------
An omni model that scores 62% might be fusing video and audio, or it might have
learned to ignore the video entirely and answer from audio. **The accuracy
number cannot tell you which**, and the second is what you get by default,
because during training one modality is usually sufficient and ignoring the
other is the cheaper way to reduce loss.

`omni_eval.py` claims its ablation grid detects this. A harness that has never
been *shown* to catch the bug it targets is a harness you are trusting on faith,
so this suite constructs modality-ignoring models on purpose and asserts they
are caught:

    fusion_skill=0, video_skill=0  ->  must report NO FUSION and "video ignored"
    healthy model                  ->  must NOT report either

Both directions matter. A diagnostic that fires on everything is as useless as
one that fires on nothing.

It also pins the answer normalisation, where the failure modes run in opposite
directions: under-normalise and a correct spoken answer ("Twenty-three.") scores
wrong; over-normalise and "not Paris" scores as "Paris".
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "05_video_speech" / "04_omni_eval"))

from omni_eval import (  # noqa: E402
    CONDITIONS,
    has_negation,
    AblationReport,
    OmniQuestion,
    build_questions,
    normalize_answer,
    score_spoken_response,
    simulated_model,
)


def evaluate(video_skill, audio_skill, fusion_skill, n=120, seed=7):
    """Run the full ablation grid against a simulated model."""
    questions = build_questions(n)
    report = AblationReport()
    rng = random.Random(seed)
    for question in questions:
        for condition in CONDITIONS:
            spoken = simulated_model(question, condition, video_skill,
                                     audio_skill, fusion_skill, rng)
            scored = score_spoken_response(spoken, question.answer)
            report.add(condition, scored["correct"], question.is_cross_modal)
    return report


def test_catches_ignored_modality(r: Results) -> None:
    """The load-bearing claim: a model ignoring video must be caught."""
    blind = evaluate(video_skill=0.0, audio_skill=0.8, fusion_skill=0.0)

    r.check(blind.fusion_gain < 0.03,
            "a non-fusing model shows ~zero fusion gain",
            f"got {blind.fusion_gain:+.1%}")
    r.check(blind.video_reliance < 0.03,
            "a video-ignoring model shows ~zero video reliance",
            f"got {blind.video_reliance:+.1%}")

    diagnosis = " ".join(blind.diagnose())
    r.check("NO FUSION" in diagnosis,
            "the diagnosis names the failure explicitly",
            f"got: {diagnosis[:120]}")
    r.check("Video is being IGNORED" in diagnosis,
            "the diagnosis identifies WHICH modality is ignored",
            f"got: {diagnosis[:160]}")

    # The mirror image must also be caught.
    deaf = evaluate(video_skill=0.8, audio_skill=0.0, fusion_skill=0.0)
    r.check("Audio is being IGNORED" in " ".join(deaf.diagnose()),
            "an audio-ignoring model is caught symmetrically",
            f"audio_reliance {deaf.audio_reliance:+.1%}")

    # And the crucial contrast: the headline accuracy does NOT reveal it.
    r.check(blind.accuracy("both") > 0.15,
            "the broken model still posts a non-trivial headline score",
            f"'both' accuracy {blind.accuracy('both'):.1%} — this is exactly "
            "why accuracy alone lets it pass review")


def test_does_not_cry_wolf(r: Results) -> None:
    """A genuinely fusing model must NOT be flagged."""
    healthy = evaluate(video_skill=0.8, audio_skill=0.8, fusion_skill=0.8)

    r.check(healthy.fusion_gain > 0.10,
            "a fusing model shows a large fusion gain",
            f"got {healthy.fusion_gain:+.1%}")

    diagnosis = " ".join(healthy.diagnose())
    r.check("GENUINE FUSION" in diagnosis,
            "the diagnosis confirms genuine fusion",
            f"got: {diagnosis[:120]}")
    r.check("NO FUSION" not in diagnosis,
            "a healthy model is NOT falsely flagged — the diagnostic "
            "discriminates rather than always firing")
    r.check("IGNORED" not in diagnosis,
            "no modality is falsely reported as ignored")

    r.check(healthy.accuracy("both") > healthy.accuracy("video_only")
            and healthy.accuracy("both") > healthy.accuracy("audio_only"),
            "both-stream accuracy exceeds either single stream")


def test_prior_floor_catches_broken_benchmark(r: Results) -> None:
    """
    A benchmark answerable from the text prompt must be flagged FIRST.

    If the prior floor is high, every other number is measuring a language
    model, so the diagnosis must stop and say so rather than confidently
    reporting a fusion gain computed from noise.
    """
    report = AblationReport()
    for _ in range(50):
        # High accuracy in EVERY condition, including no inputs at all.
        for condition in CONDITIONS:
            report.add(condition, 1.0, is_cross_modal=False)

    r.check(report.prior_floor > 0.9,
            "the prior floor detects text-answerable questions",
            f"got {report.prior_floor:.1%}")

    diagnosis = report.diagnose()
    r.check("BENCHMARK PROBLEM" in " ".join(diagnosis),
            "a broken benchmark is called out explicitly")
    r.check(not any("FUSION" in line for line in diagnosis),
            "diagnosis STOPS at the benchmark problem",
            "reporting a fusion gain computed from a broken question set "
            "would be worse than reporting nothing")


def test_answer_normalization(r: Results) -> None:
    """Spoken answers must normalise — without normalising away meaning."""
    equivalent = [
        ("Paris", "paris", "case"),
        ("It's Paris", "Paris", "carrier phrase"),
        ("uh, Paris", "Paris", "disfluency"),
        ("Twenty-three", "23", "hyphenated number word"),
        ("twenty three", "23", "spaced number word"),
        ("The Eiffel Tower", "Eiffel Tower", "leading article"),
        ("I think it's Friday.", "friday", "carrier plus punctuation"),
        ("three", "3", "single number word"),
    ]
    for spoken, reference, label in equivalent:
        got = score_spoken_response(spoken, reference)["correct"]
        r.check(got == 1.0, f"accepts equivalent phrasing: {label}",
                f"{spoken!r} vs {reference!r} -> "
                f"{normalize_answer(spoken)!r} vs {normalize_answer(reference)!r}")

    # Over-normalisation must NOT destroy meaning.
    distinct = [
        ("Paris", "London", "different answers"),
        ("not Paris", "Paris", "NEGATION must survive normalisation"),
        ("23", "32", "transposed digits"),
        # "7" IS a substring of "17" — character matching scores this correct.
        ("7", "17", "a number that is a SUBSTRING of the reference"),
        ("the woman on the left", "the woman on the right",
         "answers differing in one decisive token"),
        ("nobody", "three people", "negative vs positive answer"),
    ]
    for spoken, reference, label in distinct:
        got = score_spoken_response(spoken, reference)["correct"]
        r.check(got == 0.0, f"rejects genuinely different: {label}",
                f"{spoken!r} scored as matching {reference!r} — "
                f"normalised to {normalize_answer(spoken)!r}")

    # Negation detection must survive contractions and punctuation, since
    # normalisation strips apostrophes and would otherwise hide "isn't".
    for text in ("not Paris", "it isn't Paris", "nobody", "never",
                 "there is no one"):
        r.check(has_negation(text), f"detects negation in {text!r}")
    for text in ("Paris", "three people", "the note taker"):
        r.check(not has_negation(text),
                f"does not see negation in {text!r}",
                "a false positive here would reject correct answers")


def test_asr_uncertainty(r: Results) -> None:
    """ASR error must be reported as a band, never silently absorbed."""
    clean = score_spoken_response("Paris", "Paris", asr_wer=0.0)
    r.check(clean["asr_uncertainty"] == 0.0,
            "a perfect transcriber contributes no uncertainty")

    noisy = score_spoken_response("Paris", "Paris", asr_wer=0.1)
    r.check(noisy["asr_uncertainty"] > 0.0,
            "a 10% WER transcriber contributes uncertainty",
            f"got {noisy['asr_uncertainty']}")

    # Longer answers have more words to corrupt.
    short = score_spoken_response("Paris", "Paris", asr_wer=0.1)
    long = score_spoken_response("the woman on the left",
                                 "the woman on the left", asr_wer=0.1)
    r.check(long["asr_uncertainty"] > short["asr_uncertainty"],
            "longer answers carry more ASR uncertainty",
            f"short {short['asr_uncertainty']:.3f} vs "
            f"long {long['asr_uncertainty']:.3f}")

    # Correctness must be independent of the WER parameter — the band is
    # reported alongside, not subtracted from, the score.
    r.check(score_spoken_response("Paris", "Paris", asr_wer=0.9)["correct"] == 1.0,
            "the WER parameter never alters the correctness verdict",
            "subtracting it would invent a precision nobody has")

    for bad in (-0.1, 1.5):
        try:
            score_spoken_response("a", "a", asr_wer=bad)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects out-of-range asr_wer={bad}")


def test_question_set_is_balanced(r: Results) -> None:
    """Unbalanced requirements make the diagnostics uninterpretable."""
    questions = build_questions(60)

    video_only = sum(1 for q in questions if q.requires == {"video"})
    audio_only = sum(1 for q in questions if q.requires == {"audio"})
    cross = sum(1 for q in questions if q.is_cross_modal)

    r.check(video_only > 0 and audio_only > 0 and cross > 0,
            "all three requirement classes are represented",
            f"video={video_only} audio={audio_only} cross={cross}")
    spread = min(video_only, audio_only, cross) / max(video_only, audio_only, cross)
    r.check(spread > 0.5,
            "the classes are roughly balanced",
            f"video={video_only} audio={audio_only} cross={cross} — an "
            "unbalanced set computes a reliance figure from too few items")

    r.check(OmniQuestion("x", "q", "a", {"video", "audio"}).is_cross_modal,
            "a question needing both is cross-modal")
    r.check(not OmniQuestion("x", "q", "a", {"video"}).is_cross_modal,
            "a video-only question is not cross-modal")

    empty = AblationReport()
    r.check(empty.accuracy("both") == 0.0, "an empty report does not crash")
    r.check(empty.fusion_gain == 0.0, "an empty report has zero fusion gain")

    try:
        AblationReport().add("bogus", 1.0, False)
        caught = False
    except ValueError:
        caught = True
    r.check(caught, "rejects an unknown ablation condition")


def main() -> int:
    r = Results("Omni evaluation — does the model use BOTH streams?")
    test_catches_ignored_modality(r)
    test_does_not_cry_wolf(r)
    test_prior_floor_catches_broken_benchmark(r)
    test_answer_normalization(r)
    test_asr_uncertainty(r)
    test_question_set_is_balanced(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
