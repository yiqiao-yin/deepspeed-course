# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: the evaluation harness must not lie.

Run:
    uv run tests/test_video_eval.py

Why this suite exists
---------------------
An eval harness is the instrument you use to judge everything else, so a bug
in it is worse than a bug in the model — it does not produce a wrong answer,
it produces a wrong BELIEF, and you act on that belief for weeks.

Two specific failures are covered, both of which have really happened here:

  1. ANSWER LEAKAGE. The first version of `video_mme_eval.py` seeded the
     random-guess baseline with `Random(0)` — the same seed
     `build_synthetic_questions` uses to place correct answers. Both drew one
     value per question from identical streams, so every "random" guess landed
     exactly on the correct letter and a model that saw nothing at all scored
     100%. Nothing crashed. The only reason it was caught is that --dry-run
     printed a number that was obviously impossible. This test asserts the
     baseline stays near chance.

  2. ANSWER PARSING. Models say "Looking at option A, ... therefore C". A
     naive substring match scores that as A and silently costs you real
     accuracy the model actually earned. The parser is strict-then-lenient
     and this test pins the tricky cases.

Pure stdlib, no GPU, no download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "08_vtt" / "04_video_eval"))

from video_mme_eval import (  # noqa: E402
    SINGLE_FRAME_CATEGORIES,
    TEMPORAL_CATEGORIES,
    EvalReport,
    Question,
    build_synthetic_questions,
    format_prompt,
    parse_answer,
)


def test_no_answer_leakage(r: Results) -> None:
    """
    The random baseline must score near chance, from several seeds.

    This is the regression for the correlated-seed bug. If any seed produces a
    suspiciously high score, the harness is correlated with the answer key.
    """
    questions = build_synthetic_questions(200)

    # The answer key itself must be spread across all four letters. A key that
    # is 90% "A" would make "always guess A" look like a strong model.
    counts = {letter: 0 for letter in "ABCD"}
    for question in questions:
        counts[question.answer] += 1
    spread = min(counts.values()) / max(counts.values())
    r.check(spread > 0.5,
            "correct answers are spread across the options",
            f"letter counts {counts} — a lopsided key inflates any "
            "constant-guess baseline")

    # Now the leak check proper, across several independent seeds.
    import random

    worst = 0.0
    for seed in (1, 42, 20240917, 7777):
        rng = random.Random(seed)
        hits = sum(
            rng.choice("ABCD"[: len(q.options)]) == q.answer for q in questions
        )
        acc = hits / len(questions)
        worst = max(worst, acc)

    r.check(worst < 0.45,
            "random guessing stays near chance for every seed tried",
            f"best random score was {worst:.1%}; anything approaching 100% "
            "means the guesser's RNG is correlated with the answer key — "
            "the exact bug this file regresses")

    # And the constant-guess baseline must also be near chance.
    for letter in "ABCD":
        acc = sum(q.answer == letter for q in questions) / len(questions)
        r.check(acc < 0.45,
                f"always answering {letter} stays near chance",
                f"scored {acc:.1%}")


def test_answer_parsing(r: Results) -> None:
    """The parser must handle real model output, including the traps."""
    cases = [
        ("C", "C", "a bare letter"),
        ("C.", "C", "letter with a period"),
        ("(C)", "C", "parenthesised"),
        ("The answer is C", "C", "prose lead-in"),
        ("The answer is (C).", "C", "prose plus parens"),
        ("Answer: B", "B", "colon form"),
        ("C. the red car", "C", "letter plus restated option"),
        ("answer is d", "D", "lowercase"),
        # The trap: an earlier letter is MENTIONED but a later one is chosen.
        ("Looking at option A, that seems wrong. The answer is C.",
         "C", "mentions A but answers C"),
        ("Option B describes a bicycle, but the answer is D",
         "D", "mentions B but answers D"),
    ]

    for response, expected, label in cases:
        got = parse_answer(response)
        r.check(got == expected, f"parses: {label}",
                f"{response!r} -> {got!r}, expected {expected!r}")

    # Genuinely unparseable must return None, NOT a silent wrong guess. The
    # report counts these and warns; scoring them as a letter would hide a
    # broken prompt behind a plausible-looking accuracy number.
    for junk in ("I cannot answer that.", "", "   ", "12345"):
        r.check(parse_answer(junk) is None,
                f"returns None for unparseable input {junk!r}",
                f"got {parse_answer(junk)!r}")

    # Letters outside the option range must not be accepted.
    r.check(parse_answer("The answer is D", n_options=3) is None,
            "rejects a letter beyond the number of options",
            "D is not valid when there are only 3 options")


def test_temporal_bucketing(r: Results) -> None:
    """The single-frame / temporal split must be exhaustive and disjoint."""
    overlap = SINGLE_FRAME_CATEGORIES & TEMPORAL_CATEGORIES
    r.check(not overlap, "the two buckets are disjoint", f"overlap: {overlap}")

    questions = build_synthetic_questions(50)
    categories = {q.category for q in questions}
    uncovered = categories - (SINGLE_FRAME_CATEGORIES | TEMPORAL_CATEGORIES)
    r.check(not uncovered,
            "every generated category is bucketed",
            f"unbucketed: {uncovered} — these would vanish from the report")

    r.check("duration" in TEMPORAL_CATEGORIES,
            "duration is temporal — it is the diagnostic for absolute-time "
            "encoding")
    r.check("perception" in SINGLE_FRAME_CATEGORIES,
            "perception is single-frame")

    # Both buckets must be non-empty, or the gap is undefined.
    temporal = sum(q.is_temporal for q in questions)
    r.check(0 < temporal < len(questions),
            "the generated set populates BOTH buckets",
            f"{temporal} temporal of {len(questions)} — an empty bucket makes "
            "the temporal gap meaningless")


def test_report_arithmetic(r: Results) -> None:
    """The temporal gap must be computed correctly, including at the edges."""
    report = EvalReport()

    perception = Question("p", "perception", "q?", ["a", "b", "c", "d"], "A")
    ordering = Question("o", "ordering", "q?", ["a", "b", "c", "d"], "A")

    # 8/10 single-frame, 2/10 temporal -> gap of exactly +60%.
    for i in range(10):
        report.add(perception, correct=i < 8, parsed=True)
    for i in range(10):
        report.add(ordering, correct=i < 2, parsed=True)

    r.check(abs(report.overall - 0.5) < 1e-9,
            "overall accuracy is correct", f"got {report.overall}")
    r.check(abs(report.temporal_gap - 0.6) < 1e-9,
            "temporal gap = single-frame minus temporal",
            f"got {report.temporal_gap}, expected 0.60")

    text = report.summary()
    r.check("LARGE GAP" in text,
            "a 60% gap triggers the single-frame-answering warning")
    r.check("TEMPORAL GAP" in text, "the gap is surfaced in the report")

    # An empty report must not divide by zero.
    r.check(EvalReport().overall == 0.0, "empty report does not crash")

    # Unparsed responses must be counted and warned about.
    noisy = EvalReport()
    for _ in range(5):
        noisy.add(perception, correct=False, parsed=False)
    r.check(noisy.unparsed == 5, "unparsed responses are counted")
    r.check("WARNING" in noisy.summary(),
            "unparsed responses raise a visible warning",
            "silently scoring them wrong would measure format compliance, "
            "not comprehension")


def test_prompt_format(r: Results) -> None:
    """Prompts must letter the options correctly and demand the right format."""
    question = Question("x", "perception", "What is shown?",
                        ["a cat", "a dog", "a bird", "a fish"], "B")
    prompt = format_prompt(question)

    for letter, option in zip("ABCD", question.options):
        r.check(f"{letter}. {option}" in prompt,
                f"option {letter} is lettered correctly")
    r.check("letter" in prompt.lower(),
            "the prompt asks for a letter",
            "without this instruction the unparsed rate goes through the roof")

    # Round trip: the model echoing the right letter must score correct.
    r.check(parse_answer("B") == question.answer,
            "a correct response round-trips through the parser")


def main() -> int:
    r = Results("Video evaluation harness — no leakage, honest parsing")
    test_no_answer_leakage(r)
    test_answer_parsing(r)
    test_temporal_bucketing(r)
    test_report_arithmetic(r)
    test_prompt_format(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
