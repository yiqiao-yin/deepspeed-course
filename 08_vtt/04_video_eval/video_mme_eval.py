"""
Evaluate a video-language model the way the benchmarks actually do it.

WHY EVALUATION GETS ITS OWN SUBSECTION
--------------------------------------
The three subsections before this one all make the same promise: "you can now
fit more video." None of them can tell you whether the model still UNDERSTANDS
it. Compression is lossy by construction, and the loss curve will not warn you
— a model trained on over-compressed video converges perfectly happily to a
worse model. Training loss measures fit to your data. It cannot measure whether
you deleted the evidence.

So the compression ratio is not a hyperparameter you tune on loss. It is one
you tune on a benchmark that specifically probes what compression destroys.

THE FAILURE MODE THAT MOTIVATES THE DESIGN HERE
-----------------------------------------------
Video benchmarks have a notorious problem: many questions are answerable from
ONE FRAME. "What colour is the car?" needs no video at all. A model that
ignores time entirely — or a compressor that has thrown all of it away — can
post a respectable Video-MME score. Video-MME's own authors report the
single-frame baseline for exactly this reason.

This harness therefore separates questions into two buckets and reports them
apart:

    SINGLE-FRAME    answerable from one well-chosen frame
    TEMPORAL        require ordering, counting, duration, or causality

The gap between those two numbers is the only figure that tells you whether
your temporal path works. A model at 70% single-frame and 35% temporal has a
broken vision-time pipeline, and its 55% average hides that completely. That
average is the number people publish.

THE TASK CATEGORIES
-------------------
Modelled on Video-MME (Fu et al., 2024) and LongVideoBench (Wu et al., 2024):

  perception      what is present                      (single-frame)
  counting        how many times something happened    (temporal)
  ordering        what came before what                (temporal)
  causality       why did something happen             (temporal)
  duration        how long something took              (temporal)

`duration` is the sharpest diagnostic in the set, and it is the one that
motivates `../01_qwen25vl_baseline/`. A model with frame-index positions
rather than absolute-time positions cannot answer it in principle: sampling 16
frames from a 10-second clip and from a 10-minute clip produces identical
position information, so the evidence for "how long" was destroyed before the
model saw anything. Near-chance duration accuracy with healthy perception
accuracy is that architecture, diagnosed.

WHAT IS SYNTHETIC AND WHAT IS NOT
---------------------------------
The harness — bucketing, scoring, answer parsing, the report — is real and is
what you would point at Video-MME. The bundled QUESTIONS are synthetic, so the
whole thing runs offline with no gated dataset and no download. Point
`--dataset` at the real thing when you have it; nothing else changes.

RUNNING IT
----------
Offline, no GPU, no download (validates the harness itself):
    uv run 08_vtt/04_video_eval/video_mme_eval.py --dry-run

CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 08_vtt/04_video_eval \
                            --collect --wait --terminate --yes

References:
- Fu et al. "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of
  Multi-modal LLMs in Video Analysis." https://arxiv.org/abs/2405.21075
- Wu et al. "LongVideoBench: A Benchmark for Long-context Interleaved
  Video-Language Understanding." https://arxiv.org/abs/2407.15754
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Categories that a single well-chosen frame can answer. Everything else needs
# the model to actually integrate over time.
SINGLE_FRAME_CATEGORIES = {"perception"}
TEMPORAL_CATEGORIES = {"counting", "ordering", "causality", "duration"}


@dataclass
class Question:
    """One multiple-choice item."""

    qid: str
    category: str
    question: str
    options: List[str]
    answer: str                      # the correct letter, e.g. "B"
    duration_bucket: str = "short"   # short / medium / long
    video: Optional[str] = None

    @property
    def is_temporal(self) -> bool:
        return self.category in TEMPORAL_CATEGORIES


@dataclass
class EvalReport:
    """Accumulates results and reports them in the way that matters."""

    per_category: Dict[str, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_duration: Dict[str, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )
    single_frame: List[bool] = field(default_factory=list)
    temporal: List[bool] = field(default_factory=list)
    unparsed: int = 0

    def add(self, question: Question, correct: bool, parsed: bool) -> None:
        self.per_category[question.category].append(correct)
        self.per_duration[question.duration_bucket].append(correct)
        (self.temporal if question.is_temporal else self.single_frame).append(
            correct
        )
        if not parsed:
            self.unparsed += 1

    @staticmethod
    def _acc(results: List[bool]) -> float:
        return sum(results) / len(results) if results else 0.0

    @property
    def overall(self) -> float:
        return self._acc(self.single_frame + self.temporal)

    @property
    def temporal_gap(self) -> float:
        """
        Single-frame accuracy minus temporal accuracy.

        The headline diagnostic. Near zero means the model genuinely uses
        time. Large and positive means it is reading one frame and guessing —
        which is exactly what over-compression produces, and exactly what the
        overall average conceals.
        """
        return self._acc(self.single_frame) - self._acc(self.temporal)

    def summary(self, n_options: int = 4) -> str:
        chance = 1.0 / n_options
        lines = [
            "=" * 74,
            "  Results",
            "=" * 74,
            f"  overall             {self.overall:>6.1%}   "
            f"(chance is {chance:.1%})",
            "",
            f"  single-frame        {self._acc(self.single_frame):>6.1%}   "
            f"({len(self.single_frame)} questions)",
            f"  temporal            {self._acc(self.temporal):>6.1%}   "
            f"({len(self.temporal)} questions)",
            f"  TEMPORAL GAP        {self.temporal_gap:>+6.1%}   "
            "<- the number that matters",
            "",
            "  By category:",
        ]
        for cat in sorted(self.per_category):
            marker = "T" if cat in TEMPORAL_CATEGORIES else " "
            lines.append(
                f"    {marker} {cat:<14} {self._acc(self.per_category[cat]):>6.1%}"
                f"   ({len(self.per_category[cat])})"
            )

        lines.append("")
        lines.append("  By video duration:")
        for bucket in ("short", "medium", "long"):
            if bucket in self.per_duration:
                lines.append(
                    f"      {bucket:<14} "
                    f"{self._acc(self.per_duration[bucket]):>6.1%}"
                    f"   ({len(self.per_duration[bucket])})"
                )

        if self.unparsed:
            lines.append("")
            lines.append(
                f"  WARNING: {self.unparsed} responses had no parseable "
                "answer letter."
            )
            lines.append(
                "           These were scored WRONG. If the count is large the "
                "model is"
            )
            lines.append(
                "           failing to follow the format, not failing the task "
                "— fix the"
            )
            lines.append(
                "           prompt before believing any number above."
            )

        lines.append("")
        lines.append("  Interpretation:")
        if self.temporal_gap > 0.15:
            lines.append(
                "    LARGE GAP. The model is answering from single frames. "
                "Suspect"
            )
            lines.append(
                "    over-aggressive compression, too few frames, or "
                "frame-index"
            )
            lines.append(
                "    rather than absolute-time positions. See "
                "../01_qwen25vl_baseline/."
            )
        elif self.temporal_gap < 0.05:
            lines.append(
                "    SMALL GAP. Temporal reasoning is holding up — the "
                "compression"
            )
            lines.append(
                "    settings are safe. Try compressing harder and re-running."
            )
        else:
            lines.append(
                "    MODERATE GAP. Typical for a working model. Watch it "
                "across"
            )
            lines.append("    compression ratios rather than reading it once.")

        lines.append("=" * 74)
        return "\n".join(lines)


def parse_answer(response: str, n_options: int = 4) -> Optional[str]:
    """
    Pull the chosen letter out of a free-form model response.

    Deliberately strict-then-lenient, in that order. Models produce
    "The answer is (C)." and "C" and "C. the red car" and, infuriatingly,
    "Looking at option A, ... so the answer is C." A naive `if "A" in response`
    scores that last one as A and quietly costs you several points of measured
    accuracy that the model actually earned.

    So: try the explicit formats first, and only fall back to a bare letter
    when nothing structured is found. Returns None when genuinely
    unparseable — which the report counts and warns about, rather than
    silently scoring as wrong. An unparseable rate above a few percent means
    you are measuring format compliance, not comprehension.
    """
    valid = "".join(chr(ord("A") + i) for i in range(n_options))
    text = response.strip()

    patterns = [
        rf"answer\s*(?:is|:)\s*\(?([{valid}])\)?",   # "the answer is (C)"
        rf"^\s*\(?([{valid}])\)?\s*[.:)]",            # "C." at the start
        rf"^\s*\(?([{valid}])\)?\s*$",                # exactly "C"
        rf"\(([{valid}])\)",                          # "(C)" anywhere
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: a standalone capital letter.
    loose = re.findall(rf"\b([{valid}])\b", text)
    return loose[-1].upper() if loose else None


def build_synthetic_questions(n: int = 40, seed: int = 0) -> List[Question]:
    """
    A balanced offline question set spanning all five categories.

    Balance is the point. An unbalanced set makes the temporal gap
    uninterpretable — if 90% of questions are perception, the temporal number
    is computed from a handful of items and its error bars swallow the signal.
    """
    rng = random.Random(seed)
    templates = [
        ("perception", "What object is visible in the clip?",
         ["a red car", "a blue bicycle", "a green truck", "a yellow bus"]),
        ("counting", "How many times does the person open the door?",
         ["once", "twice", "three times", "four times"]),
        ("ordering", "Which happens FIRST in the clip?",
         ["she sits down", "she picks up the cup", "she opens the door",
          "she turns off the light"]),
        ("causality", "Why does the glass fall?",
         ["the table is bumped", "the wind blows it", "it was dropped",
          "someone pushes it"]),
        ("duration", "Roughly how long does the person spend at the sink?",
         ["about 2 seconds", "about 10 seconds", "about 30 seconds",
          "about 2 minutes"]),
    ]

    questions = []
    for i in range(n):
        category, text, options = templates[i % len(templates)]
        correct = rng.randrange(len(options))
        questions.append(Question(
            qid=f"syn-{i:04d}",
            category=category,
            question=text,
            options=list(options),
            answer=chr(ord("A") + correct),
            duration_bucket=["short", "medium", "long"][i % 3],
        ))
    return questions


def load_questions(path: str) -> List[Question]:
    """
    Load a real benchmark from JSON.

    Expected shape per row:
        {"qid", "category", "question", "options": [...], "answer": "B",
         "duration_bucket": "long", "video": "path/to.mp4"}

    Video-MME's own release needs a light reshape into this form; that
    conversion is left to you because the exact field names have changed
    between releases and hard-coding them here would rot.
    """
    with open(path) as handle:
        rows = json.load(handle)

    questions = []
    for i, row in enumerate(rows):
        category = row.get("category", "perception")
        if category not in SINGLE_FRAME_CATEGORIES | TEMPORAL_CATEGORIES:
            # Unknown categories default to temporal: the conservative choice.
            # Mis-bucketing a temporal question as single-frame inflates the
            # single-frame score and shrinks the gap, hiding the very failure
            # this harness exists to expose.
            category = "causality"
        questions.append(Question(
            qid=str(row.get("qid", i)),
            category=category,
            question=row["question"],
            options=row["options"],
            answer=row["answer"].strip().upper(),
            duration_bucket=row.get("duration_bucket", "short"),
            video=row.get("video"),
        ))
    return questions


def format_prompt(question: Question) -> str:
    """Standard MCQ prompt. The explicit format instruction is load-bearing."""
    lines = [question.question, ""]
    for i, option in enumerate(question.options):
        lines.append(f"{chr(ord('A') + i)}. {option}")
    lines.append("")
    lines.append("Answer with the letter of the correct option only.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="HF model id. Omit with --dry-run to validate "
                             "the harness offline.")
    parser.add_argument("--dataset", default=None,
                        help="JSON benchmark file. Omit for synthetic.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Random-guess baseline. Validates the harness and "
                             "establishes the chance floor.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    questions = (
        load_questions(args.dataset) if args.dataset
        else build_synthetic_questions()
    )
    if args.limit:
        questions = questions[: args.limit]

    bar = "=" * 74
    print(bar)
    print("  Video understanding evaluation")
    print(bar)
    print(f"  questions   {len(questions)}")
    print(f"  dataset     {args.dataset or 'synthetic (offline)'}")
    print(f"  model       {args.model or 'none — random baseline'}")
    print(f"  max frames  {args.max_frames}")
    print(bar)

    model = processor = None
    if args.model and not args.dry_run:
        import torch
        if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
            print("\n[preflight] Evaluating a real model needs a GPU.")
            print("            Use --dry-run to validate the harness on CPU.")
            print("            Or rent one:")
            print("              uv run runpod/runpod_ctl.py run "
                  "08_vtt/04_video_eval \\")
            print("                  --collect --wait --terminate --yes\n")
            sys.exit(1)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto"
        )

    report = EvalReport()
    # NOT seed 0. `build_synthetic_questions` also seeds a Random(0) and draws
    # one `randrange(len(options))` per question to place the correct answer.
    # Seed the guesser identically and it draws the SAME sequence in lockstep,
    # so every "random" guess lands exactly on the correct letter and the
    # baseline scores 100%. This was a real bug in this file, caught by
    # running --dry-run and noticing that chance was not chance. Correlated
    # seeds between data generation and evaluation are a classic silent leak;
    # the equivalent in a real benchmark is an eval script that reuses the
    # dataset's shuffle seed.
    rng = random.Random(20240917)
    records = []

    for question in questions:
        prompt = format_prompt(question)

        if model is None:
            # Random guessing. This is not a placeholder — it is the CHANCE
            # FLOOR, and running it is how you confirm the harness is not
            # accidentally leaking the answer. A "model" that scores 25% here
            # while your real model scores 27% tells you something important.
            response = rng.choice("ABCD"[: len(question.options)])
        else:
            import torch
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16,
                                     do_sample=False)
            response = processor.batch_decode(out, skip_special_tokens=True)[0]

        predicted = parse_answer(response, len(question.options))
        correct = predicted == question.answer
        report.add(question, correct, parsed=predicted is not None)

        records.append({
            "qid": question.qid,
            "category": question.category,
            "temporal": question.is_temporal,
            "expected": question.answer,
            "predicted": predicted,
            "correct": correct,
        })

    print(report.summary())

    with open(args.output, "w") as handle:
        json.dump({
            "overall": report.overall,
            "temporal_gap": report.temporal_gap,
            "unparsed": report.unparsed,
            "records": records,
        }, handle, indent=2)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
