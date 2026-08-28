"""
Evaluating a video-speech-to-speech model — does it actually use both streams?

THE QUESTION THAT MATTERS, AND WHY ACCURACY DOES NOT ANSWER IT
--------------------------------------------------------------
You have built an omni model. It takes video and speech in, it speaks back, and
it scores 62% on a benchmark. Two systems produce that number:

    A.  genuinely fuses what it sees with what it hears
    B.  ignores the video entirely and answers from audio alone

They are indistinguishable from the score. They are *completely* different
systems, and B is what you get by default -- because during training, one
modality is usually sufficient for most examples, so the cheapest way to reduce
loss is to learn one stream well and treat the other as noise.

This is the omni-modal analogue of the single-frame problem in
`08_vtt/04_video_eval/`, and it is worse, because there are now two ways to
cheat instead of one.

**Accuracy cannot detect it. Ablation can.**

THE ABLATION GRID
-----------------
Run the same benchmark four times, degrading the input:

    both          video + audio          the real number
    video_only    audio muted
    audio_only    video blanked
    neither       both removed           the text-prior floor

Four numbers, three diagnostics, each of which the single "both" score hides
completely:

    fusion_gain     = both - max(video_only, audio_only)
                      Near zero means NO FUSION. The model picked whichever
                      single stream was better and ignored the other. This is
                      the headline number of the whole harness.

    video_reliance  = both - audio_only
                      How much does removing video hurt? Near zero means the
                      video encoder is decoration.

    audio_reliance  = both - video_only
                      Same for audio.

    prior_floor     = neither
                      Well above chance means the *benchmark* is broken:
                      questions answerable from the text prompt alone. You are
                      measuring a language model, not an omni model. Always
                      check this first -- if it is high, nothing else here
                      means anything.

WHY SCORING SPEECH IS ITS OWN PROBLEM
-------------------------------------
This family speaks its answers, which breaks exact-match scoring in a way that
text models never had to deal with.

    - The model says "twenty-three"; the reference says "23". Both correct.
    - The model says "uh, I think it's Paris". Correct, with disfluency.
    - You must transcribe before you can score, and **the ASR makes its own
      mistakes** -- so a model penalty and a transcription error are recorded
      identically.

That last point is the one that quietly corrupts published numbers. If your ASR
has 5% WER, roughly 5% of your "model errors" are not model errors, and the
effect is not uniform: it hits rare words, names, and numbers hardest, which
are exactly what benchmark answers are made of.

`score_spoken_response` normalises aggressively before comparing and reports an
ASR-attributable band alongside the score, so the uncertainty is visible rather
than absorbed.

WHAT IS REAL HERE
-----------------
The ablation grid, the scoring, the normalisation, the diagnostics, the report
-- all real, and all what you would point at OmniEval or LVOmniBench. The
bundled questions and the simulated model are synthetic so the harness runs
offline with no download, and so the diagnostics can be *proved* to detect a
modality-ignoring model rather than merely claimed to.

Covered by `tests/test_omni_eval.py`.

References:
- Wang et al. "OmniEval: A Benchmark for Evaluating Omni-modal Models with
  Visual, Auditory, and Textual Inputs." https://arxiv.org/abs/2506.20960
- Xu et al. "Qwen2.5-Omni Technical Report." https://arxiv.org/abs/2503.20215
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# The four input conditions. Order matters for the report.
CONDITIONS = ("both", "video_only", "audio_only", "neither")


@dataclass
class OmniQuestion:
    """
    One benchmark item, annotated with what it genuinely requires.

    `requires` is the important field and the one benchmarks usually omit.
    Without it you cannot tell whether a low score means the model is bad or
    the question set was answerable from one stream all along.
    """

    qid: str
    question: str
    answer: str
    requires: Set[str] = field(default_factory=set)   # {"video"}, {"audio"}, or both
    category: str = "general"

    @property
    def is_cross_modal(self) -> bool:
        """True when the question needs video AND audio together."""
        return {"video", "audio"} <= self.requires


def normalize_answer(text: str) -> str:
    """
    Aggressively normalise a spoken answer before comparison.

    Every rule here exists because the un-normalised version scores a correct
    answer as wrong:

        "Twenty-three."   vs "23"      -> number words
        "uh, Paris"       vs "Paris"   -> disfluency
        "It's Paris"      vs "Paris"   -> carrier phrase
        "PARIS"           vs "Paris"   -> case
        "the Eiffel Tower" vs "Eiffel Tower" -> article

    Under-normalising makes a good model look bad. Over-normalising makes a bad
    model look good -- strip too much and "not Paris" scores as "Paris". So
    negation is deliberately preserved.
    """
    text = text.lower().strip()

    # Disfluencies. Bounded by word edges so "author" does not lose "uh".
    text = re.sub(r"\b(uh|um|er|ah|hmm|mm)\b", " ", text)

    # Carrier phrases, only at the start.
    text = re.sub(r"^(well|so|okay|ok|i think|i believe|it'?s|that'?s|"
                  r"the answer is|it is|there are|there is)\s+", "", text)

    # Number words -> digits. Answers are full of these.
    words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16",
        "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    }
    text = re.sub(r"[-‐-―]", " ", text)
    text = " ".join(words.get(tok, tok) for tok in text.split())

    # Compound numbers: "20 3" -> "23".
    text = re.sub(r"\b(\d0) (\d)\b", r"\1\2", text)
    text = re.sub(r"\b20(\d)\b", lambda m: str(20 + int(m.group(1))), text)

    # Articles, but NOT negations.
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


_NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "cannot", "cant", "isnt", "arent", "wasnt", "werent",
    "doesnt", "didnt", "dont", "wont", "hasnt", "havent",
}


def has_negation(text: str) -> bool:
    """
    Whether an answer is negated.

    Checked separately from normalisation because normalisation is designed to
    *discard* surface differences, and negation is the one surface difference
    that reverses the meaning. Contractions are stripped of their apostrophe
    first so "isn't" and "isnt" both match.
    """
    stripped = re.sub(r"['’]", "", text.lower())
    tokens = re.sub(r"[^\w\s]", " ", stripped).split()
    return any(tok in _NEGATIONS for tok in tokens)


def _contains_subsequence(haystack: List[str], needle: List[str]) -> bool:
    """
    Whether `needle` appears as a CONTIGUOUS run of tokens inside `haystack`.

    Contiguity matters: "the woman on the left" and "the woman on the right"
    share four of five tokens, and a set-overlap test would call them equal.
    """
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def score_spoken_response(
    response: str, reference: str, asr_wer: float = 0.0
) -> Dict[str, float]:
    """
    Score one spoken answer, and be honest about ASR contamination.

    Args:
        response: The ASR transcript of what the model said.
        reference: The expected answer.
        asr_wer: Word error rate of the ASR used, as a fraction. Feed the real
            measured value for your transcriber.

    Returns:
        `correct` (0.0/1.0), plus `asr_uncertainty` — the fraction of this
        item's words the ASR could plausibly have corrupted. Aggregated across
        the set, that becomes an error band on the headline score.

        The band is reported rather than subtracted. Subtracting would invent
        a precision nobody has; reporting makes the reader aware that a 2-point
        difference between two systems may be entirely transcription noise.
    """
    if not 0.0 <= asr_wer <= 1.0:
        raise ValueError(f"asr_wer must be in [0, 1], got {asr_wer}")

    norm_response = normalize_answer(response)
    norm_reference = normalize_answer(reference)

    # Negation must agree BEFORE any containment test. "not Paris" contains
    # "Paris", so a naive substring match scores a flat contradiction as
    # correct -- a bug this file shipped with, caught by its own test suite.
    # Negation inverts meaning and containment cannot see it.
    if has_negation(response) != has_negation(reference):
        correct = False
    else:
        # Token-level containment, not raw substring. A spoken answer
        # legitimately carries extra words ("Paris, the capital") and
        # legitimately omits them ("Paris" for "the city of Paris").
        #
        # Comparing tokens rather than characters also fixes the other
        # substring trap: "7" IS a substring of "17", so character matching
        # scores the wrong number as correct on exactly the short numeric
        # answers benchmarks are full of.
        resp_tokens = norm_response.split()
        ref_tokens = norm_reference.split()
        correct = (
            resp_tokens == ref_tokens
            or _contains_subsequence(resp_tokens, ref_tokens)
            or _contains_subsequence(ref_tokens, resp_tokens)
        )

    n_words = max(len(norm_reference.split()), 1)
    # Probability the ASR corrupted at least one word of a short answer.
    uncertainty = 1.0 - (1.0 - asr_wer) ** n_words

    return {"correct": 1.0 if correct else 0.0, "asr_uncertainty": uncertainty}


@dataclass
class AblationReport:
    """
    Accumulates the four conditions and computes the diagnostics.

    The design principle: **make the failure mode impossible to miss.** A
    single accuracy number lets a modality-ignoring model pass review. Four
    numbers side by side with an explicit fusion gain does not.
    """

    scores: Dict[str, List[float]] = field(
        default_factory=lambda: {c: [] for c in CONDITIONS}
    )
    cross_modal: Dict[str, List[float]] = field(
        default_factory=lambda: {c: [] for c in CONDITIONS}
    )
    asr_uncertainties: List[float] = field(default_factory=list)
    n_options: int = 4

    def add(self, condition: str, correct: float, is_cross_modal: bool,
            asr_uncertainty: float = 0.0) -> None:
        if condition not in self.scores:
            raise ValueError(f"unknown condition {condition!r}; "
                             f"expected one of {CONDITIONS}")
        self.scores[condition].append(correct)
        if is_cross_modal:
            self.cross_modal[condition].append(correct)
        if condition == "both":
            self.asr_uncertainties.append(asr_uncertainty)

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def accuracy(self, condition: str) -> float:
        return self._mean(self.scores[condition])

    def cross_modal_accuracy(self, condition: str) -> float:
        return self._mean(self.cross_modal[condition])

    @property
    def fusion_gain(self) -> float:
        """
        How much the model gains from having BOTH streams.

        The headline. `both` minus the better single modality. Near zero means
        the model is not fusing — it found whichever stream was more useful and
        ignored the other, and its impressive score is a single-modality score
        wearing an omni-modal costume.
        """
        return self.accuracy("both") - max(self.accuracy("video_only"),
                                           self.accuracy("audio_only"))

    @property
    def video_reliance(self) -> float:
        """Drop in accuracy when video is removed. Near zero: video ignored."""
        return self.accuracy("both") - self.accuracy("audio_only")

    @property
    def audio_reliance(self) -> float:
        """Drop in accuracy when audio is removed. Near zero: audio ignored."""
        return self.accuracy("both") - self.accuracy("video_only")

    @property
    def prior_floor(self) -> float:
        """
        Accuracy with NO video and NO audio.

        Check this first. Well above chance means the benchmark is answerable
        from the text prompt, and every other number here is measuring a
        language model rather than an omni model.
        """
        return self.accuracy("neither")

    @property
    def asr_band(self) -> float:
        """Mean fraction of items the ASR could plausibly have corrupted."""
        return self._mean(self.asr_uncertainties)

    def summary(self) -> str:
        chance = 1.0 / self.n_options
        lines = [
            "=" * 74,
            "  Ablation grid",
            "=" * 74,
        ]
        for condition in CONDITIONS:
            n = len(self.scores[condition])
            lines.append(
                f"  {condition:<12} {self.accuracy(condition):>6.1%}   "
                f"({n} questions)"
            )

        lines += [
            "",
            f"  FUSION GAIN        {self.fusion_gain:>+6.1%}   "
            "<- both minus the better single stream",
            f"  video reliance     {self.video_reliance:>+6.1%}",
            f"  audio reliance     {self.audio_reliance:>+6.1%}",
            f"  text-prior floor   {self.prior_floor:>6.1%}   "
            f"(chance is {chance:.1%})",
        ]

        if self.asr_band > 0:
            lines.append(
                f"  ASR error band     +/-{self.asr_band:>5.1%}   "
                "of items could be transcription noise"
            )

        if self.cross_modal["both"]:
            lines += [
                "",
                "  On questions that genuinely require BOTH streams:",
                f"    both           {self.cross_modal_accuracy('both'):>6.1%}",
                f"    video_only     {self.cross_modal_accuracy('video_only'):>6.1%}",
                f"    audio_only     {self.cross_modal_accuracy('audio_only'):>6.1%}",
            ]

        lines += ["", "  Diagnosis:"]
        lines.extend(f"    {line}" for line in self.diagnose())
        lines.append("=" * 74)
        return "\n".join(lines)

    def diagnose(self) -> List[str]:
        """Turn the four numbers into the sentence a reader needs."""
        out: List[str] = []
        chance = 1.0 / self.n_options

        # Check the benchmark before the model. Always.
        if self.prior_floor > chance + 0.15:
            out.append(
                f"BENCHMARK PROBLEM. {self.prior_floor:.0%} accuracy with NO "
                "video and NO audio."
            )
            out.append(
                "  The questions are answerable from the text prompt alone. "
                "Fix the"
            )
            out.append(
                "  question set before reading anything else here — you are "
                "measuring a"
            )
            out.append("  language model, not an omni model.")
            return out

        if self.fusion_gain < 0.03:
            out.append(
                f"NO FUSION. Adding the second stream gains only "
                f"{self.fusion_gain:+.1%}."
            )
            out.append(
                "  The model is answering from ONE modality and ignoring the "
                "other. Its"
            )
            out.append(
                "  headline score is a single-modality score. Suspect the "
                "alignment —"
            )
            out.append("  see ../02_thinker_talker/ for TMRoPE.")
        elif self.fusion_gain < 0.10:
            out.append(
                f"WEAK FUSION ({self.fusion_gain:+.1%}). Some cross-modal use, "
                "less than expected."
            )
        else:
            out.append(
                f"GENUINE FUSION ({self.fusion_gain:+.1%}). The model needs "
                "both streams."
            )

        if self.video_reliance < 0.03:
            out.append(
                "  Video is being IGNORED — removing it costs almost nothing."
            )
        if self.audio_reliance < 0.03:
            out.append(
                "  Audio is being IGNORED — removing it costs almost nothing."
            )

        if self.asr_band > 0.05:
            out.append(
                f"  ASR noise (+/-{self.asr_band:.0%}) is large enough to "
                "swamp small differences."
            )
        return out


# ---------------------------------------------------------------------------
# Offline question set and a simulated model
# ---------------------------------------------------------------------------

def build_questions(n: int = 60, seed: int = 0) -> List[OmniQuestion]:
    """
    A balanced set spanning video-only, audio-only, and cross-modal needs.

    Balance is load-bearing. If every question needs only audio, the video
    reliance figure is computed from nothing and its error bars swallow the
    signal. A third each keeps all three diagnostics meaningful.
    """
    templates = [
        ("what_worn", "What is the person wearing?", "a red jacket",
         {"video"}, "visual"),
        ("count_people", "How many people are visible?", "three",
         {"video"}, "visual"),
        ("what_said", "What did the speaker say the deadline was?", "friday",
         {"audio"}, "auditory"),
        ("tone", "What tone of voice did the speaker use?", "frustrated",
         {"audio"}, "auditory"),
        ("who_spoke", "Which person on screen was speaking?",
         "the woman on the left", {"video", "audio"}, "cross-modal"),
        ("said_while", "What did he say while pointing at the chart?",
         "revenue doubled", {"video", "audio"}, "cross-modal"),
    ]

    rng = random.Random(seed)
    questions = []
    for i in range(n):
        key, text, answer, requires, category = templates[i % len(templates)]
        questions.append(OmniQuestion(
            qid=f"{key}-{i:03d}",
            question=text,
            answer=answer,
            requires=set(requires),
            category=category,
        ))
    rng.shuffle(questions)
    return questions


def simulated_model(
    question: OmniQuestion,
    condition: str,
    video_skill: float,
    audio_skill: float,
    fusion_skill: float,
    rng: random.Random,
) -> str:
    """
    A stand-in model with *configurable* per-modality competence.

    This exists to prove the harness works. A real modality-ignoring model is
    hard to obtain on demand; a simulated one with `video_skill=0.0` is
    trivial, and lets `tests/test_omni_eval.py` assert that the diagnostics
    ACTUALLY DETECT it rather than merely claiming they would.

    A harness that has never been shown to catch the bug it targets is a
    harness you are trusting on faith.

    Args:
        video_skill: P(correct) on video-only questions when video is present.
        audio_skill: Likewise for audio.
        fusion_skill: P(correct) on cross-modal questions when BOTH are present.
            Set to 0 to simulate a model that never fuses.
    """
    has_video = condition in ("both", "video_only")
    has_audio = condition in ("both", "audio_only")

    needs_video = "video" in question.requires
    needs_audio = "audio" in question.requires

    if question.is_cross_modal:
        # Needs both present AND the ability to combine them.
        p = fusion_skill if (has_video and has_audio) else 0.0
    elif needs_video:
        p = video_skill if has_video else 0.0
    elif needs_audio:
        p = audio_skill if has_audio else 0.0
    else:
        p = 0.5

    # A wrong answer is a plausible distractor, not silence — a model that
    # said nothing would be trivially detectable and is not the failure we
    # are modelling.
    if rng.random() < p:
        spoken = rng.choice([question.answer,
                             f"uh, {question.answer}",
                             f"I think it's {question.answer}."])
        return spoken
    return rng.choice(["a blue hat", "seven", "monday", "cheerful", "nobody"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="HF model id. Omit to run the simulated model.")
    parser.add_argument("--dataset", default=None,
                        help="JSON benchmark. Omit for the offline set.")
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--asr-wer", type=float, default=0.05,
                        help="Measured WER of your transcriber. Drives the "
                             "reported error band.")
    parser.add_argument("--video-skill", type=float, default=0.8,
                        help="Simulated model only.")
    parser.add_argument("--audio-skill", type=float, default=0.8,
                        help="Simulated model only.")
    parser.add_argument("--fusion-skill", type=float, default=0.7,
                        help="Simulated model only. Set 0 to simulate a model "
                             "that never fuses — and watch the harness catch it.")
    parser.add_argument("--output", default="omni_eval_results.json")
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset) as handle:
            rows = json.load(handle)
        questions = [
            OmniQuestion(
                qid=str(row.get("qid", i)),
                question=row["question"],
                answer=row["answer"],
                requires=set(row.get("requires", ["video", "audio"])),
                category=row.get("category", "general"),
            )
            for i, row in enumerate(rows)
        ]
    else:
        questions = build_questions(args.questions)

    bar = "=" * 74
    print(bar)
    print("  Video-speech-to-speech evaluation — modality ablation")
    print(bar)
    print(f"  questions   {len(questions)}")
    print(f"  dataset     {args.dataset or 'synthetic (offline)'}")
    print(f"  model       {args.model or 'simulated'}")
    print(f"  ASR WER     {args.asr_wer:.1%}")
    if not args.model:
        print(f"  skills      video={args.video_skill} audio={args.audio_skill} "
              f"fusion={args.fusion_skill}")
    print(bar)

    model = processor = None
    if args.model:
        import os
        try:
            import torch
        except ImportError:
            print("\n[preflight] PyTorch is not installed. Install it with:")
            print("            uv pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu121\n")
            sys.exit(1)
        if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
            print("\n[preflight] Evaluating a real omni model needs a GPU.")
            print("            Omit --model to validate the harness on CPU.")
            print("            Or rent one:")
            print("              uv run runpod/runpod_ctl.py run "
                  "09_vss/04_omni_eval \\")
            print("                  --collect --wait --terminate --yes\n")
            sys.exit(1)
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model,
                                                  trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                          dtype=torch.bfloat16,
                                          device_map="auto")

    report = AblationReport()
    rng = random.Random(12345)
    records = []

    for question in questions:
        for condition in CONDITIONS:
            if model is None:
                spoken = simulated_model(question, condition,
                                         args.video_skill, args.audio_skill,
                                         args.fusion_skill, rng)
            else:
                # A real run masks the inputs per condition and transcribes the
                # spoken output. Left explicit rather than hidden, because the
                # masking IS the experiment: silence the audio track, blank the
                # video frames, and everything else stays identical.
                raise NotImplementedError(
                    "Wire your model's generate() here, masking inputs per "
                    "condition, then transcribe the spoken output before "
                    "scoring. See the README."
                )

            scored = score_spoken_response(spoken, question.answer,
                                           asr_wer=args.asr_wer)
            report.add(condition, scored["correct"], question.is_cross_modal,
                       scored["asr_uncertainty"])

            if condition == "both":
                records.append({
                    "qid": question.qid,
                    "category": question.category,
                    "requires": sorted(question.requires),
                    "cross_modal": question.is_cross_modal,
                    "spoken": spoken,
                    "expected": question.answer,
                    "correct": scored["correct"],
                })

    print(report.summary())

    with open(args.output, "w") as handle:
        json.dump({
            "accuracy": {c: report.accuracy(c) for c in CONDITIONS},
            "fusion_gain": report.fusion_gain,
            "video_reliance": report.video_reliance,
            "audio_reliance": report.audio_reliance,
            "prior_floor": report.prior_floor,
            "asr_band": report.asr_band,
            "records": records,
        }, handle, indent=2)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
