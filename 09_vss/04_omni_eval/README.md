# 09.4 — Does it actually use both streams?

You have built an omni model. It takes video and speech in, it speaks back, and
it scores **62%** on a benchmark. Two completely different systems produce that
number:

| | |
|---|---|
| **A** | genuinely fuses what it sees with what it hears |
| **B** | ignores the video entirely and answers from audio alone |

**Accuracy cannot tell them apart.** And B is what you get *by default* —
because during training one modality is usually sufficient for most examples, so
the cheapest way to reduce loss is to learn one stream well and treat the other
as noise.

This is the omni-modal analogue of the single-frame problem in
[`../../08_vtt/04_video_eval/`](../../08_vtt/04_video_eval/), and it is worse:
there are now **two** ways to cheat instead of one.

## The ablation grid

Run the same benchmark four times, degrading the input:

| Condition | Input |
|---|---|
| `both` | video + audio — the real number |
| `video_only` | audio muted |
| `audio_only` | video blanked |
| `neither` | both removed — the text-prior floor |

Four numbers, four diagnostics, every one of which the single score hides:

```
fusion_gain    = both - max(video_only, audio_only)
video_reliance = both - audio_only
audio_reliance = both - video_only
prior_floor    = neither
```

**`fusion_gain` is the headline.** Near zero means *no fusion* — the model
picked whichever single stream was better and ignored the other, and its
impressive score is a single-modality score wearing an omni-modal costume.

> **Check `prior_floor` first.** Well above chance means the *benchmark* is
> broken: the questions are answerable from the text prompt alone, and every
> other number is measuring a language model. The harness refuses to report a
> fusion gain in that case rather than computing one from noise.

## It demonstrably catches the failure

A harness that has never been *shown* to catch the bug it targets is one you are
trusting on faith. So `omni_eval.py` ships a simulated model with configurable
per-modality competence, and the test suite constructs a broken one on purpose:

```
$ uv run omni_eval.py --video-skill 0 --fusion-skill 0

  both          30.0%      <- still a non-trivial headline score
  video_only     0.0%
  audio_only    30.0%
  neither        0.0%

  FUSION GAIN         +0.0%   <- both minus the better single stream
  video reliance      +0.0%

  Diagnosis:
    NO FUSION. Adding the second stream gains only +0.0%.
      The model is answering from ONE modality and ignoring the other. Its
      headline score is a single-modality score. Suspect the alignment —
      see ../02_thinker_talker/ for TMRoPE.
      Video is being IGNORED — removing it costs almost nothing.
```

`tests/test_omni_eval.py` asserts **both directions** — that a broken model is
caught *and* that a healthy one is not falsely flagged. A diagnostic that fires
on everything is as useless as one that fires on nothing.

## Scoring speech is its own problem

This family **speaks** its answers, which breaks exact-match scoring in ways
text models never had to deal with:

- the model says "twenty-three"; the reference says "23" — both correct
- the model says "uh, I think it's Paris" — correct, with disfluency
- **you must transcribe before you can score, and the ASR makes its own
  mistakes**

That last point quietly corrupts published numbers. At 5% WER, roughly 5% of
your "model errors" are not model errors — and the effect is not uniform. It
hits rare words, names, and numbers hardest, which is exactly what benchmark
answers are made of.

So `score_spoken_response` normalises aggressively (number words, disfluencies,
carrier phrases, articles) and reports an **ASR error band** alongside the score
rather than absorbing it. Subtracting it would invent a precision nobody has;
reporting it tells the reader that a 2-point gap between two systems may be
entirely transcription noise.

### A real bug this file shipped with

Normalisation has failure modes in *both* directions. Under-normalise and
"Twenty-three." scores wrong. Over-normalise and meaning is destroyed — the
first version scored **`"not Paris"` as matching `"Paris"`**, because `"paris"`
is a substring of `"not paris"`. Its own test suite caught it.

The fix was two-part, and the second half matters as much as the first:

- **negation is checked before any containment test** — negation inverts
  meaning and containment cannot see it
- **matching is token-level, not character-level** — because `"7"` *is* a
  substring of `"17"`, so character matching scores the wrong number as correct
  on exactly the short numeric answers benchmarks are full of

## Runs on CPU

```bash
uv run omni_eval.py                                  # simulated healthy model
uv run omni_eval.py --video-skill 0 --fusion-skill 0 # simulated broken model
uv run tests/test_omni_eval.py                       # 49 checks
```

## Running against a real model

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed transformers accelerate librosa soundfile opencv-python-headless
```

### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh
MODEL=Qwen/Qwen2.5-Omni-7B DATASET=omnieval.json ASR_WER=0.03 sbatch run_deepspeed.sh
```

The script runs the **harness self-check first** — a deliberately
modality-ignoring model that must be caught — before touching your real model.
If the grid cannot catch a known-broken model, every number after it is
meaningless.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 09_vss/04_omni_eval \
    --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods     # confirm: "Nothing is billing."
```

> **Why `python` and not the `deepspeed` launcher here?**
> Evaluation is a series of short `generate()` calls — no optimizer, no
> gradients, nothing to shard. The win comes from batching questions.

## Bring your own benchmark

`--dataset` takes JSON. The `requires` field is the one benchmarks usually omit
and the one the whole analysis depends on:

```json
[{"qid": "1", "question": "What did he say while pointing at the chart?",
  "answer": "revenue doubled", "requires": ["video", "audio"],
  "category": "cross-modal"}]
```

Unknown categories default to **cross-modal**, the conservative direction:
mis-labelling a cross-modal question as single-modality inflates the
single-stream scores and *shrinks* the fusion gain, hiding the very failure this
harness exists to expose.

Real benchmarks to point it at: [OmniEval](https://arxiv.org/abs/2506.20960)
(810 audio-visual synchronized videos, 2,617 QA pairs, explicit event
grounding), LVOmniBench, and OmniACBench.

## The workflow this completes

```
02  align the two streams on one clock (TMRoPE)
04  evaluate                            ─┐
02  adjust                                ├─ loop until fusion_gain stops moving
04  evaluate                             ─┘
03  make it real-time and interruptible
```

**Fusion gain is not a number you read once. It is the number you watch while
changing the alignment.**

## References

- Wang et al. *OmniEval.* [arXiv:2506.20960](https://arxiv.org/abs/2506.20960)
- Xu et al. *Qwen2.5-Omni Technical Report.* [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)
