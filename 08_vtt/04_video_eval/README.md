# 08.4 — Evaluation: did compression break understanding?

The three subsections before this one all make the same promise: *you can now
fit more video.* **None of them can tell you whether the model still
understands it.**

Compression is lossy by construction, and the loss curve will not warn you — a
model trained on over-compressed video converges perfectly happily to a worse
model. Training loss measures fit to your data. It cannot measure whether you
deleted the evidence.

So the compression ratio is not a hyperparameter you tune on loss. It is one
you tune on a benchmark that specifically probes what compression destroys.

## The failure mode that shapes this harness

Video benchmarks have a notorious problem: **many questions are answerable from
one frame.** *"What colour is the car?"* needs no video at all. A model that
ignores time entirely — or a compressor that has thrown all of it away — can
post a respectable Video-MME score. Video-MME's own authors report the
single-frame baseline for exactly this reason.

So this harness splits questions into two buckets and reports them **apart**:

| Bucket | Categories | Answerable from one frame? |
|---|---|---|
| single-frame | `perception` | yes |
| temporal | `counting`, `ordering`, `causality`, `duration` | no |

```
  single-frame        70.2%   (8 questions)
  temporal            35.1%   (32 questions)
  TEMPORAL GAP       +35.1%   <- the number that matters
```

**The gap is the only figure that tells you whether your temporal path works.**
A model at 70% single-frame and 35% temporal has a broken vision-time pipeline
— and its 52% average hides that completely. *That average is the number people
publish.*

### `duration` is the sharpest diagnostic in the set

And it is the one that motivates [`../01_qwen25vl_baseline/`](../01_qwen25vl_baseline/).

A model with frame-*index* positions rather than absolute-time positions cannot
answer duration questions **in principle**: sampling 16 frames from a 10-second
clip and a 10-minute clip produces identical position information, so the
evidence for "how long" was destroyed before the model saw anything.

Near-chance duration accuracy alongside healthy perception accuracy is that
architecture, diagnosed.

## The bug this file shipped with

The first version scored a **random-guess baseline at 100%**.

`build_synthetic_questions` seeds `random.Random(0)` and draws one value per
question to place the correct answer. The evaluation loop also seeded
`random.Random(0)`. Both drew from identical streams in lockstep, so every
"random" guess landed exactly on the correct letter.

Nothing crashed. Nothing warned. The only reason it was caught is that
`--dry-run` printed a number that was obviously impossible.

> **Always run the chance baseline first.** If random guessing does not land
> near 25%, the harness is leaking answers and every number after it is
> meaningless. Correlated RNG seeds between data generation and evaluation are
> a classic silent leak — the real-world equivalent is an eval script that
> reuses the dataset's shuffle seed.

`tests/test_video_eval.py` now regresses this across four independent seeds,
and also checks that no constant-guess strategy ("always answer B") beats
chance.

## Answer parsing is not a detail

Models say:

```
Looking at option A, that seems wrong. The answer is C.
```

A naive `if "A" in response` scores that as **A**, and silently costs you real
accuracy the model actually earned. `parse_answer` is strict-then-lenient: try
the explicit formats first (`answer is (C)`, `C.`, `(C)`), fall back to a bare
letter only when nothing structured is found.

Genuinely unparseable responses return `None` and are **counted and warned
about**, not silently scored wrong:

```
  WARNING: 47 responses had no parseable answer letter.
           These were scored WRONG. If the count is large the model is
           failing to follow the format, not failing the task — fix the
           prompt before believing any number above.
```

An unparsed rate above a few percent means you are measuring format
compliance, not comprehension.

## What is real and what is synthetic

**Real:** the bucketing, the scoring, the answer parsing, the report. This is
what you point at Video-MME.

**Synthetic:** the bundled questions, so the whole thing runs offline with no
gated dataset and no download. Point `--dataset` at the real thing when you
have it; nothing else changes.

```json
[{"qid": "1", "category": "ordering", "question": "...",
  "options": ["...", "..."], "answer": "B",
  "duration_bucket": "long", "video": "clips/1.mp4"}]
```

Unknown categories default to **temporal**, the conservative choice:
mis-bucketing a temporal question as single-frame inflates the single-frame
score and shrinks the gap, hiding the very failure this harness exists to
expose.

## Runs on CPU — no model, no download

```bash
uv run video_mme_eval.py --dry-run       # the chance baseline
uv run tests/test_video_eval.py          # 39 checks
```

## Running against a real model

### Setup (uv — never bare pip)

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 08_vtt/04_video_eval
uv sync                    # creates .venv, installs the LOCKED versions
uv run python video_mme_eval.py
```

`uv run` uses the project environment directly, so there is no
`activate` step. `uv sync --extra tracking` adds Weights & Biases,
which stays optional.

The lock is the point: everyone who clones resolves to identical
versions, instead of whatever `uv pip install` finds that day.
Regenerate deliberately with `uv lock --upgrade`.

<details>
<summary>Manual route, without the project</summary>

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed transformers accelerate opencv-python-headless
```

PyPI's `torch` ships CUDA wheels now, so no `--index-url` is
needed; pinning cu121 gives an older CUDA than the default wheel.
</details>


### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh
MODEL=Qwen/Qwen2.5-VL-7B-Instruct DATASET=videomme.json sbatch run_deepspeed.sh
```

The script runs the chance baseline first, then sweeps frame budgets (8, 16,
32, 64). **Watch the temporal gap across the sweep, not the overall average.**
The gap is what widens when compression has thrown away the time axis.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 08_vtt/04_video_eval \
    --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods     # confirm: "Nothing is billing."
```

> **Why `python` and not the `deepspeed` launcher here?**
> Evaluation is a series of short `generate()` calls — no optimizer, no
> gradients, nothing to shard. The win comes from batching questions, not from
> sharding the model. The DeepSpeed launcher would add process-group setup and
> buy nothing.

## The workflow this completes

```
01  train a model that can represent time
02  compress until it fits          ─┐
04  evaluate                          ├─ loop until the temporal gap stops widening
02  compress harder                 ─┘
03  when the video has no end, bound the memory instead
```

Compression ratio is not a setting you pick once. It is a dial you turn while
watching the gap.

## References

- Fu et al. *Video-MME: The First-Ever Comprehensive Evaluation Benchmark of
  Multi-modal LLMs in Video Analysis.*
  [arXiv:2405.21075](https://arxiv.org/abs/2405.21075)
- Wu et al. *LongVideoBench: A Benchmark for Long-context Interleaved
  Video-Language Understanding.*
  [arXiv:2407.15754](https://arxiv.org/abs/2407.15754)
