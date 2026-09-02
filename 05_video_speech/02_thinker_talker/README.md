# 09.2 — Thinker-Talker: two streams in, speech out

**Prerequisite:** [`../01_longcat_omni/`](../01_longcat_omni/) shows
what the frontier looks like at 560B. This subtopic is the same architecture at
a size you can actually fine-tune.

## The problem: two streams that disagree about what time it is

- **video** — 1–25 frames per second, irregular
- **audio** — 16,000 samples per second, ~50 encoder frames per second

Concatenate them and the transformer sees a flat list of tokens with no idea
which frame goes with which sound. *"What did he say while pointing at the
whiteboard?"* is unanswerable — not because the model is undertrained, but
because the fact that pointing and saying happened **at the same moment** was
never in the input.

## TMRoPE: one clock, measured in 40 ms ticks

Qwen2.5-Omni's answer is a single rule that does all the work:

> **one temporal position ID == 40 milliseconds of real time** — for *every*
> modality.

Not one ID per token. Not one ID per frame. One ID per 40 ms of wall clock. So
a video frame at t=1.00 s and an audio frame at t=1.00 s both get temporal
position **25**. They are *the same position* to attention, and co-occurrence
lives in the position encoding instead of being left for the model to infer.

Everything else follows:

| Modality | temporal | spatial |
|---|---|---|
| text | t = h = w, +1 per token | — (degenerates to 1-D RoPE) |
| audio | one ID per 40 ms frame | h, w pinned to t |
| image | constant (one instant) | h, w across the patch grid |
| video | **from the frame's real timestamp** | h, w across the patch grid |

### The trap: numbering frames by index

The obvious approach — number frames 0, 1, 2, … — has a genuinely nasty
property. Run `uv run tmrope.py`:

```
  Frame-INDEX positions drift from the audio clock:
      1 fps @  60.0s   audio ID 1500   naive video ID 60      drift   1440
      2 fps @  60.0s   audio ID 1500   naive video ID 120     drift   1380
      5 fps @  60.0s   audio ID 1500   naive video ID 300     drift   1200
     25 fps @  60.0s   audio ID 1500   naive video ID 1500    drift      0
```

**At exactly 25 fps it is correct.** 25 fps is 40 ms per frame, which is exactly
the tick, so index and time coincide.

Test on 25 fps footage and everything works. Ship it. Then someone feeds it
2 fps and by the one-minute mark the two streams describing the same instant are
1,380 IDs apart — and nothing raises. `tests/test_tmrope.py` asserts both halves:
that the coincidence is real, and that it does not generalise.

### The 2-second interleave

Sharing a clock is necessary and not sufficient. Correctly-numbered tokens can
still sit 10,000 apart in the sequence. So the layout is chunked by real time —
visual first, then that same window's audio:

```
[ video 0-2s ][ audio 0-2s ][ video 2-4s ][ audio 2-4s ]...
```

Measured effect, from `uv run tmrope.py`: worst-case cross-modal gap drops from
**142 tokens to 42** on a 6-second clip.

Two seconds is roughly the span of a spoken clause or a single gesture — the
natural unit of co-occurrence. Smaller chunks put co-occurring tokens closer but
fragment each stream's local coherence; larger ones do the reverse.

## Thinker-Talker: emitting speech without wrecking reasoning

A model that replies in speech has two jobs that pull against each other:

| Job | Wants |
|---|---|
| reason about what was seen and heard | a big language model |
| emit audio tokens at 50 Hz, in order | low latency, stability |

One autoregressive head doing both interferes. The classic symptom: speech
quality degrades exactly when reasoning gets hard — the model spends its
capacity deciding *what* to say and the prosody falls apart mid-sentence. Users
read that as the model being unsure of itself.

**Thinker** is a full language model: consumes the interleaved sequence, emits
text plus hidden states. **Talker** is a smaller dual-track autoregressive
model: consumes the Thinker's **hidden states** — not its text — and emits audio
tokens.

> ### Why hidden states and not text
> If the Talker read the emitted *text*, it would have to wait for a token to be
> decoded before speaking, and it would lose everything text does not encode:
> hesitation, emphasis, confidence. Hidden states carry that, and arrive a step
> earlier — which is a meaningful slice of the latency budget.

**The training consequence people get wrong:** the Talker's gradient flows
through the Thinker's hidden states. Freeze the Thinker completely and the
Talker can only learn to decode a representation that is not adapting to it.
Unfreeze everything and the speech loss steers the reasoning model, degrading
what it knew. **LoRA on the Thinker is the middle path**, and it is why
`train_omni.py` is built the way it is.

## Memory

| Model | Setup | VRAM |
|---|---|---|
| Qwen2.5-Omni-3B | LoRA + ZeRO-3 | ~24 GB (one card) |
| Qwen2.5-Omni-7B | LoRA + ZeRO-3 | ~40 GB |
| Frontier omni (100B+) | multi-node | [`../01_longcat_omni/`](../01_longcat_omni/) |

An omni model is **four models resident at once** — language backbone, vision
encoder, audio encoder, speech decoder — which is why `ds_config.json` uses
ZeRO-3 despite its 1.5× communication cost (3Ψ vs 2Ψ).

Two token streams also make the sequence longer than a video-only model at the
same clip length: **25 audio tokens per second on top of the video**. A
30-second clip is ~750 audio tokens before a single frame.

## Runs on CPU — and the important part does

TMRoPE is integer arithmetic, so it needs no GPU and no download:

```bash
uv run tmrope.py                 # the shared clock, and the drift without it
uv run tests/test_tmrope.py      # 58 checks
```

Those checks assert *properties*: that video and audio at the same instant share
an ID at any frame rate, that offsets compose across chunk boundaries, that
interleaving does not renumber anything, and that naive indexing drifts
unboundedly. Getting this wrong raises nothing — the model trains happily and is
simply unable to relate the streams — which is exactly why it is worth an
arithmetic proof.

## Running it

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 05_video_speech/02_thinker_talker
uv sync                    # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=2 train_omni.py
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
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed transformers accelerate peft datasets
uv pip install librosa soundfile opencv-python-headless
```

The `--index-url` is **required**, and matches what the lock pins.
PyPI's default `torch` is a CUDA 13 wheel and reports
`cuda.is_available() == False` on a pre-CUDA-13 driver.
</details>


### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh
MODEL=Qwen/Qwen2.5-Omni-7B NUM_GPUS=4 sbatch run_deepspeed.sh
sbatch run_deepspeed.sh --max-steps 20        # cheap dry run
```

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 05_video_speech/02_thinker_talker \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods     # confirm: "Nothing is billing."
```

### Direct

```bash
deepspeed --num_gpus=2 train_omni.py --deepspeed ds_config.json
```

## Data

Defaults to synthetic clips with genuine temporal structure (a moving square, a
rising tone) so a cross-modal question has a ground-truth answer — random
tensors would let a disconnected vision path look identical to a working one.

Point `--data-dir` at the shared corpus (`../data`, 8 real samples) to wire in
real media; the loader is left to you and the README says so rather than
pretending otherwise.

## Next

[`../03_duplex_streaming/`](../03_duplex_streaming/) — this model answers one
turn at a time and is deaf while it speaks. Real conversation is not like that.

## References

- Xu et al. *Qwen2.5-Omni Technical Report.* [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)
- Qwen Team. *Qwen3-Omni Technical Report.* [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)
