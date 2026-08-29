# 08.1 — Qwen2.5-VL: a model that can represent time

**Prerequisite:** [`../hf_ds_vtt_test2/`](../hf_ds_vtt_test2/) — read the LLaVA
trainer first. Nothing is hidden there, and this subsection is best understood
as a diff against it.

## Why not just use the LLaVA example?

Because of one specific, fatal limitation.

A fixed-frame model samples N frames and numbers them `0..N-1`. Sample 16
frames from a 10-second clip and 16 frames from a 10-minute lecture, and the
model receives **identical position information**. It cannot distinguish

> "he picked up the cup, then immediately drank"

from

> "he picked up the cup, and forty minutes later drank"

because the evidence was destroyed at the sampler, before the model saw
anything. Any question containing *how long*, *before*, or *after* is
unanswerable **in principle**, not merely in practice. No amount of training
fixes it.

Qwen2.5-VL aligns the temporal component of its M-RoPE positional encoding to
**absolute timestamps**. Frame 5 at t=2.0s and frame 5 at t=300.0s get
different positions. Duration survives sampling.

The second change is **native dynamic resolution**: LLaVA resizes every frame
to 336×336, so a wide establishing shot gets letterboxed and you spend visual
tokens encoding black bars. Qwen2.5-VL keeps the native aspect ratio.

| | LLaVA (2024) | Qwen2.5-VL (2025) |
|---|---|---|
| Frames | fixed N | dynamic FPS, capped |
| Resolution | fixed square | native aspect ratio |
| Temporal position | frame index | **absolute timestamp** |
| Duration questions | impossible | possible |

## What the script does

[`train_qwen25vl.py`](train_qwen25vl.py):

- **`sample_video_frames`** — two-stage sampling. Sample at a fixed *rate* so
  temporal density is constant regardless of clip length, then subsample to a
  hard `max_frames` cap so nothing OOMs. **Timestamps are preserved through
  both stages** — dropping them is the single most common way this
  architecture gets silently reduced to the one it replaced.
- **`build_time_aware_prompt`** — states the timestamps in the text as well as
  relying on M-RoPE. Belt and braces, and it makes misconfiguration *visible*:
  if the prompt says 0.0s–8.0s and the model still cannot answer "how long",
  the problem is the model, not the plumbing.
- **LoRA on the language model, vision tower frozen.** The vision tower was
  trained on far more video than you have; unfreezing it on a small dataset
  reliably makes things worse *and* costs a large slice of memory for its
  gradients and optimizer states.
- **ZeRO-3** via [`ds_config.json`](ds_config.json) — stage 3 costs 1.5× the
  communication of stage 2 (3Ψ vs 2Ψ) and is still right here, because a video
  batch needs every spare byte for activations rather than a resident copy of
  the weights.
- **Synthetic data fallback** — a bright square moving left to right at a known
  rate. Not decoration: it has genuine temporal structure, so "did the model
  learn anything about motion?" has a ground-truth answer. Random tensors
  would let a broken temporal path pass unnoticed, which is exactly the bug
  the LLaVA trainer in this repo shipped with.

## Memory, honestly

| Model | Setup | VRAM |
|---|---|---|
| Qwen2.5-VL-3B | LoRA + ZeRO-3 | ~16 GB (one consumer card) |
| Qwen2.5-VL-7B | LoRA + ZeRO-3 | ~40 GB |
| Qwen2.5-VL-72B | LoRA + ZeRO-3 | multiple 80 GB cards |

**The dominant term is not the weights.** It is the visual tokens,
quadratically. Before raising `--max-frames`, read
[`../02_token_compression/`](../02_token_compression/) — otherwise you will
buy a bigger GPU to solve a problem that a compression ratio solves for free.

`gradient_checkpointing=True` is on by default. For video that is not an
optimisation, it is usually the difference between running and OOMing.

## Running it

### Setup (uv — never bare pip)

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed transformers accelerate peft datasets \
    qwen-vl-utils opencv-python-headless
```

### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh
MAX_FRAMES=32 NUM_GPUS=4 sbatch run_deepspeed.sh   # override
```

Build the venv on a **login** node — compute nodes usually have no egress.
Adjust `--partition` (`sinfo` lists them) and point `HF_HOME` at scratch.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...

uv run runpod/runpod_ctl.py recommend 08_vtt/01_qwen25vl_baseline
uv run runpod/runpod_ctl.py run 08_vtt/01_qwen25vl_baseline \
    --dry-run --collect --wait --terminate --yes

# --dry-run caps the training step so a smoke test stays cheap;
# --terminate deletes the pod in a finally block, so a crash or
# Ctrl-C still stops the billing.
uv run runpod/runpod_ctl.py pods        # confirm: "Nothing is billing."
```

`--terminate` deletes the pod in a `finally`, so a crash or Ctrl-C still stops
the billing; an in-pod watchdog (no API key required) is the backstop.

### Direct

```bash
deepspeed --num_gpus=2 train_qwen25vl.py --deepspeed ds_config.json --max-frames 16
```

## No GPU?

This subsection genuinely needs one — it downloads a multi-GB model and needs
real VRAM for the visual tokens. `require_gpu()` stops with a clear message
rather than letting DeepSpeed die inside its CUDA extension loader.

The *algorithms* in this topic do run on CPU, and they are where the ideas are:

```bash
uv run tests/test_token_compression.py
uv run tests/test_star_memory.py
uv run 08_vtt/03_streaming_memory/stream_infer.py --frames 20000
```

## Next

[`../02_token_compression/`](../02_token_compression/) — you have a model that
represents time correctly. Now make the clip fit.
