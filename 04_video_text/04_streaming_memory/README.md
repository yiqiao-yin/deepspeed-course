# 08.3 — Streaming: watching forever in O(1) memory

Everything in [`../03_token_compression/`](../03_token_compression/) shrinks a
clip by a constant **factor**. Halve the tokens and a two-hour video is still
twice a one-hour video. **For any fixed compression ratio there exists a video
long enough to OOM you.**

A security camera does not have a length. A livestream does not have a length.
A meeting recording has one but you do not know it in advance.

So streaming asks a strictly harder question:

> **How do I watch forever in O(1) memory?**

Not "grow slowly." A system whose per-frame cost creeps upward does not degrade
gracefully at hour six — it dies.

## The three things that must be true

1. **Write is O(1).** Absorbing frame 1,000,000 costs the same as frame 1. If
   ingestion ever touches all previously seen frames, you have a batch system
   wearing a streaming costume.
2. **Read is bounded.** The context handed to the LLM has a fixed ceiling, so
   inference latency is flat and predictable.
3. **Decoupled clocks.** Ingestion runs at the camera's rate; questions arrive
   at the user's rate; neither blocks the other.

Point 3 is the one that gets designed away. If answering blocks ingestion you
drop frames — and the memory you are so carefully maintaining is now a memory
of a video that did not happen.

## STAR memory

*Flash-VStream (Zhang et al., 2024)* formalises the bargain your own memory
makes: you cannot replay last Tuesday frame by frame, but you know what
happened. **Detail decays; structure survives.**

Four buffers, each with a hard cap, each at a different level of abstraction:

| Buffer | Size | Pooling | Holds | Overflow rule |
|---|---|---|---|---|
| `M_spa` spatial | 1 frame | 8×8 | the vivid present | FIFO |
| `M_tem` temporal | 25 clusters | 4×4 | events that happened | weighted k-means |
| `M_abs` abstract | 25 entries | 1×1 | the semantic gist | momentum decay |
| `M_ret` retrieved | 3 frames | 8×8 | detail pulled back on demand | recomputed |

Note the trade running down that list: **as the retention window gets longer,
the spatial resolution gets coarser.** One frame ago you keep 8×8. A thousand
frames ago you keep a cluster centroid. *That gradient is the algorithm.*

### Why the weighted k-means is weighted

Plain k-means treats every point equally, which is exactly wrong here. After a
few consolidation rounds some entries are single frames and others are
centroids already standing for fifty. Unweighted clustering lets a one-frame
blip drag a centroid as hard as a minute of footage — rare noise gets
over-represented, and the long steady event gets smeared.

Weights make every *original* frame count once, forever, no matter how many
rounds it has survived. Same invariant as size-weighting in ToMe.

The test asserts it: after 2,000 frames, `temporal_w.sum() == 2000`. A shortfall
means consolidation is **discarding** frames rather than **merging** them.

### Why `M_ret` exists

Buffers 1–3 alone have a fatal flaw: **consolidation is irreversible.** Once an
event is a 4×4 centroid the detail is gone, and *"what colour was the car that
passed at 3pm?"* is unanswerable.

`M_ret` fixes it. Find the largest temporal clusters — the events that mattered
— and pull the nearest **actual** frames back out of the raw buffer at full
resolution. Compressed long-term structure tells you *where* to look; the raw
buffer still has the pixels.

That is retrieval over a lossy index. The same move a RAG system makes over a
vector store.

## It actually works

```
$ uv run stream_infer.py --frames 4000 --query-every 1000 --dim 512

  frame    1,000  |  write 1.664 ms (first 100 frames: 1.659 ms)
  frame    2,000  |  write 1.568 ms (first 100 frames: 1.659 ms)
  frame    3,000  |  write 1.523 ms (first 100 frames: 1.659 ms)
  frame    4,000  |  write 1.502 ms (first 100 frames: 1.659 ms)

  4,000 frames in 6.9s (583 frames/s)
  final context: 306 tokens
  a naive system would hold 256,000 tokens — 837x more
```

Write time flat. Context flat at 306 tokens. Push it to 20,000 frames — nearly
three hours of video at 2 fps — and it is still 306.

## Bounded is not enough

A buffer that satisfies the O(1) requirement by throwing everything away is
trivially "bounded" and completely useless. So `tests/test_star_memory.py`
asserts **both**:

- **Bounded** — context size byte-identical at 200, 500, 1,000 and 2,000
  frames; every buffer within its cap.
- **Remembers** — a distinctive event written 1,500 frames ago is still
  recoverable above the noise floor, and beats a random direction.

Passing one is easy. Passing both is the actual engineering.

## Runs on CPU — no model, no download

```bash
uv run stream_infer.py --frames 20000     # watch the flat line
uv run star_memory.py                     # the same demo, standalone
uv run tests/test_star_memory.py          # 23 checks
```

The synthetic stream has *real temporal structure* — slow drift within scenes,
hard cuts every ~500 frames — so the clustering has genuine events to find.
Pure noise would let a completely broken consolidation step look identical to a
working one.

## Running with a real model

### Setup (uv — never bare pip)

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 04_video_text/04_streaming_memory
uv sync                    # creates .venv, installs the LOCKED versions
uv run python stream_infer.py
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
uv pip install deepspeed transformers accelerate opencv-python-headless
```

The `--index-url` is **required**, and matches what the lock pins.
PyPI's default `torch` is a CUDA 13 wheel and reports
`cuda.is_available() == False` on a pre-CUDA-13 driver.
</details>


### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh
FRAMES=50000 sbatch run_deepspeed.sh
```

The job requests only 32 GB of host RAM, and that is part of the demonstration:
if RSS climbs over the run, something is retaining frames and the O(1) property
is broken.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 04_video_text/04_streaming_memory \
    --dry-run --collect --wait --terminate --yes
# --dry-run caps the training step so a smoke test stays cheap;
# --terminate deletes the pod in a finally block, so a crash or
# Ctrl-C still stops the billing.
uv run runpod/runpod_ctl.py pods     # confirm: "Nothing is billing."
```

> **Why `python` and not the `deepspeed` launcher here?**
> Streaming inference is inherently sequential — frames arrive in order — so
> there is no optimizer to shard and no gradient to reduce. The DeepSpeed
> launcher would set up a process group and buy nothing. Scale by running more
> streams, not by sharding one. Using a distributed launcher where there is
> nothing to distribute is cargo cult, not rigour.

## The honest trade

Every technique in this course trades memory for something.

- ZeRO trades **communication** for memory.
- Token compression trades **fidelity** for memory.
- STAR trades **detailed recall of the distant past** for the ability to run
  forever.

You cannot ask this system what the licence plate was at minute 40. That
information is genuinely gone. If you need it, you need a different system —
one that writes frames to storage and indexes them, which is a database problem
rather than a model problem.

## Next

[`../05_video_eval/`](../05_video_eval/) — you have compressed and you have
bounded. Did the model survive it? The loss curve will not tell you.

## Reference

Zhang et al. *Flash-VStream: Memory-Based Real-Time Understanding for Long
Video Streams.* [arXiv:2406.08085](https://arxiv.org/abs/2406.08085)
