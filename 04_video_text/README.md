# 08 — Video-Text Training

Video is where every memory technique in this course stops being optional.

A 448×448 frame becomes ~256 visual tokens after Qwen2.5-VL's 2×2 patch merger.
So a 64-frame clip is **16,384 visual tokens** before a single word of the
prompt — and attention is quadratic:

| Frames | Visual tokens | Attention cost vs 8 frames |
|---|---|---|
| 8 | 2,048 | 1× |
| 16 | 4,096 | 4× |
| 32 | 8,192 | 16× |
| 64 | 16,384 | 64× |
| 128 | 32,768 | 256× |

Doubling the frames does not double the cost. That table is the reason this
topic has five subsections instead of one.

> **The through-line of this whole course.** ZeRO shards what the model *is*
> and pays in communication. Token compression shrinks what the model *looks
> at* and pays in fidelity. Streaming memory bounds what the model *retains*
> and pays in recall of the distant past. Three different axes, one identical
> bargain: **you never get memory for free, you only choose the currency.**

## The track

Read them in order. Each one exists because the previous one runs out of road.

| # | Subsection | The question it answers | GPU? |
|---|---|---|---|
| — | [`01_hf_baseline/`](01_hf_baseline/) | How does video SFT work at all? (LLaVA, 2024) | yes |
| 1 | [`02_qwen25vl/`](02_qwen25vl/) | How does a *modern* video model represent time? | yes |
| 2 | [`03_token_compression/`](03_token_compression/) | The clip does not fit. What do I throw away? | partly |
| 3 | [`04_streaming_memory/`](04_streaming_memory/) | The video has no *end*. Now what? | **no** |
| 4 | [`05_video_eval/`](05_video_eval/) | Did compression break understanding? | **no** |

**Where each one runs out of road:**

- **LLaVA baseline** — fixed frame count, fixed resolution, frame-*index*
  positions. Sample 16 frames from a 10-second clip and from a 10-minute clip
  and the model sees identical position information. Duration questions are
  unanswerable *in principle*.
- **Qwen2.5-VL baseline** — fixes time, still loads every token. A long clip
  still OOMs.
- **Token compression** — shrinks by a constant *factor*. Halve the tokens and
  a two-hour video is still twice a one-hour video. For any fixed ratio there
  is a video long enough to kill you.
- **Streaming memory** — constant *bound*, not a factor. Runs forever. Pays for
  it by forgetting detail.
- **Evaluation** — none of the above tell you whether the model still
  understands. The loss curve certainly won't.

## Run it without a GPU

Unusually for this topic, **the two most interesting subsections are fully
CPU-runnable**, because their substance is algorithms rather than weights:

```bash
uv run 04_video_text/03_token_compression/token_compression.py   # the token-budget arithmetic
uv run 04_video_text/04_streaming_memory/stream_infer.py --frames 20000
uv run 04_video_text/05_video_eval/video_mme_eval.py --dry-run
```

That last one is worth running just to see the chance baseline come out at
25%. It scored 100% once — see [`05_video_eval/`](05_video_eval/).

Verify the algorithms are correct, not merely runnable:

```bash
uv run tests/test_token_compression.py   # 30 checks — ToMe, FastV, DyCoke
uv run tests/test_star_memory.py         # 23 checks — bounded AND remembers
uv run tests/test_video_eval.py          # 39 checks — no answer leakage
```

## Run it on a GPU

Everything here uses **`uv`** for packages and **`deepspeed`** for the training
runs. Never bare `pip` or bare `python` for setup.

### CoreWeave / any SLURM cluster

Every subsection ships a `run_deepspeed.sh`:

```bash
cd 04_video_text/02_qwen25vl && sbatch run_deepspeed.sh
```

Build the environment **once on a login node** — compute nodes usually have no
egress:

```bash
uv venv ~/myenv && source ~/myenv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed transformers accelerate peft datasets \
    qwen-vl-utils opencv-python-headless
```

Adjust `--partition` to match your cluster (`sinfo` lists them) and point
`HF_HOME` at scratch — `$HOME` is usually a small NFS quota and a multi-GB
model download into it fails slowly.

### RunPod

There is no SLURM on RunPod, so the pod lifecycle is driven by API instead.
Each subsection is registered separately, so you rent only what it needs:

```bash
export RUNPOD_API_KEY=...        # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py recommend 04_video_text/03_token_compression
uv run runpod/runpod_ctl.py run 04_video_text/03_token_compression \
    --collect --wait --terminate --yes
```

`--terminate` shuts the pod down in a `finally` block, so a crash, a network
failure or Ctrl-C still stops the billing. There is also an in-pod watchdog
that needs no API key. **Always confirm afterwards:**

```bash
uv run runpod/runpod_ctl.py pods          # should say "Nothing is billing."
uv run runpod/runpod_ctl.py terminate --all
```

| Subsection | Min VRAM | GPUs | Disk | Launcher |
|---|---|---|---|---|
| `04_video_text` (LLaVA baseline) | 48 GB | 2 | 120 GB | `deepspeed` |
| `04_video_text/02_qwen25vl` | 24 GB | 2 | 100 GB | `deepspeed` |
| `04_video_text/03_token_compression` | 24 GB | 1 | 60 GB | `deepspeed` |
| `04_video_text/04_streaming_memory` | 24 GB | 1 | 60 GB | `python` |
| `04_video_text/05_video_eval` | 24 GB | 1 | 60 GB | `python` |

Subsections 3 and 4 use `python`, not `deepspeed`, and that is deliberate:
streaming inference is inherently sequential and evaluation is a series of
short `generate()` calls. Neither has an optimizer to shard, so the DeepSpeed
launcher would add process-group setup and buy nothing. Using a distributed
launcher where there is nothing to distribute is cargo cult, not rigour.

See [`runpod/README.md`](../runpod/README.md) and
[`SECURITY.md`](../SECURITY.md) for the full posture — notably that the pod is
never given your API key.

## Cost discipline

Debug on the cheap card. Subsection 2's measurement sweep runs on a single
24 GB GPU at roughly $0.22/hr; the LLaVA baseline wants 2×48 GB. A shape error
found on the small rig costs 40× less than the same error found on the big one.

## Reading list

The papers behind each subsection, in the order the track uses them:

- Bai et al. **Qwen2.5-VL Technical Report** (2025) — dynamic resolution,
  absolute-time M-RoPE. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- Bolya et al. **Token Merging: Your ViT But Faster** (ICLR 2023) — ToMe.
  [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- Chen et al. **An Image is Worth 1/2 Tokens After Layer 2** (ECCV 2024) —
  FastV. [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- Shao et al. **A Survey of Multimodal Long-Context Token Compression**
  (TMLR 2026). [arXiv:2507.20198](https://arxiv.org/abs/2507.20198)
- Zhang et al. **Flash-VStream: Memory-Based Real-Time Understanding for Long
  Video Streams** (2024) — STAR memory.
  [arXiv:2406.08085](https://arxiv.org/abs/2406.08085)
- Fu et al. **Video-MME** (2024).
  [arXiv:2405.21075](https://arxiv.org/abs/2405.21075)
- Wu et al. **LongVideoBench** (2024).
  [arXiv:2407.15754](https://arxiv.org/abs/2407.15754)
