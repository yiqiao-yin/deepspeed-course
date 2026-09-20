# 06 · Qwen3-VL

LoRA fine-tuning of **Qwen3-VL-8B-Instruct**, and the measured memory curve that
tells you how many frames fit on your card.

This is the successor to [`02_qwen25vl`](../02_qwen25vl/), and the comparison is
the lesson. Qwen3-VL is not a version bump — it adds two mechanisms the earlier
model does not have, and it is 2.3× the weights.

---

## Start here, with no GPU

```bash
cd 04_video_text/06_qwen3vl
uv sync
uv run verify_arch.py --compare
```

Eight seconds, **no GPU and no weights downloaded**. It builds the real module
tree on torch's meta device and reports what actually differs between the
generations:

| | Qwen2.5-VL-3B | Qwen3-VL-8B |
|---|---|---|
| class | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| parameters | 3.75 B | **8.77 B** |
| bf16 weights | 7.5 GB | **17.5 GB** |
| vision tower | 0.67 B (17.8%) | 0.58 B (6.6%) |
| **DeepStack** | no | **yes** — layers 8, 16, 24 |
| MRoPE | `mrope` | **interleaved** |

**DeepStack** injects visual features at several depths of the language model
rather than only at the input embedding. The config carries
`deepstack_visual_indexes: [8, 16, 24]` and the module tree carries a matching
`visual.deepstack_merger_list`.

## The trap this folder exists to teach

**A vision-language model is two sub-models with independent naming, and peft
matches by name.** `verify_arch.py --compare --targets gate_proj,up_proj,down_proj`
prints this:

| target | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| `gate_proj` | 36 lang + **32 VISION** | 36 lang only |
| `up_proj` | 36 lang + **32 VISION** | 36 lang only |
| `down_proj` | 36 lang + **32 VISION** | 36 lang only |

Both vision towers use a **fused `qkv`**, so `q_proj`/`k_proj`/`v_proj`/`o_proj`
match **zero** vision modules on either model. But Qwen2.5-VL's vision MLP
reuses the language model's MLP names, so adding those three targets unfreezes
32 vision modules you did not ask for. Qwen3-VL's vision MLP is
`linear_fc1`/`linear_fc2`, so it does not.

Nothing in either `config.json` says this, and **peft reports a plausible
trainable-parameter count either way**. Two seconds on the meta device is the
difference between knowing and assuming.

---

## Hardware

| | |
|---|---|
| **`verify_arch.py`** | no GPU, no download |
| **Training** | **1 × 48 GB.** A 24 GB card is not viable — see below |
| Disk | ~120 GB (the weights are ~17 GB) |
| Time | ~10 min for a short run |

### The memory curve, measured

Measured on one A40 (47.7 GB) by this script's own `--sweep`:

| frames | visual tokens | peak VRAM | % of card |
|---:|---:|---:|---:|
| 4 | 276 | 22.1 GB | 46% |
| 8 | 540 | 25.3 GB | 53% |
| 16 | 1,068 | 31.6 GB | 66% |
| 24 | 1,596 | 37.9 GB | 79% |

It is very nearly a straight line:

$$\text{peak} \approx 18.8\,\text{GB} + 11.97\,\text{GB} \times \frac{\text{visual tokens}}{1000}$$

At ~67 visual tokens per 224×224 frame:

| card | usable |
|---|---|
| 24 GB | ~4 frames — **not viable**, the floor alone nearly fills it |
| **48 GB** | **~33 frames** |
| 80 GB | ~70 frames |

**The 18.8 GB floor is the point.** It is the bf16 weights plus fragmentation,
it does not shrink with sequence length, and it is why this lab wants one large
card rather than two small ones.

---

## Environment & Local Testing

```bash
cd 04_video_text/06_qwen3vl
uv sync
uv run verify_arch.py             # the whole architecture lesson, no GPU
```

## Running it

```bash
# the memory sweep, then a short training run
uv run deepspeed --num_gpus=1 train_qwen3vl.py

# just the sweep
uv run deepspeed --num_gpus=1 train_qwen3vl.py --probe-only

# push it until it OOMs, which is the point
uv run deepspeed --num_gpus=1 train_qwen3vl.py --sweep 8,16,32,48
```

### CoreWeave (SLURM)

```bash
sbatch run_deepspeed.sh
sbatch run_deepspeed.sh --max-steps 4        # cheap dry run
squeue -u $USER
tail -f logs/qwen3vl_<jobid>.out
```

### RunPod

```bash
uv run runpod/runpod_ctl.py recommend 04_video_text/06_qwen3vl
uv run runpod/runpod_ctl.py run 04_video_text/06_qwen3vl \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods            # confirm nothing is still running
```

`--terminate` destroys the pod from your machine in a `finally`, so a crash
mid-run still shuts it down. **Always confirm with `pods`** — an idle 48 GB card
bills by the hour.

---

## Two things the sweep does on purpose

**It asserts that the input actually changed.** The first version of this sweep
printed a flat 14.2 GB at 4, 8, 16 *and* 24 frames and exited 0 — it had passed
the frames as `videos=`, the processor resampled every clip to its default frame
rate, and the model saw the same 128-token input four times. The table looked
like a memory ceiling and was one measurement repeated. The sweep now **raises**
if the sequence length does not vary.

**It counts parameters via `ds_numel`.** Under ZeRO-3, `zero.Init` partitions
each parameter and `p.numel()` returns **0** — summing it gives an 8.77 B model
a parameter count of zero. The first run found out by dividing by it. Any naive
parameter accounting changes meaning the moment `zero.Init` fires.

Related, and the reason the config is loaded before the model: **the DeepSpeed
config must exist before `from_pretrained`**, or `zero.Init` never fires and
every rank materialises the whole model. Holding `HfDeepSpeedConfig` in a live
variable is what springs it.

---

## Expected output

```
  parameters      8.77 B   vision 0.58 B (6.6%)
  (counted via ds_numel: under ZeRO-3, numel() would report 0)

  LoRA targets ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    resolve to 144 modules, 0 of them in the vision tower
    The vision tower is FROZEN -- not by choice but because its
    attention is a fused `qkv` that these names cannot match.
trainable params: 15,335,424 || all params: 8,782,459,120 || trainable%: 0.1746
```

> **Multi-GPU is not verified.** A 2-GPU run confirmed `zero.Init` fires and
> shards correctly, then hung in a `broadcast` inside it until NCCL's watchdog
> aborted — the interconnect signature described in `tests/gpu/diagnose_nccl.sh`,
> on a rented community-cloud box. Single-GPU is measured end to end; multi-GPU
> is not, and those are different claims.

---

## References

- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Qwen team, *Qwen2.5-VL Technical Report*, [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- [`02_qwen25vl`](../02_qwen25vl/) — the previous generation, for the comparison
