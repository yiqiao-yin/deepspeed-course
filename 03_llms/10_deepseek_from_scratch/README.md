# 03_llms/10_deepseek_from_scratch — Multi-head Latent Attention

Every other topic in this section *uses* a pretrained model. This one builds an
architecture: **Multi-head Latent Attention**, the mechanism DeepSeek-V2
introduced to make long context affordable, implemented from the paper in plain
PyTorch and compared against the two alternatives it replaces.

It belongs next to `01_llm_finetuning`, which analyses MLA inside GLM-5.3 and
reports a 57× KV-cache saving computed from a config file. This folder is where
you can see *why* that number is what it is — and `tests/test_mla.py` checks
that this implementation reproduces it independently.

Everything architectural runs on a CPU. Start with `mla.py`.

## The idea in one table

Generation caches keys and values for every token seen so far, and that cache is
what makes long context expensive:

| | cache per token per layer |
|---|---|
| **MHA** | `2 × n_heads × head_dim` |
| **GQA** | `2 × n_kv_heads × head_dim` |
| **MLA** | `kv_lora_rank + qk_rope_head_dim` |

GQA shrinks the cache by sharing K/V across query heads — fewer distinct keys
and values, and some capacity goes with them. **MLA caches a low-rank latent and
reconstructs K and V from it**, so the cache stops mentioning the head count at
all:

```
 n_heads  kv_heads        mha        gqa        mla
       8         2      1,024        256         80
      16         4      2,048        512         80
      32         8      4,096      1,024         80
      64        16      8,192      2,048         80
```

MHA and GQA scale with heads. MLA does not move. You *can* flatten the GQA
column by pinning `n_kv_heads`, and that is the trade made explicit: its cache
shrinks because fewer distinct keys exist, not because they are stored more
cleverly.

## Two details that are easy to get wrong

**Decoupled RoPE.** Rotary embeddings do not commute with the low-rank
reconstruction. Rotate the reconstructed key and you can no longer fold the
up-projection into the query, so MLA stops being fast. The architecture
therefore splits the key: a compressed part carrying content, and a small
**decoupled** part carrying position, cached separately and shared across heads.
That second term is why the cache is `512 + 64 = 576` in GLM-5.3 rather than 512.

**Matrix absorption.** At inference `W_UK` folds into the query once, ahead of
time, so keys are never reconstructed — you attend directly in the latent space.
This is what makes MLA fast rather than merely small. `forward(absorbed=True)`
does it, and the test suite pins the identity, because an optimisation that
changes the answer is the failure mode that matters:

```
max |naive - absorbed| = 2.38e-07
```

## Measured: does the compression cost accuracy?

`mla.py` gives the cache sizes, which are exact arithmetic. What arithmetic
cannot tell you is what the compression costs, so
`train_deepseek_from_scratch.py` trains the same tiny language model three
times, changing only the attention module.

The task is **induction**: a marked pattern appears early and again late, and
predicting the second occurrence requires attending to the first. That is
deliberate — a task solvable from token frequencies would let all three tie, and
the tie would mean nothing.

2 layers, hidden 256, 8 heads, 8 epochs, 8,192 sequences of length 64:

| variant | cache/token | params | accuracy | loss |
|---|---|---|---|---|
| mha | 512 | 1,609,472 | 25.9% | 3.175 |
| gqa | 128 | 1,412,864 | 26.0% | 3.159 |
| **mla** | **40** | 1,396,800 | **28.3%** | **3.075** |

Chance is **1.6%** / 4.159 nats on a 64-token vocabulary, so all three learned
the task.

**Read this carefully.** Across three seeds (42/43/44) the accuracy differences
are **within noise** — MLA ranged 26.1–28.3%, MHA 25.6–25.9%. What held in all
three runs is the *loss* ordering, `mla < gqa < mha`. The defensible claim is
that MLA cached **12.8× less without costing accuracy here**, not that it is
better. One small model on one synthetic task is not a ranking.

## Hardware requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | 8 GB | Any CUDA card. The model is ~1.6M parameters; memory is the O(seq²) attention matrix, so `--seq-len` is the knob |
| GPUs | 1 | The comparison is about the KV cache, which data parallelism does not change |
| Disk | < 1 GB | Nothing is downloaded; sequences are generated with torch |
| Host RAM | 8 GB | Data is generated in-process |

**No GPU?** The architecture is the point, and it needs none:

```bash
uv run mla.py                              # cache table + absorption identity
uv run ../../tests/test_mla.py             # 19 property assertions
uv run train_deepseek_from_scratch.py --list-variants
ALLOW_CPU=1 uv run train_deepseek_from_scratch.py --variant all --epochs 2 \
    --train-seqs 512 --seq-len 64 --layers 2 --hidden 128
```

Without `ALLOW_CPU=1` the script stops at a preflight and tells you the above,
instead of dying inside torch's extension loader with `CUDA_HOME environment
variable is not set`.

## Environment & Local Testing

```bash
cd 03_llms/10_deepseek_from_scratch
uv sync                        # creates .venv from the COMMITTED uv.lock
```

`uv.lock` is committed, so every reader installs the same versions. torch is
pinned to the cu128 index: PyPI's default torch is a CUDA 13 wheel that installs
cleanly on a 550/570 driver and then reports `cuda.is_available() == False`
while `nvidia-smi` happily shows the card.

## Running it

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh --variant all        # arguments forwarded via "$@"
squeue -u $USER
tail -f logs/deepseek_from_scratch_<jobid>.out
scancel <jobid>
```

Cheap dry run first:

```bash
sbatch run_deepspeed.sh --max-steps 20
```

### RunPod (creates the pod and shuts it down)

```bash
uv run runpod/runpod_ctl.py run 03_llms/10_deepseek_from_scratch \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # must say "Nothing is billing."
```

The pod is **never given `RUNPOD_API_KEY`** — see [SECURITY.md](../../SECURITY.md).

### Direct

```bash
deepspeed --num_gpus=1 train_deepseek_from_scratch.py --variant all
deepspeed --num_gpus=1 train_deepseek_from_scratch.py --variant mla --epochs 20
```

## Configuration notes

- `train_batch_size` is **omitted** from `ds_config.json` so any `--num_gpus`
  works. DeepSpeed derives it as `micro_batch × grad_accum × world_size`.
- **ZeRO stage 0, and that is the lesson.** ZeRO shards optimizer state,
  gradients and parameters — *none of which is the KV cache*. The cache is an
  inference cost ZeRO never touches, which is exactly why MLA exists as a
  separate idea. A reader who has only met ZeRO tends to assume it covers both.
- fp16 and bf16 are both **disabled**. The quantity being compared is a loss
  difference of order 0.1 nats between three implementations; running one in
  reduced precision would measure the dtype.
- `--seq-len` is the memory knob: attention is O(seq²).

## On provenance

This is an **independent implementation from the published paper** — DeepSeek-AI,
*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
Model*, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), §2.1. No
third-party code is vendored here.

That matters: several popular "DeepSeek from scratch" repositories carry **no
licence at all**, which under default copyright means all rights reserved. Code
being public and readable on GitHub does not make it reusable, and this
repository is MIT. Implementing from the paper is the clean route, and it is
also the better one — it is what let the cache accounting be cross-checked
against the GLM-5.3 analysis in `01_llm_finetuning`.

## Where to go next

`../01_llm_finetuning/train_glm53_ds.py --plan` shows the same mechanism inside
a real 755 GB model, where MLA's compression is what makes a 1M-token context
possible at all: 94 GB of cache instead of 5,360 GB.
