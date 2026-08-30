# 08.2 — Token compression: ZeRO for activations

> **Token compression is to video what ZeRO is to parameters.**
>
> ZeRO shards what the model *is* and pays in communication.
> Compression shrinks what the model *looks at* and pays in fidelity.
> Different terms of the same memory equation — and they compose.

## The problem

A 448×448 frame is ~256 visual tokens after Qwen2.5-VL's 2×2 patch merger. A
64-frame clip is 16,384 visual tokens. Attention is O(N²), so:

```
  8 frames ->   2,048 tokens   attention cost     1.0x
 16 frames ->   4,096 tokens   attention cost     4.0x
 32 frames ->   8,192 tokens   attention cost    16.0x
 64 frames ->  16,384 tokens   attention cost    64.0x
128 frames ->  32,768 tokens   attention cost   256.0x
```

(Run `uv run token_compression.py` to reproduce that table.)

This is why "just sample more frames" stops working almost immediately, and
why every frontier video paper of the last two years is, underneath the
branding, a **memory** paper.

## Three families, three notions of "important"

[`token_compression.py`](token_compression.py) implements each one as its
actual algorithm, not its diagram.

### 1. ToMe — spatial merging, query-agnostic

*Bolya et al., ICLR 2023.* Merges the `r` most similar token pairs per layer.

Split tokens into two sets by **alternating index** — neighbouring image
patches, the ones most likely to be redundant, land on opposite sides and can
therefore be matched. One matmul gives all pairwise similarities; each token
in A proposes an edge to its best partner in B; the top `r` edges merge.

Two details that are easy to get wrong and expensive to debug:

- **Similarity is measured on the attention *keys*, not the features.** Keys
  already encode "what information does this token offer" — that is their job —
  and they are far better behaved than raw features, whose magnitude swamps
  their direction.
- **Merging must be size-weighted.** A plain mean is wrong after the first
  merge: a token standing for 8 patches would get the same vote as one standing
  for 1, and the drift compounds every layer. Size-weighting is what makes
  ToMe *stackable*.

And the one almost everybody skips:

> **Proportional attention.** Merging two identical keys into one halves that
> region's share of the softmax denominator — so large uniform regions fade out
> *precisely because they were compressible*. Adding `log(s)` to the attention
> logits reproduces the softmax you would have gotten from `s` copies of the
> key. It costs one add.
>
> `tests/test_token_compression.py` asserts this identity holds to 1e-6, and
> also asserts the uncorrected version is measurably wrong — otherwise the test
> would be proving nothing.

### 2. FastV — attention-guided pruning, query-aware

*Chen et al., ECCV 2024.* Run K layers normally (K=2), then drop the visual
tokens the text is not looking at.

The observation: after ~layer 2, visual tokens receive dramatically less
attention per token than text tokens, with a very long tail. The model has
already extracted what it needs from most patches; carrying them through the
remaining 30 layers is waste.

**ToMe merges what is self-similar. FastV keeps what the text is looking at.**
Those are different notions of importance and they compose — ToMe in the vision
tower to cut redundancy, FastV in the LLM to cut irrelevance.

The cost nobody mentions: ranking tokens needs an explicit attention matrix,
and FlashAttention never materialises one. In practice you run layer K with
eager attention purely to get the scores. That is a real slowdown at one layer,
traded against a shorter sequence for every layer after it.

### 3. DyCoke-style — temporal merging

The redundancy the other two structurally cannot see: patch (i,j) at time *t*
is usually near-identical to patch (i,j) at *t+1*. Nothing moved. Flatten the
clip into one sequence and that structure is gone.

Inside a sliding window, keep the first frame whole as an **anchor** and drop
only positions that changed relative to it. Anchoring rather than chaining
frame-to-frame matters: with frame-to-frame comparison, slow drift (a gradual
pan, a fade) is below threshold at every single step, so every frame gets
dropped and the accumulated change goes unrecorded.

The trade-off: this is *content-adaptive*, so a static lecture compresses
enormously and a sports clip barely at all. Great for quality, annoying for
engineering — the output length now varies per sample, and batching it means
padding, which gives back exactly what you just saved.

## Measure, don't estimate

[`train_compressed.py`](train_compressed.py) runs a real DeepSpeed step with
compression on and off and reports `torch.cuda.max_memory_allocated()`.

This is not ceremony. **Three ways the predicted win fails to materialise, all
common, all looking identical from the outside:**

| Symptom | Real cause | Where the fix lives |
|---|---|---|
| Cut tokens 2×, memory barely moved | Optimizer states dominated, not activations | ZeRO — earlier in this course |
| Cut tokens 2×, step time barely moved | MLP (linear in N) dominated attention (quadratic) | Nowhere — you were not far enough along the curve |
| Cut tokens 4×, loss degraded | Ratio tuned on more static video than yours | [`../04_video_eval/`](../04_video_eval/) |

All three read as "compression didn't help." Only measurement separates them,
and the fix differs in every case. `TokenBudget` reports the attention and MLP
terms **separately** so you can see which regime you are in before optimising
the wrong thing.

## Runs on CPU — and should

The algorithms are pure PyTorch on plain tensors. No model, no GPU, no
download. Verify them on a laptop *before* renting an 80 GB card:

```bash
uv run token_compression.py              # the token-budget arithmetic
uv run tests/test_token_compression.py   # 30 checks
```

Those 30 checks assert mathematical properties, not shapes — because
compression code fails in a uniquely nasty way: **it always "works."** Drop the
wrong tokens and the model still runs, the loss still decreases, and the only
symptom is a benchmark score a few points below the paper's, which you will
blame on the learning rate. Nothing raises.

## Running the measurement on a GPU

### Setup (uv — never bare pip)

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 08_vtt/02_token_compression
uv sync                    # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=1 train_compressed.py
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
uv pip install deepspeed transformers accelerate peft opencv-python-headless
```

PyPI's `torch` ships CUDA wheels now, so no `--index-url` is
needed; pinning cu121 gives an older CUDA than the default wheel.
</details>


### CoreWeave / SLURM

```bash
sbatch run_deepspeed.sh          # sweeps 8, 16, 32 frames
FRAMES=64 sbatch run_deepspeed.sh
```

One GPU on purpose: sharding across devices would mix ZeRO's saving into a
number meant to isolate the effect of sequence length. Measure the variable you
are changing.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 08_vtt/02_token_compression \
    --dry-run --collect --wait --terminate --yes
# --dry-run caps the training step so a smoke test stays cheap;
# --terminate deletes the pod in a finally block, so a crash or
# Ctrl-C still stops the billing.
uv run runpod/runpod_ctl.py pods     # confirm: "Nothing is billing."
```

Cheapest subsection in the topic — one 24 GB card, roughly $0.22/hr.

## Next

[`../03_streaming_memory/`](../03_streaming_memory/) — everything here shrinks
cost by a constant **factor**. Halve the tokens and a two-hour video is still
twice a one-hour video. For any fixed ratio there is a video long enough to OOM
you. When the video has no length at all, you need a constant **bound**.

## References

- Bolya et al. *Token Merging: Your ViT But Faster.* ICLR 2023.
  [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- Chen et al. *An Image is Worth 1/2 Tokens After Layer 2.* ECCV 2024.
  [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- Shao et al. *A Survey of Multimodal Long-Context Token Compression.*
  TMLR 2026. [arXiv:2507.20198](https://arxiv.org/abs/2507.20198)
