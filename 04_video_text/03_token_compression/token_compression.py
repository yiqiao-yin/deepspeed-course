"""
Visual token compression for video-language models.

WHY THIS FILE EXISTS
--------------------
The whole course is about one question: *the thing does not fit in memory,
so what do we throw away and when?* ZeRO answers it for **parameters** —
shard the optimizer states, the gradients, the weights, and pay in
communication. This file answers the same question for **activations** in a
video model, where the dominant cost is not the weights at all.

Do the arithmetic. Qwen2.5-VL encodes a 448x448 frame into roughly 256 visual
tokens after its 2x2 patch merger. A 64-frame clip is therefore

    64 frames x 256 tokens = 16,384 visual tokens

before a single word of the prompt. Self-attention is O(N^2) in sequence
length, so going from a 16-frame clip to a 64-frame clip does not cost 4x --
it costs 16x. This is why "just sample more frames" stops working almost
immediately, and it is why every frontier video paper of the last two years
is, underneath the branding, a **memory** paper.

    ZeRO shards what the model *is*.
    Token compression shrinks what the model *looks at*.

Both are lossy in a controlled way, both trade compute for memory, and both
have a knob you are expected to tune rather than a setting you turn on.

WHAT IS IMPLEMENTED HERE
------------------------
Three families, each reduced to its actual algorithm rather than its diagram:

  1. `bipartite_soft_matching` / `merge_wavg` -- ToMe (Bolya et al., ICLR
     2023). Training-free SPATIAL merging. Merges the r most similar token
     pairs per layer. This is the one you should reach for first.

  2. `fastv_select` -- FastV (Chen et al., ECCV 2024). Attention-guided
     SPATIAL pruning. Runs K layers normally, then drops the visual tokens
     the text is not looking at.

  3. `dycoke_temporal_merge` -- DyCoke-style TEMPORAL merging. Adjacent video
     frames are highly redundant *at the same spatial position*; a static
     background patch is near-identical for seconds at a time. Spatial
     methods cannot see this because they treat the clip as one long bag of
     tokens.

Plus `TokenBudget`, which measures what you actually saved instead of
trusting the ratio you asked for.

Everything is plain PyTorch on plain tensors. No model is downloaded, nothing
needs a GPU, and every function here is covered by
`tests/test_token_compression.py`. That is deliberate: these are the pieces
you can verify on a laptop, so they are the pieces you *should* verify on a
laptop before renting an 80 GB card.

REFERENCES
----------
- Bolya et al. "Token Merging: Your ViT But Faster." ICLR 2023.
  https://arxiv.org/abs/2210.09461
- Chen et al. "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play
  Inference Acceleration for Large Vision-Language Models." ECCV 2024.
  https://arxiv.org/abs/2403.06764
- Shao et al. "When Tokens Talk Too Much: A Survey of Multimodal Long-Context
  Token Compression." TMLR 2026. https://arxiv.org/abs/2507.20198
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch


# ---------------------------------------------------------------------------
# 1. ToMe -- bipartite soft matching (spatial, training-free)
# ---------------------------------------------------------------------------

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Build a ToMe merge/unmerge pair that removes `r` tokens.

    THE ALGORITHM, AND WHY IT IS SHAPED THIS WAY

    The obvious way to merge similar tokens is to cluster them. Clustering N
    tokens is expensive and, worse, iterative -- you cannot afford it inside
    every transformer block. ToMe's insight is that you do not need a good
    clustering, you need a *cheap* one that is slightly better than random,
    applied many times. So:

      1. Split the tokens into two sets by alternating index. Set A gets the
         even positions, set B the odd ones. Alternating beats splitting
         down the middle because neighbouring image patches -- which are the
         ones most likely to be redundant -- land on opposite sides and can
         therefore be matched to each other.

      2. Compute the cosine similarity of every token in A against every
         token in B. This is one matmul.

      3. Each token in A proposes an edge to its single most similar partner
         in B. Now take the `r` edges with the highest similarity globally
         and merge only those. Everything else is left alone.

    Step 3 is the important one. A fixed *number* of merges per layer, not a
    fixed threshold, is what makes the sequence length deterministic -- which
    is what lets the whole batch stay rectangular and the kernels stay fast.
    A threshold would give you a different N per sample and you would be back
    to padding.

    WHY `metric` IS THE ATTENTION KEYS

    You would think to measure similarity on the token features `x`. Do not.
    The keys K already encode "what information does this token offer to
    others" -- that is their job in attention -- and they are far better
    behaved than the raw features, which carry magnitude information that
    swamps the direction. Bolya et al. ablate this; keys win. Pass
    `key.mean(dim=1)` when you have multi-head keys of shape (B, H, N, C).

    Args:
        metric: (B, N, C) similarity space. Use attention keys, not features.
        r: How many tokens to remove. Clamped to at most half the sequence,
           since set A only has N//2 tokens to give away.
        class_token: If True, token 0 is protected from being merged away.
            A video model's prompt/BOS token must survive; a bare ViT's CLS
            token must survive.

    Returns:
        (merge, unmerge). `merge(x)` maps (B, N, C) -> (B, N - r, C).
        `unmerge(x)` scatters back to (B, N, C) -- needed only if a
        downstream head requires per-patch outputs (segmentation, dense
        prediction). Pure video QA never calls it.
    """
    protected = 1 if class_token else 0
    n_tokens = metric.shape[1]

    # Set A has ceil(N/2) tokens; we cannot remove more than it holds, minus
    # anything we are protecting. Silently clamping (rather than raising) is
    # what the reference implementation does, and it makes `r` safe to sweep.
    r = min(r, (n_tokens - protected) // 2)
    if r <= 0:
        identity = lambda x, mode="mean", size=None: x  # noqa: E731
        return identity, identity

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)  # (B, Na, Nb) cosine similarity

        if class_token:
            # The class token lives at index 0, hence in set A. Make every
            # edge out of it maximally unattractive so it is never among the
            # top-r and never gets merged into something else.
            scores[..., 0, :] = -torch.inf

        # Each A-token's best partner in B, and how good that match is.
        node_max, node_idx = scores.max(dim=-1)

        # Rank A-tokens by match quality; the top r get merged.
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        src_idx = edge_idx[..., :r, :]     # A-tokens that disappear
        unm_idx = edge_idx[..., r:, :]     # A-tokens that survive
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Fold the r matched A-tokens into their B-partners."""
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        b_sz, _, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(b_sz, -1, c))
        src = src.gather(dim=-2, index=src_idx.expand(b_sz, r, c))
        dst = dst.scatter_reduce(
            -2, dst_idx.expand(b_sz, r, c), src, reduce=mode
        )
        # Survivors first, then destinations. Order is arbitrary but must be
        # consistent between merge and unmerge.
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Scatter merged tokens back to their original N positions."""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b_sz, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(b_sz, r, c))

        out = torch.zeros(
            b_sz, n_tokens, c, device=x.device, dtype=x.dtype
        )
        out[..., 1::2, :] = dst
        out.scatter_(
            dim=-2,
            index=(2 * unm_idx).expand(b_sz, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2, index=(2 * src_idx).expand(b_sz, r, c), src=src
        )
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply `merge` as a SIZE-WEIGHTED average, and track how big each token got.

    WHY WEIGHTING MATTERS

    A plain mean is wrong after the first merge. Suppose token P already
    represents 8 original patches and token Q represents 1. Averaging them
    unweighted gives the lone patch Q the same say as all 8 of P's patches
    combined -- so a single outlier patch can hijack a token that stood for a
    large uniform region. Weighting by size keeps every *original* patch
    contributing equally no matter how many merge rounds it has been through,
    which is what makes ToMe stackable across layers rather than degrading.

    So we merge `x * size` (a sum of contributions), merge `size` (a count),
    and divide.

    Returns:
        (x, size) with x of shape (B, N-r, C) and size of shape (B, N-r, 1).
        Feed `size` to `proportional_attention_bias` on the next attention.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    return x / size, size


def proportional_attention_bias(size: torch.Tensor) -> torch.Tensor:
    """
    The log-size bias that keeps attention honest after merging.

    THE PROBLEM. Softmax attention over merged tokens quietly changes the
    answer. If tokens P and Q had identical keys and you merge them into one,
    that key now appears once in the softmax denominator instead of twice --
    so the region it represents loses half its influence, purely as an
    artefact of compression. Merge aggressively and large uniform regions
    (sky, walls, a static background) fade out exactly because they were
    compressible.

    THE FIX. Add log(s) to the attention logits, where s is how many original
    tokens this one stands for:

        attn = softmax(q k^T / sqrt(d) + log s)

    Since exp(logit + log s) = s * exp(logit), this reproduces the softmax you
    would have gotten from s identical copies of the key. Attention behaves as
    if nothing was merged. It costs one add.

    Args:
        size: (B, N, 1) token sizes from `merge_wavg`.

    Returns:
        (B, 1, 1, N) bias, broadcastable over batch and heads, ready to add
        to raw attention logits before the softmax.
    """
    return size.log()[:, None, None, :, 0]


# ---------------------------------------------------------------------------
# 2. FastV -- attention-guided pruning (spatial, training-free)
# ---------------------------------------------------------------------------

def fastv_select(
    attn_weights: torch.Tensor,
    visual_start: int,
    visual_end: int,
    keep_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Choose which visual tokens survive, from the attention the text pays them.

    THE OBSERVATION THAT MOTIVATES IT

    Chen et al. measured where an LVLM's attention actually goes and found
    something embarrassing: after roughly layer 2, visual tokens receive
    dramatically less attention per token than text tokens, and the
    distribution is extremely long-tailed. The model has already extracted
    what it needed from most patches. Carrying all of them through the
    remaining 30 layers is pure waste -- and the paper's headline result is
    that you can drop half of them after layer 2 with essentially no loss on
    image *or* video benchmarks.

    HOW THIS DIFFERS FROM ToMe, AND WHY YOU MIGHT WANT BOTH

    ToMe is query-agnostic: it merges what is *self-similar*, without knowing
    what was asked. FastV is query-aware: it keeps what the *text is looking
    at*. Those are different notions of "important" and they compose --
    ToMe inside the vision encoder to cut redundancy, FastV in the LLM to cut
    irrelevance. The survey (arXiv 2507.20198) calls these the
    transformation-based and elimination-based families.

    THE COST NOBODY MENTIONS

    FastV needs an explicit attention matrix to rank tokens, and
    FlashAttention never materialises one. In practice you run layer K with
    eager attention purely to get the scores, then switch back. That is a real
    slowdown at one layer, traded against a shorter sequence for all the
    layers after it. Worth it when the sequence is long -- which for video it
    always is.

    WHY THE LAST QUERY ROW

    We rank by the attention that the *final* position pays each visual token.
    In a decoder the last position is the one about to generate, so its query
    is the closest available proxy for "what does the model need right now".
    Averaging over all text rows is the plausible alternative; it dilutes the
    signal with positions that have already been answered.

    Args:
        attn_weights: (B, H, Q, K) post-softmax attention from one layer.
        visual_start: First index of the visual span (inclusive).
        visual_end: End of the visual span (exclusive).
        keep_ratio: Fraction of visual tokens to retain. 0.5 is the paper's
            setting.

    Returns:
        (B, n_keep) LongTensor of absolute token indices to keep, sorted
        ascending. Sorting matters: positional embeddings and causal masks
        both assume monotone order, and an unsorted gather silently scrambles
        the sequence.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    n_visual = visual_end - visual_start
    if n_visual <= 0:
        raise ValueError(
            f"empty visual span: [{visual_start}, {visual_end})"
        )

    n_keep = max(1, int(round(n_visual * keep_ratio)))
    if n_keep >= n_visual:
        return (
            torch.arange(visual_start, visual_end, device=attn_weights.device)
            .unsqueeze(0)
            .expand(attn_weights.shape[0], -1)
        )

    # Attention paid BY the last query TO each visual token, averaged over
    # heads. Different heads specialise, and we want a token that any head
    # cares about to survive -- so mean, not max of a single head.
    scores = attn_weights[:, :, -1, visual_start:visual_end].mean(dim=1)

    top = scores.topk(n_keep, dim=-1).indices
    return (top + visual_start).sort(dim=-1).values


# ---------------------------------------------------------------------------
# 3. Temporal merging (DyCoke-style)
# ---------------------------------------------------------------------------

def dycoke_temporal_merge(
    frames: torch.Tensor,
    window: int = 4,
    similarity_threshold: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Drop tokens that a nearby frame already told us, position by position.

    THE REDUNDANCY SPATIAL METHODS CANNOT SEE

    Flatten a clip into one sequence and ToMe or FastV will happily compress
    it -- but they are looking for tokens that resemble each other *anywhere*.
    Video's biggest redundancy is more specific than that: patch (i, j) at
    time t is usually near-identical to patch (i, j) at time t+1. Nothing
    moved. At 8 fps most of the frame is unchanged between neighbours, and
    that structure is invisible to a method that has already discarded the
    time axis.

    So: inside a sliding window of `window` frames, keep the first frame whole
    as the anchor, and for each later frame keep only the positions that
    actually changed relative to that anchor. Static background collapses to
    one copy. Motion survives everywhere it occurs.

    WHY AN ANCHOR RATHER THAN CHAINING FRAME-TO-FRAME

    Comparing each frame to its immediate predecessor sounds better and is
    worse. Slow drift -- a gradual pan, a fade -- is below threshold at every
    single step, so every frame gets dropped, and the accumulated change goes
    unrecorded. Anchoring to a fixed reference bounds the error you can
    accumulate inside a window by construction.

    WHAT THE THRESHOLD COSTS YOU

    Unlike ToMe's fixed `r`, this is content-adaptive: a static lecture
    recording compresses enormously, a fast sports clip barely at all. That is
    the right behaviour for quality and an annoyance for engineering, because
    the output length now varies per sample. Batch it and you are padding
    again. The returned mask is what you use to keep the accounting honest.

    Args:
        frames: (B, T, N, C) -- batch, frames, tokens-per-frame, channels.
        window: Frames per window. The anchor is re-taken every `window`
            frames, so this bounds how stale a reference can get.
        similarity_threshold: Cosine similarity above which a token is
            considered already-explained by the anchor. Higher keeps more.

    Returns:
        (kept, mask). `mask` is (B, T, N) bool, True where a token survives.
        `kept` is (B, T, N, C) with dropped positions zeroed -- dense, so it
        stays batchable; use `mask` to gather the ragged version or to
        build the attention mask.
    """
    if frames.dim() != 4:
        raise ValueError(
            f"expected (B, T, N, C), got shape {tuple(frames.shape)}"
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    b_sz, t_len, n_tok, _ = frames.shape
    normed = frames / (frames.norm(dim=-1, keepdim=True) + 1e-6)

    mask = torch.ones(b_sz, t_len, n_tok, dtype=torch.bool, device=frames.device)

    for start in range(0, t_len, window):
        stop = min(start + window, t_len)
        anchor = normed[:, start : start + 1]           # (B, 1, N, C)
        rest = normed[:, start + 1 : stop]              # (B, W-1, N, C)
        if rest.shape[1] == 0:
            continue
        # Position-wise cosine similarity against the anchor. The sum over C
        # is a dot product of unit vectors, i.e. cosine, done without
        # materialising an (N, N) matrix -- we only ever compare like
        # position to like position.
        sim = (rest * anchor).sum(dim=-1)               # (B, W-1, N)
        mask[:, start + 1 : stop] = sim < similarity_threshold

    return frames * mask.unsqueeze(-1), mask


# ---------------------------------------------------------------------------
# 4. Measurement
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """
    Account for what compression actually bought you.

    Written because the reported number and the real number diverge more
    often than anyone admits. Three ways that happens:

      - You ask for keep_ratio=0.5 and get 0.51, because `int(round(...))`
        and a protected class token do not divide evenly.
      - You cut tokens by 4x and wall-clock barely moves, because at your
        sequence length the model was bound by weight loading, not attention.
      - You cut tokens by 2x and memory barely moves, because activations
        were never the dominant term -- the optimizer states were, and that
        is a ZeRO problem, not a token problem.

    The last one is the one that wastes a week. Attention is quadratic, the
    MLP is linear, and which dominates depends on where you are on the curve.
    `attention_flops_ratio` and `mlp_flops_ratio` are reported separately so
    you can see which regime you are in before you optimise the wrong thing.
    """

    original_tokens: int
    compressed_tokens: int
    hidden_size: int = 3584
    n_layers: int = 28
    notes: list[str] = field(default_factory=list)

    @property
    def keep_ratio(self) -> float:
        """Fraction of tokens surviving. The number you *actually* got."""
        if self.original_tokens == 0:
            return 0.0
        return self.compressed_tokens / self.original_tokens

    @property
    def attention_flops_ratio(self) -> float:
        """
        Remaining attention cost. Quadratic, so this is the big win.

        Halving tokens quarters this term.
        """
        return self.keep_ratio ** 2

    @property
    def mlp_flops_ratio(self) -> float:
        """
        Remaining MLP cost. Linear -- halving tokens only halves it.

        For short sequences the MLP dominates and compression underdelivers
        against the quadratic intuition. This is the term that explains a
        disappointing benchmark.
        """
        return self.keep_ratio

    @property
    def kv_cache_bytes_saved(self) -> int:
        """
        Bytes of KV cache freed, at bf16 (2 bytes), K and V, every layer.

        For streaming inference this is usually the number that matters more
        than FLOPs: the cache is what grows without bound and what eventually
        OOMs a long-video session. See `03_streaming_memory/`.
        """
        removed = self.original_tokens - self.compressed_tokens
        return removed * self.hidden_size * 2 * 2 * self.n_layers

    def summary(self) -> str:
        """Human-readable report. Print this, do not trust the ratio you set."""
        lines = [
            f"tokens        {self.original_tokens:>8,} -> "
            f"{self.compressed_tokens:>8,}  ({self.keep_ratio:.1%} kept)",
            f"attention     {self.attention_flops_ratio:.1%} of original "
            f"(quadratic term)",
            f"mlp           {self.mlp_flops_ratio:.1%} of original "
            f"(linear term)",
            f"kv cache      {self.kv_cache_bytes_saved / 1e9:.2f} GB freed "
            f"@ bf16, {self.n_layers} layers",
        ]
        lines.extend(f"note          {n}" for n in self.notes)
        return "\n".join(lines)


def count_visual_tokens(
    num_frames: int,
    height: int = 448,
    width: int = 448,
    patch_size: int = 14,
    merge_size: int = 2,
) -> int:
    """
    Visual tokens a Qwen2.5-VL-style encoder produces for a clip.

    The patch merger is the part people forget. The ViT sees (H/p)x(W/p)
    patches, then a 2x2 merge divides that by 4 before anything reaches the
    LLM. Leave it out and you overestimate by 4x, conclude the clip cannot
    possibly fit, and reach for compression you did not need.

    At the defaults: 448/14 = 32, so 32x32 = 1024 patches, merged 2x2 -> 256
    tokens per frame.
    """
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size and merge_size must be positive")

    grid_h = height // patch_size
    grid_w = width // patch_size
    per_frame = (grid_h // merge_size) * (grid_w // merge_size)
    return num_frames * per_frame


if __name__ == "__main__":
    # A worked example of the arithmetic that motivates the whole folder.
    print("=" * 70)
    print("Visual token budget -- Qwen2.5-VL geometry, 448x448 frames")
    print("=" * 70)

    for n_frames in (8, 16, 32, 64, 128):
        n_tok = count_visual_tokens(n_frames)
        # Attention cost relative to the 8-frame case, quadratic in N.
        rel = (n_tok / count_visual_tokens(8)) ** 2
        print(f"  {n_frames:>3} frames -> {n_tok:>7,} tokens"
              f"   attention cost {rel:>7.1f}x the 8-frame clip")

    print()
    print("=" * 70)
    print("What compression buys on a 64-frame clip")
    print("=" * 70)

    base = count_visual_tokens(64)
    for name, ratio in [("ToMe r=25%/layer", 0.75),
                        ("FastV keep 50%", 0.50),
                        ("DyCoke (static scene)", 0.30),
                        ("stacked", 0.15)]:
        budget = TokenBudget(base, int(base * ratio))
        print(f"\n{name}")
        print("  " + budget.summary().replace("\n", "\n  "))
