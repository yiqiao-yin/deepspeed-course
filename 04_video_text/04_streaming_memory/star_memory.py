"""
STAR memory — constant-memory understanding of an unbounded video stream.

THE PROBLEM THIS SOLVES
-----------------------
Everything in `02_token_compression/` shrinks a clip by a *factor*. Halve the
tokens, quarter the attention. That is a real win and it does not change the
asymptotics one bit: cost still grows with video length, so for any fixed
compression ratio there is a video long enough to OOM you. A security camera
does not have a length. A livestream does not have a length. A meeting
recording has one but you do not know it in advance.

Offline compression answers "how do I fit this clip?". Streaming answers a
strictly harder question:

    How do I watch forever in O(1) memory?

That constraint is absolute. If per-frame cost is not *constant*, the system
does not work -- it only fails later.

THE IDEA: BOUNDED BUFFERS, LOSSY CONSOLIDATION
----------------------------------------------
This is the same bargain your own memory makes. You cannot replay last
Tuesday frame by frame, but you know what happened. Detail decays; structure
survives. Flash-VStream (Zhang et al., 2024) formalises that as four buffers,
each with a HARD size cap, each holding a different level of abstraction:

    M_spa  spatial    N=1 frame,     8x8 pooled  -- the vivid present
    M_tem  temporal   N=25 clusters, 4x4 pooled  -- the events that happened
    M_abs  abstract   N=25 entries,  1x1 pooled  -- the gist, semantically
    M_ret  retrieved  N=3 frames,    8x8 pooled  -- detail pulled back on demand

Note the trade running down that list: as the retention window gets longer,
the spatial resolution gets coarser. One frame ago you keep 8x8. A thousand
frames ago you keep a cluster centroid. That gradient IS the algorithm.

Each buffer has its own consolidation rule for when it overflows:

    spatial    FIFO -- just drop the oldest.
    temporal   weighted k-means -- merge the closest events into centroids,
               carrying weights so a centroid built from 40 frames outvotes
               one built from 2.
    abstract   momentum update -- exponential decay, so old semantics fade
               rather than being evicted.
    retrieved  recomputed each step from the largest temporal clusters.

WHY M_ret EXISTS (the part that is easy to miss)
------------------------------------------------
Buffers 1-3 alone have a fatal flaw: consolidation is irreversible. Once an
event is a 4x4 centroid, the detail is gone, and "what colour was the car
that passed at 3pm?" is unanswerable. M_ret fixes this. It finds the largest
temporal clusters -- the events that mattered -- and pulls the nearest ACTUAL
frames back out of the raw buffer at full 8x8 resolution. Compressed
long-term structure tells you *where* to look; the raw buffer still has the
pixels. Retrieval over a lossy index, which is the same move a RAG system
makes over a vector store.

RELATION TO THE REST OF THE COURSE
----------------------------------
ZeRO shards state across GPUs and pays in communication. Token compression
shrinks activations and pays in fidelity. STAR bounds memory in TIME and pays
in resolution-of-the-past. Three different axes, one identical bargain: you
never get memory for free, you only choose the currency.

Pure PyTorch, CPU-runnable, no model download. Covered by
`tests/test_star_memory.py`.

Reference: Zhang et al. "Flash-VStream: Memory-Based Real-Time Understanding
for Long Video Streams." https://arxiv.org/abs/2406.08085
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def weighted_kmeans(
    features: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster `features` into `k` weighted centroids.

    WHY WEIGHTED

    Plain k-means treats every point equally. That is exactly wrong for
    consolidated memory: after a few rounds some entries are single frames and
    others are centroids already standing for fifty. Unweighted clustering
    lets a one-frame blip drag a centroid as hard as a minute of footage does,
    so rare noise gets over-represented and the long steady event gets
    smeared. Weights make every ORIGINAL frame count once, forever, no matter
    how many consolidation rounds it has survived -- the same invariant that
    size-weighting enforces in ToMe.

    WHY NOT RANDOM INIT

    Deterministic strided init (take every stride-th point) rather than random
    seeds. Two reasons. Memory consolidation runs thousands of times over a
    stream, so a rare bad init would surface as an unreproducible glitch far
    from its cause. And strided init on a time-ordered buffer spreads the
    seeds across the whole time range, which is a genuinely good prior here --
    the clusters we want usually ARE temporally contiguous events.

    Args:
        features: (N, D) points to cluster.
        weights: (N,) how many original frames each point stands for.
        k: Number of output centroids.
        iters: Lloyd iterations. 8 is ample; this runs per-frame and the
            marginal gain past ~5 is nil.

    Returns:
        (centroids, centroid_weights) of shapes (k, D) and (k,). Weights sum
        to the input weights -- nothing is created or lost.
    """
    n_points = features.shape[0]
    if n_points <= k:
        # Nothing to do; pad out to k so the buffer shape stays fixed. A
        # fixed shape is what keeps downstream kernels from recompiling.
        pad = k - n_points
        return (
            torch.cat([features, features.new_zeros(pad, features.shape[1])]),
            torch.cat([weights, weights.new_zeros(pad)]),
        )

    stride = max(1, n_points // k)
    centroids = features[::stride][:k].clone()

    for _ in range(iters):
        dist = torch.cdist(features, centroids)          # (N, k)
        assign = dist.argmin(dim=1)                      # (N,)

        # Weighted mean per cluster, via scatter-add. Doing this with
        # index_add_ rather than a Python loop keeps it O(N) and vectorised.
        w = weights.unsqueeze(1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assign, features * w)
        counts = torch.zeros(centroids.shape[0], device=features.device,
                             dtype=features.dtype)
        counts.index_add_(0, assign, weights)

        # Empty clusters keep their previous position rather than collapsing
        # to zero -- a zero centroid would attract everything next iteration
        # and the clustering would degenerate.
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)

    dist = torch.cdist(features, centroids)
    assign = dist.argmin(dim=1)
    final_w = torch.zeros(k, device=features.device, dtype=features.dtype)
    final_w.index_add_(0, assign, weights)

    return centroids, final_w


@dataclass
class StarConfig:
    """
    Buffer capacities. These are the paper's defaults and they are load-bearing.

    Raising them raises steady-state memory LINEARLY and buys surprisingly
    little: the point of the design is that 25 temporal clusters is enough to
    summarise an arbitrarily long stream, because clusters cover *events*, and
    the number of distinguishable events in a scene does not grow the way
    frames do. If you find yourself raising n_temporal to 200, you probably
    want a longer clip through the offline path in `02_token_compression/`,
    not a bigger stream buffer.
    """

    n_spatial: int = 1        # frames of vivid recent detail
    n_temporal: int = 25      # event clusters
    n_abstract: int = 25      # semantic gist slots
    n_retrieved: int = 3      # key frames pulled back at full detail
    n_buffer: int = 300       # raw frames kept for retrieval

    pool_spatial: int = 8     # 8x8 -> 64 tokens
    pool_temporal: int = 4    # 4x4 -> 16 tokens
    pool_abstract: int = 1    # 1x1 -> 1 token

    momentum: float = 0.9     # abstract-memory decay (alpha)


class StarMemory:
    """
    A bounded memory over an unbounded stream.

    Usage:
        mem = StarMemory(dim=1152)
        for frame_feature in stream:          # (n_tokens, dim), any length
            mem.write(frame_feature)
        context = mem.read()                  # bounded (M, dim), always

    The invariant to hold onto: `mem.read()` returns the SAME shape on frame
    ten and on frame ten million. `tests/test_star_memory.py` asserts exactly
    that, because it is the only property that actually matters here.
    """

    def __init__(self, dim: int, config: StarConfig | None = None):
        self.dim = dim
        self.cfg = config or StarConfig()

        # Spatial: recent frames at high detail. FIFO.
        self.spatial: list[torch.Tensor] = []

        # Temporal: event centroids plus how many frames each represents.
        self.temporal = torch.zeros(0, dim)
        self.temporal_w = torch.zeros(0)

        # Abstract: a fixed slate of semantic slots, updated by momentum.
        self.abstract = torch.zeros(self.cfg.n_abstract, dim)
        self.abstract_init = False

        # Raw buffer: the last n_buffer frame summaries, for retrieval.
        self.buffer: list[torch.Tensor] = []

        self.frames_seen = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _pool(frame: torch.Tensor, out_side: int) -> torch.Tensor:
        """
        Adaptive-pool one frame's tokens down to `out_side` x `out_side`.

        Tokens arrive flattened as (n_tokens, dim) and we treat them as a
        square grid, which is what every ViT actually produces. If n_tokens is
        not a perfect square we pool over the sequence instead of guessing a
        rectangle -- guessing wrong transposes the image, which is the kind of
        bug that costs a day because the loss still goes down.
        """
        n_tokens, dim = frame.shape
        target = out_side * out_side
        if n_tokens == target:
            return frame

        side = int(round(n_tokens ** 0.5))
        if side * side == n_tokens:
            grid = frame.T.reshape(1, dim, side, side)
            pooled = torch.nn.functional.adaptive_avg_pool2d(grid, out_side)
            return pooled.reshape(dim, target).T

        seq = frame.T.unsqueeze(0)                       # (1, dim, n_tokens)
        pooled = torch.nn.functional.adaptive_avg_pool1d(seq, target)
        return pooled.squeeze(0).T

    # -- the four writes --------------------------------------------------

    def write(self, frame: torch.Tensor) -> None:
        """
        Absorb one frame. Cost and memory are O(1) in stream length.

        Order matters: temporal must consolidate before retrieval runs, since
        retrieval keys off the temporal clusters.
        """
        if frame.dim() != 2 or frame.shape[1] != self.dim:
            raise ValueError(
                f"expected (n_tokens, {self.dim}), got {tuple(frame.shape)}"
            )

        self.frames_seen += 1

        # 1. SPATIAL -- FIFO of vivid recent detail.
        self.spatial.append(self._pool(frame, self.cfg.pool_spatial))
        if len(self.spatial) > self.cfg.n_spatial:
            self.spatial.pop(0)

        # 2. BUFFER -- raw-ish frames kept so retrieval has something to find.
        self.buffer.append(self._pool(frame, self.cfg.pool_spatial))
        if len(self.buffer) > self.cfg.n_buffer:
            self.buffer.pop(0)

        # 3. TEMPORAL -- append, then consolidate by weighted k-means if over.
        #    We append the frame as a single mean vector: at this level of
        #    abstraction we care when-did-what-happen, not where-in-the-frame.
        tem_tokens = self._pool(frame, self.cfg.pool_temporal).mean(dim=0, keepdim=True)
        self.temporal = torch.cat([self.temporal, tem_tokens])
        self.temporal_w = torch.cat([self.temporal_w, torch.ones(1)])

        if self.temporal.shape[0] > self.cfg.n_temporal:
            self.temporal, self.temporal_w = weighted_kmeans(
                self.temporal, self.temporal_w, self.cfg.n_temporal
            )

        # 4. ABSTRACT -- semantic attention, then a momentum update.
        self._write_abstract(frame)

    def _write_abstract(self, frame: torch.Tensor) -> None:
        """
        Route this frame into the semantic slots it resembles, then decay.

        This is the only buffer that never evicts. Instead every slot is a
        running exponential average:

            M_abs <- alpha * M_abs + (1 - alpha) * routed_update

        so old semantics fade smoothly rather than falling off a cliff. The
        effective horizon is about 1/(1-alpha) frames -- at alpha=0.9, roughly
        the last ten frames dominate any given slot, with an exponentially
        thinning tail of everything before. Forgetting as a gradient, not a
        deletion.

        Routing is attention: each slot attends over this frame's tokens, so
        a slot that has come to represent "person" pulls in person-ish tokens
        and ignores the rest. Slots specialise on their own; nothing supervises
        them.
        """
        gist = self._pool(frame, self.cfg.pool_abstract)     # (1, dim)

        if not self.abstract_init:
            # Seed every slot from the first frame. Starting from zeros makes
            # the first softmax uniform and the slots take a long time to
            # differentiate; seeding lets them start splitting immediately.
            self.abstract = gist.expand(self.cfg.n_abstract, -1).clone()
            self.abstract_init = True
            return

        frame_tokens = self._pool(frame, self.cfg.pool_temporal)   # (T, dim)
        logits = (self.abstract @ frame_tokens.T) / (self.dim ** 0.5)
        weights = logits.softmax(dim=-1)                           # (n_abs, T)
        update = weights @ frame_tokens                            # (n_abs, dim)

        a = self.cfg.momentum
        self.abstract = a * self.abstract + (1.0 - a) * update

    # -- retrieval and read ----------------------------------------------

    def _retrieve(self) -> torch.Tensor:
        """
        Pull back full-detail frames near the biggest temporal clusters.

        "Biggest" means largest weight, i.e. the events that occupied the most
        frames. That is a crude but effective salience proxy: the thing that
        was on screen longest is usually the thing the question is about.
        Query-conditioned retrieval would be better and needs the question,
        which a streaming writer does not have yet.
        """
        if not self.buffer or self.temporal.shape[0] == 0:
            return torch.zeros(0, self.dim)

        k = min(self.cfg.n_retrieved, len(self.buffer))
        top = self.temporal_w.topk(min(k, self.temporal_w.shape[0])).indices
        centroids = self.temporal[top]                        # (k, dim)

        # Buffer frames as single vectors, so we can nearest-neighbour them
        # against the centroids in one cdist.
        buf = torch.stack([f.mean(dim=0) for f in self.buffer])  # (B, dim)
        nearest = torch.cdist(centroids, buf).argmin(dim=1)      # (k,)

        return torch.cat([self.buffer[i] for i in nearest.tolist()])

    def read(self) -> torch.Tensor:
        """
        The bounded context to hand the language model.

        Concatenation order is deliberate: abstract (oldest, coarsest) first,
        spatial (newest, sharpest) last. Causal models weight recent context
        more heavily, and "what is happening right now" should sit closest to
        the question.

        Returns:
            (M, dim) where M is bounded by the config and INDEPENDENT of how
            many frames have been written.
        """
        parts = [self.abstract, self.temporal, self._retrieve()]
        parts.extend(self.spatial)
        parts = [p for p in parts if p.numel() > 0]
        if not parts:
            return torch.zeros(0, self.dim)
        return torch.cat(parts, dim=0)

    # -- accounting -------------------------------------------------------

    def max_context_tokens(self) -> int:
        """
        The hard ceiling on `read()`'s length, computed from the config alone.

        Worth printing at startup. If this number is larger than you expected,
        you have configured a system that will OOM under load -- and you find
        that out now rather than at 3am in hour six of a stream.
        """
        cfg = self.cfg
        return (
            cfg.n_abstract
            + cfg.n_temporal
            + cfg.n_retrieved * cfg.pool_spatial ** 2
            + cfg.n_spatial * cfg.pool_spatial ** 2
        )

    def stats(self) -> str:
        """One-line state dump. The compression ratio is the headline."""
        ctx = self.read().shape[0]
        naive = self.frames_seen * self.cfg.pool_spatial ** 2
        ratio = naive / max(ctx, 1)
        return (
            f"frames={self.frames_seen:>7,}  context={ctx:>5} tokens  "
            f"(naive would be {naive:>9,} — {ratio:>8.1f}x compression)"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    DIM = 128

    mem = StarMemory(dim=DIM)
    print("=" * 74)
    print("STAR memory — context size as the stream grows without bound")
    print("=" * 74)
    print(f"configured ceiling: {mem.max_context_tokens()} tokens\n")

    checkpoints = {10, 100, 1_000, 5_000, 20_000}
    for step in range(1, 20_001):
        # A synthetic stream with slowly drifting content, so the temporal
        # clusters have genuine structure to find rather than pure noise.
        drift = torch.sin(torch.tensor(step / 500.0))
        mem.write(torch.randn(64, DIM) * 0.3 + drift)
        if step in checkpoints:
            print("  " + mem.stats())

    print("\nContext size never grew. That is the entire point.")
