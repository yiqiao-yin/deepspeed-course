"""
Attention over RNN hidden states — the mechanisms, on plain tensors.

WHY THIS FILE EXISTS
--------------------
`train_rnn_stock_data_ds.py` ends its forward pass with:

    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])          # <- 59 hidden states discarded

Sixty days go in; the model keeps the last hidden state and throws the other
fifty-nine away. Everything the sequence knew has to survive by being squeezed
through one fixed-width vector, and whatever did not fit is gone.

That is precisely the bottleneck attention was invented to remove. Bahdanau et
al. (2014) introduced it for exactly this complaint in translation -- one
context vector is not enough -- and the fix transfers directly: instead of
taking the last hidden state, take a *learned weighted average* of all of them.

    context = sum_t  alpha_t * h_t,     sum_t alpha_t = 1

The model decides which of the sixty days mattered. On this task that is
interpretable in a way it rarely is elsewhere: you can plot alpha over the
window and see whether the model is looking at last week or at the same period
last quarter.

WHAT IS IMPLEMENTED
-------------------
    additive_attention        Bahdanau (2014) -- a small MLP scores each state
    dot_product_attention     Luong (2015) / scaled dot-product -- cheaper
    TemporalAttention         the pooling layer that replaces out[:, -1, :]
    causal_mask               for when the mask IS needed (see the warning)
    series_decomposition      DLinear's trend/seasonal split
    input_attention           DA-RNN stage 1 -- attention across FEATURES

Plain PyTorch, no GPU, no download. Covered by `tests/test_attention_layers.py`.

THE HONEST CAVEAT, UP FRONT
---------------------------
Zeng et al. (2022) took a set of transformer forecasters and compared them
against a one-layer linear model. The linear model won on nine datasets, often
by a wide margin. Their argument: self-attention is permutation-invariant, so
positional encoding is patching back the very ordering that a time series is
made of.

So `series_decomposition` is here for a reason. This module gives you attention
AND the linear baseline that may well beat it, in the same spirit as the Theil
U statistic already reported by the trainer: a technique you have not compared
against a trivial alternative is a technique you cannot claim anything about.

References:
- Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and
  Translate." https://arxiv.org/abs/1409.0473
- Luong et al. "Effective Approaches to Attention-based NMT."
  https://arxiv.org/abs/1508.04025
- Qin et al. "A Dual-Stage Attention-Based RNN for Time Series Prediction."
  IJCAI 2017. https://arxiv.org/abs/1704.02971
- Lim et al. "Temporal Fusion Transformers." IJF 2021.
  https://arxiv.org/abs/1912.09363
- Zeng et al. "Are Transformers Effective for Time Series Forecasting?"
  AAAI 2023. https://arxiv.org/abs/2205.13504
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Lower-triangular mask: position t may attend to 0..t, never to t+1.

    WHEN YOU NEED THIS, AND WHEN YOU DO NOT

    This matters less here than people assume, and getting the reasoning right
    matters more than applying it reflexively.

    NOT needed for the many-to-one setup this folder uses. The model reads a
    window of 60 *past* days and emits one number for day 61. Every element of
    that window is already historical, so a state at window-position 10 may
    legitimately attend to window-position 50 -- both are in the past relative
    to the thing being predicted. Masking here would throw away real
    information for no reason.

    REQUIRED the moment you do any of:

      - **per-timestep losses.** If you supervise the output at every position
        (predict t+1 from each t), then position 10's prediction must not see
        position 50, or you have leaked the answer.
      - **autoregressive multi-step forecasting**, where generated steps feed
        back in.
      - **encoder-decoder**, where the decoder must not read ahead.

    This is the in-model analogue of the scaler leak in section 5 of the
    write-up: both are the future contaminating the past, and both produce
    excellent metrics and worthless models.

    Returns:
        (seq_len, seq_len) bool tensor, True where attention is ALLOWED.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                 device=device))


def _apply_mask(scores: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Set disallowed positions to -inf BEFORE the softmax.

    Zeroing the weights *after* softmax is the tempting shortcut and it is
    wrong: the masked positions still contributed to the denominator, so the
    surviving weights no longer sum to 1 and the context vector is silently
    scaled down. Nothing raises; the model just trains slightly wrong.
    """
    if mask is None:
        return scores
    return scores.masked_fill(~mask, float("-inf"))


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def dot_product_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Scaled dot-product attention (Luong-style, as used inside transformers).

        score_t = q . k_t / sqrt(d)
        alpha   = softmax(score)
        context = sum_t alpha_t v_t

    THE 1/sqrt(d) IS NOT COSMETIC. For random q, k with unit-variance
    components, q.k has variance d. At d=128 the scores have standard deviation
    ~11, softmax of which is effectively one-hot -- so the gradient through the
    softmax is ~0 and attention stops learning before it starts. Dividing by
    sqrt(d) restores unit variance. This is the single most common bug in a
    hand-rolled attention layer, and the symptom is "attention weights are
    always one-hot and the model ignores everything else".

    Args:
        query: (B, D) the thing doing the looking -- typically the final
            hidden state.
        keys: (B, T, D) the states being looked at.
        values: (B, T, D) what to average. Defaults to `keys`, which is the
            usual choice when attending over RNN outputs.
        mask: (T,) or (B, T) bool, True where allowed.

    Returns:
        (context, weights) of shapes (B, D) and (B, T).
    """
    if values is None:
        values = keys
    if query.dim() != 2 or keys.dim() != 3:
        raise ValueError(
            f"expected query (B, D) and keys (B, T, D), got "
            f"{tuple(query.shape)} and {tuple(keys.shape)}"
        )

    d = query.shape[-1]
    scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (d ** 0.5)
    scores = _apply_mask(scores, mask)
    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
    return context, weights


def additive_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    W_query: nn.Linear,
    W_keys: nn.Linear,
    v: nn.Linear,
    values: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Additive (Bahdanau) attention: a small MLP scores each state.

        score_t = v^T tanh(W_q q + W_k k_t)

    WHY BOTHER, GIVEN DOT-PRODUCT IS CHEAPER

    Dot product requires query and key to live in the same space and to be
    comparable by inner product. Additive attention learns the comparison, so
    it works when they do not -- different dimensionalities, or a query that is
    a decoder state and keys that are encoder states.

    On a small univariate series it also tends to be better behaved: with
    hidden_size=50 and ~2,400 training samples, the learned scorer has more
    capacity to be selective than a raw inner product does, and there is little
    enough data that the extra parameters are not the binding constraint.

    Bahdanau's original motivation is exactly this folder's problem, phrased
    for translation: a single fixed-length context vector is a bottleneck.

    Args:
        query: (B, Dq)
        keys: (B, T, Dk)
        W_query: Linear(Dq -> Da)
        W_keys: Linear(Dk -> Da)
        v: Linear(Da -> 1)
        values: (B, T, Dk), defaults to keys.
        mask: (T,) or (B, T) bool, True where allowed.

    Returns:
        (context, weights) of shapes (B, Dk) and (B, T).
    """
    if values is None:
        values = keys

    # (B, 1, Da) + (B, T, Da) -> broadcast over time
    scored = torch.tanh(W_query(query).unsqueeze(1) + W_keys(keys))
    scores = v(scored).squeeze(-1)                       # (B, T)
    scores = _apply_mask(scores, mask)
    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
    return context, weights


# ---------------------------------------------------------------------------
# The layer that replaces out[:, -1, :]
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Pool a sequence of RNN hidden states into one context vector.

    A drop-in replacement for `out[:, -1, :]`. Same input, same output shape,
    but the model chooses which timesteps to use instead of being forced to
    rely on the last one.

    THE PROPERTY THAT MAKES THIS SAFE TO ADOPT

    Attention *contains* last-state pooling as a special case: put all the mass
    on t = T-1 and you recover `out[:, -1, :]` exactly. So this cannot be
    strictly worse in representational terms -- if last-state really is
    optimal, the model can learn to say so. What you pay is parameters and the
    risk of overfitting them, which on ~2,400 samples is a real risk rather
    than a theoretical one.

    `tests/test_attention_layers.py` asserts that equivalence numerically,
    because "it generalises the old behaviour" is the kind of claim that is
    easy to state and easy to get subtly wrong.
    """

    def __init__(self, hidden_size: int, attention_dim: int = 64,
                 kind: str = "additive"):
        super().__init__()
        if kind not in ("additive", "dot"):
            raise ValueError(f"kind must be 'additive' or 'dot', got {kind!r}")
        self.kind = kind
        self.hidden_size = hidden_size

        if kind == "additive":
            self.W_query = nn.Linear(hidden_size, attention_dim, bias=False)
            self.W_keys = nn.Linear(hidden_size, attention_dim, bias=False)
            self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, H) — the full RNN output, not just the last.
            mask: optional (T,) or (B, T) bool. Usually None here; see
                `causal_mask` for when it is not.

        Returns:
            (context, weights) of shapes (B, H) and (B, T). Keep the weights —
            they are the interpretable output, and plotting them is the main
            reason to prefer this over a transformer on this task.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"expected (B, T, H), got {tuple(hidden_states.shape)}"
            )
        # The last state is the natural query: "given where I ended up, which
        # earlier states should I revisit?"
        query = hidden_states[:, -1, :]

        if self.kind == "dot":
            return dot_product_attention(query, hidden_states, mask=mask)
        return additive_attention(query, hidden_states, self.W_query,
                                  self.W_keys, self.v, mask=mask)


class InputAttention(nn.Module):
    """
    DA-RNN stage 1: attention across FEATURES rather than across time.

    Qin et al. (2017) split attention in two. Stage 2 is `TemporalAttention`
    above -- which timesteps matter. Stage 1 asks a different question: at each
    timestep, which *input series* matter?

    That is the stage this folder cannot yet use, and saying so is more useful
    than pretending otherwise: `input_size=1`, so there is exactly one series
    and the attention is over a set of size one, which softmax turns into the
    constant 1. It becomes meaningful the moment you take the write-up's own
    advice and add volume, realized volatility, and the individual delta_p that
    are already computed and thrown away.

    It is implemented here so that step is a config change rather than a
    rewrite.
    """

    def __init__(self, n_features: int, hidden_size: int,
                 attention_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.W_hidden = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_series = nn.Linear(n_features, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, n_features, bias=False)

    def forward(self, x_t: torch.Tensor, hidden: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (B, F) the input features at one timestep.
            hidden: (B, H) the encoder hidden state carried in.

        Returns:
            (weighted_input, weights), both (B, F). The input is *rescaled*
            rather than pooled — every feature survives, but scaled by its
            relevance, so downstream shapes are unchanged.
        """
        scored = torch.tanh(self.W_hidden(hidden) + self.W_series(x_t))
        weights = F.softmax(self.v(scored), dim=-1)      # (B, F)
        return x_t * weights, weights


# ---------------------------------------------------------------------------
# The baseline that may beat all of it
# ---------------------------------------------------------------------------

def series_decomposition(x: torch.Tensor, kernel_size: int = 25
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a series into trend (moving average) and seasonal (the remainder).

    This is DLinear's entire preprocessing step, and DLinear -- one linear
    layer per component -- beat every transformer forecaster Zeng et al. tested,
    on nine datasets.

    There is a pleasing circularity here worth noticing. The *target* in this
    folder is already a moving-average deviation: `delta = Close - MA`. So the
    "seasonal" component DLinear extracts is the same kind of object the task
    is built from. Applying a second decomposition on top is not obviously
    useful -- and finding that out costs one experiment rather than one
    architecture.

    Args:
        x: (B, T, C) input series.
        kernel_size: moving-average window. Odd values keep the output
            centred; even values need asymmetric padding and shift the trend
            by half a step.

    Returns:
        (seasonal, trend), both (B, T, C), and they sum back to `x` exactly.
        The test asserts that -- a decomposition that does not reconstruct is
        not a decomposition.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, C), got {tuple(x.shape)}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    # Replicate-pad the ends so the trend is defined over the whole series
    # rather than being shorter than the input. Zero-padding would drag the
    # trend toward zero at both ends, which on a price-derived series creates
    # a large artificial edge effect.
    pad_front = (kernel_size - 1) // 2
    pad_back = kernel_size - 1 - pad_front

    padded = torch.cat(
        [x[:, :1].repeat(1, pad_front, 1), x, x[:, -1:].repeat(1, pad_back, 1)],
        dim=1,
    )
    trend = F.avg_pool1d(
        padded.transpose(1, 2), kernel_size=kernel_size, stride=1
    ).transpose(1, 2)

    return x - trend, trend


if __name__ == "__main__":
    torch.manual_seed(0)
    bar = "=" * 76
    B, T, H = 4, 60, 50

    print(bar)
    print("  Attention over RNN hidden states")
    print(bar)
    print(f"  batch {B}, window {T} days, hidden {H}")

    states = torch.randn(B, T, H)

    print()
    print("  Replacing out[:, -1, :] with a weighted average of all 60 states:")
    for kind in ("additive", "dot"):
        attn = TemporalAttention(H, kind=kind)
        ctx, w = attn(states)
        print(f"    {kind:<9} context {tuple(ctx.shape)}   "
              f"weights {tuple(w.shape)}   sum={w.sum(-1).mean():.6f}")

    print()
    print("  Last-state pooling is the special case where all mass sits on t=T-1:")
    one_hot = torch.zeros(B, T); one_hot[:, -1] = 1.0
    ctx_onehot = torch.bmm(one_hot.unsqueeze(1), states).squeeze(1)
    print(f"    max|attention(one-hot) - out[:, -1, :]| = "
          f"{(ctx_onehot - states[:, -1, :]).abs().max():.2e}")
    print("    So attention CONTAINS the current model; it cannot be strictly")
    print("    worse in representational terms, only harder to fit.")

    print()
    print(bar)
    print("  The 1/sqrt(d) scaling is not cosmetic")
    print(bar)
    # Averaged over many draws: a single sample is far too noisy to show a
    # trend, and printing one would be a demo that does not demonstrate.
    TRIALS, T_KEYS = 400, 16
    print(f"  mean over {TRIALS} random draws, {T_KEYS} keys "
          f"(uniform attention would be {1 / T_KEYS:.4f})\n")
    print(f"  {'d':>5}  {'max weight, unscaled':>22}  {'max weight, scaled':>20}")
    print("  " + "-" * 52)
    for d in (8, 32, 128, 512):
        q = torch.randn(TRIALS, d)
        k = torch.randn(TRIALS, T_KEYS, d)
        logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)
        raw = F.softmax(logits, dim=-1).max(-1).values.mean()
        scaled = F.softmax(logits / d ** 0.5, dim=-1).max(-1).values.mean()
        print(f"  {d:>5}  {raw:>22.4f}  {scaled:>20.4f}")
    print()
    print("  Unscaled, the largest weight climbs toward 1.0 as d grows — the")
    print("  softmax saturates, its gradient goes to zero, and the layer stops")
    print("  learning before it has started. Scaled, it stays flat and close to")
    print("  uniform, which is where a freshly initialised layer should begin.")

    print()
    print(bar)
    print("  Causal masking: needed less often than people assume")
    print(bar)
    m = causal_mask(5)
    print("  allowed[t][s] (True = position t may look at s):")
    for t in range(5):
        print("    t=%d  %s" % (t, "".join("1" if v else "." for v in m[t])))
    print()
    print("  NOT needed for this folder's many-to-one setup: all 60 days are")
    print("  already in the past relative to day 61. REQUIRED the moment you")
    print("  add per-timestep losses or autoregressive multi-step decoding.")

    print()
    print(bar)
    print("  DLinear decomposition — the baseline that may beat all of it")
    print(bar)
    series = torch.randn(2, 40, 1).cumsum(dim=1)      # a random walk
    seasonal, trend = series_decomposition(series, kernel_size=25)
    print(f"  input {tuple(series.shape)} -> seasonal + trend")
    print(f"  max reconstruction error: "
          f"{(seasonal + trend - series).abs().max():.2e}")
    print()
    print("  Zeng et al. (2022) found one linear layer on these two components")
    print("  beat every transformer forecaster they tested, on nine datasets.")
    print("  Report Theil U against it before claiming attention helped.")
    print(bar)
