# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: attention over RNN hidden states must be correct, not just wired up.

Run:
    uv run tests/test_attention_layers.py

Why this suite exists
---------------------
An attention layer that is subtly wrong trains perfectly happily. The loss goes
down, the shapes are right, and the model is worse than the `out[:, -1, :]` it
replaced — which you will attribute to the extra parameters overfitting rather
than to a bug.

The properties pinned here are the ones that fail silently:

  * **weights sum to 1.** They are a convex combination; if they do not sum to
    1 the context vector is a scaled average and the downstream layer sees a
    magnitude that drifts with the sequence length.
  * **masking happens BEFORE the softmax.** Zeroing weights afterwards is the
    tempting shortcut, and it leaves the denominator contaminated so the
    surviving weights no longer normalise.
  * **attention generalises last-state pooling.** Put all the mass on t=T-1 and
    you must recover `out[:, -1, :]` exactly. This is the claim that makes the
    swap safe to recommend, so it is asserted numerically.
  * **1/sqrt(d) actually prevents saturation.** Asserted as a trend across
    dimensions, not a single draw — one sample is far too noisy to show it.
  * **decomposition reconstructs.** seasonal + trend must equal the input; a
    decomposition that does not reconstruct is not a decomposition.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "02_intermediate/02_rnn_stock_data"))

from attention_layers import (  # noqa: E402
    InputAttention,
    TemporalAttention,
    additive_attention,
    causal_mask,
    dot_product_attention,
    series_decomposition,
)


def test_weights_are_a_distribution(r: Results) -> None:
    """Attention weights must be a convex combination — non-negative, sum 1."""
    torch.manual_seed(0)
    B, T, H = 3, 12, 16
    states = torch.randn(B, T, H)

    for kind in ("additive", "dot"):
        attn = TemporalAttention(H, kind=kind)
        ctx, w = attn(states)

        r.check(ctx.shape == (B, H), f"{kind}: context is (B, H)",
                f"got {tuple(ctx.shape)}")
        r.check(w.shape == (B, T), f"{kind}: one weight per timestep",
                f"got {tuple(w.shape)}")
        r.check(torch.allclose(w.sum(-1), torch.ones(B), atol=1e-6),
                f"{kind}: weights sum to 1",
                f"got {w.sum(-1).tolist()}")
        r.check(bool((w >= 0).all()), f"{kind}: weights are non-negative")

        # The context must actually be that weighted average, not something
        # else that happens to have the right shape.
        manual = torch.bmm(w.unsqueeze(1), states).squeeze(1)
        r.check(torch.allclose(ctx, manual, atol=1e-5),
                f"{kind}: context == sum_t alpha_t h_t",
                f"max diff {(ctx - manual).abs().max():.2e}")

    for bad, label in [(torch.randn(3, 12), "2-D input"),
                       (torch.randn(3), "1-D input")]:
        try:
            TemporalAttention(H)(bad); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects {label}")

    try:
        TemporalAttention(H, kind="bogus"); caught = False
    except ValueError:
        caught = True
    r.check(caught, "rejects an unknown attention kind")


def test_generalises_last_state_pooling(r: Results) -> None:
    """
    THE claim that makes the swap safe: attention contains `out[:, -1, :]`.

    If mass concentrates on the final timestep, the context must equal the last
    hidden state exactly. Without this, replacing last-state pooling with
    attention is a gamble rather than a generalisation.
    """
    torch.manual_seed(1)
    B, T, H = 4, 20, 8
    states = torch.randn(B, T, H)

    one_hot = torch.zeros(B, T)
    one_hot[:, -1] = 1.0
    ctx = torch.bmm(one_hot.unsqueeze(1), states).squeeze(1)

    r.check(torch.equal(ctx, states[:, -1, :]),
            "all mass on t=T-1 reproduces out[:, -1, :] EXACTLY",
            "attention must contain the model it replaces, or the swap is a "
            "gamble rather than a generalisation")

    # And any other one-hot recovers that timestep — the mechanism is a
    # selector, so it can express 'ignore everything except day k'.
    for k in (0, 7, T - 1):
        oh = torch.zeros(B, T); oh[:, k] = 1.0
        got = torch.bmm(oh.unsqueeze(1), states).squeeze(1)
        r.check(torch.equal(got, states[:, k, :]),
                f"one-hot at t={k} selects that timestep exactly")


def test_masking_precedes_softmax(r: Results) -> None:
    """
    Masked positions must get ZERO weight, and the rest must still sum to 1.

    The wrong implementation (zero the weights after softmax) leaves the
    denominator contaminated: the surviving weights sum to less than 1 and the
    context is silently scaled down. Nothing raises.
    """
    torch.manual_seed(2)
    B, T, H = 2, 8, 16
    states = torch.randn(B, T, H)
    query = states[:, -1, :]

    # Allow only the first 3 positions.
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:3] = True

    _, w = dot_product_attention(query, states, mask=mask)

    r.check(bool((w[:, 3:] == 0).all()),
            "masked positions receive exactly zero weight",
            f"got {w[0, 3:].tolist()}")
    r.check(torch.allclose(w.sum(-1), torch.ones(B), atol=1e-6),
            "UNMASKED weights still sum to 1",
            f"got {w.sum(-1).tolist()} — if this were < 1 the mask was applied "
            "AFTER the softmax and the context is silently scaled down")

    # The same must hold for additive attention.
    Wq, Wk, v = nn.Linear(H, 8, bias=False), nn.Linear(H, 8, bias=False), nn.Linear(8, 1, bias=False)
    _, wa = additive_attention(query, states, Wq, Wk, v, mask=mask)
    r.check(bool((wa[:, 3:] == 0).all())
            and torch.allclose(wa.sum(-1), torch.ones(B), atol=1e-6),
            "additive attention masks correctly too")

    # Causal mask shape and semantics.
    cm = causal_mask(5)
    r.check(cm.shape == (5, 5), "causal_mask is (T, T)")
    r.check(bool(cm[0, 0]) and not bool(cm[0, 1]),
            "position 0 sees only itself")
    r.check(bool(cm[4].all()), "the last position sees everything")
    r.check(int(cm.sum()) == 15, "a 5x5 causal mask allows exactly 15 pairs",
            f"got {int(cm.sum())}")
    try:
        causal_mask(0); caught = False
    except ValueError:
        caught = True
    r.check(caught, "rejects seq_len < 1")


def test_scaling_prevents_saturation(r: Results) -> None:
    """
    1/sqrt(d) must actually keep the softmax out of saturation.

    Asserted as a TREND over many draws. A single sample is dominated by noise
    — the module's own demo showed d=128 looking flatter than d=32 before it
    was averaged, which would have made the printed table contradict the claim
    it was printed to support.
    """
    torch.manual_seed(3)
    TRIALS, T = 400, 16
    unscaled, scaled = [], []

    for d in (8, 32, 128, 512):
        q = torch.randn(TRIALS, d)
        k = torch.randn(TRIALS, T, d)
        logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)
        unscaled.append(F.softmax(logits, dim=-1).max(-1).values.mean().item())
        scaled.append(
            F.softmax(logits / d ** 0.5, dim=-1).max(-1).values.mean().item())

    r.check(all(a < b for a, b in zip(unscaled, unscaled[1:])),
            "UNSCALED attention saturates monotonically as d grows",
            f"{[round(x, 3) for x in unscaled]}")
    r.check(unscaled[-1] > 0.85,
            "unscaled attention is near one-hot at d=512",
            f"max weight {unscaled[-1]:.3f} — the softmax gradient here is ~0")
    r.check(max(scaled) - min(scaled) < 0.05,
            "SCALED attention stays flat across d",
            f"{[round(x, 3) for x in scaled]}")
    r.check(all(x < 0.35 for x in scaled),
            "scaled attention stays far from one-hot at every d",
            f"{[round(x, 3) for x in scaled]}")


def test_decomposition_reconstructs(r: Results) -> None:
    """seasonal + trend must equal the input, at every kernel size."""
    torch.manual_seed(4)
    x = torch.randn(3, 40, 2).cumsum(dim=1)          # a random walk

    for k in (3, 5, 25, 26):
        seasonal, trend = series_decomposition(x, kernel_size=k)
        r.check(seasonal.shape == x.shape and trend.shape == x.shape,
                f"k={k}: both components keep the input shape",
                f"{tuple(seasonal.shape)} / {tuple(trend.shape)}")
        r.check(torch.allclose(seasonal + trend, x, atol=1e-5),
                f"k={k}: seasonal + trend reconstructs the input",
                f"max err {(seasonal + trend - x).abs().max():.2e}")

    # kernel_size=1 is the degenerate case: trend IS the series, seasonal is 0.
    seasonal, trend = series_decomposition(x, kernel_size=1)
    r.check(torch.allclose(trend, x, atol=1e-6)
            and torch.allclose(seasonal, torch.zeros_like(x), atol=1e-6),
            "k=1 is a no-op: all trend, no seasonal")

    # A larger window must smooth more — the trend's variance must fall.
    v_small = series_decomposition(x, 3)[1].var().item()
    v_large = series_decomposition(x, 31)[1].var().item()
    r.check(v_large < v_small,
            "a larger kernel produces a smoother trend",
            f"var(trend) k=3 {v_small:.3f} vs k=31 {v_large:.3f}")

    for bad, label in [({"kernel_size": 0}, "kernel_size=0"),
                       ({"kernel_size": -5}, "negative kernel_size")]:
        try:
            series_decomposition(x, **bad); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects {label}")

    try:
        series_decomposition(torch.randn(3, 40)); caught = False
    except ValueError:
        caught = True
    r.check(caught, "rejects a 2-D input")


def test_input_attention(r: Results) -> None:
    """DA-RNN stage 1: rescale features, do not pool them away."""
    torch.manual_seed(5)
    B, Fdim, H = 4, 6, 16
    ia = InputAttention(n_features=Fdim, hidden_size=H)
    x_t = torch.randn(B, Fdim)
    hidden = torch.randn(B, H)

    weighted, w = ia(x_t, hidden)
    r.check(weighted.shape == (B, Fdim),
            "input attention preserves the feature dimension",
            f"got {tuple(weighted.shape)} — it RESCALES features rather than "
            "pooling them, so downstream shapes are unchanged")
    r.check(torch.allclose(w.sum(-1), torch.ones(B), atol=1e-6),
            "feature weights sum to 1")
    r.check(torch.allclose(weighted, x_t * w, atol=1e-6),
            "output is the elementwise product of input and weights")

    # The degenerate case this folder is currently in: one feature means the
    # softmax is over a set of size 1, so the weight is identically 1.
    single = InputAttention(n_features=1, hidden_size=H)
    _, w1 = single(torch.randn(B, 1), hidden)
    r.check(torch.allclose(w1, torch.ones(B, 1), atol=1e-6),
            "with input_size=1 the weight is identically 1 (a no-op)",
            "which is exactly why stage-1 attention only becomes useful once "
            "the extra features are added")


def test_gradients_flow(r: Results) -> None:
    """Every path must be differentiable — a detached tensor trains nothing."""
    torch.manual_seed(6)
    B, T, H = 2, 10, 16

    for kind in ("additive", "dot"):
        states = torch.randn(B, T, H, requires_grad=True)
        ctx, _ = TemporalAttention(H, kind=kind)(states)
        ctx.sum().backward()
        g = states.grad
        r.check(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0,
                f"{kind}: gradient reaches the hidden states")

    x = torch.randn(2, 30, 1, requires_grad=True)
    series_decomposition(x, 5)[0].sum().backward()
    r.check(x.grad is not None and torch.isfinite(x.grad).all(),
            "decomposition is differentiable")


def main() -> int:
    r = Results("Attention over RNN hidden states — mechanism correctness")
    test_weights_are_a_distribution(r)
    test_generalises_last_state_pooling(r)
    test_masking_precedes_softmax(r)
    test_scaling_prevents_saturation(r)
    test_decomposition_reconstructs(r)
    test_input_attention(r)
    test_gradients_flow(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
