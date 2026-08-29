"""
Modern time-series forecasting primitives — the ideas, on plain tensors.

WHY THIS FILE EXISTS
--------------------
`attention_layers.py` next door answered "can attention replace `out[:, -1, :]`?"
The measured answer on this task was: yes it can, and it does not help, because
a one-step forecast of a smooth mean-reversion signal is a regime where
persistence is close to optimal.

That is a fact about the *problem setup*, not about deep learning. So this file
changes the setup rather than adding more architectures to a losing race:

  * **longer horizons.** Persistence degrades as H grows -- predicting
    delta-bar 20 days out is a genuinely different task from predicting it
    tomorrow, and it is where a model can earn its parameters.
  * **architectures built for series**, not borrowed from language. The four
    primitives below are what actually moved the field between 2019 and 2023,
    and none of them is an RNN.

WHAT IS IMPLEMENTED
-------------------
    trend_basis / seasonality_basis   N-BEATS interpretable basis expansion
    doubly_residual_stack             N-BEATS backcast/forecast residual flow
    patchify                          PatchTST -- the idea that made
                                      transformers work on series
    dilated_receptive_field           TCN -- how many layers to see H days
    causal_conv1d                     a convolution that cannot look forward
    multi_scale_decompose             TimeMixer-style multi-resolution split

Plain PyTorch, no GPU, no download. Covered by `tests/test_ts_forecasting.py`.

THE FRAMING WORTH KEEPING
-------------------------
Zeng et al. (2022) showed a one-layer linear model beating every transformer
forecaster of its day. PatchTST (2023) answered it. TimeMixer (2024) answered
that. And in 2025 a position paper surveyed the whole exchange and concluded
there are **no champions** -- the models are close and the rankings are
sensitive to hyperparameter search.

So the point of this file is not "here is the winner". It is that these are
four genuinely distinct inductive biases, they are cheap to try, and the only
way to know which suits your series is to run all of them against persistence.

References:
- Oreshkin et al. "N-BEATS." ICLR 2020. https://arxiv.org/abs/1905.10437
- Bai, Kolter & Koltun. "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling." 2018. https://arxiv.org/abs/1803.01271
- Nie et al. "A Time Series is Worth 64 Words." ICLR 2023.
  https://arxiv.org/abs/2211.14730
- Wang et al. "TimeMixer." ICLR 2024. https://arxiv.org/abs/2405.14616
- Zeng et al. "Are Transformers Effective for Time Series Forecasting?"
  AAAI 2023. https://arxiv.org/abs/2205.13504
- "Position: There are no Champions in Long-Term Time Series Forecasting."
  2025. https://arxiv.org/abs/2502.14045
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# N-BEATS: basis expansion
# ---------------------------------------------------------------------------

def trend_basis(horizon: int, degree: int = 3, device=None) -> torch.Tensor:
    r"""
    Polynomial basis for the trend block: rows are 1, t, t^2, ... t^degree.

    N-BEATS's interpretable variant does not let a block output arbitrary
    numbers. It outputs *coefficients* on a fixed basis, so the forecast is

        y_hat = theta^T B,      B[i, t] = (t / H)^i

    and `theta` is directly readable: theta[0] is the level, theta[1] the
    slope, theta[2] the curvature. That is the whole interpretability claim,
    and it is structural rather than a post-hoc explanation.

    THE CONSTRAINT IS THE POINT. A low-degree polynomial cannot represent a
    wiggle, so a trend block is *forced* to model trend and leave the rest to
    the residual. Compare with an unconstrained MLP head, which will happily
    fit the noise and leave nothing behind for the next block.

    Time is normalised to t/H so the basis is scale-free in the horizon -- with
    raw t, the t^3 row at H=100 spans six orders of magnitude and the linear
    layer that consumes it is badly conditioned.

    Returns:
        (degree + 1, horizon).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    t = torch.arange(horizon, dtype=torch.float32, device=device) / horizon
    return torch.stack([t ** i for i in range(degree + 1)])


def seasonality_basis(horizon: int, n_harmonics: int = 4,
                      device=None) -> torch.Tensor:
    r"""
    Fourier basis for the seasonality block: cos and sin at n_harmonics rates.

        B = [cos(2*pi*1*t), sin(2*pi*1*t), ..., cos(2*pi*K*t), sin(2*pi*K*t)]

    Every row is periodic over the horizon by construction, so a seasonality
    block cannot express trend -- the complement of the constraint in
    `trend_basis`. Together they partition the signal, which is what makes the
    decomposition mean something.

    Returns:
        (2 * n_harmonics, horizon).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")

    t = torch.arange(horizon, dtype=torch.float32, device=device) / horizon
    rows = []
    for k in range(1, n_harmonics + 1):
        rows.append(torch.cos(2 * torch.pi * k * t))
        rows.append(torch.sin(2 * torch.pi * k * t))
    return torch.stack(rows)


def doubly_residual_stack(
    x: torch.Tensor, blocks: List[nn.Module]
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    r"""
    N-BEATS's residual flow: each block explains what the previous ones could not.

    Every block emits TWO things -- a backcast (its reconstruction of the
    input) and a forecast (its contribution to the output):

        residual_0 = x
        residual_n = residual_{n-1} - backcast_n
        forecast   = sum_n forecast_n

    Subtracting the backcast is the mechanism. Block 2 never sees the raw
    series; it sees only what block 1 failed to explain. So the blocks
    specialise without being told to, and the sequence of residuals is a
    decomposition you can plot.

    Contrast with a plain deep MLP, where every layer sees a transformation of
    the whole input and nothing forces division of labour.

    Args:
        x: (B, L) the lookback window, flattened.
        blocks: modules returning (backcast (B, L), forecast (B, H)).

    Returns:
        (forecast, final_residual, per_block_forecasts). The residual is worth
        returning: if it is still large, the stack is too shallow.
    """
    if not blocks:
        raise ValueError("doubly_residual_stack needs at least one block")

    residual = x
    forecast = None
    parts = []
    for block in blocks:
        backcast, block_forecast = block(residual)
        residual = residual - backcast
        parts.append(block_forecast)
        forecast = block_forecast if forecast is None else forecast + block_forecast
    return forecast, residual, parts


# ---------------------------------------------------------------------------
# PatchTST: patching
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    Cut a series into overlapping patches — PatchTST's central idea.

    WHY THIS MADE TRANSFORMERS WORK ON TIME SERIES

    Point-wise attention over a series treats each timestep as a token, which
    is wrong in a way that is easy to state: a single timestep carries almost
    no semantic content on its own, whereas a *word* does. Attention between
    two individual timesteps is mostly attention between two noise samples.

    Patching fixes three things at once:

      1. **Local semantics.** A patch of 16 days has a shape -- a rise, a
         reversal -- that a single day does not.
      2. **Quadratic cost, divided.** Attention is O(N^2) in tokens. Patching
         with stride s cuts N by ~s, so cost falls by ~s^2. That is what lets
         PatchTST use a long lookback at all.
      3. **Longer history for the same budget.**

    Args:
        x: (B, L) or (B, L, C).
        patch_len: timesteps per patch.
        stride: step between patch starts. stride == patch_len gives
            non-overlapping patches; stride < patch_len overlaps them, which
            softens the arbitrary boundaries.

    Returns:
        (B, n_patches, patch_len) for 2-D input, or
        (B, C, n_patches, patch_len) for 3-D.

        n_patches = (L - patch_len) // stride + 1
    """
    if patch_len < 1 or stride < 1:
        raise ValueError(
            f"patch_len and stride must be >= 1, got {patch_len}, {stride}")
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(-1)                       # (B, L, 1)
    if x.dim() != 3:
        raise ValueError(f"expected (B, L) or (B, L, C), got {tuple(x.shape)}")

    length = x.shape[1]
    if length < patch_len:
        raise ValueError(
            f"sequence length {length} is shorter than patch_len {patch_len}")

    # (B, C, n_patches, patch_len)
    patches = x.transpose(1, 2).unfold(dimension=-1, size=patch_len,
                                       step=stride)
    return patches.squeeze(1) if squeeze else patches


def n_patches(length: int, patch_len: int, stride: int) -> int:
    """
    How many patches `patchify` will produce. Useful for sizing a Linear head
    without a dry-run forward pass — a shape mismatch there surfaces as a
    runtime error halfway through the first epoch.
    """
    if length < patch_len:
        raise ValueError(
            f"sequence length {length} is shorter than patch_len {patch_len}")
    return (length - patch_len) // stride + 1


# ---------------------------------------------------------------------------
# TCN: dilated causal convolution
# ---------------------------------------------------------------------------

def dilated_receptive_field(kernel_size: int, n_layers: int,
                            dilation_base: int = 2) -> int:
    r"""
    How far back a stack of dilated convolutions can see.

        R = 1 + (k - 1) * (b^L - 1) / (b - 1)     for dilations b^0 .. b^(L-1)

    THE NUMBER YOU MUST CHECK BEFORE TRAINING A TCN. If R is smaller than your
    lookback window, the model *cannot* use the early part of it -- and nothing
    will tell you. Training proceeds, the loss falls, and a chunk of your input
    is structurally invisible.

    With k=3 and doubling dilations, R grows EXPONENTIALLY in depth: 4 layers
    reach 31 steps, 6 layers reach 127. That is the TCN's argument against
    RNNs -- constant path length to any input, so no vanishing gradient over
    the window, and every timestep computed in parallel.

    Bai et al. (2018) concluded that convolutions should be the *default*
    starting point for sequence modelling, not RNNs.
    """
    if kernel_size < 1 or n_layers < 1 or dilation_base < 1:
        raise ValueError("kernel_size, n_layers, dilation_base must be >= 1")
    if dilation_base == 1:
        return 1 + (kernel_size - 1) * n_layers
    return 1 + (kernel_size - 1) * (dilation_base ** n_layers - 1) // (dilation_base - 1)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor,
                  dilation: int = 1) -> torch.Tensor:
    """
    1-D convolution that cannot see the future.

    Ordinary `conv1d` with `padding=k//2` is centred: the output at t depends
    on inputs at t+1..t+k//2. For a forecaster that is look-ahead bias hidden
    inside a layer -- the same class of bug as the scaler leak, and just as
    silent.

    The fix is to pad the LEFT only, by `dilation * (k - 1)`, and drop the
    overhang on the right. `tests/test_ts_forecasting.py` verifies causality
    directly: perturb input t+1 and assert output t does not move.

    Args:
        x: (B, C_in, L)
        weight: (C_out, C_in, K)
        dilation: spacing between taps.

    Returns:
        (B, C_out, L) — same length as the input.
    """
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(
            f"expected x (B, C, L) and weight (Co, Ci, K), got "
            f"{tuple(x.shape)} and {tuple(weight.shape)}"
        )
    k = weight.shape[-1]
    pad = dilation * (k - 1)
    out = F.conv1d(F.pad(x, (pad, 0)), weight, dilation=dilation)
    return out


# ---------------------------------------------------------------------------
# TimeMixer: multi-scale decomposition
# ---------------------------------------------------------------------------

def multi_scale_decompose(x: torch.Tensor, scales: Tuple[int, ...] = (1, 2, 4)
                          ) -> List[torch.Tensor]:
    """
    Downsample a series to several resolutions — TimeMixer's premise.

    A financial series carries structure at more than one timescale at once:
    microstructure noise at the daily level, a swing over weeks, a regime over
    quarters. A single fixed resolution has to pick one, and averages the rest
    away.

    Downsampling by 2 and 4 produces coarser views where slow structure is
    visible and fast noise has been averaged out. A model that mixes across
    them can use both, rather than trading one for the other.

    Args:
        x: (B, L, C).
        scales: downsampling factors. 1 is the original resolution.

    Returns:
        one tensor per scale, with lengths L, L/2, L/4, ...
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, L, C), got {tuple(x.shape)}")
    if any(s < 1 for s in scales):
        raise ValueError(f"scales must all be >= 1, got {scales}")

    out = []
    for s in scales:
        if s == 1:
            out.append(x)
        else:
            pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=s, stride=s)
            out.append(pooled.transpose(1, 2))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    bar = "=" * 78

    print(bar)
    print("  Modern time-series primitives")
    print(bar)

    print("\n  N-BEATS interpretable basis — the CONSTRAINT is the point")
    tb = trend_basis(horizon=20, degree=3)
    sb = seasonality_basis(horizon=20, n_harmonics=4)
    print(f"    trend basis       {tuple(tb.shape)}  rows: 1, t, t^2, t^3")
    print(f"    seasonality basis {tuple(sb.shape)}  rows: cos/sin at 4 rates")
    print(f"    trend row sums    {[round(float(r.sum()), 2) for r in tb]}")
    print(f"    seasonal row sums {[round(float(r.sum()), 4) for r in sb[:4]]}"
          "  <- ~0: periodic, so it cannot express trend")

    print("\n  PatchTST — attention cost falls quadratically with the stride")
    L = 60
    print(f"    {'patch_len':>10} {'stride':>7} {'tokens':>8} "
          f"{'attn cost vs point-wise':>26}")
    print("    " + "-" * 54)
    for pl, st in [(1, 1), (8, 8), (16, 8), (16, 16)]:
        n = n_patches(L, pl, st)
        print(f"    {pl:>10} {st:>7} {n:>8} {(n / L) ** 2:>25.3f}x")

    print("\n  TCN receptive field — exponential in depth")
    print(f"    {'layers':>7} {'k=2':>7} {'k=3':>7} {'k=5':>7}")
    print("    " + "-" * 30)
    for L_ in (2, 4, 6, 8):
        print(f"    {L_:>7} " + " ".join(
            f"{dilated_receptive_field(k, L_):>7}" for k in (2, 3, 5)))
    print("    With a 60-day window you need k=3, 5 layers (R=63) to see all")
    print("    of it. Fewer layers and the early days are structurally invisible.")

    print("\n  Causal convolution — verify, do not assume")
    x = torch.randn(1, 1, 12)
    w = torch.randn(1, 1, 3)
    base = causal_conv1d(x, w, dilation=2)
    x2 = x.clone(); x2[0, 0, 8] += 100.0            # perturb a LATER timestep
    pert = causal_conv1d(x2, w, dilation=2)
    changed = (base - pert).abs().squeeze() > 1e-6
    print(f"    perturbed t=8; outputs that moved: "
          f"{[i for i, c in enumerate(changed) if c]}")
    print("    All >= 8. No earlier output moved, so the layer cannot look ahead.")

    print("\n  TimeMixer multi-scale views")
    series = torch.randn(2, 64, 1)
    for s, v in zip((1, 2, 4), multi_scale_decompose(series, (1, 2, 4))):
        print(f"    scale {s}: {tuple(v.shape)}  std={v.std():.4f}")
    print("    Coarser scales have lower variance — fast noise averaged out,")
    print("    slow structure retained.")
    print(bar)
