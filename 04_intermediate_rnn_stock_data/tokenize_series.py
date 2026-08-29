"""
Treating the mean-reversion signal as a LANGUAGE — quantization for token models.

THE IDEA
--------
A language model predicts the next word, and a word is just an index into a
finite dictionary. The mean-reversion signal delta-bar = P - MA is continuous,
but it is also **bounded in practice**: over any finite sample it lives in some
range, and prices do not deviate from their own moving average without limit.

So bin it. Slice the observed range into B levels, replace each value with its
bin index, and you have turned a real-valued series into a sequence of tokens
over a vocabulary of size B. Now every tool built for language applies:
attention, cross-entropy, sampling, pretraining.

This is not an analogy someone stretched. It is what several real systems do:

    WaveNet (2016)   quantized raw audio to 256 mu-law levels and modelled it
                     with a categorical softmax
    Chronos (2024)   scales and quantizes time series into a fixed vocabulary,
                     then trains a T5 language model on the tokens with
                     cross-entropy, and samples to get probabilistic forecasts

WHAT YOU GAIN
-------------
1. **A full predictive distribution, free.** A softmax over B bins IS a
   distribution. Sample from it for intervals, take the argmax for a point
   forecast, read its entropy for a confidence estimate. The write-up's own
   §9 recommends "predict a distribution, not a point" — this gets it as a
   by-product rather than as extra machinery.

2. **Heavy tails stop being a problem.** MSE is quadratic, so a handful of
   crash days dominate the gradient. Cross-entropy over bins is bounded per
   example: being wrong by one bin and wrong by two hundred cost a similar
   amount. The write-up already suggests Huber loss for this reason;
   tokenization is a stronger version of the same move.

3. **The architecture question goes away.** Any next-token model works
   unchanged.

WHAT YOU PAY, AND IT IS NOT SUBTLE
-----------------------------------
1. **A hard accuracy floor.** With B bins you can never predict better than
   half a bin width. `quantization_error_floor` computes it — do that FIRST,
   because if the floor is near your target RMSE the experiment is over before
   it starts.

2. **Ordinality is destroyed.** Cross-entropy treats bin 5 and bin 6 as exactly
   as different as bin 5 and bin 200. A language model does not care that
   "cat" and "cats" are adjacent, but here the labels have an order and
   throwing it away is a real loss. `soft_targets` restores some of it.

3. **The bin edges are fitted.** They must come from the TRAIN split only —
   the same rule as the scaler in §5, for the same reason. Test values outside
   the fitted range clip to the end bins, and `clip_rate` measures how often.

Plain PyTorch, no GPU, no download. Covered by `tests/test_tokenize_series.py`.

References:
- van den Oord et al. "WaveNet." 2016. https://arxiv.org/abs/1609.03499
- Ansari et al. "Chronos: Learning the Language of Time Series." 2024.
  https://arxiv.org/abs/2403.07815
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Quantizer:
    """
    Maps a real-valued series to token indices and back.

    Attributes:
        edges: (B + 1,) bin boundaries, ascending. Fitted on TRAIN ONLY.
        scheme: "uniform" or "quantile" — see `fit`.
    """

    edges: torch.Tensor
    scheme: str = "uniform"

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> torch.Tensor:
        """Bin midpoints — what a token decodes back to."""
        return (self.edges[:-1] + self.edges[1:]) / 2

    @classmethod
    def fit(cls, train_values: torch.Tensor, n_bins: int = 256,
            scheme: str = "uniform") -> "Quantizer":
        """
        Choose bin edges from the TRAINING split only.

        THE TWO SCHEMES DIFFER IN WHERE THEY SPEND RESOLUTION

        **uniform** — equal-width bins across [min, max]. Simple, and the bin
        width is constant so the error floor is uniform. But delta-bar is
        bell-ish: most mass sits near zero, so most bins go to tails that are
        almost never visited, and the crowded middle gets coarse resolution
        exactly where precision matters most.

        **quantile** — edges at empirical quantiles, so every bin holds roughly
        equal PROBABILITY MASS. Narrow bins in the dense middle, wide bins in
        the tails. This is usually the better choice for a bell-shaped signal,
        and it is what makes the token distribution roughly uniform, which in
        turn makes cross-entropy well conditioned.

        The trade: with quantile bins the error floor is no longer uniform —
        it is small in the middle and large in the tails. That is normally the
        right trade, but it means a single "floor" number is an average.

        Args:
            train_values: (N,) or (N, 1) — TRAINING data only.
            n_bins: vocabulary size. 256 is one byte; 4096 is Chronos-scale.
            scheme: "uniform" or "quantile".
        """
        if scheme not in ("uniform", "quantile"):
            raise ValueError(f"scheme must be uniform or quantile, got {scheme!r}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        v = train_values.flatten().float()
        if v.numel() == 0:
            raise ValueError("cannot fit a quantizer on an empty series")

        if scheme == "uniform":
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                hi = lo + 1e-6
            edges = torch.linspace(lo, hi, n_bins + 1)
        else:
            qs = torch.linspace(0, 1, n_bins + 1)
            edges = torch.quantile(v, qs)
            # Ties in the data produce duplicate edges, which would create
            # zero-width bins that no value can ever fall into — dead entries
            # in the vocabulary. Nudge them apart.
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-8
        return cls(edges=edges, scheme=scheme)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Real values -> token indices in [0, n_bins - 1].

        Values outside the fitted range CLIP to the end bins rather than
        raising. That is the right behaviour for a non-stationary series — a
        test period can legitimately exceed anything seen in training — but it
        is lossy, so check `clip_rate` before trusting the result.
        """
        v = values.flatten().float()
        idx = torch.bucketize(v, self.edges[1:-1].contiguous(), right=False)
        return idx.clamp(0, self.n_bins - 1).reshape(values.shape)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token indices -> the midpoint of their bin."""
        return self.centers.to(tokens.device)[tokens.long()]

    def clip_rate(self, values: torch.Tensor) -> float:
        """
        Fraction of values falling outside the fitted range.

        Above a percent or so, the tails are being flattened and the model
        cannot represent the extremes at all — which on financial data is
        exactly where the interesting days live. Widen the range, add bins, or
        refit on a longer window.
        """
        v = values.flatten().float()
        out = ((v < self.edges[0]) | (v > self.edges[-1])).float().mean()
        return float(out)

    def quantization_error_floor(self, values: torch.Tensor) -> float:
        """
        The RMSE a PERFECT token model would still incur. Compute this first.

        Encoding then decoding loses everything finer than a bin, so

            floor = RMSE(values, decode(encode(values)))

        is a hard lower bound on any tokenized model's error. For uniform bins
        of width w, quantization noise is approximately uniform on
        [-w/2, w/2], giving floor ~ w / sqrt(12) — but this returns the
        MEASURED value, which is what matters when bins are quantile-spaced or
        clipping is occurring.

        **If this is close to your target RMSE, tokenization cannot work at
        that resolution and no amount of modelling will fix it.**
        """
        v = values.flatten().float()
        recon = self.decode(self.encode(v))
        return float(torch.sqrt(((v - recon) ** 2).mean()))


def scale_windows(windows: torch.Tensor, eps: float = 1e-6
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalise each lookback window by its OWN mean and scale, Chronos-style.

    WHY THIS IS NOT OPTIONAL ON FINANCIAL DATA

    Fitting bin edges on the training split and applying them to the test split
    fails the moment the test period leaves the training range -- which for a
    non-stationary series is not an edge case, it is the normal case. Measured
    on AAPL delta-bar, 2015-2025:

        train range   [-21.07, 30.44]
        test range    [-50.46, 32.97]     <- the 2022 drawdown
        clip rate     3.49% of test values pinned to an end bin

    And the damage does not respond to resolution. Going from 4-bit to 12-bit
    moves the error floor 2.53 -> 2.16 and then stops, because the residual is
    clipping, not bin width. Sixteen times the vocabulary buys nothing.

    The fix is the "scaling" half of Chronos's "scaling and quantization":
    normalise each window before binning, so the vocabulary describes SHAPE
    relative to local context rather than absolute level. A window sitting at
    -50 and a window sitting at -10 with the same shape now map to the same
    tokens, and the tokens stay in range because they are always relative.

    You give up absolute level, and get it back at decode time from the stored
    offset and scale. That trade is almost always correct here: the level of
    delta-bar is itself non-stationary, so a model that had to represent it
    absolutely would be learning a moving target.

    Args:
        windows: (N, L) lookback windows.
        eps: floor on the scale, so a flat window cannot divide by zero.

    Returns:
        (scaled, offset, scale), each (N, L) / (N, 1) / (N, 1). Invert with
        `scaled * scale + offset`.
    """
    if windows.dim() != 2:
        raise ValueError(f"expected (N, L), got {tuple(windows.shape)}")

    offset = windows.mean(dim=1, keepdim=True)
    # Mean ABSOLUTE deviation, not std: delta-bar has heavy tails, and a single
    # crash day inflates std enough to squash every other window toward zero.
    scale = (windows - offset).abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return (windows - offset) / scale, offset, scale


def soft_targets(tokens: torch.Tensor, n_bins: int,
                 sigma: float = 1.0) -> torch.Tensor:
    r"""
    Turn hard token labels into a Gaussian over NEIGHBOURING bins.

    THE PROBLEM THIS FIXES

    Plain cross-entropy is ordinal-blind. If the truth is bin 100, predicting
    101 and predicting 250 receive the same loss — but one is nearly right and
    the other is nonsense. For language that blindness is correct ("cat" and
    "cats" being adjacent in the vocabulary means nothing). For a quantized
    real number it throws away the single most useful piece of structure the
    labels have.

    Replacing the one-hot target with

        q_j  proportional to  exp( -(j - target)^2 / (2 sigma^2) )

    makes the loss aware of distance again: the model is rewarded for landing
    near the right bin, not only on it. This is label smoothing with a metric,
    and it is the cheapest available repair.

    sigma is in BINS. sigma=1 spreads over immediate neighbours; sigma=0
    degenerates to one-hot.

    Returns:
        (..., n_bins), each row summing to 1.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    grid = torch.arange(n_bins, device=tokens.device, dtype=torch.float32)
    if sigma == 0:
        return torch.nn.functional.one_hot(tokens.long(), n_bins).float()

    d = grid.view(*([1] * tokens.dim()), n_bins) - tokens.unsqueeze(-1).float()
    logits = -(d ** 2) / (2 * sigma ** 2)
    return torch.softmax(logits, dim=-1)


def expected_value(probs: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
    """
    Collapse a predicted distribution to a point forecast.

    Two ways to do this and they are not equivalent:

        argmax      the single most likely bin — the MODE
        expectation sum_j p_j * center_j — the MEAN

    For a symmetric, unimodal prediction they roughly agree. For a **bimodal**
    one — the model thinks the signal will either revert hard or break out —
    the mean lands between the two modes, in a bin the model considers
    unlikely. That is bad as a forecast and highly informative as a signal, and
    it is visible only if you look at the whole distribution rather than a
    collapsed number.

    Expectation minimises squared error, so it is the right choice when you
    are being scored on RMSE. Which is itself a reason to be suspicious of
    RMSE on a task like this.
    """
    return (probs * quantizer.centers.to(probs.device)).sum(-1)


def distribution_stats(probs: torch.Tensor,
                       quantizer: Quantizer) -> Tuple[float, float, float]:
    """
    Mean, standard deviation and entropy (in bits) of a predicted distribution.

    Entropy is the interesting one and it has no analogue in a point forecast.
    log2(n_bins) means the model is saying "no idea" — for 256 bins that is
    8 bits. A confident forecast might be 3-4. Watching entropy over a test
    period tells you WHEN the model thinks it knows something, which is
    strictly more actionable than a single RMSE for the whole period.
    """
    centers = quantizer.centers.to(probs.device)
    mean = (probs * centers).sum(-1)
    var = (probs * (centers - mean.unsqueeze(-1)) ** 2).sum(-1)
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(-1)
    return float(mean.mean()), float(var.sqrt().mean()), float(entropy.mean())


if __name__ == "__main__":
    torch.manual_seed(0)
    bar = "=" * 78

    # A stand-in with the right shape: bell-ish, centred near zero, heavy tails.
    n = 4000
    signal = torch.randn(n) * 8.0 + torch.randn(n).sign() * torch.rand(n) * 4
    train, test = signal[:2800], signal[2800:]

    print(bar)
    print("  Quantizing the mean-reversion signal into a vocabulary")
    print(bar)
    print(f"  train range  [{train.min():.2f}, {train.max():.2f}]"
          f"   std {train.std():.2f}")

    print()
    print("  THE FLOOR — the RMSE a PERFECT token model still pays")
    print(f"  {'bits':>5} {'bins':>6}  {'uniform floor':>15}  {'quantile floor':>16}"
          f"  {'clip rate':>10}")
    print("  " + "-" * 60)
    for bits in (4, 6, 8, 10, 12):
        b = 2 ** bits
        qu = Quantizer.fit(train, b, "uniform")
        qq = Quantizer.fit(train, b, "quantile")
        print(f"  {bits:>5} {b:>6}  {qu.quantization_error_floor(test):>15.4f}"
              f"  {qq.quantization_error_floor(test):>16.4f}"
              f"  {qu.clip_rate(test):>10.4f}")
    print()
    print("  Read this BEFORE building anything. If the floor is close to the")
    print("  RMSE you are trying to beat, the experiment is already over.")

    print()
    print(bar)
    print("  Uniform vs quantile: where the resolution goes")
    print(bar)
    for scheme in ("uniform", "quantile"):
        q = Quantizer.fit(train, 256, scheme)
        tok = q.encode(train)
        counts = torch.bincount(tok, minlength=256).float()
        used = int((counts > 0).sum())
        print(f"  {scheme:<9} bins used {used:>4}/256   "
              f"busiest bin holds {counts.max() / len(train):.1%} of the data")
    print()
    print("  Uniform bins spend most of the vocabulary on tails that are")
    print("  almost never visited. Quantile bins equalise the mass, which")
    print("  makes the token distribution — and cross-entropy — better behaved.")

    print()
    print(bar)
    print("  Ordinality: what cross-entropy throws away")
    print(bar)
    q = Quantizer.fit(train, 64, "quantile")
    t = torch.tensor([32])
    for sigma in (0.0, 1.0, 3.0):
        st = soft_targets(t, 64, sigma)[0]
        nz = int((st > 1e-4).sum())
        print(f"  sigma={sigma:<4} mass on bin 32: {st[32]:.4f}   "
              f"bins with >1e-4: {nz}")
    print()
    print("  sigma=0 is plain one-hot: bin 33 is punished exactly as hard as")
    print("  bin 250. Spreading the target over neighbours restores the")
    print("  distance information the labels had all along.")
    print(bar)
