# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: modern time-series primitives and value tokenization.

Run:
    uv run tests/test_ts_forecasting.py

Why this suite exists
---------------------
Covers `02_intermediate/02_rnn_stock_data/modern_ts_layers.py` (N-BEATS bases,
PatchTST patching, TCN dilation arithmetic) and `tokenize_series.py` (treating
the mean-reversion signal as a vocabulary).

The properties pinned here are the ones that fail silently:

  * **causal convolution is actually causal.** A centred `conv1d` with
    `padding=k//2` reads t+1..t+k//2 — look-ahead bias hidden inside a layer,
    the same class of bug as the scaler leak and just as quiet. Verified by
    perturbing a future input and asserting earlier outputs do not move.

  * **the receptive field formula is right.** If R < lookback, the early part
    of the window is structurally invisible and nothing warns you. Checked
    against a directly measured field, not just against itself.

  * **encode/decode round-trips within a bin.** The whole tokenization idea
    rests on the error floor being computable and honest.

  * **the N-BEATS bases are complementary.** Trend cannot express periodicity;
    seasonality cannot express trend. If either could, the decomposition would
    be meaningless.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "02_intermediate/02_rnn_stock_data"))

from modern_ts_layers import (  # noqa: E402
    causal_conv1d,
    dilated_receptive_field,
    multi_scale_decompose,
    n_patches,
    patchify,
    seasonality_basis,
    trend_basis,
)
from tokenize_series import (  # noqa: E402
    Quantizer,
    distribution_stats,
    expected_value,
    scale_windows,
    soft_targets,
)


def test_nbeats_bases_are_complementary(r: Results) -> None:
    """Trend cannot be periodic; seasonality cannot trend. That is the point."""
    H = 24
    tb = trend_basis(H, degree=3)
    sb = seasonality_basis(H, n_harmonics=4)

    r.check(tb.shape == (4, H), "trend basis is (degree+1, H)",
            f"got {tuple(tb.shape)}")
    r.check(sb.shape == (8, H), "seasonality basis is (2*harmonics, H)",
            f"got {tuple(sb.shape)}")

    r.check(torch.allclose(tb[0], torch.ones(H)),
            "the degree-0 trend row is constant (the level term)")
    r.check(bool((tb[1].diff() > 0).all()),
            "the degree-1 row is strictly increasing (the slope term)")

    # Every seasonality row must integrate to ~0 over the horizon — that is
    # what makes it unable to represent a level or a trend.
    sums = sb.sum(dim=1).abs()
    r.check(bool((sums < 1e-4).all()),
            "every seasonality row sums to ~0, so it cannot express trend",
            f"max |sum| = {sums.max():.2e}")

    # And the trend rows do NOT sum to zero — the complement.
    r.check(abs(float(tb[0].sum()) - H) < 1e-4,
            "the trend level row sums to H, so it CAN express a level",
            f"got {float(tb[0].sum())}")

    # Time is normalised to t/H, so the basis is scale-free in the horizon.
    tb_long = trend_basis(1000, degree=3)
    r.check(float(tb_long[3].max()) <= 1.0 + 1e-6,
            "t^3 stays bounded at large H (time is normalised to t/H)",
            f"max {float(tb_long[3].max())} — un-normalised t^3 at H=1000 "
            "would be 1e9 and the consuming Linear would be ill-conditioned")

    for bad, label in [((0, 3), "horizon=0"), ((10, -1), "degree=-1")]:
        try:
            trend_basis(*bad); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"trend_basis rejects {label}")
    try:
        seasonality_basis(10, 0); caught = False
    except ValueError:
        caught = True
    r.check(caught, "seasonality_basis rejects n_harmonics=0")


def test_patching(r: Results) -> None:
    """Patch count arithmetic must match what patchify produces."""
    B, L = 3, 60
    x = torch.arange(B * L, dtype=torch.float32).reshape(B, L)

    for pl, st in [(16, 8), (16, 16), (8, 4), (60, 60)]:
        p = patchify(x, pl, st)
        expect = n_patches(L, pl, st)
        r.check(p.shape == (B, expect, pl),
                f"patch_len={pl} stride={st}: shape matches n_patches()",
                f"got {tuple(p.shape)}, n_patches said {expect}")

    # Non-overlapping patches must tile the series exactly, in order.
    p = patchify(x, 10, 10)
    r.check(torch.equal(p.reshape(B, -1), x),
            "stride == patch_len tiles the input exactly, preserving order")

    # Overlapping patches must share their overlap.
    p = patchify(x, 16, 8)
    r.check(torch.equal(p[0, 0, 8:], p[0, 1, :8]),
            "stride < patch_len makes consecutive patches share their overlap")

    # 3-D input keeps the channel axis.
    p3 = patchify(torch.randn(B, L, 2), 16, 8)
    r.check(p3.shape == (B, 2, n_patches(L, 16, 8), 16),
            "3-D input yields (B, C, n_patches, patch_len)",
            f"got {tuple(p3.shape)}")

    for args, label in [((torch.randn(2, 5), 10, 5), "patch_len > length"),
                        ((torch.randn(2, 20), 0, 5), "patch_len=0"),
                        ((torch.randn(2, 20), 5, 0), "stride=0")]:
        try:
            patchify(*args); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"patchify rejects {label}")


def test_causality_and_receptive_field(r: Results) -> None:
    """
    The load-bearing property: a causal conv must not see the future.

    An ordinary centred conv1d WOULD, and nothing about the training run would
    reveal it — the same silent-leak class as the scaler bug in section 5.
    """
    torch.manual_seed(0)
    L = 16
    x = torch.randn(1, 1, L)
    w = torch.randn(1, 1, 3)

    for dilation in (1, 2, 4):
        base = causal_conv1d(x, w, dilation=dilation)
        r.check(base.shape == (1, 1, L),
                f"dilation={dilation}: output length equals input length",
                f"got {tuple(base.shape)}")

        # Perturb a late timestep; no EARLIER output may move.
        t_pert = 10
        x2 = x.clone(); x2[0, 0, t_pert] += 100.0
        moved = (causal_conv1d(x2, w, dilation=dilation) - base).abs().squeeze() > 1e-5
        earlier = [i for i in range(t_pert) if bool(moved[i])]
        r.check(not earlier,
                f"dilation={dilation}: perturbing t={t_pert} moves NO earlier output",
                f"outputs {earlier} moved — the layer is reading the future")
        r.check(bool(moved[t_pert]),
                f"dilation={dilation}: it does affect its own timestep "
                "(the test is not vacuous)")

    # Receptive field formula, checked against a MEASURED field rather than
    # against itself: perturb t=0 and see how far the influence reaches.
    for k, n in [(2, 3), (3, 4), (3, 5)]:
        predicted = dilated_receptive_field(k, n)
        length = predicted + 20
        sig = torch.zeros(1, 1, length)
        weights = [torch.ones(1, 1, k) for _ in range(n)]
        pert = sig.clone(); pert[0, 0, 0] = 1.0
        a, b = sig, pert
        for i, wt in enumerate(weights):
            a = causal_conv1d(a, wt, dilation=2 ** i)
            b = causal_conv1d(b, wt, dilation=2 ** i)
        influenced = ((a - b).abs().squeeze() > 1e-8).nonzero().flatten()
        measured = int(influenced.max()) + 1 if len(influenced) else 0
        r.check(measured == predicted,
                f"k={k}, {n} layers: measured receptive field {measured} "
                f"matches the formula",
                f"formula said {predicted}")

    r.check(dilated_receptive_field(3, 5) >= 60,
            "k=3 with 5 layers covers a 60-day window (R=63)",
            f"R={dilated_receptive_field(3, 5)} — fewer layers and the early "
            "days are structurally invisible")
    r.check(dilated_receptive_field(3, 4) < 60,
            "k=3 with 4 layers does NOT (R=31) — the check matters")


def test_quantizer_roundtrip(r: Results) -> None:
    """Encode/decode must land in the right bin, and the floor must be honest."""
    torch.manual_seed(1)
    train = torch.randn(3000) * 5.0

    for scheme in ("uniform", "quantile"):
        q = Quantizer.fit(train, n_bins=256, scheme=scheme)
        r.check(q.n_bins == 256, f"{scheme}: n_bins is as requested")
        r.check(len(q.centers) == 256, f"{scheme}: one center per bin")
        r.check(bool((q.edges.diff() > 0).all()),
                f"{scheme}: edges are strictly increasing (no dead bins)",
                "duplicate edges create zero-width bins nothing can land in")

        tok = q.encode(train)
        r.check(bool(((tok >= 0) & (tok < 256)).all()),
                f"{scheme}: every token is a valid index")

        # Round-trip error must be bounded by the widest bin.
        err = (q.decode(tok) - train).abs().max()
        widest = q.edges.diff().max()
        r.check(float(err) <= float(widest),
                f"{scheme}: round-trip error is within one bin width",
                f"err {float(err):.4f} vs widest bin {float(widest):.4f}")

    # More bins must lower the floor — on in-range data, monotonically.
    q_floors = [Quantizer.fit(train, b, "uniform").quantization_error_floor(train)
                for b in (16, 64, 256, 1024)]
    r.check(all(a > b for a, b in zip(q_floors, q_floors[1:])),
            "more bins strictly lowers the quantization floor (in-range data)",
            f"{[round(f, 4) for f in q_floors]}")

    # Quantile bins must spread the mass more evenly than uniform bins.
    qu = Quantizer.fit(train, 64, "uniform")
    qq = Quantizer.fit(train, 64, "quantile")
    spread_u = torch.bincount(qu.encode(train), minlength=64).float().std()
    spread_q = torch.bincount(qq.encode(train), minlength=64).float().std()
    r.check(spread_q < spread_u,
            "quantile bins equalise token frequency better than uniform",
            f"std of counts: quantile {spread_q:.1f} vs uniform {spread_u:.1f}")

    # Out-of-range values must CLIP, not crash or wrap.
    q = Quantizer.fit(train, 64, "uniform")
    extreme = torch.tensor([-1e6, 1e6])
    tok = q.encode(extreme)
    r.check(int(tok[0]) == 0 and int(tok[1]) == 63,
            "out-of-range values clip to the end bins",
            f"got {tok.tolist()}")
    r.check(q.clip_rate(extreme) == 1.0,
            "clip_rate reports 100% when everything is out of range")
    r.check(q.clip_rate(train) < 0.01,
            "clip_rate is ~0 on the data the bins were fitted to")

    for args, label in [((train, 1), "n_bins=1"),
                        ((torch.tensor([]), 16), "empty series")]:
        try:
            Quantizer.fit(*args); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"Quantizer.fit rejects {label}")
    try:
        Quantizer.fit(train, 16, "bogus"); caught = False
    except ValueError:
        caught = True
    r.check(caught, "Quantizer.fit rejects an unknown scheme")


def test_window_scaling_fixes_clipping(r: Results) -> None:
    """
    THE result: per-window scaling is what makes tokenization viable here.

    Without it, a test period that leaves the training range clips, and adding
    bins does not help because the residual is clipping rather than resolution.
    """
    torch.manual_seed(2)
    # Train windows near zero; test windows shifted far away — a regime change,
    # which is exactly what section 7 says to expect.
    train = torch.randn(400, 60) * 3.0
    test = torch.randn(200, 60) * 3.0 + 40.0

    q_raw = Quantizer.fit(train.flatten(), 256, "uniform")
    clip_raw = q_raw.clip_rate(test.flatten())

    tr_s, _, _ = scale_windows(train)
    te_s, off, scale = scale_windows(test)
    q_scaled = Quantizer.fit(tr_s.flatten(), 256, "uniform")
    clip_scaled = q_scaled.clip_rate(te_s.flatten())

    r.check(clip_raw > 0.5,
            "WITHOUT scaling, a shifted test period clips heavily",
            f"clip rate {clip_raw:.1%}")
    r.check(clip_scaled < 0.01,
            "WITH per-window scaling, clipping essentially vanishes",
            f"clip rate {clip_scaled:.4%}")
    r.check(clip_scaled < clip_raw / 10,
            "scaling improves the clip rate by more than 10x",
            f"{clip_raw:.4f} -> {clip_scaled:.4f}")

    # Scaling must be exactly invertible, or the decoded forecast is wrong.
    r.check(torch.allclose(te_s * scale + off, test, atol=1e-4),
            "scale_windows is invertible: scaled * scale + offset == original",
            f"max err {(te_s * scale + off - test).abs().max():.2e}")
    r.check(torch.allclose(te_s.mean(dim=1), torch.zeros(len(te_s)), atol=1e-5),
            "each scaled window has ~zero mean")

    # A flat window must not divide by zero.
    flat = torch.ones(3, 60)
    s, _, sc = scale_windows(flat)
    r.check(bool(torch.isfinite(s).all()) and bool((sc > 0).all()),
            "a constant window does not produce NaN or a zero scale")

    try:
        scale_windows(torch.randn(60)); caught = False
    except ValueError:
        caught = True
    r.check(caught, "scale_windows rejects a 1-D input")


def test_soft_targets_and_distribution(r: Results) -> None:
    """Ordinality repair, and the distribution utilities built on it."""
    n_bins = 64
    t = torch.tensor([32, 0, 63])

    hard = soft_targets(t, n_bins, sigma=0.0)
    r.check(torch.allclose(hard.sum(-1), torch.ones(3)),
            "sigma=0 targets sum to 1")
    r.check(float(hard[0, 32]) == 1.0,
            "sigma=0 is exactly one-hot (plain cross-entropy)")

    soft = soft_targets(t, n_bins, sigma=2.0)
    r.check(torch.allclose(soft.sum(-1), torch.ones(3), atol=1e-5),
            "soft targets still sum to 1")
    r.check(float(soft[0, 32]) > float(soft[0, 34]) > float(soft[0, 40]),
            "mass decreases with distance from the true bin",
            "this is what restores the ordering that cross-entropy discards")
    r.check(float(soft[0, 33]) > float(hard[0, 33]),
            "a neighbouring bin gets more mass than under one-hot")

    # Wider sigma must spread further.
    narrow = soft_targets(t, n_bins, 1.0)[0]
    wide = soft_targets(t, n_bins, 5.0)[0]
    r.check(float(narrow[32]) > float(wide[32]),
            "larger sigma spreads the target further")

    try:
        soft_targets(t, n_bins, -1.0); caught = False
    except ValueError:
        caught = True
    r.check(caught, "soft_targets rejects negative sigma")

    # expected_value must minimise squared error — check it recovers a center.
    q = Quantizer.fit(torch.randn(2000) * 4, n_bins, "uniform")
    onehot = torch.zeros(1, n_bins); onehot[0, 20] = 1.0
    r.check(torch.allclose(expected_value(onehot, q), q.centers[20:21], atol=1e-5),
            "expected_value of a one-hot distribution is that bin's center")

    uniform = torch.full((1, n_bins), 1.0 / n_bins)
    mean_, std_, ent = distribution_stats(uniform, q)
    import math
    r.check(abs(ent - math.log2(n_bins)) < 1e-3,
            f"a uniform distribution has entropy log2({n_bins}) = "
            f"{math.log2(n_bins):.1f} bits",
            f"got {ent:.4f} — this is the 'no idea' reading")
    _, _, ent_sharp = distribution_stats(onehot, q)
    r.check(ent_sharp < 0.01,
            "a one-hot distribution has ~0 entropy (maximal confidence)",
            f"got {ent_sharp:.4f}")


def test_multi_scale(r: Results) -> None:
    """Coarser views must be shorter and smoother."""
    torch.manual_seed(3)
    x = torch.randn(2, 64, 1)
    views = multi_scale_decompose(x, (1, 2, 4))

    r.check([v.shape[1] for v in views] == [64, 32, 16],
            "each scale halves the length",
            f"got {[v.shape[1] for v in views]}")
    r.check(torch.equal(views[0], x), "scale 1 is the input, untouched")
    stds = [float(v.std()) for v in views]
    r.check(stds[0] > stds[1] > stds[2],
            "coarser scales have lower variance (fast noise averaged out)",
            f"{[round(s, 3) for s in stds]}")

    for args, label in [((torch.randn(2, 64), (1, 2)), "2-D input"),
                        ((x, (0, 2)), "scale 0")]:
        try:
            multi_scale_decompose(*args); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"multi_scale_decompose rejects {label}")


def main() -> int:
    r = Results("Modern TS primitives and value tokenization")
    test_nbeats_bases_are_complementary(r)
    test_patching(r)
    test_causality_and_receptive_field(r)
    test_quantizer_roundtrip(r)
    test_window_scaling_fixes_clipping(r)
    test_soft_targets_and_distribution(r)
    test_multi_scale(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
