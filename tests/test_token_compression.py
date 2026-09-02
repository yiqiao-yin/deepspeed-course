# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: visual token compression must be CORRECT, not merely smaller.

Run:
    uv run tests/test_token_compression.py

Why this suite exists
---------------------
Compression code fails in a uniquely nasty way: it always "works". Drop the
wrong tokens and the model still runs, the loss still decreases, and the only
symptom is a benchmark score a few points below what the paper reported --
which you will blame on the learning rate. Nothing raises.

So we assert the mathematical properties instead of eyeballing shapes:

  * ToMe merges the MOST SIMILAR pair, not an arbitrary one.
  * Size-weighted merging is exactly the mean of the ORIGINAL tokens a merged
    token represents -- which is the property that makes ToMe stackable.
  * The log-size attention bias reproduces, to floating point, the softmax you
    would have gotten without merging. This is the one people skip.
  * FastV keeps the highest-attention tokens and returns them SORTED, since an
    unsorted gather silently scrambles positional order.
  * Temporal merging drops a static background and preserves motion.

Everything runs on CPU in a couple of seconds. That is the point: these are
the properties you can prove before you rent an 80 GB card.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "04_video_text" / "03_token_compression"))

from token_compression import (  # noqa: E402
    TokenBudget,
    bipartite_soft_matching,
    count_visual_tokens,
    dycoke_temporal_merge,
    fastv_select,
    merge_wavg,
    proportional_attention_bias,
)


def test_tome_matching(r: Results) -> None:
    """ToMe must merge the most similar pair, and only r of them."""
    torch.manual_seed(0)

    # Six tokens. Positions 0 and 1 are made near-identical; everything else
    # is random. Index 0 is in set A, index 1 is in set B, so 0 -> 1 is a
    # legal edge and should be the single best one.
    metric = torch.randn(1, 6, 8)
    metric[0, 0] = metric[0, 1] + 1e-4

    x = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).repeat(1, 1, 4)

    merge, unmerge = bipartite_soft_matching(metric, r=1)
    merged, size = merge_wavg(merge, x)

    r.check(merged.shape == (1, 5, 4),
            "ToMe removes exactly r tokens",
            f"got {tuple(merged.shape)}, expected (1, 5, 4)")

    # Exactly one token should now have size 2; the rest size 1.
    sizes = size.flatten().tolist()
    r.check(sorted(sizes) == [1.0, 1.0, 1.0, 1.0, 2.0],
            "exactly one merged token, of size 2",
            f"sizes were {sorted(sizes)}")

    # The merged token must be the mean of tokens 0 and 1, i.e. 0.5 --
    # proving it merged the SIMILAR pair and not some random pair.
    merged_vals = merged.flatten(start_dim=1)[0, ::4].tolist()
    r.check(any(abs(v - 0.5) < 1e-5 for v in merged_vals),
            "merged the most-similar pair (0 and 1 -> mean 0.5)",
            f"merged values were {merged_vals}")

    # Round trip must restore the original count.
    restored = unmerge(merged)
    r.check(restored.shape == x.shape,
            "unmerge restores the original token count",
            f"got {tuple(restored.shape)}")


def test_tome_weighted_average_is_exact(r: Results) -> None:
    """
    After several merge rounds a token must equal the plain mean of the
    ORIGINAL tokens it stands for.

    This is the property that justifies size-weighting. An unweighted mean
    drifts -- a token representing 8 patches would get the same vote as a
    token representing 1 -- and the drift compounds every layer. If this test
    passes, ToMe is safe to stack; if it fails, deep stacks quietly degrade.
    """
    torch.manual_seed(1)
    n = 16
    x = torch.randn(1, n, 6)
    metric = torch.randn(1, n, 6)

    running, size = x.clone(), None
    total = torch.ones(1, n, 1)

    # Track, per surviving token, exactly which originals it absorbed.
    for _ in range(3):
        merge, _ = bipartite_soft_matching(metric, r=2)
        running, size = merge_wavg(merge, running, size)
        # The metric must follow the same merge, or round 2 indexes garbage.
        metric, total = merge_wavg(merge, metric, total)

    # Conservation: the sizes must still sum to the original token count.
    # If they do not, tokens were double-counted or silently dropped.
    r.check(abs(size.sum().item() - n) < 1e-4,
            "token sizes conserve the original count across merge rounds",
            f"sizes summed to {size.sum().item()}, expected {n}")

    # The size-weighted sum of merged tokens must equal the sum of originals.
    lhs = (running * size).sum(dim=1)
    rhs = x.sum(dim=1)
    r.check(torch.allclose(lhs, rhs, atol=1e-4),
            "weighted merge preserves the total feature mass exactly",
            f"max abs diff {(lhs - rhs).abs().max().item():.2e}")


def test_proportional_attention(r: Results) -> None:
    """
    The log-size bias must reproduce unmerged softmax exactly.

    Setup: two tokens share an identical key, so merging them loses nothing
    informationally -- but halves their share of the softmax denominator. The
    bias is supposed to undo precisely that. We check it numerically rather
    than trusting the derivation.
    """
    torch.manual_seed(2)
    d = 8
    query = torch.randn(1, 1, 1, d)

    key_shared = torch.randn(d)
    key_other = torch.randn(d)

    # Uncompressed: the shared key appears TWICE.
    keys_full = torch.stack([key_shared, key_shared, key_other])[None, None]
    logits_full = (query @ keys_full.transpose(-1, -2)) / d ** 0.5
    attn_full = logits_full.softmax(dim=-1)
    # Total mass on the shared key = the two duplicate columns.
    mass_full = attn_full[..., 0] + attn_full[..., 1]

    # Compressed: it appears ONCE, with size 2.
    keys_merged = torch.stack([key_shared, key_other])[None, None]
    logits_merged = (query @ keys_merged.transpose(-1, -2)) / d ** 0.5
    size = torch.tensor([[[2.0], [1.0]]])

    naive = logits_merged.softmax(dim=-1)[..., 0]
    corrected = (logits_merged + proportional_attention_bias(size)).softmax(
        dim=-1
    )[..., 0]

    r.check(torch.allclose(corrected, mass_full, atol=1e-6),
            "log-size bias exactly reproduces unmerged attention mass",
            f"corrected={corrected.item():.6f} vs true={mass_full.item():.6f}")

    # And confirm the bug is real: without the bias you are measurably wrong.
    r.check((naive - mass_full).abs().item() > 1e-3,
            "without the bias, merged attention is measurably wrong",
            f"naive={naive.item():.6f} vs true={mass_full.item():.6f} "
            "-- if these matched, the test setup would be proving nothing")


def test_tome_protects_class_token(r: Results) -> None:
    """Token 0 must survive even when it is the most mergeable thing present."""
    metric = torch.ones(1, 8, 4)          # everything maximally similar
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

    merge, _ = bipartite_soft_matching(metric, r=3, class_token=True)
    merged, size = merge_wavg(merge, x)

    r.check(merged.shape[1] == 5, "removes r tokens with class_token set",
            f"got {merged.shape[1]}, expected 5")
    # Token 0's value is 0.0 and it must appear untouched with size 1.
    zero_idx = [i for i, v in enumerate(merged.flatten().tolist())
                if abs(v) < 1e-6]
    r.check(bool(zero_idx) and abs(size.flatten()[zero_idx[0]].item() - 1.0) < 1e-6,
            "class token survives unmerged (size stays 1)",
            f"merged={merged.flatten().tolist()}, sizes={size.flatten().tolist()}")


def test_fastv_selection(r: Results) -> None:
    """FastV must keep the highest-attention tokens, in sorted order."""
    b, h, q, k = 1, 4, 10, 10
    attn = torch.zeros(b, h, q, k)

    # Visual span is [2, 10). Give tokens 3, 5, 7 strong attention from the
    # last query row; the rest near-zero.
    attn[:, :, -1, 2:] = 0.01
    for idx in (3, 5, 7):
        attn[:, :, -1, idx] = 0.9

    keep = fastv_select(attn, visual_start=2, visual_end=10, keep_ratio=3 / 8)

    r.check(keep.shape == (1, 3), "keeps round(n_visual * ratio) tokens",
            f"got {tuple(keep.shape)}")
    r.check(keep[0].tolist() == [3, 5, 7],
            "keeps the highest-attention tokens, sorted ascending",
            f"got {keep[0].tolist()}, expected [3, 5, 7]")

    # Sorting is load-bearing: RoPE and the causal mask both assume order.
    monotone = bool((keep[0][1:] > keep[0][:-1]).all())
    r.check(monotone, "indices are strictly increasing (positional safety)")

    # keep_ratio=1.0 must be a genuine no-op, not an off-by-one.
    allkeep = fastv_select(attn, 2, 10, keep_ratio=1.0)
    r.check(allkeep[0].tolist() == list(range(2, 10)),
            "keep_ratio=1.0 returns the whole visual span unchanged",
            f"got {allkeep[0].tolist()}")

    # Guard rails.
    for bad, label in [(0.0, "keep_ratio=0"), (1.5, "keep_ratio>1")]:
        try:
            fastv_select(attn, 2, 10, keep_ratio=bad)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects invalid {label}")


def test_temporal_merge(r: Results) -> None:
    """
    Static regions must collapse; moving regions must survive.

    A compressor that drops motion is worse than useless -- motion is the only
    reason the clip is a video rather than a photograph.
    """
    b, t, n, c = 1, 8, 6, 4
    frames = torch.zeros(b, t, n, c)

    # Tokens 0-3: a frozen background, identical in every frame.
    frames[:, :, 0:4, :] = torch.randn(1, 1, 4, c)
    # Tokens 4-5: changing every frame -- this is the "action".
    frames[:, :, 4:6, :] = torch.randn(b, t, 2, c) * 5

    kept, mask = dycoke_temporal_merge(frames, window=4,
                                       similarity_threshold=0.9)

    r.check(kept.shape == frames.shape,
            "returns a dense tensor of the original shape",
            f"got {tuple(kept.shape)}")

    # Anchors -- the first frame of each window -- are always kept whole.
    r.check(bool(mask[0, 0].all()) and bool(mask[0, 4].all()),
            "window anchors (frames 0 and 4) are kept in full")

    static_kept = mask[0, :, 0:4].float().mean().item()
    motion_kept = mask[0, :, 4:6].float().mean().item()

    r.check(motion_kept > static_kept,
            "keeps more moving tokens than static ones",
            f"motion {motion_kept:.2f} vs static {static_kept:.2f}")
    # Non-anchor static tokens should be almost entirely dropped: 2 of 8
    # frames are anchors, so the ceiling for a perfectly static token is 0.25.
    r.check(static_kept <= 0.30,
            "static background collapses to roughly the anchor frames only",
            f"static retention {static_kept:.2f}, expected <= 0.30")

    # Dropped positions must be zeroed, so a stale value cannot leak through.
    r.check(bool((kept[~mask] == 0).all()),
            "dropped tokens are zeroed, not left stale")

    # window=1 means every frame is its own anchor: a guaranteed no-op.
    _, mask1 = dycoke_temporal_merge(frames, window=1)
    r.check(bool(mask1.all()), "window=1 is a no-op (every frame is an anchor)")


def test_token_accounting(r: Results) -> None:
    """The budget arithmetic must match the model geometry it claims."""
    # Qwen2.5-VL: 448/14 = 32 patches per side, 2x2 merge -> 16x16 = 256.
    per_frame = count_visual_tokens(1)
    r.check(per_frame == 256,
            "448x448 with patch 14 and 2x2 merge gives 256 tokens/frame",
            f"got {per_frame}")

    r.check(count_visual_tokens(64) == 16384,
            "a 64-frame clip is 16,384 visual tokens",
            f"got {count_visual_tokens(64)}")

    # Forgetting the 2x2 merger is the classic 4x overestimate.
    unmerged = count_visual_tokens(1, merge_size=1)
    r.check(unmerged == 4 * per_frame,
            "omitting the patch merger overestimates by exactly 4x",
            f"{unmerged} vs {per_frame}")

    budget = TokenBudget(original_tokens=16384, compressed_tokens=8192)
    r.check(abs(budget.keep_ratio - 0.5) < 1e-9, "keep_ratio computed correctly")
    r.check(abs(budget.attention_flops_ratio - 0.25) < 1e-9,
            "attention cost is quadratic in the keep ratio",
            f"got {budget.attention_flops_ratio}")
    r.check(abs(budget.mlp_flops_ratio - 0.5) < 1e-9,
            "mlp cost is linear in the keep ratio")

    # 8192 removed x 3584 hidden x 2 bytes x 2 (K and V) x 28 layers.
    expected = 8192 * 3584 * 2 * 2 * 28
    r.check(budget.kv_cache_bytes_saved == expected,
            "kv cache saving matches K+V, bf16, all layers",
            f"got {budget.kv_cache_bytes_saved}, expected {expected}")

    r.check(TokenBudget(0, 0).keep_ratio == 0.0,
            "empty budget does not divide by zero")


def main() -> int:
    r = Results("Visual token compression — algorithmic correctness")
    test_tome_matching(r)
    test_tome_weighted_average_is_exact(r)
    test_proportional_attention(r)
    test_tome_protects_class_token(r)
    test_fastv_selection(r)
    test_temporal_merge(r)
    test_token_accounting(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
