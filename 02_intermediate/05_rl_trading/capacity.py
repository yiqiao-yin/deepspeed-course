#!/usr/bin/env python3
"""
Where DeepSpeed earns its place here, and where it does not.

    uv run capacity.py                # the ladder and the ZeRO table
    uv run capacity.py --measure      # build each rung and measure it

THE HONEST ANSWER FOR THE HEADLINE ARMS IS "NOWHERE"
------------------------------------------------------
The policies in `train_rl.py` are **4,866 parameters** for GRPO and
**9,731** for PPO. Weights, gradients and Adam state together come
to about 78 and 156 kilobytes. ZeRO partitions exactly those three
things, so sharding them across eight cards saves under twenty
kilobytes a card and costs an all-gather on every step.

That is not a criticism of ZeRO. It is the same measurement labs
1 to 5 of topic 8 made about a 14k-parameter controller and lab 6
made about a dynamics model: **the thing that decides whether
sharding helps is the size of what you are sharding**, and here
the bottleneck is a sequential rollout over 2,992 trading days
that cannot be parallelised at all, because the position on day t
depends on the action on day t-1.

SO WHERE IS THE REAL MODEL?
-----------------------------
A policy that looks at nine sectors independently is nine copies
of a tiny network. A policy that attends ACROSS assets -- deciding
what to hold in energy partly because of what technology is doing
-- is one model whose width is set by the number of assets, and
that is the architecture an actual systematic desk runs. Scale it
from nine sectors to five hundred stocks with a real lookback and
it stops being small.

WHAT THIS TABLE SHOWS, AND THE REASON IT MATTERS
--------------------------------------------------
**Activations dominate, and ZeRO does not shard activations.**

    rung  assets   d    L       params  wt+Adam  activ  ratio  ZeRO-3
    xl       500 1024  12  151,028,736  2.42 GB  14.7G   6.1x     12%
    xxl      500 2048  16  805,373,952 12.89 GB  39.3G   3.1x     22%

At the `xl` rung there is **six times** more memory in activations
than in everything ZeRO can partition, so sharding the parameters
perfectly across eight cards moves the total only 17.2 -> 15.0 GB.
At `xxl` the parameters have grown faster than the activations and
the ratio falls to 3.1x, which is why the saving rises to 22% --
ZeRO helps more as the model gets wider relative to its input, and
that is the shape of the trade rather than a flat rule.

`06_protein_folding` reaches the identical conclusion from a
completely different direction: an Evoformer trunk is ~100k
parameters and fills a card with `O(N^2)` pair representations, so
ZeRO-1 and ZeRO-3 differ by 4.6%. Two domains with nothing in
common, one systems fact.

The lever that works on activations is gradient checkpointing,
which trades compute for memory and is a different tool. Naming
the right tool for the measured bottleneck is the whole point of
measuring it.
"""

from __future__ import annotations

import argparse

# (name, assets, d_model, layers, lookback)
LADDER = [
    ("sectors-tiny", 9, 64, 2, 1),
    ("sectors", 9, 128, 2, 20),
    ("midcap", 50, 256, 4, 40),
    ("large", 100, 512, 6, 60),
    ("xl", 500, 1024, 12, 60),
    ("xxl", 500, 2048, 16, 60),
]

BYTES_FP32 = 4
# Adam in fp32 keeps the parameter, its gradient, and two moments.
ADAM_BYTES_PER_PARAM = 16


def params(n_assets: int, d_model: int, layers: int, n_feat: int = 9) -> int:
    """
    A cross-sectional transformer: self-attention ACROSS assets at
    each timestep, then a feed-forward block.

    Per layer: 4 d^2 for the attention projections, 8 d^2 for the
    MLP at the usual 4x expansion. Plus the input projection and
    the scalar position head.
    """
    per_layer = 12 * d_model * d_model + 4 * d_model
    return layers * per_layer + n_feat * d_model + d_model + 1


def activations(n_assets: int, d_model: int, layers: int,
                lookback: int, batch: int = 1) -> int:
    """
    Stored activations for one forward pass, in bytes.

    About ten tensors of shape (batch, assets, lookback, d_model)
    survive per layer for the backward pass -- the residual stream,
    the projections, the attention output, the MLP hidden state.
    This is the term ZeRO cannot touch.
    """
    per_layer = batch * n_assets * lookback * d_model * 10
    return per_layer * layers * BYTES_FP32


def table(n_cards: int = 8) -> list[dict]:
    rows = []
    for name, n, d, L, lb in LADDER:
        p = params(n, d, L)
        w = p * ADAM_BYTES_PER_PARAM
        a = activations(n, d, L, lb)
        rows.append({
            "name": name, "assets": n, "d_model": d, "layers": L,
            "lookback": lb, "params": p,
            "gb_weights": w / 1e9,
            "gb_act": a / 1e9,
            # ZeRO-3 partitions parameters, gradients and optimizer
            # state. Activations are paid IN FULL by every rank.
            "gb_zero3": w / n_cards / 1e9,
            "gb_total_1card": (w + a) / 1e9,
            "gb_total_zero3": (w / n_cards + a) / 1e9,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=int, default=8)
    ap.add_argument("--measure", action="store_true",
                    help="actually build each rung on the meta device "
                         "and count, instead of trusting the arithmetic")
    a, _ = ap.parse_known_args()

    rows = table(a.cards)
    print("=" * 96)
    print(f"  THE CROSS-SECTIONAL LADDER — and what ZeRO-3 across "
          f"{a.cards} cards can and cannot shard")
    print("=" * 96)
    print(f"  {'rung':<14}{'assets':>7}{'d':>6}{'L':>4}{'params':>14}"
          f"{'wt+Adam':>10}{'activ':>9}{'1 card':>9}{'ZeRO-3':>9}{'saved':>8}")
    for r in rows:
        saved = 1 - r["gb_total_zero3"] / r["gb_total_1card"]
        print(f"  {r['name']:<14}{r['assets']:>7}{r['d_model']:>6}"
              f"{r['layers']:>4}{r['params']:>14,}"
              f"{r['gb_weights']:>9.2f}G{r['gb_act']:>8.2f}G"
              f"{r['gb_total_1card']:>8.2f}G{r['gb_total_zero3']:>8.2f}G"
              f"{saved:>8.0%}")

    if a.measure:
        import torch
        import torch.nn as nn

        print(f"\n  MEASURED — each rung built on the meta device, "
              f"no memory allocated")
        print(f"  {'rung':<14}{'predicted':>14}{'actual':>14}{'error':>9}")
        for name, n, d, L, lb in LADDER:
            with torch.device("meta"):
                enc = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d, nhead=max(1, d // 64),
                        dim_feedforward=4 * d, batch_first=True),
                    num_layers=L)
                inp = nn.Linear(9, d)
                head = nn.Linear(d, 1)
            actual = sum(p.numel() for m in (enc, inp, head)
                         for p in m.parameters())
            pred = params(n, d, L)
            print(f"  {name:<14}{pred:>14,}{actual:>14,}"
                  f"{abs(pred - actual) / actual:>8.1%}")
        print("\n  The arithmetic is a model of the architecture, not a")
        print("  measurement of it. Where they disagree, believe the build.")

    # The ACTUAL trained policies, not a ladder rung. An earlier
    # version quoted the `sectors` rung's 395,521 parameters here as
    # though it were the thing that produced the results. It is not:
    # the ladder is a family of hypothetical cross-sectional
    # transformers, and the arms in `train_rl.py` are plain MLPs an
    # eightieth that size. Quoting the wrong number would have
    # overstated this lab's own model by 40x in the one paragraph a
    # reader is most likely to carry away.
    TRAINED = {"GRPO (no critic)": 4_866, "PPO (with a critic)": 9_731}
    big = rows[-1]
    xl = rows[-2]
    print(f"\n" + "=" * 96)
    print("  WHAT THIS MEANS FOR THIS LAB")
    print("=" * 96)
    print("  Every published number came from these, and they are tiny:")
    for nm, n in TRAINED.items():
        mb = n * ADAM_BYTES_PER_PARAM / 1e6
        print(f"    {nm:<22}{n:>9,} params   {mb:>6.2f} MB of weights, "
              f"gradients and Adam state")
    least = min(TRAINED.values()) * ADAM_BYTES_PER_PARAM / 1e6
    print(f"\n  Sharding {least:.2f} MB across {a.cards} cards saves "
          f"{least / a.cards:.3f} MB a card and costs an")
    print(f"  all-gather every step. There is nothing for ZeRO to do, so")
    print(f"  `train_rl.py` runs plain python by default and says why.")
    print(f"\n  The real bottleneck is not memory at all: the rollout is")
    print(f"  SEQUENTIAL over {2992:,} trading days, because the position on")
    print(f"  day t depends on the action on day t-1. That does not shard.")

    print(f"\n  At the top of the ladder it becomes a real question:")
    for r in (xl, big):
        ratio = r["gb_act"] / r["gb_weights"]
        saved = 1 - r["gb_total_zero3"] / r["gb_total_1card"]
        print(f"    {r['name']:<5} {r['params'] / 1e6:>6.0f}M params   "
              f"wt+Adam {r['gb_weights']:>5.1f} GB -> "
              f"{r['gb_zero3']:>4.1f} GB a card")
        print(f"          activations {r['gb_act']:>5.1f} GB, paid IN FULL by "
              f"every rank ({ratio:.1f}x)")
        print(f"          total {r['gb_total_1card']:>5.1f} -> "
              f"{r['gb_total_zero3']:>5.1f} GB, a {saved:.0%} saving")
    print(f"\n  `06_protein_folding` reaches the same conclusion from a")
    print(f"  completely different direction: an Evoformer trunk is ~100k")
    print(f"  parameters and fills a card with O(N^2) pair representations,")
    print(f"  so ZeRO-1 and ZeRO-3 differ by 4.6%. Two unrelated domains,")
    print(f"  one fact -- shard the thing that is actually large, and when")
    print(f"  that is the activations the tool is gradient checkpointing.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
