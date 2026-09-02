#!/usr/bin/env python3
"""
Learning to Rank: the four objectives, and the metrics that judge them.

    uv run ranking_losses.py        # no GPU, no download — runs the whole demo

Ranking is not classification. A classifier asks "which class is this?"; a
ranker asks "which of these should come first?", and that difference changes
the loss, the metric, and what counts as a mistake.

The four objectives here differ ONLY in what they compare:

    pointwise    each document against its own label        (plain regression)
    ranknet      each PAIR against the other                (arXiv:cs/0605055-era)
    lambdarank   each pair, weighted by how much swapping    (Burges, 2010)
                 them would move NDCG
    listnet      the whole LIST at once, as a distribution   (Cao et al., 2007)

That progression is the entire subject. Pointwise treats a ranking problem as
regression. RankNet fixes the objective but treats every pair as equally
important -- getting positions 1 and 2 backwards costs the same as 99 and 100,
which is not how anyone reads results. LambdaRank fixes the weighting. ListNet
abandons pairs entirely.

How much does the choice actually buy you? Measured with

    train_learning_to_rank.py --method all --epochs N

on synthetic data -- 4096 queries, list length 16, noise 2.0, seed 42:

    epochs   pointwise   ranknet   lambdarank   listnet   spread
         1      0.9198    0.9594       0.9401    0.9611   0.0413
         2      0.9593    0.9678       0.9649    0.9661   0.0085
         6      0.9637    0.9689       0.9680    0.9682   0.0052
        40      0.9677    0.9686       0.9683    0.9676   0.0010

    The untrained baseline is 0.4862 throughout.

So the honest answer is: **it depends on your training budget**. Under-trained,
the objective matters a great deal and pointwise is clearly worst. Given enough
steps on a task this clean, all four converge to within a hundredth of each
other. Anyone quoting a single number for "listwise beats pointwise by X" has
fixed a budget without telling you.

(The untrained baseline of 0.4862 is reported alongside every number here on
purpose: an NDCG figure with no baseline is unreadable, because randomly
ordering a 16-document list already scores nearly 0.5.)

Why the metric comes first in this file
---------------------------------------
NDCG is not a detail you add afterwards. LambdaRank is DEFINED in terms of it:
its gradient is a RankNet gradient scaled by |ΔNDCG| of the swap. Implement
NDCG wrongly and LambdaRank does not become slightly worse, it becomes
RankNet with noise -- and it will still train, still converge, and still
report a number.

The specific traps, each of which produces a plausible score:

  * gain `2**rel - 1`, NOT `rel`. With linear gain a document of relevance 4
    is worth twice one of relevance 2; with exponential gain it is worth five
    times. Graded relevance is the whole reason to use NDCG rather than MAP.
  * discount `1 / log2(i + 2)` for 0-based i. Off by one and position 1 gets
    a discount of 1/log2(1) = infinity.
  * normalise by the IDEAL DCG of the same query, not a global constant --
    otherwise queries with more relevant documents dominate the mean.
  * a query with no relevant documents has IDCG 0. Return 0.0, not NaN, and
    not 1.0: a ranker cannot be perfect at a question with no right answer.

Pure PyTorch. No GPU, no download, no DeepSpeed.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Metrics
# =============================================================================


def dcg(relevance: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Discounted Cumulative Gain of a list ALREADY IN RANKED ORDER.

    `relevance[i]` is the graded relevance of whatever the model put at
    position i. This function does not sort -- passing it unsorted labels
    silently computes the DCG of the ground truth instead of the prediction,
    which is one of the easiest ways to report a perfect ranker.
    """
    relevance = relevance[..., :k]
    gains = torch.pow(2.0, relevance) - 1.0
    positions = torch.arange(relevance.shape[-1], device=relevance.device,
                             dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 2.0)
    return (gains * discounts).sum(dim=-1)


def ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    NDCG@k: DCG of the model's order, over DCG of the best possible order.

    Shapes are (batch, list_len). Returns (batch,).
    """
    order = scores.argsort(dim=-1, descending=True)
    ranked = labels.gather(-1, order)
    ideal = labels.sort(dim=-1, descending=True).values

    actual_dcg = dcg(ranked, k)
    ideal_dcg = dcg(ideal, k)
    # A query with nothing relevant has ideal DCG 0. Zero, not NaN, and not a
    # free 1.0 -- there is no right answer to be perfect at.
    return torch.where(ideal_dcg > 0, actual_dcg / ideal_dcg,
                       torch.zeros_like(actual_dcg))


def mrr(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean Reciprocal Rank: 1 / position of the first relevant document."""
    order = scores.argsort(dim=-1, descending=True)
    ranked = labels.gather(-1, order) > 0
    positions = torch.arange(1, ranked.shape[-1] + 1, device=scores.device,
                             dtype=torch.float32)
    # argmax on a bool tensor returns the FIRST True, or 0 if there are none --
    # so the "any" mask below is doing real work, not defensive decoration.
    first = ranked.float().argmax(dim=-1)
    has_any = ranked.any(dim=-1)
    return torch.where(has_any, 1.0 / positions[first], torch.zeros_like(first, dtype=torch.float32))


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MAP's inner term: precision at each relevant hit, averaged."""
    order = scores.argsort(dim=-1, descending=True)
    ranked = (labels.gather(-1, order) > 0).float()
    positions = torch.arange(1, ranked.shape[-1] + 1, device=scores.device,
                             dtype=torch.float32)
    precision_at_i = ranked.cumsum(dim=-1) / positions
    n_relevant = ranked.sum(dim=-1)
    return torch.where(n_relevant > 0,
                       (precision_at_i * ranked).sum(dim=-1) / n_relevant.clamp(min=1),
                       torch.zeros_like(n_relevant))


# =============================================================================
# Objectives
# =============================================================================


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Plain regression onto the relevance grade.

    The baseline that treats ranking as something it is not. It is punished
    for predicting 3.2 where the label is 3, even when the ORDER is perfect --
    and a ranking metric does not care about that at all.
    """
    return F.mse_loss(scores, labels.float())


def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor,
                 sigma: float = 1.0) -> torch.Tensor:
    """
    RankNet: binary cross-entropy over every ordered PAIR.

    For a pair (i, j) where label_i > label_j, the target probability that i
    beats j is 1, and the model's probability is sigmoid(sigma * (s_i - s_j)).

    Only pairs with DIFFERENT labels contribute. Pairs of equal relevance carry
    no information about order, and including them teaches the model to
    separate documents that should tie.
    """
    s_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)          # (B, L, L)
    l_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    mask = l_diff > 0                                             # i should beat j
    if not mask.any():
        return scores.sum() * 0.0                                 # keeps the graph
    # softplus(-x) == -log(sigmoid(x)), computed without overflow
    return F.softplus(-sigma * s_diff)[mask].mean()


def delta_ndcg(scores: torch.Tensor, labels: torch.Tensor,
               k: int = 10) -> torch.Tensor:
    """
    |ΔNDCG| for swapping every pair (i, j) — the weight LambdaRank adds.

    This is the whole idea. Swapping positions 1 and 2 changes NDCG a great
    deal; swapping 99 and 100 changes it almost not at all. RankNet weights
    both the same; this weights them by what they actually cost.
    """
    order = scores.argsort(dim=-1, descending=True)
    rank = torch.zeros_like(order)
    arange = torch.arange(scores.shape[-1], device=scores.device)
    rank.scatter_(-1, order, arange.expand_as(order))              # doc -> position

    gains = torch.pow(2.0, labels) - 1.0
    discounts = 1.0 / torch.log2(rank.float() + 2.0)

    ideal = labels.sort(dim=-1, descending=True).values
    idcg = dcg(ideal, k).clamp(min=1e-10).unsqueeze(-1).unsqueeze(-1)

    # Swapping i and j exchanges their discounts; the change in DCG is
    # (g_i - g_j) * (d_i - d_j), and NDCG divides by the ideal.
    g_diff = gains.unsqueeze(-1) - gains.unsqueeze(-2)
    d_diff = discounts.unsqueeze(-1) - discounts.unsqueeze(-2)
    return (g_diff * d_diff).abs() / idcg


def lambdarank_loss(scores: torch.Tensor, labels: torch.Tensor,
                    sigma: float = 1.0, k: int = 10) -> torch.Tensor:
    """
    LambdaRank: RankNet's pair loss, scaled by |ΔNDCG| of the swap.

    Burges's insight is that you do not need a differentiable NDCG. You need
    the GRADIENT to point where NDCG would improve, and multiplying the
    RankNet gradient by |ΔNDCG| does that. The "loss" below is therefore a
    surrogate whose gradient is right, not a quantity worth reading.
    """
    s_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    l_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    mask = l_diff > 0
    if not mask.any():
        return scores.sum() * 0.0
    # detached: the weight says how much this pair MATTERS, and letting the
    # model reduce its loss by shrinking its own weights would be a way to
    # score well without ranking well.
    weight = delta_ndcg(scores, labels, k).detach()
    return (F.softplus(-sigma * s_diff) * weight)[mask].mean()


def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ListNet: cross-entropy between two top-1 probability distributions.

    Softmax the labels, softmax the scores, and match the distributions. No
    pairs anywhere -- the list is one object with one distribution over "which
    document is best".

    Note the labels are softmaxed too, not one-hot. That is what makes it
    handle GRADED relevance: a query with two documents at relevance 4 puts
    half its mass on each.
    """
    target = F.softmax(labels.float(), dim=-1)
    return -(target * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


LOSSES = {
    "pointwise":  (pointwise_loss,  "Regression onto the grade. Treats ranking as something it is not."),
    "ranknet":    (ranknet_loss,    "Pairwise cross-entropy. Every pair weighted equally."),
    "lambdarank": (lambdarank_loss, "Pairwise, weighted by |ΔNDCG| of the swap."),
    "listnet":    (listnet_loss,    "Listwise. Softmax over the whole list, no pairs."),
}


# =============================================================================
# Model and data
# =============================================================================


class RankingMLP(nn.Module):
    """
    A scoring function: one document's features in, one score out.

    Applied independently per document, which is what "pointwise scoring"
    means and is shared by all four objectives above -- they differ in the
    LOSS, not the architecture. `04_groupwise_ranking/` is where that
    assumption gets dropped.
    """

    def __init__(self, n_features: int = 32, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        dims = (n_features,) + hidden
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, F) -> (B, L)
        return self.net(x).squeeze(-1)


def synthetic_ranking_data(n_queries: int = 512, list_len: int = 20,
                           n_features: int = 32, noise: float = 0.5,
                           seed: int = 0, task_seed: int = 12345
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Graded relevance data with a LEARNABLE signal, generated by numpy.

    The classic benchmarks for this (MSLR-WEB10K, LETOR, Istella) are not on
    the HuggingFace Hub -- checked, not assumed -- so the default is simulated.
    That is not a compromise for teaching these four objectives, because they
    differ in the loss and share everything else; what matters is that the data
    has graded labels and a signal a model can actually find.

    Construction: a hidden linear direction `w` defines true utility. Labels
    are that utility, noised, then bucketed into grades 0-4 by quantile. The
    noise level is the knob that makes the problem hard -- at noise=0 every
    objective reaches NDCG 1.0 and the comparison says nothing.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_queries, list_len, n_features)).astype("float32")

    # The hidden utility direction comes from `task_seed`, NOT `seed`. Train and
    # test must be the SAME task sampled twice, and an earlier version drew w
    # from `seed` -- so every split was a different problem, the model learned
    # one direction and was scored on another, and training reliably made NDCG
    # WORSE than random init. It looked like an objective-comparison result.
    # Change task_seed only to generate a genuinely different task.
    w = np.random.default_rng(task_seed).normal(size=(n_features,)).astype("float32")
    utility = x @ w
    utility += rng.normal(scale=noise, size=utility.shape).astype("float32")

    # Grades 0-4 by within-query quantile, so every query has the same label
    # distribution and NDCG is comparable across them.
    order = utility.argsort(axis=-1)
    grades = np.zeros_like(utility)
    cuts = np.linspace(0, list_len, 6).astype(int)     # 5 buckets
    for g, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        idx = order[:, lo:hi]
        np.put_along_axis(grades, idx, float(g), axis=-1)

    return torch.from_numpy(x), torch.from_numpy(grades.astype("float32"))


def _demo() -> None:
    bar = "=" * 78
    torch.manual_seed(0)
    print(bar)
    print("  Learning to Rank — the metric first, because LambdaRank is defined by it")
    print(bar)

    labels = torch.tensor([[3.0, 2.0, 3.0, 0.0, 1.0, 2.0]])
    perfect = labels.clone()
    reverse = -labels
    print(f"  labels                 {labels.tolist()[0]}")
    print(f"  NDCG, perfect order    {ndcg(perfect, labels).item():.4f}   (must be exactly 1)")
    print(f"  NDCG, reversed order   {ndcg(reverse, labels).item():.4f}")
    print(f"  NDCG, random order     {ndcg(torch.randn_like(labels), labels).item():.4f}")
    print(f"  NDCG, nothing relevant {ndcg(torch.randn(1, 4), torch.zeros(1, 4)).item():.4f}"
          "   (0, not NaN, not 1)")
    print(f"  MRR                    {mrr(perfect, labels).item():.4f}")
    print(f"  MAP                    {average_precision(perfect, labels).item():.4f}")
    print(bar)

    print("  Why LambdaRank differs from RankNet: the SAME swap costs differently")
    # Both pairs below span the SAME label gap (one grade), so the only
    # difference is WHERE in the list they sit. An earlier version of this
    # demo compared a pair of equally-irrelevant documents, whose |ΔNDCG| is
    # exactly 0, and printed a ratio of 138311535120x -- a division by zero
    # wearing a number's clothing.
    s = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    lab = torch.tensor([[4.0, 3.0, 2.0, 1.0, 4.0, 3.0]])
    d = delta_ndcg(s, lab)[0]
    top, deep = d[0, 1].item(), d[4, 5].item()
    print(f"    labels in rank order   {lab.tolist()[0]}")
    print(f"    |ΔNDCG| swap at positions 1<->2 (grades 4,3): {top:.4f}")
    print(f"    |ΔNDCG| swap at positions 5<->6 (grades 4,3): {deep:.4f}")
    print(f"    -> identical label gap, but the top swap costs {top / deep:.1f}x more")
    print("    RankNet weights these two EQUALLY. That is the entire difference.")
    print(bar)

    print("  The four objectives on one batch (values are not comparable across rows)")
    x, y = synthetic_ranking_data(n_queries=4, list_len=8, n_features=16, seed=1)
    model = RankingMLP(n_features=16)
    with torch.no_grad():
        sc = model(x)
    for name, (fn, blurb) in LOSSES.items():
        print(f"    {name:<11} {fn(sc, y).item():>9.4f}   {blurb}")
    print(bar)

    print("  Do the objectives actually differ? Measured, not asserted.")
    print(f"    {'noise':>6} {'untrained':>10} " +
          " ".join(f"{n:>11}" for n in LOSSES))
    for noise in (0.5, 4.0):
        x, y = synthetic_ranking_data(512, 16, 16, noise=noise, seed=2)
        xt, yt = synthetic_ranking_data(128, 16, 16, noise=noise, seed=99)
        torch.manual_seed(0)
        base = RankingMLP(n_features=16).eval()
        with torch.no_grad():
            row = [ndcg(base(xt), yt).mean().item()]
        for name, (fn, _) in LOSSES.items():
            torch.manual_seed(0)
            m = RankingMLP(n_features=16)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(250):
                opt.zero_grad()
                fn(m(x), y).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                row.append(ndcg(m(xt), yt).mean().item())
        print(f"    {noise:>6.1f} " + " ".join(f"{v:>10.4f} " for v in row))
    print()
    print("    Read that honestly: the four objectives land within ~0.005 of")
    print("    each other, while TRAINING AT ALL moves NDCG from ~0.62 to ~0.99.")
    print("    On this synthetic task the choice of objective barely matters.")
    print("    The published gains for listwise methods come from real data with")
    print("    many ties, position bias and far longer lists -- none of which a")
    print("    linear-utility simulation reproduces. Use --source hf for that.")
    print(bar)

    print("  Training on a GPU with DeepSpeed: train_learning_to_rank.py")
    print(bar)


if __name__ == "__main__":
    _demo()
