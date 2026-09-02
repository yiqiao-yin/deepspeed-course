# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: learning-to-rank objectives and metrics.

Run:
    uv run tests/test_ranking_losses.py

Why this suite exists
---------------------
Every mistake available in ranking code produces a plausible number and a model
that trains. There is no crash to catch:

  * linear gain instead of `2**rel - 1` — NDCG still lands in [0, 1] and still
    rises during training, it just stops distinguishing grade 4 from grade 2
  * an off-by-one discount — position 1 gets `1/log2(1) = inf`
  * normalising by a global constant rather than the query's own ideal DCG —
    queries with more relevant documents dominate the average
  * a query with nothing relevant returning 1.0 — the model looks perfect on
    exactly the questions it cannot answer
  * LambdaRank whose ΔNDCG weight does not depend on POSITION — it silently
    degenerates into RankNet, which is a real algorithm, so nothing looks wrong
  * a pairwise loss that includes equal-label pairs — the model is taught to
    separate documents that should tie

So these assert PROPERTIES, and where a property has a counterexample the
counterexample is asserted too: it is not enough that top swaps weigh more than
deep ones, the test pins the ratio at a specific list.

CPU only. No GPU, no download.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "02_intermediate/03_learning_to_rank"))

from ranking_losses import (  # noqa: E402
    LOSSES, RankingMLP, average_precision, dcg, delta_ndcg, lambdarank_loss,
    listnet_loss, mrr, ndcg, pointwise_loss, ranknet_loss,
    synthetic_ranking_data)


def test_ndcg_basics(r: Results) -> None:
    labels = torch.tensor([[3.0, 2.0, 3.0, 0.0, 1.0, 2.0]])

    r.check(abs(ndcg(labels, labels).item() - 1.0) < 1e-6,
            "a perfect ordering scores exactly NDCG 1.0")

    worst = ndcg(-labels, labels).item()
    r.check(0.0 < worst < 1.0,
            f"the worst ordering scores strictly between 0 and 1 ({worst:.4f})",
            "NDCG cannot reach 0 while any relevant document is in the list")

    # Scale and shift invariance: NDCG reads the ORDER, nothing else.
    s = torch.tensor([[5.0, 1.0, 4.0, -2.0, 0.0, 2.0]])
    base = ndcg(s, labels).item()
    r.check(abs(ndcg(s * 100, labels).item() - base) < 1e-6,
            "NDCG is invariant to score scale")
    r.check(abs(ndcg(s + 7, labels).item() - base) < 1e-6,
            "NDCG is invariant to a score shift")

    r.check(ndcg(torch.randn(1, 5), torch.zeros(1, 5)).item() == 0.0,
            "a query with NO relevant documents scores 0.0",
            "returning 1.0 makes a model look perfect on the questions it "
            "cannot answer; NaN poisons the mean")


def test_ndcg_uses_exponential_gain(r: Results) -> None:
    """The property that separates NDCG from a rank correlation."""
    # One document of grade 4, versus two of grade 2 placed just below it.
    # With exponential gain 2**4-1 = 15 dominates 2*(2**2-1) = 6.
    labels = torch.tensor([[4.0, 2.0, 2.0]])
    top_first = ndcg(torch.tensor([[3.0, 2.0, 1.0]]), labels).item()
    top_last = ndcg(torch.tensor([[1.0, 3.0, 2.0]]), labels).item()
    r.check(top_first > top_last,
            "burying the grade-4 document costs more than burying a grade-2")

    g = dcg(torch.tensor([[4.0]])).item()
    r.check(abs(g - 15.0) < 1e-6,
            f"gain is 2**rel - 1, so a single grade-4 gives DCG 15 (got {g:.2f})",
            "linear gain would give 4 and would make grade 4 worth twice "
            "grade 2 rather than five times")

    d = dcg(torch.tensor([[1.0, 1.0]])).item()
    r.check(abs(d - (1.0 + 1.0 / torch.log2(torch.tensor(3.0)).item())) < 1e-6,
            "the discount is 1/log2(i+2), so position 1 is undiscounted",
            "an off-by-one makes position 1 infinite")


def test_mrr_and_map(r: Results) -> None:
    labels = torch.tensor([[0.0, 0.0, 3.0, 1.0]])
    # first relevant at position 3 -> 1/3
    s = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    r.check(abs(mrr(s, labels).item() - 1 / 3) < 1e-6,
            "MRR is 1/position of the first relevant document")
    r.check(mrr(torch.randn(1, 4), torch.zeros(1, 4)).item() == 0.0,
            "MRR is 0 when nothing is relevant, not 1")

    perfect = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    r.check(abs(average_precision(perfect, torch.tensor([[1.0, 1.0, 0.0, 0.0]])).item() - 1.0) < 1e-6,
            "MAP is 1.0 when every relevant document is at the top")
    r.check(average_precision(torch.randn(1, 4), torch.zeros(1, 4)).item() == 0.0,
            "MAP is 0 when nothing is relevant")


def test_delta_ndcg_depends_on_position(r: Results) -> None:
    """
    LambdaRank's whole contribution. If this fails it IS RankNet.
    """
    # Two pairs with an identical label gap (4 vs 3), one at the top of the
    # list and one at the bottom. Only the position differs.
    scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    labels = torch.tensor([[4.0, 3.0, 2.0, 1.0, 4.0, 3.0]])
    d = delta_ndcg(scores, labels)[0]
    top, deep = d[0, 1].item(), d[4, 5].item()

    r.check(top > deep * 3,
            f"a swap at the top weighs far more than the same swap deep down "
            f"({top:.4f} vs {deep:.4f}, {top / max(deep, 1e-12):.1f}x)",
            "if these are equal, LambdaRank has silently become RankNet -- "
            "which still trains and still reports a number")

    r.check(torch.allclose(d, d.transpose(-1, -2), atol=1e-6),
            "the swap cost is symmetric in (i, j)")

    same = delta_ndcg(scores, torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]))[0]
    r.check(float(same.abs().max()) < 1e-6,
            "swapping two equally-relevant documents costs exactly 0")


def test_losses_prefer_correct_order(r: Results) -> None:
    """Every objective must score a correct ordering below a wrong one."""
    labels = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    good = torch.tensor([[4.0, 3.0, 2.0, 1.0]])       # correct order
    bad = torch.tensor([[1.0, 2.0, 3.0, 4.0]])        # reversed

    for name, (fn, _) in LOSSES.items():
        g, b = fn(good, labels).item(), fn(bad, labels).item()
        r.check(g < b,
                f"{name}: a correct ordering has lower loss than a reversed one "
                f"({g:.4f} < {b:.4f})")

    # Pairwise losses must IGNORE equal-label pairs. With all labels equal
    # there is no ordering information, so the loss must be exactly 0.
    flat = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    for name, fn in (("ranknet", ranknet_loss), ("lambdarank", lambdarank_loss)):
        v = fn(torch.randn(1, 4), flat).item()
        r.check(abs(v) < 1e-6,
                f"{name}: all-equal labels give exactly 0 loss ({v:.6f})",
                "including equal pairs teaches the model to separate documents "
                "that should tie")


def test_pairwise_and_listwise_are_scale_sensitive(r: Results) -> None:
    """
    A property that distinguishes the objectives from the metric.

    NDCG is scale-invariant; the LOSSES are not, and must not be -- a model
    that is confidently right should be rewarded over one that is barely right.
    """
    labels = torch.tensor([[3.0, 0.0]])
    weak = torch.tensor([[0.1, -0.1]])
    strong = torch.tensor([[5.0, -5.0]])
    for name, fn in (("ranknet", ranknet_loss), ("listnet", listnet_loss)):
        r.check(fn(strong, labels).item() < fn(weak, labels).item(),
                f"{name}: confident-and-correct beats barely-correct")

    r.check(abs(ndcg(weak, labels).item() - ndcg(strong, labels).item()) < 1e-9,
            "...while NDCG cannot tell them apart, since it only reads order")


def test_listnet_is_a_distribution_match(r: Results) -> None:
    labels = torch.tensor([[3.0, 1.0, 0.0]])
    # Scores equal to labels give the minimum for ListNet: identical softmaxes.
    exact = listnet_loss(labels.clone(), labels).item()
    other = listnet_loss(torch.tensor([[1.0, 3.0, 0.0]]), labels).item()
    r.check(exact < other,
            f"ListNet is minimised when the score distribution matches the "
            f"label distribution ({exact:.4f} < {other:.4f})")

    # Graded relevance: two equally-relevant documents should split the mass,
    # which is what softmaxing the LABELS (rather than one-hot) buys.
    tie = torch.tensor([[4.0, 4.0, 0.0]])
    balanced = listnet_loss(torch.tensor([[2.0, 2.0, 0.0]]), tie).item()
    lopsided = listnet_loss(torch.tensor([[6.0, 0.0, 0.0]]), tie).item()
    r.check(balanced < lopsided,
            "ListNet prefers splitting probability between tied documents",
            "one-hot targets would make it pick an arbitrary winner")


def test_gradients_reach_the_model(r: Results) -> None:
    """Cheapest possible proof that each objective can actually train."""
    torch.manual_seed(0)
    x, y = synthetic_ranking_data(16, 8, 12, noise=0.3, seed=5)

    for name, (fn, _) in LOSSES.items():
        torch.manual_seed(0)
        model = RankingMLP(n_features=12)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        first = ndcg(model(x), y).mean().item()
        for _ in range(60):
            opt.zero_grad()
            fn(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            last = ndcg(model(x), y).mean().item()
        r.check(last > first,
                f"{name}: 60 steps improve NDCG on the training set "
                f"({first:.4f} -> {last:.4f})",
                "an objective that cannot overfit 16 queries has a broken "
                "gradient path and will still train and still look fine")


def test_synthetic_data_is_learnable_and_shared(r: Results) -> None:
    """
    The generator bug that cost a full debugging cycle.

    The hidden utility direction must come from `task_seed`, not `seed`. When
    it came from `seed`, train and test were DIFFERENT tasks: the model learned
    one direction, was scored on another, and training reliably made NDCG worse
    than random init -- which read exactly like a result about the objectives.
    """
    x_tr, y_tr = synthetic_ranking_data(64, 10, 8, noise=0.0, seed=1)
    x_te, y_te = synthetic_ranking_data(64, 10, 8, noise=0.0, seed=2)

    torch.manual_seed(0)
    model = RankingMLP(n_features=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        ranknet_loss(model(x_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_ndcg = ndcg(model(x_tr), y_tr).mean().item()
        test_ndcg = ndcg(model(x_te), y_te).mean().item()

    r.check(test_ndcg > 0.8,
            f"a model trained on one split generalises to another "
            f"({test_ndcg:.4f})",
            "if this collapses toward random, the two splits are different "
            "tasks and every comparison built on them is meaningless")
    r.check(abs(train_ndcg - test_ndcg) < 0.15,
            f"train and test agree ({train_ndcg:.4f} vs {test_ndcg:.4f})")

    # Labels must be graded, not binary -- NDCG's gain needs something to bite.
    grades = set(y_tr.unique().tolist())
    r.check(len(grades) >= 4,
            f"labels are graded, not binary ({sorted(grades)})")


def main() -> int:
    r = Results("Learning to rank — objectives, metrics and the data generator")
    test_ndcg_basics(r)
    test_ndcg_uses_exponential_gain(r)
    test_mrr_and_map(r)
    test_delta_ndcg_depends_on_position(r)
    test_losses_prefer_correct_order(r)
    test_pairwise_and_listwise_are_scale_sensitive(r)
    test_listnet_is_a_distribution_match(r)
    test_gradients_reach_the_model(r)
    test_synthetic_data_is_learnable_and_shared(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
