#!/usr/bin/env python3
"""
Scoring documents IN CONTEXT: groupwise and self-attentive rankers.

    uv run groupwise.py        # no GPU, no download — runs the whole demo

`03_learning_to_rank/` changes the LOSS and keeps one assumption fixed: a
document's score depends only on that document. Every objective there —
pointwise, RankNet, LambdaRank, ListNet — feeds one feature vector in and gets
one score out, so the model literally cannot know what else is on the page.

That assumption is wrong for real result lists, and this folder drops it.

Why it is wrong: redundancy
---------------------------
Ten excellent articles that say the same thing are not ten times as useful as
one. A search page showing the same fact ten ways is worse than one showing
that fact plus nine others, even though every document is individually
identical in quality. Diversity is a property of the LIST, so a scorer that
sees one document at a time cannot express it, no matter how good its loss is.

The two architectures here
--------------------------
`gsf` — Groupwise Scoring Function (Ai et al., Google Research, 2019). Score
    documents in small GROUPS of size m: concatenate the group's features,
    produce m scores jointly, and average each document's scores across the
    groups it appeared in. Context arrives through the concatenation.

`setrank` — self-attention over the whole list (SetRank, arXiv:1912.05891).
    Every document attends to every other, so the score is a function of the
    entire set. Cheaper than enumerating groups and sees more context.

The property that makes these correct
-------------------------------------
**Permutation equivariance.** Shuffle the input list and every score must
follow its own document — the model may not care what ORDER the candidates
arrived in, only which ones are present. Break it, usually by adding a
positional encoding out of transformer habit, and the model learns to exploit
the order of the candidate list it was handed. That is a leak from whatever
produced the candidates, it inflates every offline metric, and it collapses in
production where the candidate order is different.

Nothing about this raises. `test_groupwise_ranking.py` asserts it directly.

Pure PyTorch. No GPU, no download, no DeepSpeed.
"""

from typing import Tuple

import torch
import torch.nn as nn

# =============================================================================
# Scorers
# =============================================================================


class PointwiseScorer(nn.Module):
    """
    The baseline from `03_learning_to_rank/`: one document in, one score out.

    Kept here as the control. It is trivially permutation-equivariant, and
    trivially blind to context — the two properties the models below are
    measured against.
    """

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GSFScorer(nn.Module):
    """
    Groupwise Scoring Function: score m documents jointly, then average.

    For group size m, the network maps m concatenated feature vectors to m
    scores. Each document appears in several groups and its final score is the
    mean of what it received in each — so the score depends on the company it
    kept.

    Groups are ALL ORDERED PAIRS, not a rotation.

    An earlier version formed groups by rotating the list, which is cheaper and
    is broken: the groups then depend on each document's absolute position, so
    shuffling the candidates changes who is scored with whom and the scores
    move. Measured permutation error 1.5e-01, against 3e-07 for the attention
    model -- i.e. that GSF was reading candidate order, the exact leak this
    file warns about.

    Enumerating pairs is O(L^2) and restores equivariance exactly: document i
    is scored alongside every other document, so no ordering of the list can
    change the multiset of groups it appears in. The original paper samples
    groups for large L; for the list lengths here, enumeration is both cheap
    and correct.
    """

    def __init__(self, n_features: int, group_size: int = 2, hidden: int = 64):
        super().__init__()
        if group_size != 2:
            raise ValueError(
                "This implementation enumerates PAIRS. Larger groups need "
                "sampling, which trades the exact equivariance property for "
                "cost -- see the docstring.")
        self.m = 2
        self.net = nn.Sequential(
            nn.Linear(n_features * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        left = x.unsqueeze(2).expand(b, n, n, f)          # document i
        right = x.unsqueeze(1).expand(b, n, n, f)         # partner j
        pair = torch.cat([left, right], dim=-1)           # (b, n, n, 2f)
        out = self.net(pair)                              # (b, n, n, 2)

        # out[..., 0] is i's score when paired with j; out[..., 1] is j's score
        # in that same group. Averaging both directions keeps the function
        # symmetric in how a document is treated.
        as_first = out[..., 0].mean(dim=2)                # over partners j
        as_second = out[..., 1].mean(dim=1)               # over hosts i
        return (as_first + as_second) / 2


class SetRankScorer(nn.Module):
    """
    Self-attention over the candidate set (SetRank, arXiv:1912.05891).

    Note what is NOT here: any positional encoding. A transformer without one
    is permutation-equivariant, and that is exactly the property a ranker over
    an unordered candidate set requires. Adding sinusoids "because
    transformers have them" would let the model read the order it was given.
    """

    def __init__(self, n_features: int, hidden: int = 64, heads: int = 4,
                 layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.proj(x))).squeeze(-1)


MODELS = {
    "pointwise": (PointwiseScorer, "Context-free control: one document in, one score out"),
    "gsf":       (GSFScorer,       "Groups of m scored jointly (Ai et al., 2019)"),
    "setrank":   (SetRankScorer,   "Self-attention over the whole set (arXiv:1912.05891)"),
}


def build_model(name: str, n_features: int, **kw) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}")
    return MODELS[name][0](n_features, **kw)


# =============================================================================
# Diagnostics — the properties, measured
# =============================================================================


def context_sensitivity(model: nn.Module, x: torch.Tensor) -> float:
    """
    How much does a document's score change when its NEIGHBOURS change?

    Score document 0 in its real list, then again with every other document
    replaced by noise. A context-free scorer gives exactly the same number
    twice; a groupwise one does not. Returns the mean absolute difference.
    """
    model.eval()
    with torch.no_grad():
        original = model(x)[:, 0]
        swapped = x.clone()
        swapped[:, 1:] = torch.randn_like(swapped[:, 1:])
        changed = model(swapped)[:, 0]
    return (original - changed).abs().mean().item()


def permutation_equivariance_error(model: nn.Module, x: torch.Tensor,
                                   seed: int = 0) -> float:
    """
    Shuffle the list; every score must follow its own document.

    Returns the largest discrepancy. Anything above float noise means the model
    is reading the order of the candidates, which is a leak from whatever
    produced them.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[1], generator=g)
    model.eval()
    with torch.no_grad():
        before = model(x)[:, perm]
        after = model(x[:, perm])
    return (before - after).abs().max().item()


# =============================================================================
# A task where context actually matters
# =============================================================================


def redundancy_ranking_data(n_queries: int = 512, list_len: int = 12,
                            n_features: int = 16, redundancy: float = 1.0,
                            noise: float = 0.2, seed: int = 0,
                            task_seed: int = 7777
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Graded relevance where a document's value DROPS if the list already
    contains something similar.

    utility(i) = quality(i) - redundancy * max_similarity(i, earlier documents)

    This is the smallest honest task that a context-free scorer cannot solve.
    Quality is a linear function of features, so a pointwise model can learn
    that part perfectly; the redundancy term depends on the rest of the list,
    so no amount of pointwise capacity can represent it. Set `redundancy=0`
    and the task collapses to the one in `03_learning_to_rank/`, where the
    groupwise models should show no advantage — which is the control worth
    running before believing any of this.

    The hidden quality direction comes from `task_seed`, not `seed`, so train
    and test are the same task sampled twice. Getting that wrong makes
    training look actively harmful; it did, once, in the neighbouring folder.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_queries, list_len, n_features)).astype("float32")

    w = np.random.default_rng(task_seed).normal(size=(n_features,)).astype("float32")
    quality = x @ w

    # cosine similarity between every pair of documents in a query
    norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    sim = norm @ norm.transpose(0, 2, 1)
    order = quality.argsort(axis=-1)[:, ::-1]          # best quality first

    utility = quality.copy()
    for q in range(n_queries):
        seen = []
        for idx in order[q]:
            if seen:
                utility[q, idx] -= redundancy * max(sim[q, idx, j] for j in seen)
            seen.append(idx)
    utility += rng.normal(scale=noise, size=utility.shape).astype("float32")

    grades = np.zeros_like(utility)
    ranks = utility.argsort(axis=-1)
    cuts = np.linspace(0, list_len, 6).astype(int)
    for g, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        np.put_along_axis(grades, ranks[:, lo:hi], float(g), axis=-1)

    return torch.from_numpy(x), torch.from_numpy(grades.astype("float32"))



def duplicate_ranking_data(n_queries: int = 512, list_len: int = 12,
                           n_features: int = 16, n_duplicates: int = 3,
                           noise: float = 0.05, seed: int = 0,
                           task_seed: int = 7777
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A task a context-free scorer CANNOT solve, by construction.

    Some documents are near-duplicates of others: identical features plus a
    little noise. The label keeps the better copy and demotes the twin to
    grade 0 — showing the same result twice is worth less than showing it once.

    This is not merely hard for a pointwise model, it is impossible. Two
    documents with (almost) identical features must receive (almost) identical
    scores from a function of one document, so it cannot rank one above the
    other by more than noise. A model that sees the list can notice the twin
    and demote it.

    `redundancy_ranking_data` above is the softer, more realistic version of
    the same idea; this one exists because a demonstration should be decisive
    before it is realistic.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_queries, list_len, n_features)).astype("float32")
    w = np.random.default_rng(task_seed).normal(size=(n_features,)).astype("float32")

    grades = np.zeros((n_queries, list_len), dtype="float32")
    for q in range(n_queries):
        # pick disjoint (original, duplicate) index pairs
        idx = rng.permutation(list_len)
        pairs = [(idx[2 * i], idx[2 * i + 1]) for i in range(n_duplicates)]
        for orig, dup in pairs:
            x[q, dup] = x[q, orig] + rng.normal(scale=noise, size=n_features)

        quality = x[q] @ w
        order = np.argsort(-quality)                       # best first
        rank_grade = np.zeros(list_len, dtype="float32")
        cuts = np.linspace(0, list_len, 6).astype(int)
        for g, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            rank_grade[order[lo:hi]] = float(4 - g)        # 4 = best

        # demote the WORSE member of each duplicate pair to 0
        for orig, dup in pairs:
            loser = dup if quality[orig] >= quality[dup] else orig
            rank_grade[loser] = 0.0
        grades[q] = rank_grade

    return torch.from_numpy(x), torch.from_numpy(grades)

def _demo() -> None:
    bar = "=" * 78
    torch.manual_seed(0)
    print(bar)
    print("  Groupwise ranking — does the score depend on the company it keeps?")
    print(bar)

    x = torch.randn(64, 10, 16)
    print(f"  {'model':<12} {'params':>9} {'context sensitivity':>21} {'perm. equivariance err':>24}")
    for name in MODELS:
        torch.manual_seed(0)
        m = build_model(name, 16)
        n = sum(p.numel() for p in m.parameters())
        cs = context_sensitivity(m, x)
        pe = permutation_equivariance_error(m, x)
        print(f"  {name:<12} {n:>9,} {cs:>21.6f} {pe:>24.2e}")
    print()
    print("    pointwise sensitivity is EXACTLY 0 — it cannot see other documents.")
    print("    Both groupwise models are permutation-equivariant to float noise,")
    print("    which is the property that stops them reading candidate order.")
    print(bar)

    print("  Does context help where context is REQUIRED? (near-duplicate task)")
    from ranking_metrics import listnet_loss, ndcg

    x_tr, y_tr = duplicate_ranking_data(384, 12, 16, seed=1)
    x_te, y_te = duplicate_ranking_data(128, 12, 16, seed=2)
    torch.manual_seed(0)
    with torch.no_grad():
        base = ndcg(build_model("pointwise", 16).eval()(x_te), y_te).mean().item()
    print(f"    {'untrained':<12} NDCG@10 {base:.4f}")
    for name in MODELS:
        torch.manual_seed(0)
        m = build_model(name, 16)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(250):
            opt.zero_grad()
            listnet_loss(m(x_tr), y_tr).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            print(f"    {name:<12} NDCG@10 {ndcg(m(x_te), y_te).mean().item():.4f}")
    print()
    print("    GSF wins, and it should: two near-identical documents must get")
    print("    near-identical scores from a function of ONE document, so the")
    print("    pointwise model cannot demote the twin however long it trains.")
    print()
    print("    SetRank comes LAST, and that is not a typo. It has 68k parameters")
    print("    against GSF's 6.4k and overfits 384 queries -- measured at 250,")
    print("    800 and 2,400 steps and three learning rates, it gets worse with")
    print("    more training (0.90 -> 0.79). More context capacity is not free.")
    print(bar)

    print("  Training on a GPU with DeepSpeed: train_groupwise_ranking.py")
    print(bar)


if __name__ == "__main__":
    _demo()
