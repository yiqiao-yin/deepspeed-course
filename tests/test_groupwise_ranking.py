# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: groupwise ranking architectures.

Run:
    uv run tests/test_groupwise_ranking.py

Why this suite exists
---------------------
A groupwise ranker fails in ways that leave the loss curve looking healthy, so
there is nothing to catch by running it:

  * the group is formed by ROTATING the list, so a document's score depends on
    where it sits in the array. During training the array is usually in label
    order, so the model learns to read the answer key and reports a superb
    NDCG. This is not hypothetical — the first GSF written for this folder did
    exactly that, with a permutation error of 1.5e-01.
  * the "groupwise" model ignores its context input entirely and collapses to a
    pointwise scorer. It still trains, still scores well, and every claim made
    about it is then false.
  * the synthetic task draws a fresh hidden utility direction per call, so
    train and test are unrelated problems and training makes the metric worse.
    This course has already shipped that bug once, in the sibling folder.

So the two properties are asserted directly, with counterexamples where a
one-sided assertion would be trivially satisfiable:

    equivariance   shuffling candidates must permute scores identically —
                   AND a deliberately position-dependent model must FAIL the
                   same check, or the check proves nothing.
    context        groupwise sensitivity must be > 0 — AND pointwise must be
                   EXACTLY 0, which is the control.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "..", "02_intermediate", "04_groupwise_ranking")
sys.path.insert(0, os.path.abspath(FOLDER))

from groupwise import (MODELS, GSFScorer, PointwiseScorer, SetRankScorer,  # noqa: E402
                       build_model, context_sensitivity,
                       duplicate_ranking_data, permutation_equivariance_error,
                       redundancy_ranking_data)
from ranking_metrics import dcg, listnet_loss, ndcg  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class PositionLeaker(torch.nn.Module):
    """
    A scorer that reads candidate POSITION. Exists so the equivariance check
    has something it must reject: a test that only ever sees equivariant models
    would pass if the checker returned 0.0 unconditionally.
    """

    def __init__(self, n_features: int = 16):
        super().__init__()
        self.lin = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        pos = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
        return self.lin(x).squeeze(-1) + pos[None, :]


def main() -> None:
    torch.manual_seed(0)
    bar = "=" * 74
    print(bar)
    print("  test_groupwise_ranking.py")
    print(bar)

    x = torch.randn(8, 12, 16)

    # ---- permutation equivariance -------------------------------------------
    print("\n  -- permutation equivariance --")
    for name in MODELS:
        m = build_model(name, 16).eval()
        err = permutation_equivariance_error(m, x)
        check(f"{name} is equivariant (err={err:.2e})", err < 1e-4,
              f"error {err:.3e}: this model reads candidate order")

    leak_err = permutation_equivariance_error(PositionLeaker(16).eval(), x)
    check(f"the checker REJECTS a position-dependent model (err={leak_err:.2e})",
          leak_err > 1e-2,
          "a model that adds its own index passed the equivariance check, "
          "so the check is not testing anything")

    # explicit: equivariance means permuting the input permutes the output,
    # not merely that the score SET is unchanged.
    m = build_model("setrank", 16).eval()
    perm = torch.randperm(12)
    with torch.no_grad():
        a = m(x)[:, perm]
        b = m(x[:, perm])
    check("scores permute WITH the candidates, not just as a set",
          torch.allclose(a, b, atol=1e-4),
          f"max deviation {(a - b).abs().max().item():.3e}")

    # ---- context sensitivity ------------------------------------------------
    print("\n  -- context sensitivity --")
    ctx_point = context_sensitivity(build_model("pointwise", 16).eval(), x)
    check(f"pointwise sensitivity is EXACTLY 0 (got {ctx_point:.2e})",
          ctx_point == 0.0,
          "a function of one document changed its score when other documents "
          "changed, which is impossible unless the model is not pointwise")
    for name in ("gsf", "setrank"):
        ctx = context_sensitivity(build_model(name, 16).eval(), x)
        check(f"{name} sensitivity > 0 (got {ctx:.4f})", ctx > 1e-3,
              "this 'groupwise' model ignores its context and has collapsed "
              "to a pointwise scorer")

    # ---- the models are actually different ----------------------------------
    print("\n  -- architectures --")
    check("PointwiseScorer / GSFScorer / SetRankScorer are distinct classes",
          len({PointwiseScorer, GSFScorer, SetRankScorer}) == 3)
    sizes = {n: sum(p.numel() for p in build_model(n, 16).parameters())
             for n in MODELS}
    check(f"setrank has more parameters than gsf ({sizes['setrank']:,} > "
          f"{sizes['gsf']:,})", sizes["setrank"] > sizes["gsf"])
    for name in MODELS:
        out = build_model(name, 16)(x)
        check(f"{name} returns one score per document {tuple(out.shape)}",
              out.shape == (8, 12), f"got {tuple(out.shape)}")

    # SetRank must NOT carry a positional encoding: a candidate set has no
    # meaningful order, and encoding one is the leak above wearing a hat.
    setrank_params = dict(build_model("setrank", 16).named_parameters())
    check("setrank has no positional-encoding parameter",
          not any("pos" in k for k in setrank_params),
          f"found {[k for k in setrank_params if 'pos' in k]}")

    # ---- gradients reach every model ----------------------------------------
    print("\n  -- gradients --")
    y = torch.randint(0, 5, (8, 12)).float()
    for name in MODELS:
        m = build_model(name, 16)
        listnet_loss(m(x), y).backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        nonzero = sum(1 for g in grads if g.abs().sum() > 0)
        check(f"{name}: {nonzero}/{len(grads)} tensors got a nonzero gradient",
              grads and nonzero == len(grads),
              "a parameter is disconnected from the loss")

    # ---- the tasks --------------------------------------------------------
    print("\n  -- synthetic tasks --")
    for fn, label in ((duplicate_ranking_data, "duplicate"),
                      (redundancy_ranking_data, "redundancy")):
        # train and test must be the SAME task. Drawing a new hidden direction
        # per call is the bug that made training reduce NDCG in folder 03.
        xa, ya = fn(64, 12, 16, seed=1)
        xb, yb = fn(64, 12, 16, seed=2)
        check(f"{label}: different seeds give different data",
              not torch.equal(xa, xb))
        w = torch.linalg.lstsq(xa.reshape(-1, 16), ya.reshape(-1, 1)).solution
        agree = ndcg((xb.reshape(-1, 16) @ w).reshape(64, 12), yb).mean().item()
        check(f"{label}: a fit on seed 1 transfers to seed 2 "
              f"(NDCG {agree:.3f} > 0.6)", agree > 0.6,
              "train and test are unrelated tasks — check task_seed is fixed")

    xd, yd = duplicate_ranking_data(128, 12, 16, n_duplicates=3, seed=0)
    check("duplicate: labels are grades in [0, 4]",
          float(yd.min()) == 0.0 and float(yd.max()) == 4.0,
          f"range [{yd.min()}, {yd.max()}]")
    # the planted twins must really be near-identical in feature space, or the
    # task is not the one the folder claims to demonstrate
    d = torch.cdist(xd[0], xd[0])
    d.fill_diagonal_(float("inf"))
    check(f"duplicate: each list holds near-identical pairs "
          f"(min distance {d.min():.3f})", d.min() < 0.5,
          "no planted duplicates found — a pointwise model can solve this "
          "task after all, so the folder's claim would be wrong")
    # and the demoted twin must be graded 0 while its partner is not
    i, j = divmod(int(d.argmin()), d.shape[1])
    check("duplicate: exactly one member of a twin pair is demoted to 0",
          (yd[0, i] == 0) != (yd[0, j] == 0),
          f"grades {yd[0, i].item()} and {yd[0, j].item()}")

    # ---- the metric module is a real duplicate, not a stub ------------------
    print("\n  -- metrics (duplicated on purpose, so verify them here too) --")
    labels = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    perfect = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    check("NDCG is 1.0 for a perfect ordering",
          abs(ndcg(perfect, labels).item() - 1.0) < 1e-6)
    check("NDCG is 0.0 when nothing is relevant",
          ndcg(perfect, torch.zeros(1, 4)).item() == 0.0)
    check("gain is exponential: a lone grade-4 gives DCG 15",
          abs(dcg(torch.tensor([[4.0, 0.0, 0.0]])).item() - 15.0) < 1e-6)

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
