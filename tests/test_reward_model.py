# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: the Bradley-Terry reward objective and its two surprises.

Run:
    uv run tests/test_reward_model.py

Why this suite exists
---------------------
A reward model is four lines of loss and two properties that bite people:

  1. **Only differences are identified.** Shift every score by a constant and
     the loss is unchanged, so any statement about an absolute reward value is
     meaningless. Downstream this is why RLHF needs a KL leash — off its
     training distribution the model is not merely inaccurate, it is arbitrary.

  2. **Loss falls while accuracy stays flat.** Widening an already-correct gap
     keeps reducing the loss forever. A training curve that looks great is
     compatible with the ranking never improving, so pairwise ACCURACY is the
     metric and loss is not.

Both are asserted here, plus the float32 caveat on (1) that the module itself
discovered: the objective is exactly shift-invariant in real arithmetic and only
approximately so in floating point, because subtracting two large nearby numbers
is catastrophic cancellation.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "05_huggingface_reward_model"))

from reward_modeling import bradley_terry_loss, is_shift_invariant  # noqa: E402


def test_objective(r: Results) -> None:
    """The loss must reward correct ordering and punish the reverse."""
    chosen = torch.tensor([2.0, 1.5, 0.5])
    rejected = torch.tensor([1.0, 1.0, -0.5])

    good = bradley_terry_loss(chosen, rejected)
    bad = bradley_terry_loss(rejected, chosen)

    r.check(good["loss"].item() < bad["loss"].item(),
            "correct ordering has a lower loss than the reverse",
            f"{good['loss'].item():.6f} vs {bad['loss'].item():.6f} — a sign "
            "error would invert this and nothing else would notice")
    r.check(good["accuracy"].item() == 1.0,
            "accuracy is 1.0 when every pair is ranked correctly")
    r.check(bad["accuracy"].item() == 0.0,
            "accuracy is 0.0 when every pair is ranked backwards")

    # A tied pair sits exactly at the decision boundary: -log sigmoid(0) = log 2.
    tied = bradley_terry_loss(torch.tensor([1.0]), torch.tensor([1.0]))
    r.check(abs(tied["loss"].item() - torch.tensor(2.0).log().item()) < 1e-6,
            "a tied pair costs exactly log(2)",
            f"got {tied['loss'].item():.6f}, expected {torch.tensor(2.0).log().item():.6f}")

    try:
        bradley_terry_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))
        caught = False
    except ValueError:
        caught = True
    r.check(caught, "mismatched shapes are rejected")


def test_shift_invariance(r: Results) -> None:
    """
    Surprise 1: absolute reward values carry no information.

    Asserted with a tolerance, deliberately. The objective is exactly
    shift-invariant in real arithmetic; float32 loses mantissa bits when you
    subtract two large nearby numbers, so exact equality is the WRONG test and
    a version that demanded it would fail for large shifts.
    """
    chosen = torch.tensor([2.0, 1.5, 0.5, -1.0])
    rejected = torch.tensor([1.0, 1.4, -0.5, -0.5])
    base = bradley_terry_loss(chosen, rejected)["loss"].item()

    for shift in (0.0, 1.0, 10.0, 100.0, -50.0):
        moved = bradley_terry_loss(chosen + shift, rejected + shift)["loss"].item()
        r.check(abs(moved - base) < 1e-4,
                f"shifting every score by {shift:+g} leaves the loss unchanged",
                f"{base:.9f} vs {moved:.9f}")
        r.check(is_shift_invariant(chosen, rejected, shift=shift),
                f"is_shift_invariant reports True for a shift of {shift:+g}")

    # Accuracy must be *exactly* invariant — it is a comparison, not arithmetic.
    for shift in (0.0, 1000.0, -1000.0):
        acc = bradley_terry_loss(chosen + shift, rejected + shift)["accuracy"]
        r.check(acc.item() == 0.75,
                f"accuracy is EXACTLY invariant under a shift of {shift:+g}",
                f"got {acc.item()}")

    # The float32 caveat, asserted rather than claimed: a large enough shift
    # does perturb the loss, which is why the tolerance above exists.
    huge = bradley_terry_loss(chosen + 1e4, rejected + 1e4)["loss"].item()
    r.check(huge != base,
            "a very large shift DOES perturb the loss in float32",
            f"{base!r} vs {huge!r} — if these were bit-identical, the "
            "tolerance in is_shift_invariant would be unmotivated")


def test_loss_falls_while_accuracy_is_flat(r: Results) -> None:
    """
    Surprise 2: a beautiful loss curve can mean nothing.

    This is why the READMEs tell you to plot accuracy.
    """
    losses, accuracies = [], []
    for gap in (0.5, 1.0, 2.0, 5.0, 10.0):
        out = bradley_terry_loss(torch.tensor([gap]), torch.tensor([0.0]))
        losses.append(out["loss"].item())
        accuracies.append(out["accuracy"].item())

    r.check(all(a > b for a, b in zip(losses, losses[1:])),
            "loss decreases monotonically as an already-correct gap widens",
            f"{[round(x, 6) for x in losses]}")
    r.check(all(a == 1.0 for a in accuracies),
            "accuracy is pinned at 100% the entire time",
            f"{accuracies} — the loss improved 10,000x while the ranking "
            "did not change at all")

    # The loss is unbounded below, so there is no natural stopping point.
    r.check(bradley_terry_loss(torch.tensor([100.0]),
                               torch.tensor([0.0]))["loss"].item() < 1e-30,
            "the loss approaches zero without bound — no natural convergence")


def test_margin(r: Results) -> None:
    """A per-example margin must make strong pairs harder to satisfy."""
    chosen = torch.tensor([2.0])
    rejected = torch.tensor([1.0])

    plain = bradley_terry_loss(chosen, rejected)["loss"].item()
    with_margin = bradley_terry_loss(
        chosen, rejected, margin=torch.tensor([1.0])
    )["loss"].item()

    r.check(with_margin > plain,
            "requiring a margin raises the loss for the same score gap",
            f"{plain:.6f} -> {with_margin:.6f}; this is how a 'much better' "
            "annotation is made to demand more separation than 'slightly "
            "better'")

    # A margin of 0 must be a genuine no-op.
    zero = bradley_terry_loss(chosen, rejected,
                              margin=torch.tensor([0.0]))["loss"].item()
    r.check(abs(zero - plain) < 1e-9, "a zero margin is a no-op",
            f"{plain:.9f} vs {zero:.9f}")


def main() -> int:
    r = Results("Bradley-Terry reward modelling — objective and its surprises")
    test_objective(r)
    test_shift_invariance(r)
    test_loss_falls_while_accuracy_is_flat(r)
    test_margin(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
