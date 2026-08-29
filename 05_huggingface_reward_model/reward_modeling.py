"""
Bradley-Terry reward modelling — the objective, on plain tensors.

WHAT A REWARD MODEL IS FOR
--------------------------
Humans are unreliable at absolute scores and reliable at comparisons. Nobody
can tell you consistently whether a response is a 7 or an 8 out of 10; almost
everyone can tell you which of two responses they prefer.

So a reward model is not trained to predict a rating. It is trained so that the
DIFFERENCE between its scores reproduces observed preferences:

    P(y_w beats y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )

Maximising the likelihood of the observed comparisons gives the loss:

    L = -log sigmoid( r(x, y_w) - r(x, y_l) )

which is logistic regression on score differences, and is the entirety of
`RewardTrainer`.

THE PROPERTY THAT SURPRISES PEOPLE
----------------------------------
**Only differences are identified.** Add a constant to every score and the loss
does not change -- sigmoid sees only the gap. (Exactly so in real arithmetic;
in float32 large shifts cost precision, which is its own lesson -- see
`is_shift_invariant`.) So:

  * "our reward model scores 0.8" is a meaningless statement
  * two reward models with wildly different score ranges can be equally good
  * you cannot compare reward values across models, or across training runs of
    the same model

This is not a quirk to work around; it is a property of the objective, and it
is proved numerically in `__main__` and asserted in `tests/test_reward_model.py`.

WHY IT MATTERS DOWNSTREAM
-------------------------
Shift-invariance means the reward model is only anchored where it saw data. Off
distribution it is not merely inaccurate, it is *arbitrary* -- and an
unconstrained RL optimiser will find where it is arbitrary and go there. That is
reward hacking, and it is why the RLHF objective carries a KL leash to the
reference model, and why `05_huggingface_dpo` (which needs no reward model at
all) is often the better trade.

Plain PyTorch, no model, no GPU, no download. Covered by
`tests/test_reward_model.py`.

References:
- Bradley & Terry. "Rank Analysis of Incomplete Block Designs" (1952).
- Christiano et al. "Deep RL from Human Preferences." https://arxiv.org/abs/1706.03741
- Ouyang et al. "InstructGPT." https://arxiv.org/abs/2203.02155
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    The reward-model objective.

    Args:
        chosen_rewards: (B,) scalar score for the preferred response.
        rejected_rewards: (B,) scalar score for the dispreferred one.
        margin: (B,) optional per-example margin. Use when your annotations
            carry a STRENGTH of preference ("much better" vs "slightly
            better"); the model is then asked to separate the strong pairs
            further. Llama 2 used this. Without it every pair is treated as
            equally decisive, which throws away real signal.

    Returns:
        dict with `loss`, `accuracy` (fraction of pairs ranked correctly), and
        `margin` (mean score gap).

    Accuracy is the metric to actually watch. The loss keeps falling as the
    model separates already-correct pairs further, so it can improve while the
    ranking does not.
    """
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError(
            f"shape mismatch: chosen {tuple(chosen_rewards.shape)} vs "
            f"rejected {tuple(rejected_rewards.shape)}"
        )

    diff = chosen_rewards - rejected_rewards
    if margin is not None:
        diff = diff - margin

    return {
        "loss": -F.logsigmoid(diff).mean(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        "margin": diff.mean().detach(),
    }


def is_shift_invariant(
    chosen: torch.Tensor, rejected: torch.Tensor,
    shift: float = 100.0, atol: float = 1e-5,
) -> bool:
    """
    Check that adding a constant to every score leaves the loss unchanged.

    NOTE THE TOLERANCE, AND WHY IT IS NOT EXACT

    The objective is *mathematically* shift-invariant: the loss depends only on
    `chosen - rejected`, so any constant added to both cancels exactly.

    Floating point does not cooperate. Computing `(x + 1000) - (y + 1000)` in
    float32 is **catastrophic cancellation** -- the shared magnitude eats the
    mantissa bits that encoded the small difference you actually care about.
    Shift by 1000 and the loss moves in the 6th decimal place; shift far enough
    and the difference is destroyed entirely.

    This is not pedantry. A reward model whose outputs drift to large absolute
    values is numerically losing the very signal it is trained on, and the
    symptom is a training curve that goes quiet for no visible reason.
    Reward-model implementations often add a small penalty on the mean score
    for exactly this reason -- it keeps the scores near zero, where the
    subtraction is well conditioned.

    Args:
        chosen, rejected: (B,) scores.
        shift: constant added to both.
        atol: absolute tolerance. Exact equality is the wrong test; see above.
    """
    a = bradley_terry_loss(chosen, rejected)["loss"]
    b = bradley_terry_loss(chosen + shift, rejected + shift)["loss"]
    return bool(torch.allclose(a, b, atol=atol))


if __name__ == "__main__":
    torch.manual_seed(0)
    bar = "=" * 74

    chosen = torch.tensor([2.0, 1.5, 0.5, -1.0])
    rejected = torch.tensor([1.0, 1.4, -0.5, -0.5])

    print(bar)
    print("  Bradley-Terry reward modelling")
    print(bar)
    out = bradley_terry_loss(chosen, rejected)
    print(f"  loss      {out['loss'].item():.6f}")
    print(f"  accuracy  {out['accuracy'].item():.1%}   "
          "(3 of 4 pairs ranked correctly)")
    print(f"  margin    {out['margin'].item():+.4f}")

    print()
    print(bar)
    print("  Only DIFFERENCES are identified")
    print(bar)
    for shift in (0.0, 10.0, 100.0, -1000.0):
        loss = bradley_terry_loss(chosen + shift, rejected + shift)["loss"]
        print(f"  shift every score by {shift:>+8.1f}  ->  loss {loss.item():.9f}")
    print()
    print("  Unchanged to ~6 decimal places. The absolute scale of a reward")
    print("  model is meaningless — 'our RM scores 0.8' says nothing, and two")
    print("  RMs with different ranges can be equally good.")
    print()
    print("  But look closely: the digits DO drift. The objective is exactly")
    print("  shift-invariant in real arithmetic; float32 is not. Computing")
    print("  (x+1000) - (y+1000) is catastrophic cancellation — the shared")
    print("  magnitude eats the mantissa bits holding the small difference.")
    print("  A reward model whose scores drift large is numerically losing the")
    print("  signal it trains on, which is why implementations often penalise")
    print("  the mean score to keep it near zero.")

    print()
    print(bar)
    print("  Loss falls while accuracy does NOT improve")
    print(bar)
    print("  Widening an already-correct gap keeps reducing the loss:\n")
    print(f"  {'gap':>6}  {'loss':>10}  {'accuracy':>9}")
    print("  " + "-" * 30)
    for gap in (0.5, 1.0, 2.0, 5.0, 10.0):
        c = torch.tensor([gap]); r = torch.tensor([0.0])
        o = bradley_terry_loss(c, r)
        print(f"  {gap:>6.1f}  {o['loss'].item():>10.6f}  "
              f"{o['accuracy'].item():>9.1%}")
    print()
    print("  Accuracy is pinned at 100% throughout. Watch ACCURACY, not loss —")
    print("  a falling loss is compatible with learning nothing new.")
    print(bar)
