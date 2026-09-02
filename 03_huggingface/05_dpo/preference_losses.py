"""
The offline preference-optimization family — DPO and its descendants.

WHY THIS FILE SITS NEXT TO A GRPO EXAMPLE
-----------------------------------------
`grpo_gsm8k_train.py` in this folder is *online* RL: it samples rollouts, scores
them with a verifier, and updates. That is the right tool when you have a cheap
ground-truth checker.

Most alignment work does not. It has a pile of `(prompt, chosen, rejected)`
pairs and no verifier, and for that the offline family below is both simpler and
an order of magnitude cheaper — no rollouts, no reward model, often no critic
and no reference model either.

The two families are usually described as competitors. They are better
understood by **what each one deletes from the RLHF pipeline**, because the
deletions are different and people conflate them constantly:

    full RLHF   SFT model + reward model + critic + reference model
    DPO         deletes the REWARD MODEL   (and the rollouts)
    GRPO        deletes the CRITIC         (keeps rollouts and reference)
    ORPO        deletes the REFERENCE MODEL (and the separate SFT stage)
    SimPO       deletes the REFERENCE MODEL (and normalises by length)

"DPO eliminates the reward model" and "GRPO eliminates the critic" are two
different sentences about two different components. Saying either is "the one
that removes the extra model" is how the confusion starts.

WHAT IS IMPLEMENTED
-------------------
Six losses, each as its actual formula rather than its description:

    dpo_loss     Rafailov et al., May 2023   — the founding paper
    ipo_loss     Azar et al., Oct 2023       — fixes DPO's overfitting
    cpo_loss     Xu et al., Jan 2024         — drops the reference model
    kto_loss     Ethayarajh et al., Feb 2024 — needs no PAIRS at all
    orpo_loss    Hong et al., Mar 2024       — folds SFT and alignment into one
    simpo_loss   Meng et al., May 2024       — reference-free, length-normalised

Plus `compare_losses`, which runs all six on the same batch so the differences
are visible rather than asserted.

Everything is plain PyTorch on plain tensors — no model, no GPU, no download.
The inputs are per-sequence log-probabilities, which is all any of these losses
actually consume. Covered by `tests/test_preference_losses.py`.

THE PROPERTY WORTH SEEING
-------------------------
Run `__main__` and look at the length experiment. DPO, IPO and CPO score a
response by the **sum** of its token log-probs, so two responses of identical
per-token quality get different scores purely because one is longer. ORPO and
SimPO divide by length and the difference vanishes.

Note the careful claim. Length-unnormalised objectives let length **into the
reward at all**; which direction that then pushes depends on your preference
data, and the widely-reported drift toward verbosity comes from the interaction
with datasets whose chosen responses are already longer. The narrow claim is
what this file computes; the broad one is what the SimPO paper argues.

References:
- Rafailov et al. "Direct Preference Optimization." https://arxiv.org/abs/2305.18290
- Azar et al. "A General Theoretical Paradigm..." https://arxiv.org/abs/2310.12036
- Xu et al. "Contrastive Preference Optimization." https://arxiv.org/abs/2401.08417
- Ethayarajh et al. "KTO: Model Alignment as Prospect Theoretic Optimization."
  https://arxiv.org/abs/2402.01306
- Hong et al. "ORPO: Monolithic Preference Optimization." https://arxiv.org/abs/2403.07691
- Meng et al. "SimPO: Simple Preference Optimization." https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class PreferenceBatch:
    """
    One batch of preference data, reduced to what the losses actually consume.

    Every loss here takes per-sequence log-probabilities, not logits and not
    token ids. Reducing to that is the point: it makes the *differences between
    the objectives* visible, instead of burying them under tokenisation and
    masking code that is identical for all six.

    Attributes:
        policy_chosen_logps: (B,) sum of token log-probs of the chosen response
            under the model being trained.
        policy_rejected_logps: (B,) same for the rejected response.
        ref_chosen_logps: (B,) under the frozen reference model. `None` for the
            reference-free losses — and passing `None` to a loss that needs it
            raises, rather than silently substituting zeros.
        ref_rejected_logps: (B,) same for rejected.
        chosen_lengths: (B,) token counts. Only SimPO and ORPO use these, which
            is exactly why those two behave differently on length.
        rejected_lengths: (B,) same for rejected.
    """

    policy_chosen_logps: torch.Tensor
    policy_rejected_logps: torch.Tensor
    ref_chosen_logps: Optional[torch.Tensor] = None
    ref_rejected_logps: Optional[torch.Tensor] = None
    chosen_lengths: Optional[torch.Tensor] = None
    rejected_lengths: Optional[torch.Tensor] = None

    def require_reference(self, name: str) -> None:
        """
        Fail loudly when a reference-based loss is handed no reference.

        Substituting zeros would be catastrophic and silent: the implicit
        reward would collapse to the raw policy log-prob, the loss would still
        decrease, and you would have trained something that is not DPO while
        the logs looked fine.
        """
        if self.ref_chosen_logps is None or self.ref_rejected_logps is None:
            raise ValueError(
                f"{name} needs a reference model. Passing None would silently "
                "reduce the implicit reward to a raw log-prob and train a "
                "different objective than the one you asked for. Use "
                "cpo_loss, orpo_loss or simpo_loss if you want to drop the "
                "reference model deliberately."
            )

    def require_lengths(self, name: str) -> None:
        """Fail loudly when a length-normalised loss is handed no lengths."""
        if self.chosen_lengths is None or self.rejected_lengths is None:
            raise ValueError(
                f"{name} normalises by sequence length and needs "
                "chosen_lengths / rejected_lengths. Without them it silently "
                "degenerates into a reference-free DPO and loses the length "
                "de-biasing that is its entire point."
            )


# ---------------------------------------------------------------------------
# DPO — Rafailov et al., May 2023
# ---------------------------------------------------------------------------

def dpo_loss(batch: PreferenceBatch, beta: float = 0.1) -> Dict[str, torch.Tensor]:
    r"""
    Direct Preference Optimization: the closed-form escape from RLHF.

    THE DERIVATION IN ONE LINE

    Under a KL-regularised RL objective the optimal policy has a closed form,
    which can be *inverted* to express the reward in terms of the policy:

        r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) + beta * log Z(x)

    Substitute that into a Bradley-Terry preference model and the intractable
    partition function Z(x) **cancels between the two responses**, leaving a
    plain binary classification loss:

        L = -log sigmoid( beta * (r_chosen - r_rejected) )

    That cancellation is the whole paper. It is why no reward model is needed:
    the policy *is* the reward model, read off implicitly.

    WHAT IT COSTS

    You still hold a frozen reference model in memory, and you are limited to
    preferences someone already collected — DPO cannot query a verifier or
    score a sample it has not seen. When you *do* have a cheap checker,
    throwing it away to collect pairwise human preferences is a strange trade,
    which is where GRPO comes in.

    Args:
        batch: needs policy and reference log-probs.
        beta: KL strength. Lower means the policy may stray further from the
            reference. 0.1 is the usual starting point; 0.01-0.5 is the range
            people actually use.

    Returns:
        dict with `loss` (scalar), and `chosen_rewards` / `rejected_rewards`
        (B,) — the *implicit* rewards, which are the thing worth logging.
        Their margin is the single most useful DPO diagnostic.
    """
    batch.require_reference("dpo_loss")

    chosen_logratio = batch.policy_chosen_logps - batch.ref_chosen_logps
    rejected_logratio = batch.policy_rejected_logps - batch.ref_rejected_logps

    logits = chosen_logratio - rejected_logratio
    loss = -F.logsigmoid(beta * logits).mean()

    return {
        "loss": loss,
        "chosen_rewards": beta * chosen_logratio.detach(),
        "rejected_rewards": beta * rejected_logratio.detach(),
        "margin": (beta * logits).detach(),
    }


# ---------------------------------------------------------------------------
# IPO — Azar et al., Oct 2023
# ---------------------------------------------------------------------------

def ipo_loss(batch: PreferenceBatch, tau: float = 0.1) -> Dict[str, torch.Tensor]:
    r"""
    Identity Preference Optimization: DPO without the overfitting.

    THE PROBLEM IT FIXES

    DPO's `-log sigmoid(.)` is unbounded below in the margin: pushing the
    margin from 10 to 20 still reduces the loss, so with a deterministic or
    near-deterministic preference dataset the optimiser keeps driving the
    margin up long after the preference has been learned. Because that margin
    is a log-ratio against the reference, "keep pushing" means "keep walking
    away from the reference model" — the KL term stops restraining anything
    and the model degenerates.

    IPO replaces the sigmoid with a **squared loss around a fixed target**:

        L = ( (log-ratio margin) - 1/(2*tau) )^2

    Now the objective has a *minimum at a finite margin*. Once the model is
    confident enough, further separation is actively penalised. The regulariser
    goes back to doing its job.

    Args:
        batch: needs policy and reference log-probs.
        tau: regularisation strength. The optimal margin is 1/(2*tau), so
            SMALLER tau targets a LARGER margin — the opposite direction from
            DPO's beta, which is a reliable source of confusion when switching.

    Returns:
        dict with `loss` and the implicit rewards.
    """
    batch.require_reference("ipo_loss")

    chosen_logratio = batch.policy_chosen_logps - batch.ref_chosen_logps
    rejected_logratio = batch.policy_rejected_logps - batch.ref_rejected_logps

    logits = chosen_logratio - rejected_logratio
    target = 1.0 / (2.0 * tau)
    loss = ((logits - target) ** 2).mean()

    return {
        "loss": loss,
        "chosen_rewards": chosen_logratio.detach(),
        "rejected_rewards": rejected_logratio.detach(),
        "margin": logits.detach(),
        "target_margin": torch.tensor(target),
    }


# ---------------------------------------------------------------------------
# CPO — Xu et al., Jan 2024
# ---------------------------------------------------------------------------

def cpo_loss(
    batch: PreferenceBatch, beta: float = 0.1, label_smoothing: float = 0.0
) -> Dict[str, torch.Tensor]:
    r"""
    Contrastive Preference Optimization: DPO with the reference model deleted.

    THE APPROXIMATION

    CPO drops pi_ref entirely and contrasts raw policy log-probs:

        L = -log sigmoid( beta * (log pi(y_w) - log pi(y_l)) )

    Formally this is DPO under a uniform reference prior. Practically it halves
    your memory: a 7B run no longer holds a second frozen 7B model. That is the
    entire pitch, and it is a big one — the reference model is pure overhead
    that contributes no gradient.

    WHAT YOU GIVE UP

    The KL anchor. Nothing now pins the policy near where it started, so CPO
    is normally paired with an SFT loss term on the chosen response to keep it
    anchored to something. The CPO paper trains that way; this function returns
    the preference term alone so it can be compared like-for-like with the
    others, and the caller adds the NLL term.

    Args:
        batch: reference log-probs are NOT required.
        beta: scaling on the log-prob difference.
        label_smoothing: in [0, 0.5). Treats a fraction of labels as flipped,
            which bounds the loss and helps when annotators disagree. This is
            cDPO / robust-DPO smoothing, applicable here too.
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(
            f"label_smoothing must be in [0, 0.5), got {label_smoothing}"
        )

    logits = batch.policy_chosen_logps - batch.policy_rejected_logps

    if label_smoothing > 0:
        loss = -(
            (1 - label_smoothing) * F.logsigmoid(beta * logits)
            + label_smoothing * F.logsigmoid(-beta * logits)
        ).mean()
    else:
        loss = -F.logsigmoid(beta * logits).mean()

    return {
        "loss": loss,
        "chosen_rewards": beta * batch.policy_chosen_logps.detach(),
        "rejected_rewards": beta * batch.policy_rejected_logps.detach(),
        "margin": (beta * logits).detach(),
    }


# ---------------------------------------------------------------------------
# KTO — Ethayarajh et al., Feb 2024
# ---------------------------------------------------------------------------

def kto_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    r"""
    Kahneman-Tversky Optimization: alignment without preference PAIRS.

    THE DATA PROBLEM IT SOLVES

    Every loss above needs `(prompt, chosen, rejected)` — two responses to the
    same prompt, ranked. That data is expensive and unnatural to collect. What
    organisations actually have is a pile of individual outputs with a thumbs
    up or thumbs down, unpaired, often wildly imbalanced.

    KTO consumes exactly that. Each example is one response plus one bit.

    THE PROSPECT-THEORY PART

    Kahneman and Tversky's value function is asymmetric: losses hurt more than
    equivalent gains please. KTO borrows the shape, measuring each response
    against a reference point z0 (the batch's mean KL from the reference) and
    applying separate weights to desirable and undesirable examples:

        desirable:    L = w_D * (1 - sigmoid( beta * (logratio - z0) ))
        undesirable:  L = w_U * (1 - sigmoid( beta * (z0 - logratio) ))

    THE KNOB THAT MATTERS

    `desirable_weight` / `undesirable_weight` exist because unpaired data is
    almost never balanced. The paper's guidance is to keep
    `w_D * n_desirable` roughly equal to `w_U * n_undesirable`. Ignore this on
    a 10:1 dataset and the model optimises almost entirely for the majority
    class while the loss curve looks perfectly healthy.

    Args:
        policy_logps: (B,) log-probs of each response under the policy.
        ref_logps: (B,) same under the reference model.
        labels: (B,) bool or 0/1 — True/1 means desirable.
        beta: scaling on the log-ratio.
        desirable_weight, undesirable_weight: class weights, per above.

    Returns:
        dict with `loss`, the implicit `rewards`, and `z0` (the reference point
        actually used — worth logging, since it drifts during training).
    """
    if policy_logps.shape != ref_logps.shape or policy_logps.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: policy {tuple(policy_logps.shape)}, "
            f"ref {tuple(ref_logps.shape)}, labels {tuple(labels.shape)}"
        )

    logratios = policy_logps - ref_logps
    labels = labels.bool()

    # The reference point: the batch's mean KL estimate, clamped at zero.
    # Detached on purpose — z0 is a moving baseline, not something to
    # backpropagate through. Letting gradient flow here lets the model reduce
    # its loss by dragging the baseline instead of improving the response,
    # which is the same degenerate solution a learned critic can fall into.
    z0 = torch.clamp(logratios.mean().detach(), min=0)

    desirable = desirable_weight * (1 - torch.sigmoid(beta * (logratios - z0)))
    undesirable = undesirable_weight * (1 - torch.sigmoid(beta * (z0 - logratios)))

    losses = torch.where(labels, desirable, undesirable)

    return {
        "loss": losses.mean(),
        "rewards": (beta * logratios).detach(),
        "z0": z0,
        "n_desirable": labels.sum(),
        "n_undesirable": (~labels).sum(),
    }


# ---------------------------------------------------------------------------
# ORPO — Hong et al., Mar 2024
# ---------------------------------------------------------------------------

def orpo_loss(
    batch: PreferenceBatch, lambda_: float = 1.0
) -> Dict[str, torch.Tensor]:
    r"""
    Odds Ratio Preference Optimization: one stage, no reference model.

    WHAT IT COLLAPSES

    Everything above assumes a pipeline: supervised fine-tune first, then align
    the result. ORPO's claim is that the SFT stage *already* contains the
    signal, because minimising NLL on chosen responses raises their likelihood
    without ever lowering the likelihood of bad ones. So it adds a mild penalty
    on the disfavoured response directly into the SFT objective:

        L = NLL(y_w) - lambda * log sigmoid( log(odds(y_w) / odds(y_l)) )

    where odds(y) = p(y) / (1 - p(y)) and p is the *length-normalised*
    sequence likelihood.

    One stage. No reference model, no separate alignment run, no second frozen
    copy in memory. For small models and modest budgets this is the cheapest
    thing in the family.

    WHY ODDS RATIO RATHER THAN LOG-PROB RATIO

    The odds ratio is deliberately gentler. A log-prob ratio grows without
    bound as p_l goes to zero, so it would let the alignment term overwhelm the
    NLL term and wreck the fluency the SFT part is building. The odds ratio
    saturates, which keeps the two terms comparable — the entire reason ORPO
    can share one objective instead of running two stages.

    Args:
        batch: needs lengths. No reference model required.
        lambda_: weight on the odds-ratio term. The paper uses 0.1-1.0. Too
            large and fluency degrades; too small and it is just SFT.

    Returns:
        dict with `loss`, plus `sft_loss` and `or_loss` separately — you want
        both curves, because ORPO failures are almost always one term
        dominating the other.
    """
    batch.require_lengths("orpo_loss")

    # Length-normalised sequence log-likelihood. ORPO works in average
    # log-prob space so that odds are comparable across response lengths.
    chosen_avg = batch.policy_chosen_logps / batch.chosen_lengths
    rejected_avg = batch.policy_rejected_logps / batch.rejected_lengths

    # log odds(y) = log p - log(1 - p), computed stably from log p.
    # log1p(-exp(x)) underflows badly near x = 0; log(-expm1(x)) is the stable
    # form for x < 0, which average log-probs always are.
    def _log_odds(avg_logp: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(avg_logp, max=-1e-6)
        return clamped - torch.log(-torch.expm1(clamped))

    log_odds_ratio = _log_odds(chosen_avg) - _log_odds(rejected_avg)

    or_loss = -F.logsigmoid(log_odds_ratio).mean()
    sft_loss = -chosen_avg.mean()          # NLL on the chosen response

    return {
        "loss": sft_loss + lambda_ * or_loss,
        "sft_loss": sft_loss.detach(),
        "or_loss": or_loss.detach(),
        "log_odds_ratio": log_odds_ratio.detach(),
    }


# ---------------------------------------------------------------------------
# SimPO — Meng et al., May 2024
# ---------------------------------------------------------------------------

def simpo_loss(
    batch: PreferenceBatch, beta: float = 2.0, gamma: float = 0.5
) -> Dict[str, torch.Tensor]:
    r"""
    Simple Preference Optimization: reference-free, length-normalised, with a margin.

    THE LENGTH ARGUMENT

    DPO's implicit reward is a **sum** of token log-probs. Sum more terms and
    you get a systematically different number, so response length leaks into
    the reward through a route that has nothing to do with quality. This is the
    mechanism behind the widely-reported result that DPO-tuned models get
    longer and more verbose.

    SimPO's fix is to make the reward the **average** log-prob — which is also
    what the model maximises at generation time, so the training objective and
    the decoding objective finally agree:

        L = -log sigmoid( beta * (avg_logp(y_w) - avg_logp(y_l)) - gamma )

    THE MARGIN

    `gamma` demands a *minimum* separation before the loss is satisfied.
    Without it, a pair that is barely ordered contributes almost no gradient;
    with it, the model must push past a threshold. This is what lets SimPO drop
    the reference model without collapsing — gamma supplies the floor the KL
    term used to.

    Run `__main__` to see the length effect measured rather than asserted.

    Args:
        batch: needs lengths. No reference model required.
        beta: 2.0-2.5 typically. Note this is much larger than DPO's 0.1,
            because average log-probs are far smaller in magnitude than sums —
            carrying DPO's beta across is a common and quiet mistake.
        gamma: target margin, 0.5-1.5. gamma/beta is the ratio that actually
            matters.
    """
    batch.require_lengths("simpo_loss")

    chosen_avg = batch.policy_chosen_logps / batch.chosen_lengths
    rejected_avg = batch.policy_rejected_logps / batch.rejected_lengths

    logits = beta * (chosen_avg - rejected_avg) - gamma
    loss = -F.logsigmoid(logits).mean()

    return {
        "loss": loss,
        "chosen_rewards": (beta * chosen_avg).detach(),
        "rejected_rewards": (beta * rejected_avg).detach(),
        "margin": logits.detach(),
    }


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

def needs_reference_model(name: str) -> bool:
    """
    Whether a method holds a second frozen copy of the model in memory.

    The single biggest practical difference between these methods, and the one
    that decides whether a 7B run fits on your card.
    """
    return name in {"DPO", "IPO", "KTO"}


def compare_losses(batch: PreferenceBatch) -> Dict[str, Dict[str, float]]:
    """
    Run every applicable loss on the same batch.

    Absolute values are NOT comparable across methods — each has its own scale,
    its own beta convention, and IPO is a squared error while the rest are
    log-sigmoid. Compare *behaviour* instead: which ones respond to a reference
    model, which respond to length, and which are indifferent.
    """
    out: Dict[str, Dict[str, float]] = {}

    out["DPO"] = {"loss": dpo_loss(batch)["loss"].item()}
    out["IPO"] = {"loss": ipo_loss(batch)["loss"].item()}
    out["CPO"] = {"loss": cpo_loss(batch)["loss"].item()}
    if batch.chosen_lengths is not None:
        out["ORPO"] = {"loss": orpo_loss(batch)["loss"].item()}
        out["SimPO"] = {"loss": simpo_loss(batch)["loss"].item()}
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    bar = "=" * 76

    print(bar)
    print("  The offline preference-optimization family")
    print(bar)
    print(f"  {'method':<8} {'year':<10} {'reference model?':<18} "
          f"{'needs pairs?':<14} {'length-normalised?'}")
    print("  " + "-" * 72)
    rows = [
        ("DPO",   "May 2023", True,  True,  False),
        ("IPO",   "Oct 2023", True,  True,  False),
        ("CPO",   "Jan 2024", False, True,  False),
        ("KTO",   "Feb 2024", True,  False, False),
        ("ORPO",  "Mar 2024", False, True,  True),
        ("SimPO", "May 2024", False, True,  True),
    ]
    for name, year, ref, pairs, norm in rows:
        print(f"  {name:<8} {year:<10} {'yes' if ref else 'NO':<18} "
              f"{'yes' if pairs else 'NO':<14} {'yes' if norm else 'no'}")

    # ---- a shared batch -------------------------------------------------
    batch = PreferenceBatch(
        policy_chosen_logps=torch.tensor([-10.0, -12.0, -8.0]),
        policy_rejected_logps=torch.tensor([-14.0, -13.0, -15.0]),
        ref_chosen_logps=torch.tensor([-11.0, -12.5, -9.0]),
        ref_rejected_logps=torch.tensor([-13.0, -12.0, -14.0]),
        chosen_lengths=torch.tensor([20.0, 25.0, 15.0]),
        rejected_lengths=torch.tensor([22.0, 24.0, 18.0]),
    )

    print()
    print(bar)
    print("  Same batch, every loss")
    print(bar)
    for name, vals in compare_losses(batch).items():
        print(f"  {name:<8} loss = {vals['loss']:.6f}")
    print("\n  Absolute values are NOT comparable across methods — different")
    print("  scales, different beta conventions, and IPO is a squared error")
    print("  while the rest are log-sigmoid. Compare behaviour, below.")

    # ---- who actually reads the reference model? ------------------------
    print()
    print(bar)
    print("  Perturb ONLY the reference model. Who notices?")
    print(bar)
    shifted = PreferenceBatch(
        policy_chosen_logps=batch.policy_chosen_logps,
        policy_rejected_logps=batch.policy_rejected_logps,
        ref_chosen_logps=batch.ref_chosen_logps - 3.0,
        ref_rejected_logps=batch.ref_rejected_logps + 3.0,
        chosen_lengths=batch.chosen_lengths,
        rejected_lengths=batch.rejected_lengths,
    )
    base, moved = compare_losses(batch), compare_losses(shifted)
    for name in base:
        delta = moved[name]["loss"] - base[name]["loss"]
        verdict = "USES the reference" if abs(delta) > 1e-9 else "reference-free"
        print(f"  {name:<8} delta = {delta:>+12.6f}   {verdict}")

    # ---- the length experiment ------------------------------------------
    print()
    print(bar)
    print("  The length experiment — identical QUALITY, different LENGTH")
    print(bar)
    print("  Two responses with the SAME average log-prob per token (-0.5),")
    print("  one twice as long as the other. Quality is identical by")
    print("  construction. Does the objective prefer one?\n")

    per_token = -0.5
    print(f"  Every response below has per-token log-prob {per_token} — i.e. "
          "IDENTICAL quality.\n")
    print(f"  {'length':>8}  {'sum log-prob':>14}  {'avg log-prob':>14}")
    print("  " + "-" * 42)
    for n in (10.0, 20.0, 40.0, 80.0):
        print(f"  {int(n):>8}  {per_token * n:>14.2f}  {per_token:>14.2f}")

    print()
    print("  The sum-based score falls linearly with length; the average-based")
    print("  score does not move. DPO, IPO and CPO score with the SUM, so")
    print("  LENGTH ENTERS THE REWARD even when quality is held fixed.")
    print("  ORPO and SimPO normalise by length, so it cannot.")
    print()

    # Same per-token quality, chosen is twice as long as rejected.
    short_len, long_len = 20.0, 40.0
    length_batch = PreferenceBatch(
        policy_chosen_logps=torch.tensor([per_token * long_len]),
        policy_rejected_logps=torch.tensor([per_token * short_len]),
        ref_chosen_logps=torch.tensor([per_token * long_len]),
        ref_rejected_logps=torch.tensor([per_token * short_len]),
        chosen_lengths=torch.tensor([long_len]),
        rejected_lengths=torch.tensor([short_len]),
    )
    c = cpo_loss(length_batch)
    s = simpo_loss(length_batch)
    print(f"  CPO   margin  {c['margin'].item():>+8.4f}   nonzero purely from "
          "the length difference")
    print(f"  SimPO margin  {s['margin'].item():>+8.4f}   exactly -gamma — "
          "length has cancelled")
    print()
    print("  NOTE ON DIRECTION. Here the longer response scores LOWER, because")
    print("  more tokens means a more negative sum. The reported real-world")
    print("  drift is toward LONGER outputs, which comes from the interaction")
    print("  with preference data where chosen responses are already longer.")
    print("  The claim this experiment supports is the narrower, sturdier one:")
    print("  a length-unnormalised objective lets length into the reward at")
    print("  all. Which direction it then pushes depends on your data.")
    print(bar)
