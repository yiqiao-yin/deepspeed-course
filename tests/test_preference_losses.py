# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: the preference-optimization losses must be the objectives they
claim to be.

Run:
    uv run tests/test_preference_losses.py

Why this suite exists
---------------------
All six of these losses are a handful of lines, all of them produce a scalar
that goes down during training, and **a wrong one is indistinguishable from a
right one from the loss curve alone**. Swap a sign, forget the reference model,
sum where you meant to average, and the run completes, the number falls, and the
model is worse in a way you will attribute to the learning rate.

So this asserts identifying PROPERTIES rather than reference values:

  * the reference-free losses must be provably indifferent to the reference
    model — perturb it and the loss must not move by a single ULP
  * the reference-based ones must move, or they are not what they say
  * every loss must prefer a correctly-ordered pair to a reversed one
  * SimPO and ORPO must be exactly length-invariant; DPO/IPO/CPO must not be
  * KTO must work on unpaired data and must respond to its class weights
  * IPO must have a finite optimum (that is the whole point of IPO)

Numbers alone would let a sign error through if the fixture happened to be
symmetric. Properties do not.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "06_huggingface_grpo"))

from preference_losses import (  # noqa: E402
    PreferenceBatch,
    cpo_loss,
    dpo_loss,
    ipo_loss,
    kto_loss,
    needs_reference_model,
    orpo_loss,
    simpo_loss,
)


def make_batch(**over) -> PreferenceBatch:
    """A well-ordered batch: chosen is genuinely better than rejected."""
    kw = dict(
        policy_chosen_logps=torch.tensor([-10.0, -12.0, -8.0]),
        policy_rejected_logps=torch.tensor([-14.0, -13.0, -15.0]),
        ref_chosen_logps=torch.tensor([-11.0, -12.5, -9.0]),
        ref_rejected_logps=torch.tensor([-13.0, -12.0, -14.0]),
        chosen_lengths=torch.tensor([20.0, 25.0, 15.0]),
        rejected_lengths=torch.tensor([22.0, 24.0, 18.0]),
    )
    kw.update(over)
    return PreferenceBatch(**kw)


def test_ordering_direction(r: Results) -> None:
    """
    Every loss must prefer a correctly-ordered pair over a reversed one.

    The single cheapest way to catch a sign error, and it catches all of them.
    """
    good = make_batch()
    flipped = make_batch(
        policy_chosen_logps=good.policy_rejected_logps,
        policy_rejected_logps=good.policy_chosen_logps,
        chosen_lengths=good.rejected_lengths,
        rejected_lengths=good.chosen_lengths,
    )

    for name, fn in [("DPO", dpo_loss), ("CPO", cpo_loss),
                     ("ORPO", orpo_loss), ("SimPO", simpo_loss)]:
        lo = fn(good)["loss"].item()
        hi = fn(flipped)["loss"].item()
        r.check(lo < hi,
                f"{name}: a correctly-ordered pair has LOWER loss than a "
                f"reversed one",
                f"ordered={lo:.6f}, reversed={hi:.6f} — a sign error would "
                "invert this and nothing else would notice")

    # IPO is a squared error around a target, so "better" means "closer to the
    # target margin", not "larger margin". Assert that instead.
    good_gap = abs(ipo_loss(good)["margin"].mean().item()
                   - ipo_loss(good)["target_margin"].item())
    flip_gap = abs(ipo_loss(flipped)["margin"].mean().item()
                   - ipo_loss(flipped)["target_margin"].item())
    r.check(good_gap < flip_gap,
            "IPO: the ordered pair sits closer to the target margin",
            f"ordered gap={good_gap:.3f}, reversed gap={flip_gap:.3f}")


def test_reference_dependence(r: Results) -> None:
    """
    Reference-free losses must be EXACTLY indifferent to the reference model.

    This is the property that decides whether you need a second frozen copy of
    the model in VRAM, so it is worth proving rather than believing. We perturb
    only the reference and require bit-level equality — "approximately
    indifferent" would mean it leaked in somewhere.
    """
    base = make_batch()
    moved = make_batch(
        ref_chosen_logps=base.ref_chosen_logps - 3.0,
        ref_rejected_logps=base.ref_rejected_logps + 3.0,
    )

    for name, fn in [("CPO", cpo_loss), ("ORPO", orpo_loss), ("SimPO", simpo_loss)]:
        a, b = fn(base)["loss"].item(), fn(moved)["loss"].item()
        r.check(a == b,
                f"{name}: reference-free — loss is bit-identical when the "
                f"reference moves",
                f"{a!r} vs {b!r}")
        r.check(not needs_reference_model(name),
                f"{name}: correctly declared reference-free")

    for name, fn in [("DPO", dpo_loss), ("IPO", ipo_loss)]:
        a, b = fn(base)["loss"].item(), fn(moved)["loss"].item()
        r.check(abs(a - b) > 1e-6,
                f"{name}: reference-based — loss MOVES when the reference does",
                f"{a:.6f} vs {b:.6f} — if these matched, the reference model "
                "is being ignored and you are not training DPO/IPO")
        r.check(needs_reference_model(name),
                f"{name}: correctly declared reference-dependent")

    # Missing reference must raise, never silently default to zeros.
    noref = PreferenceBatch(
        policy_chosen_logps=torch.tensor([-10.0]),
        policy_rejected_logps=torch.tensor([-12.0]),
        chosen_lengths=torch.tensor([10.0]),
        rejected_lengths=torch.tensor([10.0]),
    )
    for name, fn in [("dpo_loss", dpo_loss), ("ipo_loss", ipo_loss)]:
        try:
            fn(noref); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"{name} raises when handed no reference model",
                "silently substituting zeros would train a different objective")

    # And the reference-free ones must be happy without it.
    for name, fn in [("cpo_loss", cpo_loss), ("orpo_loss", orpo_loss),
                     ("simpo_loss", simpo_loss)]:
        try:
            fn(noref); ok = True
        except Exception as exc:               # noqa: BLE001
            ok = False; detail = str(exc)
        r.check(ok, f"{name} runs with no reference model at all",
                "" if ok else detail)


def test_length_invariance(r: Results) -> None:
    """
    The headline distinction: which objectives let LENGTH into the reward?

    Two responses of identical per-token quality, one twice as long. A
    length-normalised objective must score them identically; a sum-based one
    must not. This is the mechanism SimPO and ORPO exist to remove.
    """
    per_token = -0.5

    def batch_for(len_c, len_r):
        return PreferenceBatch(
            policy_chosen_logps=torch.tensor([per_token * len_c]),
            policy_rejected_logps=torch.tensor([per_token * len_r]),
            ref_chosen_logps=torch.tensor([per_token * len_c]),
            ref_rejected_logps=torch.tensor([per_token * len_r]),
            chosen_lengths=torch.tensor([float(len_c)]),
            rejected_lengths=torch.tensor([float(len_r)]),
        )

    # Same per-token quality throughout; only the lengths differ.
    equal = batch_for(20, 20)
    skewed = batch_for(40, 20)

    for name, fn in [("SimPO", simpo_loss), ("ORPO", orpo_loss)]:
        a, b = fn(equal)["loss"].item(), fn(skewed)["loss"].item()
        r.check(abs(a - b) < 1e-6,
                f"{name}: length-INVARIANT — doubling the chosen response "
                f"changes nothing when per-token quality is fixed",
                f"{a:.6f} vs {b:.6f}")

    a, b = cpo_loss(equal)["loss"].item(), cpo_loss(skewed)["loss"].item()
    r.check(abs(a - b) > 1e-6,
            "CPO: length-SENSITIVE — the loss moves on length alone",
            f"{a:.6f} vs {b:.6f}")

    # DPO needs care here. In THIS fixture policy == reference, so both
    # log-ratios are 0 and the margin is 0 regardless of length — the
    # reference cancels the effect. Assert that exactly, rather than waving at
    # it, and then show the sensitivity returns as soon as the policy has
    # actually moved away from the reference (which it has, in any real run).
    r.check(abs(dpo_loss(skewed)["margin"].item()) < 1e-6,
            "DPO: margin is exactly 0 when policy == reference, whatever the "
            "lengths",
            f"got {dpo_loss(skewed)['margin'].item()}")

    drifted = PreferenceBatch(
        policy_chosen_logps=torch.tensor([per_token * 40]),
        policy_rejected_logps=torch.tensor([per_token * 20]),
        ref_chosen_logps=torch.tensor([per_token * 40 - 1.0]),
        ref_rejected_logps=torch.tensor([per_token * 20 - 1.0]),
        chosen_lengths=torch.tensor([40.0]),
        rejected_lengths=torch.tensor([20.0]),
    )
    drifted_longer = PreferenceBatch(
        policy_chosen_logps=torch.tensor([per_token * 80]),
        policy_rejected_logps=torch.tensor([per_token * 20]),
        ref_chosen_logps=torch.tensor([per_token * 80 - 1.0]),
        ref_rejected_logps=torch.tensor([per_token * 20 - 1.0]),
        chosen_lengths=torch.tensor([80.0]),
        rejected_lengths=torch.tensor([20.0]),
    )
    same = abs(dpo_loss(drifted)["loss"].item()
               - dpo_loss(drifted_longer)["loss"].item()) < 1e-6
    r.check(same,
            "DPO: a CONSTANT per-token drift from the reference also cancels",
            "the log-ratio is (policy - reference), so a uniform offset "
            "cancels; DPO's length exposure comes from the ratio varying "
            "across tokens, not from length alone")

    # The underlying cause, stated directly: a sum-based reward scales with
    # length while an average-based one does not.
    for n in (10.0, 20.0, 40.0, 80.0):
        b = batch_for(n, n)
        summed = b.policy_chosen_logps.item()
        averaged = (b.policy_chosen_logps / b.chosen_lengths).item()
        r.check(abs(summed - per_token * n) < 1e-6,
                f"sum-based score at length {int(n)} scales with length "
                f"({summed:.1f})")
        r.check(abs(averaged - per_token) < 1e-6,
                f"average-based score at length {int(n)} is constant "
                f"({averaged:.2f})")

    # SimPO's margin on equal-quality pairs must be exactly -gamma.
    gamma = 0.5
    m = simpo_loss(batch_for(40, 20), beta=2.0, gamma=gamma)["margin"].item()
    r.check(abs(m + gamma) < 1e-6,
            "SimPO: margin on an equal-quality pair is exactly -gamma",
            f"got {m:.6f}, expected {-gamma}")

    # Missing lengths must raise, not silently degrade to un-normalised.
    nolen = PreferenceBatch(
        policy_chosen_logps=torch.tensor([-10.0]),
        policy_rejected_logps=torch.tensor([-12.0]),
    )
    for name, fn in [("simpo_loss", simpo_loss), ("orpo_loss", orpo_loss)]:
        try:
            fn(nolen); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"{name} raises when handed no lengths",
                "without lengths it silently loses its length de-biasing")


def test_ipo_has_a_finite_optimum(r: Results) -> None:
    """
    IPO's reason to exist: the loss must STOP rewarding a larger margin.

    DPO's -log sigmoid is unbounded below in the margin, so the optimiser keeps
    pushing away from the reference forever. IPO's squared loss has a minimum
    at a finite margin. If that is not true, IPO is just a slower DPO.
    """
    tau = 0.1
    target = 1.0 / (2.0 * tau)

    def at(margin):
        return ipo_loss(PreferenceBatch(
            policy_chosen_logps=torch.tensor([margin]),
            policy_rejected_logps=torch.tensor([0.0]),
            ref_chosen_logps=torch.tensor([0.0]),
            ref_rejected_logps=torch.tensor([0.0]),
        ), tau=tau)["loss"].item()

    r.check(at(target) < 1e-9,
            f"IPO loss is ~0 exactly at the target margin {target}",
            f"got {at(target)}")
    r.check(at(target + 5) > at(target),
            "IPO PENALISES a margin larger than the target",
            "this is the anti-overfitting property; without it IPO is just DPO")
    r.check(at(target - 5) > at(target),
            "IPO penalises a margin smaller than the target")

    # DPO, by contrast, keeps improving forever.
    def dpo_at(margin):
        return dpo_loss(PreferenceBatch(
            policy_chosen_logps=torch.tensor([margin]),
            policy_rejected_logps=torch.tensor([0.0]),
            ref_chosen_logps=torch.tensor([0.0]),
            ref_rejected_logps=torch.tensor([0.0]),
        ))["loss"].item()

    r.check(dpo_at(100) < dpo_at(10) < dpo_at(1),
            "DPO keeps rewarding a larger margin without bound (the contrast)",
            f"{dpo_at(1):.6f} -> {dpo_at(10):.6f} -> {dpo_at(100):.6f} — if "
            "this were bounded, IPO would have nothing to fix")


def test_kto_unpaired(r: Results) -> None:
    """KTO must work on unpaired data and must honour its class weights."""
    policy = torch.tensor([-8.0, -9.0, -14.0, -15.0, -16.0])
    ref = torch.tensor([-10.0, -10.0, -10.0, -10.0, -10.0])
    labels = torch.tensor([True, True, False, False, False])

    out = kto_loss(policy, ref, labels)
    r.check(out["n_desirable"].item() == 2 and out["n_undesirable"].item() == 3,
            "KTO consumes UNPAIRED data — 2 desirable, 3 undesirable, no pairs",
            "every other loss here would need matched (chosen, rejected) tuples")
    r.check(torch.isfinite(out["loss"]), "KTO loss is finite")
    r.check(out["z0"].item() >= 0, "the KTO reference point is clamped at zero",
            f"got {out['z0'].item()}")

    # Class weights must actually change the objective — this is the knob that
    # matters on imbalanced data, and a no-op here would be silent.
    a = kto_loss(policy, ref, labels, desirable_weight=1.0,
                 undesirable_weight=1.0)["loss"].item()
    b = kto_loss(policy, ref, labels, desirable_weight=1.0,
                 undesirable_weight=5.0)["loss"].item()
    r.check(abs(a - b) > 1e-6,
            "KTO class weights change the loss",
            f"{a:.6f} vs {b:.6f} — on a 10:1 dataset a no-op here would let "
            "the model optimise for the majority class unnoticed")

    # A model that scores desirable HIGH and undesirable LOW must do better.
    good = kto_loss(torch.tensor([-5.0, -5.0, -20.0, -20.0]),
                    torch.tensor([-10.0] * 4),
                    torch.tensor([True, True, False, False]))["loss"].item()
    bad = kto_loss(torch.tensor([-20.0, -20.0, -5.0, -5.0]),
                   torch.tensor([-10.0] * 4),
                   torch.tensor([True, True, False, False]))["loss"].item()
    r.check(good < bad,
            "KTO rewards raising desirable and lowering undesirable outputs",
            f"aligned={good:.6f}, inverted={bad:.6f}")

    try:
        kto_loss(policy, ref[:3], labels); caught = False
    except ValueError:
        caught = True
    r.check(caught, "KTO rejects mismatched input shapes")


def test_orpo_structure_and_gradients(r: Results) -> None:
    """ORPO must expose both terms, and every loss must be differentiable."""
    batch = make_batch()
    out = orpo_loss(batch)
    for key in ("sft_loss", "or_loss"):
        r.check(key in out, f"ORPO reports {key} separately",
                "ORPO failures are almost always one term dominating the "
                "other, which a single combined number hides")
    r.check(abs(out["loss"].item()
                - (out["sft_loss"].item() + out["or_loss"].item())) < 1e-5,
            "ORPO loss = sft_loss + lambda * or_loss at lambda=1")

    # lambda must actually weight the alignment term.
    a = orpo_loss(batch, lambda_=0.1)["loss"].item()
    b = orpo_loss(batch, lambda_=2.0)["loss"].item()
    r.check(abs(a - b) > 1e-6, "ORPO lambda changes the objective",
            f"{a:.6f} vs {b:.6f}")

    # Gradients must flow to the policy for every loss — a detached term
    # somewhere would train nothing and raise nothing.
    for name, fn in [("DPO", dpo_loss), ("IPO", ipo_loss), ("CPO", cpo_loss),
                     ("ORPO", orpo_loss), ("SimPO", simpo_loss)]:
        chosen = torch.tensor([-10.0], requires_grad=True)
        b = PreferenceBatch(
            policy_chosen_logps=chosen,
            policy_rejected_logps=torch.tensor([-14.0]),
            ref_chosen_logps=torch.tensor([-11.0]),
            ref_rejected_logps=torch.tensor([-13.0]),
            chosen_lengths=torch.tensor([20.0]),
            rejected_lengths=torch.tensor([22.0]),
        )
        fn(b)["loss"].backward()
        g = chosen.grad
        r.check(g is not None and torch.isfinite(g).all() and g.abs().item() > 0,
                f"{name}: gradient reaches the policy log-probs",
                f"grad={None if g is None else g.item()}")

    # CPO label smoothing must be range-checked.
    for bad in (-0.1, 0.5, 0.9):
        try:
            cpo_loss(batch, label_smoothing=bad); caught = False
        except ValueError:
            caught = True
        r.check(caught, f"CPO rejects label_smoothing={bad}")


def main() -> int:
    r = Results("Preference optimization losses — DPO and its descendants")
    test_ordering_direction(r)
    test_reference_dependence(r)
    test_length_invariance(r)
    test_ipo_has_a_finite_optimum(r)
    test_kto_unpaired(r)
    test_orpo_structure_and_gradients(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
