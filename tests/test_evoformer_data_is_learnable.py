# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: the synthetic MSAs actually carry a learnable signal.

Run:
    uv run tests/test_evoformer_data_is_learnable.py

Why this suite exists
---------------------
`01_basics/02_convnet` shipped `x = randn(...)` with `y = randint(...)` --
labels independent of inputs, zero mutual information, and a script that
exited 0 advising the reader to train longer against an unreachable target.
The lab sat at the information-theoretic ceiling for months and every check
was green. `POSTMORTEMS.md` has the full account.

Synthetic protein data is an easy place to repeat that, because a contact map
and an MSA can be generated independently and the result looks completely
plausible. So this suite asserts, on a HELD-OUT split, that:

    coupling = 1.0   the contacts ARE recoverable
    coupling = 0.0   the contacts are NOT recoverable   <- the counterexample
    depth    = 1     the contacts are NOT recoverable   <- why MSAs exist

The middle line is the one that keeps the suite honest. A learnability check
that never sees unlearnable data would pass while returning True
unconditionally -- which is exactly the shape of the bug `beats_chance()`
shipped with.

Two levels, on purpose
----------------------
**Level 1 is model-free.** APC-corrected mutual information between MSA
columns is the classical contact predictor, and it answers "is the signal in
the data" in milliseconds with no training run to confound it. If the signal
is not here, no architecture can find it.

**Level 2 uses the model the lab actually ships** -- `EvoformerStack`, at the
lab's own token convention, on a held-out split. This is the part
`test_synthetic_data_is_learnable.py` got wrong for months by building its own
small MLP: a learnability test that measures a model no learner runs is the
same failure as validating a library version no learner installs. If the
Evoformer cannot recover contacts the MI can, that is a finding about the
model, and this suite is where it would surface.

Thresholds are deliberately loose
---------------------------------
Level 2 asserts `prec@K > 3x base rate`, not a specific accuracy. A threshold
pinned to the number this machine happens to produce would fail on half the
hardware in the course, and tightening it would trade a real property for a
fragile one -- the reasoning `test_synthetic_data_is_learnable.py` documents
for its own loose `>25%`. What is being asserted is *the signal is reachable*,
not *the signal is reachable to three decimal places*.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "02_evoformer"))

from _srcload import Results                                    # noqa: E402
from evoformer import EvoformerConfig, EvoformerStack           # noqa: E402
from synthetic_msa import (SyntheticConfig,                     # noqa: E402
                           SyntheticContactDataset, eval_mask,
                           signal_strength)

SEEDS = (0, 1, 2)


# =============================================================================
# Level 1 -- is the signal in the DATA? (no model, milliseconds)
# =============================================================================


def test_signal_exists_in_data(r: Results) -> None:
    coupled = [
        signal_strength(SyntheticConfig(n_res=48, n_seq=64, coupling=1.0), s)
        for s in SEEDS
    ]
    base = float(np.mean([s["base_rate"] for s in coupled]))
    prec = float(np.mean([s["top_k_precision"] for s in coupled]))
    gap = float(np.mean([s["separation"] for s in coupled]))

    r.check(
        prec > 0.8,
        f"coupled MSAs: contacts recoverable by MI alone "
        f"(prec@K {prec:.3f} vs base rate {base:.3f})",
        "If mutual information cannot separate contacts from background, the "
        "generator is not planting coevolution and no model will find any.",
    )
    r.check(
        gap > 0.3,
        f"coupled MSAs: MI gap is large (contact - background = {gap:.3f} nats)",
    )


def test_uncoupled_data_is_unlearnable(r: Results) -> None:
    """THE COUNTEREXAMPLE. Must fail, or the suite above proves nothing."""
    un = [
        signal_strength(SyntheticConfig(n_res=48, n_seq=64, coupling=0.0), s)
        for s in SEEDS
    ]
    base = float(np.mean([s["base_rate"] for s in un]))
    prec = float(np.mean([s["top_k_precision"] for s in un]))
    per_seed = [round(s["top_k_precision"], 3) for s in un]

    r.check(
        prec < 4 * base,
        f"UNCOUPLED MSAs are NOT recoverable (prec@K {prec:.3f} vs base rate "
        f"{base:.3f}; per seed {per_seed})",
        "The counterexample is learnable, which means the generator is leaking "
        "the contact map through something other than coevolution. An earlier "
        "version leaked it through a SHARED mutation event for each pair -- "
        "when the event did not fire, both positions kept the query residue "
        "together, correlating them at coupling=0.0.",
    )


def test_depth_one_is_unlearnable(r: Results) -> None:
    """A single sequence has no statistics. This is WHY MSAs exist."""
    one = [
        signal_strength(SyntheticConfig(n_res=48, n_seq=1, coupling=1.0), s)
        for s in SEEDS
    ]
    base = float(np.mean([s["base_rate"] for s in one]))
    prec = float(np.mean([s["top_k_precision"] for s in one]))
    r.check(
        prec < 4 * base,
        f"depth-1 'alignments' are NOT recoverable (prec@K {prec:.3f} vs "
        f"base rate {base:.3f})",
        "Perfect coupling with one sequence still carries no information: "
        "coevolution is a property of a population, not a sequence.",
    )


# =============================================================================
# Level 2 -- can THE MODEL THE LAB SHIPS find it, on a held-out split?
# =============================================================================


def _precision_at_k(logits: torch.Tensor, contacts: torch.Tensor,
                    mask: torch.Tensor) -> float:
    """Mean precision@K over a batch, K = true contact count per chain."""
    out = []
    for b in range(logits.shape[0]):
        n = logits.shape[-1]
        tri = torch.triu(mask[b].bool() & ~torch.eye(n, dtype=bool), diagonal=1)
        scores = logits[b][tri]
        labels = contacts[b][tri]
        k = int(labels.sum().item())
        if k == 0:
            continue
        top = labels[torch.argsort(scores, descending=True)][:k]
        out.append(top.mean().item())
    return float(np.mean(out)) if out else 0.0


def _train_and_eval(coupling: float, seed: int, steps: int = 150) -> dict:
    """
    Train the lab's own EvoformerStack briefly and score a HELD-OUT split.

    Held out matters: contact maps are small enough to memorise, and
    memorisation proves the opposite of learning. The eval chains are
    generated from a different seed and never trained on.
    """
    torch.manual_seed(seed)
    cfg = SyntheticConfig(n_res=32, n_seq=32, n_contacts=16,
                          min_sep=6, coupling=coupling)
    train = SyntheticContactDataset(32, cfg, seed=seed)
    held_out = SyntheticContactDataset(16, cfg, seed=seed + 1000)

    model = EvoformerStack(EvoformerConfig(n_blocks=1))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scored = torch.from_numpy(eval_mask(cfg)).float()

    loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    model.train()
    step = 0
    while step < steps:
        for msa, contacts, mask in loader:
            if step >= steps:
                break
            logits = model(msa)
            # Loss only on scored pairs -- the near-diagonal band is excluded
            # so the model cannot bank score on |i-j| it can read off indices.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, contacts, weight=mask, reduction="sum"
            ) / mask.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

    model.eval()
    precs, base = [], []
    with torch.no_grad():
        for msa, contacts, mask in torch.utils.data.DataLoader(held_out,
                                                               batch_size=4):
            precs.append(_precision_at_k(model(msa), contacts, mask))
            n = contacts.shape[-1]
            tri = torch.triu(scored.bool() & ~torch.eye(n, dtype=bool), 1)
            base.append(contacts[:, tri].mean().item())
    return {"prec": float(np.mean(precs)), "base": float(np.mean(base))}


def test_model_learns_coupled_contacts(r: Results) -> None:
    got = _train_and_eval(coupling=1.0, seed=0)
    r.check(
        got["prec"] > 3 * got["base"],
        f"the lab's own EvoformerStack recovers coupled contacts on HELD-OUT "
        f"chains (prec@K {got['prec']:.3f} vs base rate {got['base']:.3f})",
        "The MI test says the signal is in the data. If the model cannot find "
        "it, that is a finding about the trunk, not the generator.",
    )


def test_model_fails_on_uncoupled_contacts(r: Results) -> None:
    """THE COUNTEREXAMPLE, at the model level."""
    got = _train_and_eval(coupling=0.0, seed=0)
    r.check(
        got["prec"] < 4 * got["base"],
        f"the same model does NOT recover UNCOUPLED contacts "
        f"(prec@K {got['prec']:.3f} vs base rate {got['base']:.3f})",
        "The model is scoring above chance on data with no signal, so it is "
        "reading something other than coevolution -- most likely a residual "
        "index shortcut that the eval mask was supposed to remove.",
    )


def main() -> int:
    r = Results("Synthetic MSAs: coevolution is present, and absent on purpose")
    test_signal_exists_in_data(r)
    test_uncoupled_data_is_unlearnable(r)
    test_depth_one_is_unlearnable(r)
    test_model_learns_coupled_contacts(r)
    test_model_fails_on_uncoupled_contacts(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
