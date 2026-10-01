# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: the AlphaFold2 Evoformer trunk.

Run:
    uv run tests/test_evoformer.py

Why this suite exists
---------------------
Every failure mode of a triangle operation produces a tensor of the right
shape. `z` goes in as ``[B, N, N, c_z]`` and comes out as ``[B, N, N, c_z]``
whether or not the operation has a triangle in it, whether or not the model
reads residue order instead of residue content, and whether or not the MSA is
being used at all. CONTRIBUTING.md section 7 is explicit: assert properties,
not shapes.

So every property below is paired with a counterexample where one exists. It
is not enough that the correct triangular update propagates a constraint
through a third residue -- `TriangleMultiplicativeUpdate(...,
broken_no_third_index=True)` must be shown NOT to, or the check would pass on
an implementation that ignores its input.

The five properties, and what breaks if each is wrong
-----------------------------------------------------
1. **Residue-permutation equivariance.** Relabel the residues; the contact map
   must permute identically. A model that keys on position index rather than
   residue content scores well on a fixed ordering and is worthless on a new
   protein. `02_intermediate/04_groupwise_ranking` shipped precisely this bug
   -- a GSF with permutation error 1.5e-01 that was reading candidate order,
   which at training time is label order -- and only the property test caught
   it.

2. **Non-query MSA row permutation invariance.** The order homologs arrive in
   carries no information, and `OuterProductMean` averages it away. Asserted
   in both directions: permuting rows 1.. must change nothing, and permuting
   the QUERY row must change something, or the first half is satisfied by a
   model that ignores the MSA entirely.

3. **Triangle closure.** Evidence on edges (i, k) and (j, k) must reach edge
   (i, j) -- that is the entire purpose of the operation, and the whole reason
   the architecture can reason about geometry.

4. **Contact symmetry.** A contact is a property of the unordered pair, so
   logits must be symmetric by construction rather than approximately so after
   training.

5. **The cubic exponent.** The triangle attention logit tensor must grow as
   N^3 and the pair representation as N^2. This is the claim the whole DeepSpeed
   argument rests on: it is why ZeRO does not help and why
   `DS4Sci_EvoformerAttention` exists. Asserted as a measured exponent rather
   than quoted from the docstring, because a docstring cannot regress.

These checks have been watched failing
--------------------------------------
Per CLAUDE.md, a check that has never rejected bad input is not a check. Each
of these was run against a deliberately sabotaged trunk before being trusted:

    sabotage                                    caught by
    ------------------------------------------  --------------------------------
    output leaks residue POSITION into logits   equivariance      (err 1.30e+01)
    OuterProductMean weighted by row order      row invariance    (err 1.49e-02)
    "correct" triangle op loses its third index closure           (delta 0.0)
    memory table claims quadratic, not cubic    cubic exponent    (4.0x not 8.0x)
    contact head not symmetrised                symmetry          (asym 2.87e+00)

One of those sabotages had to be written twice, which is worth recording. The
first attempt at the row-order one scaled each MSA row by a position-dependent
constant *before* `OuterProductMean`'s LayerNorm -- and LayerNorm normalises a
per-row scale straight back out, so the invariance survived and the test looked
blind. The sabotage was wrong, not the check. Applying the same weighting after
the projection broke the invariance immediately. **A sabotage that fails to
break anything is evidence about the sabotage first.**

A note on tolerances
--------------------
Equivariance is asserted at 1e-5 rather than exactly. These are float32
reductions over different memory layouts, so bitwise equality is the wrong
test -- the same mistake `test_reward_model.py` documents for shift-invariance.
The counterexample checks use a margin far above that floor, so the tolerance
is not doing any work to make a failing implementation pass.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "02_evoformer"))

from _srcload import Results                              # noqa: E402
from evoformer import (EvoformerConfig, EvoformerStack,   # noqa: E402
                       TriangleAttention,
                       TriangleMultiplicativeUpdate, memory_table)
from synthetic_msa import make_batch                      # noqa: E402

EQUIV_TOL = 1e-5


def _model(seed: int = 0, n_blocks: int = 1, broken: bool = False) -> EvoformerStack:
    torch.manual_seed(seed)
    cfg = EvoformerConfig(n_blocks=n_blocks)
    return EvoformerStack(cfg, broken_no_third_index=broken).eval()


# =============================================================================
# 1. Residue permutation equivariance
# =============================================================================


def test_residue_permutation_equivariance(r: Results) -> None:
    model = _model()
    msa = make_batch(batch_size=1, n_res=16, n_seq=24, seed=1)["msa"]

    with torch.no_grad():
        base = model(msa)
        perm = torch.randperm(msa.shape[-1])
        got = model(msa[:, :, perm])
        want = base[:, perm][:, :, perm]

    err = (got - want).abs().max().item()
    r.check(
        err < EQUIV_TOL,
        f"contact map is equivariant to residue permutation (err {err:.2e})",
        "The model is reading residue ORDER, not residue content. At training "
        "time residue order is label order, so this scores well and "
        "generalises to nothing.",
    )

    # Without this, a constant function passes the check above for free.
    spread = base.std().item()
    r.check(
        spread > 1e-3,
        f"contact map actually varies with input (std {spread:.4f})",
        "A constant output is trivially equivariant. If this fails, the "
        "equivariance check above is proving nothing.",
    )


# =============================================================================
# 2. MSA row permutation -- invariant for homologs, NOT for the query
# =============================================================================


def test_msa_row_permutation(r: Results) -> None:
    model = _model()
    msa = make_batch(batch_size=1, n_res=16, n_seq=24, seed=2)["msa"]
    n_seq = msa.shape[1]

    with torch.no_grad():
        base = model(msa)

        # Rows 1.. are homologs: order carries no information.
        keep_query = torch.cat(
            [torch.zeros(1, dtype=torch.long), 1 + torch.randperm(n_seq - 1)]
        )
        shuffled = model(msa[:, keep_query, :])
        invariance_err = (shuffled - base).abs().max().item()

        # Row 0 is the sequence being folded. Swapping it in is a DIFFERENT
        # problem, and the output must say so.
        swap_query = torch.arange(n_seq)
        swap_query[0], swap_query[1] = 1, 0
        swapped = model(msa[:, swap_query, :])
        query_delta = (swapped - base).abs().max().item()

    r.check(
        invariance_err < EQUIV_TOL,
        f"invariant to permuting NON-QUERY MSA rows (err {invariance_err:.2e})",
        "OuterProductMean averages over sequences, so homolog order cannot "
        "matter. If this fails, something downstream of the MSA is "
        "order-dependent.",
    )
    r.check(
        query_delta > 1e-3,
        f"swapping the QUERY row does change the output (delta {query_delta:.2e})",
        "If moving the query changes nothing, the model is ignoring the MSA "
        "and the invariance check above is vacuous.",
    )


# =============================================================================
# 3. Triangle closure, with the permanent counterexample
# =============================================================================


def test_triangle_closure(r: Results) -> None:
    torch.manual_seed(3)
    n, c_z, c_hidden = 8, 16, 8
    i, j, k = 0, 1, 5

    good = TriangleMultiplicativeUpdate(c_z, c_hidden, "outgoing").eval()
    bad = TriangleMultiplicativeUpdate(
        c_z, c_hidden, "outgoing", broken_no_third_index=True
    ).eval()
    bad.load_state_dict(good.state_dict())      # identical weights

    z = torch.zeros(1, n, n, c_z)
    # Evidence must vary across channels: norm_in is a LayerNorm over the
    # channel axis, and a constant vector normalises to zeros. Planting
    # constants here measures 0.0 for BOTH implementations and proves nothing.
    z_ev = z.clone()
    z_ev[0, i, k] = torch.randn(c_z)
    z_ev[0, j, k] = torch.randn(c_z)

    with torch.no_grad():
        d_good = (good(z_ev)[0, i, j] - good(z)[0, i, j]).abs().max().item()
        d_bad = (bad(z_ev)[0, i, j] - bad(z)[0, i, j]).abs().max().item()
        shapes_match = good(z_ev).shape == bad(z_ev).shape

    r.check(
        d_good > 1e-3,
        f"evidence on (i,k) and (j,k) reaches (i,j) (delta {d_good:.3e})",
        "sum_k a_ik * b_jk is not propagating through the third residue. The "
        "operation has no triangle in it and the trunk cannot reason about "
        "geometry.",
    )
    r.check(
        d_bad < 1e-9,
        f"the third-index-free variant does NOT propagate (delta {d_bad:.3e})",
        "The counterexample is leaking a path from (i,k)/(j,k) to (i,j). If "
        "it propagates too, the check above distinguishes nothing.",
    )
    r.check(
        shapes_match,
        "correct and broken variants produce IDENTICAL shapes",
        "This is the point of the whole suite: a shape assertion cannot tell "
        "these two apart.",
    )


def test_outgoing_and_incoming_differ(r: Results) -> None:
    """Both directions exist because they are different. Pin that."""
    torch.manual_seed(4)
    n, c_z, c_hidden = 8, 16, 8
    out = TriangleMultiplicativeUpdate(c_z, c_hidden, "outgoing").eval()
    inc = TriangleMultiplicativeUpdate(c_z, c_hidden, "incoming").eval()
    inc.load_state_dict(out.state_dict())

    # An asymmetric pair representation, or the two directions coincide.
    z = torch.randn(1, n, n, c_z)
    with torch.no_grad():
        delta = (out(z) - inc(z)).abs().max().item()

    r.check(
        delta > 1e-3,
        f"outgoing and incoming are different operations (delta {delta:.3e})",
        "With the same weights on an asymmetric input these must disagree. If "
        "they agree, one direction's einsum is wrong and half of every block "
        "is redundant.",
    )


# =============================================================================
# 4. Contact symmetry
# =============================================================================


def test_contact_map_symmetric(r: Results) -> None:
    model = _model(seed=5)
    msa = make_batch(batch_size=2, n_res=16, n_seq=16, seed=5)["msa"]
    with torch.no_grad():
        logits = model(msa)
    asym = (logits - logits.transpose(-1, -2)).abs().max().item()
    r.check(
        asym < 1e-6,
        f"contact logits are symmetric by construction (asym {asym:.2e})",
        "A contact is a property of the unordered pair {i, j}. An asymmetric "
        "head can disagree with itself about the same contact.",
    )


# =============================================================================
# 5. The cubic exponent -- the claim the DeepSpeed argument rests on
# =============================================================================


def test_memory_scaling_exponents(r: Results) -> None:
    rows = {row["n_res"]: row for row in memory_table((128, 256, 512, 1024))}

    pair_ratios, logit_ratios = [], []
    for a, b in ((128, 256), (256, 512), (512, 1024)):
        pair_ratios.append(rows[b]["pair_mb"] / rows[a]["pair_mb"])
        logit_ratios.append(rows[b]["logits_mb"] / rows[a]["logits_mb"])

    r.check(
        all(abs(x - 4.0) < 0.01 for x in pair_ratios),
        f"pair representation is QUADRATIC: 4.0x per doubling {pair_ratios}",
    )
    r.check(
        all(abs(x - 8.0) < 0.01 for x in logit_ratios),
        f"triangle attention logits are CUBIC: 8.0x per doubling {logit_ratios}",
        "If this is not 8x, the memory argument for DS4Sci_EvoformerAttention "
        "in the README is wrong.",
    )

    # And the gap widens -- which is why this is a wall and not a constant.
    ratios = [rows[n]["ratio"] for n in (128, 256, 512, 1024)]
    r.check(
        all(b > a for a, b in zip(ratios, ratios[1:])),
        f"logits/pair ratio grows with length {[round(x, 1) for x in ratios]}",
    )


def test_triangle_attention_really_is_cubic(r: Results) -> None:
    """
    The exponent above is arithmetic. This checks the implementation actually
    builds the tensor that arithmetic describes, by counting elements in the
    real einsum rather than trusting the formula.
    """
    torch.manual_seed(6)
    c_z, c_hidden, n_heads = 16, 8, 4
    att = TriangleAttention(c_z, c_hidden, n_heads, "starting").eval()

    seen: dict[int, int] = {}
    for n in (8, 16):
        z = torch.randn(1, n, n, c_z)
        captured = []

        real_softmax = torch.softmax

        def spy(x, dim=None, **kw):
            captured.append(x.numel())
            return real_softmax(x, dim=dim, **kw)

        torch.softmax = spy
        try:
            with torch.no_grad():
                att(z)
        finally:
            torch.softmax = real_softmax
        seen[n] = max(captured)

    got = seen[16] / seen[8]
    r.check(
        abs(got - 8.0) < 0.01,
        f"the attention logits actually built scale 8x per doubling ({got:.2f}x)",
        f"Measured {seen}. The implementation is not materialising an "
        "N^3 tensor, so either it is already tiled or the shapes are wrong -- "
        "either way the README's memory claim does not describe this code.",
    )


# =============================================================================
# 6. The broken variant still trains -- which is why it is dangerous
# =============================================================================


def test_broken_variant_is_silently_plausible(r: Results) -> None:
    good = _model(seed=7, broken=False)
    bad = _model(seed=7, broken=True)
    msa = make_batch(batch_size=2, n_res=16, n_seq=16, seed=7)["msa"]

    with torch.no_grad():
        g, b = good(msa), bad(msa)

    r.check(g.shape == b.shape, "broken model: same output shape")
    r.check(
        torch.isfinite(b).all().item(),
        "broken model: finite, plausible logits",
        "The counterexample must look healthy. One that produced NaNs would "
        "be caught by anything and would not represent the real failure mode.",
    )
    r.check(
        (g - b).abs().max().item() > 1e-3,
        "but the two models genuinely disagree",
        "If identical, `broken_no_third_index` is not wired through the stack "
        "and every closure check above is running on the same code twice.",
    )


def main() -> int:
    r = Results("Evoformer: triangle operations, symmetries, and the memory wall")
    test_residue_permutation_equivariance(r)
    test_msa_row_permutation(r)
    test_triangle_closure(r)
    test_outgoing_and_incoming_differ(r)
    test_contact_map_symmetric(r)
    test_memory_scaling_exponents(r)
    test_triangle_attention_really_is_cubic(r)
    test_broken_variant_is_silently_plausible(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
