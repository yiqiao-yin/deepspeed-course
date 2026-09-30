# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: the AlphaFold3 Pairformer trunk.

Run:
    uv run tests/test_pairformer.py

Why this suite exists SEPARATELY from test_evoformer.py
-------------------------------------------------------
`06_protein_folding/02_evoformer` and `03_pairformer` are two folders by
design -- their memory profiles genuinely differ, which is the same reason
`04_reward_model`, `05_dpo` and `07_online_dpo` are separate. The cost of that
decision is that the triangle operations are **written twice**, by hand, in
both folders.

So this suite re-asserts the shared properties independently rather than
trusting that `test_evoformer.py` covers them. A fix applied to one copy of
`TriangleMultiplicativeUpdate` and not the other is exactly the failure the
duplication invites, and it would be silent: both copies produce correct
shapes either way.

The claim this folder makes, and how it is checked
---------------------------------------------------
AlphaFold3 deleted the MSA representation from the trunk. The tempting
conclusion is "so the memory problem is solved". It is not, and the suite
pins both halves of that:

1. **The deletion is real.** `MSAModule.forward` returns only the pair
   representation, and no tensor allocated inside a `PairformerBlock` has an
   N_seq axis. Checked by watching every tensor the trunk actually produces,
   not by reading the signature.

2. **The asymptote is unchanged.** The triangle attention logits are
   byte-identical between the two architectures, and AF3's percentage saving
   *shrinks* as proteins get longer. If someone "optimises" the Pairformer
   into an architecture with a smaller exponent, this check fails and the
   README's central claim needs rewriting -- which is the right outcome.

These checks have been watched failing
--------------------------------------
    sabotage                                  caught by
    ----------------------------------------  ------------------------------
    MSA smuggled back into a trunk block      no-N_seq-axis (108 tensors)
    MSAModule returns (z, m)                  returns-a-single-tensor
    AF3 claimed to halve the cubic term       triangle-cost-identical
    MSA row attention logits omitted          MSA-row-attn-dominates

The last one reproduces a bug that actually shipped in this folder: the first
version of `trunk_activation_table` left out AF2's MSA row attention logits --
the largest MSA-side tensor -- and reported AF3 saving 1.2% where the honest
figure is 24%. The check exists because the omission already happened once.

The second sabotage found a defect in this suite rather than in the code. When
`MSAModule` returned a tuple, the follow-up shape check raised
`AttributeError` instead of failing, so CI would have gone red naming a type
error rather than the finding -- the same shape as the `beats_chance()` bug in
`test_synthetic_data_is_learnable.py`. It is now guarded on the check above it.

A note on what is NOT asserted
-------------------------------
Nothing here claims AF3 is more or less accurate than AF2. This suite measures
activation shapes and symmetries, which is what it can measure honestly on a
CPU with no weights. Accuracy is a claim about trained frontier models and
belongs to the papers, not to a 100k-parameter teaching trunk.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "03_pairformer"))

from _srcload import Results                                   # noqa: E402
from pairformer import (AttentionPairBias, MSAModule,          # noqa: E402
                        PairformerBlock, PairformerConfig,
                        PairformerStack, TriangleAttention,
                        TriangleMultiplicativeUpdate,
                        trunk_activation_table)
from synthetic_msa import make_batch                           # noqa: E402

EQUIV_TOL = 1e-5

# Distinct on purpose: a tensor carrying 24 is carrying sequences, and a
# tensor carrying 16 is carrying residues. Equal values would make the
# N_seq-axis check below unable to tell them apart.
N_RES, N_SEQ = 16, 24


def _model(seed: int = 0, n_blocks: int = 1, broken: bool = False):
    torch.manual_seed(seed)
    return PairformerStack(
        PairformerConfig(n_blocks=n_blocks), broken_no_third_index=broken
    ).eval()


# =============================================================================
# 1. The deletion is real
# =============================================================================


def test_msa_module_returns_only_pair(r: Results) -> None:
    torch.manual_seed(0)
    cfg = PairformerConfig()
    mod = MSAModule(cfg).eval()
    m = torch.randn(1, N_SEQ, N_RES, cfg.c_m)
    z = torch.randn(1, N_RES, N_RES, cfg.c_z)
    with torch.no_grad():
        out = mod(m, z)

    single = r.check(
        isinstance(out, torch.Tensor),
        "MSAModule returns a single tensor, not (m, z)",
        "If it returns the MSA representation too, the trunk can keep using "
        "it and the AF3 deletion is cosmetic.",
    )
    # Guarded on the check above. Without the guard, a module that returns a
    # tuple makes the next line raise AttributeError instead of failing --
    # and CI then goes red naming a type error rather than the finding. That
    # is exactly how `beats_chance()` in test_synthetic_data_is_learnable.py
    # shipped unable to report the bug it was written for.
    if single:
        r.check(
            tuple(out.shape) == (1, N_RES, N_RES, cfg.c_z),
            f"MSAModule returns the PAIR representation {tuple(out.shape)}",
        )
    else:
        r.check(False, "MSAModule returns the PAIR representation",
                f"got {type(out).__name__}, cannot check its shape")


def test_trunk_allocates_no_sequence_axis(r: Results) -> None:
    """
    The structural claim, checked by watching every tensor the trunk produces.

    A signature can promise the MSA is gone while a block quietly rebuilds
    something N_seq-shaped inside. Hooks see what actually gets allocated.
    """
    model = _model(n_blocks=2)
    msa = make_batch(batch_size=1, n_res=N_RES, n_seq=N_SEQ, seed=1)["msa"]

    seen_in_trunk: list[tuple] = []
    seen_in_msa_module: list[tuple] = []

    def record(store):
        def hook(_mod, _inp, out):
            for t in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(t, torch.Tensor):
                    store.append(tuple(t.shape))
        return hook

    handles = []
    for blk in model.blocks:
        for sub in blk.modules():
            handles.append(sub.register_forward_hook(record(seen_in_trunk)))
    for sub in model.msa_module.modules():
        handles.append(sub.register_forward_hook(record(seen_in_msa_module)))

    with torch.no_grad():
        model(msa)
    for h in handles:
        h.remove()

    trunk_with_seq = [s for s in seen_in_trunk if N_SEQ in s]
    msa_with_seq = [s for s in seen_in_msa_module if N_SEQ in s]

    r.check(
        not trunk_with_seq,
        f"no tensor in the Pairformer trunk has an N_seq axis "
        f"({len(seen_in_trunk)} tensors checked)",
        f"found {trunk_with_seq[:4]} -- the MSA representation is still in "
        "the trunk, so this is not the AlphaFold3 architecture.",
    )
    r.check(
        bool(msa_with_seq),
        f"but the MSA module DOES see sequences ({len(msa_with_seq)} tensors)",
        "If nothing anywhere carries an N_seq axis, the MSA is not being read "
        "at all and the check above is vacuous -- the model would be folding "
        "from a single sequence.",
    )


def test_attention_pair_bias_is_single_row(r: Results) -> None:
    """AF3's trunk attention acts on one row, not N_seq rows."""
    torch.manual_seed(0)
    cfg = PairformerConfig()
    att = AttentionPairBias(cfg.c_s, cfg.c_z, cfg.c_hidden, cfg.n_heads).eval()
    s = torch.randn(1, N_RES, cfg.c_s)
    z = torch.randn(1, N_RES, N_RES, cfg.c_z)
    with torch.no_grad():
        out = att(s, z)
    r.check(
        tuple(out.shape) == (1, N_RES, cfg.c_s),
        f"AttentionPairBias maps [b, N_res, c_s] -> {tuple(out.shape)}",
        "AF2's equivalent operates on [b, N_seq, N_res, c_m]. That difference "
        "is the architecture.",
    )


# =============================================================================
# 2. The asymptote is unchanged -- the claim the folder rests on
# =============================================================================


def test_triangle_cost_is_identical_between_architectures(r: Results) -> None:
    t = trunk_activation_table(n_res=384, n_seq=128)
    tri = t["triangle attention logits"]
    r.check(
        tri["af2"] == tri["af3"] and tri["af3"] > 0,
        f"triangle attention logits are IDENTICAL in AF2 and AF3 "
        f"({tri['af3']:.1f} MB)",
        "If AF3's triangle cost differs, the README's central claim -- that "
        "the deletion removes a constant and not the asymptote -- is wrong.",
    )
    r.check(
        t["MSA representation"]["af3"] == 0.0
        and t["MSA row attention logits"]["af3"] == 0.0,
        "AF3 carries neither the MSA representation nor its attention logits",
    )
    r.check(
        t["MSA row attention logits"]["af2"]
        > 10 * t["MSA representation"]["af2"],
        "AF2's MSA ROW ATTENTION dominates its MSA representation "
        f"({t['MSA row attention logits']['af2']:.1f} MB vs "
        f"{t['MSA representation']['af2']:.1f} MB)",
        "An early version of the table omitted the attention logits entirely "
        "and reported AF3 saving 1.2% instead of 24%. The largest MSA-side "
        "tensor is the one that is easiest to forget.",
    )


def test_af3_saving_shrinks_with_length(r: Results) -> None:
    """
    The lesson: a constant-factor architectural saving loses to an asymptote.

    If this ever stops shrinking, either the table is wrong or someone has
    changed the exponent -- both worth failing a build over.
    """
    savings, cubic_share = [], []
    for n in (128, 256, 512, 1024):
        t = trunk_activation_table(n_res=n, n_seq=128)
        af2 = sum(v["af2"] for v in t.values())
        af3 = sum(v["af3"] for v in t.values())
        savings.append(100 * (af2 - af3) / af2)
        cubic_share.append(100 * t["triangle attention logits"]["af3"] / af3)

    r.check(
        all(b < a for a, b in zip(savings, savings[1:])),
        f"AF3's percentage saving SHRINKS with length "
        f"{[round(x, 1) for x in savings]}",
        "The MSA terms are quadratic at worst; the triangle term is cubic. "
        "The saving must decay.",
    )
    r.check(
        all(b > a for a, b in zip(cubic_share, cubic_share[1:]))
        and cubic_share[-1] > 90,
        f"the cubic term takes over AF3's budget "
        f"{[round(x, 1) for x in cubic_share]}",
    )


def test_triangle_attention_still_builds_a_cubic_tensor(r: Results) -> None:
    """Measured from the real einsum in THIS folder's copy, not the table."""
    torch.manual_seed(0)
    att = TriangleAttention(16, 8, 4, "starting").eval()
    seen = {}
    real_softmax = torch.softmax
    for n in (8, 16):
        captured = []

        def spy(x, dim=None, **kw):
            captured.append(x.numel())
            return real_softmax(x, dim=dim, **kw)

        torch.softmax = spy
        try:
            with torch.no_grad():
                att(torch.randn(1, n, n, 16))
        finally:
            torch.softmax = real_softmax
        seen[n] = max(captured)

    got = seen[16] / seen[8]
    r.check(
        abs(got - 8.0) < 0.01,
        f"AF3's triangle attention is still CUBIC ({got:.2f}x per doubling)",
        f"measured {seen}",
    )


# =============================================================================
# 3. Shared properties, re-asserted because the code is duplicated
# =============================================================================


def test_residue_permutation_equivariance(r: Results) -> None:
    model = _model(seed=2)
    msa = make_batch(batch_size=1, n_res=N_RES, n_seq=N_SEQ, seed=2)["msa"]
    with torch.no_grad():
        base = model(msa)
        perm = torch.randperm(N_RES)
        err = (model(msa[:, :, perm])
               - base[:, perm][:, :, perm]).abs().max().item()
    r.check(err < EQUIV_TOL,
            f"contact map is equivariant to residue permutation ({err:.2e})")
    r.check(base.std().item() > 1e-3,
            f"contact map actually varies with input (std {base.std():.4f})")


def test_msa_row_permutation(r: Results) -> None:
    model = _model(seed=3)
    msa = make_batch(batch_size=1, n_res=N_RES, n_seq=N_SEQ, seed=3)["msa"]
    with torch.no_grad():
        base = model(msa)
        keep = torch.cat([torch.zeros(1, dtype=torch.long),
                          1 + torch.randperm(N_SEQ - 1)])
        inv = (model(msa[:, keep, :]) - base).abs().max().item()
        swap = torch.arange(N_SEQ)
        swap[0], swap[1] = 1, 0
        delta = (model(msa[:, swap, :]) - base).abs().max().item()
    r.check(inv < EQUIV_TOL,
            f"invariant to permuting NON-QUERY MSA rows ({inv:.2e})")
    r.check(delta > 1e-3,
            f"swapping the QUERY row does change the output ({delta:.2e})",
            "Otherwise the MSA module is being ignored and the invariance "
            "above is vacuous.")


def test_triangle_closure(r: Results) -> None:
    """The duplicated copy needs its own closure check. See the docstring."""
    torch.manual_seed(4)
    n, c_z, c_hidden = 8, 16, 8
    i, j, k = 0, 1, 5
    good = TriangleMultiplicativeUpdate(c_z, c_hidden, "outgoing").eval()
    bad = TriangleMultiplicativeUpdate(c_z, c_hidden, "outgoing",
                                       broken_no_third_index=True).eval()
    bad.load_state_dict(good.state_dict())

    z = torch.zeros(1, n, n, c_z)
    z_ev = z.clone()
    # Varied across channels: a constant vector normalises to zeros under
    # norm_in and would measure 0.0 for BOTH implementations.
    z_ev[0, i, k] = torch.randn(c_z)
    z_ev[0, j, k] = torch.randn(c_z)

    with torch.no_grad():
        d_good = (good(z_ev)[0, i, j] - good(z)[0, i, j]).abs().max().item()
        d_bad = (bad(z_ev)[0, i, j] - bad(z)[0, i, j]).abs().max().item()

    r.check(d_good > 1e-3,
            f"evidence on (i,k) and (j,k) reaches (i,j) ({d_good:.3e})")
    r.check(d_bad < 1e-9,
            f"the third-index-free variant does NOT propagate ({d_bad:.3e})")


def test_contact_map_symmetric(r: Results) -> None:
    model = _model(seed=5)
    msa = make_batch(batch_size=2, n_res=N_RES, n_seq=N_SEQ, seed=5)["msa"]
    with torch.no_grad():
        logits = model(msa)
    asym = (logits - logits.transpose(-1, -2)).abs().max().item()
    r.check(asym < 1e-6, f"contact logits are symmetric ({asym:.2e})")


def test_interchangeable_with_evoformer(r: Results) -> None:
    """
    Same inputs, same output shape as 02_evoformer's stack.

    The two folders share a training script shape, data and metrics on
    purpose: a comparison between architectures is only meaningful if
    everything except the architecture is held fixed.
    """
    model = _model(seed=6)
    batch = make_batch(batch_size=2, n_res=N_RES, n_seq=N_SEQ, seed=6)
    with torch.no_grad():
        out = model(batch["msa"])
    r.check(
        tuple(out.shape) == tuple(batch["contacts"].shape),
        f"contact logits {tuple(out.shape)} match the label shape "
        f"{tuple(batch['contacts'].shape)}",
    )


def main() -> int:
    r = Results("Pairformer: the MSA deletion, and the asymptote that survived")
    test_msa_module_returns_only_pair(r)
    test_trunk_allocates_no_sequence_axis(r)
    test_attention_pair_bias_is_single_row(r)
    test_triangle_cost_is_identical_between_architectures(r)
    test_af3_saving_shrinks_with_length(r)
    test_triangle_attention_still_builds_a_cubic_tensor(r)
    test_residue_permutation_equivariance(r)
    test_msa_row_permutation(r)
    test_triangle_closure(r)
    test_contact_map_symmetric(r)
    test_interchangeable_with_evoformer(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
