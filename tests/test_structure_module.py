# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: SE(3) equivariance, guaranteed versus learned.

Run:
    uv run tests/test_structure_module.py

Why this suite exists
---------------------
A protein has no preferred position or orientation, so a structure module must
satisfy

    predict(R x + t) == R predict(x) + t

for every rotation R and translation t. Every way of getting this wrong
produces coordinates of the right shape, trains to a plausible loss, and
scores normally on a held-out split drawn from the same distribution -- which
is precisely the failure a held-out split cannot catch, because the test data
is oriented like the training data.

The three heads make the distinction measurable rather than rhetorical:

    ipa          SE(3)-invariant BY CONSTRUCTION (AlphaFold2, Algorithm 22)
    diffusion    standard attention on coordinates; AlphaFold3 dropped IPA and
                 learns the symmetry from augmentation instead
    mlp          no notion of frames at all -- the strawman

Three points, not two. With only `ipa` and `mlp` the result reads as
"correct versus broken" and teaches nothing; the middle arm is a real
architecture shipped by a real frontier model, and seeing it land six orders
of magnitude from the guarantee is the finding.

What is asserted, and what is deliberately not
-----------------------------------------------
Asserted: that IPA's equivariance is exact to float precision, that the
AF3-style head's is measurably not, that the gap between them is enormous, and
that augmentation shrinks the gap **without closing it**.

NOT asserted: that AF2's structure module is better than AF3's. It is not --
AF3 gave up the guarantee to gain generality, because IPA needs residue frames
built from N, CA and C atoms and ligands, ions and nucleic acids have no
backbone to build one from. This suite measures a symmetry, which is what it
can measure honestly on a CPU with no weights. Which architecture predicts
better structures is a claim about trained frontier models and belongs to the
papers.

These checks have been watched failing
--------------------------------------
    sabotage                               caught by
    -------------------------------------  ------------------------------
    IPA leaks a global coordinate          ipa-exact      (5.22e-02)
    AF3 head stops reading coordinates     af3-not-equiv  (3.42e-16)
    FAPE replaced by raw-coordinate RMSD   fape-invariant (0.44 vs 10.00)
    frames built as reflections            det = -1.000000

The third sabotage found a defect in this suite. The FAPE check originally
transformed prediction **and** truth together -- which a raw RMSD also
survives, because moving two point sets by the same rigid transform leaves
the distance between them unchanged. It passed on the sabotage, i.e. it was
vacuous. The check now transforms the **prediction alone**, which is the
property that actually distinguishes FAPE from RMSD, and the sabotage fails
it by a factor of twenty.

Why float64
-----------
IPA's error is float NOISE, and in float32 the noise floor (~1e-7) sits close
enough to a genuinely small-but-nonzero error to be ambiguous. In float64 the
separation is unmistakable: 1e-16 against 1e-02. Using the wider dtype here is
not precision theatre -- it is what makes "exact" distinguishable from "very
good".
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "04_structure_module"))

from _srcload import Results                                    # noqa: E402
from structure import (StructureConfig, StructureModule,        # noqa: E402
                       apply_frame, equivariance_error,
                       fape_loss, frames_from_backbone,
                       invert_frame, random_se3)

SEEDS = (0, 1, 2)


def _mean_err(head: str) -> float:
    return sum(equivariance_error(head, seed=s) for s in SEEDS) / len(SEEDS)


# =============================================================================
# 1. The three-way separation
# =============================================================================


def test_ipa_is_exactly_equivariant(r: Results) -> None:
    err = _mean_err("ipa")
    r.check(
        err < 1e-12,
        f"IPA is SE(3)-equivariant to float precision ({err:.2e})",
        "IPA scores point pairs by distances measured in shared frames, which "
        "a global transform cannot change. A nonzero error means points are "
        "being compared in global space somewhere, and the guarantee is gone.",
    )


def test_af3_style_head_is_not_equivariant(r: Results) -> None:
    """THE COUNTEREXAMPLE. Must fail the symmetry, or the check above is free."""
    err = _mean_err("diffusion")
    r.check(
        err > 1e-3,
        f"the AF3-style head is measurably NOT equivariant ({err:.2e})",
        "If this head were already equivariant, IPA's guarantee would be "
        "demonstrating nothing -- the test would pass on any implementation.",
    )


def test_mlp_head_is_badly_broken(r: Results) -> None:
    err = _mean_err("mlp")
    r.check(
        err > 1e-2,
        f"the MLP strawman is badly non-equivariant ({err:.2e})",
    )


def test_the_gap_is_enormous(r: Results) -> None:
    ipa, diff, mlp = _mean_err("ipa"), _mean_err("diffusion"), _mean_err("mlp")
    r.check(
        diff / max(ipa, 1e-300) > 1e8,
        f"guaranteed and learned differ by >1e8x "
        f"(ipa {ipa:.1e}, diffusion {diff:.1e})",
    )
    r.check(
        ipa < diff < mlp,
        f"the three heads order as expected: {ipa:.1e} < {diff:.1e} < {mlp:.1e}",
        "If the MLP beats the AF3-style head, the middle arm is not doing "
        "what it claims and the comparison is mislabelled.",
    )


# =============================================================================
# 2. Augmentation shrinks the gap without closing it
# =============================================================================


def test_augmentation_helps_but_does_not_guarantee(r: Results) -> None:
    """
    AlphaFold3 recovers the symmetry by randomly reorienting training
    examples. Does that actually work? Measured, not assumed.
    """
    def train(augment: bool, seed: int, steps: int = 120):
        torch.manual_seed(seed)
        cfg = StructureConfig(n_blocks=2)
        model = StructureModule(cfg, "diffusion")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        b, n = 4, 16
        s = torch.randn(b, n, cfg.c_s)
        z = torch.randn(b, n, n, cfg.c_z)
        t_true = torch.cumsum(
            torch.randn(b, n, 3) * 0.5 + torch.tensor([3.8, 0.0, 0.0]), dim=1)
        R_true = torch.eye(3).expand(b, n, 3, 3).contiguous()
        for i in range(steps):
            R0, t0, Rt, tt = R_true, t_true, R_true, t_true
            if augment:
                Rg, tg = random_se3(b, seed=10_000 + i)
                R0 = Rg.unsqueeze(1) @ R_true
                t0 = torch.einsum("bij,bnj->bni", Rg, t_true) + tg.unsqueeze(1)
                Rt, tt = R0, t0
            Rp, tp = model(s, z, R0, t0)
            loss = fape_loss(Rp, tp, Rt, tt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def err(model, seed):
        cfg = StructureConfig(n_blocks=2)
        torch.manual_seed(seed)
        b, n = 1, 16
        s = torch.randn(b, n, cfg.c_s)
        z = torch.randn(b, n, n, cfg.c_z)
        t0 = torch.cumsum(
            torch.randn(b, n, 3) * 0.5 + torch.tensor([3.8, 0.0, 0.0]), dim=1)
        R0 = torch.eye(3).expand(b, n, 3, 3).contiguous()
        with torch.no_grad():
            _, tp = model(s, z, R0, t0)
            Rg, tg = random_se3(b, seed=seed + 7)
            _, tpg = model(s, z, Rg.unsqueeze(1) @ R0,
                           torch.einsum("bij,bnj->bni", Rg, t0) + tg.unsqueeze(1))
            exp = torch.einsum("bij,bnj->bni", Rg, tp) + tg.unsqueeze(1)
            scale = (tp - tp.mean(1, keepdim=True)).norm(dim=-1).mean()
            return ((tpg - exp).norm(dim=-1).mean() / scale).item()

    plain = sum(err(train(False, s), s) for s in (0, 1)) / 2
    augmented = sum(err(train(True, s), s) for s in (0, 1)) / 2

    r.check(
        augmented < plain,
        f"augmentation SHRINKS the equivariance error "
        f"({plain:.2e} -> {augmented:.2e})",
        "AlphaFold3's whole strategy for recovering the symmetry is random "
        "rotation of training examples. If it does not help here, this "
        "folder's account of AF3 is wrong.",
    )
    r.check(
        augmented > 1e-6,
        f"but it does NOT reach a guarantee ({augmented:.2e}, vs IPA's ~1e-16)",
        "A learned symmetry is a statement about the training distribution. "
        "If augmentation made it exact, there would be no reason to prefer "
        "IPA and the architectural argument would collapse.",
    )


# =============================================================================
# 3. FAPE must be invariant, or the model learns an arbitrary orientation
# =============================================================================


def test_fape_is_invariant_under_global_transform(r: Results) -> None:
    torch.manual_seed(5)
    b, n = 2, 12
    t_true = torch.cumsum(
        torch.randn(b, n, 3, dtype=torch.float64) * 0.5
        + torch.tensor([3.8, 0.0, 0.0], dtype=torch.float64), dim=1)
    R_true = torch.eye(3, dtype=torch.float64).expand(b, n, 3, 3).contiguous()
    t_pred = t_true + torch.randn_like(t_true) * 0.3
    R_pred = R_true.clone()

    base = fape_loss(R_pred, t_pred, R_true, t_true).item()
    Rg, tg = random_se3(b, dtype=torch.float64, seed=11)

    def xf(R, t):
        return (Rg.unsqueeze(1) @ R,
                torch.einsum("bij,bnj->bni", Rg, t) + tg.unsqueeze(1))

    # Transform the PREDICTION ONLY. This is the property that matters and the
    # one that distinguishes FAPE from a plain RMSD.
    #
    # An earlier version of this check transformed prediction AND truth
    # together -- which raw RMSD also survives, because moving two point sets
    # by the same rigid transform leaves the distance between them alone. The
    # check passed on a sabotage that replaced FAPE with RMSD, i.e. it was
    # vacuous. Found by running that sabotage, not by reading the test.
    pred_moved = fape_loss(*xf(R_pred, t_pred), R_true, t_true).item()
    r.check(
        abs(base - pred_moved) < 1e-9,
        f"FAPE ignores a global SE(3) applied to the PREDICTION ALONE "
        f"({base:.6f} vs {pred_moved:.6f})",
        "A rotated-but-correct structure is correct. A loss that disagrees "
        "makes the model burn capacity learning an arbitrary orientation. "
        "Raw RMSD fails this; FAPE passes because both sides are expressed "
        "in their own frames before being compared.",
    )

    # The weaker property, kept because it is cheap and would catch a loss
    # that is somehow sensitive to absolute position.
    both_moved = fape_loss(*xf(R_pred, t_pred), *xf(R_true, t_true)).item()
    r.check(
        abs(base - both_moved) < 1e-9,
        f"FAPE is unchanged when BOTH are transformed ({both_moved:.6f})",
    )

    # And it must still be a real loss -- zero only when the structures match.
    perfect = fape_loss(R_true, t_true, R_true, t_true).item()
    r.check(
        perfect < base and perfect < 0.05,
        f"FAPE is near zero for a perfect prediction ({perfect:.4f} < "
        f"{base:.4f})",
        "An invariant that is invariant to everything is a constant.",
    )


# =============================================================================
# 4. Frames
# =============================================================================


def test_frames_are_orthonormal_and_equivariant(r: Results) -> None:
    torch.manual_seed(6)
    b, n = 2, 10
    ca = torch.cumsum(torch.randn(b, n, 3, dtype=torch.float64) * 0.5
                      + torch.tensor([3.8, 0.0, 0.0], dtype=torch.float64), dim=1)
    nx = ca + torch.randn(b, n, 3, dtype=torch.float64) * 0.1
    cx = ca + torch.randn(b, n, 3, dtype=torch.float64) * 0.1

    R, t = frames_from_backbone(nx, ca, cx)
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    orth = (R.transpose(-1, -2) @ R - eye).abs().max().item()
    det = torch.det(R)

    r.check(orth < 1e-10, f"frames are orthonormal (R^T R - I = {orth:.2e})")
    r.check(
        (det - 1.0).abs().max().item() < 1e-10,
        f"frames are ROTATIONS, not reflections (det = {det.mean():.6f})",
        "A reflection would mirror the structure -- chemically a different "
        "molecule.",
    )

    Rg, tg = random_se3(b, dtype=torch.float64, seed=13)
    def mv(x):
        return torch.einsum("bij,bnj->bni", Rg, x) + tg.unsqueeze(1)
    R2, t2 = frames_from_backbone(mv(nx), mv(ca), mv(cx))
    err = (R2 - Rg.unsqueeze(1) @ R).abs().max().item()
    r.check(err < 1e-10, f"frames are equivariant to a global SE(3) ({err:.2e})")


def test_frame_roundtrip(r: Results) -> None:
    torch.manual_seed(7)
    b, n, p = 2, 8, 5
    R, t = random_se3(b, dtype=torch.float64, seed=17)
    R = R.unsqueeze(1).expand(b, n, 3, 3).contiguous()
    t = t.unsqueeze(1).expand(b, n, 3).contiguous()
    pts = torch.randn(b, n, p, 3, dtype=torch.float64)
    back = invert_frame(R, t, apply_frame(R, t, pts))
    err = (back - pts).abs().max().item()
    r.check(err < 1e-10, f"apply_frame and invert_frame round-trip ({err:.2e})")


def main() -> int:
    r = Results("Structure module: a guaranteed symmetry vs a learned one")
    test_ipa_is_exactly_equivariant(r)
    test_af3_style_head_is_not_equivariant(r)
    test_mlp_head_is_badly_broken(r)
    test_the_gap_is_enormous(r)
    test_augmentation_helps_but_does_not_guarantee(r)
    test_fape_is_invariant_under_global_transform(r)
    test_frames_are_orthonormal_and_equivariant(r)
    test_frame_roundtrip(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
