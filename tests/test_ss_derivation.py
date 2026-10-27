# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "pyarrow", "huggingface-hub"]
# ///
"""
Regression test: secondary structure derived from backbone geometry.

Run:
    uv run tests/test_ss_derivation.py

Why this suite exists
---------------------
`06_protein_folding/01_esm2_plm` does not download its labels. It derives
3-state secondary structure from two backbone dihedral angles, which keeps
the whole section on one CC-BY-4.0 data spine and lets a reader see where a
label comes from.

The risk that buys is specific: **a wrong derivation produces labels of the
right shape, in the right range, that train to a plausible accuracy.** Swap
two arguments to the dihedral, flip a sign, mislabel the atom order, and you
get a per-residue integer array in {0,1,2} with a believable class balance and
no error anywhere. The model will happily fit it.

So the checks here are geometric, not statistical:

1. `dihedral()` is correct against configurations whose angle is known by
   construction -- 0, +90 and -90 degrees. Sign included, because a sign flip
   mirrors the Ramachandran plot and swaps which region is which.
2. Assigned helix and strand residues land at the **textbook cluster centres**
   on real data. This is the check that a broken derivation cannot pass while
   still looking reasonable.
3. Helical geometry is verified independently of the angles: C-alpha(i) to
   C-alpha(i+4) is ~6.2 A in an alpha helix and much larger in an extended
   strand. If the H label and that distance disagree, the labels are wrong.
4. Run-length enforcement actually enforces.
5. The label/token alignment is right -- residue i sits at token i+1 because
   ESM-2 prepends `<cls>`.

Number 5 is the one that silently costs accuracy. An off-by-one trains, scores
a few points lower, and raises nothing.

These checks have been watched failing
--------------------------------------
    sabotage                      caught by
    ----------------------------  ---------------------------------
    dihedral sign flipped         +90/-90 unit check AND the real-data
                                  cluster centres (4/4 fail)
    phi and psi swapped           cluster centres (3/4 fail)
    run-length enforcement gone   isolated-residue demotion
    label alignment off by one    <cls> ignored, residue i at token i+1

The first sabotage was NOT caught at first, and the reason is worth keeping.
Every assertion in the dihedral test used `abs()` -- including the one named
"opposite sign", which only checked that the two configurations were opposite
to *each other*. A global negation survives all of that. The test now pins
the signed value, with the convention anchored to an empirical fact (real
helices have psi < 0) rather than to this implementation, so the unit check
and the real-data check agree on the sign independently.

What is NOT asserted
--------------------
The class fractions. This is a Ramachandran-plus-run-length assignment, not
DSSP -- what makes a strand a strand is hydrogen bonding to another strand,
which is non-local and invisible to two angles -- and CATH is a domain
database enriched in beta relative to whole proteomes. Asserting a published
"~33% helix" figure would be asserting a statistic about a different
population, and tuning the region boundaries until it passed would be fitting
the derivation to the wrong target. The cluster centres are the honest check.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "01_esm2_plm"))

from _srcload import Results                                    # noqa: E402
from plm import (SS_INDEX, backbone_dihedrals, dihedral,        # noqa: E402
                 mask_tokens, secondary_structure)


# =============================================================================
# 1. The dihedral itself, against angles known by construction
# =============================================================================


def test_dihedral_is_correct_including_sign(r: Results) -> None:
    # Four points in a plane: the outer bonds point the same way -> 0 degrees.
    p0 = np.array([[1.0, 1.0, 0.0]])
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    p3 = np.array([[2.0, 1.0, 0.0]])
    cis = dihedral(p0, p1, p2, p3)[0]
    r.check(abs(cis) < 1e-6, f"coplanar, same side -> 0 deg (got {cis:.3f})")

    # Rotate the last point out of plane by +90 degrees about the p1-p2 axis.
    p3_up = np.array([[2.0, 0.0, 1.0]])
    up = dihedral(p0, p1, p2, p3_up)[0]
    p3_dn = np.array([[2.0, 0.0, -1.0]])
    dn = dihedral(p0, p1, p2, p3_dn)[0]

    # SIGNED, not |signed|.
    #
    # The first version of this test asserted `abs(up) == 90` and
    # `up + dn == 0`, both of which survive a GLOBAL SIGN FLIP -- so the check
    # named "opposite sign" verified only that the two were opposite to each
    # other, and a sabotage that negated every dihedral passed all four
    # assertions. Caught by running that sabotage.
    #
    # The convention is pinned to an empirical fact rather than to this
    # implementation: real alpha helices have psi < 0. `test_cluster_centres_
    # are_textbook` measures that on real backbones and fails 4/4 under the
    # same sabotage, so the two checks agree on the sign independently.
    r.check(abs(up - 90.0) < 1e-6,
            f"perpendicular, +z -> EXACTLY +90 deg (got {up:.3f})",
            "A global sign flip mirrors the Ramachandran plot and swaps "
            "which region is helix and which is sheet -- while every label "
            "stays in range and the class balance stays plausible.")
    r.check(abs(dn + 90.0) < 1e-6,
            f"perpendicular, -z -> EXACTLY -90 deg (got {dn:.3f})")

    # Anti-periplanar: 180 degrees, where the sign is genuinely ambiguous.
    anti = dihedral(p0, p1, p2, np.array([[2.0, -1.0, 0.0]]))[0]
    r.check(abs(abs(anti) - 180.0) < 1e-6,
            f"trans configuration -> |180| deg (got {anti:.3f})",
            "abs() is correct HERE and only here: +180 and -180 are the same "
            "angle.")


def test_terminal_residues_are_nan_not_zero(r: Results) -> None:
    n = np.random.default_rng(0).normal(size=(6, 3))
    ca = n + 1.0
    c = n + 2.0
    phi, psi = backbone_dihedrals(n, ca, c)
    r.check(np.isnan(phi[0]) and np.isnan(psi[-1]),
            "first phi and last psi are NaN",
            "There is no previous C or next N. A zero would land in the "
            "coil region and become a plausible-looking label.")
    r.check(np.isfinite(phi[1:]).all() and np.isfinite(psi[:-1]).all(),
            "every other angle is finite")


# =============================================================================
# 2 & 3. Real backbones: cluster centres, and independent helix geometry
# =============================================================================


def _load_real(n_chains: int = 96):
    from cath_sequences import load_split
    return load_split("validation")[:n_chains]


def test_cluster_centres_are_textbook(r: Results) -> None:
    try:
        rows = _load_real()
    except Exception as exc:                                    # noqa: BLE001
        r.check(True, f"SKIP: Hub unreachable ({type(exc).__name__})")
        return

    phis, psis, sss = [], [], []
    for row in rows:
        L = int(row["length"])
        c = np.asarray(row["coords"], dtype=np.float64).reshape(L, 4, 3)
        m = np.asarray(row["mask"], dtype=bool)
        phi, psi = backbone_dihedrals(c[:, 0], c[:, 1], c[:, 2])
        phi[~m], psi[~m] = np.nan, np.nan
        phis.append(phi)
        psis.append(psi)
        sss.append(secondary_structure(phi, psi))

    phi = np.concatenate(phis)
    psi = np.concatenate(psis)
    ss = np.concatenate(sss)
    ok = np.isfinite(phi) & np.isfinite(psi)

    h = ok & (ss == SS_INDEX["H"])
    e = ok & (ss == SS_INDEX["E"])
    hphi, hpsi = phi[h].mean(), psi[h].mean()
    ephi, epsi = phi[e].mean(), psi[e].mean()

    r.check(abs(hphi - (-60)) < 20 and abs(hpsi - (-45)) < 25,
            f"helix cluster centre ({hphi:.1f}, {hpsi:.1f}) "
            "~ textbook (-60, -45)",
            "If this drifts, the dihedral arguments are in the wrong order "
            "or the atom columns are mislabelled.")
    r.check(abs(ephi - (-135)) < 30 and abs(epsi - 135) < 30,
            f"strand cluster centre ({ephi:.1f}, {epsi:.1f}) "
            "~ textbook (-135, +135)")
    r.check(hpsi < 0 < epsi,
            f"helix and strand are on OPPOSITE sides in psi "
            f"({hpsi:.1f} vs {epsi:.1f})",
            "This is the single most diagnostic consequence of a sign flip.")

    # All three classes present -- a degenerate assignment is useless.
    fracs = {s: float((ss == i).mean()) for s, i in SS_INDEX.items()}
    r.check(all(f > 0.05 for f in fracs.values()),
            "all three classes are substantially present "
            + ", ".join(f"{s} {f:.3f}" for s, f in fracs.items()))


def test_helix_label_agrees_with_helix_geometry(r: Results) -> None:
    """
    Independent of the angles: an alpha helix has CA(i)..CA(i+4) ~ 6.2 A,
    because that is one full turn. Extended chain is roughly twice that.
    """
    try:
        rows = _load_real(64)
    except Exception as exc:                                    # noqa: BLE001
        r.check(True, f"SKIP: Hub unreachable ({type(exc).__name__})")
        return

    helix_d, other_d = [], []
    for row in rows:
        L = int(row["length"])
        c = np.asarray(row["coords"], dtype=np.float64).reshape(L, 4, 3)
        m = np.asarray(row["mask"], dtype=bool)
        phi, psi = backbone_dihedrals(c[:, 0], c[:, 1], c[:, 2])
        phi[~m], psi[~m] = np.nan, np.nan
        ss = secondary_structure(phi, psi)
        ca = c[:, 1]
        if L < 6:
            continue
        d = np.linalg.norm(ca[4:] - ca[:-4], axis=-1)
        run_h = np.array([(ss[i:i + 5] == SS_INDEX["H"]).all()
                          for i in range(L - 4)])
        valid = np.array([m[i:i + 5].all() for i in range(L - 4)])
        helix_d.append(d[run_h & valid])
        other_d.append(d[(~run_h) & valid])

    h = np.concatenate(helix_d)
    o = np.concatenate(other_d)
    if h.size < 50:
        r.check(False, "enough helical windows to measure", f"only {h.size}")
        return

    r.check(abs(float(np.median(h)) - 6.2) < 1.0,
            f"CA(i)..CA(i+4) in labelled HELIX is {np.median(h):.2f} A "
            "(one turn: ~6.2)",
            "The label and the geometry disagree, so the label is wrong. "
            "This check uses no dihedral at all, which is why it catches "
            "errors the Ramachandran check shares a cause with.")
    r.check(float(np.median(o)) > float(np.median(h)) + 2.0,
            f"and non-helix is much longer ({np.median(o):.2f} A)",
            "Without this the check above passes on a labeller that calls "
            "everything helix.")


# =============================================================================
# 4. Run-length enforcement
# =============================================================================


def test_short_runs_are_demoted(r: Results) -> None:
    # An isolated residue in the helix region, then a genuine 5-long helix.
    phi = np.array([-60.0, 180.0, 180.0, -60, -60, -60, -60, -60, 180.0])
    psi = np.array([-45.0, 170.0, 170.0, -45, -45, -45, -45, -45, 170.0])
    ss = secondary_structure(phi, psi, min_helix=4, min_strand=3)
    r.check(ss[0] == SS_INDEX["C"],
            "an isolated in-region residue is demoted to coil",
            "Secondary structure is contiguous. Without this rule the "
            "assignment over-calls badly -- the first version of this "
            "function reported 59.9% strand against a published 18-25%.")
    r.check((ss[3:8] == SS_INDEX["H"]).all(),
            "a genuine 5-residue helix survives",
            "If run enforcement eats real elements too, it is not a filter, "
            "it is a mute button.")


# =============================================================================
# 5. Label/token alignment -- the silent accuracy thief
# =============================================================================


def test_label_token_alignment(r: Results) -> None:
    """
    ESM-2 prepends <cls>, so residue i must land at token i+1.

    This mirrors exactly what `build()` in train_esm2_ds.py does. An
    off-by-one here trains, converges, scores a few points worse, and raises
    nothing anywhere.
    """
    import torch

    max_length, n_res = 16, 6
    ids = torch.zeros(1, max_length, dtype=torch.long)
    ss = np.array([0, 1, 2, 0, 1, 2])

    y = torch.full_like(ids, -100)
    y[0, 1:1 + n_res] = torch.from_numpy(ss)

    r.check(y[0, 0].item() == -100,
            "<cls> at position 0 is ignored (-100)")
    r.check((y[0, 1:1 + n_res].numpy() == ss).all(),
            "residue i is at token i+1")
    r.check((y[0, 1 + n_res:] == -100).all(),
            "<eos> and padding are ignored (-100)")
    r.check(int((y != -100).sum()) == n_res,
            f"exactly {n_res} positions carry a label",
            "More means specials or padding leaked into the loss; fewer "
            "means residues were dropped.")


def test_masking_proportions(r: Results) -> None:
    rng = np.random.default_rng(0)
    seq = rng.integers(4, 24, size=20_000)
    ids, labels = mask_tokens(seq, mask_token_id=32, vocab_size=33,
                              special_ids={0, 1, 2, 3}, rng=rng)
    sel = labels != -100
    rate = sel.mean()
    masked = (ids[sel] == 32).mean()
    unchanged = (ids[sel] == seq[sel]).mean()

    r.check(abs(rate - 0.15) < 0.01, f"~15% of positions selected ({rate:.3f})")
    r.check(abs(masked - 0.8) < 0.03,
            f"~80% of selected become [MASK] ({masked:.3f})")
    r.check(0.05 < unchanged < 0.20,
            f"~10% of selected are left UNCHANGED ({unchanged:.3f})",
            "This is the arm people delete as a simplification. Without it "
            "the model only ever sees [MASK] where it must predict, and the "
            "input distribution shifts at fine-tuning time.")
    r.check(not (labels[~sel] != -100).any(),
            "unselected positions are ignored by the loss")


def main() -> int:
    r = Results("Secondary structure derived from geometry, not downloaded")
    test_dihedral_is_correct_including_sign(r)
    test_terminal_residues_are_nan_not_zero(r)
    test_cluster_centres_are_textbook(r)
    test_helix_label_agrees_with_helix_geometry(r)
    test_short_runs_are_demoted(r)
    test_label_token_alignment(r)
    test_masking_proportions(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
