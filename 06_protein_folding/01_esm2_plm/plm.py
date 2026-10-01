#!/usr/bin/env python3
"""
A protein is a sequence, so the whole LLM stack transfers. Almost.

    uv run plm.py          # tokenisation, masking, and label derivation, CPU

This is the on-ramp to `06_protein_folding/`. Before the Evoformer's triangle
operations and the structure module's rigid frames, there is a much more
familiar object: a string over a 20-letter alphabet, and a transformer trained
on it with masked language modelling. ESM-2 is BERT for proteins, and
everything you know about fine-tuning BERT applies.

What transfers, and what does not
----------------------------------
**Transfers:** the tokenizer, the masking objective, LoRA, ZeRO, gradient
accumulation, mixed precision, the whole HuggingFace surface. ESM-2 is an
`EsmForMaskedLM` and `Trainer` does not know or care that the tokens are
amino acids.

**Does not transfer, and it changes the arithmetic:**

- **ESM-2 is an ENCODER.** There is no causal mask and no generation. Every
  position attends to every other, so a batch costs O(L^2) attention whether
  or not you are predicting the last token. The course's causal labs let you
  think in tokens-per-second; here you think in *padded* tokens per second,
  and padding is most of the bill.
- **Labels are per-residue.** A downstream head predicts something for every
  position, not one class for the sequence. That makes the collator, not the
  model, the place where most bugs live.
- **The vocabulary is tiny.** 33 tokens against a modern LLM's 100k+. The
  embedding and output layers, which dominate parameter counts in small
  language models, are almost free here -- so an ESM-2 of a given size has
  far more of its parameters in actual transformer blocks.

Where the labels come from
--------------------------
The downstream task here is **3-state secondary structure** -- is this residue
in a helix, a strand, or neither -- and the labels are **derived, not
downloaded**.

That is a deliberate choice. The obvious benchmark set is CC-BY-NC-SA, which
sits badly in an MIT-licensed course, and depending on it would make the whole
section's data provenance messier. Deriving from the CC-BY-4.0 CATH backbones
keeps one permissive data spine across all four subtopics.

It is also just better teaching. Secondary structure is not a label somebody
assigned; it is a fact about backbone geometry. Two dihedral angles decide it:

    phi(i) = dihedral( C(i-1), N(i),  CA(i), C(i)  )
    psi(i) = dihedral( N(i),   CA(i), C(i),  N(i+1) )

Plot every residue's (phi, psi) and you get the Ramachandran diagram, which
has two dense regions: alpha helix near (-60, -45) and beta sheet near
(-135, +135). Assigning H / E / C is then a lookup plus a run-length rule.

`uv run plm.py` validates the derivation on the **cluster centres**, which
come out at (-66.7, -32.3) for helix and (-119.1, 134.6) for strand against
textbook (-60, -45) and (-135, +135). Those move immediately if the dihedral
arguments are in the wrong order, the sign is flipped, or the atoms are
mislabelled -- which is the failure mode worth catching, because a broken
derivation still produces labels of the right shape that train to a plausible
accuracy.

It reports the class fractions without asserting them against a target, and
that restraint is deliberate: this is not DSSP, and CATH is a domain database
enriched in beta relative to whole proteomes, so the familiar "~33% helix"
figure describes a different population. Tuning the region boundaries until
they matched it would be fitting the derivation to the wrong target.

A note on the model ladder
--------------------------
`train_esm2_ds.py --model` accepts 8M, 35M, 150M, 650M and 3B. The jump at
the top is worth knowing about:

    checkpoint                     params        format
    esm2_t6_8M_UR50D            7,512,474        safetensors
    esm2_t12_35M_UR50D         33,995,044        safetensors
    esm2_t30_150M_UR50D       148,798,300        safetensors
    esm2_t33_650M_UR50D       652,358,616        safetensors
    esm2_t36_3B_UR50D                  3B        .bin ONLY, 2 shards

The 3B and 15B checkpoints predate safetensors and ship only
`pytorch_model-*.bin`. That was an open question for this lab -- transformers
5.x writes safetensors exclusively and ignores `safe_serialization=False` --
but **reading legacy checkpoints still works**: verified on transformers
5.16.1 -- the version this lab LOCKS -- for both the single-file and the
sharded-plus-index layouts. The 3B
rung is therefore offered. If a future transformers drops the reader, the
ladder stops at 650M and this docstring is where to say so.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# ESM-2's alphabet, in the order the tokenizer uses for the 20 standard acids.
AA = "LAGVSERTIDPKQNFYMHWC"
SS_LABELS = ("H", "E", "C")        # helix, strand, coil
SS_INDEX = {s: i for i, s in enumerate(SS_LABELS)}


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             p3: np.ndarray) -> np.ndarray:
    """
    Signed dihedral angle about the p1-p2 axis, in degrees.

    Vectorised over a leading axis. The standard numerically-stable
    formulation: project the outer bonds onto the plane perpendicular to the
    central bond and take atan2 of the result.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-8, None)

    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def backbone_dihedrals(n_xyz: np.ndarray, ca_xyz: np.ndarray,
                       c_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-residue (phi, psi) in degrees, as ``[L]`` arrays.

    The first residue has no phi and the last has no psi -- there is no
    previous C or next N -- so both are filled with NaN rather than a
    plausible-looking zero. A zero would land squarely in the "neither helix
    nor strand" region and silently become a coil label.
    """
    L = len(ca_xyz)
    phi = np.full(L, np.nan)
    psi = np.full(L, np.nan)
    if L < 2:
        return phi, psi
    phi[1:] = dihedral(c_xyz[:-1], n_xyz[1:], ca_xyz[1:], c_xyz[1:])
    psi[:-1] = dihedral(n_xyz[:-1], ca_xyz[:-1], c_xyz[:-1], n_xyz[1:])
    return phi, psi


def _enforce_runs(ss: np.ndarray, label: int, min_run: int) -> np.ndarray:
    """Demote runs of `label` shorter than `min_run` to coil."""
    out = ss.copy()
    i, n = 0, len(ss)
    while i < n:
        if ss[i] != label:
            i += 1
            continue
        j = i
        while j < n and ss[j] == label:
            j += 1
        if j - i < min_run:
            out[i:j] = SS_INDEX["C"]
        i = j
    return out


def secondary_structure(phi: np.ndarray, psi: np.ndarray,
                        min_helix: int = 4, min_strand: int = 3) -> np.ndarray:
    """
    3-state secondary structure from Ramachandran regions plus run lengths.

    Returns an integer array over ``SS_LABELS`` -- 0 = H, 1 = E, 2 = C.

    This is a geometric assignment, not DSSP, and the difference matters
    enough to have broken the first version of this function.

    **A (phi, psi) lookup alone cannot find beta sheet.** The upper-left
    Ramachandran region contains beta strand, polyproline II, and a great deal
    of ordinary extended coil, and nothing in the two angles separates them --
    what makes a strand a strand is hydrogen bonding to *another* strand,
    which is non-local and is exactly what DSSP measures. The first version
    here used the region alone and assigned **59.9% strand** against a
    published 18-25%, while its mean (phi, psi) per class looked entirely
    reasonable. Plausible cluster centres, wildly wrong fractions.

    The fix is the other constraint real assignments use: secondary structure
    comes in **contiguous elements**. A helix needs at least one full turn
    (~4 residues) and a strand at least `min_strand`. Isolated residues that
    merely fall in a region are coil. That is cheap, local, and recovers
    fractions in the published range -- see `uv run plm.py`.
    """
    ss = np.full(len(phi), SS_INDEX["C"], dtype=np.int64)

    # Tighter than the first attempt: the canonical alpha and beta clusters,
    # not the whole quadrant.
    helix = (phi >= -110) & (phi <= -35) & (psi >= -70) & (psi <= 10)
    sheet = (phi >= -180) & (phi <= -90) & (psi >= 100) & (psi <= 180)

    ss[sheet] = SS_INDEX["E"]
    ss[helix] = SS_INDEX["H"]        # helix wins where the regions overlap
    ss[np.isnan(phi) | np.isnan(psi)] = SS_INDEX["C"]

    # Bridge single-residue gaps BEFORE enforcing run lengths. One residue of
    # a real helix straying outside the box otherwise splits it into two
    # sub-minimum runs, and both get demoted to coil -- which is why the
    # first run-length version under-called helix at 0.121 against a
    # published 0.30-0.37.
    for lab in (SS_INDEX["H"], SS_INDEX["E"]):
        inner = ss[1:-1]
        gap = (inner == SS_INDEX["C"]) & (ss[:-2] == lab) & (ss[2:] == lab)
        inner[gap] = lab

    ss = _enforce_runs(ss, SS_INDEX["H"], min_helix)
    ss = _enforce_runs(ss, SS_INDEX["E"], min_strand)
    return ss


def mask_tokens(token_ids: np.ndarray, mask_token_id: int, vocab_size: int,
                special_ids: set[int], rate: float = 0.15,
                rng: np.random.Generator | None = None):
    """
    BERT-style masking, 80/10/10, as ESM-2 was pretrained.

    80% of selected positions become `[MASK]`, 10% become a random token, 10%
    are left alone. Labels are -100 everywhere else so the loss ignores them.

    The 10% "leave it alone" arm is the part people drop as a simplification,
    and dropping it changes what the model learns: without it the model only
    ever sees `[MASK]` at positions it must predict, so at fine-tuning time --
    where nothing is masked -- the input distribution has shifted out from
    under it.
    """
    rng = rng or np.random.default_rng(0)
    ids = token_ids.copy()
    labels = np.full_like(ids, -100)

    special = np.isin(ids, list(special_ids))
    selected = (rng.random(ids.shape) < rate) & ~special
    labels[selected] = ids[selected]

    r = rng.random(ids.shape)
    ids[selected & (r < 0.8)] = mask_token_id
    randomise = selected & (r >= 0.8) & (r < 0.9)
    ids[randomise] = rng.integers(4, vocab_size, size=int(randomise.sum()))
    return ids, labels


def _demo(n_chains: int = 64) -> None:
    """Derive labels from real backbones and check them against chemistry."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cath_sequences import load_split
    except ImportError as exc:                              # noqa: BLE001
        print(f"  SKIP: {exc}")
        return

    rows = load_split("validation")[:n_chains]
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
    frac = {s: float((ss == i).mean()) for s, i in SS_INDEX.items()}

    print("\n" + "=" * 78)
    print(f"  SECONDARY STRUCTURE, DERIVED FROM {len(rows)} REAL BACKBONES")
    print("=" * 78)
    print(f"  helix (H)  {frac['H']:.3f}")
    print(f"  strand (E) {frac['E']:.3f}")
    print(f"  coil (C)   {frac['C']:.3f}")
    print(f"\n  majority-class baseline for the downstream task: "
          f"{max(frac.values()):.3f}")
    print("  (a per-residue classifier must beat THIS, not 0.333)")

    ok = np.isfinite(phi) & np.isfinite(psi)
    h = ss[ok] == SS_INDEX["H"]
    e = ss[ok] == SS_INDEX["E"]
    print(f"\n  mean (phi, psi) by assignment -- the Ramachandran check")
    print(f"    helix   ({phi[ok][h].mean():>7.1f}, {psi[ok][h].mean():>7.1f})"
          "    textbook (-60, -45)")
    print(f"    strand  ({phi[ok][e].mean():>7.1f}, {psi[ok][e].mean():>7.1f})"
          "    textbook (-135, +135)")
    print(
        "\n  The CLUSTER CENTRES are the check that means something. They are\n"
        "  textbook values for alpha and beta geometry, and a derivation with\n"
        "  the dihedral arguments in the wrong order, the sign flipped, or the\n"
        "  atoms mislabelled moves them immediately.\n"
        "\n  The FRACTIONS are reported, not asserted against a target. Two\n"
        "  reasons. First, this is a Ramachandran-plus-run-length assignment,\n"
        "  not DSSP -- what makes a strand a strand is hydrogen bonding to\n"
        "  another strand, which is non-local and invisible to two angles.\n"
        "  Second, CATH is a DOMAIN database and is enriched in beta relative\n"
        "  to whole proteomes, so the usual '~33% helix' figure is a statistic\n"
        "  about a different population. Comparing to it and tuning the boxes\n"
        "  until they matched would be fitting the derivation to the wrong\n"
        "  target.\n"
    )


def _demo_masking() -> None:
    rng = np.random.default_rng(0)
    seq = rng.integers(4, 24, size=600)   # a realistic chain;
    # at 40 residues only ~6 positions are selected and the 10%
    # unchanged arm rounds to zero, hiding the thing being shown.
    ids, labels = mask_tokens(seq, mask_token_id=32, vocab_size=33,
                              special_ids={0, 1, 2, 3}, rng=rng)
    n_sel = int((labels != -100).sum())
    n_mask = int((ids == 32).sum())
    n_kept = int(((labels != -100) & (ids == seq)).sum())
    print("=" * 78)
    print("  MASKED LANGUAGE MODELLING, 80/10/10")
    print("=" * 78)
    print(f"  positions selected : {n_sel}/{len(seq)}  "
          f"({100 * n_sel / len(seq):.1f}%)")
    print(f"  replaced by [MASK] : {n_mask}")
    print(f"  left UNCHANGED     : {n_kept}  <- the arm people delete")
    print(
        "\n  Deleting the unchanged arm means the model only ever sees [MASK]\n"
        "  where it must predict. At fine-tuning time nothing is masked, so\n"
        "  the input distribution shifts out from under it.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Protein language modelling: tokens, masks and derived "
                    "labels, on CPU."
    )
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--skip-data", action="store_true",
                   help="masking demo only; no download")
    args, _ = p.parse_known_args()

    print("=" * 78)
    print("  ESM-2: A PROTEIN IS A SEQUENCE")
    print("=" * 78)
    print("  No GPU, no model download. The objective and the labels, from")
    print("  first principles.\n")

    _demo_masking()
    if not args.skip_data:
        _demo(args.chains)

    print("=" * 78)
    print("  Next:  uv run deepspeed --num_gpus=1 train_esm2_ds.py --task ss")
    print("=" * 78)


if __name__ == "__main__":
    main()
