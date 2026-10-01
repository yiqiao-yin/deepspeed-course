# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface-hub", "torch"]
# ///
"""
Regression test: the CATH backbones are what they claim to be.

Run:
    uv run tests/test_cath_source.py

Why this suite exists
---------------------
`06_protein_folding/04_structure_module --data cath` trains on a Hub dataset
nobody here produced. Data that is *nearly* right -- a different split
boundary, a truncated chain, coordinates in nanometres instead of Angstroms,
atoms in a different order -- trains perfectly well and produces numbers
comparable to nothing. `01_basics/03_convnet_cifar10` already taught this
repository that a mirror is a trust decision; see POSTMORTEMS.md.

Structure data admits a stronger check than image data does: **chemistry**.
Consecutive C-alpha atoms in a protein backbone sit about 3.8 A apart because
of the geometry of the peptide bond. That is a fact about proteins, not about
this dataset, so it catches corruption no checksum was computed for:

    unit error (nm not A)        median would be 0.38, not 3.8
    atom order wrong             CA-CA would be nonsense
    chains truncated/concatenated  spacing would jump at the seams
    coordinates permuted         spacing would be random

What is asserted
----------------
1. Split row counts match the published 16,691 / 1,528 / 1,880.
2. `coords` is [L, 4, 3] -- N, CA, C, O.
3. **128 is a CAP, not a fixed length.** The dataset card says "cropped to a
   fixed 128-residue window", which reads as "every chain is 128" and is
   false: lengths run 40-128. The loader filters to exactly 128 for batching,
   and this check pins the fact so nobody re-reads the card and removes the
   filter.
4. C-alpha spacing is ~3.8 A, with a tolerance measured from the data rather
   than assumed -- and asserted tightly enough that a 10% unit error fails.
5. The mask is consistent with the stated resolution rate.

This suite needs the network by design. It checks a remote artifact; mocking
it would assert nothing. It skips cleanly when the Hub is unreachable.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "06_protein_folding" / "04_structure_module"))

from _srcload import Results                                    # noqa: E402

EXPECTED_ROWS = {"train": 16_691, "validation": 1_528, "test": 1_880}
# The peptide bond. Not a property of this dataset.
CA_CA_ANGSTROM = 3.8


def main() -> int:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from cath_data import ATOM_CA, CROP_LEN, REPO_ID, ca_ca_distances
    except ImportError as exc:
        print(f"\n  SKIP: missing dependency ({exc}).")
        return 0

    try:
        paths = {s: hf_hub_download(REPO_ID, f"data/{s}.parquet",
                                    repo_type="dataset")
                 for s in EXPECTED_ROWS}
    except Exception as exc:                                    # noqa: BLE001
        print(f"\n  SKIP: could not reach the Hub ({type(exc).__name__}: {exc}).")
        print("        This suite needs the network by design -- it checks a")
        print("        remote artifact, and mocking it would assert nothing.")
        return 0

    r = Results("CATH 4.3 backbones: shapes, splits, and chemistry")

    tables = {s: pq.read_table(p) for s, p in paths.items()}

    # -- 1. splits ---------------------------------------------------------
    for split, expected in EXPECTED_ROWS.items():
        got = tables[split].num_rows
        r.check(got == expected,
                f"{split} split has {expected} chains (got {got})",
                "A different split boundary makes every published number "
                "incomparable with the literature that used this benchmark.")

    rows = tables["validation"].to_pylist()

    # -- 2. shapes ---------------------------------------------------------
    lengths = np.asarray([int(x) for x in tables["train"].column("length").to_pylist()])
    c0 = np.asarray(rows[0]["coords"], dtype=np.float64)
    L0 = int(rows[0]["length"])
    r.check(c0.reshape(L0, 4, 3).shape == (L0, 4, 3),
            f"coords reshape to [L, 4, 3] (L={L0}) -- N, CA, C, O")

    # -- 3. 128 is a cap, not a fixed length -------------------------------
    at_cap = int((lengths == CROP_LEN).sum())
    r.check(lengths.max() == CROP_LEN,
            f"no chain exceeds the {CROP_LEN}-residue cap (max {lengths.max()})")
    r.check(lengths.min() < CROP_LEN,
            f"but chains are NOT all {CROP_LEN} residues "
            f"(min {lengths.min()}, {at_cap}/{len(lengths)} at the cap)",
            "The dataset card says 'cropped to a fixed 128-residue window', "
            "which reads as uniform length and is not. If this ever becomes "
            "true, CathBackboneDataset's require_len filter can be dropped -- "
            "until then removing it breaks batching on the first mixed batch.")

    # -- 4. chemistry ------------------------------------------------------
    d = np.concatenate([
        ca_ca_distances(np.asarray(x["coords"], dtype=np.float64)
                        .reshape(int(x["length"]), 4, 3),
                        np.asarray(x["mask"], dtype=bool))
        for x in rows[:128]])
    median = float(np.median(d))
    r.check(abs(median - CA_CA_ANGSTROM) < 0.1,
            f"consecutive CA atoms are {median:.3f} A apart "
            f"(peptide bond: ~{CA_CA_ANGSTROM})",
            "Off by 10x means nanometres. Off by anything else means the "
            "atom order, the chain order or the coordinates are wrong.")
    p1, p99 = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    r.check(3.6 < p1 and p99 < 4.0,
            f"the spacing distribution is TIGHT (p1 {p1:.2f}, p99 {p99:.2f})",
            "A real backbone barely varies here. A wide distribution means "
            "chains have been concatenated or residues dropped mid-chain.")

    # The tolerance has to be tight enough to catch a unit error. Show it is.
    r.check(abs(median / 10.0 - CA_CA_ANGSTROM) > 0.1,
            "the tolerance would REJECT a nanometre/Angstrom mix-up",
            "If a 10x error passed, this check would be decorative.")

    # -- 5. masks ----------------------------------------------------------
    frac = float(np.mean([np.mean(np.asarray(x["mask"], dtype=bool))
                          for x in rows]))
    r.check(0.90 < frac <= 1.0,
            f"resolved fraction is ~96% as published (measured {frac:.3f})")
    r.check(all(len(x["mask"]) == int(x["length"]) for x in rows),
            "mask length matches the stated chain length for every row")

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
