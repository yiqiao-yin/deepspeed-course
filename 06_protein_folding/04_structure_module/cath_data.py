#!/usr/bin/env python3
"""
Real backbones from `ajiang2025/cath-4.3-backbone`.

    uv run cath_data.py        # download, and check it is what it claims

Used by `train_structure_ds.py --data cath`. The synthetic arm is the default
because it always runs; this is the arm with real geometry in it.

What the dataset is
-------------------
CATH 4.3 protein backbones prepared for generative modelling: **CC-BY-4.0**,
240 MB, 16,691 / 1,528 / 1,880 chains.

    seq          str        amino acid sequence
    coords       [L, 4, 3]  N, CA, C, O in Angstroms
    mask         [L]        which residues are resolved
    name         str        e.g. "12as.A"
    cath         list[str]  CATH superfamily codes

**128 is a cap, not a fixed length.** The dataset card describes "a fixed
128-residue window", which reads as "every chain is 128 residues" and is not
what the data contains. Measured:

    split        n       length range   median   chains at exactly 128
    train    16,691         40 - 128      128    12,806  (77%)
    validation 1,528        40 - 128      128     1,064  (70%)
    test       1,880        40 - 128      128     1,047  (56%)

Nearly a quarter of the training chains are shorter, which default DataLoader
collation cannot batch -- it would raise on the first mixed batch, after the
download. `require_len=128` below keeps only the exact-length chains and
reports how many it dropped, which leaves 12,806 training chains: far more
than this lab needs, and no padding logic anywhere.

Padding is the alternative and it is worse here. An unresolved or padded
residue has placeholder coordinates, and feeding a placeholder into a frame
construction produces a frame pointing at nothing -- which then participates
in every FAPE term as though it were real geometry.

A mirror is a trust decision, so assert it
-------------------------------------------
`01_basics/03_convnet_cifar10` shipped a lab whose data source was slow enough
to make it unrunnable, and the fix -- loading from a mirror -- introduced a
second risk: data that is *nearly* right trains fine and produces numbers
comparable to nothing. `tests/test_cath_source.py` therefore checks shapes,
split counts, mask consistency and **physics**.

The physics check is the strong one, and structure data admits a better
version than image data does. Consecutive C-alpha atoms in a real protein
backbone sit about **3.8 A** apart -- that is a property of the peptide bond,
not of this dataset. Measured over 64 validation chains (7,419 bonds):

    median 3.806 A, p1 3.72, p99 3.90

A permuted, truncated, unit-confused or subtly corrupted download fails that.
No checksum needed; chemistry is the checksum.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

REPO_ID = "ajiang2025/cath-4.3-backbone"
CROP_LEN = 128
# Index of each atom in the `coords` axis of length 4.
ATOM_N, ATOM_CA, ATOM_C, ATOM_O = 0, 1, 2, 3


def load_split(split: str = "train") -> list[dict]:
    """Download one split's parquet and return it as a list of row dicts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split {split!r}")
    path = hf_hub_download(REPO_ID, f"data/{split}.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise RuntimeError(
            f"{REPO_ID} split {split!r} parsed to zero rows. The layout has "
            "changed; do not train on an empty set."
        )
    return rows


def ca_ca_distances(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Distances between consecutive resolved C-alpha atoms, in Angstroms."""
    ca = coords[:, ATOM_CA, :]
    ok = mask[:-1] & mask[1:]
    return np.linalg.norm(ca[1:] - ca[:-1], axis=-1)[ok]


class CathBackboneDataset(torch.utils.data.Dataset):
    """
    Backbones filtered to exactly `require_len` residues, as the three atoms
    a frame needs plus a resolved-mask.

    Each item is ``(n_xyz, ca_xyz, c_xyz, mask)``, all float32. The training
    script builds frames from the first three and uses the C-alpha positions
    as the FAPE target.

    Chains whose resolved fraction falls below `min_resolved` are dropped
    rather than padded: an unresolved residue's coordinates are placeholders,
    and feeding a placeholder into a frame construction produces a frame
    pointing at nothing.
    """

    def __init__(self, rows: list[dict], min_resolved: float = 0.9,
                 require_len: int | None = CROP_LEN,
                 limit: int | None = None) -> None:
        self.items: list[tuple[torch.Tensor, ...]] = []
        dropped_short = dropped_unresolved = 0
        for row in rows:
            mask = np.asarray(row["mask"], dtype=bool)
            # Keep only exact-length chains, so a batch is a tensor and not a
            # padding problem. See the module docstring: 128 is a cap.
            if require_len is not None and len(mask) != require_len:
                dropped_short += 1
                continue
            if mask.mean() < min_resolved:
                dropped_unresolved += 1
                continue
            coords = np.asarray(row["coords"], dtype=np.float32)
            coords = coords.reshape(len(mask), 4, 3)
            self.items.append((
                torch.from_numpy(coords[:, ATOM_N, :].copy()),
                torch.from_numpy(coords[:, ATOM_CA, :].copy()),
                torch.from_numpy(coords[:, ATOM_C, :].copy()),
                torch.from_numpy(mask.astype(np.float32)),
            ))
            if limit is not None and len(self.items) >= limit:
                break
        self.dropped_short = dropped_short
        self.dropped_unresolved = dropped_unresolved
        if not self.items:
            raise RuntimeError(
                f"no chains survived filtering ({dropped_short} not exactly "
                f"{require_len} residues, {dropped_unresolved} below "
                f"{min_resolved:.0%} resolved). Relax require_len or "
                "min_resolved rather than training on an empty dataset."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download the CATH backbones and check the chemistry."
    )
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=256)
    args, _ = p.parse_known_args()

    print("=" * 78)
    print(f"  {REPO_ID}  --  real backbones, CC-BY-4.0")
    print("=" * 78)

    rows = load_split(args.split)
    print(f"  split {args.split!r:<12} {len(rows)} chains")

    coords0 = np.asarray(rows[0]["coords"], dtype=np.float64)
    print(f"  coords shape       {coords0.shape}   (N, CA, C, O)")
    print(f"  name               {rows[0]['name']}")
    print(f"  cath               {rows[0]['cath']}")

    d = np.concatenate([
        ca_ca_distances(np.asarray(r["coords"], dtype=np.float64).reshape(-1, 4, 3),
                        np.asarray(r["mask"], dtype=bool))
        for r in rows[:64]
    ])
    print(f"\n  CA-CA over 64 chains: n={d.size}  median={np.median(d):.3f} A"
          f"  p1={np.percentile(d, 1):.2f}  p99={np.percentile(d, 99):.2f}")
    print("  (the peptide bond puts this near 3.8 A in any real protein --")
    print("   it is a property of chemistry, not of this dataset)")

    ds = CathBackboneDataset(rows, limit=args.limit)
    print(f"\n  usable chains      {len(ds)} of the first "
          f"{len(ds) + ds.dropped_short + ds.dropped_unresolved} scanned")
    print(f"  dropped            {ds.dropped_short} not exactly {CROP_LEN} "
          f"residues, {ds.dropped_unresolved} below 90% resolved")
    n_xyz, ca_xyz, c_xyz, mask = ds[0]
    print(f"  sample item        N {tuple(n_xyz.shape)}  CA {tuple(ca_xyz.shape)}"
          f"  C {tuple(c_xyz.shape)}  mask {tuple(mask.shape)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
