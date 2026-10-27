#!/usr/bin/env python3
"""
Real MSAs and real contacts, from `ChrisHayduk/nanofold-public`.

    uv run nanofold_data.py        # download 2 shards and check what arrived

Used by `train_evoformer_ds.py --data nanofold`. The synthetic generator in
`synthetic_msa.py` is the default because it always runs; this is the arm that
shows the same trunk on data nobody made up.

What the dataset is
-------------------
10,000 training chains and 1,000 validation chains, derived from
OpenProteinSet / OpenFold PDB-chain data plus RCSB mmCIF coordinates, filtered
to single-chain monomers of 40-256 residues at <= 3.0 A resolution.
**CC-BY-4.0**, 1.02 GB, 29 training shards.

The columns this file uses:

    aatype             [L]            query sequence, integer-coded
    msa                [N_seq, L]     tokenised A3M; row 0 IS the query
    ca_coords          [L, 3]         C-alpha coordinates in Angstroms
    ca_mask            [L]            which residues are resolved
    length             int
    msa_depth          int

Token convention matches `synthetic_msa.py` exactly -- 0-19 amino acids, 20
unknown, 21 gap, 22 mask -- so both data paths feed the trunk identically
shaped integers and `--data` really is a one-word switch.

Why hf_hub_download and not load_dataset(streaming=True)
--------------------------------------------------------
Measured, not assumed. On this dataset:

    load_dataset(..., streaming=True)      no rows after 15 minutes, killed
    hf_hub_download(one shard)             32.5 MB in 1.9 s  (17.5 MB/s)

The bottleneck is the streaming machinery over these nested array columns, not
bandwidth -- the full 1.02 GB takes about a minute fetched directly. This
repository has already shipped one lab that was unrunnable because its data
source was slow enough to outlive the orchestrator's window while reporting
success (`01_basics/03_convnet_cifar10`, see POSTMORTEMS.md). Measuring the
fetch is cheap; discovering it on a rented GPU is not.

Contacts are defined from C-alpha, deliberately
------------------------------------------------
Two residues are in contact when their C-alpha atoms are within 8 A. The CASP
convention uses C-beta at 8 A, which is slightly better because it points along
the side chain -- but C-beta would have to be read out of `atom14_positions`,
and the atom ordering in that column has not been verified here. An 8 A
C-alpha threshold is a recognised definition, it is reproducible from a column
whose meaning is documented, and it is stated plainly rather than implied. Do
not silently switch to atom14 without checking the ordering first.

As in the synthetic path, only long-range pairs (|i - j| >= min_sep) are
scored: residues adjacent in sequence are always in contact, so including them
lets a model score well by reading the index difference.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

REPO_ID = "ChrisHayduk/nanofold-public"
N_TRAIN_SHARDS = 29
CONTACT_THRESHOLD_A = 8.0

TOK_UNKNOWN = 20
TOK_GAP = 21
TOK_MASK = 22
N_TOKENS = 23


def load_shards(n_shards: int = 2, split: str = "train") -> list[dict]:
    """
    Download and parse `n_shards` parquet shards into a list of row dicts.

    Two shards is ~65 MB and ~700 chains, which is plenty for a teaching run.
    Raise `n_shards` for a longer one; all 29 is 1.02 GB.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if split != "train":
        raise ValueError("only the train split is sharded as train-*-of-29")
    n_shards = max(1, min(n_shards, N_TRAIN_SHARDS))

    rows: list[dict] = []
    for i in range(n_shards):
        path = hf_hub_download(
            REPO_ID,
            f"data/train-{i:05d}-of-{N_TRAIN_SHARDS:05d}.parquet",
            repo_type="dataset",
        )
        rows.extend(pq.read_table(path).to_pylist())
    if not rows:
        raise RuntimeError(
            f"{REPO_ID}: {n_shards} shard(s) parsed to zero rows. The dataset "
            "layout has changed; do not train on an empty set."
        )
    return rows


def contacts_from_ca(ca: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Boolean contact map from C-alpha coordinates, 8 A threshold.

    Unresolved residues (mask False) are never in contact with anything --
    their coordinates are placeholders, and treating a placeholder as a
    position at the origin would invent a dense cluster of false contacts at
    whatever the padding value happens to be.
    """
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    contacts = d < CONTACT_THRESHOLD_A
    valid = mask[:, None] & mask[None, :]
    contacts &= valid
    np.fill_diagonal(contacts, False)
    return contacts


class NanofoldContactDataset(torch.utils.data.Dataset):
    """
    Fixed-size crops of real chains, with their MSAs and C-alpha contact maps.

    Chains shorter than `n_res` are skipped rather than padded, and the count
    is reported. Padding would put masked rows and columns into every pair
    tensor and quietly change what the base rate means -- an easy way to make
    a metric look better than it is.
    """

    def __init__(
        self,
        rows: list[dict],
        n_res: int = 64,
        n_seq: int = 64,
        min_sep: int = 6,
        seed: int = 0,
    ) -> None:
        self.n_res = n_res
        self.n_seq = n_seq
        self.min_sep = min_sep
        rng = np.random.default_rng(seed)

        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        skipped_short = skipped_unresolved = 0

        for row in rows:
            length = int(row["length"])
            if length < n_res:
                skipped_short += 1
                continue

            ca = np.asarray(row["ca_coords"], dtype=np.float64).reshape(length, 3)
            ca_mask = np.asarray(row["ca_mask"], dtype=bool).reshape(length)
            msa = np.asarray(row["msa"], dtype=np.int64)
            if msa.ndim != 2:
                msa = msa.reshape(-1, length)

            ca, ca_mask, msa = ca[:n_res], ca_mask[:n_res], msa[:, :n_res]
            if ca_mask.sum() < 0.5 * n_res:
                skipped_unresolved += 1
                continue

            # Keep the query (row 0) and sample homologs for the rest.
            depth = msa.shape[0]
            if depth > n_seq:
                pick = np.concatenate(
                    [[0], 1 + rng.choice(depth - 1, n_seq - 1, replace=False)]
                )
            else:
                pick = np.concatenate(
                    [np.arange(depth), np.zeros(n_seq - depth, dtype=int)]
                )
            sampled = np.clip(msa[pick], 0, N_TOKENS - 1)

            contacts = contacts_from_ca(ca, ca_mask)
            self.items.append(
                (
                    torch.from_numpy(sampled.astype(np.int64)),
                    torch.from_numpy(contacts).float(),
                )
            )

        self.skipped_short = skipped_short
        self.skipped_unresolved = skipped_unresolved
        if not self.items:
            raise RuntimeError(
                f"no chains survived filtering at n_res={n_res} "
                f"({skipped_short} too short, {skipped_unresolved} too "
                "unresolved). Lower --n-res or load more shards; training on "
                "an empty dataset would silently report a base rate of zero."
            )
        self._scored = self._build_scored_mask()

    def _build_scored_mask(self) -> torch.Tensor:
        idx = np.arange(self.n_res)
        sep = np.abs(idx[:, None] - idx[None, :])
        return torch.from_numpy(sep >= self.min_sep).float()

    def scored_mask(self) -> torch.Tensor:
        """Which pairs count: long-range only, as in the synthetic path."""
        return self._scored

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        msa, contacts = self.items[i]
        return msa, contacts, self._scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download nanofold-public shards and report what arrived."
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--n-res", type=int, default=64)
    parser.add_argument("--n-seq", type=int, default=64)
    args, _ = parser.parse_known_args()

    print("=" * 78)
    print(f"  {REPO_ID}  --  real MSAs, real contacts")
    print("=" * 78)

    rows = load_shards(n_shards=args.shards)
    print(f"  rows parsed          : {len(rows)}")
    depths = [int(r["msa_depth"]) for r in rows]
    lengths = [int(r["length"]) for r in rows]
    print(f"  chain length         : {min(lengths)}-{max(lengths)} "
          f"(median {int(np.median(lengths))})")
    print(f"  MSA depth            : {min(depths)}-{max(depths)} "
          f"(median {int(np.median(depths))})")

    ds = NanofoldContactDataset(rows, n_res=args.n_res, n_seq=args.n_seq)
    print(f"  usable at n_res={args.n_res:<4}: {len(ds)} chains "
          f"({ds.skipped_short} too short, {ds.skipped_unresolved} unresolved)")

    msa, contacts, scored = ds[0]
    tri = torch.triu(scored.bool(), diagonal=1)
    print(f"\n  sample MSA           : {tuple(msa.shape)}  "
          f"tokens {int(msa.min())}..{int(msa.max())}")
    print(f"  sample contact map   : {tuple(contacts.shape)}")
    print(f"  long-range base rate : {contacts[tri].mean():.4f}")
    print("\n  Contacts are C-alpha pairs within "
          f"{CONTACT_THRESHOLD_A:.0f} A, long-range only.")
    print("=" * 78)


if __name__ == "__main__":
    main()
