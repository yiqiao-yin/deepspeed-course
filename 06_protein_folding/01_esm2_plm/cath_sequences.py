#!/usr/bin/env python3
"""
CATH 4.3 sequences and backbones, for `01_esm2_plm`.

A deliberate near-copy of `04_structure_module/cath_data.py` -- same dataset,
same CC-BY-4.0 licence, same direct-parquet access. Duplicated rather than
imported, per the repository's no-shared-module rule: each folder must run
without the others existing.

This copy differs in what it keeps. The structure module needs N, CA and C to
build rigid frames; this lab needs the **sequence** plus the same three atoms
to derive secondary structure labels. What it does *not* do is filter to a
fixed length -- a language model handles variable-length input natively and
the tokenizer pads, so the structure module's `require_len` filter would throw
away a quarter of the data for nothing.

See `04_structure_module/cath_data.py` for the dataset's full description,
including the measured fact that the dataset card's "cropped to a fixed
128-residue window" is wrong: lengths run 40-128.
"""

from __future__ import annotations

REPO_ID = "ajiang2025/cath-4.3-backbone"


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
            f"{REPO_ID} split {split!r} parsed to zero rows -- the layout has "
            "changed. Do not train on an empty set."
        )
    return rows
