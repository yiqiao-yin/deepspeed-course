#!/usr/bin/env python3
"""
Synthetic MSAs that carry the signal an Evoformer actually eats: coevolution.

    uv run synthetic_msa.py      # show that the signal is there, and measure it

Why this file exists at all
---------------------------
This course has shipped the same data bug twice, and it is documented in
`POSTMORTEMS.md`. `01_basics/02_convnet` drew `x = randn(...)` and
`y = randint(...)` -- labels independent of inputs, so **zero mutual
information**. Ten percent on ten classes was not a poor result, it was the
information-theoretic ceiling, and the script cheerfully advised the reader to
train longer.

Random sequences with random contact maps would be exactly that bug in a lab
coat. So this generator does not plant "some learnable pattern". It reproduces
**the specific statistical structure real MSAs have and the Evoformer exists to
exploit**:

    residues that touch in 3D mutate together across evolution

If position i changes and position j must change with it to keep the fold
stable, then columns i and j of the alignment are statistically coupled. That
coupling is the *only* evidence AlphaFold has about contacts, and here it is
the only evidence too.

How a sequence is generated
---------------------------
1. Draw a query sequence over the 20 amino acids.
2. Plant a set of **long-range** contacts (i, j), |i - j| >= `min_sep`.
3. For each homolog, mutate positions at rate `mut_rate`:
   - an *uncoupled* position mutates to a uniformly random residue;
   - for a *coupled* pair (i, j), i and j draw **independent** mutation events,
     and then with probability `coupling` residue j is overwritten as a fixed
     per-pair function of residue i, so the two columns covary.

`coupling=1.0` gives perfect covariation, `coupling=0.0` gives none at all.

The independence of those two draws is load-bearing and was got wrong first
time: sharing one mutation event between i and j couples them through the
event itself, so the `coupling=0.0` arm stayed learnable. See the comment at
the site in `make_msa`.

Measured by `uv run synthetic_msa.py`, 48 residues, depth 64, APC-corrected
mutual information, precision at K where K is the true contact count:

    coupling     gap (nats)     prec@K        (base rate 0.014)
        0.00        -0.069       0.000
        0.25         0.119       0.385
        0.50         0.407       0.923
        1.00         1.110       1.000

    depth 1          0.000       0.077
    depth 2         -0.000       0.000
    depth 8          0.387       0.462
    depth 32         0.870       1.000

Two design decisions that are load-bearing
------------------------------------------
**1. Contacts are long-range only, and the near-diagonal band is excluded from
both the loss and the metrics.** In a real protein, residues i and i+1 are
always in contact, so a model can score well on "contacts" by learning
`|i - j| == 1` and ignoring the MSA entirely. That is a shortcut which passes
every accuracy check while learning nothing about coevolution. Excluding the
band means the *only* route to a good score is the statistics -- which is what
makes the counterexample below decisive rather than suggestive.

**2. The unlearnable arm is kept forever.** `coupling=0.0` produces MSAs whose
columns are independent, so the contact map is unrecoverable in principle --
mutual information between contacting and non-contacting pairs is identical.
`tests/test_evoformer_data_is_learnable.py` asserts that this arm **FAILS**.
A learnability test that never sees unlearnable data passes while returning
True unconditionally, which is the bug `beats_chance()` shipped with.

There is a third arm worth knowing about: `n_seq=1`. A single sequence has no
statistics to measure, so contacts are unrecoverable no matter how strong the
coupling. That is not a limitation of this generator -- **it is why MSAs exist**,
and it is why ESMFold giving up MSAs was a surprising result rather than an
obvious one.

Token convention
----------------
Deliberately matched to `ChrisHayduk/nanofold-public`, so `--data synthetic`
and `--data nanofold` feed the model identically shaped integers:

    0-19  the twenty amino acids
    20    unknown
    21    gap
    22    mask
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

N_AMINO_ACIDS = 20
TOK_UNKNOWN = 20
TOK_GAP = 21
TOK_MASK = 22
N_TOKENS = 23


# =============================================================================
# Generation
# =============================================================================


@dataclass
class SyntheticConfig:
    """
    Knobs for the generator.

    `coupling` is the one that matters. It is the difference between a dataset
    that teaches the model coevolution and a dataset that teaches it nothing,
    and it is swept in the demonstration at the bottom of this file.
    """

    n_res: int = 48            # residues per chain
    n_seq: int = 64            # sequences per MSA (alignment depth)
    n_contacts: int = 24       # planted long-range contacts
    min_sep: int = 6           # |i - j| below this is never a planted contact
    coupling: float = 1.0      # 0.0 = no coevolution, 1.0 = perfect
    mut_rate: float = 0.5      # fraction of positions that mutate per homolog


def make_contact_map(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Plant `n_contacts` long-range contacts.

    Returns a symmetric boolean ``[n_res, n_res]`` array with a zero diagonal.
    Only pairs with ``|i - j| >= cfg.min_sep`` are ever set, for the reason in
    the module docstring: near-diagonal contacts are guessable from the indices
    alone and would let a model cheat.
    """
    contacts = np.zeros((cfg.n_res, cfg.n_res), dtype=bool)
    candidates = [
        (i, j)
        for i in range(cfg.n_res)
        for j in range(i + cfg.min_sep, cfg.n_res)
    ]
    if not candidates:
        raise ValueError(
            f"n_res={cfg.n_res} and min_sep={cfg.min_sep} leave no candidate "
            "pairs; increase n_res or decrease min_sep"
        )
    n = min(cfg.n_contacts, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)

    # Each residue takes part in at most one coupled pair. Overlapping pairs
    # would make the mutation rule ambiguous (two partners fighting over one
    # position), and the resulting statistics would be muddier than the code
    # suggests -- a silent weakening of the very signal this file promises.
    used: set[int] = set()
    for idx in chosen:
        i, j = candidates[idx]
        if i in used or j in used:
            continue
        used.update((i, j))
        contacts[i, j] = contacts[j, i] = True
    return contacts


def make_msa(
    contacts: np.ndarray, cfg: SyntheticConfig, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate one MSA whose column pairs covary exactly at the planted contacts.

    Returns ``[n_seq, n_res]`` of int64 tokens. Row 0 is the query, unmutated,
    matching the convention in real alignments (and in `nanofold-public`, where
    ``msa[0] == aatype`` was verified to hold).
    """
    n_res = cfg.n_res
    query = rng.integers(0, N_AMINO_ACIDS, size=n_res)

    pairs = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(contacts)))]
    partner = {}
    # A fixed substitution map per pair: residue a at i implies (a + shift) at
    # j. Any bijection works; a per-pair shift is the simplest one that makes
    # the two columns dependent without making them identical.
    for i, j in pairs:
        shift = int(rng.integers(1, N_AMINO_ACIDS))
        partner[i] = (j, shift)

    coupled_positions = set()
    for i, (j, _) in partner.items():
        coupled_positions.update((i, j))

    msa = np.tile(query, (cfg.n_seq, 1))
    for s in range(1, cfg.n_seq):           # row 0 stays the query
        row = msa[s]

        # Independent positions mutate on their own.
        for p in range(n_res):
            if p in coupled_positions:
                continue
            if rng.random() < cfg.mut_rate:
                row[p] = rng.integers(0, N_AMINO_ACIDS)

        # Coupled pairs. The mutation events for i and j are drawn
        # INDEPENDENTLY, and coupling is applied afterwards by making j a
        # function of i.
        #
        # An earlier version of this loop drew ONE event for the pair -- "if
        # the pair mutates, mutate both" -- and that shared indicator made i
        # and j dependent even at coupling=0.0, because when the event did not
        # fire, both positions kept their query residue together. The
        # `coupling=0.0` arm measured 1.9x separation and 0.271 top-L
        # precision against a 0.014 base rate: the supposedly unlearnable
        # counterexample was quietly learnable. Caught by running
        # `uv run synthetic_msa.py`, not by reading it.
        for i, (j, shift) in partner.items():
            i_mutates = rng.random() < cfg.mut_rate
            j_mutates = rng.random() < cfg.mut_rate
            if i_mutates:
                row[i] = int(rng.integers(0, N_AMINO_ACIDS))
            if rng.random() < cfg.coupling:
                row[j] = (row[i] + shift) % N_AMINO_ACIDS   # j follows i
            elif j_mutates:
                row[j] = int(rng.integers(0, N_AMINO_ACIDS))  # j goes its own way
    return msa.astype(np.int64)


def eval_mask(cfg: SyntheticConfig) -> np.ndarray:
    """
    Which pairs count, as a boolean ``[n_res, n_res]``.

    Everything with ``|i - j| < min_sep`` is excluded, along with the diagonal.
    Both the training loss and every reported metric use this mask, so a model
    cannot bank score on the trivially-predictable band near the diagonal.
    """
    idx = np.arange(cfg.n_res)
    sep = np.abs(idx[:, None] - idx[None, :])
    return sep >= cfg.min_sep


def make_batch(
    batch_size: int = 4,
    n_res: int = 48,
    n_seq: int = 64,
    coupling: float = 1.0,
    seed: int = 0,
    cfg: SyntheticConfig | None = None,
) -> dict[str, torch.Tensor]:
    """
    A batch of independent synthetic chains.

    Returns a dict of tensors:
        ``msa``       int64  ``[B, n_seq, n_res]``
        ``contacts``  float  ``[B, n_res, n_res]``  the labels, symmetric
        ``mask``      float  ``[B, n_res, n_res]``  which pairs are scored
    """
    cfg = cfg or SyntheticConfig(n_res=n_res, n_seq=n_seq, coupling=coupling)
    rng = np.random.default_rng(seed)

    msas, maps = [], []
    for _ in range(batch_size):
        contacts = make_contact_map(cfg, rng)
        msas.append(make_msa(contacts, cfg, rng))
        maps.append(contacts)

    m = eval_mask(cfg)
    return {
        "msa": torch.from_numpy(np.stack(msas)),
        "contacts": torch.from_numpy(np.stack(maps)).float(),
        "mask": torch.from_numpy(np.tile(m, (batch_size, 1, 1))).float(),
    }


class SyntheticContactDataset(torch.utils.data.Dataset):
    """
    A fixed-size dataset of independent chains, generated once up front.

    Generated eagerly rather than on the fly so that a train/eval split is a
    real split: the eval chains are never seen during training. Measuring
    contact accuracy on chains the model trained on would report memorisation,
    which is the mistake `test_synthetic_data_is_learnable.py` exists to
    prevent and the reason it asserts on a HELD-OUT split.
    """

    def __init__(self, n_items: int, cfg: SyntheticConfig, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.items = []
        m = eval_mask(cfg)
        for _ in range(n_items):
            contacts = make_contact_map(cfg, rng)
            self.items.append(
                (
                    torch.from_numpy(make_msa(contacts, cfg, rng)),
                    torch.from_numpy(contacts).float(),
                    torch.from_numpy(m).float(),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.items[i]


# =============================================================================
# Measuring the signal, without training anything
# =============================================================================


def column_mutual_information(msa: np.ndarray, apc: bool = True) -> np.ndarray:
    """
    Mutual information between every pair of MSA columns, in nats.

    This is the classical pre-deep-learning contact predictor, and it is here
    for a reason that has nothing to do with nostalgia: **it measures whether
    the data carries the signal at all, with no model in the loop.**

    If MI at planted contacts is not clearly above MI elsewhere, then no
    architecture can recover the contacts, and a model that scores well is
    scoring on something else. Asserting on this catches a broken generator in
    milliseconds, without the confound of a training run.

    The APC correction is not optional dressing
    -------------------------------------------
    Raw MI estimated from a finite alignment is **biased upward**, and the bias
    grows with the entropy of the two columns -- 64 sequences spread over a
    20x20 joint table leaves most cells empty, and emptier tables look more
    informative than they are. Measured here, that bias alone put ~0.75 nats on
    every pair and made high-entropy positions rank above genuinely coupled
    ones.

    Average Product Correction subtracts the part of MI(i, j) predicted by the
    row and column averages:

        MI_apc(i, j) = MI(i, j) - MI(i, .) * MI(., j) / MI(., .)

    It is the standard correction in the coevolution literature (Dunn et al.
    2008) for exactly this reason, so using it here is what the field does, not
    a convenience.

    Returns a symmetric ``[n_res, n_res]`` array with a zero diagonal.
    """
    n_seq, n_res = msa.shape
    a = N_AMINO_ACIDS

    # One-hot counts per column: [n_res, a]
    onehot = np.zeros((n_res, n_seq, a))
    for p in range(n_res):
        col = msa[:, p]                              # [n_seq]
        ok = col < a                                 # ignore gap/unknown/mask
        onehot[p, np.arange(n_seq)[ok], col[ok]] = 1.0

    counts = onehot.sum(axis=1)                       # [n_res, a]
    p_single = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

    # Joint counts: [n_res, n_res, a, a]
    joint = np.einsum("isa,jsb->ijab", onehot, onehot)
    joint = joint / np.maximum(joint.sum(axis=(2, 3), keepdims=True), 1e-12)

    outer = p_single[:, None, :, None] * p_single[None, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * (np.log(joint) - np.log(outer))
    mi = np.nansum(np.where(joint > 0, term, 0.0), axis=(2, 3))
    np.fill_diagonal(mi, 0.0)

    if apc:
        # Average Product Correction. Row/column means exclude the diagonal,
        # which is why the sums divide by (n_res - 1) rather than n_res.
        denom = max(n_res - 1, 1)
        row_mean = mi.sum(axis=1) / denom
        total_mean = mi.sum() / max(n_res * denom, 1)
        if total_mean > 1e-12:
            mi = mi - np.outer(row_mean, row_mean) / total_mean
        np.fill_diagonal(mi, 0.0)
    return mi


def signal_strength(
    cfg: SyntheticConfig, seed: int = 0
) -> dict[str, float]:
    """
    Summarise how separable contacts are from non-contacts, by MI alone.

    Returns ``mi_contact``, ``mi_background``, ``separation`` (their difference
    in nats -- a difference rather than a ratio, because APC-corrected MI is
    signed) and ``top_l_precision`` -- the fraction of the top-L scoring pairs
    that are real contacts, L = n_res, which is the standard metric in the
    contact prediction literature and the one that actually decides whether
    this data is usable.
    """
    rng = np.random.default_rng(seed)
    contacts = make_contact_map(cfg, rng)
    msa = make_msa(contacts, cfg, rng)
    mi = column_mutual_information(msa)

    m = eval_mask(cfg)
    scored = m & ~np.eye(cfg.n_res, dtype=bool)
    pos = contacts & scored
    neg = (~contacts) & scored

    mi_pos = float(mi[pos].mean()) if pos.any() else 0.0
    mi_neg = float(mi[neg].mean()) if neg.any() else 0.0

    # Precision at K, where K is the number of contacts that actually exist.
    #
    # NOT top-L with L = n_res, which is the field convention: real proteins
    # have roughly L long-range contacts, but this generator plants fewer
    # (residues take part in at most one pair, so n_res=48 caps the count at
    # 24 and dedup lands near 13). Scoring the top 48 against 13 possible hits
    # caps precision at 13/48 = 0.271 no matter how perfect the signal is --
    # which is exactly the plateau the first run of this file printed for
    # every coupling >= 0.5. A ceiling imposed by the metric looks identical
    # to a ceiling imposed by the data, and reporting it would have understated
    # the signal by a factor of four.
    tri = np.triu(scored, k=1)
    order = np.argsort(-mi[tri])
    labels = contacts[tri][order]
    k = int(contacts[tri].sum())
    top_k = labels[:k]
    precision = float(top_k.mean()) if top_k.size else 0.0

    return {
        "mi_contact": mi_pos,
        "mi_background": mi_neg,
        "separation": mi_pos - mi_neg,
        "top_k_precision": precision,
        "n_contacts": float(k),
        "base_rate": float(contacts[tri].mean()),
    }


# =============================================================================
# Demonstration
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic coevolving MSAs: show the signal is real, and "
                    "that the counterexample has none."
    )
    parser.add_argument("--n-res", type=int, default=48)
    parser.add_argument("--n-seq", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    print("=" * 78)
    print("  SYNTHETIC MSAs: DOES THE DATA CARRY A SIGNAL?")
    print("=" * 78)
    print(
        "  Measured by mutual information between MSA columns -- no model, no\n"
        "  training. If contacts are not separable here, they are not\n"
        "  recoverable by anything.\n"
    )

    print(f"  {'coupling':>9}  {'MI contact':>11}  {'MI background':>14}"
          f"  {'gap (nats)':>11}  {'prec@K':>11}")
    print("  " + "-" * 74)
    for coupling in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfg = SyntheticConfig(
            n_res=args.n_res, n_seq=args.n_seq, coupling=coupling
        )
        s = signal_strength(cfg, seed=args.seed)
        print(
            f"  {coupling:>9.2f}  {s['mi_contact']:>11.4f}  "
            f"{s['mi_background']:>14.4f}  {s['separation']:>11.4f}"
            f"  {s['top_k_precision']:>10.3f}"
        )

    base = signal_strength(
        SyntheticConfig(n_res=args.n_res, n_seq=args.n_seq), seed=args.seed
    )["base_rate"]
    print(f"\n  base rate (random guessing prec@K): {base:.3f}")

    print("\n  Alignment depth, at full coupling:")
    print(f"  {'n_seq':>9}  {'gap (nats)':>11}  {'prec@K':>11}")
    print("  " + "-" * 40)
    for n_seq in (1, 2, 8, 32, 128, 512):
        cfg = SyntheticConfig(n_res=args.n_res, n_seq=n_seq, coupling=1.0)
        s = signal_strength(cfg, seed=args.seed)
        print(
            f"  {n_seq:>9}  {s['separation']:>11.4f}"
            f"  {s['top_k_precision']:>10.3f}"
        )

    print(
        "\n  Read the two tables together:\n"
        "    coupling 0.0  -> contacts and background are indistinguishable.\n"
        "                     This is the permanent counterexample, and the\n"
        "                     test suite asserts a model FAILS on it.\n"
        "    n_seq    1    -> one sequence has no statistics. Depth is not a\n"
        "                     nice-to-have; it IS the evidence.\n"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
