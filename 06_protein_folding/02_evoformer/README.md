# Evoformer — the AlphaFold2 trunk, and the memory wall ZeRO cannot move

The pair representation is `O(N_res²)`. Triangle attention is `O(N_res³)`.
Neither is a parameter, so **no ZeRO stage touches either of them** — which is
why DeepSpeed ships a *kernel* for this model family rather than a sharding
strategy.

That sentence is the whole topic. Everything below is built to make it
concrete rather than quotable.

---

## What you get

| File | What it is |
|---|---|
| `evoformer.py` | The trunk, from scratch, **runs on CPU**. Triangle multiplicative update, triangle attention, MSA row/column attention, outer product mean — plus the memory table |
| `synthetic_msa.py` | Coevolving MSAs, **runs on CPU**. Measures its own signal by mutual information, and ships a counterexample with none |
| `nanofold_data.py` | Real MSAs and real contacts from `ChrisHayduk/nanofold-public` (CC-BY-4.0, 1.02 GB) |
| `train_evoformer_ds.py` | DeepSpeed training: contact prediction, with the `DS4Sci_EvoformerAttention` toggle |
| `ds_config.json` | ZeRO-1 + bf16. Read the `_zero_comment` — the stage choice is the lesson |
| `ds_config_z3.json` | ZeRO-3, provided so you can **measure that it does not help** |

**Start with the two CPU files.** Most of the teaching is in them, they need no
GPU and no download, and they take about a minute each.

---

## Why this is in a DeepSpeed course

Every other example in this course has its memory problem in the *parameters* —
weights, gradients, optimizer state — which is exactly what ZeRO shards. This
one does not. The Evoformer trunk here is ~100k parameters; the thing that
fills the card is an activation, and it grows with the cube of the protein
length.

Printed by `uv run evoformer.py`, at AlphaFold2's real `c_z=128`, 4 heads, bf16:

| N_res | pair rep | triangle logits | ratio |
|---|---|---|---|
| 128 | 4.2 MB | 16.8 MB | 4.0× |
| 256 | 16.8 MB | 134.2 MB | 8.0× |
| 512 | 67.1 MB | 1073.7 MB | 16.0× |
| 1024 | 268.4 MB | 8589.9 MB | 32.0× |

Two of those logit tensors per block, 48 blocks in AlphaFold2 proper.

DeepSpeed's answer is [`DS4Sci_EvoformerAttention`](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/),
from the DeepSpeed4Science × OpenFold collaboration: it tiles the attention and
never materialises the cubic tensor. Reported at **13× peak memory reduction**
for OpenFold. It lives in the DeepSpeed this lab already locks:

```python
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
```

---

## Quick start

```bash
cd 06_protein_folding/02_evoformer
uv sync

uv run evoformer.py           # the trunk + the memory table   (CPU, ~1 min)
uv run synthetic_msa.py       # does the data carry signal?    (CPU, seconds)

uv run deepspeed --num_gpus=1 train_evoformer_ds.py
uv run deepspeed --num_gpus=1 train_evoformer_ds.py --ds-evoformer-attn
```

### The three runs worth making, in order

1. **Baseline.** Note peak memory and precision@K.
2. **`--ds-evoformer-attn`.** Peak memory should fall, and the longer the
   protein the more it falls.
3. **`--deepspeed_config ds_config_z3.json`.** ZeRO-3 shards the parameters and
   does essentially nothing for peak memory. By now this course has trained
   the "reach for stage 3" reflex into you; this is the case where it is the
   wrong reflex, and seeing it fail is worth more than being told.

`--n-res` is the knob that will run you out of memory. That is deliberate — it
is the cubic term.

---

## Hardware

| | |
|---|---|
| Declared | **24 GB, 1 GPU** |
| Why 1 GPU | The bottleneck is an activation every rank pays in full. More ranks does not help; a fused kernel does. |
| bf16 | **Required** for `--ds-evoformer-attn` — the kernel has no fp32 path. Needs Ampere (A100 / RTX 3090) or newer. On Volta/Turing switch `ds_config.json` to fp16. |

```bash
# See exactly what would be provisioned and run, without spending anything:
uv run runpod/runpod_ctl.py run 06_protein_folding/02_evoformer --dry-run

# The real thing. --terminate is driven from YOUR machine in a finally, so an
# exception here still tears the pod down.
uv run runpod/runpod_ctl.py run 06_protein_folding/02_evoformer \
    --collect --wait --terminate --yes

uv run runpod/runpod_ctl.py pods      # confirm it is gone
```

Always `--dry-run` first on a lab you have not rented before, and always check
`pods` afterwards: an abandoned pod bills until it is terminated.

---

## The data, and why it is built the way it is

### `--data synthetic` (default)

Real MSAs carry one thing the Evoformer eats: **coevolution.** Residues that
touch in 3D mutate together across evolution, and that covariance is the only
evidence AlphaFold has about contacts.

So the generator does not plant "some learnable pattern" — it plants exactly
that statistical structure. Contacting column pairs are coupled; everything
else is independent.

Measured by `uv run synthetic_msa.py` (48 residues, depth 64, APC-corrected
mutual information, precision at K where K is the true contact count, base rate
**0.014**):

| coupling | MI gap (nats) | prec@K |
|---|---|---|
| 0.00 | −0.069 | **0.000** |
| 0.25 | 0.119 | 0.385 |
| 0.50 | 0.407 | 0.923 |
| 1.00 | 1.110 | 1.000 |

| depth | MI gap | prec@K |
|---|---|---|
| 1 | 0.000 | 0.077 |
| 2 | −0.000 | 0.000 |
| 8 | 0.387 | 0.462 |
| 32 | 0.870 | 1.000 |

Read those two tables together. `coupling=0.0` is the **permanent
counterexample** — data with no signal, which
`tests/test_evoformer_data_is_learnable.py` asserts a model *fails* on. And
depth 1 is unlearnable at perfect coupling, because coevolution is a property
of a population, not a sequence. **That is why MSAs exist.**

Two design decisions worth knowing before you change anything:

- **Only long-range contacts (|i−j| ≥ 6) are planted or scored.** In a real
  protein residues i and i+1 always touch, so a model can score well by reading
  the index difference and ignoring the MSA entirely. Excluding the band means
  coevolution is the only route to a good score.
- **The base rate is printed with every result.** Contact maps are sparse —
  ~1–3% positive — so 0.9 accuracy is worse than predicting "no contact"
  everywhere. Compare against the base rate, never against 0.5.

### `--data nanofold`

Real chains: [`ChrisHayduk/nanofold-public`](https://huggingface.co/datasets/ChrisHayduk/nanofold-public),
CC-BY-4.0, 1.02 GB, derived from OpenProteinSet/OpenFold. Verified by loading
one shard:

```
rows parsed          : 345
chain length         : 40-256 (median 145)
MSA depth            : 1-2048 (median 984)
usable at n_res=64   : 323 chains (19 too short, 3 unresolved)
long-range base rate : 0.0310
```

Contacts are Cα pairs within 8 Å. The CASP convention uses Cβ, which is
slightly better — but Cβ means reading `atom14_positions`, whose atom ordering
has **not** been verified here, so this lab uses the column whose meaning is
documented and says so rather than implying otherwise.

> **Do not use `load_dataset(streaming=True)` on this dataset.** Measured: no
> rows after 15 minutes, killed at the timeout, while `hf_hub_download` of one
> shard took **1.9 s (17.5 MB/s)**. The bottleneck is the streaming machinery
> over nested array columns, not bandwidth. `01_basics/03_convnet_cifar10`
> already shipped once as a lab that could not finish because its data source
> was too slow — see `POSTMORTEMS.md`.

---

## Expected output

### `uv run evoformer.py` — verified on CPU

```
  TRIANGLE CLOSURE:  evidence on (i,k) and (j,k) must reach (i,j)
  correct  sum_k a_ik * b_jk    ->  change at (i,j) = 9.495e-01
  broken   a_ij * b_ij          ->  change at (i,j) = 0.000e+00

  TWO SYMMETRIES, ONE VACUITY CHECK
  residue permutation  -> equivariance error 2.38e-07
  sequence permutation -> invariance   error 2.38e-07
  contact logit spread -> std          0.4090  (must be > 0)
```

### `uv run deepspeed --num_gpus=1 train_evoformer_ds.py`

**Not yet verified on hardware.** This lab has been run end to end on CPU and
through its full logic suite, but the GPU path — peak memory with and without
`--ds-evoformer-attn`, throughput, and the ZeRO-3 comparison — has not been
measured on the declared 24 GB card. The numbers are deliberately absent rather
than estimated: a published figure a reader cannot reproduce costs them a day
deciding their own correct setup is broken.

What the script prints is precision@K on **held-out** chains against the base
rate, plus peak GPU memory and the kernel status. A capped run
(`--max-steps N`) says so explicitly and tells you not to read its precision as
a result.

---

## A result worth not overclaiming

On this synthetic data, classical APC-corrected mutual information reaches
prec@K **1.000**, while the trunk in `tests/test_evoformer_data_is_learnable.py`
— one block, 150 steps, CPU — reaches **0.313** against a base rate of 0.023.

That is 13× the base rate, so the model is genuinely learning. But it is *below
the classical baseline*, and the honest reason is that this generator plants
**purely pairwise** coupling, which is precisely what MI is optimal for. The
Evoformer's advantage is indirect and higher-order coupling on real alignments,
which the synthetic arm does not contain by construction.

So: the synthetic path proves the plumbing, the symmetries and the counter-
example. It does not prove the architecture is better than a 2008 statistic,
and this README will not pretend otherwise. Use `--data nanofold` for that
argument.

---

## Environment & Local Testing

```bash
cd 06_protein_folding/02_evoformer
uv sync                        # installs the committed lock
uv run evoformer.py            # CPU, ~1 min, no download
uv run synthetic_msa.py        # CPU, seconds, no download
uv run nanofold_data.py --shards 1   # ~33 MB download, verifies the real data
```

Dependencies: `torch`, `deepspeed>=0.19` (for `DS4Sci_EvoformerAttention`),
`numpy`, and `pyarrow` + `huggingface-hub` for the `--data nanofold` path only.
No `transformers`, no OpenFold.

From the repository root:

```bash
uv run tests/test_evoformer.py                     # 16 property checks
uv run tests/test_evoformer_data_is_learnable.py   # 6 checks, ~18 s
./tests/run_all.sh                                 # everything
```

### What the tests assert, and why

| Property | Why a shape check misses it |
|---|---|
| Residue-permutation **equivariance** | A model reading residue order scores well on fixed ordering and generalises to nothing. `02_intermediate/04_groupwise_ranking` shipped exactly this |
| Non-query MSA row **invariance** | Homolog order carries no information. Asserted in both directions — permuting the *query* must change the output, or the check is vacuous |
| **Triangle closure** | `TriangleMultiplicativeUpdate(broken_no_third_index=True)` is kept permanently: it runs, trains, returns the right shape, and has no triangle in it |
| The **cubic exponent** | Asserted from the real einsum, not the docstring. If it is not 8× per doubling, the memory argument above is wrong |
| Data **learnability** | On a held-out split, using the lab's own model — plus `coupling=0.0` asserted to FAIL |

Every one of these was run against a deliberately sabotaged trunk before being
trusted; the sabotage table is in the docstring of `tests/test_evoformer.py`.

---

## Shutting down

```bash
uv run runpod/runpod_ctl.py pods         # list
uv run runpod/runpod_ctl.py stop <id>    # terminate
```

Termination is driven from your machine in a `finally`; the pod never receives
`RUNPOD_API_KEY`. See `SECURITY.md`.

---

## References

- Jumper et al. 2021, *Highly accurate protein structure prediction with AlphaFold*,
  Nature 596. Supplementary Algorithms 6–15 are the blocks in `evoformer.py`,
  named to match.
- [DS4Sci_EvoformerAttention](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/)
  and the [DeepSpeed4Science initiative](https://www.deepspeed.ai/deepspeed4science/).
- Dunn et al. 2008, for the average product correction used in `synthetic_msa.py`.
- [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) —
  the best visual walkthrough of the trunk.
