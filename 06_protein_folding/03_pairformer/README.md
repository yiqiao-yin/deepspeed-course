# Pairformer — AlphaFold3 deleted the MSA representation, and the wall stayed

**Read [`../02_evoformer`](../02_evoformer/) first.** This folder's result is
the *difference* between the two trunks, and it does not mean anything on its
own.

AlphaFold3 replaced the Evoformer with the **Pairformer**, and the headline
change is a deletion: the MSA representation is gone from the trunk. What
survives is a single (sequence) representation and the pair representation.

The obvious conclusion is that the memory problem got solved. It did not.

---

## The number

`uv run pairformer.py`, per block, at AlphaFold's own widths (N_res=384,
N_seq=128, bf16):

| tensor | Evoformer (AF2) | Pairformer (AF3) |
|---|---|---|
| MSA representation | 6.3 MB | — |
| MSA row attention logits | 151.0 MB | — |
| single representation | — | 0.3 MB |
| pair representation | 37.7 MB | 37.7 MB |
| **triangle attention logits** | **453.0 MB** | **453.0 MB** |
| **TOTAL** | **648.0 MB** | **491.0 MB** |

AF3 saves **24.2%** per block. Genuinely worth having — and note where it
comes from: not the MSA representation itself (6.3 MB) but its *row attention
logits* (151 MB), a `[N_seq, heads, N_res, N_res]` tensor. AF3's
pair-weighted averaging has no query–key product at all.

And the bottom row is **identical**, because the Pairformer runs the same four
triangle operations the Evoformer does.

## The lesson

| N_res | AF2 total | AF3 total | AF3 saving | cubic share of AF3 |
|---|---|---|---|---|
| 128 | 39.8 MB | 21.1 MB | **47.1%** | 79.6% |
| 256 | 222.3 MB | 151.2 MB | 32.0% | 88.8% |
| 512 | 1417.7 MB | 1141.2 MB | 19.5% | 94.1% |
| 1024 | 9948.9 MB | 8859.2 MB | **11.0%** | 97.0% |

At 128 residues the MSA deletion halves the trunk. At 1024 it is worth 11%,
because the term it was competing with grows eight times faster per doubling.

> **A constant-factor architectural saving loses to an asymptote, always,
> eventually.** O(N_res³) before, O(N_res³) after.

Which is why `DS4Sci_EvoformerAttention` matters *just as much* to AF3-class
models as to AF2-class ones. The kernel's name does not say so, and that is
the single most useful thing to carry out of this folder.

---

## Quick start

```bash
cd 06_protein_folding/03_pairformer
uv sync

uv run pairformer.py          # the comparison               (CPU, ~1 min)
uv run synthetic_msa.py       # the data                     (CPU, seconds)

uv run deepspeed --num_gpus=1 train_pairformer_ds.py
uv run deepspeed --num_gpus=1 train_pairformer_ds.py --ds-evoformer-attn
```

`train_pairformer_ds.py` is deliberately `train_evoformer_ds.py` with one
substitution — `PairformerStack` for `EvoformerStack`. Same data, same loss,
same metric, same configs, same seeds. A comparison between architectures is
only worth something when everything except the architecture is held fixed.

---

## What actually changed, block by block

| | Evoformer (AF2) | Pairformer (AF3) |
|---|---|---|
| carries | `m[N_seq, N_res]`, `z[N_res, N_res]` | `s[N_res]`, `z[N_res, N_res]` |
| MSA in the trunk | yes, 48 blocks | **no** |
| MSA module | — | 4 blocks, then discarded |
| MSA row attention | gated self-attention | **removed** — pair-weighted averaging instead |
| triangular multiplicative update | out + in | out + in *(unchanged)* |
| triangle attention | start + end | start + end *(unchanged)* |
| single-rep attention | — | `AttentionPairBias` |

`MSAModule.forward` returns **only** the pair representation. That return type
is the architecture: after it, no tensor in the model has an N_seq axis, and
`tests/test_pairformer.py` asserts that by watching all 106 tensors the trunk
actually allocates rather than by reading a signature.

**MSA pair-weighted averaging** (AF3 Algorithm 10) is the operation that
replaced row-wise gated self-attention. Each MSA row is averaged along the
residue axis using weights read off the pair representation:

$$
w_{ij} = \mathrm{softmax}_j\big(\mathrm{Linear}(z_{ij})\big), \qquad
m_{si} \leftarrow g_{si} \odot \sum_j w_{ij}\, v_{sj}
$$

The MSA never computes its own query–key product. That is why the module can
be four blocks instead of forty-eight, and why the 151 MB line above
disappears.

---

## Why this is a separate folder from `02_evoformer`

`CONTRIBUTING.md` §2 says a *family* of related methods belongs in one folder
behind a flag — the `03_llms/05_dpo --method` precedent, where six folders
would have been six copies of one file. Two folders here is a deliberate
exception, argued on the repo's own terms: `04_reward_model`, `05_dpo` and
`07_online_dpo` are separate because their **memory profiles genuinely
differ**, and that is exactly what differs here. Memory profile is this
course's subject.

**The cost, stated plainly:** the triangle operations are written twice, by
hand. A fix applied to one copy and not the other would be silent — both
produce correct shapes either way. So `tests/test_pairformer.py` re-asserts
permutation equivariance, row invariance and triangle closure *independently*
of `tests/test_evoformer.py` rather than assuming they are covered.

---

## Hardware

| | |
|---|---|
| Declared | **24 GB, 1 GPU** |
| Why 1 GPU | Same reason as `02_evoformer`: the bottleneck is an activation every rank pays in full |
| bf16 | Required for `--ds-evoformer-attn` (no fp32 path). Ampere or newer |

```bash
uv run runpod/runpod_ctl.py run 06_protein_folding/03_pairformer --dry-run

uv run runpod/runpod_ctl.py run 06_protein_folding/03_pairformer \
    --collect --wait --terminate --yes

uv run runpod/runpod_ctl.py pods      # an abandoned pod bills until terminated
```

---

## Expected output

### `uv run pairformer.py` — verified on CPU

The two tables above, plus:

```
  THE SAME TWO SYMMETRIES, ON THE AF3 TRUNK
  residue permutation  -> equivariance error 3.28e-07
  sequence permutation -> invariance   error 3.58e-07
  contact logit spread -> std          0.6236
```

Both symmetries hold for the same reasons they hold in `02_evoformer`. **The
MSA deletion changed the cost, not the symmetries.**

### `uv run deepspeed --num_gpus=1 train_pairformer_ds.py`

**Not yet verified on hardware.** The CPU modules and the full logic suite
have been run; the GPU path — peak memory with and without the kernel, and
throughput against `02_evoformer` — has not been measured on the declared
24 GB card. No numbers for it appear here rather than estimated ones.

The honest comparison to make once you have a card: run `02_evoformer` and
`03_pairformer` at the *same* `--n-res` and `--n-seq`, with and without
`--ds-evoformer-attn`, and check the four peak-memory figures against the
table at the top of this file.

---

## Environment & Local Testing

```bash
cd 06_protein_folding/03_pairformer
uv sync
uv run pairformer.py           # CPU, ~1 min, no download
uv run synthetic_msa.py        # CPU, seconds
uv run nanofold_data.py --shards 1   # ~33 MB, verifies the real data
```

Dependencies: `torch`, `deepspeed>=0.19`, `numpy`, plus `pyarrow` and
`huggingface-hub` for `--data nanofold` only. No `transformers`, no OpenFold.

`synthetic_msa.py` and `nanofold_data.py` are **byte-identical copies** of
`02_evoformer`'s. That is the no-shared-module rule: each folder must run
without the other existing.

From the repository root:

```bash
uv run tests/test_pairformer.py    # 19 checks
./tests/run_all.sh
```

Four sabotages were run against this suite before it was trusted — MSA
smuggled back into the trunk, `MSAModule` returning `(z, m)`, a claim that AF3
halves the cubic term, and the MSA row attention logits omitted from the
table. The last reproduces a bug that actually shipped in this folder. Table
in the suite's docstring.

---

## References

- Abramson et al. 2024, *Accurate structure prediction of biomolecular
  interactions with AlphaFold 3*, Nature 630. Algorithm 8 (MSA module),
  Algorithm 10 (pair-weighted averaging), Algorithm 17 (Pairformer stack) and
  Algorithm 24 (attention with pair bias) are the blocks in `pairformer.py`.
- [DS4Sci_EvoformerAttention](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/) —
  named for AF2, relevant to both.
- [Architectural highlights of AlphaFold3](https://www.blopig.com/blog/2024/08/architectural-highlights-of-alphafold3/)
