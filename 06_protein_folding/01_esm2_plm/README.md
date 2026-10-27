# ESM-2 — a protein is a sequence, so the whole LLM stack transfers

The on-ramp to `06_protein_folding/`. Before triangle attention and rigid
frames there is a much more familiar object: a string over a 20-letter
alphabet, and a BERT trained on it. ESM-2 *is* BERT, pretrained on UniRef
instead of Wikipedia, and everything you know about fine-tuning transfers.

Almost. The parts that do not are the interesting ones.

---

## What transfers, and what does not

**Transfers unchanged:** tokenizer, masked-LM objective, LoRA, ZeRO, gradient
accumulation, mixed precision, the whole HuggingFace surface. `Trainer` does
not know the tokens are amino acids.

**Does not, and it changes the arithmetic:**

- **ESM-2 is an encoder.** No causal mask, no generation. Every position
  attends to every other, so you think in *padded* tokens per second, and
  padding is most of the bill.
- **Labels are per-residue.** The collator, not the model, is where the bugs
  live — see the alignment note below.
- **The vocabulary is 33 tokens**, against a modern LLM's 100k+. Embedding and
  output layers are almost free, so an ESM-2 of a given size has far more of
  its parameters in actual transformer blocks.

---

## Measured on hardware

1 × RTX 3080 Ti Laptop (16 GB), `--task ss`, 1,024 train / 128 held-out
chains, 2 epochs:

| `--model` | accuracy | majority baseline | margin | peak GPU | wall clock | trainable |
|---|---|---|---|---|---|---|
| 150M | 0.7928 | 0.503 | **+29.0 pts** | 3.88 GB | 185 s | all |
| 650M `--use-lora` | 0.7965 | 0.503 | +29.3 pts | 4.18 GB | 200 s | 2.03M / 653M (0.31%) |
| **3B `--use-lora`** | **0.8049** | 0.503 | **+30.2 pts** | **10.93 GB** | 581 s | 4.43M / 2.84B (0.16%) |

**Compare against the majority class, never against 1/3.** Secondary structure
is unbalanced, so a model that always predicts the most common class scores
~0.50 while learning nothing. Every run prints the baseline and the margin
beside the accuracy.

Note what scale buys here: 150M → 3B is a twenty-fold increase in parameters
for **1.2 accuracy points**. That is worth seeing before you reach for a bigger
checkpoint on a different task.

### The 3B rung: an open question, now closed

The 3B and 15B checkpoints predate safetensors and ship only
`pytorch_model-*.bin`. Whether transformers 5.x would still load them was the
one thing gating this lab, because **transformers 5.x writes safetensors
exclusively and ignores `safe_serialization=False`**.

Reading legacy checkpoints still works. Verified on **transformers 5.16.1** -- the version this lab locks:

| layout | result |
|---|---|
| single `pytorch_model.bin` | loads, parameters match |
| sharded `pytorch_model-0000N-of-0000M.bin` + index | loads, parameters match |
| **the real `facebook/esm2_t36_3B_UR50D`** | **downloads, loads, trains — 0.8049 at 10.93 GB** |

So the 3B rung is offered. If a future transformers drops the reader, the
ladder stops at 650M and `plm.py`'s docstring is where to say so.

---

## Quick start

```bash
cd 06_protein_folding/01_esm2_plm
uv sync

uv run plm.py                       # the objective and the labels (CPU, ~1 min)

uv run deepspeed --num_gpus=1 train_esm2_ds.py --task ss --model 150M
uv run deepspeed --num_gpus=1 train_esm2_ds.py --task ss --model 650M --use-lora
uv run deepspeed --num_gpus=1 train_esm2_ds.py --task mlm --model 35M
```

---

## Where the labels come from

The downstream task is 3-state secondary structure — helix, strand, or
neither — and the labels are **derived, not downloaded**.

That is partly licensing: the obvious benchmark set is CC-BY-NC-SA, which sits
badly in an MIT course, and deriving from the CC-BY-4.0 CATH backbones keeps
one permissive data spine across all four subtopics. But it is also better
teaching. Secondary structure is not a label somebody assigned; it is a fact
about backbone geometry:

```
phi(i) = dihedral( C(i-1), N(i),  CA(i), C(i)  )
psi(i) = dihedral( N(i),   CA(i), C(i),  N(i+1) )
```

Plot them and you get the Ramachandran diagram, with two dense regions: helix
near (−60, −45) and sheet near (−135, +135).

### It took two attempts, and the failure is instructive

**A (φ, ψ) lookup alone cannot find β-sheet.** The upper-left region contains
β-strand, polyproline-II *and* a great deal of ordinary extended coil, and
nothing in the two angles separates them — what makes a strand a strand is
hydrogen bonding to *another* strand, which is non-local and is exactly what
DSSP measures. The first version here used the region alone and assigned
**59.9% strand** against a published 18–25%, while its cluster centres looked
entirely reasonable.

The fix is the other constraint real assignments use: secondary structure
comes in **contiguous elements**. A helix needs at least one full turn
(~4 residues), a strand at least 3, and single-residue gaps are bridged first.

### How the derivation is validated

Not by matching a published fraction. `uv run plm.py` reports the fractions
without asserting them, because this is not DSSP *and* CATH is a domain
database enriched in β relative to whole proteomes — the familiar "~33% helix"
figure describes a different population, and tuning the region boundaries
until it matched would be fitting the derivation to the wrong target.

What is asserted is geometry:

| check | measured |
|---|---|
| helix cluster centre | (−66.5, −32.2) vs textbook (−60, −45) |
| strand cluster centre | (−119.2, 134.0) vs textbook (−135, +135) |
| **CA(i)→CA(i+4) in labelled helix** | **6.25 Å** vs one turn ≈ 6.2 |
| and in non-helix | 11.64 Å |

The third row is the strong one: it uses **no dihedral at all**, so it catches
errors that the Ramachandran check shares a cause with.

---

## The alignment that silently costs accuracy

ESM-2 prepends `<cls>`, so **residue i is at token i+1**. An off-by-one there
trains, converges, scores a few points lower, and raises nothing anywhere.
`tests/test_ss_derivation.py` asserts it for exactly that reason.

---

## Hardware

| | |
|---|---|
| Declared | **24 GB, 1 GPU** |
| 150M | 3.88 GB, full fine-tune |
| 650M | 4.18 GB with `--use-lora` |
| 3B | 10.93 GB with `--use-lora`; will OOM without it |

`ds_config.json` uses **ZeRO stage 2, and here the ordinary reasoning
applies** — which it did not in the other three folders. ESM-2 650M has 652M
*real* parameters, so optimizer state and gradients are gigabytes and sharding
them buys real memory. Compare `02_evoformer`, where the model is 100k
parameters and the stage is nearly irrelevant because the memory is in
activations. Same course, same knob, opposite conclusion.

```bash
uv run runpod/runpod_ctl.py run 06_protein_folding/01_esm2_plm --dry-run
uv run runpod/runpod_ctl.py run 06_protein_folding/01_esm2_plm \
    --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # an abandoned pod bills until terminated
```

---

## Environment & Local Testing

```bash
cd 06_protein_folding/01_esm2_plm
uv sync
uv run plm.py              # CPU; downloads 240 MB of CATH for the label demo
uv run plm.py --skip-data  # masking demo only, no download
```

Dependencies: `torch`, `deepspeed>=0.19`, `transformers==5.16.1`, `peft==0.20.0`
(pinned to match the rest of the course and `tests/test_config_kwargs.py`),
plus `pyarrow` and `huggingface-hub` for the data.

From the repository root:

```bash
uv run tests/test_ss_derivation.py    # 22 checks
./tests/run_all.sh
```

Four sabotages were run before the suite was trusted. The first — a global
dihedral sign flip — was **not caught**, because every assertion used `abs()`,
including the one named "opposite sign". The test now pins the signed value,
with the convention anchored to the empirical fact that real helices have
ψ < 0 rather than to this implementation.

---

## References

- Lin et al. 2023, *Evolutionary-scale prediction of atomic-level protein
  structure with a language model*, Science 379 — ESM-2 and ESMFold.
- [ESM Cambrian](https://www.evolutionaryscale.ai/blog/esm-cambrian) — ESM-C,
  300M/600M/6B, MIT-licensed and stronger than ESM-2 today. Not used here
  because it needs the `esm` package rather than being `transformers`-native,
  which would obscure the point that the ordinary LLM stack applies.
- Ramachandran et al. 1963, for the plot the labels come from.
