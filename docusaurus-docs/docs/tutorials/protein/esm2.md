---
sidebar_position: 39
---

# ESM-2: a protein is a sequence

The on-ramp to this section. Before [triangle attention](./evoformer.md) and
[rigid frames](./structure-module.md) there is a far more familiar object: a
string over a 20-letter alphabet, and a BERT trained on it. ESM-2 *is* BERT,
pretrained on UniRef instead of Wikipedia.

Everything you know about fine-tuning transfers. Almost — and the exceptions
are the interesting part.

> New to the section? [What goes in, what comes out](./shapes.md) states the
> input and output shapes these models work in, in about two minutes.

## What does not transfer

- **ESM-2 is an encoder.** No causal mask, no generation. Every position
  attends to every other, so you think in *padded* tokens per second, and
  padding is most of the bill.
- **Labels are per-residue**, so the collator is where the bugs live.
- **The vocabulary is 33 tokens**, against a modern LLM's 100k+. Embedding and
  output layers are almost free, so more of the parameter budget sits in
  actual transformer blocks.

## Measured

1 × RTX 3080 Ti (16 GB), 3-state secondary structure, 1,024 train / 128
held-out chains:

| model | accuracy | majority baseline | margin | peak GPU | trainable |
|---|---|---|---|---|---|
| 150M | 0.7928 | 0.503 | **+29.0 pts** | 3.88 GB | all |
| 650M + LoRA | 0.7965 | 0.503 | +29.3 pts | 4.18 GB | 2.03M / 653M |
| **3B + LoRA** | **0.8049** | 0.503 | **+30.2 pts** | 10.93 GB | 4.43M / 2.84B |

:::tip Compare against the majority class, never 1/3
Secondary structure is unbalanced. A model that always predicts the most
common class scores ~0.50 here while learning nothing. Every run prints the
baseline and the margin beside the accuracy.
:::

Note what scale buys: 150M → 3B is **twenty times the parameters for 1.2
accuracy points**. Worth seeing before reaching for a bigger checkpoint.

### The 3B checkpoint: a question that had to be answered by running it

ESM-2's 3B and 15B checkpoints predate safetensors and ship only
`pytorch_model-*.bin`. Whether transformers 5.x would load them gated this
whole lab, because transformers 5.x **writes** safetensors exclusively and
ignores `safe_serialization=False`.

Reading still works. Verified on transformers 5.16.1 (the version the lab locks) — single-file layout,
sharded-plus-index layout, and finally the real `facebook/esm2_t36_3B_UR50D`,
which downloads, loads and trains to 0.8049 at 10.93 GB with LoRA.

## Where the labels come from

They are **derived, not downloaded**. Partly licensing — the obvious benchmark
set is CC-BY-NC-SA, and deriving from the CC-BY-4.0 CATH backbones keeps one
permissive data spine across the whole section. But mostly because secondary
structure is not a label somebody assigned; it is a fact about geometry:

$$
\phi_i = \text{dihedral}\big(C_{i-1}, N_i, C\alpha_i, C_i\big)
\qquad
\psi_i = \text{dihedral}\big(N_i, C\alpha_i, C_i, N_{i+1}\big)
$$

Plot them and you get the Ramachandran diagram, with helix near $(-60, -45)$
and sheet near $(-135, +135)$.

### A (φ, ψ) lookup alone cannot find β-sheet

The first version of the derivation used the regions alone and assigned
**59.9% strand** against a published 18–25% — while its cluster centres looked
entirely reasonable. Plausible geometry, wildly wrong fractions.

The reason is real biology: the upper-left Ramachandran region contains
β-strand, polyproline-II *and* a great deal of ordinary extended coil, and
nothing in two angles separates them. What makes a strand a strand is
hydrogen bonding to *another* strand — non-local, and exactly what DSSP
measures.

The fix is the other constraint real assignments use: secondary structure
comes in **contiguous elements**. A helix needs one full turn (~4 residues),
a strand at least 3, single-residue gaps bridged first.

### How it is validated — and what is deliberately not asserted

Not the class fractions. This is not DSSP, *and* CATH is a domain database
enriched in β relative to whole proteomes, so the familiar "~33% helix" figure
describes a different population. Tuning the region boundaries until it
matched would be fitting the derivation to the wrong target.

What is asserted is geometry:

| check | measured | reference |
|---|---|---|
| helix cluster centre | (−66.5, −32.2) | (−60, −45) |
| strand cluster centre | (−119.2, 134.0) | (−135, +135) |
| **CA(i)→CA(i+4), labelled helix** | **6.25 Å** | one turn ≈ 6.2 |
| CA(i)→CA(i+4), non-helix | 11.64 Å | — |

The third row uses **no dihedral at all**, which is why it catches errors the
Ramachandran check shares a cause with.

## Two traps

**Label/token alignment.** ESM-2 prepends `<cls>`, so residue $i$ is at token
$i+1$. An off-by-one trains, converges, scores a few points lower, and raises
nothing.

**A test that cannot fail.** The dihedral unit test originally asserted
`abs(angle) == 90` and that two configurations were "opposite" — both of which
survive a *global sign flip*, so a sabotage negating every dihedral passed all
four assertions. A sign flip mirrors the Ramachandran plot and swaps which
region is helix and which is sheet, while every label stays in range. The test
now pins the signed value, anchored to the empirical fact that real helices
have ψ < 0.

## Figures

This page has none, and that is deliberate — the section's animated figures
all illustrate structure, and nothing here would be clarified by a picture of
a sequence. They live on [Evoformer](./evoformer.md) (contact maps, the cubic
memory wall, coevolution, the trunk learning) and
[Structure module](./structure-module.md) (SE(3) equivariance, prediction
superimposed on truth).

All six are regenerated by one uv-managed script, which trains nothing of its
own and modifies no lab:

```bash
uv run scripts/make_protein_animations.py            # the four analytic ones
uv run scripts/make_protein_animations.py --only trunk-refinement
uv run scripts/make_protein_animations.py --only prediction-vs-truth
```

## The ZeRO stage, for once, behaves normally

`ds_config.json` uses **stage 2, and here the ordinary reasoning applies** —
which it did not anywhere else in this section. ESM-2 650M has 652M *real*
parameters, so optimizer state and gradients are gigabytes and sharding them
buys real memory.

Compare [Evoformer](./evoformer.md), where the model is 100k parameters, the
memory is all activations, and stage 3 saves 4.6%. Same course, same knob,
opposite conclusion — because the memory is in a different place.
