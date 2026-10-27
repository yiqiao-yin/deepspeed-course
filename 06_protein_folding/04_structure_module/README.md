# Structure module — a symmetry you can prove, or one you have to learn

`02_evoformer` and `03_pairformer` produce a pair representation: the model's
hypothesis about which residues touch. This folder turns that into
coordinates, and the interesting part is a decision AlphaFold3 made that
AlphaFold2 did not.

A protein has no preferred position or orientation. Rotate it, move it, and it
is the same molecule. So any model that predicts coordinates must satisfy

```
predict(R x + t) == R predict(x) + t
```

for every rotation `R` and translation `t`. There are exactly two ways to get
that, and they are not equally verifiable.

---

## The measurement

`uv run structure.py`, error relative to the size of the molecule, three
seeds, float64:

| head | equivariance error | guaranteed? |
|---|---|---|
| `ipa` | **3.6e-16** | **yes — by construction** |
| `diffusion` | 4.5e-02 | no — must be learned |
| `mlp` | 2.7e-01 | no — and not trying |

**IPA's figure is float noise.** AlphaFold2's Invariant Point Attention
compares points only after mapping them into per-residue local frames, and the
distance between two points in a shared frame cannot change when the whole
molecule moves. It is arithmetic, not training. The model never saw a rotation
and does not need to.

**AlphaFold3 removed IPA.** Its diffusion module uses ordinary attention
blocks over atom coordinates, which are invariant to nothing, and recovers the
symmetry by randomly rotating and translating every training example.

### Does the augmentation work? Measured, on hardware

| arm | FAPE | vs do-nothing | equivariance (trained) |
|---|---|---|---|
| `--head ipa` | **1.7972** | **+18.8%** | **1.086e-15** |
| `--head diffusion` | 2.2785 | −3.0% | 5.075e-02 |
| `--head diffusion --augment` | 2.2917 | −3.6% | **7.807e-03** |
| `--head mlp` | 2.2662 | −2.4% | 3.335e-01 |

Three things to read off that:

1. **Augmentation genuinely works** — 5.1e-02 → 7.8e-03, a 6.5× reduction.
   AlphaFold3's strategy is not hand-waving.
2. **And it is still not a guarantee.** Thirteen orders of magnitude above
   IPA. A guarantee holds on inputs you never tested; a learned symmetry holds
   on inputs resembling the ones you did, and degrades quietly elsewhere —
   which a held-out split drawn from the same distribution cannot detect.
3. **Only IPA learns the task at all here**, and that needs scoping. This lab
   gives the head *no* sequence and *no* pair features (`s` and `z` are
   zeros), so geometry is the only signal — and reading geometry is exactly
   what IPA can do and plain attention cannot. Inside a full AlphaFold the
   trunk supplies rich features the other heads could use, and the gap would
   be far smaller. **This is a statement about this lab, not about AF2 versus
   AF3.**

---

## So why did AlphaFold3 give up the guarantee?

Generality, not carelessness.

IPA needs a residue frame built from N, CA and C atoms. AlphaFold3 predicts
ligands, ions, nucleic acids and modified residues — most of which have no
backbone and therefore no frame to build. Dropping the architectural guarantee
is what let the model treat everything as atoms.

That is the real trade, and it is worth stating plainly: **a symmetry you can
prove, against a model that can represent more things.** Neither choice is
obviously right, which is why both shipped.

---

## Quick start

```bash
cd 06_protein_folding/04_structure_module
uv sync

uv run structure.py                     # the comparison      (CPU, ~1 min)
uv run cath_data.py --split validation  # the data + chemistry (240 MB)

uv run deepspeed --num_gpus=1 train_structure_ds.py --head ipa
uv run deepspeed --num_gpus=1 train_structure_ds.py --head diffusion --augment
uv run deepspeed --num_gpus=1 train_structure_ds.py --data cath --head ipa
```

### The task

Take a backbone, perturb it by `--noise` Angstroms, and put it back — which is
the structure module's real job, and literally the job of AF3's diffusion
head. Scored by **FAPE**: predicted and true coordinates compared inside each
residue's own frame, so a globally rotated but otherwise perfect prediction
costs nothing.

Every result is printed against a **do-nothing baseline** — the FAPE of the
noisy input, unrefined. A model that does not beat that has learned nothing,
and no absolute FAPE number tells you which side of the line you are on.

---

## Hardware

| | |
|---|---|
| Declared | **24 GB, 1 GPU** |
| Actually used | **0.16 GB** — the cheapest lab in the section; no cubic term here |
| Precision | **fp32, deliberately** |

The precision choice is not inherited. This lab's whole subject is the
difference between 1e-16 and 1e-02, and bf16 has about three decimal digits of
mantissa — a reduced-precision run cannot represent the guarantee it is
supposed to be demonstrating, and the IPA arm would show a "nonzero" error
that is purely dtype noise. Its sibling folders enable bf16 because
`DS4Sci_EvoformerAttention` requires it; there is no such kernel here.

```bash
uv run runpod/runpod_ctl.py run 06_protein_folding/04_structure_module --dry-run
uv run runpod/runpod_ctl.py run 06_protein_folding/04_structure_module \
    --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # an abandoned pod bills until terminated
```

---

## The data, and three attempts at it

`--data synthetic` (default) builds backbones from **alpha helices and beta
strands** — rise 1.5 Å and ~100°/residue for helices, ~3.3 Å extended for
strands — concatenated at random orientations.

It took three tries, and the failures are the useful part:

| generator | improvement over do-nothing |
|---|---|
| random walk, 3.8 Å steps | **+1.0%** |
| helix/strand CA trace, N and C at fixed global offsets | **−0.7%** |
| helix/strand CA trace, N and C in each residue's local frame | **+18.8%** |
| *(real CATH backbones, for comparison)* | *+33.1%* |

- A **random walk has nothing to denoise toward**. The true positions are
  themselves random, so +1.0% is not a weak model — it is close to the
  information-theoretic ceiling for that data. This is the
  `01_basics/02_convnet` bug in a lab coat, and the thing that exposed it was
  running the same lab on real CATH data and getting +33%.
- Structure in the CA trace **was not enough**. With N and C at constant
  global offsets, every frame pointed the same way regardless of where the
  chain went, so the frames carried no local information and the model did
  *worse* than nothing. The frames have to track the backbone.

`--data cath` is [`ajiang2025/cath-4.3-backbone`](https://huggingface.co/datasets/ajiang2025/cath-4.3-backbone),
CC-BY-4.0, 240 MB, 16,691 / 1,528 / 1,880 chains.

> **128 is a cap, not a fixed length.** The dataset card says "cropped to a
> fixed 128-residue window", which reads as uniform and is not: lengths run
> **40–128**, with 12,806 of 16,691 training chains at the cap. Default
> collation cannot batch mixed lengths, so the loader filters to exactly 128
> and reports what it dropped. `tests/test_cath_source.py` pins this so nobody
> re-reads the card and removes the filter.

---

## Environment & Local Testing

```bash
cd 06_protein_folding/04_structure_module
uv sync
uv run structure.py                      # CPU, ~1 min, no download
uv run cath_data.py --split validation   # 240 MB, checks the chemistry
```

From the repository root:

```bash
uv run tests/test_structure_module.py    # 14 checks, ~11 s
uv run tests/test_cath_source.py         # 11 checks, needs the network
./tests/run_all.sh
```

### What the tests assert

| Property | Why it matters |
|---|---|
| IPA equivariance exact (< 1e-12) | the guarantee, as arithmetic |
| AF3-style head measurably **not** equivariant | without this, the check above is free |
| three heads order `ipa < diffusion < mlp` | a mislabelled middle arm would show up here |
| augmentation shrinks the error **but does not close it** | both halves of AF3's strategy |
| FAPE invariant when the **prediction alone** is transformed | the property that separates FAPE from RMSD |
| frames orthonormal, det **+1**, equivariant | a reflection is a different molecule |

Four sabotages were run before the suite was trusted — IPA leaking a global
coordinate, the AF3 head made accidentally invariant, FAPE replaced by raw
RMSD, and frames built as reflections. The third found a defect **in the
suite**: the FAPE check originally transformed prediction *and* truth
together, which raw RMSD also survives, so it was vacuous. Table in the
suite's docstring.

---

## References

- Jumper et al. 2021, Nature 596 — Supplementary Algorithm 21 (frames),
  22 (IPA), 23 (backbone update), 28 (FAPE). Named to match in `structure.py`.
- Abramson et al. 2024, Nature 630 — the diffusion module that replaced IPA.
- [Flash Invariant Point Attention](https://arxiv.org/pdf/2505.11580) — on
  making IPA cheap enough that the guarantee is not a performance tax.
