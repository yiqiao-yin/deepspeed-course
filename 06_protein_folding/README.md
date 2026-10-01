# 06 — Protein folding

Structural biology as a **systems** topic. The models here are small; the
tensors are not.

## The point of this section

The rest of the course teaches one habit: when a model does not fit, shard it.
ZeRO-1 shards optimizer state, ZeRO-2 adds gradients, ZeRO-3 adds parameters,
and expert parallelism shards the experts. Every one of those moves
*parameters* off a card.

AlphaFold-class models break that habit. Their memory problem is in the
**activations**, and it scales with the length of the protein rather than the
size of the model:

| | cost | is it a parameter? |
|---|---|---|
| pair representation | O(N_res²) | no |
| triangle attention logits | **O(N_res³)** | no |

A ~100k-parameter trunk can exhaust a 24 GB card on a 512-residue chain, and
sharding the parameters across eight GPUs changes nothing, because every rank
materialises the cubic tensor in full, every step. Measured: ZeRO-1 0.65 GB
against ZeRO-3 0.62 GB, a 4.6% difference for 1.5x the communication.

`01_esm2_plm` is the control. ESM-2 650M has 652M *real* parameters, so there
the ordinary reasoning applies and stage 2 earns its keep. Same course, same
knob, opposite conclusion -- because the memory is in a different place.

This is why DeepSpeed ships
[`DS4Sci_EvoformerAttention`](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/)
— a fused kernel from the DeepSpeed4Science × OpenFold collaboration that tiles
the attention and never builds the logit tensor — instead of another sharding
strategy. Structural biology is not a guest in this course; it is the case
DeepSpeed built a kernel for.

It also extends the through-line from sections 04 and 05. ZeRO shards what the
model *is*; token compression shrinks what it *looks at*; STAR memory bounds
what it *retains*. Here the currency changes again: **the memory is in the
reasoning itself**, and the only way to shrink it is to not write it down.

## Subtopics

| Folder | Level | What it teaches |
|---|---|---|
| `01_esm2_plm` | sequence | a protein is a sequence, so the LLM stack transfers wholesale — ESM-2, LoRA, ZeRO |
| **`02_evoformer`** | pairs (AF2) | triangle operations, coevolution, and the cubic wall. **Start here** |
| `03_pairformer` | pairs (AF3) | the same trunk with the MSA representation deleted — and why the asymptote survives |
| `04_structure_module` | coordinates | pair representation → 3D, IPA vs a diffusion head, and SE(3) equivariance |

> **Status:** all four subtopics are built and verified on hardware
> (1 x RTX 3080 Ti, 16 GB). The one thing still unverified anywhere in the
> section is `DS4Sci_EvoformerAttention` itself, which needs a CUDA toolkit
> the verification box lacks -- only its fallback path is tested, and no
> memory number for the kernel appears anywhere.

## AF2 and AF3 are two folders on purpose

`CONTRIBUTING.md` §2 says a *family* of related methods belongs in one folder
behind a flag — the `03_llms/05_dpo --method` precedent, where six folders
would have been six copies of one file. The AF2/AF3 trunks are split anyway,
and the reason is the same one that split `04_reward_model` from `05_dpo` from
`07_online_dpo`: **their memory profiles genuinely differ.**

| | `02_evoformer` (AF2) | `03_pairformer` (AF3) |
|---|---|---|
| MSA representation in the trunk | yes, 48 blocks | **deleted** |
| MSA module | — | 4 blocks, no row-wise gated self-attention |
| Triangle operations | yes | yes |
| Trunk activation cost | O(N_seq × N_res²) **+** O(N_res²) | O(N_res²) |

The dominant AF2 term is not the MSA representation itself but its **row
attention logits**, `[N_seq, heads, N_res, N_res]` — 151 MB against the
representation's 6.3 MB at N_res=384, N_seq=128. AF3's pair-weighted averaging
has no query-key product, so that tensor disappears entirely.

The payoff, read across the two folders: both run the *same* triangle
operations, so the AF3 trunk is still O(N_res²) in memory and O(N_res³) in
time. Measured, the deletion is worth **24% per block at 384 residues and 11%
at 1024** — a real saving that shrinks as the cubic term takes over.
**Deleting the MSA representation removes a large constant, not the
asymptote.** That is why the kernel still matters for AF3-class models.

The price of the split is that the triangle operations are written twice. That
is the house style — this repository duplicates on purpose — but it means a
fix to one must be applied to the other, and each folder's test suite asserts
the shared properties independently so a one-sided fix is caught.

## Where ColabFold fits

Most people meet this field through a Colab notebook: ColabFold, which wraps
AlphaFold2 with an MMseqs2 MSA search and hands back a PDB file and a pLDDT
plot. It is an excellent tool and it is **inference only** — no optimizer,
nothing to distribute, nothing to train.

That is precisely why it is not a lab here. Running a pretrained folding model
teaches you what the field can do; it teaches you nothing about why it was
hard to build. This section takes the part you can actually train, at a scale
you can actually afford, and makes the expensive part visible.

## What can be run without a GPU

More than you would expect. Each subtopic ships its algorithm as a plain
PyTorch module that runs on CPU in about a minute:

```bash
cd 02_evoformer
uv run evoformer.py           # triangle ops, the two symmetries, the memory table
uv run synthetic_msa.py       # coevolution, measured — and a counterexample with none
```

This follows `03_llms/10_deepseek_from_scratch/mla.py` and
`03_llms/11_moe/moe.py`: when the substance of a topic is an algorithm rather
than weights, ship it as something a reader can run and read, beside the
DeepSpeed script that trains it.
