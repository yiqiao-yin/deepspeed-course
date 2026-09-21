# 11 · Mixture of Experts

Routing, load balancing, and expert parallelism — built from the DeepSeek-V3
paper and measured rather than asserted.

This is the companion to [`10_deepseek_from_scratch`](../10_deepseek_from_scratch/).
DeepSeek-V2 and V3 have **two** headline architectural contributions, and that
folder covers one of them. MLA compresses what the model *remembers*; MoE
changes what the model *computes*. Read them together.

---

## The one-sentence version

An MoE layer holds `N` experts and fires only `k` of them, so **parameter count
and FLOP count stop being the same number** — GLM-5.3 is ~743 B parameters
running at the cost of ~30 B. The router that decides this is 0.12 B of those
743 B, about **0.016% of the model**.

## The finding this folder exists to show

The usual telling is "without load balancing the router collapses, so balance
it." Measured on CPU over 300 steps — numbers you can reproduce with
`uv run moe.py` — that is half true, and the wrong half is the interesting one:

| experts | groups | balance | eval loss | max/min | dead | purity |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 4 | none | **0.144** | 264:1 | 0 | **0.975** |
| 16 | 4 | bias | 0.246 | 1.5 | 0 | 0.793 |
| 64 | 4 | none | **0.310** | ∞ | **12** | 0.932 |
| 64 | 4 | bias | 0.634 | 3.0 | 0 | 0.662 |

**Collapse is real, but it needs surplus.** At 64 experts for a 4-group task,
12 experts receive nothing at all — paid for, never trained. At 16 experts
nothing dies, but one expert still handles 264× the traffic of another.

**Balancing is a tax, not an improvement — at world size 1.** Read the loss
column: balancing makes the model *worse* in every configuration measured here,
and specialisation drops with it. That scope is load-bearing: **a 2-GPU run
reported the opposite**, with `--balance none` 33× worse and diverging. See
_[An unresolved disagreement](#an-unresolved-disagreement)_ below. That is not a defect in this implementation — it is the trade DeepSeek-V3
names explicitly, and the reason they went looking for a cheaper mechanism:

> "However, too large an auxiliary loss will impair the model performance."
> — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §2.1.2

**So why balance at all?** Because the unbalanced router is a better model and a
much worse *program*, and that only becomes visible once the experts live on
different GPUs. Under expert parallelism each rank owns a slice of the experts
and every rank waits at an all-to-all for the slowest one. A 264:1 imbalance is
one GPU doing 264× the work while its peers idle at a barrier.

**This is why MoE belongs in a DeepSpeed course rather than an architecture
course.** The balancing mechanism is not there to make the model better. It is
there to make the model *schedulable*.

---

## An unresolved disagreement

Everything in the table above was measured **single-process, at world size 1**.
A 2 × RTX 3090 run of `train_moe_ds.py` under DeepSpeed reported the reverse:

| world size 2, NCCL, 500 steps, one run each | eval loss |
|---|---:|
| `--balance bias` | 0.024 |
| `--balance none` | **0.795** — and *rising* through training (0.515 → 0.827) |

A rising loss is divergence, not poor specialisation, so this is a different
phenomenon rather than a louder version of the same one.

- At world size 1 the table holds across **six seeds**, no overlap between the
  groups (none 0.138–0.206, bias 0.246–0.315). Not seed luck.
- Reproducing it on **two gloo ranks on CPU**, with gradients all-reduced as
  data parallelism does, did **not** reverse the ordering (none 0.0044 vs bias
  0.0170). Plain data parallelism does not explain it.
- The 2-GPU observation is **one run per arm**.

The ordering is established at world size 1 and **unresolved above it**. If the
reversal holds under repetition, it is a stronger version of this topic's
thesis — balancing would be a requirement for convergence once experts span
ranks, not merely a tax paid for schedulability — and this folder will be
rewritten around it.

## Hardware

| | |
|---|---|
| **Minimum** | none for `moe.py` — it is plain PyTorch on CPU |
| **Recommended** | 1 × 24 GB for the training script |
| **Required for `--expert-parallel`** | **2 GPUs.** The experts are partitioned across ranks; with one rank there is nothing to partition and the script raises rather than pretending |
| Disk | ~20 GB (no model downloads; the data is synthetic) |
| Time | `moe.py` ≈ 1 min · a full training run ≈ 10 min |

---

## Environment & Local Testing

```bash
cd 03_llms/11_moe
uv sync
```

The routing lesson needs **no GPU at all**:

```bash
uv run moe.py
```

That prints the parameter accounting and runs the full six-way comparison
(3 strategies × 2 expert counts) on CPU in about a minute.

---

## Running it

```bash
# the default path: this repo's MoELayer, every expert on every rank
uv run deepspeed --num_gpus=2 train_moe_ds.py --balance bias

# the comparison that matters — run both and read the loss AND the balance
uv run deepspeed --num_gpus=2 train_moe_ds.py --balance none
uv run deepspeed --num_gpus=2 train_moe_ds.py --balance bias

# expert parallelism: experts partitioned across ranks, all-to-all dispatch
uv run deepspeed --num_gpus=2 train_moe_ds.py --expert-parallel \
    --deepspeed_config ds_config_ep.json

# cheap pipeline check
uv run deepspeed --num_gpus=2 train_moe_ds.py --max-steps 20
```

### CoreWeave (SLURM)

```bash
sbatch run_deepspeed.sh                      # full run
sbatch run_deepspeed.sh --max-steps 20       # cheap dry run
squeue -u $USER
tail -f logs/moe_<jobid>.out
```

### RunPod

```bash
uv run runpod/runpod_ctl.py recommend 03_llms/11_moe
uv run runpod/runpod_ctl.py run 03_llms/11_moe \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods            # confirm nothing is still running
```

`--terminate` destroys the pod from your machine in a `finally`, so a crash mid-run
still shuts it down. **Always confirm with `pods` afterwards** — an idle GPU bills
by the hour.

---

## Two paths, and they are different code

| | default | `--expert-parallel` |
|---|---|---|
| Implementation | `MoELayer` in `moe.py`, from the paper | `deepspeed.moe.layer.MoE` |
| Experts live | on every rank | **partitioned across ranks** |
| Dispatch | an index loop | all-to-all |
| `--balance` | applies | ignored — DeepSpeed's gate has its own |
| Shows you | the *mechanism* | the *systems consequence* |

They are not interchangeable and the script does not pretend they are.

### Expert parallelism is not ZeRO

Both split a model across GPUs. They split **different things**, and this is
the distinction the folder is here to teach:

- **ZeRO** shards optimizer state, gradients and parameters of the *same*
  model. Every rank still runs every layer, on different **data**.
- **EP** shards the **experts**. Every rank holds a different *subset of the
  model* and runs it on tokens routed to it from every other rank.

So EP introduces a communication pattern ZeRO never has — an all-to-all in the
forward and another in the backward — and its cost is set by the **busiest**
expert, not the average one.

They compose. `ds_config_ep.json` uses ZeRO stage 1 alongside EP, and the
comment in that file explains why not a higher stage.

---

## Two details this implementation gets right on purpose

Both are places where a reasonable-looking implementation silently diverges
from the paper, and in both cases **the model trains fine either way** — which
is exactly what makes them worth stating. Both are asserted in
`tests/test_moe_routing.py`.

**1. The bias steers selection only.** It decides *which* experts fire and never
touches the weight their output is multiplied by:

> "Note that the bias term is only used for routing. The gating value, which
> will be multiplied with the FFN output, is still derived from the original
> affinity score."

Fold the bias into the gate and load-balancing pressure starts perturbing the
model's actual output — reintroducing the coupling the aux-loss-free design
exists to remove.

**2. The affinity is a sigmoid, normalised among the selected experts** — not a
softmax over the top-k logits. Eq. 15 is `s = Sigmoid(u·eᵢ)`; Eq. 13 normalises
among selected scores. V2 used softmax; V3 changed it.

---

## Files

| File | What it is |
|---|---|
| `moe.py` | The algorithm. CPU-runnable, no GPU, no downloads. Start here. |
| `train_moe_ds.py` | Trains it under DeepSpeed; `--expert-parallel` for EP. |
| `ds_config.json` | ZeRO-1, fp32. The default path. |
| `ds_config_ep.json` | The expert-parallel config. Diff it against the above. |
| `run_deepspeed.sh` | SLURM batch script. |
| `../../tests/test_moe_routing.py` | 20 property assertions, no GPU. |

---

## Expected output

`uv run moe.py`, abridged:

```
  component            parameters
  ------------------ ------------
  router                    2,048
  shared experts           65,536
  routed experts        1,048,576
  ------------------ ------------
  TOTAL                 1,116,160
  ACTIVE / token          198,656   (18% of total)

  The router is 0.18% of the layer and decides where the other 99.82% goes.
```

A capped training run (`--max-steps 20`) prints its routing numbers and then
says plainly that the run was capped and they do not yet mean anything. **A
short run is a pipeline check, not a result** — success there means DeepSpeed
launched, every rank ran, and the loss moved.

> **Every path here is now verified on hardware.** The expert-parallel entry
> was run on 2 x RTX 3090 and 8/8 on 2 x H100 80 GB, with `expert-parallel`,
> `ds_config_ep` and `ep_size` markers confirmed in the output — so the
> all-to-all genuinely ran rather than the control entry. The default path was
> separately measured on 2 x A40 (eval loss 0.7036, entropy 0.933, 0 dead
> experts, purity 0.866, rc=0), which also confirms the `update_bias`
> all-reduce works under NCCL and not only under the gloo suite.

---

## References

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §2.1.2 — Eqs. 12–16, auxiliary-loss-free load balancing
- Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization*, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — fine-grained and shared experts
- Fedus et al., *Switch Transformers*, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) — the auxiliary loss
- Shazeer et al., *Outrageously Large Neural Networks*, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) — top-k gating, and the first description of routing collapse
