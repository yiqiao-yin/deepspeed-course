# 02_intermediate/04_groupwise_ranking — Groupwise Ranking

The previous folder changed the **loss** and held the scorer fixed. This one
does the opposite: the loss is frozen at ListNet for every model, and what
varies is how much of the candidate list the scorer may look at — a plain
pointwise MLP, **GSF** (Ai et al., 2019), and **SetRank** (arXiv:1912.05891),
which is self-attention over the candidate set. Because only the architecture
moves, the comparison is controlled.

It belongs in a DeepSpeed course for a sharper reason than the last folder. A
groupwise scorer is O(L²) in the list length and a query **cannot** be split
across devices: a group that spans two ranks scores each document against a
*subset* of its real candidates, silently, with no error and a loss that still
goes down. Data parallelism here shards queries, and that constraint is the
lesson.

Everything here runs on a CPU. Start with `groupwise.py`.

## What this demonstrates

- **Two properties that decide whether a groupwise model is right or broken**,
  measured rather than asserted, and printed next to every score:

  | property | meaning | control |
  |---|---|---|
  | context sensitivity | does a document's score change when its *neighbours* change? | pointwise must be **exactly 0** |
  | permutation equivariance | shuffle the candidates and the scores must permute identically | a position-reading model must **fail** |

  A "groupwise" model with sensitivity ≈ 0 has collapsed into a pointwise one,
  and every claim made about it is then false. A model that is *not*
  equivariant is reading candidate **order** — which at training time is
  usually the label order. That is a leak that looks like a result.

- **The leak is not hypothetical.** The first GSF written for this folder
  formed its groups by rotating the list, which depends on absolute position.
  It scored well. Its permutation error was **1.5e-01**, and the property test
  is what caught it. It now enumerates all ordered pairs and measures 9.5e-07.

- **A task a pointwise scorer provably cannot solve.** `--task duplicate`
  plants near-identical documents in each list and demotes the twin to grade 0
  — showing the same result twice is worth less than showing it once. Two
  documents with almost identical features must get almost identical scores
  from a function of *one* document, so no amount of training closes the gap.
  This matters because the softer, more realistic `--task redundancy` does
  **not** separate the models convincingly (see below), and a demonstration
  should be decisive before it is realistic.

- **More context capacity is not free.** SetRank has ~10× GSF's parameters and
  loses to it here. At smaller data scales it loses badly and gets *worse* with
  more training — measured at 250/800/2400 steps across three learning rates.

## Measured results

`--model all`, 4096 train / 1024 test queries, 12 documents per list, 30 epochs:

| task | pointwise | gsf | setrank | untrained |
|---|---|---|---|---|
| `duplicate` (context **required**) | 0.9369 | **0.9903** | 0.9834 | 0.5431 |
| `redundancy` (context merely helps) | 0.9913 | **0.9970** | 0.9964 | 0.6260 |

Groupwise scoring buys **+0.053 NDCG** where context is required and **+0.006**
where it is only useful. Both numbers are worth publishing: the second is the
honest reminder that on tasks where a document's value doesn't really depend on
its neighbours, an O(L²) scorer buys you almost nothing for a large constant
factor.

| model | parameters | context sensitivity | perm. error |
|---|---|---|---|
| pointwise | 5,313 | 0.000000 | 0.00e+00 |
| gsf | 6,402 | 0.737396 | 9.54e-07 |
| setrank | 68,097 | 0.666664 | 2.19e-05 |

## Hardware requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | 8 GB | Any CUDA card. The largest model is 68k parameters; memory is the O(list_len²) activation, so `--list-len` is the knob — not the parameter count |
| GPUs | 1 (2 to see sharding) | 2 is the smallest number that exercises query-level sharding |
| Disk | < 1 GB | Nothing is downloaded; data is generated with numpy |
| Host RAM | 8 GB | Data is generated in-process |

**No GPU?** The models are tiny and the CPU paths are first-class:

```bash
uv run groupwise.py                                # models + both property checks
uv run ../../tests/test_groupwise_ranking.py       # 27 property assertions
ALLOW_CPU=1 uv run train_groupwise_ranking.py --model all
```

Without `ALLOW_CPU=1` the script stops at a preflight and tells you the above,
instead of dying inside torch's extension loader with `CUDA_HOME environment
variable is not set`.

## Environment & Local Testing

```bash
cd 02_intermediate/04_groupwise_ranking
uv sync                        # creates .venv from the COMMITTED uv.lock
```

`uv.lock` is committed, so every reader installs the same versions. torch is
pinned to the cu128 index: PyPI's default torch is a CUDA 13 wheel that
installs cleanly on a 550/570 driver and then reports
`cuda.is_available() == False` while `nvidia-smi` happily shows the card.

Quick checks that need no GPU and no download:

```bash
uv run groupwise.py
uv run ../../tests/test_groupwise_ranking.py
uv run train_groupwise_ranking.py --list-models
```

## Running it

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh --model all       # arguments are forwarded via "$@"
squeue -u $USER
tail -f logs/groupwise_ranking_<jobid>.out
scancel <jobid>
```

Cheap dry run first — two optimizer steps, proves the pipeline assembles
without burning a queue slot:

```bash
sbatch run_deepspeed.sh --max-steps 2
```

`NUM_GPUS=1 sbatch run_deepspeed.sh` works too; `ds_config.json` omits
`train_batch_size` so DeepSpeed derives it and no edit is needed.

### RunPod (creates the pod and shuts it down)

```bash
uv run runpod/runpod_ctl.py run 02_intermediate/04_groupwise_ranking \
    --dry-run --collect --wait --terminate --yes
```

- `--dry-run` runs the `--max-steps 2` path — a few cents, not an hour.
- `--collect` copies the logs back before the pod dies.
- `--terminate` shuts the pod down in a `finally`, and there is a keyless
  in-pod watchdog as backstop. **The pod is never given `RUNPOD_API_KEY`** —
  see `SECURITY.md`.

Then confirm nothing is still billing:

```bash
uv run runpod/runpod_ctl.py pods
```

Drop `--dry-run` for the real sweep. It costs minutes, not hours.

### Direct (single pod, GPU in the shell)

```bash
deepspeed --num_gpus=2 train_groupwise_ranking.py --model all
deepspeed --num_gpus=2 train_groupwise_ranking.py --model gsf --list-len 24
deepspeed --num_gpus=1 train_groupwise_ranking.py --model all --task redundancy
```

## Expected output

From an `ALLOW_CPU=1` run of `--model all` — **measured, not illustrative**:

```
==============================================================================
  Groupwise Ranking
==============================================================================
  task          duplicate
  train / test  4096 / 1024 queries
  list length   12 documents  (16 features each)
  models        pointwise, gsf, setrank
  loss          listnet (held FIXED across models)
  world size    1
==============================================================================
  pointwise    NDCG@10 0.9369   context 0.000000   perm_err 0.00e+00
  gsf          NDCG@10 0.9903   context 0.737396   perm_err 9.54e-07
  setrank      NDCG@10 0.9834   context 0.666664   perm_err 2.19e-05
==============================================================================
  untrained baseline   NDCG@10 0.5431
  best                 gsf (0.9903)
  gain over pointwise  +0.0534
```

`context` and `perm_err` are printed beside NDCG on purpose. A groupwise NDCG
is only evidence for groupwise scoring if context sensitivity is above zero and
the permutation error is at float noise; the script prints a **warning** and
tells you to treat the score as unproven when `perm_err` exceeds 1e-4.

Multi-GPU output is not yet verified on hardware; the single-GPU and CPU paths
above are.

## Configuration notes

- `train_batch_size` is **omitted** from `ds_config.json` so any `--num_gpus`
  works. DeepSpeed derives it as `micro_batch × grad_accum × world_size`.
- `train_micro_batch_size_per_gpu: 32` counts **queries**, not documents —
  32 × 12 = 384 documents per step.
- **`--list-len` is the memory knob.** GSF enumerates every ordered pair and
  SetRank attends over the whole set, so both are O(L²). Doubling the list
  quadruples the activation; doubling the hidden size does not.
- fp16 and bf16 are both **disabled**. A 0.005 NDCG difference between two
  architectures is the quantity being measured, and running one of them in
  reduced precision would measure the dtype instead. These models are a few
  thousand parameters; fp32 costs nothing.
- ZeRO **stage 0**: sharding optimizer state for a 68k-parameter model adds
  communication to save kilobytes, and does not touch the O(L²) activation that
  actually dominates.

## Where to go next

`../03_learning_to_rank/` is the other half of this pair: it freezes the
architecture and varies the **objective** (pointwise / RankNet / LambdaRank /
ListNet) instead.
