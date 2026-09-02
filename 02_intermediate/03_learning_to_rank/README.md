# 02_intermediate/03_learning_to_rank — Learning to Rank

Ranking is not classification, and the difference lives entirely in the loss.
This example implements the four classical learning-to-rank objectives —
**pointwise**, **RankNet**, **LambdaRank**, **ListNet** — behind one shared
scoring network, so `--method all` is a controlled comparison rather than four
runs that differ in a dozen ways. It belongs in a DeepSpeed course because
ranking breaks the assumption every data-parallel example so far has relied on:
the unit of data is a **query with its whole candidate list**, so a batch is a
batch of *lists*, and sharding must split queries rather than documents.

Everything mathematical here runs on a CPU. Start with `ranking_losses.py`.

## What this demonstrates

- **Query-level sharding, and why document-level sharding is silently wrong.**
  A pairwise loss compares documents *within* one query; a listwise softmax
  normalises *over* one query. Split a candidate list across two ranks and both
  become a different computation — with no error, no warning, and a loss curve
  that still goes down. `train_learning_to_rank.py` shards `x_tr[rank::world]`
  and says so in a comment, because this is the failure mode.
- **ZeRO stage 0 as a deliberate choice.** The model is ~4k parameters.
  Sharding optimizer state across ranks would add communication to save
  kilobytes. The config says stage 0 and explains why — picking ZeRO-3 because
  it is the biggest number is cargo cult.
- **fp16 disabled on purpose.** LambdaRank weights each pair by |ΔNDCG|, which
  is order 1e-4 for deep swaps. In fp16 those weights flush toward zero and
  LambdaRank quietly degenerates into RankNet — a real algorithm, so nothing
  looks broken. This is the clearest example in the course of a precision
  choice that changes an *objective* rather than a number.
- **A measured result rather than the textbook one.** See below.

## The honest headline

The literature's ordering (listwise > pairwise > pointwise) is real, but on
this data it is **a function of training budget**, and reporting one number
without the budget would be misleading. Measured with
`train_learning_to_rank.py --method all --epochs N`:

| epochs | pointwise | ranknet | lambdarank | listnet | spread |
|---|---|---|---|---|---|
| 1 | 0.9198 | 0.9594 | 0.9401 | **0.9611** | **0.0413** |
| 2 | 0.9593 | 0.9678 | 0.9649 | 0.9661 | 0.0085 |
| 6 | 0.9637 | 0.9689 | 0.9680 | 0.9682 | 0.0052 |
| 40 | 0.9677 | 0.9686 | 0.9683 | 0.9676 | 0.0010 |

Untrained baseline: **0.4862** at every row.

Listwise wins clearly when training is short — at one epoch ListNet leads
pointwise by 0.041 — and the ordering dissolves by epoch 40, where the spread is
0.0010 and *pointwise is second*. Training at all moves NDCG from 0.49 to 0.97;
the objective moves it by 0.001 at convergence. Anyone quoting a single "listwise
beats pointwise by X" has fixed a budget without telling you. Published gains come from real corpora with ties,
position bias and long lists — which is what `--source hf` is for.

## Hardware requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | 8 GB | Any CUDA card. The model is ~4k parameters; memory is dominated by the O(list_len²) pairwise term, so `--list-len` is the knob |
| GPUs | 1 (2 to see sharding) | 2 is the smallest number that exercises query-level sharding |
| Disk | < 1 GB | `--source synthetic` downloads nothing. `--source hf` adds ~90 MB (MiniLM + SciDocs) |
| Host RAM | 8 GB | Data is generated in-process with numpy |

**No GPU?** This example is genuinely small, and the CPU paths are documented
rather than grudging:

```bash
uv run ranking_losses.py                          # objectives, metrics, a demo
uv run ../../tests/test_ranking_losses.py         # 33 property assertions
ALLOW_CPU=1 uv run train_learning_to_rank.py --method all
```

Without `ALLOW_CPU=1` the script stops at a preflight and tells you the above,
instead of dying inside torch's extension loader with `CUDA_HOME environment
variable is not set`.

## Environment & Local Testing

```bash
cd 02_intermediate/03_learning_to_rank
uv sync                        # creates .venv from the COMMITTED uv.lock
uv sync --extra real-data      # only if you want --source hf
```

`uv.lock` is committed, so every reader installs the same versions. torch is
pinned to the cu128 index: PyPI's default torch is a CUDA 13 wheel that
installs cleanly on a 550/570 driver and then reports
`cuda.is_available() == False` while `nvidia-smi` happily shows the card.

Quick checks that need no GPU and no download:

```bash
uv run ranking_losses.py
uv run ../../tests/test_ranking_losses.py
uv run train_learning_to_rank.py --list-methods
```

## Running it

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh --method all      # arguments are forwarded via "$@"
squeue -u $USER
tail -f logs/learning_to_rank_<jobid>.out
scancel <jobid>
```

Cheap dry run first — two optimizer steps, proves the pipeline assembles
without waiting for a queue slot to be wasted:

```bash
sbatch run_deepspeed.sh --max-steps 2
```

`NUM_GPUS=1 sbatch run_deepspeed.sh` works too; `ds_config.json` omits
`train_batch_size` so DeepSpeed derives it and no edit is needed.

### RunPod (creates the pod and shuts it down)

```bash
uv run runpod/runpod_ctl.py run 02_intermediate/03_learning_to_rank \
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
deepspeed --num_gpus=2 train_learning_to_rank.py --method all
deepspeed --num_gpus=2 train_learning_to_rank.py --method lambdarank --epochs 40
deepspeed --num_gpus=1 train_learning_to_rank.py --source hf   # real corpus
```

## Expected output

From an `ALLOW_CPU=1` run with `--method all --epochs 40` — **measured, not
illustrative**. Your numbers will differ slightly with a different seed:

```
==============================================================================
  Learning to Rank
==============================================================================
  source        synthetic
  train / test  4096 / 1024 queries
  list length   16 documents  (32 features each)
  methods       pointwise, ranknet, lambdarank, listnet
  world size    1
==============================================================================
  pointwise    NDCG@10 0.9677   MRR 1.0000   MAP 0.9924
  ranknet      NDCG@10 0.9686   MRR 1.0000   MAP 0.9924
  lambdarank   NDCG@10 0.9683   MRR 1.0000   MAP 0.9924
  listnet      NDCG@10 0.9676   MRR 1.0000   MAP 0.9925
==============================================================================
  untrained baseline   NDCG@10 0.4862
  best                 ranknet (0.9686)
  spread across methods 0.0010
==============================================================================
```

MRR is 1.0000 for all four, and that is a fact about the DATA, not a bug: the
generator grades documents 0-4 in roughly equal proportions, so about 80% of a
list is "relevant" under MRR's binary reading and any trained model puts
something relevant first. MRR is simply uninformative on graded data with this
many relevant documents — which is worth seeing once, since it is a metric
people report by reflex. NDCG is the one to read here.

The untrained baseline is printed on purpose: random ordering of a 16-document
list already scores ~0.62 NDCG, so a bare "0.97" means nothing without it.

Multi-GPU output is not yet verified on hardware; the single-GPU and CPU paths
above are.

## Configuration notes

- `train_batch_size` is **omitted** from `ds_config.json` so any `--num_gpus`
  works. DeepSpeed derives it as
  `micro_batch × grad_accum × world_size`; hardcoding it and then changing
  `--num_gpus` is the most common breakage in this course.
- `train_micro_batch_size_per_gpu: 64` counts **queries**, not documents —
  64 × 16 = 1,024 documents per step.
- `--list-len` is the memory knob, not the parameter count: the pairwise losses
  are O(L²).
- `--noise 0` makes every method reach NDCG 1.0 and the comparison says
  nothing. The default of 2.0 is what keeps the task hard enough to separate
  them.

## Where to go next

`../04_groupwise_ranking/` holds the scoring function fixed the *other* way: it
freezes the loss at ListNet and changes the architecture, so documents are
scored **in context** instead of one at a time.
