# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Rules here are terse on purpose. Where a rule exists because something shipped
broken, the hook names the incident and links to the full account in
[`POSTMORTEMS.md`](POSTMORTEMS.md). **Read the linked section before undoing a
rule that looks arbitrary** — most of them are load-bearing and none are
stylistic.

Each account carries an **incident record** citing the commits that introduced
and fixed it, so every claim in that file can be checked against the code
rather than taken on trust — see
[the incident record convention](#every-postmortem-carries-an-incident-record).

## What this repository is

A teaching course, not an application. Each numbered directory
(`01_basics/01_neuralnet` … `06_protein_folding/04_structure_module`) is a
**self-contained, runnable DeepSpeed example** that escalates in difficulty:
toy MLP → CNN → LSTM → Bayesian MCMC → HuggingFace/TRL fine-tuning → GRPO RL →
LoRA SFT of 20B models → video-text → video-speech-to-speech → protein
structure.

There is no package and no shared library. Directories deliberately duplicate code rather than import from each other — a reader should be able to open one folder and run it without touching the rest. **Do not refactor shared logic into a common module.** (`require_gpu()` appears verbatim in ~34 files on purpose.)

There *is* a regression suite in `tests/` (CPU-only, runs in CI) and a GPU tier in `tests/gpu/`; see [What can and cannot be run here](#what-can-and-cannot-be-run-here). Tooling lives in `runpod/` for provisioning GPUs on demand, and `scripts/` for scaffolding and drift auditing.

Contributions from outside are welcome and governed by `CONTRIBUTING.md`, which is written to double as a spec an agent can follow. Read it before adding an example — it encodes the three-platform contract below. Repo is MIT (`LICENSE`).

`MOVED.md` maps the pre-reorganisation paths: five top-level numbers used to be
reused, and every folder now sits at `NN_section/NN_topic`. Folders were moved
with `git mv`, so `git log --follow` and `git blame` work through the rename.

## Architecture: which folders only make sense together

### The alignment thread spans four topics in `03_llms/`

`04`–`07` are not independent examples; they are one escalating argument about
**what you can delete from the RLHF pipeline**, and the deletions are different:

| Folder | Deletes | Reference model? |
|---|---|---|
| `03_llms/04_reward_model` | — (this IS the pipeline) | — |
| `03_llms/05_dpo` | the **reward model** (`--method` covers 6 objectives) | LoRA removes it |
| `03_llms/06_grpo` | the **critic** | yes |
| `03_llms/07_online_dpo` | — (re-adds sampling; needs a judge) | yes |

> "DPO removes the reward model" and "GRPO removes the critic" are two different
> claims about two different components. Conflating them is the most common
> confusion in this area, and the docs say so in three places on purpose.

The book pages run `rlhf-reward-modeling` → `preference-optimization` → `grpo-*`
→ `online-preference-methods` → `beyond-grpo`, ordered by **when the literature
arrived relative to GRPO (Feb 5, 2024)**. That ordering is deliberate and the
pages carry dated tables because the families genuinely straddle it — KTO
precedes GRPO by three days, ORPO and SimPO follow.

### `02_intermediate/03` and `04` are a matched pair

They vary opposite halves of the same system, and only make sense read together:

| Folder | Held fixed | Varied |
|---|---|---|
| `03_learning_to_rank` | the scorer | the **objective** — pointwise / RankNet / LambdaRank / ListNet |
| `04_groupwise_ranking` | the objective (ListNet) | the **architecture** — pointwise / GSF / SetRank |

Two findings there are load-bearing and easy to undo by "tidying":

- The published spread between objectives **depends on training budget** (0.041
  at 1 epoch, 0.001 at 40). The docs give the budget with every number on
  purpose; a single "listwise beats pointwise by X" would be meaningless.
- `04`'s two property checks — **context sensitivity** and **permutation
  equivariance** — are not decoration. The first GSF written here scored well
  and had a permutation error of 1.5e-01, i.e. it was reading candidate order,
  which at training time is label order. Only the property test caught it.

### `03_llms/01_llm_finetuning` holds three entry points

Not one. `train_ds.py` (Llama SFT, the original), `train_glm53_ds.py` (GLM-5.3,
a 755 GB sparse MoE) and `train_qwen38_ds.py` (Qwen3.8-27B, hybrid
linear/full attention). The two frontier scripts share a shape worth reusing:

```bash
uv run train_glm53_ds.py --plan          # architecture + capacity, from config.json
uv run train_qwen38_ds.py --verify-arch  # build the real module tree, no weights
```

**`--verify-arch` is the technique to copy.** It builds the model on torch's
**meta device** — no memory, no weight download, about two seconds for 743 B
parameters — and checks that the LoRA target names resolve against the real
module tree. That catches, for free, the three things that otherwise fail only
*after* a multi-hundred-gigabyte download: an unsupported architecture, target
names that match nothing, and parameter arithmetic that disagrees with the
implementation.

It also surfaced a fact no amount of reading the checkpoint would: transformers
**fuses** GLM-5.3's 256 experts into 3D tensors at runtime
(`mlp.experts.gate_up_proj` is `(256, 4096, 6144)`) although the checkpoint
stores them per expert. **Checkpoint layout and runtime module tree are not the
same thing**, and peft can only wrap `Linear`/`Embedding`/`Conv1D`, so freezing
the experts there is the only expressible option rather than merely the wise one.

The folder also holds `analyze_kimi_k3.py` — see
[not every entry point is a training script](#not-every-entry-point-is-a-training-script).

### `03_llms/10` and `11` are the two halves of one paper

DeepSeek-V2 and V3 each make **two** architectural contributions, and the course
covers them in adjacent folders that only make sense read together:

| Folder | Shrinks | Mechanism |
|---|---|---|
| `10_deepseek_from_scratch` | what the model **remembers** | MLA: cache a latent, reconstruct K/V |
| `11_moe` | what the model **computes** | MoE: hold N experts, fire k |

Both ship a CPU-runnable pure-algorithm module (`mla.py`, `moe.py`) beside a
DeepSpeed training script, which is the shape to copy for anything whose
substance is an algorithm rather than weights.

The distinction that took a reader asking to surface: **MLA is not the router.**
The MLA page cites the DeepSeek-V2 paper — whose title contains
"Mixture-of-Experts" — and has no routing content at all. Conflating them is
the same class of error as conflating "DPO removes the reward model" with
"GRPO removes the critic".

**`11_moe` is also where expert parallelism lives**, and EP is a *different
axis* from ZeRO: ZeRO shards optimizer state, gradients and parameters of the
same model while every rank runs every layer on different data; EP shards the
**experts**, so each rank holds a different subset of the model and tokens
reach it by all-to-all. DeepSpeed ships `deepspeed/moe/` (`ep_router.py`,
`sharded_moe.py`) for this. It is the reason the topic belongs in this course
rather than an architecture course.

### `06_protein_folding` is one argument against a reflex

The four subtopics escalate, but the section exists to make a single systems
claim that the rest of the course sets up and then breaks:

| Folder | What it is | Reads the thesis |
|---|---|---|
| `01_esm2_plm` | ESM-2, sequence only — no MSA, no geometry | the control: an ordinary transformer |
| `02_evoformer` | the AlphaFold2 trunk | **where the thesis is measured** |
| `03_pairformer` | the AlphaFold3 trunk — MSA deleted from the trunk | what deleting a representation buys |
| `04_structure_module` | IPA + FAPE → coordinates | SE(3) invariance by construction |

**The thesis: AlphaFold-class memory lives in activations, not parameters, so
no ZeRO stage helps.** The trunk this course builds has ~100k parameters; what
fills the card is the pair representation (`O(N²)`) and the triangle attention
logits (`O(N³)`). Measured in `02_evoformer`: **ZeRO-1 0.65 GB vs ZeRO-3
0.62 GB — 4.6%**, for 1.5× the communication. That is the point of the lab,
and it is the reason DeepSpeed shipped a fused *kernel*
(`DS4Sci_EvoformerAttention`) for this model family rather than another
sharding strategy.

Three things that are easy to break by tidying:

- **`ds_config.json` and `ds_config_z3.json` must differ in the stage and
  nothing else.** They once differed in `reduce_bucket_size` too — ZeRO-1 left
  it unset, so DeepSpeed's 5e8 default made the comparison read 1.58 vs
  0.62 GB and appear to prove ZeRO-3 saves 2.5×, the exact opposite of the
  lab's finding. A config comparison is only a comparison if one variable moves.
- **`DS4Sci_EvoformerAttention` is UNVERIFIED here** and labelled so in four
  places. It needs a CUDA toolkit (`nvcc`); the PyPI `nvidia-cuda-nvcc-cu12`
  wheels ship only `ptxas`. No memory number is published for it — do not
  estimate one.
- **`02` and `03` are a matched pair**, like `02_intermediate/03`+`04`. The
  AF2→AF3 saving from deleting the MSA representation **decays with length**:
  measured at 58.5% (32 residues), 24.2% (128), 12.5% (256). It is a constant
  subtracted from an $O(N_{res}^3)$ term, so quoting one figure without the
  length is meaningless — and the length that figure came from is the thing
  most easily lost in an edit.

`synthetic_msa.py` generates coevolving alignments scored by **APC-corrected**
mutual information (Dunn 2008). Raw MI is biased by column entropy, and the
first counterexample written here was contaminated — one shared mutation event
per coupled pair correlated the columns even at `coupling=0.0`.

### Sections 04, 05 and 06 are multi-subtopic

Most sections hold flat topics. **`04_video_text/`, `05_video_speech/` and
`06_protein_folding/` escalate internally**, so each holds several numbered
subtopics:

```
04_video_text/{01_hf_baseline, 02_qwen25vl, 03_token_compression,
               04_streaming_memory, 05_video_eval, 06_qwen3vl}
05_video_speech/{01_longcat_omni, 02_thinker_talker,
                 03_duplex_streaming, 04_omni_eval, data/}
06_protein_folding/{01_esm2_plm, 02_evoformer,
                    03_pairformer, 04_structure_module}
```

Each subtopic keeps the full six-file contract independently and is registered
separately in `runpod/runpod_ctl.py` under a **nested key** (`"05_video_speech/02_thinker_talker"`),
so a reader rents a 24 GB card for the tractable subtopic instead of the
frontier model's unobtainable hardware. `tests/test_runpod_ctl.py` only requires
*top-level* numbered dirs in that table; nested entries are additive.

`05_video_speech/data/` (44 MB of real video+audio) is **shared across its subtopics**
rather than duplicated four times into git history — override with
`VSS_DATA_DIR`. Sharing sample *media* this way does not violate the
no-shared-module rule, which is about logic.

The through-line worth knowing when editing either topic: **every frontier
technique in both is a memory technique.** ZeRO shards what the model *is*;
token compression shrinks what it *looks at*; STAR memory bounds what it
*retains*. Same bargain, different currency.

## The per-example contract

Every example folder follows the same six-file shape. When adding or editing an example, keep it:

| File | Role |
|---|---|
| `train_*.py` | Training entry point; calls `deepspeed.initialize(...)` and reads the JSON config |
| `ds_config*.json` | DeepSpeed config — ZeRO stage, fp16/bf16, optimizer, batch sizes |
| `run_deepspeed.sh` (or `submit_job.sh`, `run_training.sh`, `run_2xB200.sh`) | SLURM batch script, or a bare launcher for single-pod platforms |
| `README.md` | Full standalone walkthrough: hardware, setup, run command, expected output |
| `pyproject.toml` | Makes the folder a **uv project**: dependencies, `requires-python`, `package = false`, W&B under an optional `tracking` extra |
| `uv.lock` | **Committed.** `cd <example> && uv sync` must work from a fresh clone — enforced by `tests/test_runpod_ctl.py` |

Larger examples add `HARDWARE_REQUIREMENTS.md` / `HARDWARE_GUIDE.md` / `MODEL_IMPROVEMENT_STRATEGY.md`.

Batch size consistency is enforced by DeepSpeed at startup: `train_batch_size == train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`. Changing `--num_gpus` in a launcher without updating the JSON is the most common breakage.

## Running examples

```bash
cd 01_basics/01_neuralnet
deepspeed --num_gpus=1 train_ds_enhanced.py          # direct, e.g. RunPod / single pod
sbatch run_deepspeed.sh                              # SLURM, e.g. CoreWeave
```

SLURM workflow: `sbatch <script>` → `squeue -u $USER` → `tail -f logs/<name>_<jobid>.out` → `scancel <jobid>`. Every batch script does `mkdir -p logs` and writes `logs/*_%j.{out,err}`.

## Tooling: always `uv`

Environments and package installs use **`uv`**, never bare `pip` or conda — including
for throwaway checks. **Every example folder is a uv project** with a committed
`uv.lock`, so the first thing a reader does is:

```bash
cd 01_basics/02_convnet && uv sync && uv run deepspeed --num_gpus=1 train_ds.py
```

Locks are per folder rather than a workspace, matching the no-shared-module
rule: one folder must run without the other 22 existing. Regenerate with
`uv lock --upgrade`, deliberately. Ad-hoc commands still use uv directly:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed wandb

uv run script.py                    # run a script
uv run --no-project python -c "..." # one-off, no project env
```

Every example README documents its own `uv` setup under **Environment & Local Testing**.

**A custom torch index pins its companions too.** Any package with a compiled
extension linked against torch — `torchvision`, `torchaudio` — must appear in
`[tool.uv.sources]` whenever torch does; with `explicit = true` only the packages
named there come from that index, and a mismatched pair fails with
`RuntimeError: operator torchvision::nms does not exist`, which reads like a
torch bug and is not one. `tests/test_torch_index_pins.py` guards it by reading
the **lock**, not the pyproject.
→ [postmortem](POSTMORTEMS.md#a-custom-torch-index-pins-its-companions-too-or-nothing-works)

### Which examples skip the `deepspeed` launcher

Five, and each for a stated reason. Do not "fix" these by adding DeepSpeed —
using a distributed launcher where there is nothing to distribute is cargo cult.
They carry `launcher="python"` in `runpod/runpod_ctl.py`:

| Example | Why |
|---|---|
| `03_llms/09_multi_agency` | drives TRL's `GRPOTrainer` directly |
| `04_video_text/04_streaming_memory` | streaming *inference* — sequential, no optimizer |
| `04_video_text/05_video_eval` | evaluation — short `generate()` calls |
| `05_video_speech/03_duplex_streaming` | duplex inference — slices arrive in order |
| `05_video_speech/04_omni_eval` | evaluation — modality-ablation `generate()` calls |

Every other example uses both `uv` and `deepspeed`. If you add a sixth
exception, say so explicitly and explain why.

## What can and cannot be run here

This distinction governs how to verify a change:

| Section | Scale | Verification |
|---|---|---|
| `01_basics/`, `02_intermediate/` | Synthetic or tiny data, ≤1M params, 1–2 GPUs | **Runnable end to end** on a single machine |
| `03_llms/`, `04_video_text/`, `05_video_speech/` | Real model downloads (GBs to 1.1 TB), multi-GPU, up to 560B params | **Not runnable locally.** Verify logic only |
| `06_protein_folding/` | ~100k params, synthetic or CATH data, **1 × 24 GB** | **Runnable** — the trunks train on one modest card |

`06_protein_folding/` is the third case and the exception to the usual
reading of this table: it sits high in the numbering but needs no frontier
hardware, because its models are small and its cost is in activations. Verify
changes there by **running them**, not by writing a mock.

(The numbers are per section and get reused, so name the section — `03_llms/03_ocr`,
not `03`.)

For the second group, do not attempt a full training run to validate a change —
it will not fit, and a partial run proves nothing. Write or extend a **logic test**
in `tests/` instead, which exercises the changed code path without a GPU or a
model download:

### The big exception: fifteen modules ARE fully CPU-runnable

Their substance is *algorithms, objectives and policy* rather than weights, so
they need no GPU and no download. **Run these directly rather than mocking
them:**

| Module | What it is |
|---|---|
| `02_intermediate/03_learning_to_rank/ranking_losses.py` | pointwise / RankNet / LambdaRank / ListNet, plus NDCG, MRR, MAP |
| `02_intermediate/04_groupwise_ranking/groupwise.py` | GSF / SetRank, and the two property checks that police them |
| `03_llms/05_dpo/preference_losses.py` | DPO / IPO / CPO / KTO / ORPO / SimPO, plain tensors |
| `03_llms/04_reward_model/reward_modeling.py` | Bradley-Terry objective |
| `04_video_text/03_token_compression/token_compression.py` | ToMe / FastV / DyCoke |
| `04_video_text/04_streaming_memory/star_memory.py` | STAR bounded memory bank |
| `04_video_text/05_video_eval/video_mme_eval.py` | eval harness |
| `05_video_speech/02_thinker_talker/tmrope.py` | the 40 ms shared clock — pure integer arithmetic |
| `05_video_speech/03_duplex_streaming/duplex.py` | turn-taking policy, barge-in, RTF |
| `05_video_speech/04_omni_eval/omni_eval.py` | modality-ablation grid |
| `03_llms/10_deepseek_from_scratch/mla.py` | MHA / GQA / MLA behind one interface, and the cache arithmetic |
| `03_llms/11_moe/moe.py` | top-k routing, three load-balancing strategies, the params-vs-active table |
| `04_video_text/06_qwen3vl/verify_arch.py` | builds Qwen3-VL-8B on the **meta device** — DeepStack, and whether LoRA targets resolve |
| `06_protein_folding/02_evoformer/evoformer.py` | triangle attention and multiplicative update, and the cubic memory table |
| `06_protein_folding/02_evoformer/synthetic_msa.py` | coevolving MSAs, measured by APC-corrected mutual information |

**Assert mathematical properties, not shapes.** Every bug this repo has shipped
in these areas ran fine and was quietly wrong, and a shape assertion would have
passed on all of them. Established patterns to copy:

- `test_token_compression.py` — the log-size attention identity holds to 1e-6,
  **and** the uncorrected version is measurably wrong (otherwise the test proves
  nothing).
- `test_tmrope.py` — video and audio at the same instant share a position ID at
  *any* frame rate; and naive frame-index numbering is asserted to be correct at
  exactly 25 fps (40 ms/frame == the tick) and to drift ~1,380 positions at 2 fps.
  Testing one frame rate would have shipped the bug.
- `test_star_memory.py` — bounded **and** remembers. Either alone is trivially
  satisfiable by a broken implementation.
- `test_omni_eval.py` — builds a modality-ignoring model on purpose and asserts
  the harness catches it, *and* that a healthy model is not falsely flagged.
- `test_preference_losses.py` — reference-free losses are **bit-identical** when
  the reference model moves; reference-based ones must move. Also asserts IPO
  has a finite optimum while DPO improves without bound, which is the entire
  point of IPO.
- `test_reward_model.py` — shift-invariance asserted **with a tolerance**, plus
  a check that a huge shift *does* perturb the loss. The objective is exactly
  shift-invariant in real arithmetic and only approximately so in float32
  (catastrophic cancellation), so exact equality is the wrong test.

### Running the tests

```bash
./tests/run_all.sh                 # all 39 suites, no GPU, no downloads
uv run tests/test_ds_configs.py    # one suite

# what CI actually runs — run_all.sh alone does not reproduce it
uv run --no-project python -m compileall -q <training scripts>   # see .github/workflows/tests.yml
```

Tests use [PEP 723](https://peps.python.org/pep-0723/) inline dependency metadata,
so `uv run` provisions each one automatically. `tests/_srcload.py` extracts a single
function from a training script via `ast` so tests run against the **actual shipped
source** without importing torch/deepspeed/trl. See `tests/README.md`.

When you change an example, update its README's **Environment & Local Testing**
section if the dependencies, GPU count, or download size changed.

After editing an example, run the drift audit and read its findings:

```bash
uv run scripts/audit_readmes.py
```

It is **advisory, not a gate** — it over-reports because teaching READMEs contain
illustrative code and remediation advice that legitimately differ from the
shipped source. Triage by hand; see `scripts/README.md`.

### The GPU tier: `tests/gpu/`

Not run by CI and not run by `run_all.sh`. These need real hardware and answer
questions a CPU cannot:

| Script | Question |
|---|---|
| `probe_device_binding.py` | does each rank bind to its OWN GPU? (10 s, no data, no model) |
| `diagnose_nccl.sh` | is multi-GPU NCCL working on this box at all? |
| `verify_uv_sync_cuda.sh` | does the committed lock actually install and import on a CUDA box? |
| `verify_02_modern_cifar.sh` | `01_basics/03_convnet_cifar10`'s modern script, end to end |
| `verify_04_multi_gpu.sh` | the multi-GPU path — rank guards, barriers, device binding |
| `verify_05_ocr_models.sh` | `03_llms/03_ocr` loads and steps on the declared card |
| `verify_05_reward_model.sh` | `03_llms/04_reward_model` end to end |
| `verify_06_online_dpo.sh` | `03_llms/07_online_dpo` end to end |
| `validate_llava_vision_path.py` | the vision tower actually receives pixels |

Run `probe_device_binding.py` first when a multi-GPU job hangs. Correct devices
plus a hang means the binding is fine and the interconnect is not, which is
`diagnose_nccl.sh`'s territory. `verify_uv_sync_cuda.sh` is the executable form
of the rule below that a harness which does not install the artifact verifies
nothing.

### Multi-rank logic can be verified without multiple GPUs

Most of what breaks in a distributed guard is not CUDA. It is control flow —
who does the work, who waits, and whether the collective is even a legal call.
All of that runs on **gloo, on CPU, in two processes**, and it runs against the
*shipped* source rather than a paraphrase:

```python
# two processes, real process group, torchvision stubbed so nothing downloads
dist.init_process_group(backend="gloo", rank=rank, world_size=2)
fn = load_function(REPO / "01_basics/03_convnet_cifar10/cifar10_deepspeed.py",
                   "download_cifar10",
                   extra_globals={"torchvision": _stub, ...})   # tests/_srcload.py
```

That run proved: exactly one rank downloaded, rank 1 blocked for the entire
4.0 s rank 0 spent, neither raised. The last of those is the one that matters
most — it is a live check that `barrier(device_ids=...)` is a **legal call at
the locked torch**, which is precisely what a `TypeError` had broken.

What gloo does **not** cover is the NCCL device binding itself: two ranks
colliding on cuda:0 needs two real CUDA devices. So this technique retires the
control-flow risk and leaves exactly one question for real hardware. Say which
is which when reporting — "verified" and "believed correct" are different
claims, and a reader betting a GPU-hour deserves to know which they are getting.

**Check what hardware you actually have before declaring a thing unrunnable.**
`nvidia-smi` costs a second; three rounds of a fix went out "verified" by static
analysis on a box that had a GPU all along.

## Distributed rules

These are the ones that cost the most time to rediscover.

- **Only rank 0 downloads, and the others must wait on a barrier.**
  `torchvision.datasets.*(download=True)` does **no locking**; `huggingface_hub`
  does. Guard the download on rank, then `barrier()` — without the barrier rank 1
  reads a directory rank 0 is still writing, which fails only *sometimes*.
  `tests/test_multigpu_download_guard.py` enforces it (AST-based, because several
  scripts rank-guard their *printing* while downloading unguarded).
  → [postmortem](POSTMORTEMS.md#only-rank-0-downloads-and-the-others-must-wait-on-a-barrier)
- **Bind the device before any collective that runs before `deepspeed.initialize()`.**
  Call `torch.cuda.set_device(local_rank)` and pass `device_ids=` to the barrier,
  or every rank all-reduces on cuda:0 and hangs. Pass **no `timeout=`**: torch
  2.13 accepts one, torch 2.11 — what every lab locks — raises `TypeError`.
  **Verify an API against the version the lab's `uv.lock` resolves**, not against
  whatever is on the box.
- **Derive `local_rank` from `LOCAL_RANK`, not from argv.** `torchrun` sets only
  the environment variable, so an `args.local_rank` defaulting to `-1` binds
  every rank to cuda:0.
- **A rank guard may wrap printing, logging and saving; it must never wrap a
  collective.** Run the collective on every rank and guard only what it prints.
  Tear down deliberately — `barrier()` then `destroy_process_group()` on every
  rank. Related: an unused expert produces no gradient, so
  `if p.grad is not None` makes ranks disagree on how many all-reduces to issue.
  → [postmortem](POSTMORTEMS.md#guard-the-output-never-the-collective)
- **Size multi-GPU jobs per GPU, not in aggregate.** Weights shard under ZeRO-3;
  activations, gather buffers and fragmentation do not.
  → [postmortem](POSTMORTEMS.md#sizing-multi-gpu-jobs-model-it-per-gpu-not-in-aggregate)

Two signatures worth recognising on sight:

| Signature | Means |
|---|---|
| OOM whose *requested* allocation is trivially small (60 MiB) on a card with GB spare | sharding never happened — under ZeRO-3 the DeepSpeed config must exist *before* `from_pretrained`, or `zero.Init` never fires. Build `SFTConfig`/`TrainingArguments` first. |
| a collective with `NumelIn=1` (or 1152) timing out | not a memory event. The box advertises peer-to-peer it cannot perform; `nvidia-smi topo -m` showing `SYS` is the tell, `NCCL_P2P_DISABLE=1` the fix, at a real throughput cost. |

**"Rent a bigger box" was the wrong answer three times running** — twice the
symptom looked like size and was not, and a larger machine would have masked it.
→ [postmortem](POSTMORTEMS.md#rent-a-bigger-box-was-the-wrong-answer-three-times-running)

## Two target platforms

The README's central distinction, which shapes every launcher script:

- **CoreWeave** — shared SLURM HPC cluster. You SSH to a login node and *submit*; you never run training interactively. Scripts carry `#SBATCH` headers (`--gres=gpu:N`, `--partition=h200-low`, `--time`, `--mem`).
- **RunPod** — single-user pod with direct GPU access. Run `deepspeed ...` straight in the shell; the `#SBATCH` lines are inert comments.

`05_video_speech/01_longcat_omni/run_2xB200.sh` shows the non-SLURM style: preflight checks on GPU count, free disk, and RAM before launching.

### The three-platform contract

**Check it, do not reason about it:**

```bash
uv run scripts/check_contract.py 10_my_topic   # one example, 33 checks
uv run scripts/check_contract.py               # the whole repo
uv run scripts/check_contract.py -v <folder>   # show passing checks too
```

`CONTRIBUTING.md` states the contract in full; the short version, because every
new or edited example must satisfy it:

| Reader | Requirement |
|---|---|
| **no GPU** | `require_gpu()` called *before* torch/deepspeed are imported; message says why it stopped, what they can still do, and how to rent a GPU; `ALLOW_CPU=1` honoured |
| **CoreWeave** | a `run_deepspeed.sh` with `#SBATCH` headers and a cheap `--max-steps` dry-run path |
| **RunPod** | an `EXAMPLES` entry in `runpod/runpod_ctl.py`, and the README documents `run <ex> --dry-run --collect --wait --terminate --yes` plus confirming with `pods` |

Put heavy imports *inside* `main()`, after the preflight. Import torch at module
scope and a CPU-only reader gets a CUDA traceback before the message ever runs.

`check_contract.py` is advisory — older examples predate parts of the contract
and failing the build on those would water the checks down until they catch
nothing. The non-negotiable subset (`EXAMPLES` registration, `bash -n`, `#SBATCH`
presence) is in `tests/test_runpod_ctl.py` and does fail CI.

> When it flags something, check whether the CHECKER is wrong before changing
> working code — three of its original checks were over-strict and were fixed in
> the checker.
> → [postmortem](POSTMORTEMS.md#watch-a-checker-fail-before-trusting-it)

### The contract can be RUN, not only checked

`check_contract.py` reads the source. Two of its three readers can be
*executed*, cheaply, and executing them catches things reading does not.

**Reader A costs nothing.** Hide the GPU and run the script:

```bash
CUDA_VISIBLE_DEVICES="" uv run python train_x.py ; echo "exit=$?"
```

That proves the message actually prints, that it names something the reader
*can* run, and — the part a static check is weakest on — that the **exit code
is non-zero**. A guard that prints a beautiful message and exits 0 passes every
grep and tells a calling script the run succeeded. Also check `ALLOW_CPU=1`
still gets past it, since that escape hatch is part of the contract.

**Reader C must run on the DECLARED hardware.** This is the one that matters
and it is easy to get wrong for the most understandable reason: when the cheap
24 GB card has no capacity, a 48 GB card is right there and the run will
certainly pass on it. It proves nothing. `min_vram_gb: 24` is a claim *about
24 GB*, and verifying it on 48 GB is how `03_llms/03_ocr` shipped a lab that
OOMed for every learner who believed the manifest.

    04_video_text/02_qwen25vl verified on 2 x RTX 3090 -- the declared 24 GB,
    not the A40 that was available -- 3/3 steps, adapter saved, rc=0, zero OOM.

A few steps is enough. `--max-steps 3` proves `uv sync` resolved, the model
loaded, and nothing OOMed, without paying for a full fine-tune. What you are
testing is the *contract*, not the model. One thing worth watching in that log:
seeing `.../<lab>/.venv/bin/python` is proof the harness installed from the
lab's **committed lock** rather than from whatever the container image shipped —
the distinction that hid eight broken labs.
→ [postmortem](POSTMORTEMS.md#a-verification-harness-that-does-not-install-the-artifact-verifies-nothing)

### The RunPod harness

**Never give the pod `RUNPOD_API_KEY`** — termination is driven from the local
machine in a `finally`, with a keyless in-pod watchdog as backstop. See
`SECURITY.md`.

- **`--wait-seconds` defaults to 1800.** Too short for anything with a large
  download, and with `--terminate` the pod is destroyed mid-download.
- **GitHub rate-limits anonymous clones from cloud IP ranges**, so a pod can
  fail with `could not read Username for 'https://github.com'` on a public repo.
  There is a codeload tarball fallback; no credential is ever placed on the pod.
- Four bugs that each made a **failed** run look successful are now pinned by
  assertions in `tests/test_runpod_ctl.py`.
  → [postmortem](POSTMORTEMS.md#the-runpod-harness-lies-less-than-it-used-to)

## Correctness rules with a history

Each of these exists because something shipped broken. The hook is here; the
evidence is in `POSTMORTEMS.md`.

- **Synthetic data must carry a signal, and the summary must not lie.** Labels
  independent of inputs make chance the information-theoretic ceiling, and the
  script will still print "Finished Successfully" while advising the reader to
  train longer. Calibrate noise **against the model and optimizer the lab
  actually ships**, and publish a range rather than a number.
  → [postmortem](POSTMORTEMS.md#synthetic-data-must-carry-a-signal-and-the-summary-must-not-lie)
- **A short run must not be reported as a failure.** Clawdeck runs these with
  `--epochs 1` / `--max-steps 20`. Say the run was capped and what a real one
  looks like. This has shipped twice, months apart.
- **Any hyperparameter expressed in epochs is a bug waiting for a short run.**
  A `warmup_epochs` hardcoded to 5 spent a `--epochs 1` run's only epoch at a
  fifth of the target rate: 24.32% → 61.21% once fixed.
- **A slow data source can make a lab unrunnable, and it looks like success.**
  CIFAR-10 from `cs.toronto.edu` is ~400× slower than the HF mirror and outlived
  the orchestrator window, so the job reported *finished* without one training
  step. Capping steps does not cap the download. A mirror is a trust decision —
  `tests/test_cifar10_source.py` asserts counts, shapes and canonical **label
  order**.
  → [postmortem](POSTMORTEMS.md#a-slow-data-source-can-make-a-lab-unrunnable-and-it-looks-like-success)
- **A measured claim is scoped to the configuration it was measured in.**
  `11_moe`'s load-balancing finding reversed at world size 2; it was scoped to
  world size 1 and the disagreement written down as unresolved rather than
  rewritten around one unreplicated run. **It has now shipped twice**: the
  second time the figure was correct and the residue count attached to it was
  not, which doubled the published saving. A number and the configuration it
  was measured in are one indivisible fact; quoting the number alone is a
  different claim, not a shorter one. `tests/test_published_protein_numbers.py`
  recomputes every analytic figure from the shipped function and pins the
  measured ones to a single owning table.
  → [postmortem](POSTMORTEMS.md#a-measured-claim-is-scoped-to-the-configuration-it-was-measured-in)
- **Never fabricate expected output.** If it has not been run, mark it *not yet
  verified on hardware*.
- **Library API drift comes in three classes** — a rejected kwarg (fails at
  construction), a removed attribute (fails at the save step, after training),
  a vanished symbol (fails at import, before anything runs). All three are
  syntactically valid, so `compileall` catches none. `tests/test_config_kwargs.py`
  covers them and checks its own pinned versions against every lab's `uv.lock`.
  When bumping a library, expect it to fail and read it as a to-do list.
  → [postmortem](POSTMORTEMS.md#library-api-drift-comes-in-three-classes-and-only-one-is-obvious)
- **A lab is its COMMAND, not just its code.** `03_llms/03_ocr` OOMed because
  the manifest omitted `--use-lora`, not because the hardware claim was wrong.
  `parse_known_args()` — used by contract, since the launcher injects
  `--local_rank` — makes a misplaced flag silent. When editing a manifest, **edit
  by index, not by string match**: two labs both have a `train_ds.py` and their
  commands were byte-identical.
  → [postmortem](POSTMORTEMS.md#a-lab-is-its-command-not-just-its-code)
- **A passing static suite is evidence about the questions it asks, never
  coverage of the ones it does not.** That OCR bug was green on three checkers at
  once. → [postmortem](POSTMORTEMS.md#three-green-checkers-one-broken-lab)
- **Watch a checker fail before trusting it.** A check you have not seen reject
  bad input is not a check — four here shipped unable to fail. Run a new check
  against the **unfixed** tree first, and keep the counterexample in the suite
  permanently. **A substring is not a fact about the program**: ask the AST
  whether the thing is called, reachable, and in the branch you think it is.
  → [postmortem](POSTMORTEMS.md#watch-a-checker-fail-before-trusting-it)

### Every postmortem carries an incident record

A war story with no commit references cannot be verified and cannot be
counted. `POSTMORTEMS.md` went thirteen months citing none, which made it an
anthology rather than a record. **A new postmortem must arrive with its
record**, and `tests/test_incident_records.py` fails CI if one does not:

```markdown
## The thing that broke

> **Incident record** · class `silent-wrong` ·
> introduced [`72bf410`](<repo>/commit/72bf410) ·
> fixed [`dabdd7b`](<repo>/commit/dabdd7b) ·
> detector [`tests/test_clawdeck_manifest.py`](<repo>/blob/main/tests/test_clawdeck_manifest.py)
```

Seven classes, and the split is the point of keeping them:

| class | meaning |
|---|---|
| `silent-wrong` | ran clean, plausible output, incorrect |
| `silent-noop` | reported success having done nothing |
| `false-green` | a check passed input it should have rejected |
| `hang` | no error, no progress |
| `fails-loud` | crashed — the easy class, and the rare one here |
| `doc-drift` | a published claim diverged from the code |
| `cross-cutting` | a lesson spanning incidents, not a defect itself |

```bash
uv run scripts/incidents.py          # the table, with latency from git
uv run scripts/incidents.py --csv    # regenerate incidents.csv
```

`incidents.csv` is **derived and committed**, and the suite fails if it drifts
from the prose — regenerate it in the same commit that adds a record.
Latencies are computed from git rather than written down, so they cannot rot.

Two things that are easy to get wrong here:

- **One record describes one instance.** The scoped-claim section covers two,
  and listing all their commits together dated the fix ten days *before* the
  introduction. Negative latency is now a hard failure rather than a number
  absorbed into the median.
- **CI needs `fetch-depth: 0`.** `actions/checkout` defaults to a shallow
  clone, which makes every SHA unresolvable — and the suite would then pass by
  finding nothing to check. It detects the shallow case and fails instead.

What the corpus currently supports, as measurements rather than impressions:
**10 of 13 defects did not crash**, and latency ranges from same-day to 367
days. That asymmetry — the quiet ones survive a year — is the argument for
every property-based test in `tests/`.

## The Clawdeck lab manifest

`clawdeck.yaml` at the repo root is the **only** integration point with
[clawdeck-app.com](https://clawdeck-app.com), which boots a GPU box, clones
this repo and builds a Lab picker from it. Never put Clawdeck-specific code in
a training script.

Every directory with a `pyproject.toml` must appear in it, and
`tests/test_clawdeck_manifest.py` **fails CI** if one does not — because the
symptom otherwise shows up in a different product with no error on either side.
That already happened once: Clawdeck hardcoded `01_basic_neuralnet`, this repo
restructured, and every Clawdeck boot failed its pre-install until a human
noticed.

The subtle check is `gpu.count`. Where a `ds_config.json` hardcodes
`train_batch_size`, `micro` and `grad_accum`, it has pinned the GPU count and
DeepSpeed asserts it at startup. `01_basics/04_rnn` and
`03_llms/02_trl_sft` both require **2** GPUs and were both registered in
`EXAMPLES` as needing 1 — fixed, and now cross-checked.

Note that `scripts/check_contract.py` is **advisory and not in CI**, so the
manifest is guarded by a `tests/` suite instead. Its per-example "registered in
clawdeck.yaml" note is a convenience for contributors, not the gate.

Two rules exist because Clawdeck reads this file **live from `main`**, cached
five minutes, with no deploy and no review step — whatever lands is in front of
learners within minutes:

- **`gpu.count` must be 1, 2, 4 or 8**, with `min_vram_gb` ≤ 180 at counts 1–2
  and ≤ 80 at 4–8. Clawdeck books an *exact* count from a fixed catalog, so
  `count: 3` is unbookable and the lab dead-ends as "Needs a different machine".
  The table in `tests/test_clawdeck_manifest.py` mirrors their catalog; if it
  changes, that constant is what to edit.
- **Any run entry without `--num_gpus` is advertised as "Runs now — no GPU
  needed"** and shown first, even for locked labs. So such an entry must really
  run on CPU, or carry `needs_gpu: true`. That field covers the five examples
  that deliberately skip the deepspeed launcher, and Clawdeck reads it as of
  their v169.

  The check for this is **reachability-based, not string-based**, and that
  distinction is the whole difficulty. Two guard forms exist here —
  `require_gpu()` and an inline
  `if not torch.cuda.is_available() and ALLOW_CPU != "1"` — and mere presence
  proves nothing, because four scripts put the inline form inside
  `if args.model:` and default `--model` to `None`. Six manifest entries sit in
  exactly that position and all six exit 0 on a CPU-only box. `gpu_guards()` in
  `tests/test_clawdeck_manifest.py` therefore returns each guard with the flags
  gating it, and treats a guard as reachable when it is unconditional **or** the
  command passes any gating flag — over-flagging is recoverable with
  `needs_gpu`, under-flagging ships a lie to a learner.

Clawdeck **fails open**: an unreachable or malformed manifest yields zero labs,
so the Lab tab disappears and plain compute keeps working. The flip side is that
a broken manifest looks like *"this course has no labs"* rather than an error —
which is why the CI gate is the thing making that trade safe.

`main` is **branch-protected**: the `Logic tests (no GPU required)` check is
required, `enforce_admins` is on, and force-pushes and deletions are blocked. A
direct push to `main` is rejected — including for repo admins — so **every
change goes through a PR**, and the manifest cannot reach learners without the
gate having passed. Emergency escape hatch, if you ever truly need it:

```bash
gh api -X DELETE repos/yiqiao-yin/deepspeed-course/branches/main/protection
# ... fix, push ... then re-enable, or the last window stays open
```

## Scaffolding a new example

```bash
uv run scripts/new_example.py 10_my_topic --title "My Topic" --vram 24
```

Writes the four files with the contract already met — `require_gpu()` wired, a
portable `ds_config.json` (omits `train_batch_size` so any `--num_gpus` works),
a SLURM script with secrets left commented, a pre-headed README, and a test stub.
It prints the `runpod_ctl.py` line to add but deliberately does **not** edit that
shared file itself.

Register the printed line, then `./tests/run_all.sh` is green before any of your
own code exists. Skip the registration and exactly one check fails — that is the
suite enforcing its own checklist, not a broken scaffold.

`scripts/` holds five tools. Three answer questions about an example; two
generate figures for the book:

| Tool | Question | Gate? |
|---|---|---|
| `new_example.py` | "give me a skeleton that already satisfies the contract" | — |
| `check_contract.py` | "does this example work for all three readers?" | advisory |
| `audit_readmes.py` | "has this README drifted from the code?" | advisory, over-reports |
| `make_protein_animations.py` | six figures for `06_protein_folding` (PNG + GIF) | — |
| `make_casp14_figure.py` | the CASP14 panel, from the public archive + RCSB | — |

The two figure scripts carry **PEP 723 inline metadata and no `pyproject.toml`**,
deliberately. A `pyproject.toml` would oblige an entry in `clawdeck.yaml` (CI
enforces it), and these are authoring tools rather than labs — a learner should
never be offered "run the figure generator" as an exercise. `uv run
scripts/<name>.py` still provisions them.

## Conventions to preserve

- **Secrets are commented placeholders — and this is load-bearing.** Credential
  lines appear as:

  ```bash
  # export WANDB_API_KEY="your_key_here"
  ```

  They must stay **commented and quoted**. An uncommented
  `export WANDB_API_KEY=<ENTER_KEY_HERE>` is a **bash syntax error** — `<` is a
  redirection operator — so the script aborts on that line and never reaches the
  training command. Seven SLURM scripts shipped that way and could never run.
  `tests/test_runpod_ctl.py` now runs `bash -n` over every shell script to stop
  this recurring. Never substitute a real key.
- **W&B is optional and soft.** Training scripts wrap `import wandb` in `try/except ImportError` and only enable tracking when `WANDB_API_KEY` is set. Keep new scripts runnable with no W&B installed.
- **Heavy docstrings and comments.** Line-by-line explanatory comments (including on `#SBATCH` directives) are the pedagogical point, not clutter. Match the surrounding density.
- **Type hints** on function signatures throughout the Python scripts.
- Scripts print banner blocks (`"=" * 80`, emoji headers) around phases — expected output in the READMEs matches this, so changing print formatting invalidates docs.
- **Fail loudly, never silently.** Every serious bug this repo has shipped ran
  fine and was quietly wrong: a frame extractor returning one image repeated, a
  collator silently dropping `pixel_values`, a scaler fit before the train/test
  split, an eval harness whose RNG was correlated with the answer key (a random
  baseline scored 100%), and a spoken-answer scorer that matched `"not Paris"`
  against `"Paris"` by substring. Raise rather than returning a placeholder, and
  if a pipeline can be misconfigured into doing nothing, **assert that it did
  something** — the multimodal collators check that pixels and audio features
  actually arrived.

### Not every entry point is a training script

...and the name should say so. `03_llms/01_llm_finetuning/analyze_kimi_k3.py` is
named `analyze_`, not `train_`, because it does not train — Kimi K3 is 2.78 T
parameters / 1,561 GB and its remote code does not import on the pinned
transformers, so there is no run to have. Two things follow that generalise:

- **It carries no `require_gpu()`, deliberately, and says so in a comment.**
  `--plan` reads a JSON file over HTTPS; `--verify-arch` builds on the meta
  device. A guard that can never fire is decoration, and the same cargo-cult
  objection applies as to a distributed launcher with nothing to distribute.
- **A script with no end-to-end run makes its DERIVATIONS the only testable
  surface.** `tests/test_kimi_k3_plan.py` runs the shipped functions against a
  fixture config and caught three defects on its first run — all populated fields
  of the right type holding plausible small integers.
  → [postmortem](POSTMORTEMS.md#a-script-with-no-end-to-end-run-makes-its-derivations-the-only-testable-surface)

## Documentation site

`docusaurus-docs/` is a Docusaurus 3 site mirroring the examples, deployed to GitHub Pages at `https://yiqiao-yin.github.io/deepspeed-course/` by `.github/workflows/deploy-docs.yml` — it triggers **only** on pushes to `main` that touch `docusaurus-docs/**`.

```bash
cd docusaurus-docs
npm install
npm start          # local dev server with hot reload
npm run build      # must pass before pushing — CI runs this with NODE_OPTIONS=--max-old-space-size=4096
npm run serve      # preview the production build
```

There are **two** CI workflows:

- `deploy-docs.yml` — builds and deploys the site. Runs only on pushes touching `docusaurus-docs/**`. `onBrokenLinks`, `onBrokenAnchors` and `onBrokenMarkdownLinks` are all `throw`, so link rot fails the build.
- `tests.yml` — runs every suite in `tests/` plus a `compileall` over all training scripts, on every push and PR.

- Every doc page needs `---\nsidebar_position: N\n---` frontmatter **and** an entry in `sidebars.js` under `tutorialSidebar` — a page missing from `sidebars.js` is orphaned and nothing in the Docusaurus build warns you. `tests/test_docs_style.py` now checks this.
- KaTeX math (`remark-math` + `rehype-katex`) and Mermaid (`@docusaurus/theme-mermaid`) are enabled; tutorial pages use ```` ```mermaid ```` blocks liberally.
- When you change an example's code or hardware requirements, update both its folder `README.md` and the corresponding page under `docusaurus-docs/docs/tutorials/`.
- The site is **dark-mode only** (`colorMode: {defaultMode:'dark', disableSwitch:true}`). Mermaid uses ELK layout with a dark-blue palette, both set **globally** in `docusaurus.config.js`. Diagrams are optional, but one that is added must declare all **five** house `classDef`s:

  ```
  classDef deep   fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
  classDef dark   fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
  classDef base   fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
  classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
  classDef steel  fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
  ```

  `deep` for subgraphs/containers, `base` for ordinary nodes, `bright` for the
  node the eye should land on. (Names are historical — `steel` is actually the
  lightest, not `bright`.)

  **Never put `%%{init: ...}%%` or `layout: elk` inside a diagram.** It overrides
  the global config and drifts silently. `tests/test_docs_style.py` enforces the
  palette, the absence of inline overrides, label quoting, and that the config
  still sets what CONTRIBUTING.md publishes — all 48 diagram pages conform.

- **The site's own background is `#000000`**, not the dark blue of the Mermaid
  palette above — `custom.css` sets `--ifm-background-color: #000000` and dark
  mode is the only mode. Those are two different colour systems and it is easy
  to assume one from the other.
- **`headTags` in `docusaurus.config.js` carries the iOS home-screen icon**, and
  it is load-bearing: iOS Safari ignores `<link rel="icon">` for "Add to Home
  Screen" and shows a **letter tile** without an `apple-touch-icon`. The site
  did exactly that — a bare "D" — until the tags were added. The icons under
  `static/img/` (180 for iOS, 192 and 512 for the manifest) are **pre-flattened
  onto black on purpose**: iOS does not honour transparency in home-screen icons
  and composites it to black regardless, so doing it deliberately makes the
  result the site's own background instead of the renderer's choice. iOS also
  masks the corners, hence the padding.

- **Verifying a deployed page needs a content check, not a status code.** A 200 only proves *a* page is there, not the new one — and a literal `grep` for text inside a KaTeX block will fail because it renders into split HTML spans. Match on plain prose instead.
