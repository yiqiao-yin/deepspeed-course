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

### `07_physical_ai` is a single escalating argument about embodiment

Seven labs, and each one exists because the previous one ran out of
something. They are not independent examples and the later ones assume
the earlier measurements:

| Folder | Adds | The thing it measured |
|---|---|---|
| `01_obstacle_hopper` | a body | the one-off bonus must be ONE-OFF, or standing on the box beats crossing it |
| `02_biped_stairs` | a second leg | 0.04-0.10 m treads climb reliably, above that they do not. **Every later lab's `STEP_MAX = 0.12` comes from here** |
| `03_terrain_vision` | an eye | a camera helps on ascent and nowhere else, because a corridor has nothing to decide |
| `04_terrain_navigation` | a second dimension | climb it or go round it -- the first lab with a routing choice in it |
| `05_mars_terrain` | a planet | terrain first (part 1), then a policy on it (parts 2-4) |
| `06_mars_mjx` | a bigger machine | GPU physics is 1.50x a 16-process CPU pool; the terrain is NOT why |
| `07_terrain_vlm` | an eye that reads | a VLM scores 52.9%; the robot that only FEELS scores ~75% |

**`05_mars_terrain` is split into parts ON PURPOSE.** Lab 4 published a
result that was wrong, and the faults were in its world, its renderer
and its metric rather than in anything a policy learned. Terrain and
policy are different problems; mixing them leaves you unable to say
which one broke. Part 1 therefore contains no robot at all.

Two presets and a course, and the distinction is load-bearing:
`rover` (60 m) is what parts 1-3 measured on, `course` (34 m) is the
forced-detour task from part 4, and `regional` (20 km) exists only to
be looked at. **New presets are ADDED, never substituted**, so
published numbers keep describing the surface they were measured on.

### `08_aerial_physics` asks the same question seven times and keeps changing the answer

Topic 7 walked a robot over terrain. Topic 8 throws a two-tonne
cold-gas car into the air and gives it **seven seconds of
propellant**, which removes the one luxury every walking lab had:
you cannot stop and think. A falling vehicle must commit.

| Folder | Adds | What it measured |
|---|---|---|
| `01_aerial_stabilization` | a body | PPO beats a textbook PD controller 89% to 75% **on a quarter of the gas** |
| `02_barrel_roll` | a manoeuvre | teach the landing first and the roll is **never learned** (0% vs 95%) — curriculum ORDER decides everything |
| `03_land_on_target` | a mark to hit | a 5x5 height patch takes PPO from 56% to **0%**; terrain knowledge makes it worse, monotonically |
| `04_fly_on_mars` | a planet and a course | residual PPO on a guidance law: same arrivals, **5-8% less propellant** |
| `05_thruster_failure` | a broken nozzle | practising with faults lifts survival **72% to 89%**; a static authority check does not predict which flights survive |
| `06_learned_dynamics` | a world model | **29,000x the parameters, no better prediction** — the best rung has 18,281 |
| `07_teacher_student` | a sensor and no altimeter | PPO on 144 depth beams scores **2.5%**, no better than blind; distilled from a teacher it scores **50%** |

Three things are load-bearing across the topic and easy to undo:

- **The worst terminal must be the outcome reachable by doing
  nothing.** Labs 3 and 5 are landings, so inaction IS a hover and
  the timeout must be worse than a crash. Lab 4 is a gate course
  where the bonus pulls the policy out of a hover, so there the
  crash must be worst. Lab 5 shipped lab 4's ladder once and taught
  the policy never to try.
- **Every lab after 2 inherits `STEP_MAX = 0.12`** and the reward
  shape from lab 3, whose Φ is capped near 6. Lab 3 shipped Φ ≈ 31
  and the `(γ−1)Φ` toll outweighed the prize: the policy fled, miss
  distance climbing 12 m → 86 m with every episode timing out.
- **A graded terminal, never a flat one.** A flat crash penalty
  scores a 10 m/s dive and a 2.6 m/s arrival identically, so there
  is no gradient from crashing toward landing.

Labs 1-6 carry **no `deepspeed` launcher and no `require_gpu()`**,
deliberately: a ~14k-parameter policy stepped against MuJoCo is
faster on a CPU, measured. Lab 6 is the first to want a card and
lab 7 the first with a real supervised phase.

### `02_intermediate/02` and `05` are a matched pair, like `03` and `04`

| Folder | Scored by | Asks |
|---|---|---|
| `02_rnn_stock_data` | Theil's U, directional accuracy | can a model **predict** a price series? |
| `05_rl_trading` | money after costs, drawdown | is a prediction **worth acting on**? |

The second exists because the first cannot answer its own
follow-up. A forecast at **51.1% directional accuracy** — better
than chance, with a real +0.031 correlation — **loses 2.2% a year**
once it pays 5 bps a side. Three things decide that and none is
expressible as a squared error: holding is free and flipping is
not, a small edge is correct and unprofitable, and the objective
is a path (one return series shuffled 400 ways has an identical
mean and drawdowns from −23% to −71%).

Two rules there that look like fussiness and are not:

- **The universe is all nine SPDR sectors, and that IS the
  survivorship-bias fix.** Twelve hand-picked large caps liquid
  today return +23.70%/yr against the equal-weight index's
  +13.80%. Buy-and-hold is the control every RL arm must beat, so
  on that basket the control starts ten points ahead and RL
  "loses" for reasons unrelated to RL. Taking ALL nine sectors
  means there is no selection decision to be biased by. The
  biased basket is kept as a labelled second arm so `--bias` can
  reproduce the gap.
- **The price panel is COMMITTED.** Vendor history is not
  immutable — splits are applied retroactively and adjustments
  revised — so a lab that pins numbers to a live feed publishes
  figures nobody can check.

### The geometry lesson both navigation labs had to learn separately

What makes terrain a DECISION is not obstacle size or route length. It
is the **ratio of route to arena**:

| | route | arena | ratio | detour |
|---|---|---|---|---|
| `04_terrain_navigation` | 7 m | 20 m | 35% | 1.36x |
| lab 5, first attempt | 8 m | 60 m | **13%** | **1.10x** |
| lab 5 `course` | 8.8 m | 34 m | 26% | **1.39x** |

At 13% the robot walks past the obstacles and the task contains no
choice, while every arrival number still looks reasonable. Lab 4 found
this through ridge LENGTH; lab 5 found it through arena SIZE; it is
the same fact. A generator that scales the box without scaling the
landforms -- or vice versa -- produces a lab that runs, trains,
renders and has nothing in it.

Related, and measured twice: **capping goal distance by walking B back
along the oracle's route destroys the task.** The first stretch of any
route is the straight run before the first obstacle, so the capped
goal lands on open ground and the detour collapses to 1.00-1.07x.
Select naturally-bent A-B pairs instead.

### A hardcoded joint index will eventually address the wrong joint

Lab 5 shipped this three times in one session, and each time it looked
like a learning failure rather than a wiring failure:

- `qpos[4:]` for the leg angles -- encoded "four root joints". Freeing
  torso pitch and roll made it six, and it raised. Lucky: `qpos[3]`
  for yaw would have kept returning a number, just the pitch angle.
- `ctrl[6] = action[6]` for the yaw motor -- encoded "six leg
  actuators". Adding hip abduction made it eight, so the yaw command
  was written into an ankle and the real yaw motor was never driven.
  The robot stood perfectly upright at a 0% fall rate and never moved,
  and three training runs were blamed on observation width.
- `random_policy(rng, n=7)` -- a default action width that no longer
  matched the body.

**Resolve joints and actuators by NAME, through `jnt_qposadr` and
`actuator_trnid`.** Note that MuJoCo actuators declared as
`<position joint="hipL"/>` are UNNAMED: `mj_id2name(mjOBJ_ACTUATOR, i)`
returns `None`, so filter on the joint each actuator drives, not on
the actuator. A name filter that silently returns an empty list makes
`ctrl[[]] = []` a no-op, which is the uncontrollable-robot failure
above.

### Measure the machine before renting one

Lab 5 reached "we need more compute" and the obvious move -- rent a
GPU -- was measurably wrong. `bench_compute.py` answers it in three
minutes on any box:

| | measured |
|---|---|
| 12k-param policy on GPU, rollout batch | **1.7x SLOWER** than CPU |
| 16x more envs in one process | 1.36x -- stepping was serialised |
| 16 worker processes (`vecenv.py`) | **2.5x** |
| raising `--eval-every` from 3 to 15 | another **1.6x** |

Two findings worth carrying: the environments were stepped in a Python
`for` loop, so a 20-core box ran at a twentieth of itself; and the
in-training evaluation was **~75% of wall clock**, because it stayed
serial when training went parallel. Total 3.9x, free, on hardware
already owned. `vecenv.py` asserts the parallel pool is BIT-IDENTICAL
to the serial loop before anyone trusts its speed.

**Lab 6 asked the same question of PHYSICS and got the same answer.**
`bench_backends.py` times MuJoCo on CPU against MJX and MuJoCo Warp on
the card, with the bar written down first — *≥ 5x the CPU worker pool
or it is not worth renting*:

| | measured |
|---|---|
| CPU, 16-process pool (the baseline) | 9,337 steps/s |
| MJX, best (then **aborts** at 4096 envs) | 0.14x |
| Warp, 4096 envs | **1.50x** — bar was 5x, **FAILS** |
| physics as a share of a training step | **33-34%**, so any physics win caps at **1.5x** end to end |

Three things from it that generalise:

- **Baseline against the pool, not one environment.** One CPU env is
  5,630 steps/s and sixteen processes are 9,337. Dividing by the
  single env would have reported Warp at 2.5x instead of 1.50x. The
  comparison that flatters the new thing is rarely the one you face.
- **A backend that `abort()`s is a data point, not a lost run.** MJX's
  OOM is not a catchable exception -- XLA hits a `Check failed` and
  takes the interpreter with it, which erased Warp's results *and* the
  ablation. Each measurement now runs in its own process.
- **Warp falls back to CPU silently** and reports throughput that
  looks exactly like GPU throughput. `nvidia-smi` cannot answer this:
  the question is whether JAX and Warp can *see* the card, which they
  fail independently. Ask each backend and print it before any number.

### The first lab here that needs ZeRO-3, and why the others do not

`07_terrain_vlm` is the FIRST example in this section that calls
`deepspeed.initialize`, and the justification is a measurement rather
than the course's name. `verify_arch.py`, weights + optimizer per card:

| Qwen2.5-VL-7B | 1 card | 8 cards |
|---|---|---|
| LoRA r=16 | **15.3 GB** -- already fits | 1.9 GB |
| full fine-tune | 119.7 GB | **15.0 GB** |

So the LoRA arm runs **plain python with no launcher** and only the
full fine-tune uses `ds_config.json`. Sharding 15.3 GB that fitted
would be the same cargo cult labs 1-6 avoid, pointed the other way.

Three things from it that generalise:

- **A control decides whether a number is readable.** The lab's
  sighted arm scores 52.9%; its BLIND arm -- identical model, grey
  image, same token count -- scores 45.0%, and the majority class
  scores 55.0%. Sight helps (McNemar p = 0.006) and still loses to a
  one-line heuristic. Reporting only the first comparison would have
  been a true sentence and a false impression.
- **A working no-information control emits ONE constant.** The blind
  arm answered AROUND for all 429 items, which is what licenses the
  comparison; a blind control with VARIED answers means information
  is leaking through the pipeline.
- **The analytic capacity table excludes activations, and they are
  not small.** Predicted 7.3 GB for the 3B LoRA arm, measured
  **11.5 GB**. ZeRO-3 shards parameters, gradients and optimizer
  state -- never activations.

### An ablation is only an ablation if ONE thing moved

Lab 6 first concluded the height-field collider costs **8.6x** of all
GPU throughput. Withdrawn: it compared a MuJoCo `plane` geom against
an `hfield`, which are **two different collision routines**. The
measurement was of the collider TYPE and the conclusion named the
TERRAIN.

Re-run with the collider held fixed — a height field of zeros against
one with relief — Mars costs **1.10x** against a 3.2% spread, for
**5.5x** the contacts. Rocks: nothing. Resolution: nothing. The
terrain is exonerated, and three explanations are ruled out rather
than one confirmed.

That shape — three NEGATIVE results — is exactly what a broken knob
produces, so `tests/test_mars_mjx.py` tests the knobs rather than the
conclusions: `rocks=0` must remove 75 geoms, `plane=True` must flatten
the elevation while keeping `nhfield == 1`. One of those checks
shipped unable to fail, asserting on `hfield_data` — which MuJoCo
**normalises to [0, 1]** at compile time, so its range is 1.000 for
every terrain. The metres are in `hfield_size[2]`.

### Arrival rate cannot tell you whether a policy reads terrain

Lab 5 part 3 is the cheap experiment that should come first.
`analyze_decisions.py` replays existing checkpoints and asks, for
every landform the path comes near, whether the robot went OVER it or
AROUND it. The robot discriminates about 3:1 -- **and so does an arm
fed six constant zeros.** The behaviour is emergent from having a
body, not from perception: you learn a scarp is impassable by walking
into one.

A policy that avoids every obstacle and one that reads terrain score
identically on arrival. Three parts of that lab measured outcome and
could not have distinguished them.

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
./tests/run_all.sh                 # all 60 suites, no GPU, no downloads
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
**13 of 16 defects did not crash**, and latency ranges from same-day to 367
days. That asymmetry — the quiet ones survive a year — is the argument for
every property-based test in `tests/`.

### A lab that claims something is BETTER declares a baseline and a budget

Scoped to one tier, deliberately. Most labs make no comparative claim and
demanding this of them would be cargo cult:

| tier | claims | requirement |
|---|---|---|
| 1 | nothing comparative (`01_neuralnet` fits y = 2x + 1) | none |
| 2 | a systems fact (memory, throughput, tokens/page) | budget only |
| 3 | **X beats Y on a measured axis** | **baseline + budget, enforced** |

Two visible lines near the top of the README, in this exact form —
`tests/test_tier3_claims.py` fails CI without them, and `TIER3` in that file
is the registry:

```markdown
**Baseline:** the untrained scorer, NDCG 0.4862 at every row.
**Budget:** epochs 1 to 40; the spread is 0.041 at one epoch and 0.001 at forty.
```

Tier 3 is where this repo has been confidently wrong twice — `11_moe`'s
load-balancing finding reversed at world size 2, and the ranking spread
collapses by a factor of 40 with training budget. Both were scoping errors,
not measurement errors.

**A keyword search is the wrong check here.** The first version grepped for
the word "baseline" and flagged `05_video_speech/04_omni_eval`, which has
had a proper two-sided control all along (model B ignores the video; the
harness must flag it *and* must not flag model A) and simply never used the
word. Hence the explicit marker. A substring is not a fact about the
document.

**State a falsifier too — not enforced, costs one line.** Any check for it
would be a check for a phrase. It is convention because it works:
`03_llms/12_prefill_decode` wrote one before measuring, printed
`HOLDS`/`FAILS` against it, and got `FAILS` on its first dry run. Not a bug
— at a 512-token prompt there are only two chunks and chunking genuinely
loses. A whole regime documented because the claim was written in a form
that could lose.

**For latency or throughput, report repeats and the spread.** One timed call
is a sample, not a measurement. In `12_prefill_decode` the baseline decode
step varies ±14% while the blocked step varies ±1.5%, so the ratio between
them wanders 4×–8× while the absolutes barely move — quote the absolutes.
The same discipline caught a two-point fit returning a **negative** cost per
cached token, which a single sample would have published as a confident
constant.

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
  still sets what CONTRIBUTING.md publishes — all 67 diagram pages conform.

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
