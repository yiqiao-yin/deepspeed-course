# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A teaching course, not an application. Each top-level numbered directory (`01_basics/01_neuralnet` … `05_video_speech`) is a **self-contained, runnable DeepSpeed example** that escalates in difficulty: toy MLP → CNN → LSTM → Bayesian MCMC → HuggingFace/TRL fine-tuning → GRPO RL → LoRA SFT of 20B models → video-text → video-speech-to-speech.

There is no package and no shared library. Directories deliberately duplicate code rather than import from each other — a reader should be able to open one folder and run it without touching the rest. **Do not refactor shared logic into a common module.** (`require_gpu()` appears verbatim in ~22 files on purpose.)

There *is* a regression suite in `tests/` (CPU-only, runs in CI) and a GPU tier in `tests/gpu/`; see [What can and cannot be run here](#what-can-and-cannot-be-run-here). Tooling lives in `runpod/` for provisioning GPUs on demand, and `scripts/` for scaffolding and drift auditing.

Contributions from outside are welcome and governed by `CONTRIBUTING.md`, which is written to double as a spec an agent can follow. Read it before adding an example — it encodes the three-platform contract below. Repo is MIT (`LICENSE`).

### The alignment thread spans five topic folders

`05`–`06` are not independent examples; they are one escalating argument about
**what you can delete from the RLHF pipeline**, and the deletions are different:

| Folder | Deletes | Reference model? |
|---|---|---|
| `03_huggingface/04_reward_model` | — (this IS the pipeline) | — |
| `03_huggingface/05_dpo` | the **reward model** (`--method` covers 6 objectives) | LoRA removes it |
| `03_huggingface/06_grpo` | the **critic** | yes |
| `03_huggingface/07_online_dpo` | — (re-adds sampling; needs a judge) | yes |

> "DPO removes the reward model" and "GRPO removes the critic" are two different
> claims about two different components. Conflating them is the most common
> confusion in this area, and the docs say so in three places on purpose.

The book pages run `rlhf-reward-modeling` → `preference-optimization` → `grpo-*`
→ `online-preference-methods` → `beyond-grpo`, ordered by **when the literature
arrived relative to GRPO (Feb 5, 2024)**. That ordering is deliberate and the
pages carry dated tables because the families genuinely straddle it — KTO
precedes GRPO by three days, ORPO and SimPO follow.

### Topics 08 and 09 are multi-subtopic

Most topics are a single flat folder. **`04_video_text/` and `05_video_speech/` each contain four
numbered subtopics**, because their subject matter escalates internally:

```
04_video_text/{hf_ds_vtt_test2, 01_qwen25vl_baseline, 02_token_compression,
        03_streaming_memory, 04_video_eval}
05_video_speech/{01_longcat_flash_omni, 02_thinker_talker,
        03_duplex_streaming, 04_omni_eval, data/}
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

### Which examples skip the `deepspeed` launcher

Five, and each for a stated reason. Do not "fix" these by adding DeepSpeed —
using a distributed launcher where there is nothing to distribute is cargo cult.
They carry `launcher="python"` in `runpod/runpod_ctl.py`:

| Example | Why |
|---|---|
| `03_huggingface/09_multi_agency` | drives TRL's `GRPOTrainer` directly |
| `04_video_text/04_streaming_memory` | streaming *inference* — sequential, no optimizer |
| `04_video_text/05_video_eval` | evaluation — short `generate()` calls |
| `05_video_speech/03_duplex_streaming` | duplex inference — slices arrive in order |
| `05_video_speech/04_omni_eval` | evaluation — modality-ablation `generate()` calls |

Every other example uses both `uv` and `deepspeed`. If you add a sixth
exception, say so explicitly and explain why.

## What can and cannot be run here

This distinction governs how to verify a change:

| Examples | Scale | Verification |
|---|---|---|
| `01`–`04` | Synthetic or tiny data, ≤1M params, 1–2 GPUs | **Runnable end to end** on a single machine |
| `05`–`09` | Real model downloads (GBs to 1.1 TB), multi-GPU, up to 560B params | **Not runnable locally.** Verify logic only |

For the second group, do not attempt a full training run to validate a change —
it will not fit, and a partial run proves nothing. Write or extend a **logic test**
in `tests/` instead, which exercises the changed code path without a GPU or a
model download:

### The big exception: eight modules ARE fully CPU-runnable

Their substance is *algorithms, objectives and policy* rather than weights, so
they need no GPU and no download. **Run these directly rather than mocking
them:**

| Module | What it is |
|---|---|
| `03_huggingface/05_dpo/preference_losses.py` | DPO / IPO / CPO / KTO / ORPO / SimPO, plain tensors |
| `03_huggingface/04_reward_model/reward_modeling.py` | Bradley-Terry objective |
| `04_video_text/03_token_compression/token_compression.py` | ToMe / FastV / DyCoke |
| `04_video_text/04_streaming_memory/star_memory.py` | STAR bounded memory bank |
| `04_video_text/05_video_eval/video_mme_eval.py` | eval harness |
| `05_video_speech/02_thinker_talker/tmrope.py` | the 40 ms shared clock — pure integer arithmetic |
| `05_video_speech/03_duplex_streaming/duplex.py` | turn-taking policy, barge-in, RTF |
| `05_video_speech/04_omni_eval/omni_eval.py` | modality-ablation grid |

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

```bash
./tests/run_all.sh              # all 21 suites, no GPU, no downloads
uv run tests/test_ds_configs.py # one suite
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

## Two target platforms

The README's central distinction, which shapes every launcher script:

- **CoreWeave** — shared SLURM HPC cluster. You SSH to a login node and *submit*; you never run training interactively. Scripts carry `#SBATCH` headers (`--gres=gpu:N`, `--partition=h200-low`, `--time`, `--mem`).
- **RunPod** — single-user pod with direct GPU access. Run `deepspeed ...` straight in the shell; the `#SBATCH` lines are inert comments.

`05_video_speech/01_longcat_omni/run_2xB200.sh` shows the non-SLURM style: preflight checks on GPU count, free disk, and RAM before launching.

### The three-platform contract

**Check it, do not reason about it:**

```bash
uv run scripts/check_contract.py 10_my_topic   # one example, ~28 checks
uv run scripts/check_contract.py               # the whole repo
uv run scripts/check_contract.py -v <folder>   # show passing checks too
```

Advisory, not a CI gate — older examples predate parts of the contract and
legitimately differ, and failing the build on those would water the checks down
until they catch nothing. The non-negotiable subset (`EXAMPLES` registration,
`bash -n`, `#SBATCH` presence) is in `tests/test_runpod_ctl.py` and does fail CI.

It earns its keep: pointing it at the repo found `03_huggingface/03_ocr` requesting
`--ntasks-per-node=2` while running `deepspeed --num_gpus=2` (four processes for
two GPUs — a hang), and 13 READMEs that never told a RunPod reader how to shut
the pod down.

> When it flags something, check whether the CHECKER is wrong before changing
> working code. Three of its original checks were over-strict and were fixed in
> the checker: `import torch` at module scope is harmless on a CPU box,
> `import deepspeed` likewise (the CUDA_HOME error comes from `initialize()`,
> which `require_gpu()` precedes), and `pip install uv` is legitimate because
> you cannot bootstrap uv with uv.

`CONTRIBUTING.md` states the contract in full; the short version, because every
new or edited example must satisfy it:

| Reader | Requirement |
|---|---|
| **no GPU** | `require_gpu()` called *before* torch/deepspeed are imported; message says why it stopped, what they can still do, and how to rent a GPU; `ALLOW_CPU=1` honoured |
| **CoreWeave** | a `run_deepspeed.sh` with `#SBATCH` headers and a cheap `--max-steps` dry-run path |
| **RunPod** | an `EXAMPLES` entry in `runpod/runpod_ctl.py`, and the README documents `run <ex> --dry-run --collect --wait --terminate --yes` plus confirming with `pods` |

Put heavy imports *inside* `main()`, after the preflight. Import torch at module
scope and a CPU-only reader gets a CUDA traceback before the message ever runs.

`tests/test_runpod_ctl.py` fails if a numbered example is missing from the
`EXAMPLES` table or lacks a SLURM script, so the contract is enforced rather
than merely documented.

**Never give the pod `RUNPOD_API_KEY`** — termination is driven from the local
machine in a `finally`, with a keyless in-pod watchdog as backstop. See
`SECURITY.md`.

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

`scripts/` holds three tools, and they answer different questions:

| Tool | Question | Gate? |
|---|---|---|
| `new_example.py` | "give me a skeleton that already satisfies the contract" | — |
| `check_contract.py` | "does this example work for all three readers?" | advisory |
| `audit_readmes.py` | "has this README drifted from the code?" | advisory, over-reports |

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
- **Never fabricate expected output.** If it has not been run, mark it *not yet
  verified on hardware*. A wrong published number costs a reader a day debugging
  their own correct setup.

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
  still sets what CONTRIBUTING.md publishes — all 39 diagram pages conform.

- **Verifying a deployed page needs a content check, not a status code.** A 200 only proves *a* page is there, not the new one — and a literal `grep` for text inside a KaTeX block will fail because it renders into split HTML spans. Match on plain prose instead.
