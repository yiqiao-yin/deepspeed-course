# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A teaching course, not an application. Each top-level numbered directory (`01_basic_neuralnet` … `09_vss`) is a **self-contained, runnable DeepSpeed example** that escalates in difficulty: toy MLP → CNN → LSTM → Bayesian MCMC → HuggingFace/TRL fine-tuning → GRPO RL → LoRA SFT of 20B models → video-text (LLaVA/NLLB) → video-speech (LongCat-Flash-Omni 560B).

There is no package and no shared library. Directories deliberately duplicate code rather than import from each other — a reader should be able to open one folder and run it without touching the rest. **Do not refactor shared logic into a common module.**

There *is* a regression suite in `tests/` (CPU-only, runs in CI) and a GPU tier in `tests/gpu/`; see [What can and cannot be run here](#what-can-and-cannot-be-run-here). Tooling lives in `runpod/` for provisioning GPUs on demand.

## The per-example contract

Every example folder follows the same four-file shape. When adding or editing an example, keep it:

| File | Role |
|---|---|
| `train_*.py` | Training entry point; calls `deepspeed.initialize(...)` and reads the JSON config |
| `ds_config*.json` | DeepSpeed config — ZeRO stage, fp16/bf16, optimizer, batch sizes |
| `run_deepspeed.sh` (or `submit_job.sh`, `run_training.sh`, `run_2xB200.sh`) | SLURM batch script, or a bare launcher for single-pod platforms |
| `README.md` | Full standalone walkthrough: hardware, setup, run command, expected output |

Larger examples add `HARDWARE_REQUIREMENTS.md` / `HARDWARE_GUIDE.md` / `MODEL_IMPROVEMENT_STRATEGY.md`.

Batch size consistency is enforced by DeepSpeed at startup: `train_batch_size == train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`. Changing `--num_gpus` in a launcher without updating the JSON is the most common breakage.

## Running examples

```bash
cd 01_basic_neuralnet
deepspeed --num_gpus=1 train_ds_enhanced.py          # direct, e.g. RunPod / single pod
sbatch run_deepspeed.sh                              # SLURM, e.g. CoreWeave
```

SLURM workflow: `sbatch <script>` → `squeue -u $USER` → `tail -f logs/<name>_<jobid>.out` → `scancel <jobid>`. Every batch script does `mkdir -p logs` and writes `logs/*_%j.{out,err}`.

## Tooling: always `uv`

Environments and package installs use **`uv`**, never bare `pip` or conda — including
for throwaway checks:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed wandb

uv run script.py                    # run a script
uv run --no-project python -c "..." # one-off, no project env
```

Every example README documents its own `uv` setup under **Environment & Local Testing**.

Note that `07_huggingface_trl_multi_agency` does **not** use DeepSpeed — it drives
TRL's `GRPOTrainer` directly, so `uv` alone suffices. Every other example uses both.

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

> **Exception worth knowing: `08_vtt/02_token_compression/` and
> `08_vtt/03_streaming_memory/` are fully CPU-runnable**, because their substance
> is *algorithms* rather than weights — ToMe/FastV/DyCoke and the STAR memory
> bank are plain PyTorch on plain tensors. Same for `08_vtt/04_video_eval/`'s
> harness. Run these directly rather than mocking them, and prefer asserting
> **mathematical properties** over shapes: compression code fails by running
> fine and being quietly wrong, so `test_token_compression.py` checks that the
> log-size attention identity holds to 1e-6 and that the *uncorrected* version
> is measurably wrong. A test that only checks tensor shapes would pass on
> completely broken compression.

```bash
./tests/run_all.sh              # all suites
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

`09_vss/run_2xB200.sh` shows the non-SLURM style: preflight checks on GPU count, free disk, and RAM before launching.

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

- Every doc page needs `---\nsidebar_position: N\n---` frontmatter **and** an entry in `sidebars.js` under `tutorialSidebar` — a page missing from `sidebars.js` is orphaned.
- KaTeX math (`remark-math` + `rehype-katex`) and Mermaid (`@docusaurus/theme-mermaid`) are enabled; tutorial pages use ```` ```mermaid ```` blocks liberally.
- When you change an example's code or hardware requirements, update both its folder `README.md` and the corresponding page under `docusaurus-docs/docs/tutorials/`.
