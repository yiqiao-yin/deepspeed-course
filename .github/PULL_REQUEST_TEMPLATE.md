<!--
Thanks for contributing! The checklist below is the same one in CONTRIBUTING.md.

It is long because it is the difference between "we accept contributions" and
"your contribution gets merged". Most items are satisfied automatically if you
scaffolded with:

    uv run scripts/new_example.py <folder>

Tick what applies. Delete sections that genuinely do not (a docs-only PR does
not need the RunPod section). An unticked box with an honest note beside it is
far better than a ticked box that is not true — especially for anything you
could not verify on hardware.
-->

## What this adds

<!-- One paragraph. What does it teach, and why does it belong in a DeepSpeed
     course? If it fixes a bug, describe the bug's SYMPTOM — how would someone
     have noticed it? -->

## Type

- [ ] New example / topic
- [ ] New subsection of an existing topic
- [ ] Bug fix
- [ ] Test
- [ ] Docs
- [ ] Tooling (`runpod/`, `scripts/`, `tests/`)

---

## The three-platform contract

<!-- See CONTRIBUTING.md §3. This is the section most likely to send a PR back. -->

### Reader A — no GPU (fails gracefully)

- [ ] `require_gpu()` is called **before** torch/deepspeed are imported
- [ ] Its message says *why* it stopped, *what the reader can still do*, and *how to rent a GPU*
- [ ] `ALLOW_CPU=1` is honoured
- [ ] The README states plainly whether this runs on CPU

### Reader B — CoreWeave (SLURM)

- [ ] `run_deepspeed.sh` ships with `#SBATCH` headers
- [ ] `--ntasks-per-node=1` (the `deepspeed` launcher spawns its own workers)
- [ ] `mkdir -p logs` before `--output=logs/...`
- [ ] A dry-run path exists (`--max-steps`) and is documented
- [ ] `sbatch` → `squeue` → `tail -f` → `scancel` documented in the README

### Reader C — RunPod (rent, run, **auto shut down**)

- [ ] Registered in `EXAMPLES` in `runpod/runpod_ctl.py`, with honest `min_vram` / `gpus` / `disk`
- [ ] README documents the auto-shutdown invocation:
      `run <example> --dry-run --collect --wait --terminate --yes`
- [ ] README tells the reader to confirm with `runpod_ctl.py pods`
- [ ] Nothing added to the bootstrap echoes a credential (`env`, `printenv`, `$*_TOKEN`)

**Did you actually run it on RunPod?**

- [ ] Yes — output below is real
- [ ] No — table entry and docs only; needs verification by someone with hardware

<!-- Saying "no" here is completely fine and costs you nothing in review.
     Inventing output is not fine. -->

---

## Hard rules

- [ ] **`uv`** everywhere — no bare `pip`, no conda, including in docs and comments
- [ ] **`deepspeed`** is used — or I explain below why this is a genuine exception
- [ ] Secrets are **commented and quoted** (`# export HF_TOKEN="your_value_here"`)
- [ ] W&B is wrapped in `try/except ImportError` and gated on `WANDB_API_KEY`
- [ ] No shared logic extracted into a common module (duplication is deliberate)
- [ ] Type hints on function signatures
- [ ] Comment/docstring density matches the surrounding examples

<!-- If deepspeed is NOT used, explain why here. Inference and evaluation are
     legitimate exceptions; "I didn't get to it" is not. -->

## DeepSpeed config

- [ ] Batch invariant holds (`train_batch_size == micro × accum × num_gpus`), **or** `train_batch_size` is omitted so DeepSpeed derives it
- [ ] `fp16` and `bf16` are not both enabled
- [ ] `"auto"` appears only where a HuggingFace `Trainer` resolves it
- [ ] ZeRO-3 sets `stage3_gather_16bit_weights_on_model_save`

## Verification

```
# paste the real output of:
./tests/run_all.sh
```

- [ ] `./tests/run_all.sh` passes
- [ ] A logic test exists for anything not runnable locally
- [ ] Test registered in **both** `tests/run_all.sh` and `.github/workflows/tests.yml`
- [ ] The test asserts a **property**, not just a tensor shape
- [ ] `cd docusaurus-docs && npm run build` passes (if docs changed)
- [ ] Docs page has `sidebar_position` frontmatter **and** a `sidebars.js` entry

## Honesty

- [ ] All expected output is **real**, or explicitly marked *"not yet verified on hardware"*
- [ ] Benchmark/memory numbers are measured, not estimated — or labelled as estimates
- [ ] I have the right to contribute this, and adapted work is cited

**What I could not verify:**

<!-- e.g. "No access to a multi-GPU box — ZeRO-3 sharding path is untested."
     This is genuinely useful to a reviewer. -->

---

## Used an AI coding agent?

- [ ] Yes — and I read the full diff and stand behind every factual claim in it
- [ ] No

<!-- Using Claude Code is encouraged (CONTRIBUTING.md §10). Saying so is useful
     context for review, not a mark against the PR. Agents are reliably weak at
     three things worth double-checking yourself:
       - inventing plausible expected output
       - writing tests that pass vacuously
       - "helpfully" refactoring duplication that is deliberate -->
