# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Scaffold a new course example that already satisfies the contribution contract.

    uv run scripts/new_example.py 10_my_topic --title "My Topic" --vram 24

Why this exists
---------------
CONTRIBUTING.md describes a four-file contract and a three-platform
compatibility promise (CPU / CoreWeave / RunPod). Describing it is not the same
as making it easy to satisfy, and a contract that is tedious to satisfy is one
people satisfy approximately.

So this generates the skeleton with the contract already met. Run it, register
the example in `runpod/runpod_ctl.py` (one line — the tool prints it for you),
and `./tests/run_all.sh` should be green **before you have written a single line
of your own code**. From there every failure is genuinely yours, which is the
whole point — you are never debugging the scaffolding and your model at the same
time.

Register nothing and the suite fails with exactly one message:
`every numbered example is in the requirements table`. That is the checklist
enforcing itself, and it is the intended first experience.

What it writes
--------------
    <folder>/train_<name>.py    entry point, with require_gpu() already wired
    <folder>/ds_config.json     portable DeepSpeed config (any --num_gpus)
    <folder>/run_deepspeed.sh   SLURM batch script (CoreWeave)
    <folder>/README.md          the four required sections, pre-headed

It does NOT touch runpod/runpod_ctl.py — registering your example there is a
one-line edit you should make deliberately, and CONTRIBUTING.md walks through
it. Silently editing a shared file from a scaffold tool is how merge conflicts
are born.

Stdlib only. Nothing is installed, nothing is downloaded.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Templates
#
# Placeholders are __UPPERCASE__ tokens rather than str.format fields, because
# the generated Python and JSON are full of braces that .format() would try to
# interpret. Plain replacement keeps the templates readable as the files they
# actually produce.
# ---------------------------------------------------------------------------

TRAIN_PY = '''"""
__TITLE__ — DeepSpeed example.

TODO(contributor): replace this docstring with a real explanation.

The docstrings in this repository are the teaching material, not decoration.
Say what the example demonstrates, why it is built this way, and what the
reader should notice. Heavy comments are the house style — match the density
of the surrounding examples rather than trimming to taste.

Requirements:
    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install deepspeed
    # ... plus whatever your example needs

Running it:
    sbatch run_deepspeed.sh                       # CoreWeave / SLURM
    deepspeed --num_gpus=__GPUS__ __SCRIPT__      # direct
"""

import argparse
import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    # Imported locally so this helper stays self-contained and can be copied
    # between example scripts unchanged.
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \\"torch_adam\\": true and "
              "fp16/bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\\n")
        return

    bar = "=" * 72
    print("\\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\\n  torch.cuda.is_available() returned False.")
    print("\\n  TODO(contributor): say whether THIS example can run on CPU.")
    print("  If it can, say how (smaller model? fewer steps? ALLOW_CPU=1?).")
    print("  If it cannot, say so plainly and point at one that can.")
    print("\\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\\n  No GPU at all? These need none:")
    print("      ./tests/run_all.sh    # the full logic suite, no GPU, no downloads")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend __FOLDER__")
    print("      uv run runpod/runpod_ctl.py run __FOLDER__ \\\\")
    print("          --collect --wait --terminate --yes")
    print("\\n" + bar + "\\n")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepspeed", default="ds_config.json",
                        help="Path to the DeepSpeed config.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap steps. The RunPod --dry-run path relies on "
                             "this to keep a smoke test cheap.")
    args = parser.parse_args()

    require_gpu()

    # Imported AFTER the preflight, so a missing GPU produces our message
    # rather than a CUDA error from inside torch's import chain.
    import deepspeed
    import torch

    bar = "=" * 72
    print(bar)
    print("  __TITLE__")
    print(bar)
    print(f"  device   {torch.cuda.get_device_name(0)}")
    print(f"  gpus     {torch.cuda.device_count()}")
    print(bar)

    # ------------------------------------------------------------------
    # TODO(contributor): build your model and dataset here.
    #
    # The placeholder below is a two-parameter linear model. It exists so the
    # scaffold RUNS end to end on a real GPU immediately -- prove the plumbing
    # works, then replace it. Do not ship it.
    # ------------------------------------------------------------------
    model = torch.nn.Linear(1, 1)

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed,
    )

    steps = args.max_steps if args.max_steps > 0 else 100
    for step in range(steps):
        x = torch.randn(4, 1, device=engine.device, dtype=torch.float32)
        y = 2.0 * x + 1.0                      # the function being learned
        loss = torch.nn.functional.mse_loss(engine(x.to(engine.dtype)).float(), y)
        engine.backward(loss)
        engine.step()
        if step % 10 == 0:
            print(f"  step {step:>4}  loss {loss.item():.6f}")

    weight = model.weight.item()
    bias = model.bias.item()
    print(f"\\n  Learned Weight: {weight:.6f}   (true value: 2.000000)")
    print(f"  Learned Bias:   {bias:.6f}   (true value: 1.000000)")
    print("\\n  Scaffold ran. Now replace the model above with your example.")


if __name__ == "__main__":
    main()
'''


DS_CONFIG = '''{
  "_comment": "TODO(contributor): explain the choices below. Configs in this repo are teaching material — a reader should learn WHY stage 2, WHY bf16, from the file itself.",
  "bf16": {
    "enabled": true
  },
  "_precision_comment": "bf16 needs Ampere (A100/3090) or newer. On older cards use fp16 instead — but NEVER enable both, DeepSpeed raises at initialization and tests/test_ds_configs.py will catch it.",
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "weight_decay": 0.01
    }
  },
  "zero_optimization": {
    "stage": 2,
    "_stage_comment": "Stage 2 shards optimizer states and gradients (16-psi -> 2-psi + 14-psi/N) at the SAME 2-psi communication volume as plain data parallelism. It is free. Stage 3 also shards parameters but costs 1.5x the communication; reach for it only when stage 2 still will not fit.",
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000.0,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000.0,
    "contiguous_gradients": true
  },
  "_batch_comment": "train_batch_size is DELIBERATELY OMITTED. DeepSpeed derives it as micro x accum x num_gpus, so this config is portable to any GPU count. Pin train_batch_size only if you have a reason, and then it MUST equal micro x accum x the --num_gpus your launcher requests, or the run aborts at startup. This is the single most common breakage in the repo.",
  "train_micro_batch_size_per_gpu": __MICRO__,
  "gradient_accumulation_steps": __ACCUM__,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
'''


RUN_SH = '''#!/bin/bash
# SLURM batch script — __TITLE__
#
# CoreWeave / any SLURM cluster:
#     sbatch run_deepspeed.sh
#
# RunPod (no SLURM there — the API driver creates and TERMINATES the pod):
#     uv run runpod/runpod_ctl.py run __FOLDER__ \\
#         --collect --wait --terminate --yes

#SBATCH --gres=gpu:__GPUS__
# TODO(contributor): justify this number. Why __GPUS__ and not 1?

#SBATCH --partition=h200-low
# Update to match your cluster's partitions (check with: sinfo)

#SBATCH --time=02:00:00
# Wall-clock ceiling. The job is killed at this point, so overestimate.

#SBATCH --job-name=__JOBNAME__

#SBATCH --ntasks-per-node=1
# ONE task. The `deepspeed` launcher spawns one worker per GPU itself; letting
# SLURM also start one task per GPU gives N^2 processes and usually a hang.

#SBATCH --cpus-per-task=8
# Cores for the data pipeline. Too few starves the dataloader and the GPU
# idles between batches — which looks like a slow model and is not.

#SBATCH --mem=__MEM__G

#SBATCH --output=logs/__JOBNAME___%j.out
#SBATCH --error=logs/__JOBNAME___%j.err

set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     ${SLURM_NODELIST:-local}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:    $(date)"
echo "=================================================="

# Environment, built ONCE on a LOGIN node with uv. Compute nodes usually have
# no network egress, so building it inside the job fails.
#   uv venv ~/myenv && source ~/myenv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#   uv pip install deepspeed
if [ -f ~/myenv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/myenv/bin/activate
fi

# $HOME is usually a small NFS quota and a multi-GB model download into it
# fails slowly. Point the cache at scratch.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# Credentials, if your example needs them. KEEP THESE COMMENTED AND QUOTED.
# An uncommented `export HF_TOKEN=<ENTER_KEY_HERE>` is a bash SYNTAX ERROR —
# `<` is a redirection operator — so the script aborts on that line and never
# reaches the training command. Seven scripts shipped that way once and could
# never run. tests/test_runpod_ctl.py runs `bash -n` over every shell script
# to stop it recurring.
# export HF_TOKEN="your_value_here"
# export WANDB_API_KEY="your_value_here"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-__GPUS__}"

deepspeed --num_gpus="${NUM_GPUS}" __SCRIPT__ \\
    --deepspeed ds_config.json \\
    "$@"

echo "=================================================="
echo "End: $(date)"
echo "=================================================="
'''


README_MD = '''# __FOLDER__ — __TITLE__

TODO(contributor): one paragraph. What does this example teach, and why does it
belong in a DeepSpeed course? A reader should be able to decide from this
paragraph alone whether to keep reading.

## What this demonstrates

TODO. Be specific about the DeepSpeed mechanism at issue — ZeRO stage, offload,
pipeline parallelism, activation checkpointing. "Trains a model" is not a
mechanism.

## Hardware requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | __VRAM__ GB | TODO: per GPU |
| GPUs | __GPUS__ | TODO: why this many |
| Disk | __DISK__ GB | TODO: model download size |
| Host RAM | __MEM__ GB | TODO |

TODO: state plainly whether this runs on CPU. If it does not, say so — and say
which example teaches the same mechanic at a size that does.

## Environment & Local Testing

Packages via **`uv`**. Never bare `pip`.

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed
# TODO: your example's dependencies
```

TODO: state what a reader with **no GPU** can still do here. Every example must
answer this. Options, in descending order of usefulness:

- a CPU-runnable subset (`ALLOW_CPU=1`, a smaller model, fewer steps)
- a logic test in `tests/` that exercises the changed code path with no GPU
- an honest "this one needs a GPU; run `./tests/run_all.sh` instead"

## Running it

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh
squeue -u $USER
tail -f logs/__JOBNAME___<jobid>.out
```

Build the venv on a **login** node — compute nodes usually have no egress.
Adjust `--partition` to match your cluster (`sinfo` lists them).

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...        # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py recommend __FOLDER__
uv run runpod/runpod_ctl.py run __FOLDER__ \\
    --dry-run --collect --wait --terminate --yes

uv run runpod/runpod_ctl.py pods       # confirm: "Nothing is billing."
```

`--dry-run` caps the training step so a smoke test stays cheap. `--terminate`
deletes the pod in a `finally` block, so a crash, a network failure or Ctrl-C
still stops the billing.

### Direct

```bash
deepspeed --num_gpus=__GPUS__ __SCRIPT__ --deepspeed ds_config.json
```

## Expected output

TODO: paste the real banner and final lines from an actual run. Not invented
output — a reader compares against this to decide whether their run worked, so
fabricated output is worse than none.

```
================================================================
  __TITLE__
================================================================
  ...
```

## Configuration notes

TODO: explain the `ds_config.json` choices. Which ZeRO stage, and why? What
would you change to fit a smaller card?

Remember the invariant DeepSpeed enforces at startup:

```
train_batch_size == train_micro_batch_size_per_gpu x gradient_accumulation_steps x num_gpus
```

This config omits `train_batch_size` so DeepSpeed derives it and the config
stays portable across GPU counts. If you pin it, changing `--num_gpus` in the
launcher without updating the JSON aborts the run.
'''


TEST_PY = '''# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Logic test for __FOLDER__.

Run:
    uv run tests/__TESTNAME__

TODO(contributor): replace this with real assertions.

What belongs here
-----------------
The point of tests/ is to verify examples that CANNOT be run locally — no GPU,
no multi-GB download. So test the LOGIC of the change, not the training run.

Prefer asserting mathematical or structural PROPERTIES over shapes. The bugs
this repository has actually shipped all had the same character: the code ran
fine, the loss decreased, and the result was quietly wrong. A shape assertion
would have passed on every one of them.

`tests/_srcload.py` extracts a single function from a training script via `ast`,
so you can test the ACTUAL shipped source without importing torch or deepspeed:

    from _srcload import Results, load_function, source_contains

    fn = load_function("__FOLDER__/__SCRIPT__", "my_function")
    r.check(fn(...) == expected, "describes what is guaranteed")

Add this file to tests/run_all.sh and .github/workflows/tests.yml when it does
something real.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, source_contains  # noqa: E402

FOLDER = "__FOLDER__"


def main() -> int:
    r = Results("__TITLE__ — logic checks")

    # A starter check that is genuinely worth keeping: the preflight must be
    # present, or a CPU-only reader gets an unreadable CUDA error instead of
    # an explanation.
    r.check(
        source_contains(f"{FOLDER}/__SCRIPT__", "require_gpu"),
        "the entry point has a require_gpu() preflight",
        "Every example must fail gracefully without a GPU. See CONTRIBUTING.md.",
    )

    # TODO(contributor): add checks that would have caught a real bug.

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
'''


def slugify(name: str) -> str:
    """Turn a folder name into a safe identifier for scripts and job names."""
    base = name.split("/")[-1]
    # Strip a leading numeric prefix: 10_my_topic -> my_topic
    base = re.sub(r"^\d+_", "", base)
    return re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_") or "example"



PYPROJECT = """\
# =============================================================================
# __FOLDER__ as a self-contained uv project.
#
#     cd __FOLDER__
#     uv sync                    # creates .venv, installs the LOCKED versions
#     uv run deepspeed --num_gpus=__GPUS__ __SCRIPT__
#
# The committed uv.lock is required (CONTRIBUTING.md section 4): it is what
# makes every reader install the same versions. Without it `uv pip install`
# resolves to whatever is newest that day, which is how a tutorial that worked
# in March breaks in September with nobody having touched it.
#
# After adding your real dependencies below, run `uv lock` and COMMIT the lock.
# torch is pinned to an explicit CUDA index (cu128) below. PyPI's DEFAULT torch
# is a CUDA 13 wheel, which installs fine on a pre-CUDA-13 driver (550/570) and
# then reports cuda.is_available() == False -- silently, while nvidia-smi shows
# the card. cu128 works on old and new drivers alike.
# =============================================================================

[project]
name = "deepspeed-course-__PROJNAME__"
version = "0.1.0"
description = "__TITLE__"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.2",
    "deepspeed>=0.14",
    # TODO: add what your example actually imports, e.g.
    # "transformers>=4.40", "datasets>=2.18", "peft>=0.10",
]

[project.optional-dependencies]
# W&B stays OPTIONAL: the training script wraps `import wandb` in try/except
# and only tracks when WANDB_API_KEY is set. Making it required contradicts
# the code and tests/test_runpod_ctl.py will fail you for it.
tracking = ["wandb>=0.16"]

[tool.uv]
# Runnable example, not a distributable library. Without this uv tries to BUILD
# the folder as a package and `uv sync` fails.
package = false

# Pin the CUDA build rather than inheriting PyPI's default -- see the header.
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
"""

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder",
                        help="Folder to create, relative to the repo root. "
                             "Top-level examples are numbered (10_my_topic); "
                             "subsections may nest (08_vtt/05_my_idea).")
    parser.add_argument("--title", default=None,
                        help="Human-readable title. Defaults to the folder name.")
    parser.add_argument("--vram", type=int, default=24,
                        help="Minimum VRAM per GPU, in GB.")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--disk", type=int, default=60,
                        help="Disk needed, in GB (model downloads included).")
    parser.add_argument("--mem", type=int, default=48,
                        help="Host RAM, in GB.")
    parser.add_argument("--micro", type=int, default=4,
                        help="train_micro_batch_size_per_gpu.")
    parser.add_argument("--accum", type=int, default=1,
                        help="gradient_accumulation_steps.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing folder.")
    args = parser.parse_args()

    folder = args.folder.strip("/")
    target = REPO_ROOT / folder

    if target.exists() and not args.force:
        print(f"error: {folder} already exists. Use --force to overwrite.")
        return 1
    if args.gpus < 1 or args.vram < 1:
        print("error: --gpus and --vram must be positive.")
        return 1

    slug = slugify(folder)
    title = args.title or folder.split("/")[-1].replace("_", " ").title()
    script = f"train_{slug}.py"
    testname = f"test_{slug}.py"

    subs = {
        "__PROJNAME__": folder.replace("/", "-").replace("_", "-").lower(),
        "__FOLDER__": folder,
        "__TITLE__": title,
        "__SCRIPT__": script,
        "__TESTNAME__": testname,
        "__JOBNAME__": slug[:16],
        "__GPUS__": str(args.gpus),
        "__VRAM__": str(args.vram),
        "__DISK__": str(args.disk),
        "__MEM__": str(args.mem),
        "__MICRO__": str(args.micro),
        "__ACCUM__": str(args.accum),
    }

    def render(template: str) -> str:
        for key, value in subs.items():
            template = template.replace(key, value)
        return template

    target.mkdir(parents=True, exist_ok=True)

    written = []
    for filename, template, executable in [
        (script, TRAIN_PY, False),
        ("ds_config.json", DS_CONFIG, False),
        ("run_deepspeed.sh", RUN_SH, True),
        ("README.md", README_MD, False),
        ("pyproject.toml", PYPROJECT, False),
    ]:
        path = target / filename
        path.write_text(render(template), encoding="utf-8")
        if executable:
            # A script documented as `./run_deepspeed.sh` that is not +x is a
            # papercut this repo has shipped three times.
            path.chmod(0o755)
        written.append(path.relative_to(REPO_ROOT))

    test_path = REPO_ROOT / "tests" / testname
    if not test_path.exists():
        test_path.write_text(render(TEST_PY), encoding="utf-8")
        written.append(test_path.relative_to(REPO_ROOT))

    bar = "=" * 72
    print(bar)
    print(f"  Scaffolded {folder}")
    print(bar)
    for path in written:
        print(f"    {path}")

    print()
    print("  Next, in order:")
    print()
    print("  1. Resolve and COMMIT the lock — `uv sync` must work from a clone:")
    print()
    print(f"         cd {folder}")
    print("         uv lock && uv sync")
    print("         uv run python -c \"import torch, deepspeed\"")
    print()
    print("     tests/test_runpod_ctl.py fails without a committed uv.lock.")
    print()
    print("  2. Register the example so RunPod users can rent the right card.")
    print("     Add to EXAMPLES in runpod/runpod_ctl.py:")
    print()
    print(f'         "{folder}": dict(min_vram={args.vram}, gpus={args.gpus}, '
          f'disk={args.disk},')
    print(f'                          script="{script}",')
    print('                          note="TODO: the one thing that surprises '
          'people."),')
    print()
    print("  2. Verify the scaffold is green BEFORE you write anything:")
    print("         ./tests/run_all.sh")
    print()
    print("     Skip step 1 and you will see exactly one failure:")
    print('         FAIL  every numbered example is in the requirements table')
    print("     That is the suite enforcing the checklist, not a broken")
    print("     scaffold. Once green, every later failure is genuinely yours —")
    print("     you are never debugging the scaffolding and your model at once.")
    print()
    print("  3. Replace every TODO(contributor). Search for them:")
    print(f"         grep -rn 'TODO(contributor)' {folder} tests/{testname}")
    print()
    print("  4. Write a real test in tests/" + testname + ", then register it")
    print("     in tests/run_all.sh and .github/workflows/tests.yml.")
    print()
    print("  5. Add a docs page under docusaurus-docs/docs/tutorials/ and an")
    print("     entry in sidebars.js — a page missing from sidebars.js is")
    print("     orphaned and the build will not warn you.")
    print()
    print("  Full checklist: CONTRIBUTING.md")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
