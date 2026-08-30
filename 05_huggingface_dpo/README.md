# 05 — Offline Preference Optimization (DPO and its descendants)

`05_huggingface_trl` does supervised fine-tuning: maximise the likelihood of a
reference answer. That can only say *"this output was good."* It has no way to
say *"this one was better than that one"* — and no way at all to push
probability **down**. A maximum-likelihood objective has no mechanism for it.

This folder adds the missing downward force, from preference pairs, with no
reinforcement learning. [`../06_huggingface_grpo/`](../06_huggingface_grpo/) is
the RL answer; this is the cheaper one, and for most alignment work it is the
right one.

**Book page:** [Preference Optimization](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/huggingface/preference-optimization)

## The family, by what each one deletes

Full RLHF holds four models — policy, critic, reward model, reference. Every
method here is an argument about which you can drop.

| Method | arXiv | Date | Deletes | Reference model? |
|---|---|---|---|---|
| **DPO** | [2305.18290](https://arxiv.org/abs/2305.18290) | May 2023 | reward model, rollouts | yes |
| **IPO** | [2310.12036](https://arxiv.org/abs/2310.12036) | Oct 2023 | — (bounds DPO) | yes |
| **CPO** | [2401.08417](https://arxiv.org/abs/2401.08417) | Jan 2024 | reference model | **no** |
| **KTO** | [2402.01306](https://arxiv.org/abs/2402.01306) | Feb 2024 | the need for *pairs* | yes |
| **ORPO** | [2403.07691](https://arxiv.org/abs/2403.07691) | Mar 2024 | reference model + SFT stage | **no** |
| **SimPO** | [2405.14734](https://arxiv.org/abs/2405.14734) | May 2024 | reference model, length bias | **no** |

> **GRPO deletes the critic**, which is a different component again. "DPO
> removes the reward model" and "GRPO removes the critic" are two different
> sentences, and conflating them is the most common confusion in this area.

## Environment & Local Testing

Packages via **`uv`**. Never bare `pip`.

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 05_huggingface_dpo
uv sync                    # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=1 train_dpo.py
```

`uv run` uses the project environment directly, so there is no
`activate` step. `uv sync --extra tracking` adds Weights & Biases,
which stays optional.

The lock is the point: everyone who clones resolves to identical
versions, instead of whatever `uv pip install` finds that day.
Regenerate deliberately with `uv lock --upgrade`.

<details>
<summary>Manual route, without the project</summary>

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed transformers trl peft accelerate datasets
```

PyPI's `torch` ships CUDA wheels now, so no `--index-url` is
needed; pinning cu121 gives an older CUDA than the default wheel.
</details>


### No GPU? The objectives still run — and they are the point

`preference_losses.py` implements all six as plain tensor maths. No model, no
download, no GPU:

```bash
uv run preference_losses.py                    # the comparison, measured
uv run ../tests/test_preference_losses.py      # 58 checks
```

Two properties it computes rather than claims:

**Which methods hold a second model in VRAM?** Perturb only the reference:

```
  DPO      delta =    -0.229387   USES the reference
  IPO      delta =    -2.000000   USES the reference
  CPO      delta =    +0.000000   reference-free
  ORPO     delta =    +0.000000   reference-free
  SimPO    delta =    +0.000000   reference-free
```

Those zeros are exact — the test asserts bit-level equality. For a 7B model,
reference-free is roughly **14 GB** you get back.

**Does length leak into the reward?** Every row below has per-token log-prob
−0.5, i.e. *identical quality by construction*:

```
    length    sum log-prob    avg log-prob
        10           -5.00           -0.50
        40          -20.00           -0.50
        80          -40.00           -0.50
```

DPO, IPO and CPO score with the sum; ORPO and SimPO normalise. The narrow,
sturdy claim: a length-unnormalised objective lets length into the reward **at
all**. Which direction it then pushes depends on your data.

The training script itself needs a GPU (`require_gpu()` stops with a clear
message and points here).

## Hardware Requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | 24 GB | Qwen3-0.6B + LoRA fits ~12 GB; 7B wants ~24 GB |
| GPUs | 1 | Offline PO does no generation — far cheaper than GRPO |
| Disk | 60 GB | Model + preference dataset |

**The reference model is the swing factor** — a second frozen copy, ~14 GB at
7B. Two ways to avoid it, best first:

1. **LoRA.** The reference is the base weights with the adapter disabled, so no
   second copy exists. Usually a better move than changing objectives to save
   memory.
2. **A reference-free method** (CPO, ORPO, SimPO).

## Running it

```bash
uv run train_dpo.py --list-methods      # the table; needs no GPU
```

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh
METHOD=simpo sbatch run_deepspeed.sh
METHODS="dpo ipo simpo orpo" sbatch run_deepspeed.sh    # sweep the family
sbatch run_deepspeed.sh --max-steps 20                  # cheap dry run
```

Build the venv on a **login** node — compute nodes usually have no egress.

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py recommend 05_huggingface_dpo
uv run runpod/runpod_ctl.py run 05_huggingface_dpo \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods       # confirm: "Nothing is billing."
```

### Direct

```bash
deepspeed --num_gpus=1 train_dpo.py --deepspeed ds_config.json --method dpo
```

## Two TRL details that are easy to get wrong

**SimPO is not a `DPOTrainer` loss type.** It is
`CPOConfig(loss_type="simpo", cpo_alpha=0.0)` plus `simpo_gamma`. Leaving
`cpo_alpha` at its default of 1.0 silently trains **CPO-SimPO**, a different
method. `train_dpo.py` forces it to 0.0 and says so.

**`CPOTrainer` moved to `trl.experimental.cpo`** in recent releases. The script
tries both import paths and raises a message naming the problem rather than an
`ImportError` traceback.

## What to watch

**`rewards/margins`, not `loss`.** The loss falls for every method here whether
or not it is learning the right thing — and the scales are not comparable across
methods anyway (IPO is a squared error; the rest are log-sigmoid). The margin
between chosen and rejected is what tells you the preference was absorbed.

## Next

- [`../06_huggingface_grpo/`](../06_huggingface_grpo/) — when you have a
  **verifier** instead of preference pairs. A verifier beats any judge.
- [`../06_huggingface_online_dpo/`](../06_huggingface_online_dpo/) — when offline
  DPO plateaus because the policy has drifted away from your fixed dataset.
- [`../05_huggingface_reward_model/`](../05_huggingface_reward_model/) — the
  component DPO deletes, and when you still want it.
