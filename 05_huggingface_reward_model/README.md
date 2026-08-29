# 05 — Reward Modeling (Bradley-Terry)

Stage 2 of the classical RLHF pipeline: **SFT → reward model → PPO**. The reward
model is the artefact that lets stage 3 score responses it has never seen.

**Book page:** [RLHF and Reward Modeling](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/huggingface/rlhf-reward-modeling)

## Why build one, when DPO deletes it?

[`../05_huggingface_dpo/`](../05_huggingface_dpo/) derives an implicit reward
from the policy itself and needs no separate model — which is why most alignment
work now skips this stage. Three reasons it is still sometimes right:

1. **A reward model is reusable.** It scores anything, including outputs from a
   model you have not trained yet. A DPO run leaves behind no such artefact.
2. **Best-of-$n$ sampling** needs a scorer at inference time. Only this route
   gives you one.
3. **Online methods need a judge.**
   [`../06_huggingface_online_dpo/`](../06_huggingface_online_dpo/) consumes
   exactly what this folder produces.

## The objective

Humans are unreliable at absolute scores and reliable at comparisons. So the
model is not trained to predict a rating — it is trained so the **difference**
between its scores reproduces observed preferences:

```
P(chosen ≻ rejected | x) = sigmoid( r(x, chosen) − r(x, rejected) )
L = −log sigmoid( r(x, chosen) − r(x, rejected) )
```

Logistic regression on score differences. That is the entirety of `RewardTrainer`.

## Two properties that bite people

Both are computed by `reward_modeling.py`, not asserted here.

### 1. Only differences are identified

```
  shift every score by     +0.0  ->  loss 0.561249256
  shift every score by    +10.0  ->  loss 0.561249197
  shift every score by   +100.0  ->  loss 0.561249435
  shift every score by  -1000.0  ->  loss 0.561252117
```

So **"our reward model scores 0.8" is a meaningless statement.** Two reward
models with wildly different ranges can be equally good, and you cannot compare
reward values across models or across runs.

> **Look closely: the digits drift.** The objective is *exactly* shift-invariant
> in real arithmetic; float32 is not. Computing `(x+1000) − (y+1000)` is
> catastrophic cancellation — the shared magnitude eats the mantissa bits holding
> the small difference. A reward model whose scores drift large is numerically
> losing the signal it trains on, which is why implementations often penalise the
> mean score to keep it near zero.

**Downstream consequence:** shift-invariance means the model is only anchored
where it saw data. Off distribution it is not merely inaccurate, it is
*arbitrary* — and an unconstrained RL optimiser will find where it is arbitrary
and go there. That is reward hacking, and it is why RLHF carries a KL leash.

### 2. Loss falls while accuracy stays flat

```
     gap        loss   accuracy
     0.5    0.474077     100.0%
     2.0    0.126928     100.0%
    10.0    0.000045     100.0%
```

Widening an already-correct gap keeps reducing the loss forever. **A beautiful
training curve is compatible with the ranking never improving.**

> **Watch pairwise accuracy, not loss.** Above ~0.75 on a held-out split is
> respectable — human annotators do not agree with each other much more often
> than that.

## Environment & Local Testing

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed transformers trl peft accelerate datasets
```

**No GPU?** The objective runs on CPU with no download:

```bash
uv run reward_modeling.py
uv run ../tests/test_reward_model.py     # 24 checks
```

The training script needs a GPU; `require_gpu()` stops with a clear message.

## Hardware Requirements

| Resource | Minimum | Notes |
|---|---|---|
| VRAM | 24 GB | Qwen3-0.6B + scalar head + LoRA fits ~12 GB |
| GPUs | 1 | A reward model is lighter than the policy it will score |
| Disk | 60 GB | Model + preference dataset |

**Each example is a pair**, so a micro-batch of $N$ does $2N$ forward passes.
Size it at roughly half what you would use for SFT.

## Running it

### CoreWeave / any SLURM cluster

```bash
sbatch run_deepspeed.sh
MODEL=Qwen/Qwen2.5-7B-Instruct sbatch run_deepspeed.sh
sbatch run_deepspeed.sh --max-steps 20        # cheap dry run
```

### RunPod (creates the pod and shuts it down)

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 05_huggingface_reward_model \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # confirm: "Nothing is billing."
```

### Direct

```bash
deepspeed --num_gpus=1 train_reward_model.py --deepspeed ds_config.json
```

## One detail that fails silently

The LoRA config must use **`task_type="SEQ_CLS"`**, not `CAUSAL_LM`. With
`CAUSAL_LM` the scalar head is never trained, the run completes normally, and you
get a model that scores noise. Nothing raises.

## Next

- [`../05_huggingface_dpo/`](../05_huggingface_dpo/) — skip the reward model
  entirely; usually the better trade.
- [`../06_huggingface_online_dpo/`](../06_huggingface_online_dpo/) — use this
  model as the judge.
- [`../06_huggingface_grpo/`](../06_huggingface_grpo/) — when a ground-truth
  verifier exists, it beats any learned reward model.
