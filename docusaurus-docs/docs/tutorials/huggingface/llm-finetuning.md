---
sidebar_position: 2
---

# LLM Fine-tuning (SFT)

The most direct HuggingFace example in the course: supervised fine-tuning of a
Llama-3.2 model on a domain corpus with TRL's `SFTTrainer` and DeepSpeed ZeRO-1.

**Model:** `unsloth/Llama-3.2-3B-Instruct` · **Example:** `05_huggingface`

:::info Where this sits
This is the *plainest* configuration in the HuggingFace group — no LoRA, no RL,
no multimodal inputs. If you want to see the moving parts of a DeepSpeed +
HuggingFace run with nothing else layered on top, start here, then move to
[TRL Function Calling](/docs/tutorials/huggingface/trl-function-calling) for
loss masking and [GRPO](/docs/tutorials/huggingface/grpo-training) for RL.
:::

## 1. The Task

Domain adaptation. The base model is a general instruction-tuned Llama; the
dataset is question–answer pairs derived from **Warren Buffett's shareholder
letters, 1998–2024**:

```python
dataset = load_dataset(
    "eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024", split="train"
)
model_name = "unsloth/Llama-3.2-3B-Instruct"
```

The commented alternatives in the source show the intended axes of variation —
`openai/gsm8k` for math instead of finance, and `Llama-3.2-1B-Instruct` if 3B
does not fit:

```python
# dataset = load_dataset("openai/gsm8k", "main", split="train")
# model_name = "unsloth/Llama-3.2-1B-Instruct"
```

Swapping either is a one-line change, which makes this a good template.

## 2. Quick Start

```bash
cd 05_huggingface

export HF_TOKEN=...          # required — Llama is a gated model
sbatch run_deepspeed.sh      # SLURM / CoreWeave
deepspeed --num_gpus=2 train_ds.py   # direct / RunPod
```

:::warning `HF_TOKEN` is required, not optional
Llama models are gated on the Hub. Without a token with accepted licence terms,
`from_pretrained` fails at download. The script warns and continues with
`hf_token = None`, which then fails later and less clearly — so set it up front:

```bash
export HF_TOKEN=hf_...        # https://huggingface.co/settings/tokens
```

W&B remains genuinely optional; the script skips tracking when `WANDB_API_KEY`
is unset.
:::

## 3. Configuration

`ds_config.json`:

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": { "enabled": false },
  "zero_optimization": { "stage": 1 },
  "checkpoint": {
    "tag_validation_enabled": false,
    "partition_activations": false,
    "save_optimizer_states": false,
    "save_fp16_master_weights": false
  }
}
```

Three things are worth reading carefully, because they differ from the other
HuggingFace examples.

**ZeRO Stage 1, not 2.** Stage 1 partitions only optimizer states; Stage 2 adds
gradients at [no extra communication cost](/docs/getting-started/deepspeed-zero-stages#42-stages-1-and-2-are-communication-neutral).
Stage 2 is therefore normally the better default. Stage 1 remains defensible
when gradient accumulation is heavy — see the [stage comparison](/docs/getting-started/deepspeed-zero-stages#44-the-pareto-frontier--why-exactly-three-stages) —
but here `gradient_accumulation_steps` is 1, so **moving to Stage 2 is close to
free** and worth trying first if you are tight on memory.

**No `train_micro_batch_size_per_gpu`.** Only `train_batch_size` and
`gradient_accumulation_steps` are given, so DeepSpeed derives the micro-batch
from the world size: 32 ÷ (1 × N GPUs). At the launcher's 2 GPUs that is 16 per
GPU. Specifying two of the three fields and letting the third be derived is the
[portable pattern](/docs/reference/deepspeed-config#2-batch-size) — this config
works at any `--num_gpus` that divides 32.

**Precision is off entirely.** Both `fp16` disabled and no `bf16` block means
FP32. At 3B parameters that is $16\Psi = 48$ GB of model states, which is a lot
to leave on the table. On Ampere or newer, enable BF16:

```json
{ "bf16": { "enabled": true } }
```

This roughly halves weight and activation memory with no loss-scaling
machinery — see [why BF16](/docs/tutorials/huggingface/overview#bf16-over-fp16-for-llms).

:::note The `checkpoint` block trades recoverability for disk
`save_optimizer_states: false` and `save_fp16_master_weights: false` make
checkpoints much smaller, but a checkpoint without optimizer state **cannot
resume training correctly** — Adam's momentum and variance restart from zero,
which produces a visible loss spike. Fine for exporting a finished model;
wrong if you expect to resume after a pre-emption. On a time-limited SLURM
partition, you probably want optimizer states saved.
:::

## 4. Memory

At $\Psi = 3\times10^9$, using the [$16\Psi$ accounting](/docs/getting-started/deepspeed-zero-stages#12-where-the-memory-actually-goes):

| Configuration | Model states |
|---|---|
| Full FT, FP32/mixed, no ZeRO | **48 GB** |
| \+ ZeRO-1 across 2 GPUs | $4\Psi + 12\Psi/2 = 30$ GB per GPU |
| \+ ZeRO-2 across 2 GPUs | $2\Psi + 14\Psi/2 = 27$ GB per GPU |
| LoRA instead of full FT | ~6 GB frozen base + adapter |

So this example genuinely needs 2 capable GPUs, and it is the one HuggingFace
example here that does **full fine-tuning** rather than LoRA — which is exactly
why it is the clearest illustration of what ZeRO is for.

If you only have one GPU, the honest options are the 1B model, LoRA, or
[renting hardware](/docs/guides/runpod-setup#2a-provisioning-from-the-command-line):

```bash
uv run runpod/runpod_ctl.py recommend 05_huggingface
```

## 5. Pinned Dependencies

Unusually for this repository, `05_huggingface` ships a `requirements.txt` with
exact pins:

```
torch==2.1.0
transformers==4.51.3
accelerate==1.6.0
datasets==3.5.1
deepspeed==0.16.7
trl==0.17.0
unsloth
```

Install with `uv`:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Pinning is defensible here because TRL's API has moved repeatedly — `SFTTrainer`
argument names in particular differ between 0.1x releases. If you hit a
`TypeError` about an unexpected keyword, a version drift is the first thing to
check.

Note `torch==2.1.0` is old enough that you should confirm it matches your CUDA
toolkit before installing; see [Installation](/docs/getting-started/installation#2-prerequisites).

## 6. Suggested Changes

| Change | Why |
|---|---|
| ZeRO Stage 1 → 2 | Free memory at identical communication volume (§3) |
| Add `"bf16": {"enabled": true}` | Halves memory, no loss scaling, on Ampere+ |
| Save optimizer states | Required for correct resume after pre-emption |
| Add `"gradient_clipping": 1.0` | Cheap insurance against a single bad batch |
| Switch to LoRA | Turns a 2-GPU job into a 1-GPU job |
| Completion-only loss masking | The dataset is Q&A; training on the question wastes signal. See [loss masking](/docs/tutorials/huggingface/trl-function-calling#4-completion-only-loss-masking) |

That last one is the most consequential for output quality and the easiest to
overlook.

## 7. Troubleshooting

**`401` / gated repo on download.** `HF_TOKEN` unset, or licence terms not
accepted on the model page.

**OOM on the first forward pass.** Model-state bound at FP32 — enable BF16 and
move to ZeRO-2, or drop to the 1B model. See the
[OOM diagnosis flow](/docs/tutorials/basic/neural-network#92-diagnosis).

**Batch-size assertion.** `train_batch_size` (32) must be divisible by
`gradient_accumulation_steps × num_gpus`.

**`TypeError` from `SFTTrainer`.** TRL version drift — install the pinned
`requirements.txt` (§5).

**Download fails on a compute node.** Air-gapped cluster; pre-fetch on a login
node. See [CoreWeave Setup](/docs/guides/coreweave-setup#5-pre-fetching-for-air-gapped-nodes).

## Next Steps

- [TRL Function Calling](/docs/tutorials/huggingface/trl-function-calling) — chat templates and completion-only loss masking
- [GRPO Training](/docs/tutorials/huggingface/grpo-training) — when you can score outputs but not write them
- [ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) — the stage choice discussed in §3

## References

1. Grattafiori, A., et al. (2024). The Llama 3 Herd of Models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
2. von Werra, L., et al. (2020). TRL: Transformer Reinforcement Learning. [GitHub](https://github.com/huggingface/trl)
3. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO. *SC '20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — the SFT stage in context.
