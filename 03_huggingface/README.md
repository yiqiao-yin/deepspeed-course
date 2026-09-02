# HuggingFace Integration

Real models, real downloads, multiple GPUs. Not runnable on a laptop — verify
changes with the logic tests in `tests/` instead.

Examples 04-07 are one escalating argument about **what you can delete from the
RLHF pipeline**, and the deletions are different: DPO removes the reward model,
GRPO removes the critic. Read them in order.

## Topics

| Folder | What it is |
|---|---|
| [`01_llm_finetuning/`](01_llm_finetuning/) | LLM fine-tuning with ZeRO — the starting point. |
| [`02_trl_sft/`](02_trl_sft/) | TRL supervised fine-tuning for function calling. |
| [`03_ocr/`](03_ocr/) | Vision-language OCR, plus a measured comparison of five modern OCR models. |
| [`04_reward_model/`](04_reward_model/) | Bradley-Terry reward modelling. This IS the pipeline. |
| [`05_dpo/`](05_dpo/) | DPO and five descendants — deletes the **reward model**. |
| [`06_grpo/`](06_grpo/) | GRPO on GSM8K — deletes the **critic**. |
| [`07_online_dpo/`](07_online_dpo/) | Online DPO, Nash-MD, XPO — re-adds sampling, needs a judge. |
| [`08_gpt_oss_lora/`](08_gpt_oss_lora/) | LoRA SFT of a 20B model. |
| [`09_multi_agency/`](09_multi_agency/) | Multi-agent GRPO. |

Each folder is self-contained and follows the same six-file contract (`CONTRIBUTING.md`):
a training script, a DeepSpeed config, a launcher, a README, a `pyproject.toml` and a
committed `uv.lock`. So:

```bash
cd 03_huggingface/01_llm_finetuning
uv sync
```

works from a fresh clone with no other setup.
