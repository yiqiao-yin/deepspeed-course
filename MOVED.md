# Folder reorganisation — old paths to new

The example folders were renumbered so that **each top-level number appears
exactly once** and every example sits at `NN_section/NN_topic`. Before this, five
numbers were reused (`02`, `04`, `05`, `06`, `07`) and the root held seventeen
directories, which made the repository hard to read at a glance and impossible
to order.

The sections mirror the [book's tutorial sections](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/basic/neural-network):

| Section | Contains |
|---|---|
| `01_basics/` | MLP, CNN, CIFAR-10 CNN, RNN |
| `02_intermediate/` | Bayesian MCMC, time-series forecasting |
| `03_huggingface/` | LLM fine-tuning, TRL, OCR, and the full alignment thread |
| `04_video_text/` | video-to-text |
| `05_video_speech/` | video-speech-to-speech |

**Nothing was deleted and no history was lost** — every folder was moved with
`git mv`, so `git log --follow` and `git blame` still work through the rename.

## Mapping

| Old path | New path |
|---|---|
| `01_basic_neuralnet` | `01_basics/01_neuralnet` |
| `02_basic_convnet` | `01_basics/02_convnet` |
| `02_basic_convnet_cifar10_examples` | `01_basics/03_convnet_cifar10` |
| `03_basic_rnn` | `01_basics/04_rnn` |
| `04_bayesian_neuralnet` | `02_intermediate/01_bayesian_neuralnet` |
| `04_intermediate_rnn_stock_data` | `02_intermediate/02_rnn_stock_data` |
| `05_huggingface` | `03_huggingface/01_llm_finetuning` |
| `05_huggingface_dpo` | `03_huggingface/05_dpo` |
| `05_huggingface_ocr` | `03_huggingface/03_ocr` |
| `05_huggingface_reward_model` | `03_huggingface/04_reward_model` |
| `05_huggingface_trl` | `03_huggingface/02_trl_sft` |
| `06_huggingface_grpo` | `03_huggingface/06_grpo` |
| `06_huggingface_online_dpo` | `03_huggingface/07_online_dpo` |
| `07_huggingface_openai_gpt_oss_finetune_sft` | `03_huggingface/08_gpt_oss_lora` |
| `07_huggingface_trl_multi_agency` | `03_huggingface/09_multi_agency` |
| `08_vtt` | `04_video_text` |
| `08_vtt/01_qwen25vl_baseline` | `04_video_text/02_qwen25vl` |
| `08_vtt/02_token_compression` | `04_video_text/03_token_compression` |
| `08_vtt/03_streaming_memory` | `04_video_text/04_streaming_memory` |
| `08_vtt/04_video_eval` | `04_video_text/05_video_eval` |
| `08_vtt/hf_ds_vtt_test2` | `04_video_text/01_hf_baseline` |
| `08_vtt/test1/train_vtt.py` | `04_video_text/archive/test1_train_vtt.py` |
| `09_vss` | `05_video_speech` |
| `09_vss/01_longcat_flash_omni` | `05_video_speech/01_longcat_omni` |
| `09_vss/02_thinker_talker` | `05_video_speech/02_thinker_talker` |
| `09_vss/03_duplex_streaming` | `05_video_speech/03_duplex_streaming` |
| `09_vss/04_omni_eval` | `05_video_speech/04_omni_eval` |
| `09_vss/data` | `05_video_speech/data` |

## If you have a link that 404s

Find the old path above and use the new one. GitHub does not redirect renamed
paths, so old deep links into the repository will not resolve — that is the one
real cost of this change, and it is why this table exists.

## If you have a local clone

```bash
git pull --rebase
```

The `uv.lock` and `pyproject.toml` in each example are unaffected: they use
paths relative to their own folder, so `cd <new path> && uv sync` works exactly
as before.
