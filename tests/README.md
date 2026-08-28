# Regression Tests

Guards against the specific bugs fixed in this repository.

Two tiers, answering different questions:

| | `tests/*.py` | [`tests/gpu/*.py`](gpu/README.md) |
|---|---|---|
| Requires | nothing but `uv` | GPU + model download |
| Runs in CI | **yes**, every push | no — manual |
| Answers | *Is the code structurally correct?* | *Does a real model accept what we build?* |
| Speed | seconds | minutes |

Everything below describes the **CPU tier**. It is what protects the repository
day to day. See [`gpu/README.md`](gpu/README.md) for the GPU tier and the bug
that motivated splitting them.

## Running

```bash
./tests/run_all.sh              # everything
uv run tests/test_ds_configs.py # one suite
```

Only `uv` is needed. Each test carries [PEP 723](https://peps.python.org/pep-0723/)
inline metadata declaring its own dependencies, so `uv run` provisions an
environment automatically — nothing to install first.

## What is covered

| Suite | Guards against |
|---|---|
| `test_ds_configs.py` | Config errors that abort at startup or silently do the wrong thing, across **every** `ds_config.json` in the repo |
| `test_stock_leakage.py` | Look-ahead bias from fitting the scaler before the train/test split |
| `test_grpo_rewards.py` | A PPO value head under GRPO; surface-form and misaligned rewards |
| `test_video_frames.py` | Frame "extraction" that returns one image repeated; and `preprocess_function` failing to unwrap the processor's batch dimension |
| `test_runpod_ctl.py` | Example table drift, a non-`devel` image, a bootstrap that leaks credentials, and shell scripts that do not parse |
| `test_token_compression.py` | Compression that runs, reduces tokens, and is silently **wrong** — merging the wrong pair, unweighted averaging, missing the log-size attention correction |
| `test_star_memory.py` | A streaming buffer that leaks (grows with stream length) **or** one that is bounded because it discards everything |
| `test_video_eval.py` | An eval harness that leaks answers via correlated RNG seeds, and a parser that scores "option A is wrong, the answer is C" as A |

### `test_ds_configs.py`

Static validation of all 14 DeepSpeed configs:

- valid JSON
- the batch invariant `train_batch_size = micro × accum × num_gpus` holds for the
  GPU count the neighbouring launcher actually requests
- `fp16` and `bf16` are not both enabled, and there is no *latent* conflict where
  one is hard-enabled while the other is `"auto"`
- `"auto"` only appears where a HuggingFace `Trainer` exists to resolve it
- `offload_param` is only used with ZeRO stage 3
- stage-3 configs save a consolidated (loadable) checkpoint
- NVMe `nvme_path` does not point at a network filesystem

This suite found a bug that manual review had missed:
`09_vss/01_longcat_flash_omni/ds_config.json`
combined a hard-enabled `bf16` with `fp16: "auto"`.

### Design note

The training scripts import `torch`, `deepspeed`, `trl` and `transformers` at
module scope, so importing them to unit-test one pure function would drag in the
whole stack. Instead `_srcload.py` parses the file with `ast` and execs only the
function under test.

This means the tests run against the **actual shipped source**, not a copy that
can drift out of sync with it — while needing nothing heavier than `numpy`.

`_srcload.source_code_contains()` strips comments and string literals before
searching, because several files deliberately *describe* the bugs they used to
have; a naive text search would match that prose and report a fixed bug as still
present.

## Adding a test

```python
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]      # omit if stdlib-only
# ///
"""What this guards against, and why the bug mattered."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, load_function, source_code_contains

def main() -> int:
    r = Results("Short description")
    r.check(condition, "what is being asserted", "detail shown on failure")
    return r.finish()

if __name__ == "__main__":
    sys.exit(main())
```

Then add it to the `TESTS` array in `run_all.sh`.
