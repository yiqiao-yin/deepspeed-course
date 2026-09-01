# Vision-Language Model Fine-tuning with DeepSpeed

Minimal Vision-Language Model (VLM) fine-tuning script using DeepSpeed for distributed training on 2 RTX 4000-series NVIDIA GPUs. This example uses the Qwen2-VL-2B-Instruct model for OCR and vision-language tasks.

## Which OCR model should you actually use?

`train_ds.py` fine-tunes Qwen2-VL-2B. That teaches the training mechanics and
says nothing about *which* model to fine-tune, and the field moved: purpose-built
OCR models now compete with general VLMs several times their size, and they
differ by more than an order of magnitude in what a page costs them.

`run_modern_ocr.py` measures five of them on the same pages:

```bash
uv sync
uv run run_modern_ocr.py --list-models            # no GPU needed
python run_modern_ocr.py --models all --max-samples 16
```

### Measured on hardware

2x RTX 3090 (RunPod), torch 2.8.0+cu128, transformers 5.16, **12 rendered
pages**, greedy decoding, `max_new_tokens=256`:

| Model | Params | CER (pooled) | CER (median) | Tokens/page | Acc per 100 tok |
|---|---|---:|---:|---:|---:|
| `qwen2-vl-2b` | 2.2B | **0.0000** | 0.0000 | 164 | 0.610 |
| `qwen2.5-vl-3b` | 3.8B | **0.0000** | 0.0000 | 164 | 0.610 |
| `got-ocr2` | 580M | 0.1530 | 0.0104 | 286 | 0.296 |
| `florence-2-base` | 230M | — | — | — | blocked (see below) |
| `deepseek-ocr` | 3B MoE | — | — | — | blocked (see below) |

`qwen2-vl-2b` and `qwen2.5-vl-3b` read **12/12 pages exactly**. `got-ocr2` read
**6/12 exactly**, with a per-page CER range of **0.0000 to 1.7639** — one page
ran away and generated far more text than the reference. That single page is
what drags its pooled score to 0.1530 while its median stays at 0.0104, and it
is precisely why this metric is not clipped at 1.0.

**These pages are cleanly rendered text, not photographs.** Error rates here
are a *floor*, not a document-benchmark score — real scans bring skew, noise
and JPEG artefacts that none of this measures. Use `--source hf` for a real
corpus. A 0.0000 means "perfect on easy input", not "solved".

Note `got-ocr2`'s pooled 0.1530 against a median of 0.0104. That gap is the
signature of a **few pages failing badly** while most are near-perfect, not of
a uniformly weak model — which is exactly why the script reports both, and why
[pooled and averaged CER are not interchangeable](#the-metric-is-not-obvious).

### Two of the five could not run, and it is the same cause

Both are `trust_remote_code` models whose published code targets an **older
transformers** than the one Qwen2.5-VL requires:

| Model | Error |
|---|---|
| `deepseek-ocr` | `ImportError: cannot import name 'LlamaFlashAttention2' from transformers.models.llama.modeling_llama` |
| `florence-2-base` | `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'` |

This is worth knowing before you plan around either: running them means a
**separate pinned environment**, and pinning transformers back far enough for
them breaks Qwen2.5-VL. That is a real constraint on building one OCR pipeline
across these models, not a defect in this folder — and it is why the lock file
in this directory matters.

**What was tried, so you do not repeat it.** `tests/gpu/verify_05_ocr_models.sh`
builds a second venv with `--system-site-packages` (reusing torch) and
`transformers==4.47.1`, the last release that still contains
`LlamaFlashAttention2` — verified against the tagged sources, not from memory:

| transformers | `LlamaFlashAttention2` present |
|---|---|
| 4.46.3, 4.47.1 | yes |
| 4.49.0, 5.16.1 | no |

On that pinned environment both models load further and then stop again:
DeepSeek-OCR's `infer()` writes its transcription to disk and returns `None`,
and Florence-2 still raises the missing-config-attribute error. **Neither has a
verified accuracy number here**, and rather than publish one, this folder says
so. If you get either running, the harness is set up to score it.

> One number that nearly shipped: an earlier version of the harness took
> DeepSeek-OCR's `None` return, `str()`'d it, and scored the literal string
> `"None"` against every reference — producing a pooled CER of **0.9594** with
> a median of 0.9600 and a min/max of 0.9231/0.9710. That has the shape of a
> real measurement of a weak model. It was caught only because 0.9594 happened
> to match an unrelated sanity run exactly. The script now refuses to score any
> model whose pages all come back empty.

Getting there took four GPU runs, each surfacing the next layer:
`AutoProcessor` cannot instantiate DeepSeek-OCR at all → it needs a custom
`infer()` → which imports `addict`, `matplotlib`, then `easydict` → which then
hits the transformers version wall. Each package was discovered only *after*
several GB of weights had downloaded.

### The metric is not obvious

`ocr_metrics.py` runs on CPU and is where the scoring lives, because this is
where OCR benchmarks go quietly wrong:

```bash
uv run ocr_metrics.py                 # a demo of every trap below
uv run ../tests/test_ocr_metrics.py   # 40 property checks
```

- **CER divides by the reference**, so emitting half the page scores ~0.5. A
  prediction-length denominator lets a model improve by saying less.
- **It is not clipped at 1.0**, so runaway generation stays visible — the most
  common VLM failure on a page it cannot parse.
- **An empty reference with invented text scores 1.0**, not 0.0 and not a
  `ZeroDivisionError`.
- **Pooled ≠ averaged.** On a corpus of one 1000-character page read perfectly
  and one 2-character page read wrong, pooled CER is 0.0020 and averaged is
  0.5000 — **250x apart on identical predictions**. A benchmark that does not
  say which it used is not comparable to anything.

### Accuracy alone is the wrong axis

DeepSeek-OCR ([arXiv:2510.18234](https://arxiv.org/abs/2510.18234)) argues the
point directly: a page compressed into ~100 vision tokens decodes at ~97%
precision, falling to ~60% at 20x compression. That is the same bargain as
[`08_vtt/02_token_compression`](../08_vtt/02_token_compression/) — shrink what
the model looks at, pay in accuracy — so the table reports tokens/page beside
CER. Ranking on accuracy alone recommends a model that is half a point better
and sixty times more expensive.

### Reproduce it

```bash
bash ../tests/gpu/verify_05_ocr_models.sh 1 16     # 1 GPU, 16 pages
```

## Environment & Local Testing

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 05_huggingface_ocr
uv sync                    # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=1 train_ds.py
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
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed
uv pip install transformers datasets accelerate peft bitsandbytes qwen-vl-utils
```

The `--index-url` is **required** and matches what `uv.lock` pins.
PyPI's *default* `torch` is a CUDA 13 wheel: on a driver older than
CUDA 13 — the 550/570 series, common on rented hardware — it installs
cleanly and then reports `cuda.is_available() == False` while
`nvidia-smi` shows the card. Verified on a driver 550.127 box.
</details>

### Running

| | |
|---|---|
| Runs end to end on one machine | **No** — needs real GPU capacity |
| GPUs requested by the launcher | 2 |
| Downloads | Qwen2-VL-2B (~4 GB) |

Vision-language memory is driven by sequence length, not parameters. Cap it with the processor's `min_pixels` / `max_pixels`.

```bash
cd 05_huggingface_ocr
deepspeed --num_gpus=2 train_ds.py
```

Because a full run is not feasible on a laptop, validate changes with the logic
tests below before submitting to a cluster.


### Doing less work: `--max-steps`

Every training script here accepts `--max-steps N`, which stops after `N`
optimizer steps instead of running the full schedule. `-1` (the default) means
"train normally".

Vision-language collators are the classic silent failure: pixels get dropped and
training proceeds happily on text alone. A few steps is enough for the collator's
own assertion to fire if `pixel_values` never arrived.

```bash
# directly
uv run deepspeed --num_gpus=2 train_ds.py --max-steps 5

# through the launcher — it forwards its arguments, so this works on SLURM too
sbatch submit_job.sh --max-steps 5
```

Two things worth knowing. The flag caps **optimizer steps, not epochs**, so with
gradient accumulation of 4 a `--max-steps 5` run consumes 20 micro-batches. And
the launcher only sees the flag because its last line ends in `"$@"` — drop that
and the argument is silently swallowed, the script runs to completion, and
nothing warns you.

This is also what `runpod_ctl.py run <example> --dry-run` relies on to keep a
rented pod's bill small.

### Verifying logic without a full run

The repository ships regression tests that check the **logic** of these examples —
config validity, data handling, reward correctness — with no GPU and no model
download required:

```bash
../tests/run_all.sh
```

See [`tests/README.md`](../tests/README.md) for what each suite covers.

## Prerequisites

- 2x RTX 4000-series NVIDIA GPUs
- CUDA 11.8 or higher
- Python 3.8+
- `uv` package manager

## Getting Started

### 1. Install uv

If you haven't already installed `uv`, install it first:

```bash
pip install uv
```

### 2. Initialize Project

Create a new project directory and initialize it with `uv`:

```bash
uv init proj
cd proj
```

### 3. Install Dependencies

Install all required packages using `uv add`:

```bash
# Install PyTorch with CUDA support (CUDA 11.8)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face and training libraries
uv add transformers datasets accelerate peft

# Install DeepSpeed and optimization libraries
uv add deepspeed bitsandbytes

# Install image processing
uv add pillow

# Optional: Install Weights & Biases for experiment tracking
uv add wandb
```

### 4. Copy Training Script

Copy the `train_ds.py` file to your project directory:

```bash
cp ../train_ds.py .
```

### 5. Run Training

Run the training script with DeepSpeed using 2 GPUs:

```bash
uv run deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora
```

#### Training Options

The script supports various configuration options:

**Model Configuration:**
- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2-VL-2B-Instruct`)
- `--use-4bit`: Enable 4-bit quantization for reduced memory usage
- `--use-lora`: Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha parameter (default: 16)
- `--lora-dropout`: LoRA dropout rate (default: 0.05)

**Training Configuration:**
- `--batch-size`: Batch size per device (default: 1)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--num-epochs`: Number of training epochs (default: 1)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--output-dir`: Output directory for checkpoints (default: `outputs`)

**System Configuration:**
- `--no-deepspeed`: Disable DeepSpeed (run on single GPU)

#### Example Commands

```bash
# Basic training with default settings
uv run deepspeed --num_gpus=2 train_ds.py

# Training with 4-bit quantization and LoRA
uv run deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora

# Single GPU training (without DeepSpeed)
uv run python train_ds.py --no-deepspeed --use-4bit --use-lora

# Custom configuration
uv run deepspeed --num_gpus=2 train_ds.py \
  --use-4bit \
  --use-lora \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --output-dir ./custom_output
```

## Features

- **DeepSpeed ZeRO Stage 2**: Efficient distributed training with gradient and optimizer state partitioning
- **4-bit Quantization**: Reduced memory footprint using bitsandbytes
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **Gradient Checkpointing**: Memory optimization for large models
- **FP16 Mixed Precision**: Faster training with reduced memory usage
- **Synthetic Dataset**: Built-in sample data generation for testing

## Output

The training script will:
1. Generate a DeepSpeed configuration file (`ds_config.json`)
2. Download and load the specified model
3. Create synthetic training data (or use your custom dataset)
4. Train the model across 2 GPUs
5. Save checkpoints to the output directory

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` to 1
- Increase `--gradient-accumulation-steps`
- Enable `--use-4bit` quantization
- Enable `--use-lora` for parameter-efficient training

### DeepSpeed Initialization Errors
- Ensure CUDA toolkit is properly installed
- Verify both GPUs are accessible: `nvidia-smi`
- Check DeepSpeed installation: `ds_report`

### Model Download Issues
- Ensure stable internet connection
- HuggingFace models may require authentication for some models
- Set `HF_TOKEN` environment variable if needed

## Next Steps

- Replace synthetic dataset with your own OCR/VLM dataset
- Adjust hyperparameters based on your dataset size
- Enable W&B tracking for experiment monitoring
- Fine-tune on domain-specific vision-language tasks

## See Also

- [HARDWARE_REQUIREMENTS.md](./HARDWARE_REQUIREMENTS.md) - GPU requirements and recommendations
- [submit_job.sh](./submit_job.sh) - SLURM job submission script for CoreWeave

---

## Renting a GPU on RunPod (with auto-shutdown)

There is no SLURM on RunPod, so the pod lifecycle is driven by API instead —
including shutting it down.

```bash
export RUNPOD_API_KEY=...     # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py recommend 05_huggingface_ocr
uv run runpod/runpod_ctl.py run 05_huggingface_ocr \
    --dry-run --collect --wait --terminate --yes

uv run runpod/runpod_ctl.py pods      # must say: "Nothing is billing."
```

| Flag | Effect |
|---|---|
| `--dry-run` | Caps the training step at 300s. The pod still clones, installs and launches the **real** script, so a genuine failure still surfaces — you just do not pay for a full run. |
| `--collect` | The pod pushes its log to a private-ish ntfy topic. **No SSH needed** — RunPod exposes no log endpoint, so the pod pushes. |
| `--wait` | Blocks locally until the pod reports DONE. |
| `--terminate` | Deletes the pod in a `finally` block, so a crash, a network failure or Ctrl-C **still** stops the billing. Retries five times with backoff. |
| `--yes` | Skips the confirmation. `run` and `create` both refuse without it and print the hourly rate first. |

> ### 💸 An abandoned pod bills until terminated
> *Stopping* is not enough. Always finish with `runpod_ctl.py pods` and confirm
> it says **"Nothing is billing."**
>
> Two safety nets you get for free: an **in-pod watchdog** (`--max-hours`,
> default 6) that kills the container from the inside and needs no API
> key, and `terminate --all` as the blunt instrument.

This example is sized in `runpod/runpod_ctl.py` as **24 GB VRAM, 1 GPU(s),
60 GB disk**.

The pod is **never given `RUNPOD_API_KEY`** — putting a spending credential on
rented hardware would be the wrong trade, so termination is driven from your
machine. See [SECURITY.md](../SECURITY.md).
