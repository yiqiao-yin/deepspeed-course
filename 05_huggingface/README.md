# HuggingFace + DeepSpeed Fine-tuning

This guide walks through how to use **DeepSpeed** with **HuggingFace Transformers** to fine-tune large language models efficiently on multi-GPU setups.

## Environment & Local Testing

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 05_huggingface
uv sync                    # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=2 train_ds.py
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
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed
uv pip install transformers datasets accelerate
```

PyPI's `torch` ships CUDA wheels now, so no `--index-url` is needed;
pinning cu121 today gives an older CUDA than the default wheel.
</details>

### Running

| | |
|---|---|
| Runs end to end on one machine | **No** — needs real GPU capacity |
| GPUs requested by the launcher | 2 |
| Downloads | model weights (GBs) |

Requires a capable GPU and a model download.

```bash
cd 05_huggingface
deepspeed --num_gpus=2 train_ds.py
```

Because a full run is not feasible on a laptop, validate changes with the logic
tests below before submitting to a cluster.


### Doing less work: `--max-steps`

Every training script here accepts `--max-steps N`, which stops after `N`
optimizer steps instead of running the full schedule. `-1` (the default) means
"train normally".

From here on the cap is about money. This example downloads real weights and needs
real GPU capacity, so `--max-steps 5` on a rented card answers "is my config
valid?" for cents instead of for an hour of billing.

```bash
# directly
deepspeed --num_gpus=2 train_ds.py --max-steps 5

# through the launcher — it forwards its arguments, so this works on SLURM too
sbatch run_deepspeed.sh --max-steps 5
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

- Docker image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` (or similar)
- `uv` package manager installed
- At least 2 GPUs recommended (see [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md))
- HuggingFace account with API token (optional, for model download and upload)
- Weights & Biases account with API key (optional, for experiment tracking) 

## Project Starter

Use `uv` to start project. 

```bash
uv init project_name
````

If you do not have `uv`, please install it.

```bash
brew install uv
```

Or alternatively, you can use `pip`.

```bash
pip install uv
```

Next, add packages or dependencies

```bash
cd project_name
uv add torch transformers accelerate datasets deepspeed bitsandbytes trl unsloth wandb
```

Or add them individually:

```bash
uv add torch
uv add transformers
uv add accelerate
uv add datasets
uv add deepspeed
uv add bitsandbytes
uv add trl
uv add unsloth
uv add wandb  # Optional, for experiment tracking
```

We can examine the package dependency trees.

```bash
uv tree
```

You should expect something like the following.

```bash
root@1b0c67c74d6a:/workspace/deepspeed_project# uv tree
Resolved 97 packages in 0.68ms
deepspeed-project v0.1.0
├── accelerate v1.6.0
│   ├── huggingface-hub v0.31.1
│   │   ├── filelock v3.18.0
│   │   ├── fsspec v2025.3.0
│   │   │   └── aiohttp v3.11.18 (extra: http)
│   │   │       ├── aiohappyeyeballs v2.6.1
│   │   │       ├── aiosignal v1.3.2
│   │   │       │   └── frozenlist v1.6.0
│   │   │       ├── async-timeout v5.0.1
│   │   │       ├── attrs v25.3.0
│   │   │       ├── frozenlist v1.6.0
│   │   │       ├── multidict v6.4.3
│   │   │       │   └── typing-extensions v4.13.2
│   │   │       ├── propcache v0.3.1
│   │   │       └── yarl v1.20.0
│   │   │           ├── idna v3.10
│   │   │           ├── multidict v6.4.3 (*)
│   │   │           └── propcache v0.3.1
│   │   ├── hf-xet v1.1.0
│   │   ├── packaging v25.0
│   │   ├── pyyaml v6.0.2
│   │   ├── requests v2.32.3
│   │   │   ├── certifi v2025.4.26
│   │   │   ├── charset-normalizer v3.4.2
│   │   │   ├── idna v3.10
│   │   │   └── urllib3 v2.4.0
│   │   ├── tqdm v4.67.1
│   │   └── typing-extensions v4.13.2
│   ├── numpy v2.2.5
│   ├── packaging v25.0
│   ├── psutil v7.0.0
│   ├── pyyaml v6.0.2
│   ├── safetensors v0.5.3
│   └── torch v2.7.0
│       ├── filelock v3.18.0
│       ├── fsspec v2025.3.0 (*)
│       ├── jinja2 v3.1.6
│       │   └── markupsafe v3.0.2
│       ├── networkx v3.4.2
│       ├── nvidia-cublas-cu12 v12.6.4.1
│       ├── nvidia-cuda-cupti-cu12 v12.6.80
│       ├── nvidia-cuda-nvrtc-cu12 v12.6.77
│       ├── nvidia-cuda-runtime-cu12 v12.6.77
│       ├── nvidia-cudnn-cu12 v9.5.1.17
│       │   └── nvidia-cublas-cu12 v12.6.4.1
│       ├── nvidia-cufft-cu12 v11.3.0.4
│       │   └── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-cufile-cu12 v1.11.1.6
│       ├── nvidia-curand-cu12 v10.3.7.77
│       ├── nvidia-cusolver-cu12 v11.7.1.2
│       │   ├── nvidia-cublas-cu12 v12.6.4.1
│       │   ├── nvidia-cusparse-cu12 v12.5.4.2
│       │   │   └── nvidia-nvjitlink-cu12 v12.6.85
│       │   └── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-cusparse-cu12 v12.5.4.2 (*)
│       ├── nvidia-cusparselt-cu12 v0.6.3
│       ├── nvidia-nccl-cu12 v2.26.2
│       ├── nvidia-nvjitlink-cu12 v12.6.85
│       ├── nvidia-nvtx-cu12 v12.6.77
│       ├── sympy v1.14.0
│       │   └── mpmath v1.3.0
│       ├── triton v3.3.0
│       │   └── setuptools v80.3.1
│       └── typing-extensions v4.13.2
├── bitsandbytes v0.45.5
│   ├── numpy v2.2.5
│   └── torch v2.7.0 (*)
├── datasets v3.6.0
│   ├── dill v0.3.8
│   ├── filelock v3.18.0
│   ├── fsspec[http] v2025.3.0 (*)
│   ├── huggingface-hub v0.31.1 (*)
│   ├── multiprocess v0.70.16
│   │   └── dill v0.3.8
│   ├── numpy v2.2.5
│   ├── packaging v25.0
│   ├── pandas v2.2.3
│   │   ├── numpy v2.2.5
│   │   ├── python-dateutil v2.9.0.post0
│   │   │   └── six v1.17.0
│   │   ├── pytz v2025.2
│   │   └── tzdata v2025.2
│   ├── pyarrow v20.0.0
```

Afterwards, you should be able to expect the following folder structure:

```bash
project_name/
├── README.md
├── ds_config.json           # DeepSpeed configuration
├── train_ds.py              # Training script
├── pyproject.toml           # UV project configuration
├── uv.lock                  # UV lock file
└── results/                 # Training outputs
```

## DeepSpeed Configuration

The `ds_config.json` file controls DeepSpeed optimization settings. The most important parameter is the **ZeRO optimization stage**:

### ZeRO Optimization Stages

**Stage 1** - Optimizer State Partitioning:
- Partitions optimizer states across GPUs
- **Memory savings**: ~4x reduction
- **Recommended for**: Models that fit in GPU memory but optimizer states don't
- **Use case**: Smaller models (1B-7B parameters) on GPUs with limited memory

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

**Stage 2** - Optimizer + Gradient Partitioning:
- Partitions both optimizer states AND gradients across GPUs
- **Memory savings**: ~8x reduction
- **Recommended for**: Medium models (7B-13B parameters) or limited GPU memory
- **Use case**: Llama-3.2-3B on 2x RTX 4090 or similar

```json
{
  "zero_optimization": {
    "stage": 2
  }
}
```

**Stage 3** - Optimizer + Gradient + Parameter Partitioning:
- Partitions optimizer states, gradients, AND model parameters across GPUs
- **Memory savings**: Linear with number of GPUs
- **Recommended for**: Very large models (13B+ parameters)
- **Use case**: Large models that don't fit in single GPU memory
- **Note**: Slightly slower due to increased communication

```json
{
  "zero_optimization": {
    "stage": 3
  }
}
```

### Switching ZeRO Stages

To change the ZeRO stage, simply edit `ds_config.json`:

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 2  # Change this to 1, 2, or 3
  }
}
```

## Environment Setup

Before running the training script, set up your API tokens:

### Required: HuggingFace Token

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

### Optional: Weights & Biases API Key

For experiment tracking and visualization:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

Get your API key from: https://wandb.ai/authorize

If you don't set `WANDB_API_KEY`, the script will run without W&B tracking.

## Run Training

### Option 1: Using DeepSpeed Launcher (Recommended for 2+ GPUs)

For multi-GPU training with 2 GPUs:

```bash
uv run deepspeed --num_gpus=2 train_ds.py
```

For all available GPUs:

```bash
uv run deepspeed --num_gpus=$(nvidia-smi --list-gpus | wc -l) train_ds.py
```

### Option 2: Using Standard Python (Single GPU)

```bash
uv run python train_ds.py
```

### Option 3: Manual DeepSpeed Configuration

With custom DeepSpeed launcher arguments:

```bash
uv run deepspeed \
  --num_gpus=2 \
  --master_port=29500 \
  train_ds.py
```

## Monitoring Training

### Local Monitoring

Watch GPU utilization:

```bash
watch -n 1 nvidia-smi
```

### W&B Dashboard (if enabled)

After starting training with `WANDB_API_KEY` set, you'll see:

```
✅ Weights & Biases: Enabled
📈 W&B Run initialized: llama-3.2-3b-warren-buffett
   View at: https://wandb.ai/your-username/huggingface-deepspeed-finetuning/runs/xxxxx
```

Visit the URL to see real-time metrics, including:
- Training loss
- Learning rate
- GPU utilization
- System metrics

## Common Issues

### Out of Memory (OOM)

Try these in order:
1. Reduce `per_device_train_batch_size` in `train_ds.py`
2. Increase `gradient_accumulation_steps` in `ds_config.json`
3. Switch to higher ZeRO stage (1 → 2 → 3)
4. Enable FP16/BF16 mixed precision in `ds_config.json`

### NCCL Errors

If you see NCCL timeout errors:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

Then rerun the training command.

## Hardware Requirements

See [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) for detailed GPU requirements and recommendations for different models.

---

## Renting a GPU on RunPod (with auto-shutdown)

There is no SLURM on RunPod, so the pod lifecycle is driven by API instead —
including shutting it down.

```bash
export RUNPOD_API_KEY=...     # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py recommend 05_huggingface
uv run runpod/runpod_ctl.py run 05_huggingface \
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

This example is sized in `runpod/runpod_ctl.py` as **24 GB VRAM, 2 GPU(s),
80 GB disk**.

The pod is **never given `RUNPOD_API_KEY`** — putting a spending credential on
rented hardware would be the wrong trade, so termination is driven from your
machine. See [SECURITY.md](../SECURITY.md).
