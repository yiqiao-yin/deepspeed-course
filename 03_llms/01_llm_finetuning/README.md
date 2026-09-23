# HuggingFace + DeepSpeed Fine-tuning

This guide walks through how to use **DeepSpeed** with **HuggingFace Transformers** to fine-tune large language models efficiently on multi-GPU setups.

The folder holds **four** entry points: `train_ds.py` (Llama SFT, the original),
and three frontier-model analyses — `train_glm53_ds.py` (755 GB sparse MoE),
`train_qwen38_ds.py` (hybrid linear/full attention) and `analyze_kimi_k3.py`
(2.78 T parameters — analysis only, hence the name). The last three all start with `--plan`,
which needs no GPU and downloads nothing.

## Environment & Local Testing

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a
`pyproject.toml` and a committed `uv.lock`, so after cloning:

```bash
cd 03_llms/01_llm_finetuning
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
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed
uv pip install transformers datasets accelerate
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
| Downloads | model weights (GBs) |

Requires a capable GPU and a model download.

```bash
cd 03_llms/01_llm_finetuning
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
../../tests/run_all.sh
```

See [`tests/README.md`](../../tests/README.md) for what each suite covers.

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

uv run runpod/runpod_ctl.py recommend 03_llms/01_llm_finetuning
uv run runpod/runpod_ctl.py run 03_llms/01_llm_finetuning \
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
machine. See [SECURITY.md](../../SECURITY.md).

---

# GLM-5.3: fine-tuning a 755 GB sparse MoE

`train_glm53_ds.py` is a second, self-contained entry point in this folder.
One script, four stages — download data, download model, LoRA fine-tune,
generate — pointed by default at
[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), released
2026-08-31.

It is here because GLM-5.3 breaks the assumptions the Llama example above
rests on, and the ways it breaks them are instructive.

## Start here: the analysis needs no GPU and downloads nothing

```bash
uv run train_glm53_ds.py --plan
uv run train_glm53_ds.py --plan --model zai-org/glm-edge-1.5b-chat
uv run train_glm53_ds.py --plan --num-gpus 8 --vram-gb 80    # try to fit it
```

`--plan` reads the published `config.json` and works out where the parameters
are, which modules LoRA should adapt, what the KV cache costs, and whether your
hardware can hold the model — before anything is downloaded.

## What GLM-5.3 is

Every number below was read from the model's published `config.json` and
`model.safetensors.index.json`. That index lists all **118,629 tensor names**
and the total byte count without downloading any weights, which is how the LoRA
targets here were *verified* rather than guessed.

| | |
|---|---|
| architecture | `GlmMoeDsaForCausalLM` (`model_type: glm_moe_dsa`) |
| parameters | ~743 B, of which 8 of 256 experts fire per token |
| weights | **755.7 GB** fp8 / 1,506.7 GB bf16 |
| layers | 78 (+1 multi-token-prediction layer) |
| attention | MLA — compressed q/kv (`q_lora_rank` 2048, `kv_lora_rank` 512) |
| sparse attention | DSA indexer, on **22 of 78 layers only** |
| context | 1,048,576 tokens |
| requires | `transformers >= 5.15` |

### Almost all of it is experts

```
routed experts     724.78 B   97.5%
shared experts       2.83 B    0.4%
dense MLP layers     0.68 B    0.1%
attention           12.87 B    1.7%
router (gate)        0.12 B    0.0%
embeddings           1.90 B    0.3%
TOTAL              743.18 B
```

That computed total cross-checks against the measured 755.7 GB of fp8 bytes —
the gap is the `weight_scale_inv` block-scale tensors — which is what makes the
arithmetic trustworthy rather than merely plausible.

### MLA does not save parameters. It saves the cache

This is the one most people get backwards, including the first version of this
example's own test:

```
KV cache/token       87.8 KB  (MLA)
vanilla would be   4992.0 KB  -> 57x larger
at 1,048,576 tokens:  94 GB   vs  5,360 GB
```

On these dimensions — 64 heads × 256 head_dim is 2.7× the hidden size — MLA
attention is slightly **larger** in parameters than vanilla attention would be
(12.9 B vs 11.8 B). What it compresses is the KV cache, by 57×, and that is the
only reason a 1M-token context is physically possible.

## Why LoRA targets attention and not the experts

`lora_target_modules()` returns `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
`kv_b_proj`, `o_proj` — names verified against the safetensors index. The 256
expert MLPs, 97% of the model, are left **frozen**, and so is the router:

- An adapter on expert *k* only receives gradient when the router sends a token
  to expert *k*. At top-8 of 256 that is about **3% of tokens**, so 256 adapters
  would each train on a sliver of the data and most would stay near their
  initialisation. You would add hundreds of thousands of matrices to fine-tune
  badly.
- Attention is shared by every token on every layer, so one adapter there sees
  the whole dataset — for 2% of the parameters.
- The **router stays frozen** deliberately. Training it changes *which* experts
  fire, which is a far more destructive edit than changing how they are read;
  routing collapse is the classic way a fine-tuned MoE quietly degrades.

Copying `q_proj`/`k_proj`/`v_proj` from a Llama recipe would match **nothing**
here — and depending on the peft version that either raises or silently trains
an adapter attached to nothing while the loss still goes down, because the base
model is already good.

## Hardware: this does not fit, and the script says so first

LoRA freezes the base weights, which removes optimizer state. It does **not**
reduce what it costs to *hold* the model — all 755 GB must still be resident.

| configuration | total VRAM | holds fp8 GLM-5.3? |
|---|---|---|
| 1 × A100 80GB | 80 GB | no — 11.3× short |
| 8 × A100 80GB | 640 GB | no |
| 8 × H100 80GB | 640 GB | no |
| 8 × H200 141GB | 1,128 GB | **yes**, with room for LoRA + activations |
| 8 × B200 180GB | 1,440 GB | yes, comfortably |

The script computes this and **refuses before downloading**, because the
alternative is discovering it after 755 GB:

```
    weights on disk   755.7 GB
    needed (x1.2)     906.8 GB   (LoRA frees optimizer state, NOT the base weights)
    you have          8 x 80 GB = 640 GB
    verdict           DOES NOT FIT — short by 1.4x
    would need        ~12 x 80 GB
```

Pass `--force` to override it and OOM anyway.

## Running it

### No GPU

```bash
uv run train_glm53_ds.py --plan          # the whole analysis
uv run train_glm53_ds.py --verify-arch   # build it on the meta device, check the LoRA targets
uv run ../../tests/test_glm53_arch.py    # 26 property assertions
```

`--verify-arch` is the one to run before renting anything: it constructs the
full 743 B module tree on the meta device — no memory, no weight download —
and confirms your transformers version implements `glm_moe_dsa` and that every
LoRA target resolves. All three of those failures otherwise surface only
*after* a 755 GB download.

It also exposes why the experts must stay frozen: transformers fuses the 256
experts into 3D tensors (`mlp.experts.gate_up_proj` is `(256, 4096, 6144)`),
so they are not `nn.Linear` modules and stock peft has nothing to attach LoRA
to — regardless of whether you think adapting them is a good idea.

### CoreWeave / any SLURM cluster

```bash
sbatch run_glm53.sh --max-steps 20                        # cheap dry run
sbatch run_glm53.sh                                       # the real thing, 8 GPUs
NUM_GPUS=1 sbatch run_glm53.sh --model zai-org/glm-edge-1.5b-chat
```

`run_glm53.sh` requests `--gres=gpu:8` and 256 GB of host RAM, and forwards
`"$@"` so those extra arguments reach the script.

### RunPod

`runpod_ctl.py` sizes this folder for the Llama example above. GLM-5.3 itself
needs an 8×H200-class node, which RunPod does not reliably offer — so rent a
single card and run the proxy model:

```bash
uv run runpod/runpod_ctl.py run 03_llms/01_llm_finetuning \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # must say "Nothing is billing."
```

## What has and has not been verified

**This script has never been run against GLM-5.3 itself.** Nothing is stubbed —
the same code path runs both models, only the weights differ — but honesty about
which parts are proven matters more than a tidy claim:

| | Status |
|---|---|
| `--plan` architecture + capacity analysis | **verified**, cross-checked against measured file sizes |
| LoRA target module names for GLM-5.3 | **verified** against the published safetensors index |
| All four stages end to end | **verified on a rented RTX 3090** with `zai-org/glm-edge-1.5b-chat` — see below |
| The same four stages on GLM-5.3 | **not verified** — needs ~8×H200, unavailable to test |

### The verified run

`deepspeed --num_gpus=1 train_glm53_ds.py --model zai-org/glm-edge-1.5b-chat
--max-steps 8 --max-samples 64` on a rented RTX 3090, via
`runpod_ctl.py --collect --wait --terminate`. Trimmed, and **measured**:

```
  [1/4] dataset: tatsu-lab/alpaca
  64 examples
  sample: '### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n1.Eat a balanced diet...'

  [2/4] model: zai-org/glm-edge-1.5b-chat  (3.2 GB)
  LoRA targets: q_proj, k_proj, v_proj, o_proj

  [3/4] fine-tuning
  {'loss': '2.461', 'grad_norm': '1.349', 'mean_token_accuracy': '0.5519', 'epoch': '0.125'}
  {'loss': '1.946', 'grad_norm': '1.061', 'mean_token_accuracy': '0.5746', 'epoch': '1'}
  {'train_runtime': '61.39', 'train_loss': '2.296', 'epoch': '1'}
  adapter written to ./glm53-lora-out

  [4/4] inference
  prompt:   Explain what a mixture-of-experts layer does, in two sentences.
  response: 'A mixture-of-experts layer is a type of layer in a neural network that
             combines the predictions from multiple different neural networks...'
```

Note the LoRA targets in that run: `q_proj, k_proj, v_proj, o_proj`, because
glm-edge is a **dense** GLM with vanilla attention. The same function returns
`q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj` for GLM-5.3's MLA.
That branch is what makes this a proxy for the real thing rather than a
different program — and it is asserted in `tests/test_glm53_arch.py`.

Eight steps on 64 examples is a **pipeline test, not a result**. The loss moves
and the model generates; nothing about quality is claimed.

---

# Qwen3.8-27B: a hybrid linear/full-attention model

`train_qwen38_ds.py` is the third entry point in this folder, and the
counterpart to the GLM-5.3 one above. Same four stages — download data,
download model, LoRA fine-tune, generate — pointed at
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

Unlike GLM-5.3, **this one you can actually rent hardware for**: 55.6 GB of
weights on 2 × 80 GB (or 4 × 48 GB).

## Start here, with no GPU

```bash
uv run train_qwen38_ds.py --plan          # hybrid-layer, cache and capacity analysis
uv run train_qwen38_ds.py --verify-arch   # build the real module tree, no weights
```

## The shape of the model

| | |
|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| parameters | 27.36 B total, **26.90 B** without the vision tower |
| weights | 55.6 GB bf16 |
| layers | 64 — **48 linear attention, 16 full attention** |
| pattern | full attention every 4th layer |
| full attention | GQA, 24 q-heads / 4 kv-heads, head_dim 256 |
| linear attention | gated-delta, causal conv kernel 4, SSM state in float32 |
| context | 262,144 tokens |
| transformers | config saved with 5.8.0.dev0; **verified on 5.16.1** (this folder's lock) |

Three quarters of the layers keep **no KV cache at all** — they carry a fixed
recurrent state instead:

| | per token | at 262,144 tokens |
|---|---|---|
| hybrid (16 full layers) | **64 KB** | **17.2 GB** |
| if all 64 were full attention | 256 KB | 68.7 GB |

and the 48 linear layers hold **159 MB per sequence**, the same for one token
as for the full context. That constant is the entire argument for the design.

## The mistake this model invites

`q_proj`/`k_proj`/`v_proj`/`o_proj` — the list from every Llama recipe — **do
exist here**, on the 16 full-attention layers only. The other 48 use
`linear_attn.in_proj_{qkv,z,b,a}` and `linear_attn.out_proj`.

So the Llama default does not error, does not warn, attaches adapters to 25% of
the depth, and shows a healthy falling loss. `--plan` reports coverage as a
number so you can see it:

```
$ uv run train_qwen38_ds.py --plan --lora-scope attention-full
    layer coverage    16/64 (25% of depth)
    WARNING: 48 layers get NO adapter.

$ uv run train_qwen38_ds.py --plan
    layer coverage    64/64 (100% of depth)
```

`--verify-arch` counts them in the real module tree: `q_proj` **16**,
`in_proj_qkv` **48**. The trainer also asserts at runtime that the adapter
attached to something, because peft will happily give you zero trainable
parameters and train anyway.

## Hardware

55.6 GB of bf16 weights do not fit one 48 GB card, and LoRA does not change
that — it removes optimizer state, not the base weights.

The weights shard under ZeRO-3; activations, gather buffers and fragmentation
do **not** — every rank pays ~20 GB of those, even with gradient checkpointing.

| configuration | weight shard | + overhead | per GPU | verdict |
|---|---|---|---|---|
| 1 × 48 GB | 55.6 | 20.5 | 76.1 | no |
| 2 × 24 GB | 27.8 | 20.5 | 48.3 | no |
| 2 × 48 GB | 27.8 | 20.5 | **48.3** | **no — measured OOM on 2×L40S** |
| **2 × 80 GB** | 27.8 | 20.5 | 48.3 | **yes** |
| 4 × 48 GB | 13.9 | 20.5 | 34.4 | yes |

2 × 48 GB passes an aggregate check (96 GB vs 55.6 GB) and then OOMs at the
first step by about 1%. A "48 GB" L40S reports 44.39 GiB usable.

Hence ZeRO **stage 3**: the parameters themselves are sharded, ~28 GB per rank.

## Running it

```bash
# CoreWeave
sbatch run_qwen38.sh --max-steps 20        # cheap dry run
sbatch run_qwen38.sh

# RunPod — needs 2 x 48 GB, so check what is available first
uv run runpod/runpod_ctl.py gpus --min-vram 46
uv run runpod/runpod_ctl.py run 03_llms/01_llm_finetuning \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods           # must say "Nothing is billing."
```

> The `EXAMPLES` entry for this folder sizes the pod for `train_ds.py`
> (2 × 24 GB). For Qwen3.8 you need 2 × 48 GB — pick the GPU explicitly with
> `--gpu`, and note that 2-GPU capacity for any given card comes and goes.

> **The 30-minute `--wait-seconds` default is not enough for this model.**
> `runpod_ctl.py` stops waiting after 1800s and, with `--terminate`, then kills
> the pod — while a 55.6 GB download may still be in progress. It does say
> *"no DONE marker — the run may still be going"*, but the pod is gone by then.
> Pass a realistic window:
>
> ```bash
> uv run runpod/runpod_ctl.py run 03_llms/01_llm_finetuning \
>     --collect --wait --wait-seconds 5700 --terminate --yes --gpu "NVIDIA L40S"
> ```

## The verified run

`deepspeed --num_gpus=2 train_qwen38_ds.py --max-steps 4 --max-samples 32` on a
rented **2 × A100-SXM4-80GB** pod. Measured, not illustrative:

```
    weight shard      27.8 GB per GPU (ZeRO-3 across 2)
    + per-GPU overhead 20.5 GB
    = needed per GPU  48.3 GB
    you have          85 GB per GPU
    verdict           FITS

  [1/4] dataset: tatsu-lab/alpaca — 32 examples
  [2/4] model: Qwen/Qwen3.8-27B  (55.6 GB)
        LoRA targets: q_proj, k_proj, v_proj, o_proj, in_proj_qkv,
                      in_proj_z, in_proj_b, in_proj_a, out_proj
        layer coverage: 64/64 (100% of depth)
  [3/4] fine-tuning
        trainable: 47.5 M of 26.94 B (0.176%)   [ZeRO-3: partitioned across ranks]
        loss 2.963 -> 2.848 -> 2.389 -> 2.076
        adapter written to ./qwen38-lora-out
  [4/4] inference
        Skipped: ZeRO-3 shards the weights across ranks.
```

Four steps on 32 examples is a **pipeline test, not a result**.

Two caveats stated plainly:

- **Generation is skipped under multi-rank ZeRO-3**, by design — the weights
  are sharded, so `generate()` on rank 0 would read partial tensors. Verified
  separately on **1 × A100-SXM4-80GB** (which the corrected capacity model says
  fits at 76.1 GB needed): the adapter loads, and the model answers coherently.
  Note the raw output continues into a `<think>` block — Qwen3.8 emits
  reasoning tokens, so strip them before scoring completions.
- **400 s/step is not representative.** That pod needed `NCCL_P2P_DISABLE=1`,
  which routes every ZeRO-3 all-gather through the host rather than the GPU
  interconnect. The workaround makes the run possible, not fast.

### What it took to get there

Five attempts on real hardware, four distinct defects — worth listing because
three of them were in this script and each produced a plausible-looking failure:

| | Symptom | Cause |
|---|---|---|
| 1 | `ImportError: Qwen2VLImageProcessor requires PIL` | TRL builds an `AutoProcessor` unless given one; on a VLM repo that is the *image* processor. Fixed with `processing_class=tokenizer` |
| 2 | no output at all | `--wait-seconds` defaults to 1800 and the pod was killed mid-download |
| 3 | hang, then `rc=250` | NCCL barrier — an allreduce of **one element** — timed out after 1,800,069 ms. `nvidia-smi topo -m` showed `SYS` between the cards. Pod, not code |
| 4 | OOM at 43.73 GiB/GPU | `SFTConfig` built *after* the model, so `zero.Init` never fired and every rank held the whole model |
| 5 | OOM at 42.23 GiB/GPU | sharding now correct; the **capacity model** was wrong — see the table above |

---

# Kimi K3: reading a 2.8-trillion-parameter model you cannot run

`analyze_kimi_k3.py` is the **fourth** entry point in this folder, and the
first that deliberately does **not** train. It is named `analyze_`, not
`train_`, for exactly that reason. It analyses
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) —
2.78 T parameters, 104 B activated per token — from its published
`config.json`.

```bash
uv run analyze_kimi_k3.py --plan            # no GPU, no download, ~2 seconds
uv run analyze_kimi_k3.py --plan --audit-snippet
uv run analyze_kimi_k3.py --verify-arch     # meta device; see the caveat below
```

## Why there is no training run here

`train_qwen38_ds.py` analyses *and* trains, because 27 B fits on two 48 GB
cards. This one only analyses, and the reason is arithmetic:

| | |
|---|---|
| parameters | 2.78 T total, 104 B activated per token |
| weights on the Hub | **1,561 GB** across 96 safetensors shards |
| as shipped | **0.56 bytes/parameter** — already quantised, 2.72 T of them in `U8` |
| realistic floor | **> 1,561 GB.** 8 × B200 is 1,440 GB and falls **121 GB short** |

**Quantisation is not the way out, because it has already been taken.** The
checkpoint is denser than fp8 on arrival, so there is no 4× left. Even true
4-bit across all 2.78 T parameters is ~1,390 GB — still eight B200s for the
weights alone, before a single activation.

That is still the *optimistic* number, because it is weights-only — this
course's rule is `weights/N + overhead that does not shard`.

Two further blockers no amount of hardware fixes:

- **The remote code does not import on the transformers this course pins.** K3
  ships custom modelling code whose `auto_map` points at
  `modeling_kimi_k3.py`, which imports `OutputRecorder` from
  `transformers.utils.generic`. That symbol survives in 5.1.0 and is **gone in
  5.2.0** (verified by installing each); every lab here pins **5.16.1**, and
  `tests/test_config_kwargs.py` fails CI if any lock disagrees. So
  `--verify-arch` needs its own pinned environment, while `--plan` always works.
- **It needs `fla-core`** (flash-linear-attention) for Kimi Delta Attention,
  which is a CUDA-compiled dependency.

**This is the honest outcome, not a shortfall.** A lab that pretends a 1.5 TB
model is rentable would waste a reader's money before it taught them anything.

### "Could I just rent 8 Blackwells?"

The obvious question, and worth answering with real prices rather than
intuition. RunPod's largest card is a **B200 at 180 GB, $5.98/GPU/hr**:

| | VRAM | $/hr |
|---|---:|---:|
| 8 × B200 | **1,440 GB** | ~$48 |
| 8 × H200 | 1,128 GB | ~$29 |
| 8 × RTX PRO 6000 Blackwell (96 GB) | 768 GB | ~$14 |

Eight B200s is **121 GB short** of just holding the weights as they ship. And
the usual escape — quantise harder — is already spent, because K3 arrives at
0.56 bytes/parameter. You would need nine, which in practice means sixteen.

**This is the rare case where "rent a bigger box" is the right diagnosis** —
which is worth stating plainly, because in this course it has been the *wrong*
one three times running (an OOM that was a missing `--use-lora`, a hang that
was NCCL peer-to-peer, a capacity model that was wrong). Here the arithmetic
really does say the machine is too small. It also does not matter, because the
remote code would not import even if the machine were big enough.


## What `--plan` reports

Every number is read from `config.json`, never hardcoded:

```
  layers           93
  FULL attention    24   (26%)
  LINEAR (KDA)      69   (74%)
  pattern          every 4th layer is full attention  ->  2.9:1 linear:full
```

| | |
|---|---|
| architecture | `KimiK3ForConditionalGeneration`, text backbone `KimiLinearForCausalLM` |
| hidden / vocab | 7,168 / 163,840 |
| context | 1,048,576 tokens |
| experts | **896 routed, 16 active (1.8%), 2 shared**, sigmoid router |
| MLA cache | `kv_lora_rank 512 + qk_rope_head_dim 64` = **576 values/token/full layer** |

**97.9% of the parameters are experts.** That one number explains the model: K3
is not a dense 2.8 T model, it is a ~104 B model with an enormous lookup table
of specialists attached. The router that picks 16 of 896 is a rounding error in
the parameter count and decides where almost everything goes — which is exactly
the point [`11_moe`](../11_moe/) makes at toy scale.

## Why it is a good capstone read

K3 composes ideas that already have their own folders here:

| K3 uses | Taught in |
|---|---|
| MLA-style compressed KV (`kv_lora_rank 512`) | [`10_deepseek_from_scratch`](../10_deepseek_from_scratch/) |
| fine-grained + shared MoE, sigmoid router | [`11_moe`](../11_moe/) |
| hybrid linear/full attention layers | `train_qwen38_ds.py`, above |
| expert parallelism | `11_moe --expert-parallel` |

The hybrid layout is the one worth pausing on. **Linear-attention layers carry
no KV cache at all**, so on a 3:1 hybrid the cache scales with the 24 full
layers rather than all 93. Combined with MLA making each of those layers
cheaper, it compounds: fewer cached layers, and each one smaller.

## A widely-copied training snippet that does not run

A fine-tuning snippet for "kimi-k3" circulates on tutorial sites.
`--plan --audit-snippet` checks it against your **installed** libraries rather
than a remembered snapshot:

| in the snippet | what happens |
|---|---|
| `model_id = "kimi/kimi-k3"` | 401 — the id is `moonshotai/Kimi-K3` |
| `SFTTrainer(tokenizer=...)` | removed in transformers 5.x → `processing_class=` |
| `SFTTrainer(max_seq_length=...)` | moved out of the trainer in trl 1.x |
| `train_dataset="train.jsonl"` | a `str`, not a `Dataset` |

Its **LoRA targets, though, are fine.** `q_proj/k_proj/v_proj/o_proj` resolve to
89 modules in the full-attention layers and 211 in the linear ones — 300 in
total. That is worth stating because the obvious guess, that a linear-attention
model must use different projection names, is **wrong here**, and only building
the module tree shows it either way.

## Verifying without a GPU

```bash
uv run tests/test_kimi_k3_plan.py       # from the repo root
```

The script cannot be validated by running it end to end, so the arithmetic is
tested directly against a fixture config. It carries a permanent counterexample:
K3's `full_attn_layers` is `[4, 8, … 92, 93]` — 23 gaps of 4 and **one of 1** —
and the first version of this script took `min()` of those gaps and printed
*"every 1th layer is full attention → 2:1"*. Both halves wrong, both a populated
field of the right type holding a plausible small integer, and neither
detectable by a shape assertion.

## Preflight for any model you cannot afford to be wrong about

Everything above was found without renting anything, and none of it is
specific to K3. If you are sizing a pod for a frontier checkpoint, these four
checks cost minutes and are the ones that would otherwise fail *after* the
download.

### 1. Do not assume the checkpoint is bf16

This is the check that changed the answer here. Ask the Hub what dtypes are
actually in the file:

```python
from huggingface_hub import HfApi
info = HfApi().model_info("moonshotai/Kimi-K3")
st = info.safetensors
print(f"total parameters: {st.total:,}")
for dtype, n in sorted(st.parameters.items(), key=lambda kv: -kv[1]):
    print(f"  {dtype:5} {n:>18,}")
```

```
total parameters: 2,779,931,837,184
  U8     2,722,740,830,208
  BF16      57,179,884,544
  F32           11,122,432
```

**`U8` for 98% of the parameters means the model is already quantised**, and
every "just load it in 4-bit" plan is dead on arrival. A `config.json` with
`torch_dtype: bfloat16` and `quantization_config: null` — which is exactly what
K3 has — tells you nothing about this. The dtype census does.

### 2. The parameter count and the byte count are different questions

Two authoritative-looking sources disagree here, and picking the wrong one puts
you off by 1.8×:

| question | source | K3 |
|---|---|---|
| how much **disk and VRAM** do I need? | `model.safetensors.index.json` → `metadata.total_size` | **1,561 GB** |
| how many **parameters** is it? | the dtype census above | **2.78 T** |

```python
import json, urllib.request
u = "https://huggingface.co/<repo>/resolve/main/model.safetensors.index.json"
idx = json.load(urllib.request.urlopen(u, timeout=120))
print(f"disk needed: {idx['metadata']['total_size']/1e9:,.0f} GB")
```

They disagree because the census counts *parameters* while the index counts
*bytes*, and packed experts store **1.88 parameters per byte** — roughly 4-bit
packing plus per-block scales. Multiplying the parameter count by a bytes-per-
parameter you assumed is how the 390 GB error happened. **Use `total_size` for
capacity planning and the census only to understand what you are holding.**

### 3. Check the remote code imports *before* you rent

Custom modelling code pins you to a transformers window that nothing documents.
K3's `modeling_kimi_linear.py` does:

```python
from transformers.utils.generic import OutputRecorder, check_model_inputs
```

Bisecting that symbol takes a few minutes and no GPU:

```bash
for v in 4.57.1 5.0.0 5.1.0 5.2.0 5.5.0 5.16.1; do
  printf "  %-8s " "$v"
  uv run --no-project --with "transformers==$v" python -c \
    "from transformers.utils.generic import OutputRecorder; print('HAS')" 2>&1 | tail -1
done
```

| transformers | `OutputRecorder` |
|---|---|
| 4.57.1, 5.0.0, **5.1.0** | present |
| **5.2.0** and later | **gone** |

So K3's remote code needs **transformers ≤ 5.1**. This course pins 5.16.1
everywhere, which is why `--verify-arch` needs its own environment. Note that
a changelog would have told you "removed in 5.2.0" only if you knew to look;
installing each version and importing the symbol is the check that cannot be
wrong.

### 4. Check the compiled dependencies, and what is actually bookable

`fla-core` (Kimi Delta Attention) is CUDA-compiled, but it **installs and
imports on a CPU box**, so you can retire that risk for free:

```bash
uv run --no-project --with fla-core python -c "import fla; print('ok')"
```

And confirm the hardware exists before planning around it — RunPod's largest
card is a B200 at 180 GB:

```bash
uv run runpod/runpod_ctl.py gpus --min-vram 80 --limit 40
```

> **The order matters.** Run **3** and **4** before **1** and **2**. Sizing is the interesting question,
but an import error makes it irrelevant — and it is the cheaper check.

## What about GGUF, unsloth, llama.cpp?

A fair objection to everything above: the community routinely runs models that
"do not fit". So does that change the answer?

**For inference, yes — completely.** For what this course is about, no. The two
halves are worth separating, because conflating them is how people end up
renting the wrong machine.

### The sizes are real

`unsloth/Kimi-K3-GGUF` publishes dynamic quants of K3, and they are not
marginal — measured from the Hub, against RunPod's largest card:

| quant | size | × B200-180GB |
|---|---:|---:|
| `UD-Q1_0` | 466 GB | 2.6 |
| `UD-IQ1_S` | 594 GB | 3.3 |
| `UD-IQ2_XXS` | 711 GB | 4.0 |
| **`UD-Q2_K_XL`** | **861 GB** | **4.8** |
| `UD-Q4_K_XL` | 1,509 GB | 8.4 |
| `UD-Q8_K_XL` | 1,561 GB | 8.7 |

So `UD-Q2_K_XL` at 861 GB fits on **8 × B200 with 579 GB to spare**, and
`UD-IQ2_XXS` fits on 8 × RTX PRO 6000 Blackwell — a **~$14/hour** machine
rather than a ~$48/hour one. The section above says eight Blackwells cannot
hold K3, and that remains true *of the released checkpoint*; it is not true of
a 2-bit conversion of it.

NVIDIA also publishes `Kimi-K3-NVFP4` at 1,610 GB, which despite the name is
**larger** than the original and still will not fit on eight cards.

### Why this does not make K3 a lab here

Three reasons, and the first is the one that matters:

- **GGUF is an inference format.** llama.cpp does not train. This course is
  about ZeRO, sharding and optimizer state — a quantised inference runtime is
  a different subject that happens to share a model. You cannot LoRA-tune a
  `UD-Q2_K_XL` file.
- **`UD-Q2_K_XL` is not Kimi K3.** It is a 2-bit approximation of it. Fine for
  "can I talk to it", not fine for any claim about the model's behaviour — and
  this course's rule is that a measured claim is scoped to the configuration it
  was measured in.
- **If you go this route you probably do not want 8 GPUs at all.** llama.cpp's
  real trick is CPU offload: the binding resource becomes system RAM and disk,
  not VRAM, and a large-RAM box is far cheaper per hour than eight Blackwells.
  Renting 8 × B200 to run a 2-bit GGUF is close to the most expensive way to
  do it.

**unsloth** is worth naming separately, because it is a *fine-tuning* library
and therefore the closest thing to an on-topic answer. Its speedups target
single-GPU LoRA on models that fit; K3 at 1,561 GB is not that, and nothing in
the unsloth repo listing above is a trainable K3. `unsloth/Kimi-K3-GGUF` is a
conversion for inference, not a training path.

> **What would actually be worth measuring.** Not "can 8 GPUs hold a 2-bit GGUF" — the arithmetic above already answers that.
The open questions are the ones this page still does not claim: **how long
1,561 GB (or 861 GB) actually takes to fetch onto a pod**, and whether that
outlives the orchestrator's window. Those are measurements, not derivations,
and they are not published here because they have not been made.

## References

- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- *Kimi K3: Open Frontier Intelligence*, [arXiv:2607.24653](https://huggingface.co/papers/2607.24653)
- [`11_moe`](../11_moe/) — the router this model scales to 896 experts
- [`10_deepseek_from_scratch`](../10_deepseek_from_scratch/) — MLA from scratch
