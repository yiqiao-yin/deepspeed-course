# RunPod Automation

Rent a GPU, run a course example on it, and shut it down — from the command line.

```bash
export RUNPOD_API_KEY=...          # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py gpus --min-vram 24      # what's available, live prices
uv run runpod/runpod_ctl.py recommend 06_huggingface_grpo
uv run runpod/runpod_ctl.py run 06_huggingface_grpo --yes
uv run runpod/runpod_ctl.py pods                    # what am I paying for?
uv run runpod/runpod_ctl.py terminate <podId>       # stop paying
```

Stdlib only — no dependencies. `uv run` handles the interpreter.

> ### 💸 Billing starts the moment a pod is created
> It continues until the pod is **terminated**. *Stopping* is not enough.
> `create` and `run` both refuse without `--yes` and print the hourly rate first.
> Run `pods` when you are done and confirm it says *"Nothing is billing."*

## Commands

| Command | Does |
|---|---|
| `gpus [--min-vram N] [--max-price P]` | Live GPU catalogue with on-demand and spot prices |
| `recommend <example>` | Maps an example to its VRAM/disk needs, lists the cheapest GPUs that fit |
| `create --gpu <id>` | Creates a bare pod |
| `run <example>` | Picks a GPU, creates a pod that **clones this repo and starts training** |
| `pods` | Lists pods, per-hour cost, and SSH details |
| `terminate <id>...` | Terminates pods and stops billing |

## What `run` actually does

There is no file upload and no SSH key setup. The pod's start command bootstraps
itself from the public repository:

```bash
cd /workspace
git clone --depth 1 https://github.com/yiqiao-yin/deepspeed-course.git
cd deepspeed-course
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --system deepspeed
cd <example>
deepspeed --num_gpus=<N> <script>.py 2>&1 | tee /workspace/train.log
```

So the pod is disposable: nothing local is pushed to it, and re-running gets a
clean clone of `main`. Use `--branch` to test another branch.

## Known limitations

**No log streaming.** RunPod's REST API has no log endpoint — the `Pod` schema
exposes `portMappings` and `ports` but nothing log-shaped. To watch a run:

```bash
uv run runpod/runpod_ctl.py pods       # prints the ssh line once an IP exists
ssh root@<ip> -p <port>
tail -f /workspace/train.log
```

That needs an SSH key registered on your RunPod account. The web console also
shows container logs with no key required.

**Capacity is not guaranteed.** Popular GPUs are frequently sold out; RunPod
returns HTTP 500 *"no instances currently available"*. The tool reports that
plainly and suggests alternatives — nothing is created and nothing is billed.
Try another GPU or `--cloud COMMUNITY`.

**`09_vss` will not work here.** It needs roughly **3 TB of host RAM**, which
RunPod pods do not provide. GPU VRAM is not the binding constraint for that
example. `recommend` warns about this explicitly.

## Example requirements

`run` and `recommend` size the pod from this table (in `runpod_ctl.py`):

| Example | Min VRAM | GPUs | Disk |
|---|---|---|---|
| `01_basic_neuralnet` | 6 GB | 1 | 20 GB |
| `02_basic_convnet` | 6 GB | 1 | 20 GB |
| `02_basic_convnet_cifar10_examples` | 8 GB | 1 | 30 GB |
| `03_basic_rnn` | 8 GB | 1 | 20 GB |
| `04_bayesian_neuralnet` | 8 GB | 2 | 20 GB |
| `04_intermediate_rnn_stock_data` | 8 GB | 1 | 20 GB |
| `05_huggingface_trl` | 24 GB | 1 | 60 GB |
| `05_huggingface_ocr` | 24 GB | 1 | 60 GB |
| `06_huggingface_grpo` | 24 GB | 1 | 80 GB |
| `07_..._gpt_oss_finetune_sft` | 80 GB | 4 | 200 GB |
| `08_vtt` | 48 GB | 2 | 120 GB |
| `09_vss` | 180 GB | 2 | 2 TB (plus ~3 TB host RAM — see above) |

## Image choice

The default is a **`devel`** image:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

This matters. The `runtime` variants ship no `nvcc`, so DeepSpeed cannot
JIT-compile its fused CUDA ops and every example dies with
`CUDA_HOME environment variable is not set`. Override with `--image` only if you
are sure the replacement includes the CUDA toolkit.

## Cost discipline

1. **`pods` before you walk away.** The most expensive mistake is a forgotten pod.
2. **Develop small, train big.** Debug a shape error on an RTX 3090 at ~$0.22/hr,
   not on 4×H100. A bug found on the big rig costs 40× as much.
3. **Terminate, don't stop**, unless you specifically want to keep the volume.
4. **Watch the download clock.** Pulling 40 GB of weights bills GPU time for
   pure I/O. A network volume lets you reuse the cache across pods.

## Verified behaviour

The read-only paths (`gpus`, `recommend`, `pods`) and the full
create → running → terminate lifecycle have been exercised against the live API.
`tests/test_runpod_ctl.py` covers the offline logic — the example table, GPU
selection, bootstrap generation, and the `--yes` guard — with no API key and no
network.

```bash
uv run tests/test_runpod_ctl.py
```
