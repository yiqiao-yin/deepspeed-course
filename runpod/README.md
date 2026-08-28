# RunPod Automation

Rent a GPU, run a course example on it, and shut it down — from the command line.

```bash
export RUNPOD_API_KEY=...          # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py gpus --min-vram 24      # what's available, live prices
uv run runpod/runpod_ctl.py recommend 06_huggingface_grpo
uv run runpod/runpod_ctl.py run 06_huggingface_grpo --collect --wait --terminate --yes
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
| `fetch <topic> --wait` | Downloads results the pod pushed — **no SSH needed** |
| `smoke [examples...]` | Dry-runs several examples, one pod each, all collecting |
| `terminate <id>... \| --all` | Terminates pods and stops billing |

### Auto-termination

`--wait --terminate` runs the whole cycle in one command and shuts the pod down
in a `finally` block, so a crash, a network failure or Ctrl-C still terminates it:

```bash
uv run runpod/runpod_ctl.py run 01_basic_neuralnet \
    --dry-run --collect --wait --terminate --yes
```

Termination retries five times with backoff — verified working through three
consecutive DNS failures — and shouts loudly with the manual command if it still
cannot delete the pod.

Two more safety nets:

- **In-pod watchdog.** `--max-hours` (default 6) kills the container from the
  inside regardless of what your machine is doing. Needs no API key on the pod.
- **`terminate --all`.** The blunt instrument for cleaning up after yourself.

The pod is **never given `RUNPOD_API_KEY`** — letting it delete itself would mean
putting a spending credential on rented hardware. Termination is driven from your
machine instead. See [SECURITY.md](../SECURITY.md).

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

## Getting results back without SSH

RunPod exposes **no log endpoint** — verified against both the REST OpenAPI spec
(the `Pod` schema has `portMappings` and `ports`, nothing log-shaped) and GraphQL
introspection (no log/output/console field on any type). The pod cannot be read
from, so it has to **push**.

```bash
uv run runpod/runpod_ctl.py run 01_basic_neuralnet --dry-run --collect --yes
# ... prints:  Results topic: dsc-c1b3231b898f4856af25

uv run runpod/runpod_ctl.py fetch dsc-c1b3231b898f4856af25 --wait
```

```
  [1/6] pod up: 36303e2eb4e5
  [2/6] repo cloned
  [3/6] uv installed: uv 0.12.7
  [4/6] deepspeed installed
  [5/6] env captured
  [6/6] DONE rc=0 — log attached

  saved runpod/results/dsc-.../01_basic_neuralnet.log  (21830 bytes)
```

**How it avoids the chicken-and-egg.** The topic is generated *locally* before
the pod exists, so we know where to look before the pod has said anything. The
pod publishes progress lines and attaches its log; `fetch` polls and writes
everything to `runpod/results/<topic>/`. No SSH key, no port forwarding, no
console. Structurally the same as writing run artefacts to S3 — without needing
credentials on the pod.

Transport is [ntfy.sh](https://ntfy.sh), a no-auth pub/sub. Override with
`DSC_NTFY_SERVER` to point at your own instance.

> ### ⚠️ Topics are public — never push secrets
> See [SECURITY.md](../SECURITY.md) for the full posture.
>
> Anyone who knows the topic can read it. Topics are random 20-hex-character
> strings, so they are unguessable, but they are not *private*. The bootstrap
> pushes only `nvidia-smi`, version banners, `ds_report` and training stdout.
> **Do not extend it to echo tokens or dataset contents.** If you need
> confidentiality, run your own ntfy server via `DSC_NTFY_SERVER`.

### `--dry-run`

Caps the training step at 300 seconds. The pod still clones, installs and
launches the real script, so a genuine failure still surfaces — you just do not
pay for a full run. `01_basic_neuralnet` finishes well inside the cap.

**Verified end to end** on an RTX 3090: the run converged to
`Learned Weight: 2.000000 / Learned Bias: 1.000000` against the true `y = 2x + 1`.
See [`sample_output/`](sample_output/).

## Validating several topics at once

The examples that cannot run locally are exactly the ones most worth checking on
real hardware. `smoke` starts one pod per example, each with `--dry-run` and
`--collect`:

```bash
uv run runpod/runpod_ctl.py smoke 01_basic_neuralnet 03_basic_rnn 06_huggingface_grpo
#   Will start 3 pod(s), one per example:
#     01_basic_neuralnet    1x  6G  ~$0.13/hr
#     03_basic_rnn          1x  8G  ~$0.13/hr
#     06_huggingface_grpo   1x 24G  ~$0.22/hr
#   Combined burn rate: ~$0.48/hour
#   Refusing without --yes.
```

Add `--yes` to proceed. It prints a `fetch` line per example, then the terminate
reminder.

> ### Pods are not auto-terminated
> `--dry-run` caps the *training step*, not the pod's lifetime. The container
> keeps running — and billing — after the script exits. Always finish with:
> ```bash
> uv run runpod/runpod_ctl.py pods
> uv run runpod/runpod_ctl.py terminate <id> [<id> ...]
> ```

**Suggested order.** Start with the cheap tier to confirm the mechanism, then
spend on the expensive ones:

| Tier | Examples | ~$/hr each |
|---|---|---|
| Cheap — verify the harness | `01`, `02`, `02_cifar10`, `03`, `04*` | 0.12–0.22 |
| Mid — real models, real downloads | `05_trl`, `05_ocr`, `06_grpo`, `07_multi_agency` | 0.22–0.35 |
| Expensive — only once the above pass | `07_gpt_oss` (4×80G), `08_vtt` (2×48G) | 3–8 |
| Not viable on RunPod | `09_vss` — needs ~3 TB host RAM | — |

## Other limitations

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
