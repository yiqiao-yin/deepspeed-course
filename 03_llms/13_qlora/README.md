# QLoRA vs LoRA: what the GPU actually holds

This lab measures the infrastructure cost of keeping Qwen3-4B's **frozen**
base weights in bf16 (LoRA) or NF4 four-bit form (QLoRA). The training task,
adapter targets, optimizer, and ZeRO stage stay the same. This is a memory
comparison, not an evaluation of answer quality.

## What this demonstrates

DeepSpeed **ZeRO-2** shards gradients and optimizer state across ranks but
replicates frozen base weights. That makes the difference in base storage
visible per GPU. Do not switch the quantized arm to ZeRO-3: the version of
Transformers pinned here may skip ZeRO-3 parameter partitioning for quantized
weights, even when training exits successfully.

Both arms train rank-8 adapters on seven named Qwen linear projections using
the same four repeatable text samples. A repeated tiny training set gives a
controlled learning signal; it does not establish downstream model quality.
The measured peak includes model loading, training activations, adapter
optimizer state, and CUDA allocation. Measurements are **per GPU**, never
the sum over cards.

## Hardware requirements

| Resource | Starting budget | Notes |
|---|---:|---|
| GPU VRAM | 24 GB per GPU | Estimate only; verify on your actual card |
| GPUs | 1 | More ranks are optional, each still holds the base |
| Disk | 60 GB | Model weights, cache and environment |
| Host RAM | 48 GB | Loading and package installation |

An idealized 4.02B parameter base is about 8.04 GB in bf16 or 2.01 GB at
four bits; NF4 scales, unquantized layers, adapters and activations add to
the actual footprint. Full Adam fine-tuning of 4.02B weights alone would
need about 32 GB for two fp32 optimizer moments, before weights/gradients.
These are arithmetic estimates, **not measured output**.

## Environment & local testing

From this folder, on a machine with an appropriate CUDA GPU:

```bash
uv sync
uv run --locked python -c "import torch, deepspeed, peft, bitsandbytes"
uv run --locked deepspeed --num_gpus=1 train_qlora.py --max-steps 20
```

The committed `uv.lock` and explicit cu128 PyTorch index are part of the
example. Qwen3-4B weights download from Hugging Face on first use. No API
credentials are required for the public model. If your machine has no CUDA
GPU, you can still inspect the weight arithmetic and run the logic suite:

```bash
uv run --no-project python 03_llms/13_qlora/train_qlora.py --plan
uv run tests/test_qlora.py
./tests/run_all.sh
```

The last three commands are run from the **repository root**. A training
entry point without CUDA exits nonzero with a clear explanation.
`ALLOW_CPU=1` bypasses the preflight for debugging, but this 4B model
still cannot train on CPU. The `--plan` mode runs without torch.

## Running it

### CoreWeave / other SLURM clusters

On the login node, run `uv sync` in this folder once. Ensure the model
cache is available to compute nodes; set `HF_HOME` to a scratch directory
you own. Adjust the partition if necessary (`sinfo`). Then:

```bash
mkdir -p logs
sbatch run_deepspeed.sh --max-steps 20   # smoke test; cap per arm
sbatch run_deepspeed.sh                  # 40 optimizer steps per arm
squeue -u "$USER"
tail -f logs/qlora_<jobid>.out
scancel <jobid>                          # if you need to stop the job
```

### RunPod

From the repository root, inspect price and availability, launch with
automatic termination, and confirm no pod remains:

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py recommend 03_llms/13_qlora
uv run runpod/runpod_ctl.py run 03_llms/13_qlora \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods
```

While testing a contribution branch on your public fork, add
`--repo https://github.com/YOU/deepspeed-course --branch feature/qlora-vs-lora`
to the `run` command so the pod downloads the code you are testing.

The controller's `--dry-run` imposes a wall-clock limit, including any
first model download; it may end before any optimizer step. To measure
both arms, use a normal run after checking the pod setup and set a suitable
`--wait-seconds`. `--volume` sizes the `/workspace` model cache; `--disk`
sizes a different container filesystem. Check the hourly cost before
renting hardware.

### Direct

```bash
uv run --locked deepspeed --num_gpus=1 train_qlora.py --max-steps 20
```

`--max-steps` counts optimizer steps **per arm**, not epochs. The default
40 steps per arm is still only a short demonstration. An uncapped claim
about convergence or model quality is not warranted.

## Reading the results

Each arm runs in a separate Python process so the first model's GPU
allocations cannot carry over into the second measurement. The comparison
rejects a starting allocation above 256 MiB in either arm.

On rank 0, `comparison.json` holds results for each GPU rank and arm:

- `peak_allocated_bytes`: the larger allocated-memory peak from model
  loading or training. This differs from reserved memory and total VRAM
  reported by system tools.
- `stored_tensor_bytes`: parameter storage, buffers, and NF4 quantization
  state.
- `logical_parameters`: original parameter counts, accounting for NF4
  packing.
- `trainable_parameters`: adapter parameters updated during training.

The comparison requires different base-weight representations, equal
adapter and logical parameter counts, and adapter counts matching the
expected LoRA dimensions. QLoRA must have lower stored tensor bytes and
a lower measured peak on each GPU. For runs of at least 20 steps, both
arms must also have a lower final five-step mean loss than their initial
five-step mean.

## Measured GPU results

The RunPod run at commit `11d000d` completed successfully (`DONE rc=0`)
and reported `LoRA vs QLoRA comparison passed`.

### Run configuration

- GPU: one NVIDIA GeForce RTX 3090 with 24 GB VRAM.
- Model: `Qwen/Qwen3-4B`.
- DeepSpeed: ZeRO-2.
- Adapters: rank 8, with identical targets in both arms.
- Training: 40 optimizer steps per arm on four repeated text samples.
- Isolation: a fresh Python process for each arm.
- Container: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.

### Results

| Measurement | LoRA (bf16) | QLoRA (NF4) |
|---|---:|---:|
| Starting allocated memory | 0.00 GiB | 0.00 GiB |
| Allocated memory after model and adapter loading | 7.55 GiB | 3.28 GiB |
| Model-loading peak | 7.55 GiB | 3.94 GiB |
| Overall allocated-memory peak | 8.19 GiB | 3.94 GiB |
| Stored tensor memory | 7.55 GiB | 3.26 GiB |
| Trainable adapter parameters | 16,515,072 | 16,515,072 |
| First-step training loss | 5.3668 | 5.9130 |
| Final-step training loss | 0.0112 | 0.4633 |

QLoRA reduced peak allocated GPU memory by approximately **4.25 GiB
(52%)** in this run, calculated from the rounded values above.

Both arms started at 0.00 GiB at the displayed precision and trained the
same number of adapter parameters. All memory measurements are per GPU;
one GiB is 2^30 bytes.

### Interpretation and limitations

These results demonstrate memory savings for this model, configuration,
and short training task. The tiny repeated corpus and falling training
loss do not establish answer quality, generalization, or equivalent model
quality between the two arms.

Memory usage can change with sequence length, batch size, hardware, and
software versions.

### Run evidence

The controller collected the training log at:

```text
runpod/results/dsc-91e520033c2b42ebac97/03_llms/13_qlora.log
```

The log reports that `comparison.json` was saved inside the pod. The
confirmed local artifact is the collected log.

After completion, the controller terminated the pod and reported zero
pods still running.

[View the full RTX 3090 comparison log](results/rtx3090_comparison.log)
— GPU run tested at commit `11d000d`.
For readability, terminal color codes were removed and carriage-return
progress updates were placed on separate lines. Warnings and measurements
were preserved.

## October 2026: Cross-GPU Benchmark and Reproducibility Analysis

### Motivation for additional testing

The initial benchmark, conducted on September 24, 2026, using an NVIDIA RTX 3090, demonstrated that QLoRA substantially reduced peak allocated GPU memory compared with conventional LoRA. However, the findings were based on a single execution and did not include training-duration measurements.

Before submitting the project for review, I revisited the experiment on October 26, 2026, to evaluate the reproducibility of the original results and investigate potential training-time trade-offs. I incorporated CUDA-synchronized timing measurements, conducted three additional benchmarks on the RTX 3090, and performed two benchmarks on the RTX 4090.

These additional experiments were intended to determine whether the observed memory savings remained consistent across repeated runs and different GPU architectures. I also preserved the numerical results and training logs to support independent verification and improve the transparency and reproducibility of the project.

### Experimental objective

The experiments were designed to answer three questions:

1. Does QLoRA consistently reduce GPU memory consumption compared with standard LoRA?
2. How does four-bit quantization affect training duration?
3. Are GPU memory measurements and training results reproducible across repeated runs and different GPU architectures?

### Experimental configuration

The following configuration was used across all five benchmark runs:

* **Model:** `Qwen/Qwen3-4B`
* **Framework:** DeepSpeed ZeRO-2
* **LoRA representation:** BF16
* **QLoRA representation:** NF4 four-bit quantization
* **Training steps:** 40 per method
* **Trainable adapter parameters:** 16,515,072
* **GPUs:** NVIDIA GeForce RTX 3090 and RTX 4090
* **Container:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
* **Benchmark code commit:** `bfb19e4`
* **Experiment date:** October 26, 2026 (Michigan local time)

Both methods used the same training examples and adapter configuration.

Training duration was measured using Python's `time.perf_counter()` with CUDA synchronization before and after the training loop.

This approach measures elapsed training-loop time separately from model loading and DeepSpeed initialization.

Peak GPU memory measurements include model loading and training. Each method was executed in a separate process to prevent GPU memory allocations from one experiment from affecting the other.

### Individual GPU benchmark results

Five successful benchmark runs were completed: three using the RTX 3090 and two using the RTX 4090.

| GPU      | Run | LoRA Time (sec) | QLoRA Time (sec) | LoRA Peak (GiB) | QLoRA Peak (GiB) |
| -------- | --: | --------------: | ---------------: | --------------: | ---------------: |
| RTX 3090 |   1 |           12.73 |            15.19 |            8.19 |             3.94 |
| RTX 3090 |   2 |           14.93 |            17.98 |            8.19 |             3.94 |
| RTX 3090 |   3 |           16.04 |            18.74 |            8.19 |             3.94 |
| RTX 4090 |   1 |           16.10 |            14.57 |            8.19 |             3.94 |
| RTX 4090 |   2 |           13.88 |            15.95 |            8.19 |             3.94 |

All five experiments completed successfully and passed the LoRA-versus-QLoRA comparison validation.

### Statistical analysis

The included `analyze_results.py` script automatically reads the benchmark JSON files and calculates mean training time, sample standard deviation, relative training-time differences, and GPU memory savings.

| GPU      | Method | Runs | Mean Time (sec) | Sample Std. Dev. (sec) |
| -------- | ------ | ---: | --------------: | ---------------------: |
| RTX 3090 | LoRA   |    3 |           14.57 |                   1.68 |
| RTX 3090 | QLoRA  |    3 |           17.30 |                   1.87 |
| RTX 4090 | LoRA   |    2 |           14.99 |                   1.57 |
| RTX 4090 | QLoRA  |    2 |           15.26 |                   0.98 |

On the RTX 3090, QLoRA required approximately **18.74% more training time** than LoRA, based on the ratio of average training durations.

On the RTX 4090, QLoRA required approximately **1.80% more training time** based on the two available runs.

The RTX 4090 results demonstrated greater variation in the relative performance of the two methods. QLoRA was faster in the first run but slower in the second.

These differences highlight the importance of repeated measurements when evaluating GPU training performance.

The timing measurements are descriptive results from a small sample and do not establish a statistically reliable speed advantage for either method.

### GPU memory efficiency

Across all five experiments, QLoRA achieved a 51.86% reduction in peak allocated GPU memory compared with conventional LoRA, while both methods trained the same 16,515,072 adapter parameters.

Because the model, training data, adapter configuration, and DeepSpeed ZeRO-2 settings remained consistent, the observed difference primariliy reflects the reduced memory requirements of QLoRA's quantized base weights.

These measurements represent peak PyTorch-allocated CUDA memory, not total physical GPU memory consumption.

### Reproducibility findings

All five experiments produced consistent peak allocated-memory measurements. Within each GPU model, the repeated runs also produced identical training-loss measurements to the recorded precision. These findings support the reproducibility of the original September 2026 memory-efficiency results under the tested configurations.

Training durations, however, varied between runs. This suggests that execution time was more sensitive to runtime conditions than the measured memory requirements. Although the repeated experiments strengthen confidence in the memory-efficiency findings, the small sample size limits conclusions about training-speed differences and performance across other hardware configurations.

### Training-loss observations

Both LoRA and QLoRA showed reductions in training loss between the first and final optimization steps across all five experiments.

| GPU      | Method | First-Step Loss | Final-Step Loss |
| -------- | ------ | --------------: | --------------: |
| RTX 3090 | LoRA   |          5.3668 |          0.0112 |
| RTX 3090 | QLoRA  |          5.9130 |          0.4633 |
| RTX 4090 | LoRA   |          5.3456 |          0.0067 |
| RTX 4090 | QLoRA  |          5.9611 |          0.5710 |

Within each GPU group, the recorded loss measurements were consistent across repeated runs.

These results demonstrate that both configurations reduced training loss on the small, repeated training dataset. However, training loss alone cannot establish generalization or comparative model quality. Such conclusions would require evaluation using independent validation data.

### Experimental limitations

Although the additional experiments strengthened the reproducibility of the original findings, several limitations should be considered when interpreting the results:

1. **Sample size:** The analysis includes three RTX 3090 runs and two RTX 4090 runs. These measurements provide descriptive evidence but are insufficient to establish statistically significant performance differences.

2. **Training workload:** Each experiment used 40 optimization steps on a small, repeated dataset. This controlled configuration is useful for comparing memory requirements but does not represent large-scale fine-tuning or establish downstream model quality.

3. **Timing variability:** GPU clock speeds, system conditions, and software behavior can influence training duration. The reported times measure the training loop rather than the complete process of loading and initializing the model.

4. **Generalizability:** The measured memory savings apply to the tested model, hardware, and training configuration. Different models, sequence lengths, batch sizes, or software versions may produce different results. Peak PyTorch-allocated memory also differs from total physical GPU memory consumption.


### Reproducing the statistical analysis

The repository includes a Python script for automatically calculating the reported benchmark statistics.

From the repository root, execute:

```bash
python 03_llms/13_qlora/analyze_results.py
```

The script uses Python's standard library and does not require GPU access.

It reads the five JSON files located in:

`03_llms/13_qlora/results/2026-10-26/`

#### Recorded benchmark evidence

The original JSON measurements and corresponding GPU execution logs from October 26, 2026, are preserved below. Each experiment includes the numerical results and the training log documenting its execution.

| GPU      | Run | JSON Measurements                                 | Training Log                                             |
| -------- | --: | ------------------------------------------------- | -------------------------------------------------------- |
| RTX 3090 |   1 | [View JSON](results/2026-10-26/rtx3090_run1.json) | [View Training Log](results/2026-10-26/rtx3090_run1.log) |
| RTX 3090 |   2 | [View JSON](results/2026-10-26/rtx3090_run2.json) | [View Training Log](results/2026-10-26/rtx3090_run2.log) |
| RTX 3090 |   3 | [View JSON](results/2026-10-26/rtx3090_run3.json) | [View Training Log](results/2026-10-26/rtx3090_run3.log) |
| RTX 4090 |   1 | [View JSON](results/2026-10-26/rtx4090_run1.json) | [View Training Log](results/2026-10-26/rtx4090_run1.log) |
| RTX 4090 |   2 | [View JSON](results/2026-10-26/rtx4090_run2.json) | [View Training Log](results/2026-10-26/rtx4090_run2.log) |


The results can be independently recalculated using the accompanying analysis script.

### Conclusion

Across five benchmark runs using NVIDIA RTX 3090 and RTX 4090 GPUs, QLoRA consistently reduced peak PyTorch-allocated GPU memory by approximately **51.86%** compared with conventional LoRA while maintaining the same number of trainable adapter parameters. These findings reinforce the original September 2026 results and demonstrate the memory-efficiency advantages of NF4 quantization under the tested DeepSpeed ZeRO-2 configuration.

Training-duration measurements were less consistent, highlighting the importance of repeated experiments when evaluating performance trade-offs. By preserving the raw measurements, execution logs, and automated statistical analysis, this project provides reproducible evidence of QLoRA's memory efficiency while acknowledging the limitations of the experimental workload.

## Validation checks

At commit `2c3efd0`:

- The repository contract audit passed all 32 checks.
- With `CUDA_VISIBLE_DEVICES=-1`, training exited with code 1,
  displayed CPU-only alternatives, and produced no traceback.
- The QLoRA test suite passed all 18 checks.

### Deliberately broken implementation

In a separate temporary checkout, the peak-memory rejection condition
in `check_pair()` was replaced with `if False`.

The suite then passed 16 of 18 checks. These two checks failed:

- `broken test: increased NF4 GPU peak fails`
- `isolated comparison rejects increased NF4 GPU peak`

Restoring the original condition returned the suite to 18/18 passing.
This demonstrates that the tests detect disabled peak-memory validation.
The demonstration ran locally using simulated measurements; it did not
require a GPU. The broken change was never committed.