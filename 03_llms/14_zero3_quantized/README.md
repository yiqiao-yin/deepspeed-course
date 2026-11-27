# ZeRO-3 parameter partitioning with BF16 and NF4

## What this demonstrates

Does storing Qwen3-4B's frozen base weights in NF4 change which parameters have ZeRO-3 backing shards after DeepSpeed engine initialization?

- **Baseline and null hypothesis:** The BF16 arm supplies the stage-matched baseline. After engine initialization, the NF4 arm shows the same pattern of ZeRO-3 management and partitioning for corresponding eligible parameters.
- **Alternative hypothesis:** The NF4 arm shows a different pattern after initialization, such as packed base weights without primary ZeRO shards. An earlier difference during model loading may disappear after engine initialization. Missing or inconsistent metadata makes the comparison inconclusive for those tensors.
- **Budget:** One model load, LoRA attachment, and engine initialization per arm per launch, with two ranks and zero optimizer steps.

Within each BF16/NF4 comparison, use the same model ID and resolved checkpoint revision, ZeRO stage 3, BF16 compute, and LoRA adapter configuration. Both paths register the Hugging Face DeepSpeed config before loading. BF16 loads unquantized weights and explicitly freezes the base model. NF4 requests bitsandbytes 4-bit weights with double quantization and BF16 compute, lets the loader select the current rank's CUDA device, and applies PEFT's k-bit preparation. The paths use the same LoRA targets and settings. The locked Transformers source gates its `deepspeed.zero.Init` branch on non-quantized loading. Measure both arms after engine initialization to determine whether the loading difference persists. Compare matching named base tensors separately from the adapters and compare the same execution stage on each rank. The comparison covers the two initialization paths as implemented, including NF4-specific preparation. It measures neither training quality nor throughput.

## Hardware requirements

Use Linux and two BF16-capable NVIDIA GPUs on one host, with a driver compatible with the locked CUDA 12.8 PyTorch build. Each rank needs its own visible device. Use a filesystem accessible to both ranks for results.

The current planning settings are 24 GB VRAM per GPU and 80 GB disk in the RunPod registry, and 64 GB host RAM, eight CPUs, and a two-hour wall-time limit in the SLURM launcher. These are starting estimates. Required VRAM, disk usage, and runtime are not yet verified on hardware. Allow disk space for the model cache and locked dependencies. CPU-only machines can run the configuration plans and CPU regression tests.

## Environment & Local Testing

Run the setup from this lab directory using the committed lockfile. The lab's `.python-version` selects Python 3.12 in uv. The `src/` directory contains a scaffold stub. Launch the experiment through `train_zero3.py`. The project locks PyTorch (CUDA 12.8), DeepSpeed, Transformers, PEFT, bitsandbytes, and their dependencies.

```bash
uv sync --locked
```

Run both lightweight plans with Python 3.12. They read the local config and need no package installation, model download, or GPU allocation.

```bash
python3 train_zero3.py --arm bf16 --plan
python3 train_zero3.py --arm nf4 --plan
./run_deepspeed.sh bf16 --plan
./run_deepspeed.sh nf4 --plan
./run_deepspeed.sh bf16 --plan --model Qwen/Qwen3-0.6B
./run_deepspeed.sh nf4 --plan --model Qwen/Qwen3-0.6B
bash -n run_deepspeed.sh
```

Both plans print the model, arm, and ZeRO stage, followed by this observed output:

```text
PASS: configuration structure valid. GPU measurements: NOT PERFORMED
```

Run all four lab suites on a Linux CPU host with the same command used in CI. The runner uses the CPU build of the lab’s PyTorch release and the same Transformers version as the lab lock. It checks synthetic tensors, stubbed GPU commands, mocked engine calls, and Transformers source. Use the locked project environment for GPU runs.

```bash
bash run_cpu_tests.sh
```

## Running it

Before running either arm, record the Git commit, resolved Hugging Face checkpoint revision, hardware and driver, and whether the model and dependencies were already cached. The loader resolves the selected Hugging Face model ID. Resolve and cache one checkpoint revision for each model before the comparison, use that snapshot for both arms, and retain its revision with the results. Each rank report records the selected model ID. Dependency locking covers the Python environment. Match GPU type, rank topology, and launch conditions across arms.

### Direct Linux launch

From the lab directory, run one arm per launch:

```bash
# Primary comparison (Qwen/Qwen3-4B is the default)
./run_deepspeed.sh bf16 --model Qwen/Qwen3-4B
./run_deepspeed.sh nf4 --model Qwen/Qwen3-4B

# Smaller control (run BF16 first and inspect its partitioning)
./run_deepspeed.sh bf16 --model Qwen/Qwen3-0.6B
./run_deepspeed.sh nf4 --model Qwen/Qwen3-0.6B
```

The launcher defaults to the BF16 arm. It and `train_zero3.py` accept one `--model MODEL` argument, with choices `Qwen/Qwen3-4B` and `Qwen/Qwen3-0.6B`. Invoke `--plan` directly on the CPU or login node. It exits before GPU checks and model loading.

Run the 0.6B BF16 control first. After engine initialization, check each rank for partitioned base tensors, zero tensors classified as missing a DeepSpeed shard, and backing shard capacity consistent with the partition world size after accounting for padding. If the control fails, stop and diagnose the harness before paying for NF4 or 4B. If the control passes, run 0.6B NF4 against the same revision and assess whether partitioning differs. Report matching partitioning as evidence against the hypothesis for that checkpoint. A 0.6B result qualifies the harness on the smaller checkpoint. The Qwen3-4B question requires both arms on Qwen3-4B.

### CoreWeave SLURM

Run `uv sync --locked` on the login node, and make the model cache accessible to compute nodes before allocating GPUs if they lack outbound access. From this lab directory, submit one arm at a time. Choose a site-approved partition and wall-time budget suitable for initialization.

```bash
sbatch run_deepspeed.sh bf16
squeue -u "$USER"
job_id=12345  # Replace with the ID printed by sbatch.
tail -f "zero3_${job_id}.out"
# If you need to cancel: scancel "$job_id"
# After the first job and artifact check, submit the second arm:
sbatch run_deepspeed.sh nf4
```

Errors go to `zero3_<job-id>.err`. An expired or canceled job is incomplete. The batch script requests GPUs even if it receives `--plan`, so use the direct plan command for the no-allocation check.

### RunPod

From the repository root, configure `RUNPOD_API_KEY` in your shell and use the project helper to inspect the resource estimates. The registered automated run selects BF16. Before using automated provisioning, integrate the upstream RunPod API migration in PR #128 and verify the bootstrap and collector against the resulting branch. The following paid dry-run command uses the fork branch and automatically terminates the pod. Its 300-second launch timeout excludes setup. If initialization times out, the helper reports a failed run.

```bash
uv run runpod/runpod_ctl.py recommend 03_llms/14_zero3_quantized
uv run runpod/runpod_ctl.py run 03_llms/14_zero3_quantized \
  --repo https://github.com/hallleedustin-alt/deepspeed-course \
  --branch feature/zero3-quantization \
  --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods
```

The automated collector saves complete BF16 measurements under `runpod/results/<topic>/03_llms/14_zero3_quantized/results/runpod-<id>/<run-id>/`. It checks that the expected rank reports match the manifest and saves the manifest only once all reports have been retrieved. For both arms on RunPod, use a two-GPU pod with terminal access, check out the same research revision, run the two direct commands under an explicit wall-clock budget, and copy the complete run directories and logs off the pod before terminating it. Terminate the pod even if a run fails and verify shutdown with `uv run runpod/runpod_ctl.py pods`.

### Reading the results

Each launcher invocation writes a unique `results/<arm>/<run-id>/` directory. A complete two-rank run contains `<arm>_rank0.json`, `<arm>_rank1.json`, and `measurements_complete.json`. Compare the per-rank `before_lora`, `before_engine` (after LoRA), and `after_engine` snapshots. Inspect ZeRO management, partition classification, and primary shard capacity alongside parameter-view size and shard payload bytes. Gathered views and padding can obscure partitioning if judged by size alone. Unknown logical counts appear as `null`. Packed elements describe storage and cannot substitute for logical counts.

Treat a nonzero exit, timeout, missing manifest, or missing or mismatched rank report as unsuccessful or incomplete. Verify that the manifest and rank reports agree on run ID, arm, and world size. The manifest records collection completion. It does not establish replication, cross-rank agreement, or scientific validity. Retain logs and any partial output with its failure status.

**GPU validation is pending.** Confirm that both arms load and initialize on two GPUs, that frozen base weights and trainable adapters are identifiable, and that rank-level metadata for matching tensors supports the same interpretation across ranks. Preserve null and inconclusive comparisons, including cases where metadata is missing.
