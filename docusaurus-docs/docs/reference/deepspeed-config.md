---
sidebar_position: 1
---

# DeepSpeed Configuration Reference

Every key you are likely to need, what it actually controls, and which combinations are invalid. Organized so you can jump to a block and read only that block.

:::info How to read this page
Keys are grouped by top-level block. Each table gives the default and, where it matters, the *reason* to change it. Sections that require background link to the page that develops it — most often [ZeRO Stages](/docs/getting-started/deepspeed-zero-stages).

The authoritative upstream reference is the [DeepSpeed config-json docs](https://www.deepspeed.ai/docs/config-json/); this page covers what the course uses and the failure modes that are easy to hit.
:::

## 1. Structure

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false,

  "optimizer":  { "type": "AdamW", "params": {} },
  "scheduler":  { "type": "WarmupLR", "params": {} },
  "fp16":       { "enabled": false },
  "bf16":       { "enabled": true },
  "zero_optimization": { "stage": 2 },
  "activation_checkpointing": {},
  "aio": {}
}
```

## 2. Batch Size

The most common source of startup failures.

| Key | Meaning |
|---|---|
| `train_batch_size` | **Global** batch — across all GPUs *and* all accumulation steps |
| `train_micro_batch_size_per_gpu` | Samples per GPU per forward pass |
| `gradient_accumulation_steps` | Micro-steps before an optimizer step |

DeepSpeed asserts, at initialization:

$$
\texttt{train\_batch\_size} = \texttt{train\_micro\_batch\_size\_per\_gpu} \times \texttt{gradient\_accumulation\_steps} \times N_{\text{gpus}}
$$

```
AssertionError: Check batch related parameters. train_batch_size is not equal
to micro_batch_per_gpu * gradient_acc_step * world_size
```

**You only need to specify two of the three** — DeepSpeed derives the third. Specifying all three and getting the arithmetic wrong is the usual cause.

| Given | Derived |
|---|---|
| `train_batch_size` + `gradient_accumulation_steps` | micro-batch |
| `train_batch_size` + micro-batch | accumulation steps |
| micro-batch + `gradient_accumulation_steps` | `train_batch_size` |

:::tip Make configs portable across GPU counts
A config with all three hard-coded is pinned to one `--num_gpus`. Under HuggingFace `Trainer`, set all three to `"auto"`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

Outside `Trainer`, omit `train_batch_size` and give the other two — DeepSpeed multiplies by world size for you.
:::

### The `"auto"` mechanism

`"auto"` is a **HuggingFace convention, not a DeepSpeed feature.** `Trainer` walks the config before initialization and substitutes values from `TrainingArguments`. With raw `deepspeed.initialize` there is nothing to resolve it, and you get a parse error or a string where a number belongs. Full detail in [the integration page](/docs/tutorials/huggingface/overview#2-the-auto-mechanism).

| Config key | Resolved from |
|---|---|
| `train_micro_batch_size_per_gpu` | `per_device_train_batch_size` |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` |
| `optimizer.params.lr` | `learning_rate` |
| `optimizer.params.weight_decay` | `weight_decay` |
| `optimizer.params.betas` | `adam_beta1`, `adam_beta2` |
| `optimizer.params.eps` | `adam_epsilon` |
| `gradient_clipping` | `max_grad_norm` |
| `scheduler.params.warmup_num_steps` | `warmup_steps` |
| `scheduler.params.warmup_max_lr` | `learning_rate` |

## 3. Optimizer

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 2e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }
  }
}
```

| `type` | Notes |
|---|---|
| `Adam` | Standard. With `torch_adam: false` (default) DeepSpeed uses its fused CUDA kernel |
| `AdamW` | Decoupled weight decay. **The right default for transformers** |
| `OneBitAdam`, `ZeroOneAdam`, `OneBitLamb` | Communication-compressed variants for bandwidth-limited clusters |
| `Lamb` | Layerwise adaptive; for very large batch training |
| `SGD` | Momentum SGD. Preferred for vision — see [CIFAR-10](/docs/tutorials/basic/cifar10#42-switch-adam--sgd-with-momentum) |

Memory cost per parameter — the $K$ in the [$(4+K)\Psi$ formula](/docs/getting-started/deepspeed-zero-stages#12-where-the-memory-actually-goes):

| Optimizer | $K$ (bytes/param) |
|---|---|
| Adam / AdamW, mixed precision | **12** |
| SGD with momentum | 4 |
| SGD without momentum | 0 |
| 8-bit Adam (bitsandbytes) | ~6 |

:::note `Adam` vs `AdamW` is not cosmetic
`Adam` applies weight decay by adding $\lambda\theta$ to the gradient, so it is scaled by the adaptive term $1/(\sqrt{\hat v}+\epsilon)$ — parameters with large gradient variance get *less* decay, which is not what regularization is supposed to do. `AdamW` (Loshchilov & Hutter, 2019) decouples it, applying $\theta \leftarrow \theta - \eta\lambda\theta$ separately. For any run with nonzero weight decay, use `AdamW`.
:::

**CPU offload requires a compatible optimizer.** With `offload_optimizer.device: "cpu"`, DeepSpeed substitutes `DeepSpeedCPUAdam`, a hand-tuned AVX/OpenMP implementation. A naive CPU Adam would make offload useless. This is why offload works out of the box for Adam and not for arbitrary custom optimizers.

## 4. Scheduler

```json
{
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-5,
      "warmup_num_steps": 100,
      "total_num_steps": 10000
    }
  }
}
```

| `type` | Behaviour |
|---|---|
| `WarmupLR` | Linear warmup to `warmup_max_lr`, then constant |
| `WarmupDecayLR` | Warmup, then linear decay to zero. Requires `total_num_steps` |
| `WarmupCosineLR` | Warmup, then cosine decay |
| `OneCycle` | Cyclical schedule |

Warmup matters for Adam because the second-moment estimate $\hat v$ is unreliable in the first steps — see [bias correction](/docs/tutorials/basic/neural-network#52-momentum-and-adam). It matters independently for BatchNorm models, whose running statistics are meaningless early.

:::warning Do not configure a scheduler in two places
If `TrainingArguments` sets `lr_scheduler_type` **and** `ds_config.json` defines a `scheduler` block, DeepSpeed's wins and the HF setting is silently ignored. Pick one. Under `Trainer`, the least surprising choice is to omit the `scheduler` block entirely and let HF drive it.
:::

## 5. Mixed Precision

### BF16 — prefer this on Ampere or newer

```json
{ "bf16": { "enabled": true } }
```

8-bit exponent (same dynamic range as FP32), 7-bit mantissa. **No loss scaling needed**, no overflow at $g^2$. Requires A100, H100, B200, RTX 30xx/40xx/50xx.

### FP16 — for V100, T4, and older

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

| Key | Meaning |
|---|---|
| `loss_scale` | `0` = **dynamic** scaling. Any positive value = fixed scale |
| `initial_scale_power` | Starting scale is $2^{\texttt{power}}$; 16 → 65536 |
| `loss_scale_window` | Clean steps before the scale is doubled |
| `hysteresis` | Consecutive overflows tolerated before halving |
| `min_loss_scale` | Floor. Hitting it repeatedly means a genuine numerical problem |

Mechanism and rationale: [FP16 and dynamic loss scaling](/docs/tutorials/basic/neural-network#85-fp16-and-dynamic-loss-scaling).

:::danger Never enable `fp16` and `bf16` together
They are mutually exclusive. Setting both raises at initialization. Setting *neither* is valid and means FP32 — which is what `03_huggingface/02_trl_sft/ds_config.json` does.
:::

## 6. ZeRO Optimization

Full derivation in [ZeRO Stages](/docs/getting-started/deepspeed-zero-stages). Summary:

| Stage | Partitions | Memory/GPU (Adam) | Volume |
|---|---|---|---|
| 0 | nothing (plain DDP) | $16\Psi$ | $2\Psi$ |
| 1 | optimizer states | $4\Psi + 12\Psi/N_d$ | $2\Psi$ |
| 2 | \+ gradients | $2\Psi + 14\Psi/N_d$ | $2\Psi$ |
| 3 | \+ parameters | $16\Psi/N_d$ | $3\Psi$ |

### Stage 1 / 2

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true,
    "round_robin_gradients": false
  }
}
```

| Key | Effect |
|---|---|
| `overlap_comm` | Overlaps gradient reduction with backward compute. **Costs extra buffer memory** — roughly 4.5× the bucket size — so reduce buckets if you OOM with it on |
| `contiguous_gradients` | Copies gradients into a flat pre-allocated buffer. Prevents fragmentation; usually worth it |
| `reduce_bucket_size` | Elements per reduction bucket. Larger = fewer, bigger collectives (better bandwidth, more memory) |
| `allgather_bucket_size` | Same trade for the parameter gather |
| `round_robin_gradients` | Stage 1/2 with CPU offload: distributes gradient copies across ranks to parallelize PCIe transfers |

### Stage 3

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "sub_group_size": 1e9,
    "memory_efficient_linear": true
  }
}
```

| Key | Effect |
|---|---|
| `stage3_max_live_parameters` | Ceiling on parameters simultaneously materialized on GPU. **The primary OOM control at Stage 3** — lower it to fit, at a throughput cost |
| `stage3_prefetch_bucket_size` | How far ahead to gather. Larger hides more latency, uses more memory |
| `stage3_param_persistence_threshold` | Tensors smaller than this are never partitioned. LayerNorm gains and biases are tiny and numerous; gathering them is pure latency |
| `stage3_max_reuse_distance` | Keeps a parameter resident if it will be reused within this many parameters |
| `sub_group_size` | Parameters processed per optimizer sub-step. Lower it if the *optimizer step itself* OOMs |
| `stage3_gather_16bit_weights_on_model_save` | Consolidates shards into a loadable checkpoint |
| `memory_efficient_linear` | Reduces temporaries in linear layers |

:::danger Stage-3 checkpoints without the gather flag are shards
Omit `stage3_gather_16bit_weights_on_model_save` and `from_pretrained` cannot load the result. Recover with `zero_to_fp32.py` in the checkpoint directory:

```bash
python zero_to_fp32.py . pytorch_model.bin
```

Under LoRA this is largely moot — save adapters with `model.save_pretrained()` rather than writing the full base model.
:::

### Offload

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true, "buffer_count": 4 },
    "offload_param":     { "device": "cpu", "pin_memory": true, "buffer_count": 5, "max_in_cpu": 1e9 }
  }
}
```

| `device` | Notes |
|---|---|
| `"none"` | Disabled — keep on GPU |
| `"cpu"` | Host RAM. Budget $\approx 12\Psi$ bytes for Adam states |
| `"nvme"` | Requires `nvme_path`. **Local NVMe only** — never a network filesystem |

`pin_memory: true` allocates page-locked host memory so transfers can DMA without stalling the host. Nearly always worth it; costs non-swappable RAM.

`offload_param` requires **Stage 3**. `offload_optimizer` works with Stage 1, 2, or 3.

### NVMe

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param":     { "device": "nvme", "nvme_path": "/local_nvme", "pin_memory": true },
    "offload_optimizer": { "device": "nvme", "nvme_path": "/local_nvme", "pin_memory": true }
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
}
```

The `aio` block tunes libaio. Defaults are reasonable; tune with DeepSpeed's `ds_io` benchmark if NVMe throughput is your bottleneck.

## 7. Activation Checkpointing

Attacks activations, which ZeRO-DP does not touch. Essential for CNNs and vision-language models, where activations dominate.

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

| Key | Effect |
|---|---|
| `partition_activations` | Partition checkpointed activations across ranks (ZeRO-R) |
| `cpu_checkpointing` | Offload them to host RAM. Slower still, saves more |
| `contiguous_memory_optimization` | Copy into contiguous buffers to reduce fragmentation |
| `synchronize_checkpoint_boundary` | Insert `cuda.synchronize()` at boundaries — debugging aid |

:::note This block alone often does nothing
For HuggingFace models the recomputation is driven by the model, not the config. You also need

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False        # incompatible with checkpointing
```

or `gradient_checkpointing=True` in `TrainingArguments`. The `activation_checkpointing` block configures DeepSpeed's *own* checkpointing API, which applies when you call `deepspeed.checkpointing.checkpoint` in a custom model. Setting the block and expecting HF models to recompute is a common disappointment.
:::

## 8. Other Keys

```json
{
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false,
  "memory_breakdown": false,
  "prescale_gradients": false,
  "gradient_predivide_factor": 1.0,
  "data_types": { "grad_accum_dtype": "fp32" },
  "comms_logger": { "enabled": false, "verbose": false, "prof_all": true, "debug": false },
  "flops_profiler": { "enabled": false, "profile_step": 1, "module_depth": -1, "top_modules": 1, "detailed": true },
  "tensorboard": { "enabled": false, "output_path": "./tb/", "job_name": "run" },
  "csv_monitor": { "enabled": false, "output_path": "./csv/", "job_name": "run" }
}
```

| Key | Notes |
|---|---|
| `gradient_clipping` | Clips global grad norm. **Set it** — mandatory for RNNs and RL. Requires a cross-rank all-reduce of a scalar under ZeRO, which DeepSpeed handles |
| `steps_per_print` | Logging cadence. Set high for long runs; the print is a synchronization point |
| `wall_clock_breakdown` | Per-stage timing (forward/backward/step). Useful once, expensive always |
| `flops_profiler` | Reports achieved FLOPS and per-module cost. Excellent for finding the real bottleneck |
| `data_types.grad_accum_dtype` | `bf16` saves memory during accumulation; watch precision with many accumulation steps |
| `comms_logger` | Logs collective sizes and times — the tool for diagnosing communication-bound Stage 3 |

## 9. Complete Examples

**Small model, single node** (`01_basics/01_neuralnet`):

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": { "type": "Adam", "params": { "lr": 1e-3 } },
  "fp16": { "enabled": true }
}
```

**HuggingFace LoRA fine-tuning** — the most common configuration in this course:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 2e8,
    "allgather_bucket_size": 2e8
  },
  "optimizer": { "type": "AdamW", "params": { "lr": "auto", "weight_decay": "auto" } },
  "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": 0, "warmup_max_lr": "auto", "warmup_num_steps": "auto" } },
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**Very large model with offload** (`05_video_speech`, 560B):

```json
{
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true, "buffer_count": 4 },
    "offload_param":     { "device": "cpu", "pin_memory": true, "buffer_count": 5, "max_in_cpu": 1e9 },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 5e8,
    "stage3_gather_16bit_weights_on_model_save": true,
    "memory_efficient_linear": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

## 10. Invalid Combinations

| Combination | Result |
|---|---|
| `fp16.enabled` **and** `bf16.enabled` | Error at initialization |
| `offload_param` with stage 1 or 2 | Ignored — parameter offload needs Stage 3 |
| `"auto"` outside HuggingFace `Trainer` | Unresolved; parse error or a string where a number belongs |
| Batch fields inconsistent with `--num_gpus` | `AssertionError: Check batch related parameters` |
| `scheduler` block **and** HF `lr_scheduler_type` | DeepSpeed wins; the HF setting is silently ignored |
| Stage 3 without `stage3_gather_16bit_weights_on_model_save` | Checkpoint saved as unloadable shards |
| `offload_optimizer` to CPU with insufficient RAM | Host swaps; throughput effectively stops |
| `nvme_path` on a network filesystem | Catastrophically slow |
| `activation_checkpointing` block without `gradient_checkpointing_enable()` | No effect on HF models |
| `use_cache=True` with gradient checkpointing | Warning; checkpointing may be silently disabled |

## 11. Validating a Config

```python
import json
from deepspeed.runtime.config import DeepSpeedConfig

with open("ds_config.json") as f:
    cfg = json.load(f)
print(json.dumps(cfg, indent=2))     # catches JSON syntax errors first
```

Check the environment and which ops are available:

```bash
ds_report
```

Then run one step with `steps_per_print: 1` and `wall_clock_breakdown: true` and read what DeepSpeed prints at startup — it echoes the resolved config, including everything `"auto"` became. **That echo is the ground truth**, and comparing it against what you intended resolves most configuration confusion in one step.

## Next Steps

- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) — the theory behind §6
- [Troubleshooting](/docs/reference/troubleshooting) — symptom-first diagnosis
- [HuggingFace Integration](/docs/tutorials/huggingface/overview) — `"auto"` and strategy selection

## References

1. [DeepSpeed configuration JSON reference](https://www.deepspeed.ai/docs/config-json/) — upstream, authoritative.
2. [HuggingFace DeepSpeed integration](https://huggingface.co/docs/transformers/deepspeed)
3. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO. *SC '20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
4. Ren, J., et al. (2021). ZeRO-Offload. *USENIX ATC '21*. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)
5. Rajbhandari, S., et al. (2021). ZeRO-Infinity. *SC '21*. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)
6. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) — Adam vs AdamW.
7. Micikevicius, P., et al. (2018). Mixed Precision Training. *ICLR 2018*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
8. Chen, T., et al. (2016). Training Deep Nets with Sublinear Memory Cost. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
