# Basic Neural Network with DeepSpeed

Train a simple linear regression model using DeepSpeed for distributed training on synthetic data.

## Environment & Local Testing

### Setup with `uv`

This folder is a **self-contained `uv` project** — it ships a `pyproject.toml`
and a committed `uv.lock`, so after cloning the repository:

```bash
cd 01_basics/01_neuralnet
uv sync                                  # creates .venv, installs the LOCKED versions
uv run deepspeed --num_gpus=1 train_ds_enhanced.py
```

`uv run` uses the project environment directly, so there is no `activate` step.
Add `uv sync --extra tracking` if you want Weights & Biases; it stays optional
and the script runs without it.

Why a lock file rather than the `uv pip install` recipe: everyone who clones
this repository resolves to byte-identical versions. `uv pip install torch
deepspeed` resolves to whatever is newest that day, which is how a tutorial
that worked in March stops working in September with nobody having touched it.
Regenerate deliberately with `uv lock --upgrade`.

<details>
<summary>Prefer not to use a project? The manual route still works</summary>

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed
```

The `--index-url` here is **required**, and pins the same CUDA build as
`uv.lock`. PyPI's *default* `torch` is a CUDA 13 wheel; on a driver older
than CUDA 13 it installs fine and then reports
`cuda.is_available() == False`. Verified on a driver 550.127 box, where
`uv sync` succeeded and torch could not see the GPU at all.
</details>

> **If DeepSpeed stops with `CUDA_HOME environment variable is not set`,** you
> have the NVIDIA driver but not the CUDA *toolkit*, so it cannot JIT-compile
> its fused Adam. Add `"torch_adam": true` to the `optimizer.params` block of
> `ds_config_fp32.json` and it will use PyTorch's Adam instead. For a model with
> two parameters the difference is unmeasurable. Verified on this exact path.

### Running

| | |
|---|---|
| Runs end to end on one machine | **Yes** |
| GPUs requested by the launcher | 1 |
| Downloads | none — synthetic data |

Two learnable parameters. Runs on any CUDA GPU in seconds.

```bash
cd 01_basics/01_neuralnet
deepspeed --num_gpus=1 train_ds_enhanced.py
```


### Doing less work: `--max-steps`

Every training script here accepts `--max-steps N`, which stops after `N`
optimizer steps instead of running the full schedule. `-1` (the default) means
"train normally".

This example is seconds long anyway, so the cap is not about saving time — it is
how you check that an edit to the training loop still *takes a step* before you
trust a number it printed.

```bash
# directly
deepspeed --num_gpus=1 train_ds_enhanced.py --max-steps 5

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

## Features

- 🎯 **Simple & Educational**: Perfect introduction to DeepSpeed with minimal complexity
- 📊 **Linear Regression**: Trains y = 2x + 1 model on synthetic data
- ⚡ **DeepSpeed Integration**: Demonstrates core DeepSpeed initialization and training loop
- 🔧 **FP16 Training**: Mixed precision training for faster computation
- 💻 **Multi-GPU Ready**: Supports distributed training across multiple GPUs
- 📈 **Comprehensive Logging**: Detailed training progress with parameter convergence tracking
- 🎯 **Ground Truth Validation**: Automatic comparison of learned parameters vs. true values
- 🔄 **Optional W&B Tracking**: Seamless Weights & Biases integration (never crashes if not configured)

## Quick Start on RunPod

### 1. Initial Setup

Start with a fresh RunPod instance (recommend >= 1x RTX 4090 or A100):

```bash
# Install uv package manager
pip install uv

# Initialize new project
uv init basic-neuralnet-ds
cd basic-neuralnet-ds

# Add core dependencies
uv add "torch>=2.0.0"
uv add "deepspeed>=0.12.0"

# Optional: Weights & Biases for experiment tracking
uv add "wandb"

# Development dependencies
uv add --dev "black" "isort" "flake8"
```

### 2. Project Structure

Create the following directory structure:

```
basic-neuralnet-ds/
├── train_ds.py            # Your training script
├── ds_config.json         # DeepSpeed configuration
├── requirements.txt       # Generated by uv
└── README.md             # This file
```

### 3. DeepSpeed Configuration

:::warning Two configs ship here, and the script uses the *other* one
`train_ds_enhanced.py` loads **`ds_config_fp32.json`**, not the
`ds_config.json` shown below. The difference is `fp16.enabled`, and it matters
more than it looks:

For a problem this small — fitting `y = 2x + 1`, two learnable parameters — the
gradients are tiny, and FP16's limited range lets them **underflow to zero**.
DeepSpeed's loss scaler cannot recover them. Training then runs to completion,
prints a falling-looking loss, and never updates the parameters. It fails
*quietly*, which is the worst way to fail while learning a new framework.

Both files ship in this folder, so if you cloned the repository there is
nothing to create. The walkthrough below writes them out only because it is
also usable as a from-scratch setup on a fresh pod.
:::

Create `ds_config_fp32.json` — **this is the one the script loads**:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.01,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0
    }
  },
  "fp16": {
    "enabled": false
  },
  "gradient_clipping": 1.0
}
```

Note the learning rate is `0.01` here against `1e-3` in the FP16 file below.
That is not a typo: with two parameters and no precision loss to fight, the
larger step converges in seconds.

If you skip this file, `train_ds_enhanced.py` fails immediately at
`deepspeed.initialize(..., config="ds_config_fp32.json")` with a
file-not-found error — which is the good outcome, because the alternative
(silently falling back to FP16) is the run that trains nothing.

And `ds_config.json`, the FP16 variant — kept because FP16 is what you *would*
enable on any real model, and every later example does:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

### 4. Add Your Training Script

Copy your `train_ds.py` script to the project directory.

### 5. (Optional) Configure Weights & Biases

If you want to track experiments with W&B:

```bash
# Get your API key from https://wandb.ai/authorize
export WANDB_API_KEY="your_api_key_here"
```

**Note**: The script will work perfectly fine without W&B configured. It will simply show a helpful message and continue training.

## Running the Training

### Basic Training (without W&B)

```bash
# Single GPU
uv run deepspeed --num_gpus=1 train_ds.py

# Multi-GPU (2 GPUs)
uv run deepspeed --num_gpus=2 train_ds.py

# Multi-GPU (4 GPUs)
uv run deepspeed --num_gpus=4 train_ds.py
```

### Training with Weights & Biases Tracking

```bash
# Set your W&B API key
export WANDB_API_KEY="your_api_key_here"

# Run training (single GPU)
uv run deepspeed --num_gpus=1 train_ds.py

# Or multi-GPU
uv run deepspeed --num_gpus=4 train_ds.py
```

### Multi-Node Training

```bash
# For multi-node training
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 --master_addr="10.0.0.1" train_ds.py
```

### With Explicit Config File

```bash
uv run deepspeed --num_gpus=1 train_ds.py --deepspeed --deepspeed_config ds_config.json
```

## Configuration Options

### Model Settings

- **Model Architecture**: Simple Linear Layer (1 input → 1 output)
- **Task**: Linear regression y = 2x + 1
- **Dataset**: Synthetic data (1000 samples, randomly generated)

### Training Hyperparameters

- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Epochs**: 30
- **Batch Size**: 32 per device
- **Gradient Accumulation**: 1 step
- **Loss Function**: MSE (Mean Squared Error)

### Memory Optimization

- **Mixed Precision**: FP16
- **Train Batch Size**: 32
- **Micro Batch Size**: 32 per GPU

## Understanding the Training Script

The `train_ds.py` script demonstrates core DeepSpeed concepts:

### 1. Model Definition (train_ds.py:9-22)
```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

### 2. DeepSpeed Initialization (train_ds.py:42-46)
```python
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

### 3. Training Loop (train_ds.py:53-65)
The training loop uses DeepSpeed's `backward()` and `step()` methods instead of standard PyTorch optimizer calls.

## Monitoring Training

### Training Output

The script provides comprehensive logging with detailed progress information:

```
================================================================================
🚀 Starting DeepSpeed Linear Regression Training
================================================================================

✅ Weights & Biases: Enabled
   - API key detected and configured

📊 Dataset Information:
   - Synthetic data: y = 2.0x + 1.0
   - Training samples: 1000
   - True Weight (W): 2.0
   - True Bias (b): 1.0

🎲 Initial Model Parameters (random):
   - Weight: 0.456789
   - Bias: -0.123456

⚙️  Initializing DeepSpeed...
✅ DeepSpeed initialized successfully

💻 Training Configuration:
   - Device: cuda
   - Batch size: 32
   - Total batches per epoch: 32
   - Number of epochs: 30
   - Model dtype: torch.float16

📈 W&B Run initialized: simple-linear-model
   - Project: deepspeed-linear-regression
   - View at: https://wandb.ai/your-username/deepspeed-linear-regression/runs/abc123

================================================================================
🏋️  Training Started...
================================================================================

Epoch  0/30 | Step   0 | Loss: 5.234567
Epoch  0/30 | Step  10 | Loss: 2.345678
Epoch  0/30 | Step  20 | Loss: 1.234567
Epoch  0/30 | Step  30 | Loss: 0.654321

📈 Epoch  0 Summary: Avg Loss = 1.567890
   Current Parameters: W = 1.678901, b = 0.789012
   Parameter Errors: ΔW = 0.321099, Δb = 0.210988

...

📈 Epoch 29 Summary: Avg Loss = 0.000123
   Current Parameters: W = 1.999876, b = 1.000234
   Parameter Errors: ΔW = 0.000124, Δb = 0.000234

================================================================================
✅ Training Completed!
================================================================================

📊 Training Summary:
   - Initial Loss: 1.567890
   - Final Loss: 0.000123
   - Loss Reduction: 99.99%

🎯 Final Model Parameters:
   - Learned Weight: 1.999876
   - Learned Bias: 1.000234

🎓 Ground Truth Parameters:
   - True Weight: 2.000000
   - True Bias: 1.000000

📏 Parameter Estimation Errors:
   - Weight Error: 0.000124 (0.01%)
   - Bias Error: 0.000234 (0.02%)

🏆 Model Quality Assessment:
   ✨ Excellent! Parameters match ground truth within 1% error

📊 W&B Summary logged
   - View results at: https://wandb.ai/your-username/deepspeed-linear-regression/runs/abc123
   - W&B run finished successfully

================================================================================
🎉 Training Script Finished Successfully!
================================================================================
```

### GPU Monitoring

Watch GPU usage during training:

```bash
watch -n 0.1 nvidia-smi
```

### Training Metrics Tracked

The script automatically tracks and displays:

**During Training:**
- Step-by-step loss (every 10 steps)
- Epoch summaries (every 5 epochs)
- Current parameter estimates
- Parameter errors vs. ground truth

**Final Summary:**
- Total loss reduction
- Learned parameters vs. true parameters
- Absolute and percentage errors
- Quality assessment (Excellent/Good/Fair/Poor)

### Weights & Biases Dashboard

When W&B is enabled, you can view:

**Real-time Metrics:**
- `step_loss`: Loss at each logged step
- `epoch_avg_loss`: Average loss per epoch
- `learned_weight`: Weight parameter over time
- `learned_bias`: Bias parameter over time
- `weight_error`: Absolute error in weight
- `bias_error`: Absolute error in bias
- `weight_error_pct`: Percentage error in weight
- `bias_error_pct`: Percentage error in bias

**Final Summary:**
- Loss reduction percentage
- Final learned parameters
- Final parameter errors
- Quality score

**Project Info:**
- Project: `deepspeed-linear-regression`
- Run name: `simple-linear-model`
- Access at: https://wandb.ai/your-username/deepspeed-linear-regression

## Model Usage After Training

### Understanding the Trained Model

The model learns the linear relationship y = 2x + 1. After training:

```python
import torch
from train_ds import SimpleModel

# Load model
model = SimpleModel()
# Load trained weights if saved

# Test inference
x_test = torch.tensor([[5.0]])
y_pred = model(x_test)
print(f"Input: {x_test.item()}, Predicted: {y_pred.item()}, Expected: {2*5+1}")
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```json
// Reduce batch size in ds_config.json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 16
}
```

#### DeepSpeed Installation Issues
```bash
# Install DeepSpeed with specific CUDA version
uv add "deepspeed>=0.12.0" --extra-index-url https://download.pytorch.org/whl/cu118

# Or build from source
DS_BUILD_OPS=1 uv add "deepspeed>=0.12.0"
```

#### FP16 Training Errors
```json
// Disable FP16 if your GPU doesn't support it
{
  "fp16": {
    "enabled": false
  }
}
```

#### Multi-GPU Training Not Working
```bash
# Check GPU availability
nvidia-smi

# Verify NCCL setup
export NCCL_DEBUG=INFO

# Test with single GPU first
uv run deepspeed --num_gpus=1 train_ds.py
```

#### Weights & Biases Issues

**W&B not installed:**
```bash
# Install wandb
uv add "wandb"

# Or with pip
uv pip install wandb
```

**W&B login issues:**
```bash
# Get your API key from https://wandb.ai/authorize
export WANDB_API_KEY="your_api_key_here"

# Verify it's set
echo $WANDB_API_KEY

# Or login interactively
wandb login
```

**Script works without W&B:**
The script is designed to never crash if W&B is not configured. You'll see one of these messages:

```
📊 Weights & Biases: Not installed
   - To enable tracking: pip install wandb
   - Then: export WANDB_API_KEY=your_api_key
```

Or:

```
📊 Weights & Biases: Not configured
   - To enable: export WANDB_API_KEY=your_api_key
   - To install: pip install wandb
```

Training will continue normally without W&B tracking.

### Performance Optimization

#### For Better Memory Usage
- Reduce `train_micro_batch_size_per_gpu`
- Increase `gradient_accumulation_steps`
- Disable `fp16` if not needed

#### For Faster Training
- Use multiple GPUs with DeepSpeed
- Enable `fp16` for compatible hardware
- Increase batch size if memory allows

## System Requirements

### Minimum Requirements
- **GPU**: 1x GTX 1080 Ti (11GB VRAM) or equivalent
- **RAM**: 8GB system RAM
- **Storage**: 5GB free space
- **CUDA**: 11.1+

### Recommended Setup
- **GPU**: 1x RTX 4090 (24GB) or A100
- **RAM**: 16GB system RAM
- **Storage**: 20GB SSD
- **Network**: Not required for single-node training

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Microsoft for DeepSpeed optimization framework
- PyTorch team for the deep learning framework
- The open-source ML community

---

**Note**: This training example is designed for educational purposes to demonstrate DeepSpeed integration with minimal complexity. It's an ideal starting point for learning distributed training concepts.

---

## Renting a GPU on RunPod (with auto-shutdown)

There is no SLURM on RunPod, so the pod lifecycle is driven by API instead —
including shutting it down.

```bash
export RUNPOD_API_KEY=...     # https://console.runpod.io/user/settings

uv run runpod/runpod_ctl.py recommend 01_basics/01_neuralnet
uv run runpod/runpod_ctl.py run 01_basics/01_neuralnet \
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

This example is sized in `runpod/runpod_ctl.py` as **6 GB VRAM, 1 GPU(s),
20 GB disk**.

The pod is **never given `RUNPOD_API_KEY`** — putting a spending credential on
rented hardware would be the wrong trade, so termination is driven from your
machine. See [SECURITY.md](../../SECURITY.md).
