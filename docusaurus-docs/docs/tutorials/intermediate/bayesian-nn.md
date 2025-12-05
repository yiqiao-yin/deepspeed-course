---
sidebar_position: 1
---

# Bayesian Neural Networks

Train Bayesian neural networks using parallel tempering MCMC for uncertainty estimation.

## Overview

This example demonstrates:
- Parallel tempering (replica exchange) MCMC
- Multi-GPU distributed Bayesian inference
- Temperature-based chain swaps
- Multimodal posterior exploration

**Task:** Bayesian inference for neural network weights

## Quick Start

```bash
cd 04_bayesian_neuralnet

# SLURM submission (2 GPUs)
sbatch run_deepspeed.sh

# Direct execution
deepspeed --num_gpus=2 parallel_tempering_mcmc.py
```

## What is Parallel Tempering?

Parallel tempering runs multiple MCMC chains at different "temperatures":
- **Cold chains** (T=1): Sample from the target posterior
- **Hot chains** (T>1): Explore more freely, escape local modes
- **Swaps**: Exchange states between chains to combine benefits

```
Temperature Ladder:

T=1.0  ●─────────●─────────●  (cold: accurate samples)
       │ swap?   │ swap?   │
T=2.0  ○─────────○─────────○  (warm: more exploration)
       │ swap?   │ swap?   │
T=4.0  ○─────────○─────────○  (hot: rapid mixing)
```

## Model Architecture

```python
class BayesianMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

## How It Works

### 1. Temperature Assignment

Each GPU runs a chain at a different temperature:

```python
def get_temperature(rank, num_gpus, max_temp=4.0):
    """Assign temperature based on GPU rank."""
    if num_gpus == 1:
        return 1.0
    return 1.0 + (max_temp - 1.0) * rank / (num_gpus - 1)

# Example with 4 GPUs:
# GPU 0: T=1.0 (cold - collect samples here)
# GPU 1: T=2.0
# GPU 2: T=3.0
# GPU 3: T=4.0 (hot - explore freely)
```

### 2. MCMC Sampling

Each chain performs Metropolis-Hastings updates:

```python
def mcmc_step(model, data, temperature):
    # Propose new parameters
    old_params = get_params(model)
    new_params = propose(old_params, step_size=0.01)

    # Compute acceptance probability
    old_log_prob = log_posterior(model, data) / temperature
    set_params(model, new_params)
    new_log_prob = log_posterior(model, data) / temperature

    # Accept/reject
    if np.log(np.random.random()) < new_log_prob - old_log_prob:
        return True  # Accept
    else:
        set_params(model, old_params)
        return False  # Reject
```

### 3. Replica Exchange

Periodically swap states between adjacent chains:

```python
def attempt_swap(chain_i, chain_j, temp_i, temp_j):
    # Compute swap acceptance probability
    log_prob_i = log_posterior(chain_i)
    log_prob_j = log_posterior(chain_j)

    # Metropolis criterion for swap
    delta = (1/temp_i - 1/temp_j) * (log_prob_j - log_prob_i)

    if np.log(np.random.random()) < delta:
        # Swap parameters between chains
        swap(chain_i, chain_j)
        return True
    return False
```

## Benefits of Parallel Tempering

1. **Escape local modes**: Hot chains can jump between modes
2. **Better mixing**: Information propagates through temperature ladder
3. **Uncertainty quantification**: Sample from full posterior
4. **Parallel efficiency**: Each GPU handles one chain

## DeepSpeed Configuration

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 2,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4
    }
  },
  "fp16": {
    "enabled": false
  }
}
```

**Note:** FP16 is disabled for numerical stability in MCMC.

## Running with SLURM

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=bayesian_nn

source ~/myenv/bin/activate
deepspeed --num_gpus=2 parallel_tempering_mcmc.py
```

## Expected Output

```
Parallel Tempering MCMC with 2 GPUs

GPU 0: Temperature = 1.00 (cold chain)
GPU 1: Temperature = 4.00 (hot chain)

Iteration 100:
  Chain 0 acceptance: 0.32
  Chain 1 acceptance: 0.45
  Swap attempts: 10, accepted: 3

Iteration 1000:
  Collected 500 posterior samples from cold chain
  Mean prediction uncertainty: 0.15

Final Results:
  Posterior mean predictions: [...]
  95% credible intervals: [...]
```

## Use Cases

- **Uncertainty estimation**: Get confidence intervals on predictions
- **Model selection**: Compare models via marginal likelihood
- **Robust predictions**: Average over parameter uncertainty
- **Scientific inference**: Proper uncertainty propagation

## Troubleshooting

### Low Acceptance Rate

- Reduce step size in proposals
- Increase temperature range
- Check log posterior computation

### Poor Mixing

- Add more temperatures (use more GPUs)
- Increase swap frequency
- Adjust temperature ladder spacing

## Next Steps

- [Stock Prediction](/docs/tutorials/intermediate/stock-prediction) - Real-world application
- [HuggingFace Overview](/docs/tutorials/huggingface/overview) - Large model training
