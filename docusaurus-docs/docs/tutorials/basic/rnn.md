---
sidebar_position: 4
---

# Basic RNN (LSTM)

Train an LSTM model for time series prediction using DeepSpeed with ZeRO-2 optimization.

## Overview

This example demonstrates:
- LSTM architecture with proper initialization
- ZeRO-2 memory optimization
- Gradient clipping for RNN stability
- Validation set and early stopping
- Optional W&B experiment tracking

**Task:** Multi-frequency sine wave prediction

## Quick Start

```bash
cd 03_basic_rnn

# Single GPU
deepspeed train_rnn_deepspeed.py

# Multi-GPU
deepspeed --num_gpus=2 train_rnn_deepspeed.py
```

## Model Architecture

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

        self._initialize_weights()
```

### Proper LSTM Initialization

```python
def _initialize_weights(self):
    for name, param in self.lstm.named_parameters():
        if 'weight_ih' in name:
            # Xavier for input-hidden weights
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            # Orthogonal for hidden-hidden weights
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            # Forget gate bias = 1.0 for better gradients
            param.data.fill_(0)
            n = param.size(0)
            param.data[n//4:n//2].fill_(1.0)
```

## DeepSpeed Configuration

```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 2,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 5e-4,
      "weight_decay": 1e-5
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-4,
      "warmup_num_steps": 100
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "gradient_clipping": 1.0
}
```

## Key Features

### Gradient Clipping

Essential for RNN stability:

```json
{
  "gradient_clipping": 1.0
}
```

Prevents gradient explosion common in recurrent networks.

### ZeRO-2 Optimization

Partitions gradients and optimizer states:

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```

### Learning Rate Warmup

Stabilizes early training:

```json
{
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_num_steps": 100
    }
  }
}
```

## Dataset

Synthetic multi-frequency sine wave:

```python
def generate_data(n_samples, seq_length=50):
    t = np.linspace(0, 4*np.pi, n_samples + seq_length)

    # Multi-frequency signal
    signal = (np.sin(0.5 * t) +
              0.5 * np.sin(2.0 * t) +
              0.3 * np.sin(5.0 * t))

    # Add noise
    signal += np.random.normal(0, 0.1, signal.shape)

    return create_sequences(signal, seq_length)
```

- Training samples: 8,000 sequences
- Validation samples: 2,000 sequences
- Sequence length: 50 timesteps

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Hidden Size | 64 |
| LSTM Layers | 2 |
| Dropout | 0.2 |
| Learning Rate | 5e-4 |
| Warmup Steps | 100 |
| Epochs | 50 |
| Early Stopping | 10 epochs |
| Batch Size | 128 total |

## Expected Results

```
Training Summary:
  - Initial Loss: 1.523456
  - Final Loss: 0.012345
  - Loss Reduction: 99.19%

Validation Summary:
  - Best Val Loss: 0.015678
  - Val Loss Reduction: 98.92%

Model Quality: Excellent! (MSE < 0.05)
```

## Monitoring

### Gradient Norms

Watch for stability:
- Healthy: 0.01 - 1.0
- Exploding: > 10 (clipping should prevent)
- Vanishing: < 0.001

### W&B Metrics

When enabled, tracks:
- Step-level: loss, gradient norm, learning rate
- Epoch-level: train/val loss averages
- Final: quality assessment

## Troubleshooting

### Gradient Explosion

```
Loss: inf or NaN
```

Solutions:
- Lower gradient clipping: `"gradient_clipping": 0.5`
- Reduce learning rate: `"lr": 1e-4`
- Check initialization

### CUDA OOM

Solutions:
- Reduce `train_micro_batch_size_per_gpu`
- Increase `gradient_accumulation_steps`
- Enable ZeRO Stage 3

### Poor Convergence

- Increase warmup steps
- Adjust learning rate
- Check data normalization

## Advanced Usage

### Custom Time Series Data

```python
def get_custom_data_loaders(file_path, batch_size):
    data = pd.read_csv(file_path)

    X_train, y_train = create_sequences(data, seq_length=50)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

### GRU Alternative

```python
self.rnn = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.2
)
```

## Next Steps

- [Stock Prediction](/docs/tutorials/intermediate/stock-prediction) - Real-world RNN application
- [Bayesian Neural Networks](/docs/tutorials/intermediate/bayesian-nn) - Uncertainty estimation
