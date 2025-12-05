---
sidebar_position: 2
---

# Stock Price Prediction

Train an RNN on real stock data from Yahoo Finance for price delta prediction.

## Overview

This example demonstrates:
- Fetching real market data with `yfinance`
- Feature engineering (moving averages, deltas)
- DeepSpeed distributed training
- Optional W&B experiment tracking

**Task:** Predict stock price changes (deltas)

## Quick Start

```bash
cd 04_intermediate_rnn_stock_data

# SLURM submission
sbatch run_deepspeed.sh

# Direct execution (2 GPUs)
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py
```

## Data Pipeline

### 1. Fetch Real Data

```python
import yfinance as yf

def fetch_stock_data(ticker='AAPL', period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df['Close'].values
```

### 2. Feature Engineering

```python
def create_features(prices):
    # Price deltas
    delta_1 = np.diff(prices, n=1)

    # Moving averages
    ma_5 = rolling_mean(prices, window=5)
    ma_20 = rolling_mean(prices, window=20)

    # Combine features
    features = np.stack([delta_1, ma_5, ma_20], axis=-1)
    return features
```

### 3. Sequence Creation

```python
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict next delta
    return np.array(X), np.array(y)
```

## Model Architecture

```python
class StockRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```

**Input features:**
- Price delta (day-over-day change)
- 5-day moving average
- 20-day moving average

## DeepSpeed Configuration

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3,
      "weight_decay": 1e-5
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Stock | AAPL |
| Period | 5 years |
| Sequence Length | 20 days |
| Hidden Size | 64 |
| RNN Layers | 2 |
| Learning Rate | 1e-3 |
| Epochs | 50 |
| Batch Size | 64 |

## Running Training

### With SLURM

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=stock_rnn

export WANDB_API_KEY="your_key"  # Optional

source ~/myenv/bin/activate
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py
```

### With W&B Tracking

```bash
export WANDB_API_KEY="your_api_key"
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py
```

## Expected Output

```
Fetching AAPL stock data (5 years)...
  Total samples: 1,250 trading days
  Training samples: 1,000
  Validation samples: 250

Training Progress:
  Epoch 10: Loss = 0.0045, Val Loss = 0.0052
  Epoch 20: Loss = 0.0032, Val Loss = 0.0038
  ...

Final Results:
  Best Val Loss: 0.0035
  Prediction correlation: 0.72
```

## Visualization

The script can generate predictions vs. actuals:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual_deltas, label='Actual', alpha=0.7)
plt.plot(predicted_deltas, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Stock Price Delta Prediction')
plt.savefig('predictions.png')
```

## Important Notes

### Financial Disclaimer

This is for **educational purposes only**. Stock prediction is extremely difficult and:
- Past performance doesn't guarantee future results
- Real trading requires risk management
- Market conditions change unpredictably

### Data Considerations

- Stock prices are non-stationary
- Using deltas (changes) is more stable than raw prices
- Consider log returns for better statistical properties
- Add more features for better predictions

## Improving the Model

### Additional Features

```python
# Technical indicators
rsi = compute_rsi(prices, window=14)
macd = compute_macd(prices)
bollinger = compute_bollinger_bands(prices)

# Volume features
volume_ma = rolling_mean(volume, window=10)
```

### Alternative Architectures

```python
# LSTM for better long-term memory
self.rnn = nn.LSTM(input_size, hidden_size, num_layers)

# Attention mechanism
self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
```

## Troubleshooting

### Data Download Fails

```python
# Try different ticker or period
df = yf.download('AAPL', start='2019-01-01', end='2024-01-01')

# Or use cached data
import pandas as pd
df = pd.read_csv('aapl_data.csv')
```

### Loss Not Decreasing

- Normalize input features
- Reduce learning rate
- Add more data preprocessing

## Next Steps

- [HuggingFace Integration](/docs/tutorials/huggingface/overview) - Large language models
- [GRPO Training](/docs/tutorials/huggingface/grpo-training) - Reinforcement learning
