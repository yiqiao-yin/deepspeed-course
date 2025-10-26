# Stock Price Delta RNN with DeepSpeed 📈

This example demonstrates intermediate RNN training for stock price delta prediction using DeepSpeed, W&B tracking, and SLURM job scheduling.

## Overview

This project trains a SimpleRNN model to predict stock price deltas (price - moving average) using historical stock data from Yahoo Finance. The model learns temporal patterns in the relationship between current prices and their moving averages across multiple time periods.

## What Does This Example Do?

1. **Downloads Stock Data**: Fetches AAPL stock data (2015-2025) via `yfinance`
2. **Calculates Moving Averages**: Computes MAs for [14, 26, 50, 100, 200] periods
3. **Computes Deltas**: Calculates price - MA for each period
4. **Predicts Average Delta**: Trains SimpleRNN to predict average delta across all periods
5. **Visualizes Results**: Generates time series, distribution, and prediction plots
6. **Tracks Experiments**: Logs all metrics and visualizations to Weights & Biases

## Prerequisites

- CUDA-capable GPU(s) recommended (stock data training benefits from GPU acceleration)
- Python 3.8+
- Linux/macOS (Windows with WSL2)
- Internet connection (required for downloading stock data via yfinance)

## Quick Start

### 1. Install uv (Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Restart terminal or source your shell config
```

### 2. Initialize Project

```bash
# Create and navigate to project directory
uv init stock-rnn-deepspeed
cd stock-rnn-deepspeed
```

### 3. Install Dependencies

```bash
# Add core dependencies
uv add "torch>=2.0.0" "deepspeed>=0.12.0" "numpy>=1.24.0"

# Add Weights & Biases for experiment tracking
uv add "wandb"

# Add stock data and visualization dependencies
uv add "yfinance>=0.2.0" "pandas>=2.0.0" "matplotlib>=3.7.0" "seaborn>=0.12.0"

# Add ML utilities
uv add "scikit-learn>=1.3.0"

# Add optional dependencies
uv add "scipy>=1.10.0" "tqdm>=4.65.0"

# Add development tools (optional)
uv add --dev "black" "isort" "pytest" "jupyter"
```

### 4. Configure Weights & Biases (Optional)

To enable experiment tracking with W&B:

```bash
# Set your W&B API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY=your_api_key_here

# Or configure it interactively
wandb login
```

**If you don't configure W&B**, the script will continue training without tracking (fully optional).

### 5. Add Training Files

Copy the following files to your project directory:
- `train_rnn_stock_data.py` - Original single-machine training script
- `train_rnn_stock_data_ds.py` - DeepSpeed-enhanced version with W&B
- `train_rnn_stock_data_config.json` - DeepSpeed configuration

### 6. Run Training

```bash
# Single GPU training
uv run deepspeed --num_gpus=1 train_rnn_stock_data_ds.py

# Multi-GPU training (recommended for ZeRO-2)
uv run deepspeed --num_gpus=2 train_rnn_stock_data_ds.py

# Multi-GPU with explicit config
uv run deepspeed --num_gpus=2 --deepspeed_config=train_rnn_stock_data_config.json train_rnn_stock_data_ds.py

# Run original script (without DeepSpeed)
uv run python train_rnn_stock_data.py
```

## Files in This Folder

```
04_intermediate_rnn_stock_data/
├── train_rnn_stock_data.py          # Original single-machine training script
├── train_rnn_stock_data_ds.py       # DeepSpeed-enhanced version with W&B
├── train_rnn_stock_data_config.json # DeepSpeed configuration (ZeRO-2 + FP16)
├── run_deepspeed.sh                 # SLURM batch script for CoreWeave
└── README.md                         # This file
```

### File Descriptions

**`train_rnn_stock_data.py`**
- Original working script for single-machine training
- Uses PyTorch with standard training loop
- SimpleRNN architecture (2 layers, 50 hidden units)
- 80/20 train/test split
- Optional W&B integration

**`train_rnn_stock_data_ds.py`** ⭐
- DeepSpeed-enhanced version (recommended for production)
- Based on `03_basic_rnn/train_rnn_deepspeed.py` pattern
- **Key Enhancements:**
  - ZeRO-2 distributed optimization
  - FP16 mixed precision training
  - 70/15/15 train/val/test split
  - Proper weight initialization (Xavier + Orthogonal)
  - Early stopping (patience=10)
  - Gradient norm monitoring with NaN/Inf detection
  - Comprehensive W&B logging
  - Time series and distribution visualizations
  - Test set evaluation with RMSE

**`train_rnn_stock_data_config.json`**
- DeepSpeed configuration based on `03_basic_rnn/ds_config_rnn.json`
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler:** WarmupLR (100 steps)
- **ZeRO Stage 2:** Gradient and optimizer state partitioning
- **FP16:** Mixed precision with dynamic loss scaling
- **Gradient Clipping:** 1.0 (RNN stability)
- **Batch Sizes:** 64 global, 32 per GPU

**`run_deepspeed.sh`**
- SLURM batch script for CoreWeave cluster
- **Resources:** 2 GPUs, 8 CPUs, 32GB RAM, 2hr runtime
- **Note:** Requires internet access for yfinance downloads

## DeepSpeed Integration

This example uses DeepSpeed for efficient distributed training:

### ZeRO-2 Optimization
```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "overlap_comm": true,
  "reduce_scatter": true,
  "contiguous_gradients": true
}
```
- **Stage 2:** Partitions gradients and optimizer states across GPUs
- **Benefits:** ~2x memory reduction, enables larger batch sizes
- **Overlap Communication:** Hides communication latency with computation

### FP16 Mixed Precision
```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "min_loss_scale": 1
}
```
- **Dynamic Loss Scaling:** Prevents underflow/overflow
- **Benefits:** ~2x speedup, ~50% memory reduction
- **RNN Considerations:** Gradient clipping (1.0) prevents exploding gradients

### DeepSpeed Training Loop
```python
# Initialize DeepSpeed engine
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config='train_rnn_stock_data_config.json'
)

# Training step
for batch_X, batch_y in train_loader:
    outputs = model_engine(batch_X)
    loss = criterion(outputs, batch_y)

    model_engine.backward(loss)  # DeepSpeed handles gradient scaling
    model_engine.step()          # DeepSpeed handles optimizer step
```

## Weights & Biases Integration

Comprehensive experiment tracking with optional graceful fallback:

### Setup W&B
```bash
# Get your API key from: https://wandb.ai/authorize
export WANDB_API_KEY=your-api-key-here
```

### What Gets Logged

**Step-Level Metrics (every batch):**
- Training loss
- Gradient norms (total, RNN, FC layers)
- Learning rate

**Epoch-Level Metrics:**
- Train/validation/test loss
- Train/validation/test RMSE
- Model checkpoints (best validation)

**Visualizations:**
- Time series plots (price, MAs, deltas)
- Distribution plots (histograms, boxplots)
- Training history (loss curves)
- Prediction results (train/test)

**System Metrics:**
- GPU utilization
- Memory usage
- Training throughput

### W&B Dashboard Example
```
Project: stock-delta-rnn-deepspeed
Run: stock_rnn_AAPL_<timestamp>

Metrics:
├── train/loss
├── val/loss
├── test/loss
├── train/rmse
├── val/rmse
├── test/rmse
├── gradients/total_norm
├── gradients/rnn_norm
├── gradients/fc_norm
└── learning_rate

Visualizations:
├── time_series_plots.png
├── distribution_plots.png
├── training_history.png
└── prediction_results.png
```

## SLURM Integration (CoreWeave Cluster)

### Quick Start with SLURM

1. **Navigate to folder:**
```bash
cd 04_intermediate_rnn_stock_data
```

2. **Configure W&B API key:**
```bash
nano run_deepspeed.sh
# Replace <ENTER_KEY_HERE> with your actual API key
```

3. **Submit job:**
```bash
sbatch run_deepspeed.sh
```

4. **Monitor job:**
```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/stock_rnn_<job_id>.out

# Check for errors
tail -f logs/stock_rnn_<job_id>.err
```

### SLURM Resource Allocation

```bash
#SBATCH --gres=gpu:2        # 2 GPUs for ZeRO-2
#SBATCH --cpus-per-task=8   # 8 CPUs for data processing
#SBATCH --mem=32G           # 32GB RAM (stock data + sequences)
#SBATCH --time=02:00:00     # 2 hour max runtime
```

**Why these resources?**
- **2 GPUs:** ZeRO-2 optimization benefits from multiple GPUs
- **8 CPUs:** Stock data download, MA calculation, sequence creation
- **32GB RAM:** Large sequence datasets (60 timesteps × thousands of samples)
- **2 hours:** Conservative estimate including data download + 50 epochs

### Job Output
```bash
==================================================
Job ID: 123456
Job Name: stock_rnn_ds
Node: node-001
GPUs: 2
Start Time: 2025-10-26 10:00:00
Working Directory: /workspace/04_intermediate_rnn_stock_data
==================================================
Starting DeepSpeed Stock RNN training...
Task: Stock price delta prediction
Ticker: AAPL (configurable in script)
Expected result: Test RMSE depends on market volatility
Note: Requires internet access for yfinance data download

[yfinance] Downloading AAPL data...
[INFO] Data shape: (2701, 6)
[INFO] Using device: cuda
[INFO] DeepSpeed initialized: ZeRO-2, FP16, 2 GPUs
...
Epoch 50/50, Train Loss: 0.0015, Val Loss: 0.0018
Final Test RMSE: 2.3456
==================================================
```

## Running Locally (Without SLURM)

### Single GPU
```bash
# Set W&B API key
export WANDB_API_KEY=your-api-key-here

# Run with DeepSpeed (single GPU) using uv
uv run deepspeed --num_gpus=1 train_rnn_stock_data_ds.py

# Or with standard deepspeed command (if installed globally)
deepspeed --num_gpus=1 train_rnn_stock_data_ds.py
```

### Multiple GPUs
```bash
# Run with DeepSpeed (2 GPUs) using uv
uv run deepspeed --num_gpus=2 train_rnn_stock_data_ds.py

# Or with standard deepspeed command
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py
```

### Without DeepSpeed (Original Script)
```bash
# Run original script (single machine, no DeepSpeed) using uv
uv run python train_rnn_stock_data.py

# Or with standard python command
python train_rnn_stock_data.py
```

## Model Architecture

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(SimpleRNN, self).__init__()

        # RNN layers with ReLU activation
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def _initialize_weights(self):
        """Proper initialization for RNN stability"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
```

**Architecture Details:**
- **Input:** Sequence of 60 timesteps × 1 feature (avg_delta)
- **RNN:** 2 layers, 50 hidden units, ReLU activation
- **Output:** Single value (next avg_delta prediction)
- **Total Parameters:** ~13,101 parameters
- **Initialization:** Xavier (input) + Orthogonal (hidden)

## Data Pipeline

### 1. Download Stock Data
```python
data = yf.download('AAPL', start='2015-01-01', end='2025-09-01')
# Shape: (2701, 6) - 10+ years of daily data
```

### 2. Calculate Moving Averages
```python
ma_periods = [14, 26, 50, 100, 200]
for period in ma_periods:
    analysis_df[f'MA_{period}'] = close_prices.rolling(window=period).mean()
```

### 3. Compute Deltas
```python
for period in ma_periods:
    analysis_df[f'delta_{period}'] = analysis_df['Close'] - analysis_df[f'MA_{period}']

# Average across all deltas
analysis_df['avg_delta'] = analysis_df[delta_columns].mean(axis=1)
```

### 4. Create Sequences
```python
sequence_length = 60  # 60 days lookback
X, y = create_sequences(avg_delta_scaled, sequence_length)
# X shape: (N, 60, 1) - 60 timesteps of avg_delta
# y shape: (N, 1) - next avg_delta value
```

### 5. Train/Val/Test Split
```python
# 70% train, 15% val, 15% test
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
```

## Expected Results

### Training Convergence
- **Typical convergence:** 20-30 epochs
- **Train loss:** ~0.001-0.003 (normalized scale)
- **Val loss:** ~0.002-0.004
- **Test RMSE:** 2-5 (depends on market volatility)

### Performance Metrics
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Train RMSE | 1.5-3.0 | Lower bound limited by market noise |
| Val RMSE | 2.0-4.0 | Should be close to train RMSE |
| Test RMSE | 2.0-5.0 | Varies with market conditions |
| Training Time | 5-10 min | 2 GPUs, FP16, 50 epochs |

### Gradient Norms
- **Total norm:** 0.5-2.0 (stable training)
- **RNN norm:** 0.3-1.5 (dominant component)
- **FC norm:** 0.1-0.5 (smaller contribution)
- **Warning:** Norms >10 indicate instability

## Hyperparameters

### Key Hyperparameters (Configurable)
```python
config = {
    "ticker": "AAPL",                    # Stock ticker
    "ma_periods": [14, 26, 50, 100, 200],  # MA periods
    "sequence_length": 60,                # Lookback window
    "hidden_size": 50,                    # RNN hidden units
    "num_layers": 2,                      # RNN layers
    "learning_rate": 0.001,               # Adam LR
    "batch_size": 32,                     # Per GPU
    "epochs": 50,                         # Training epochs
}
```

### DeepSpeed Config (JSON)
```json
{
  "train_batch_size": 64,               # Global batch size
  "train_micro_batch_size_per_gpu": 32, # Per GPU batch size
  "gradient_accumulation_steps": 1,     # Accumulation (1 = disabled)
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "weight_decay": 1e-5              # L2 regularization
    }
  },
  "gradient_clipping": 1.0              # RNN stability
}
```

## Troubleshooting

### Common Issues

**1. yfinance download fails:**
```bash
# Error: Connection timeout or rate limit
# Solution: Check internet connection, retry later
# Alternative: Use cached data or different data source
```

**2. Out of memory (OOM):**
```bash
# Reduce batch size in config
"train_micro_batch_size_per_gpu": 16  # Down from 32

# Or enable gradient accumulation
"gradient_accumulation_steps": 2      # Effective batch = 16 × 2 = 32
```

**3. NaN/Inf gradients:**
```bash
# Script automatically skips bad batches and logs warning
# Check W&B for gradient norm spikes
# If persistent: reduce learning rate or disable FP16
"fp16": {"enabled": false}
```

**4. W&B not logging:**
```bash
# Check API key is set
echo $WANDB_API_KEY

# Verify W&B is installed
pip install wandb

# Script will continue without W&B if WANDB_API_KEY not set
```

**5. SLURM job fails:**
```bash
# Check error log
cat logs/stock_rnn_<job_id>.err

# Common causes:
# - Virtual environment path incorrect
# - Missing dependencies (yfinance, wandb)
# - No internet access on compute node
```

## Project Structure

After following the Quick Start, your project should look like:

```
stock-rnn-deepspeed/
├── .python-version                  # Python version specification
├── pyproject.toml                   # Project dependencies and metadata (uv)
├── train_rnn_stock_data.py          # Original single-machine script
├── train_rnn_stock_data_ds.py       # DeepSpeed-enhanced script
├── train_rnn_stock_data_config.json # DeepSpeed configuration
├── README.md                        # Documentation
├── .venv/                           # Virtual environment (auto-created)
├── wandb/                           # W&B logs (auto-created)
├── time_series_plots.png            # Generated visualization
├── distribution_plots.png           # Generated visualization
├── training_history.png             # Generated visualization
├── prediction_results.png           # Generated visualization
└── stock_delta_rnn_model.pth        # Saved model checkpoint
```

## Dependencies Reference

### Core Dependencies
- `torch>=2.0.0`: PyTorch framework for deep learning
- `deepspeed>=0.12.0`: Distributed training optimization
- `numpy>=1.24.0`: Numerical computations
- `wandb`: Experiment tracking and visualization (optional)

### Stock Data & Visualization
- `yfinance>=0.2.0`: Yahoo Finance API for stock data download
- `pandas>=2.0.0`: Data manipulation and analysis
- `matplotlib>=3.7.0`: Plotting and visualization
- `seaborn>=0.12.0`: Statistical data visualization

### ML Utilities
- `scikit-learn>=1.3.0`: MinMaxScaler, train/test split, RMSE metrics

### Optional Dependencies
- `scipy>=1.10.0`: Scientific computing utilities
- `tqdm>=4.65.0`: Progress bars for training loops

### Development Tools
- `black`: Code formatting (PEP 8)
- `isort`: Import sorting
- `pytest`: Testing framework
- `jupyter`: Interactive notebooks for experimentation
- `ipython`: Enhanced Python shell

## Next Steps

### Experimentation Ideas

1. **Different Stocks:** Change ticker in script (e.g., "TSLA", "MSFT", "NVDA")
2. **Longer Sequences:** Increase sequence_length to 120 (quarterly patterns)
3. **More Features:** Add volume, volatility, RSI, MACD
4. **Deeper Models:** Increase hidden_size or num_layers
5. **LSTM Upgrade:** Replace SimpleRNN with LSTM or GRU
6. **Ensemble:** Train multiple models with different seeds

### Advanced Topics

- **Multi-stock prediction:** Train on multiple tickers simultaneously
- **Portfolio optimization:** Use predictions for trading strategies
- **Attention mechanisms:** Add attention layers for interpretability
- **Transfer learning:** Pre-train on multiple stocks, fine-tune on target
- **Real-time inference:** Deploy model with streaming data pipeline

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)
- [uv Documentation](https://docs.astral.sh/uv/)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [RNN Best Practices](https://arxiv.org/abs/1503.04069)

## Author

**Yiqiao Yin**
[LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)

---

**Need Help?** Check the main [DeepSpeed Course README](../README.md) for setup instructions and troubleshooting tips.
