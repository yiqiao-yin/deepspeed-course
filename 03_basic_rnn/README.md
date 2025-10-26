# Enhanced DeepSpeed LSTM Training with Weights & Biases

A comprehensive example of training LSTM (Recurrent Neural Network) models using DeepSpeed for distributed training optimization with production-ready features and experiment tracking.

## Overview

This project demonstrates how to:
- Train an LSTM model on synthetic time series data
- Use DeepSpeed for distributed training and memory optimization
- **Track experiments with Weights & Biases (W&B)**
- Monitor training metrics in real-time
- Implement proper LSTM initialization (Xavier + Orthogonal)
- Use gradient clipping, mixed precision (FP16), and ZeRO-2 optimization
- Implement early stopping with validation set
- Assess model quality automatically

## Features

- 🔄 **LSTM Architecture**: 2-layer LSTM with 64 hidden units for time series prediction
- ⚡ **DeepSpeed Integration**: Multi-GPU distributed training with ZeRO-2
- 📊 **W&B Tracking**: Real-time experiment tracking and visualization
- 🎯 **Validation Set**: Proper train/val split for model evaluation
- 🛑 **Early Stopping**: Patience-based stopping to prevent overfitting
- 📈 **Gradient Monitoring**: Track gradient norms throughout training
- 🔧 **Proper Initialization**: Xavier for input weights, orthogonal for hidden weights
- 💪 **Production Ready**: Comprehensive error handling and logging

## Prerequisites

- CUDA-capable GPU(s) recommended
- Python 3.8+
- Linux/macOS (Windows with WSL2)

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
uv init rnn-deepspeed
cd rnn-deepspeed
```

### 3. Install Dependencies

```bash
# Add core dependencies
uv add "torch>=2.0.0" "deepspeed>=0.12.0" "numpy>=1.24.0"

# Add Weights & Biases for experiment tracking
uv add "wandb"

# Add optional dependencies
uv add "matplotlib>=3.7.0" "tqdm>=4.65.0" "scipy>=1.10.0"

# Add development tools (optional)
uv add --dev "black" "isort" "pytest" "jupyter"
```

### 4. Configure Weights & Biases (Optional)

To enable experiment tracking with W&B:

```bash
# Set your W&B API key
export WANDB_API_KEY=your_api_key_here

# Or configure it interactively
wandb login
```

**If you don't configure W&B**, the script will continue training without tracking (fully optional).

### 5. Add Training Files

Copy the following files to your project directory:
- `train_rnn_deepspeed.py` - Enhanced training script with W&B integration
- `ds_config_rnn.json` - DeepSpeed configuration

### 6. Run Training

```bash
# Single GPU training
uv run deepspeed train_rnn_deepspeed.py

# Multi-GPU training (specify number of GPUs)
uv run deepspeed --num_gpus=2 train_rnn_deepspeed.py

# Multi-GPU with explicit config
uv run deepspeed --num_gpus=4 --deepspeed_config=ds_config_rnn.json train_rnn_deepspeed.py
```

## Model Architecture

The project uses an enhanced LSTM-based architecture:

- **Input**: Time series sequences (50 timesteps)
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Task**: Predicting next value in multi-frequency sine wave
- **Output**: Single continuous value
- **Total Parameters**: ~34,000 trainable parameters

### Model Enhancements

1. **Proper Initialization**:
   - Xavier uniform for input-hidden weights
   - Orthogonal initialization for hidden-hidden weights
   - Forget gate bias initialized to 1.0 (LSTM trick for better gradients)
   - Xavier uniform for final FC layer

2. **Architecture Details**:
   - Input size: 1 (univariate time series)
   - Hidden size: 64
   - Number of layers: 2
   - Sequence length: 50 timesteps
   - Dropout: 0.2 (between LSTM layers)
   - Output: 1 (regression)

## DeepSpeed Configuration

The configuration includes several optimizations:

### Batch Configuration
- **Train batch size**: 128 (total across all GPUs)
- **Micro batch size per GPU**: 32
- **Gradient accumulation steps**: 2

### Optimization Features
- **Mixed Precision (FP16)**: Enabled with adaptive loss scaling
- **ZeRO Stage 2**: Memory optimization for gradients and optimizer states
- **Gradient Clipping**: 1.0 (essential for RNN stability)
- **Learning Rate Scheduling**: Warmup from 0 to 5e-4 over 100 steps

### Memory Optimization
- ZeRO-2 partitioning for gradients and optimizer states
- Overlapped communication for efficiency
- Contiguous gradients for better memory layout

## Training Details

### Dataset
- **Synthetic time series**: Multi-frequency sine waves with noise
- **Training samples**: 8,000 sequences
- **Validation samples**: 2,000 sequences
- **Sequence length**: 50 timesteps
- **Prediction horizon**: 1 step ahead
- **Signal components**:
  - Primary sine: frequency 0.5
  - Secondary sine: frequency 2.0
  - Tertiary sine: frequency 5.0
  - Gaussian noise: std 0.1

### Training Process
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam (lr=5e-4, weight_decay=1e-5)
- **Learning rate**: 5e-4 with 100-step warmup
- **Loss function**: Mean Squared Error (MSE)
- **Early stopping**: Patience of 10 epochs
- **Validation**: Evaluated every epoch

## Weights & Biases Integration

The enhanced script includes comprehensive W&B tracking:

### Automatic Detection
```python
# W&B is automatically detected if installed
# Set environment variable to enable:
export WANDB_API_KEY=your_api_key
```

### What Gets Tracked

**Step-level metrics (every 20 steps)**:
- Training loss
- Gradient norms
- Learning rate
- Current epoch
- Step number

**Epoch-level metrics**:
- Average training loss
- Average validation loss
- Average gradient norm
- Best training loss (so far)
- Best validation loss (so far)
- Patience counter
- Learning rate

**Final summary**:
- Total train loss reduction (%)
- Total validation loss reduction (%)
- Best losses achieved
- Model quality assessment
- Epochs completed

### W&B Dashboard Views

Once training starts, visit your W&B dashboard to see:
- **Loss curves**: Training and validation loss over time
- **Gradient tracking**: Monitor gradient norm stability
- **Learning rate**: Visualize warmup and schedule
- **Model comparison**: Compare multiple runs
- **Hyperparameter tracking**: All config logged automatically

Example W&B URL after training:
```
https://wandb.ai/your-username/deepspeed-rnn/runs/enhanced-lstm-timeseries
```

## Monitoring Training

### Enhanced Console Output

The script provides detailed logging with progress indicators:

```
================================================================================
🚀 Starting Enhanced DeepSpeed LSTM Training
================================================================================

✨ Enhancements in this version:
   1. Proper weight initialization (Xavier + orthogonal)
   2. Gradient norm monitoring
   3. Learning rate tracking
   4. Early stopping with patience
   5. Best model tracking
   6. Validation set evaluation
   7. Comprehensive logging with W&B support
   8. Model quality assessment

✅ Weights & Biases: Enabled
   - API key detected and configured

📊 Model Configuration:
   - Architecture: 2-layer LSTM
   - Hidden size: 64
   - Input size: 1 (univariate time series)
   - Output size: 1 (next value prediction)
   - Sequence length: 50 timesteps
   - Dropout: 0.2 (between LSTM layers)

📊 Model Parameters:
   - Total parameters: 33,921
   - Trainable parameters: 33,921

⚙️  Initializing DeepSpeed...
✅ DeepSpeed initialized successfully

💻 Training Configuration:
   - Device: cuda
   - Model dtype: torch.float16
   - Batch size per GPU: 32
   - Train batches per epoch: 250
   - Validation batches: 63
   - Total epochs: 50
   - Optimizer: Adam (lr=5e-4, weight_decay=1e-5)
   - LR schedule: Warmup (100 steps)
   - Gradient clipping: 1.0
   - Mixed precision: FP16 enabled
   - ZeRO optimization: Stage 2

📈 W&B Run initialized: enhanced-lstm-timeseries
   - Project: deepspeed-rnn
   - View at: https://wandb.ai/your-username/deepspeed-rnn/runs/xyz123

================================================================================
🏋️  Enhanced Training Started...
================================================================================

📚 Epoch   0/50 - Learning Rate: 5.000000e-05
   Step   0 | Loss: 2.345678 | Grad Norm: 0.456789
   Step  20 | Loss: 1.234567 | Grad Norm: 0.345678
   ...

📈 Epoch   0 Summary:
   - Train Loss: 1.523456
   - Val Loss: 1.456789
   - Avg Grad Norm: 0.398765
   - Learning Rate: 5.000000e-04
   ✅ New best training loss! Patience reset.
   🎯 New best validation loss: 1.456789
--------------------------------------------------------------------------------

...

📈 Epoch  49 Summary:
   - Train Loss: 0.012345
   - Val Loss: 0.015678
   - Avg Grad Norm: 0.123456
   - Learning Rate: 5.000000e-04
   ✅ New best training loss! Patience reset.
   🎯 New best validation loss: 0.015678

================================================================================
✅ Training Completed!
================================================================================

📊 Training Summary:
   - Initial Train Loss: 1.523456
   - Final Train Loss: 0.012345
   - Best Train Loss: 0.012345
   - Loss Reduction: 99.19%
   - Epochs completed: 50

📊 Validation Summary:
   - Initial Val Loss: 1.456789
   - Final Val Loss: 0.015678
   - Best Val Loss: 0.015678
   - Val Loss Reduction: 98.92%

🏆 Model Quality Assessment:
   ✨ Excellent! Model achieved MSE < 0.05 on validation set

💡 Note:
   - Task: Time series prediction (multi-frequency sine wave)
   - MSE Loss: Lower is better (perfect prediction = 0)
   - Training samples: 8,000 sequences
   - Validation samples: 2,000 sequences
   - Model is relatively small (for demonstration)

📊 W&B Summary logged
   - View results at: https://wandb.ai/your-username/deepspeed-rnn/runs/xyz123
   - W&B run finished successfully

================================================================================
🎉 Enhanced LSTM Training Script Finished Successfully!
================================================================================
```

## Model Quality Assessment

The script automatically assesses model quality based on validation MSE:

| Quality | Validation MSE | Description |
|---------|---------------|-------------|
| **Excellent** | < 0.05 | Outstanding prediction accuracy |
| **Good** | < 0.10 | Solid performance on time series |
| **Fair** | < 0.20 | Acceptable baseline |
| **Poor** | ≥ 0.20 | Consider adjustments |

## Troubleshooting

### Common Issues

#### 1. Batch Size Mismatch
```
AssertionError: Check batch related parameters
```
**Solution**: Ensure `train_batch_size = micro_batch_per_gpu × gradient_accumulation_steps × num_gpus`

Example for 2 GPUs:
```json
{
  "train_batch_size": 128,          // Total
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 2,  // 32 × 2 × 2 = 128 ✓
  "..."
}
```

#### 2. CUDA Out of Memory
**Solutions**:
- Reduce `train_micro_batch_size_per_gpu` in config (e.g., 32 → 16)
- Increase `gradient_accumulation_steps` (e.g., 2 → 4)
- Enable ZeRO Stage 3 for larger models:
  ```json
  {"zero_optimization": {"stage": 3}}
  ```

#### 3. Gradient Explosion
**Symptoms**:
```
Grad Norm: inf
Loss: 10000.000000
```
**Solution**: Gradient clipping should prevent this, but if it occurs:
- Reduce clipping threshold: `"gradient_clipping": 0.5`
- Lower learning rate: `"lr": 1e-4`
- Check weight initialization

#### 4. W&B Not Working
**Issue**: "Weights & Biases: Not configured"
**Solutions**:
```bash
# Set API key
export WANDB_API_KEY=your_key

# Or login interactively
wandb login

# Or install if missing
pip install wandb
```

**Note**: Script works without W&B - tracking is optional!

#### 5. Early Stopping Too Soon
**Issue**: Training stops before convergence
**Solution**: In `train_rnn_deepspeed.py`, adjust:
```python
patience_limit = 10  # Increase to 15 or 20
min_improvement = 1e-6  # Lower for more sensitivity
```

### Performance Tuning

**Reduce memory usage**:
- Lower batch size: `train_micro_batch_size_per_gpu: 16`
- Enable ZeRO Stage 3
- Reduce sequence length (if applicable)

**Improve convergence**:
- Adjust learning rate: try `5e-4`, `1e-3`, or `1e-4`
- Modify warmup steps: `100` → `200`
- Increase model capacity: `hidden_size: 128`

**Speed up training**:
- Increase batch size (if memory allows)
- Use more GPUs
- Tune gradient accumulation
- Disable W&B for fastest training (no overhead)

## Advanced Usage

### Custom Data

Replace the data generation function with your own:

```python
def get_custom_data_loaders(file_path: str, batch_size: int):
    """
    Load your own time series data.

    Returns:
        train_loader, val_loader with (sequences, targets)
        - sequences: [batch, seq_len, features]
        - targets: [batch, output_size]
    """
    # Load from CSV, HDF5, etc.
    data = pd.read_csv(file_path)

    # Create sequences
    X_train, y_train = create_sequences(data, sequence_length=50)
    X_val, y_val = create_sequences(data_val, sequence_length=50)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
```

### Multi-Node Training

For training across multiple machines:

```bash
# On node 0 (master)
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train_rnn_deepspeed.py

# On node 1
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=1 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train_rnn_deepspeed.py
```

### Different RNN Architectures

Modify the `LSTMModel` class:

**GRU instead of LSTM**:
```python
self.rnn = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.2
)
```

**Bidirectional LSTM**:
```python
self.lstm = nn.LSTM(..., bidirectional=True)
self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
```

**Add Attention Mechanism**:
```python
class AttentionLSTM(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.lstm = nn.LSTM(...)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Linear(hidden_size, output_size)
```

### Hyperparameter Sweeps with W&B

Create a sweep configuration:

```python
# sweep_config.yaml
program: train_rnn_deepspeed.py
method: bayes
metric:
  name: final/best_val_loss
  goal: minimize
parameters:
  hidden_size:
    values: [32, 64, 128]
  learning_rate:
    min: 1e-5
    max: 1e-3
  num_layers:
    values: [1, 2, 3]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

## Comparing to Basic Version

| Feature | Basic Version | Enhanced Version |
|---------|--------------|------------------|
| **Weight Init** | Default (random) | Xavier + Orthogonal |
| **Validation Set** | ❌ | ✅ Train/Val split |
| **Early Stopping** | ❌ | ✅ Patience-based |
| **Gradient Monitoring** | ❌ | ✅ Full tracking |
| **LR Tracking** | ❌ | ✅ Per-step logging |
| **W&B Integration** | ❌ | ✅ Full tracking |
| **Quality Assessment** | ❌ | ✅ Automatic |
| **Error Handling** | Minimal | Comprehensive |
| **Logging Detail** | Basic | Production-ready |
| **Best Model Tracking** | ❌ | ✅ Train & Val |
| **Progress Updates** | Every 20 steps | Every 20 steps + summary |
| **Expected MSE** | 0.05-0.15 | 0.01-0.05 (better) |

## Project Structure

```
rnn-deepspeed/
├── .python-version          # Python version specification
├── pyproject.toml           # Project dependencies and metadata
├── train_rnn_deepspeed.py   # Enhanced training script with W&B
├── ds_config_rnn.json       # DeepSpeed configuration
├── README.md                # This file
├── .venv/                   # Virtual environment (auto-created)
└── wandb/                   # W&B logs (auto-created)
```

## Dependencies Reference

### Core Dependencies
- `torch>=2.0.0`: PyTorch framework
- `deepspeed>=0.12.0`: Distributed training optimization
- `numpy>=1.24.0`: Numerical computations
- `wandb`: Experiment tracking and visualization (optional)

### Optional Dependencies
- `matplotlib>=3.7.0`: Plotting and visualization
- `tqdm>=4.65.0`: Progress bars
- `scipy>=1.10.0`: Scientific computing
- `pandas`: Data manipulation for custom datasets

### Development Tools
- `black`: Code formatting
- `isort`: Import sorting
- `pytest`: Testing framework
- `jupyter`: Interactive notebooks
- `ipython`: Enhanced Python shell

## Expected Results

With the enhanced script and proper configuration:

### Training Metrics
- **Initial Loss**: ~1.5-2.5 (MSE on noisy sine wave)
- **Final Loss**: ~0.01-0.05 (95-99% reduction)
- **Training Time**: ~5-10 minutes (single GPU, 50 epochs)
- **Convergence**: Typically 30-40 epochs
- **Quality**: Excellent (MSE < 0.05)

### Validation Metrics
- **Best Val Loss**: ~0.015-0.05 (should be close to train loss)
- **Generalization**: Good (validation close to training)
- **Overfitting**: Minimal (with dropout and early stopping)

## Tips for Best Results

1. **Start with defaults** - The provided configuration works well
2. **Monitor W&B dashboard** - Check for training instabilities early
3. **Watch gradient norms** - Should be < 1.0 (clipping ensures this)
4. **Check early stopping** - If triggered too early, increase patience
5. **Validate often** - Validation set helps catch overfitting
6. **Use FP16 carefully** - Helps memory but watch for numerical issues
7. **Scale up gradually** - Test on small data first, then scale

## License

This project is released under the MIT License. Feel free to use and modify as needed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Format code: `uv run black . && uv run isort .`
6. Submit a pull request

## References

- [DeepSpeed Documentation](https://deepspeed.readthedocs.io/)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [LSTM Initialization Best Practices](https://pytorch.org/docs/stable/nn.init.html)
- [uv Documentation](https://docs.astral.sh/uv/)

## Acknowledgments

- DeepSpeed team for the optimization framework
- Weights & Biases for experiment tracking platform
- PyTorch community for deep learning tools
- Sine wave data generation inspired by time series forecasting literature

---

**Note**: This enhanced version demonstrates production-ready practices for RNN training with DeepSpeed. The comprehensive logging, validation, and W&B integration make it ideal for both learning and research projects.
