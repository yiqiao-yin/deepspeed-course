# DeepSpeed RNN Training with LSTM

A comprehensive example of training LSTM (Recurrent Neural Network) models using DeepSpeed for distributed training optimization.

## Overview

This project demonstrates how to:
- Set up a Python environment using `uv` (ultrafast Python package manager)
- Train an LSTM model on synthetic time series data
- Use DeepSpeed for distributed training and memory optimization
- Configure gradient clipping, mixed precision (FP16), and ZeRO optimization

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
uv init proj
cd proj
```

### 3. Install Dependencies

```bash
# Add core dependencies
uv add "torch>=2.0.0" "deepspeed>=0.12.0" "tensorboard>=2.14.0" "numpy>=1.24.0"

# Add optional dependencies for visualization and analysis
uv add "matplotlib>=3.7.0" "tqdm>=4.65.0" "scipy>=1.10.0"

# Add development tools (optional)
uv add --dev "black" "isort" "pytest" "jupyter" "ipython"
```

### 4. Add Training Files

Create the following two files in your project directory:

#### `train_rnn_deepspeed.py`
```python
# Copy the Python training script from the artifacts above
```

#### `ds_config_rnn.json`
```json
# Copy the DeepSpeed configuration from the artifacts above
```

### 5. Run Training

```bash
# Single GPU training
uv run deepspeed train_rnn_deepspeed.py

# Multi-GPU training (specify number of GPUs)
uv run deepspeed --num_gpus=2 train_rnn_deepspeed.py

# Training with custom config file
uv run deepspeed --deepspeed_config=ds_config_rnn.json train_rnn_deepspeed.py
```

## Project Structure

```
proj/
├── .python-version          # Python version specification
├── pyproject.toml           # Project dependencies and metadata
├── train_rnn_deepspeed.py   # Main training script
├── ds_config_rnn.json       # DeepSpeed configuration
├── README.md                # This file
└── .venv/                   # Virtual environment (auto-created)
```

## Model Architecture

The project uses an LSTM-based architecture with the following features:

- **Input**: Time series sequences (50 timesteps)
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Task**: Predicting next value in multi-frequency sine wave
- **Output**: Single continuous value

### Model Parameters
- Input size: 1 (univariate time series)
- Hidden size: 64
- Number of layers: 2
- Sequence length: 50
- Dropout: 0.2 (between LSTM layers)

## DeepSpeed Configuration

The configuration includes several optimizations:

### Batch Configuration
- **Train batch size**: 128 (total across all GPUs)
- **Micro batch size per GPU**: 32
- **Gradient accumulation steps**: 2

### Optimization Features
- **Mixed Precision (FP16)**: Enabled with adaptive loss scaling
- **ZeRO Stage 2**: Memory optimization for parameters and gradients
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
- **Sequence length**: 50 timesteps
- **Prediction horizon**: 1 step ahead

### Training Process
- **Epochs**: 50
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning rate**: 5e-4 with warmup
- **Loss function**: Mean Squared Error (MSE)

## Monitoring Training

The script provides detailed logging:

```
Training LSTM model on cuda
Model parameters: 33,921
Epoch  0 | Step   0 | Loss: 2.454123
Epoch  0 | Step  20 | Loss: 1.892345
...
Epoch  0 | Average Loss: 1.234567
--------------------------------------------------
```

### TensorBoard (Optional)
To add TensorBoard logging, modify the training script:

```python
from torch.utils.tensorboard import SummaryWriter

# In main() function
writer = SummaryWriter('runs/lstm_experiment')
writer.add_scalar('Loss/Train', loss.item(), step)
```

Then run: `uv run tensorboard --logdir=runs`

## Troubleshooting

### Common Issues

1. **Batch Size Mismatch**
   ```
   AssertionError: Check batch related parameters
   ```
   **Solution**: Ensure `train_batch_size = micro_batch_per_gpu × gradient_accumulation_steps × num_gpus`

2. **CUDA Out of Memory**
   **Solutions**:
   - Reduce `train_micro_batch_size_per_gpu` in config
   - Increase `gradient_accumulation_steps`
   - Enable ZeRO Stage 3 in config

3. **Gradient Explosion**
   **Solution**: Adjust `gradient_clipping` value in config (try 0.5 or 0.25)

### Performance Tuning

- **Reduce memory usage**: Lower batch size, enable ZeRO Stage 3
- **Improve convergence**: Adjust learning rate, add learning rate scheduling
- **Speed up training**: Increase batch size (if memory allows), tune gradient accumulation

## Advanced Usage

### Custom Data
Replace the `generate_sine_wave_data()` function with your own data loader:

```python
def get_custom_data_loader(file_path: str, batch_size: int):
    # Load your time series data
    # Return DataLoader with (sequences, targets)
    pass
```

### Multi-Node Training
For training across multiple machines:

```bash
# On each node
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 --master_addr=<IP> train_rnn_deepspeed.py
```

### Different RNN Architectures
Modify the `LSTMModel` class to use:
- GRU: Replace `nn.LSTM` with `nn.GRU`
- Bidirectional: Add `bidirectional=True`
- Attention: Add attention mechanisms

## Dependencies Reference

### Core Dependencies
- `torch>=2.0.0`: PyTorch framework
- `deepspeed>=0.12.0`: Distributed training optimization
- `tensorboard>=2.14.0`: Training visualization
- `numpy>=1.24.0`: Numerical computations

### Optional Dependencies
- `matplotlib>=3.7.0`: Plotting and visualization
- `tqdm>=4.65.0`: Progress bars
- `scipy>=1.10.0`: Scientific computing

### Development Tools
- `black`: Code formatting
- `isort`: Import sorting
- `pytest`: Testing framework
- `jupyter`: Interactive notebooks
- `ipython`: Enhanced Python shell

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
- [uv Documentation](https://docs.astral.sh/uv/)