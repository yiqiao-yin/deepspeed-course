# DeepSpeed RNN Time-Series Forecasting

A high-performance distributed training implementation for LSTM-based time-series forecasting using DeepSpeed and PyTorch. This project demonstrates scalable deep learning training on synthetic multi-variate time-series data with sine and cosine patterns.

## Features

- 🚀 **Distributed Training**: Multi-GPU and multi-node training with DeepSpeed ZeRO
- 📊 **Synthetic Data Generation**: Configurable sine/cosine time-series with realistic noise and trends
- 🧠 **LSTM Architecture**: Flexible bidirectional LSTM with customizable layers and hidden sizes
- 📈 **Real-time Monitoring**: TensorBoard integration for training visualization
- 💾 **Checkpointing**: Automatic model saving and resuming capabilities
- ⚡ **Memory Optimization**: BF16 mixed precision and gradient accumulation

## Quick Start

### 1. Project Setup

```bash
# Initialize new project with uv
uv init deepspeed-rnn-timeseries
cd deepspeed-rnn-timeseries

# Add core dependencies
uv add "torch>=2.0.0" "deepspeed>=0.12.0" "tensorboard>=2.14.0" "numpy>=1.24.0"

# Add optional dependencies for visualization and analysis
uv add "matplotlib>=3.7.0" "tqdm>=4.65.0" "scipy>=1.10.0"

# Add development tools (optional)
uv add --dev "black" "isort" "pytest" "jupyter" "ipython"
```

### 2. File Setup

Create the required files in your project directory:

- `ds_config_rnn.json` - DeepSpeed configuration
- `train_rnn_deepspeed.py` - Training script

```bash
# Create necessary directories
mkdir -p logs checkpoints
```

### 3. Basic Training

```bash
# Single GPU training
uv run python train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 100

# Multi-GPU training
uv run deepspeed --num_gpus=2 train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 100
```

## Understanding the Configuration

### DeepSpeed Configuration (`ds_config_rnn.json`)

The DeepSpeed configuration file controls distributed training optimization and memory management:

#### **Batch Size Management**
```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 8
}
```
- **train_batch_size**: Global effective batch size across all GPUs
- **train_micro_batch_size_per_gpu**: Actual batch size processed per GPU per step
- **gradient_accumulation_steps**: Number of micro-batches to accumulate before updating weights
- Formula: `train_batch_size = train_micro_batch_size_per_gpu × num_gpus × gradient_accumulation_steps`

#### **Optimization Strategy**
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```
- **AdamW**: Decoupled weight decay version of Adam, better for deep networks
- **Learning rate**: Conservative 0.001 starting point for stable RNN training
- **Weight decay**: 0.01 provides mild regularization to prevent overfitting

#### **Learning Rate Scheduling**
```json
{
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 100,
            "cos_min_lr": 1e-5
        }
    }
}
```
- **Warmup**: Gradually increases learning rate from 0 to 0.001 over 100 steps
- **Cosine decay**: Smoothly decreases learning rate to 1e-5 following cosine curve
- Critical for RNN stability - prevents early training instability and late-stage overfitting

#### **Memory Optimization**
```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "contiguous_gradients": true,
        "cpu_offload": false
    }
}
```
- **ZeRO Stage 2**: Partitions gradients and optimizer states across GPUs
- **Communication overlap**: Hides communication latency during computation
- **Contiguous gradients**: Reduces memory fragmentation for better performance

#### **Precision and Stability**
```json
{
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0
}
```
- **BF16**: Mixed precision training - faster computation with minimal accuracy loss
- **Gradient clipping**: Prevents exploding gradients (common in RNN training)

## Understanding the Training Script

### Core Components

#### **1. Synthetic Data Generation**
```python
def generate_synthetic_timeseries(n_samples=10000, n_features=3, sampling_rate=0.1, noise_level=0.05)
```

The script generates realistic time-series data with multiple components:

- **Multi-frequency patterns**: Different sine/cosine frequencies per feature (1.0, 1.5, 2.0 Hz)
- **Phase relationships**: Staggered phase shifts (0, π/4, π/2) create realistic cross-correlations
- **Trend components**: Linear trends (0.001 * t) simulate real-world drift
- **Seasonal patterns**: Combined sine and cosine waves create complex seasonality
- **Realistic noise**: Gaussian noise (5% level) mimics measurement uncertainty

This approach creates data similar to financial markets, sensor readings, or economic indicators.

#### **2. LSTM Architecture**
```python
class LSTMTimeSeriesModel(nn.Module)
```

**Key architectural decisions:**

- **Bidirectional option**: Can process sequences forward and backward for better context
- **Dropout layers**: Prevent overfitting between LSTM layers and in the final projection
- **Weight initialization**: 
  - Xavier uniform for input weights (prevents vanishing gradients)
  - Orthogonal for hidden weights (maintains gradient flow)
  - Forget gate bias = 1 (helps long-term memory retention)

**Model flow:**
1. Input: `(batch_size, sequence_length, input_features)`
2. LSTM layers: Process temporal dependencies
3. Take last output: Extract final hidden state
4. Linear projection: Map to output space
5. Output: `(batch_size, output_features)`

#### **3. Training Loop Integration**

**DeepSpeed initialization:**
```python
# Multi-GPU: deepspeed launcher handles config automatically
if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(args=args, model=model, training_data=train_loader.dataset)
else:
    # Single GPU: we provide config explicitly
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(args=args, model=model, training_data=train_loader.dataset, config=config_path)
```

**Key differences:**
- **Single GPU mode**: Script provides config file via `config` parameter
- **Multi-GPU mode**: DeepSpeed launcher (`deepspeed --num_gpus=N`) automatically handles config from `--deepspeed_config` argument
- **Automatic device placement**: DeepSpeed handles GPU allocation
- **Distributed data loading**: Splits batches across available GPUs
- **Gradient synchronization**: Automatically reduces gradients across devices
- **Memory optimization**: ZeRO stages reduce memory usage per GPU

#### **4. Monitoring and Evaluation**

**TensorBoard logging:**
- Training/validation loss curves
- Learning rate schedules
- Gradient norms and parameter distributions

**Checkpoint management:**
- Best model saving based on validation loss
- Periodic checkpoints every 10 epochs
- Resumable training from any checkpoint

## Usage Guide

### Basic Training Commands

```bash
# ===== SINGLE GPU TRAINING =====
uv run python train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 100

# ===== MULTI-GPU TRAINING =====
# 2 GPUs
uv run deepspeed --num_gpus=2 train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 100

# 4 GPUs with custom architecture
uv run deepspeed --num_gpus=4 train_rnn_deepspeed.py \
    --config ds_config_rnn.json \
    --hidden_size 256 \
    --num_layers 3 \
    --sequence_length 100 \
    --epochs 200
```

### Advanced Training Options

```bash
# ===== CUSTOM DATA PARAMETERS =====
uv run deepspeed --num_gpus=2 train_rnn_deepspeed.py \
    --config ds_config_rnn.json \
    --hidden_size 512 \
    --num_layers 4 \
    --sequence_length 200 \
    --data_samples 50000 \
    --epochs 300

# ===== QUICK TESTING =====
# Small model for rapid iteration
uv run python train_rnn_deepspeed.py \
    --config ds_config_rnn.json \
    --epochs 5 \
    --data_samples 1000 \
    --hidden_size 64 \
    --sequence_length 20

# ===== DISTRIBUTED MULTI-NODE TRAINING =====
# Node 0 (master):
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=0 \
    --master_addr="192.168.1.100" --master_port=29500 \
    train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 200

# Node 1:
uv run deepspeed --num_gpus=8 --num_nodes=2 --node_rank=1 \
    --master_addr="192.168.1.100" --master_port=29500 \
    train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 200
```

### Environment Configuration

```bash
# ===== GPU SELECTION =====
CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed --num_gpus=2 \
    train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 100

# ===== DEBUG MODE =====
NCCL_DEBUG=INFO uv run deepspeed --num_gpus=4 \
    train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 50

# ===== MEMORY DEBUGGING =====
CUDA_LAUNCH_BLOCKING=1 uv run python train_rnn_deepspeed.py \
    --config ds_config_rnn.json --epochs 10
```

### Monitoring and Analysis

```bash
# ===== START TENSORBOARD =====
# In a separate terminal
uv run tensorboard --logdir=./logs --port=6006
# View at: http://localhost:6006

# ===== CHECK TRAINING PROGRESS =====
# Watch log files in real-time
tail -f ./logs/rnn_timeseries_*/events.out.tfevents.*

# ===== ANALYZE CHECKPOINTS =====
# List available checkpoints
ls -la ./checkpoints/

# Check checkpoint contents
python -c "
import torch
checkpoint = torch.load('./checkpoints/epoch_50/mp_rank_00_model_states.pt')
print(f'Checkpoint keys: {list(checkpoint.keys())}')
"
```

## Project Structure

```
deepspeed-rnn-timeseries/
├── pyproject.toml              # uv project configuration
├── uv.lock                     # uv dependency lockfile
├── ds_config_rnn.json          # DeepSpeed configuration
├── train_rnn_deepspeed.py      # Main training script
├── logs/                       # TensorBoard logs
│   └── rnn_timeseries_128h_2l/ # Model-specific logs
├── checkpoints/                # Model checkpoints
│   ├── epoch_10/              # Periodic checkpoints
│   ├── epoch_20/
│   └── best_model_epoch_XX/   # Best model based on validation
└── README.md                  # This file
```

## Command Reference

### uv Project Management

```bash
# Show dependency tree
uv tree

# Update all dependencies
uv sync

# Export requirements for other systems
uv export --format requirements-txt --output-file requirements.txt

# Run with specific Python version (if needed)
uv run --python 3.11 python train_rnn_deepspeed.py --config ds_config_rnn.json

# Or for multi-GPU
uv run --python 3.11 deepspeed --num_gpus=2 train_rnn_deepspeed.py --deepspeed_config ds_config_rnn.json

# Add new dependencies
uv add "wandb"  # For experiment tracking
uv add "plotly" # For interactive plots
```

### Model Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--hidden_size` | LSTM hidden units | 128 | 32-1024 |
| `--num_layers` | LSTM layers | 2 | 1-8 |
| `--sequence_length` | Input sequence length | 50 | 10-500 |
| `--data_samples` | Synthetic data size | 20000 | 1000-100000 |
| `--epochs` | Training epochs | 100 | 1-1000 |

### Data Generation Parameters

The synthetic data generation is controlled by these parameters in the script:

- **n_samples**: Total time steps (default: 20,000)
- **n_features**: Number of time series (default: 3)
- **sampling_rate**: Time step size (default: 0.1)
- **noise_level**: Gaussian noise standard deviation (default: 0.05)

## Performance Optimization

### Memory Optimization

1. **Reduce batch size**: Lower `train_micro_batch_size_per_gpu`
2. **Enable CPU offloading**: Set `"cpu_offload": true` in DeepSpeed config
3. **Use ZeRO Stage 3**: For very large models
4. **Gradient checkpointing**: Trade compute for memory

### Speed Optimization

1. **Increase batch size**: Higher `train_micro_batch_size_per_gpu`
2. **Reduce precision**: Use FP16 instead of BF16 (if supported)
3. **Optimize data loading**: Increase `num_workers` in DataLoader
4. **Use SSD storage**: For faster checkpoint I/O

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
# Edit ds_config_rnn.json:
"train_micro_batch_size_per_gpu": 8,  # Reduce from 16

# Or enable CPU offloading
"cpu_offload": true
```

#### DeepSpeed Installation Issues
```bash
# Install with specific CUDA version
uv add "deepspeed>=0.12.0" --extra-index-url https://download.pytorch.org/whl/cu118

# Build from source if needed
uv add "deepspeed @ git+https://github.com/microsoft/DeepSpeed.git"
```

#### Multi-GPU Communication Issues
```bash
# Check NCCL
export NCCL_DEBUG=INFO

# Use different backend
export NCCL_SOCKET_IFNAME=eth0

# Force specific NCCL version
export NCCL_VERSION=2.18.1
```

#### Slow Training
```bash
# Profile training
uv run python -m torch.profiler train_rnn_deepspeed.py --config ds_config_rnn.json --epochs 1

# Check GPU utilization
nvidia-smi -l 1

# Monitor I/O
iostat -x 1
```

## Advanced Features

### Custom Data Integration

To use your own time-series data, modify the `generate_synthetic_timeseries` function:

```python
def load_custom_data(file_path: str) -> np.ndarray:
    """Load your custom time-series data"""
    # Replace with your data loading logic
    data = np.loadtxt(file_path, delimiter=',')
    return data
```

### Model Architecture Extensions

Add custom layers or attention mechanisms:

```python
class AttentionLSTM(LSTMTimeSeriesModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8
        )
```

### Experiment Tracking

Integrate with Weights & Biases:

```bash
uv add "wandb"

# Add to script:
import wandb
wandb.init(project="rnn-timeseries", config=args)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## Acknowledgments

- **DeepSpeed Team**: For the distributed training framework
- **PyTorch Team**: For the excellent deep learning library
- **Hugging Face**: For inspiration on training infrastructure
- **Time-Series Community**: For domain expertise and best practices

---

**Happy Training! 🚀**