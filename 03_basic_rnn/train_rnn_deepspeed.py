"""
DeepSpeed RNN Training Script for Time-Series Forecasting
Synthetic data generation using sine and cosine functions
"""

import os
import json
import logging
import argparse
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.ops.adam import FusedAdam

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Time series dataset for synthetic sine/cosine data"""
    
    def __init__(self, data: np.ndarray, sequence_length: int, prediction_horizon: int = 1):
        """
        Args:
            data: Time series data of shape (n_samples, n_features)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence
        x = self.data[idx:idx + self.sequence_length]
        # Target (next prediction_horizon steps)
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        return x, y


class LSTMTimeSeriesModel(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(
        self, 
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMTimeSeriesModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Determine LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Output projection layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_size)
        
        # Project to output space
        output = self.fc(last_output)  # (batch_size, output_size)
        
        return output


def generate_synthetic_timeseries(
    n_samples: int = 10000,
    n_features: int = 3,
    sampling_rate: float = 0.1,
    noise_level: float = 0.05,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic time series data with sine and cosine components
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features (time series)
        sampling_rate: Sampling rate for time steps
        noise_level: Level of Gaussian noise to add
        seed: Random seed for reproducibility
    
    Returns:
        Synthetic time series data of shape (n_samples, n_features)
    """
    np.random.seed(seed)
    
    # Time vector
    t = np.arange(n_samples) * sampling_rate
    
    data = np.zeros((n_samples, n_features))
    
    # Generate different frequency components for each feature
    frequencies = [1.0, 1.5, 2.0]  # Different frequencies for variety
    phases = [0, np.pi/4, np.pi/2]  # Different phase shifts
    
    for i in range(n_features):
        freq = frequencies[i % len(frequencies)]
        phase = phases[i % len(phases)]
        
        # Combine sine and cosine with trend and seasonality
        trend = 0.001 * t  # Slight upward trend
        seasonal = np.sin(2 * np.pi * freq * t + phase) + 0.5 * np.cos(2 * np.pi * freq * 0.5 * t)
        noise = np.random.normal(0, noise_level, n_samples)
        
        data[:, i] = trend + seasonal + noise
    
    return data


def create_data_loaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    sequence_length: int,
    batch_size: int,
    prediction_horizon: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = TimeSeriesDataset(train_data, sequence_length, prediction_horizon)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, prediction_horizon)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    model_engine,
    epoch: int,
    writer: SummaryWriter = None
) -> Dict[str, float]:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device (handled by DeepSpeed)
        data = data.to(model_engine.local_rank)
        target = target.squeeze(-1) if target.dim() > 2 else target  # Remove extra dims
        target = target.to(model_engine.local_rank)
        
        # Forward pass
        output = model_engine(data)
        loss = criterion(output, target)
        
        # Backward pass (DeepSpeed handles this)
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log to tensorboard
        if writer and batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        if batch_idx % 100 == 0:
            logger.info(
                f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                f'Loss: {loss.item():.6f}'
            )
    
    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.squeeze(-1) if target.dim() > 2 else target
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeepSpeed RNN Time Series Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--data_samples', type=int, default=20000, help='Number of data samples to generate')
    parser.add_argument('--config', type=str, default='ds_config_rnn.json', help='DeepSpeed config file (for single GPU)')
    
    # Add DeepSpeed arguments (this adds --deepspeed_config for multi-GPU)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Set up logging directory
    log_dir = f"./logs/rnn_timeseries_{args.hidden_size}h_{args.num_layers}l"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if args.local_rank <= 0 else None
    
    # Generate synthetic data
    logger.info("Generating synthetic time series data...")
    data = generate_synthetic_timeseries(
        n_samples=args.data_samples,
        n_features=3,
        sampling_rate=0.1,
        noise_level=0.05
    )
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")
    
    # Create model
    model = LSTMTimeSeriesModel(
        input_size=3,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=3,
        dropout=0.2,
        bidirectional=False
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        sequence_length=args.sequence_length,
        batch_size=16,  # Will be overridden by DeepSpeed config
        prediction_horizon=1
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Initialize DeepSpeed
    logger.info("Initializing DeepSpeed...")
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=train_loader.dataset,
        config=args.deepspeed_config if hasattr(args, 'deepspeed_config') else args.config
    )
    
    device = torch.device(f"cuda:{args.local_rank}") if args.local_rank >= 0 else torch.device("cuda")
    
    logger.info(f"Model initialized on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, model_engine, epoch, writer)
        
        # Validate (only on main process)
        if args.local_rank <= 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            logger.info(
                f'Epoch {epoch+1}/{args.epochs} - '
                f'Train Loss: {train_metrics["loss"]:.6f}, '
                f'Val Loss: {val_metrics["loss"]:.6f}'
            )
            
            # Log to tensorboard
            if writer:
                writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
                writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
                writer.add_scalar('Learning_Rate', model_engine.get_lr()[0], epoch)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_path = f"./checkpoints/best_model_epoch_{epoch+1}.pt"
                os.makedirs("./checkpoints", exist_ok=True)
                model_engine.save_checkpoint(save_dir="./checkpoints", tag=f"epoch_{epoch+1}")
                logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_engine.save_checkpoint(save_dir="./checkpoints", tag=f"epoch_{epoch+1}")
    
    # Final evaluation
    if args.local_rank <= 0:
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Generate sample predictions
        model.eval()
        with torch.no_grad():
            # Take a sample from validation set
            sample_data, sample_target = next(iter(val_loader))
            sample_data = sample_data[:5].to(device)  # First 5 samples
            sample_target = sample_target[:5].to(device)
            
            predictions = model(sample_data)
            
            logger.info("Sample predictions vs targets:")
            for i in range(5):
                pred = predictions[i].cpu().numpy()
                target = sample_target[i].cpu().numpy()
                logger.info(f"Sample {i+1} - Pred: {pred}, Target: {target}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()