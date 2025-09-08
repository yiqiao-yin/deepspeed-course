"""Train an LSTM model using DeepSpeed on time series data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import numpy as np


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply final linear layer
        output = self.fc(last_output)
        return output


def generate_sine_wave_data(num_samples: int = 10000, sequence_length: int = 50) -> tuple:
    """
    Generates synthetic sine wave time series data for training.
    
    Args:
        num_samples: Number of training samples
        sequence_length: Length of each input sequence
    
    Returns:
        Tuple of (input_sequences, target_values)
    """
    # Generate sine wave with noise
    total_length = num_samples + sequence_length
    t = np.linspace(0, 100, total_length)
    
    # Multi-frequency sine wave with noise
    signal = (np.sin(0.5 * t) + 
             0.5 * np.sin(2 * t) + 
             0.3 * np.sin(5 * t) + 
             0.1 * np.random.randn(total_length))
    
    # Create sequences
    X = []
    y = []
    
    for i in range(num_samples):
        # Input: sequence of length `sequence_length`
        seq = signal[i:i + sequence_length]
        # Target: next value after the sequence
        target = signal[i + sequence_length]
        
        X.append(seq)
        y.append(target)
    
    X = np.array(X).reshape(-1, sequence_length, 1)  # Add feature dimension
    y = np.array(y).reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)


def get_data_loader(batch_size: int, sequence_length: int = 50) -> DataLoader:
    """
    Generates time series dataset and returns a DataLoader.
    """
    X, y = generate_sine_wave_data(num_samples=8000, sequence_length=sequence_length)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    """
    Initializes the LSTM model with DeepSpeed, trains it on time series data,
    and prints loss at intervals.
    """
    # Model hyperparameters
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    sequence_length = 50
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_rnn.json"
    )

    data_loader = get_data_loader(
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        sequence_length=sequence_length
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(model_engine.module.parameters()).dtype

    print(f"Training LSTM model on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(50):
        epoch_loss = 0.0
        num_batches = 0
        
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            outputs = model_engine(x_batch)
            loss = nn.functional.mse_loss(outputs, y_batch)

            model_engine.backward(loss)
            model_engine.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            if step % 20 == 0:
                print(f"Epoch {epoch:2d} | Step {step:3d} | Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch:2d} | Average Loss: {avg_epoch_loss:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()