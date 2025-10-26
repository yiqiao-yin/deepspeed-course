#!/usr/bin/env python3
# stock_delta_rnn.py

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# SimpleRNN model with PyTorch
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Get last time step output
        out = self.fc(out[:, -1, :])

        return out

def main():
    # Check for WANDB_API_KEY
    if 'WANDB_API_KEY' not in os.environ:
        print("Warning: WANDB_API_KEY not found in environment variables.")
        print("Please export your W&B API key with: export WANDB_API_KEY=your-api-key")
        use_wandb = False
    else:
        use_wandb = True
        print("W&B API key found. W&B tracking enabled.")

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="stock-delta-rnn",
            config={
                "ticker": "AAPL",
                "ma_periods": [14, 26, 50, 100, 200],
                "sequence_length": 60,
                "hidden_size": 50,
                "num_layers": 2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
            }
        )
        config = wandb.config
    else:
        # Define config manually if wandb is not enabled
        class Config:
            def __init__(self):
                self.ticker = "AAPL"
                self.ma_periods = [14, 26, 50, 100, 200]
                self.sequence_length = 60
                self.hidden_size = 50
                self.num_layers = 2
                self.learning_rate = 0.001
                self.batch_size = 32
                self.epochs = 50
        config = Config()

    # Set random seed
    set_seed(42)

    # Detect if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Download stock data
    print(f"Downloading {config.ticker} stock data")
    data = yf.download(config.ticker, start='2015-01-01', end='2025-09-01')
    print(f"Data shape: {data.shape}")

    # First, create a copy of the Close price column to work with
    close_prices = data['Close'].copy()

    # Create a new DataFrame to store all our calculations
    analysis_df = pd.DataFrame(index=data.index)
    analysis_df['Close'] = close_prices

    # Calculate moving averages
    for period in config.ma_periods:
        analysis_df[f'MA_{period}'] = close_prices.rolling(window=period).mean()

    # Calculate deltas
    for period in config.ma_periods:
        analysis_df[f'delta_{period}'] = analysis_df['Close'] - analysis_df[f'MA_{period}']

    # Calculate average delta
    delta_columns = [f'delta_{period}' for period in config.ma_periods]
    analysis_df['avg_delta'] = analysis_df[delta_columns].mean(axis=1)

    # Drop NaN values
    analysis_df = analysis_df.dropna()
    print(f"Data shape after dropping NaNs: {analysis_df.shape}")

    # Plot the time series
    plt.figure(figsize=(15, 10))

    # Plot 1: Stock price and MAs
    plt.subplot(3, 1, 1)
    plt.plot(analysis_df.index, analysis_df['Close'], label='Close Price')
    for period in config.ma_periods:
        plt.plot(analysis_df.index, analysis_df[f'MA_{period}'], label=f'MA_{period}')
    plt.title(f'{config.ticker} Stock Price and Moving Averages')
    plt.legend()

    # Plot 2: Individual deltas
    plt.subplot(3, 1, 2)
    for period in config.ma_periods:
        plt.plot(analysis_df.index, analysis_df[f'delta_{period}'], label=f'delta_{period}')
    plt.title('Individual Deltas (Price - MA)')
    plt.legend()

    # Plot 3: Average delta
    plt.subplot(3, 1, 3)
    plt.plot(analysis_df.index, analysis_df['avg_delta'], color='red')
    plt.title('Average Delta')
    plt.tight_layout()
    plt.savefig('time_series_plots.png')
    plt.close()

    if use_wandb:
        wandb.log({"time_series_plots": wandb.Image('time_series_plots.png')})

    # Check the distribution of average delta
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(analysis_df['avg_delta'], kde=True)
    plt.title('Distribution of Average Delta')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=analysis_df['avg_delta'])
    plt.title('Boxplot of Average Delta')
    plt.tight_layout()
    plt.savefig('distribution_plots.png')
    plt.close()

    if use_wandb:
        wandb.log({"distribution_plots": wandb.Image('distribution_plots.png')})

    # Prepare data for time series forecasting
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    avg_delta_scaled = scaler.fit_transform(analysis_df['avg_delta'].values.reshape(-1, 1))

    # Parameters for the sequence
    sequence_length = config.sequence_length

    # Create sequences
    X, y = create_sequences(avg_delta_scaled, sequence_length)

    # Split into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Initialize model
    input_size = 1  # Single feature (avg_delta)
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    output_size = 1  # Predict next value

    model = SimpleRNN(input_size, hidden_size, num_layers, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            # Reshape for RNN [batch, seq_len, features]
            batch_X = batch_X.view(-1, sequence_length, input_size)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.view(-1, sequence_length, input_size)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss
            })

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train_tensor.view(-1, sequence_length, input_size)).cpu().numpy()
        test_predict = model(X_test_tensor.view(-1, sequence_length, input_size)).cpu().numpy()

    # Inverse transform to get back to original scale
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')

    if use_wandb:
        wandb.log({
            "final_train_rmse": train_rmse,
            "final_test_rmse": test_rmse
        })

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    if use_wandb:
        wandb.log({"training_history": wandb.Image('training_history.png')})

    # Plot predictions vs actual
    # Create a dataframe with actual and predicted values
    train_data_idx = analysis_df.index[sequence_length:train_size+sequence_length]
    test_data_idx = analysis_df.index[train_size+sequence_length:len(avg_delta_scaled)]

    # Plot predictions
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_data_idx, y_train_inv, label='Actual (Train)')
    plt.plot(train_data_idx, train_predict, label='Predicted (Train)')
    plt.title('Average Delta: Training Data Prediction')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(test_data_idx, y_test_inv, label='Actual (Test)')
    plt.plot(test_data_idx, test_predict, label='Predicted (Test)')
    plt.title('Average Delta: Testing Data Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

    if use_wandb:
        wandb.log({"prediction_results": wandb.Image('prediction_results.png')})
        wandb.finish()

    print("Analysis complete. Check the generated plots.")

    # Save model
    torch.save(model.state_dict(), 'stock_delta_rnn_model.pth')
    print("Model saved to 'stock_delta_rnn_model.pth'")

if __name__ == "__main__":
    main()
