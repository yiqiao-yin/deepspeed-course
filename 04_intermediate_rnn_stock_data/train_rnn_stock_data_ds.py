#!/usr/bin/env python3
"""Enhanced Stock Price Delta RNN with DeepSpeed and W&B Integration.

This script trains a SimpleRNN on stock price deltas (price - moving averages)
using DeepSpeed for distributed training and Weights & Biases for tracking.

Features:
- Real stock data from Yahoo Finance (yfinance)
- Moving average analysis (14, 26, 50, 100, 200 periods)
- Delta calculation and prediction
- DeepSpeed ZeRO-2 optimization
- FP16 mixed precision training
- Comprehensive W&B tracking with visualizations
- Early stopping with validation set
- Gradient norm monitoring
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    # Imported locally so this helper stays self-contained and can be copied
    # between example scripts unchanged. Some of those scripts do not import
    # os/sys at module scope, so these are not always redundant.
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \"torch_adam\": true and "
              "fp16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  DeepSpeed compiles fused CUDA kernels at startup. Without a CUDA")
    print("  toolkit it aborts with a confusing CUDA_HOME error from inside")
    print("  torch's extension loader, so this check stops first.")
    print("\n  This example is small enough to run on CPU. Two config changes:")
    print('      "optimizer": {"type": "Adam", "params": {"torch_adam": true}}')
    print('      "fp16": {"enabled": false}')
    print("  then:  ALLOW_CPU=1 deepspeed --num_gpus=1 <script>.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  No GPU at all? These need none:")
    print("      ./tests/run_all.sh    # 130 logic checks, no GPU, no downloads")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py gpus --min-vram 24")
    print("      uv run runpod/runpod_ctl.py run 01_basic_neuralnet")
    print("\n" + bar + "\n")
    sys.exit(1)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleRNN(nn.Module):
    """Simple RNN model for stock delta prediction."""

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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize RNN and FC weights with Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=x.device, dtype=x.dtype)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Get last time step output
        out = self.fc(out[:, -1, :])

        return out


def download_and_prepare_stock_data(ticker, start_date, end_date, ma_periods, sequence_length):
    """
    Download stock data and prepare it for training.

    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, analysis_df)
    """
    print(f"\n📊 Downloading {ticker} stock data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    print(f"   Downloaded {len(data)} data points")

    # Prepare analysis DataFrame
    close_prices = data['Close'].copy()
    analysis_df = pd.DataFrame(index=data.index)
    analysis_df['Close'] = close_prices

    # Calculate moving averages
    print(f"\n📈 Calculating moving averages: {ma_periods}")
    for period in ma_periods:
        analysis_df[f'MA_{period}'] = close_prices.rolling(window=period).mean()

    # Calculate deltas
    for period in ma_periods:
        analysis_df[f'delta_{period}'] = analysis_df['Close'] - analysis_df[f'MA_{period}']

    # Calculate average delta
    delta_columns = [f'delta_{period}' for period in ma_periods]
    analysis_df['avg_delta'] = analysis_df[delta_columns].mean(axis=1)

    # Drop NaN values
    analysis_df = analysis_df.dropna()
    print(f"   Data shape after dropping NaNs: {analysis_df.shape}")

    # ------------------------------------------------------------------
    # Split FIRST, then scale. This ordering matters and is not cosmetic.
    #
    # MinMaxScaler.fit computes (x - min) / (max - min) over whatever it is
    # given. Fitting it on the full series before splitting would derive those
    # constants partly from the test period, so every training example would be
    # normalized using information from the future. That is look-ahead bias:
    # reported test error comes out optimistically biased, and the distortion
    # is worst exactly where it matters, around volatility spikes and regime
    # changes.
    #
    # The scaler is therefore fit on the TRAINING SLICE ONLY, and merely
    # applied (transform, never fit_transform) to validation and test.
    # ------------------------------------------------------------------
    avg_delta = analysis_df['avg_delta'].values.reshape(-1, 1)

    n_obs = len(avg_delta)
    train_end = int(n_obs * 0.7)
    val_end = int(n_obs * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(avg_delta[:train_end])   # fit here ONLY
    val_scaled = scaler.transform(avg_delta[train_end:val_end])  # transform only
    test_scaled = scaler.transform(avg_delta[val_end:])          # transform only

    # Values outside [0, 1] in validation/test are EXPECTED and correct: they
    # mean the model is seeing conditions more extreme than anything in the
    # training period, which is exactly what an honest split should surface.
    print(f"\n🔍 Scaler fit on training slice only (first {train_end} observations)")
    print(f"   Train range after scaling: "
          f"[{train_scaled.min():.3f}, {train_scaled.max():.3f}]")
    print(f"   Val range after scaling:   "
          f"[{val_scaled.min():.3f}, {val_scaled.max():.3f}]")
    print(f"   Test range after scaling:  "
          f"[{test_scaled.min():.3f}, {test_scaled.max():.3f}]")
    if test_scaled.min() < 0 or test_scaled.max() > 1:
        print("   Note: test values fall outside [0, 1]. This is expected — the "
              "test period exceeds the training range.")

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    # Build sequences WITHIN each split so no window straddles a boundary.
    # Building them before splitting would leak the tail of the training period
    # into the first `sequence_length` validation windows (and likewise for
    # test). The cost is losing `sequence_length` samples per split.
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    print(f"\n📦 Dataset splits (chronological, no shuffling):")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, analysis_df


def create_visualizations(analysis_df, ticker, ma_periods, use_wandb):
    """Create and save visualization plots."""
    print("\n📊 Creating visualizations...")

    # Plot 1: Time series
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(analysis_df.index, analysis_df['Close'], label='Close Price', linewidth=2)
    for period in ma_periods:
        plt.plot(analysis_df.index, analysis_df[f'MA_{period}'], label=f'MA_{period}', alpha=0.7)
    plt.title(f'{ticker} Stock Price and Moving Averages')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    for period in ma_periods:
        plt.plot(analysis_df.index, analysis_df[f'delta_{period}'], label=f'delta_{period}', alpha=0.7)
    plt.title('Individual Deltas (Price - MA)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(analysis_df.index, analysis_df['avg_delta'], color='red', linewidth=2)
    plt.title('Average Delta (Target Variable)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series_plots.png', dpi=150)
    plt.close()

    # Only log to wandb if it's initialized
    # if use_wandb and wandb.run is not None:
    #     wandb.log({"time_series_plots": wandb.Image('time_series_plots.png')})

    # Plot 2: Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(analysis_df['avg_delta'], kde=True, bins=50)
    plt.title('Distribution of Average Delta')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=analysis_df['avg_delta'])
    plt.title('Boxplot of Average Delta')
    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=150)
    plt.close()

    # Only log to wandb if it's initialized
    # if use_wandb and wandb.run is not None:
    #     wandb.log({"distribution_plots": wandb.Image('distribution_plots.png')})

    print("   ✅ Visualizations created and saved")


def main():
    """Enhanced stock price delta RNN training with DeepSpeed."""

    require_gpu()
    print("=" * 80)
    print("🚀 Starting Enhanced DeepSpeed Stock Price Delta RNN Training")
    print("=" * 80)

    # Check for W&B configuration
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = False

    if WANDB_AVAILABLE and wandb_api_key:
        try:
            wandb.login(key=wandb_api_key)
            use_wandb = True
            print(f"\n✅ Weights & Biases: Enabled")
        except Exception as e:
            print(f"\n⚠️  Weights & Biases: Login failed - {e}")
            use_wandb = False
    elif not WANDB_AVAILABLE:
        print(f"\n📊 Weights & Biases: Not installed")
        print(f"   - To enable tracking: pip install wandb")

    # Configuration
    ticker = "AAPL"
    ma_periods = [14, 26, 50, 100, 200]
    sequence_length = 60
    hidden_size = 50
    num_layers = 2
    total_epochs = 50
    start_date = '2015-01-01'
    end_date = '2025-09-01'

    print(f"\n📊 Configuration:")
    print(f"   - Ticker: {ticker}")
    print(f"   - MA Periods: {ma_periods}")
    print(f"   - Sequence Length: {sequence_length} days")
    print(f"   - Hidden Size: {hidden_size}")
    print(f"   - Num Layers: {num_layers}")
    print(f"   - Epochs: {total_epochs}")
    print(f"   - Date Range: {start_date} to {end_date}")

    # Set random seed
    set_seed(42)

    # Download and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, analysis_df = download_and_prepare_stock_data(
        ticker, start_date, end_date, ma_periods, sequence_length
    )

    # Initialize model
    input_size = 1
    output_size = 1

    model = SimpleRNN(input_size, hidden_size, num_layers, output_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🏗️  Model Architecture:")
    print(f"   - Type: SimpleRNN")
    print(f"   - Input size: {input_size}")
    print(f"   - Hidden size: {hidden_size}")
    print(f"   - Num layers: {num_layers}")
    print(f"   - Output size: {output_size}")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")

    # Initialize DeepSpeed
    print(f"\n⚙️  Initializing DeepSpeed...")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="train_rnn_stock_data_config.json"
    )
    print(f"✅ DeepSpeed initialized successfully")

    # Get batch size and create data loaders
    batch_size = model_engine.train_micro_batch_size_per_gpu()

    # Convert to tensors
    device = model_engine.device
    model_dtype = next(model_engine.module.parameters()).dtype

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n💻 Training Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Model dtype: {model_dtype}")
    print(f"   - Batch size per GPU: {batch_size}")
    print(f"   - Train batches per epoch: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    print(f"   - Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
    print(f"   - Gradient clipping: 1.0")
    print(f"   - Mixed precision: FP16 enabled")
    print(f"   - ZeRO optimization: Stage 2")

    # Initialize W&B run (only on rank 0)
    local_rank = model_engine.local_rank
    if use_wandb and local_rank == 0:
        wandb.init(
            project="stock-delta-rnn-deepspeed",
            name=f"rnn-{ticker}-deepspeed",
            config={
                "model": "SimpleRNN",
                "ticker": ticker,
                "ma_periods": ma_periods,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "epochs": total_epochs,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "framework": "DeepSpeed",
                "total_params": total_params,
                "zero_stage": 2,
                "fp16": True
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: stock-delta-rnn-deepspeed")
        print(f"   - View at: {wandb.run.url}")

    # Create visualizations (only on rank 0, after wandb.init)
    if local_rank == 0:
        create_visualizations(analysis_df, ticker, ma_periods, use_wandb)

    print(f"\n{'='*80}")
    print("🏋️  Enhanced Training Started...")
    print(f"{'='*80}\n")

    # Training tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    min_improvement = 1e-6

    loss_fn = nn.MSELoss()

    for epoch in range(total_epochs):
        # Training phase
        model_engine.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        epoch_grad_norms = []

        # Get current learning rate
        if lr_scheduler:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = 0.001

        print(f"\n📚 Epoch {epoch:3d}/{total_epochs} - Learning Rate: {current_lr:.6e}")

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            # Reshape for RNN [batch, seq_len, features]
            x_batch = x_batch.view(-1, sequence_length, input_size)

            outputs = model_engine(x_batch)
            loss = loss_fn(outputs, y_batch)

            # Check for NaN/Inf loss
            if not torch.isfinite(loss):
                print(f"   ⚠️  Warning: Non-finite loss at step {step}, skipping")
                continue

            model_engine.backward(loss)

            # Compute gradient norm
            total_norm = 0.0
            for p in model_engine.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if not torch.isfinite(torch.tensor(total_norm)):
                print(f"   ⚠️  Warning: Non-finite gradients at step {step}")
                model_engine.module.zero_grad()
                continue

            epoch_grad_norms.append(total_norm)
            model_engine.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if step % 10 == 0:
                print(f"   Step {step:3d} | Loss: {loss.item():.6f} | Grad Norm: {total_norm:.6f}")

                if use_wandb and local_rank == 0 and wandb.run is not None:
                    wandb.log({
                        "step_loss": loss.item(),
                        "gradient_norm": total_norm,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": step
                    })

        if num_batches == 0:
            print(f"\n❌ ERROR: All batches skipped!")
            break

        avg_train_loss = epoch_loss_sum / num_batches
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        train_losses.append(avg_train_loss)

        # Validation phase
        model_engine.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device).to(model_dtype)
                y_val = y_val.to(device).to(model_dtype)
                x_val = x_val.view(-1, sequence_length, input_size)

                outputs = model_engine(x_val)
                val_loss = loss_fn(outputs, y_val)

                if torch.isfinite(val_loss):
                    val_loss_sum += val_loss.item()
                    val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)

        # Print epoch summary
        print(f"\n📈 Epoch {epoch:3d} Summary:")
        print(f"   - Train Loss: {avg_train_loss:.6f}")
        print(f"   - Val Loss: {avg_val_loss:.6f}")
        print(f"   - Avg Grad Norm: {avg_grad_norm:.6f}")
        print(f"   - Learning Rate: {current_lr:.6e}")

        # Early stopping
        if avg_val_loss < best_val_loss - min_improvement:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"   ✅ New best validation loss! Patience reset.")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience_limit}")

        # Log to W&B (only on rank 0)
        if use_wandb and local_rank == 0 and wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "epoch_avg_grad_norm": avg_grad_norm,
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
                "patience": patience_counter
            })

        if patience_counter >= patience_limit:
            print(f"\n🛑 Early stopping triggered!")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break

        print("-" * 80)

    print(f"\n{'='*80}")
    print("✅ Training Completed!")
    print(f"{'='*80}\n")

    # Final evaluation on test set
    print("📊 Final Evaluation on Test Set...")
    model_engine.eval()
    test_loss_sum = 0.0
    test_batches = 0

    all_test_predictions = []
    all_test_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device).to(model_dtype)
            y_test = y_test.to(device).to(model_dtype)
            x_test = x_test.view(-1, sequence_length, input_size)

            outputs = model_engine(x_test)
            test_loss = loss_fn(outputs, y_test)

            if torch.isfinite(test_loss):
                test_loss_sum += test_loss.item()
                test_batches += 1

                all_test_predictions.append(outputs.cpu().numpy())
                all_test_targets.append(y_test.cpu().numpy())

    avg_test_loss = test_loss_sum / test_batches if test_batches > 0 else float('inf')

    # Concatenate all predictions
    test_predict = np.concatenate(all_test_predictions, axis=0)
    test_actual = np.concatenate(all_test_targets, axis=0)

    # Inverse transform
    test_predict_inv = scaler.inverse_transform(test_predict)
    test_actual_inv = scaler.inverse_transform(test_actual)

    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(test_actual_inv, test_predict_inv))

    # ------------------------------------------------------------------
    # Persistence baseline: predict tomorrow = today.
    #
    # An RMSE reported on its own carries no information. The target here is
    # avg_delta, which is smooth by construction (MA_200 moves by at most
    # (P_t - P_{t-200}) / 200 per day), so consecutive values are highly
    # correlated whether or not the model learned anything. A naive
    # "tomorrow equals today" predictor is therefore a strong baseline and a
    # necessary reference point.
    #
    # Theil's U = RMSE_model / RMSE_naive.
    #   U < 1  -> the model beats persistence and is adding value
    #   U >= 1 -> a one-line baseline does as well as the trained network
    # ------------------------------------------------------------------
    naive_pred = test_actual_inv[:-1]   # today's value as tomorrow's forecast
    naive_true = test_actual_inv[1:]
    naive_rmse = np.sqrt(mean_squared_error(naive_true, naive_pred))
    theil_u = test_rmse / naive_rmse if naive_rmse > 0 else float('inf')

    # Directional accuracy: for a trading signal the SIGN matters more than the
    # magnitude, and a model can have excellent RMSE while getting turning
    # points backwards. Compare against 0.5 (a coin flip).
    actual_delta = np.diff(test_actual_inv.ravel())
    predicted_delta = (test_predict_inv.ravel()[1:] - test_actual_inv.ravel()[:-1])
    directional_acc = float(np.mean(np.sign(actual_delta) == np.sign(predicted_delta)))

    print(f"\n📊 Final Results:")
    print(f"   - Initial Train Loss: {train_losses[0]:.6f}")
    print(f"   - Final Train Loss: {train_losses[-1]:.6f}")
    print(f"   - Best Val Loss: {best_val_loss:.6f}")
    print(f"   - Test Loss: {avg_test_loss:.6f}")
    print(f"   - Test RMSE: {test_rmse:.4f}")
    print(f"   - Epochs completed: {len(train_losses)}")

    print(f"\n📏 Versus baseline (this is the number that matters):")
    print(f"   - Model RMSE:            {test_rmse:.4f}")
    print(f"   - Naive RMSE (persist):  {naive_rmse:.4f}")
    print(f"   - Theil U:               {theil_u:.4f}  "
          f"({'model beats persistence' if theil_u < 1 else 'NO BETTER than persistence'})")
    print(f"   - Directional accuracy:  {directional_acc:.4f}  "
          f"({'better' if directional_acc > 0.5 else 'no better'} than a coin flip)")
    if theil_u >= 1.0:
        print("   ⚠️  Theil U >= 1 means the trained model does not beat a "
              "one-line baseline. Report this honestly rather than the raw RMSE.")

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=150)
    plt.close()

    # Log final results (only on rank 0)
    if use_wandb and local_rank == 0 and wandb.run is not None:
        wandb.log({
            "final/train_loss": train_losses[-1],
            "final/best_val_loss": best_val_loss,
            "final/test_loss": avg_test_loss,
            "final/test_rmse": test_rmse,
            "final/naive_rmse": naive_rmse,
            "final/theil_u": theil_u,
            "final/directional_accuracy": directional_acc,
            "final/epochs_completed": len(train_losses),
            # "training_history": wandb.Image('training_history.png')
        })

        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["test_rmse"] = test_rmse
        wandb.run.summary["naive_rmse"] = naive_rmse
        wandb.run.summary["theil_u"] = theil_u
        wandb.run.summary["directional_accuracy"] = directional_acc
        wandb.run.summary["test_loss"] = avg_test_loss

        print(f"\n📊 W&B Summary logged")
        print(f"   - View at: {wandb.run.url}")
        wandb.finish()

    print(f"\n{'='*80}")
    print("🎉 Stock Price Delta RNN Training Finished Successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
