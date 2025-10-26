"""Enhanced LSTM training script with DeepSpeed and Weights & Biases integration.

Improvements over basic version:
1. Comprehensive W&B tracking and visualization
2. Gradient norm monitoring
3. Learning rate tracking
4. Early stopping with patience
5. Best model tracking
6. Detailed logging and quality assessment
7. Production-ready error handling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import numpy as np
import os
import sys

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction with proper initialization.
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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM and FC weights with Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1 (LSTM trick)
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'fc.weight' in name:
                # Final FC layer
                nn.init.xavier_uniform_(param.data)

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


def generate_sine_wave_data(num_samples: int = 10000, sequence_length: int = 50, seed: int = 42) -> tuple:
    """
    Generates synthetic sine wave time series data for training.

    Args:
        num_samples: Number of training samples
        sequence_length: Length of each input sequence
        seed: Random seed for reproducibility

    Returns:
        Tuple of (input_sequences, target_values)
    """
    np.random.seed(seed)

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


def get_data_loaders(batch_size: int, sequence_length: int = 50) -> tuple:
    """
    Generates time series dataset and returns train/validation DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Generate training data
    X_train, y_train = generate_sine_wave_data(num_samples=8000, sequence_length=sequence_length, seed=42)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Generate validation data
    X_val, y_val = generate_sine_wave_data(num_samples=2000, sequence_length=sequence_length, seed=123)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def main() -> None:
    """
    Enhanced LSTM training with comprehensive W&B tracking and monitoring.
    """

    print("=" * 80)
    print("🚀 Starting Enhanced DeepSpeed LSTM Training")
    print("=" * 80)
    print("\n✨ Enhancements in this version:")
    print("   1. Proper weight initialization (Xavier + orthogonal)")
    print("   2. Gradient norm monitoring")
    print("   3. Learning rate tracking")
    print("   4. Early stopping with patience")
    print("   5. Best model tracking")
    print("   6. Validation set evaluation")
    print("   7. Comprehensive logging with W&B support")
    print("   8. Model quality assessment")

    # Check for Weights & Biases configuration
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = False

    if WANDB_AVAILABLE and wandb_api_key:
        try:
            wandb.login(key=wandb_api_key)
            use_wandb = True
            print(f"\n✅ Weights & Biases: Enabled")
            print(f"   - API key detected and configured")
        except Exception as e:
            print(f"\n⚠️  Weights & Biases: Login failed - {e}")
            print(f"   - Continuing without W&B tracking")
            use_wandb = False
    elif WANDB_AVAILABLE and not wandb_api_key:
        print(f"\n📊 Weights & Biases: Not configured")
        print(f"   - To enable: export WANDB_API_KEY=your_api_key")
    elif not WANDB_AVAILABLE:
        print(f"\n📊 Weights & Biases: Not installed")
        print(f"   - To enable tracking: pip install wandb")

    # Model hyperparameters
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    sequence_length = 50
    total_epochs = 50

    print(f"\n📊 Model Configuration:")
    print(f"   - Architecture: {num_layers}-layer LSTM")
    print(f"   - Hidden size: {hidden_size}")
    print(f"   - Input size: {input_size} (univariate time series)")
    print(f"   - Output size: {output_size} (next value prediction)")
    print(f"   - Sequence length: {sequence_length} timesteps")
    print(f"   - Dropout: 0.2 (between LSTM layers)")

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")

    print(f"\n⚙️  Initializing DeepSpeed...")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_rnn.json"
    )
    print(f"✅ DeepSpeed initialized successfully")

    # Get batch size and create data loaders
    batch_size = model_engine.train_micro_batch_size_per_gpu()
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        sequence_length=sequence_length
    )

    device = model_engine.device
    model_dtype = next(model_engine.module.parameters()).dtype

    print(f"\n💻 Training Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Model dtype: {model_dtype}")
    print(f"   - Batch size per GPU: {batch_size}")
    print(f"   - Train batches per epoch: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Total epochs: {total_epochs}")
    print(f"   - Optimizer: Adam (lr=5e-4, weight_decay=1e-5)")
    print(f"   - LR schedule: Warmup (100 steps)")
    print(f"   - Gradient clipping: 1.0")
    print(f"   - Mixed precision: FP16 enabled")
    print(f"   - ZeRO optimization: Stage 2")

    # Initialize W&B run if enabled
    if use_wandb:
        wandb.init(
            project="deepspeed-rnn",
            name="enhanced-lstm-timeseries",
            config={
                "model": "LSTM",
                "task": "time_series_prediction",
                "dataset": "synthetic_sine_wave",
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "sequence_length": sequence_length,
                "epochs": total_epochs,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "learning_rate": 5e-4,
                "weight_decay": 1e-5,
                "lr_schedule": "WarmupLR",
                "gradient_clipping": 1.0,
                "framework": "DeepSpeed",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "fp16": True,
                "zero_stage": 2,
                "enhancements": [
                    "xavier_init",
                    "orthogonal_init",
                    "gradient_monitoring",
                    "early_stopping",
                    "validation_set",
                    "lr_tracking"
                ]
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: deepspeed-rnn")
        print(f"   - View at: {wandb.run.url}")

    print(f"\n{'='*80}")
    print("🏋️  Enhanced Training Started...")
    print(f"{'='*80}\n")

    # Training tracking variables
    epoch_losses = []
    val_losses = []
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    min_improvement = 1e-6

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
            current_lr = 5e-4

        print(f"\n📚 Epoch {epoch:3d}/{total_epochs} - Learning Rate: {current_lr:.6e}")

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            outputs = model_engine(x_batch)
            loss = nn.functional.mse_loss(outputs, y_batch)

            # Check for NaN/Inf loss
            if not torch.isfinite(loss):
                print(f"   ⚠️  Warning: Non-finite loss detected at step {step}, skipping batch")
                continue

            model_engine.backward(loss)

            # Compute gradient norm before stepping
            total_norm = 0.0
            for p in model_engine.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Check for non-finite gradients
            if not torch.isfinite(torch.tensor(total_norm)):
                print(f"   ⚠️  Warning: Non-finite gradients detected at step {step}, skipping batch")
                model_engine.module.zero_grad()
                continue

            epoch_grad_norms.append(total_norm)

            model_engine.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if step % 20 == 0:
                print(f"   Step {step:3d} | Loss: {loss.item():.6f} | Grad Norm: {total_norm:.6f}")

                # Log to W&B if enabled
                if use_wandb:
                    wandb.log({
                        "step_loss": loss.item(),
                        "gradient_norm": total_norm,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": step
                    })

        # Calculate average training loss
        if num_batches == 0:
            print(f"\n❌ ERROR: All batches were skipped due to non-finite values!")
            break

        avg_train_loss = epoch_loss_sum / num_batches
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        epoch_losses.append(avg_train_loss)

        # Validation phase
        model_engine.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device).to(model_dtype)
                y_val = y_val.to(device).to(model_dtype)

                outputs = model_engine(x_val)
                val_loss = nn.functional.mse_loss(outputs, y_val)

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

        # Track best models
        if avg_train_loss < best_loss - min_improvement:
            best_loss = avg_train_loss
            patience_counter = 0
            print(f"   ✅ New best training loss! Patience reset.")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience_limit}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   🎯 New best validation loss: {best_val_loss:.6f}")

        # Log epoch metrics to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "epoch_avg_grad_norm": avg_grad_norm,
                "learning_rate": current_lr,
                "best_train_loss": best_loss,
                "best_val_loss": best_val_loss,
                "patience": patience_counter
            })

        # Early stopping check
        if patience_counter >= patience_limit:
            print(f"\n🛑 Early stopping triggered! No improvement for {patience_limit} epochs.")
            print(f"   Best training loss: {best_loss:.6f}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break

        print("-" * 80)

    print(f"\n{'='*80}")
    print("✅ Training Completed!")
    print(f"{'='*80}\n")

    # Check if we have training data
    if not epoch_losses:
        print(f"❌ CRITICAL: Training failed - no epochs completed successfully!")
        if use_wandb:
            wandb.finish()
        return

    # Calculate final statistics
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

    initial_val_loss = val_losses[0]
    final_val_loss = val_losses[-1]
    val_loss_reduction = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100

    print(f"📊 Training Summary:")
    print(f"   - Initial Train Loss: {initial_loss:.6f}")
    print(f"   - Final Train Loss: {final_loss:.6f}")
    print(f"   - Best Train Loss: {best_loss:.6f}")
    print(f"   - Loss Reduction: {loss_reduction:.2f}%")
    print(f"   - Epochs completed: {len(epoch_losses)}")

    print(f"\n📊 Validation Summary:")
    print(f"   - Initial Val Loss: {initial_val_loss:.6f}")
    print(f"   - Final Val Loss: {final_val_loss:.6f}")
    print(f"   - Best Val Loss: {best_val_loss:.6f}")
    print(f"   - Val Loss Reduction: {val_loss_reduction:.2f}%")

    # Model quality assessment based on final validation loss
    quality_score = "excellent" if best_val_loss < 0.05 else \
                   "good" if best_val_loss < 0.1 else \
                   "fair" if best_val_loss < 0.2 else "poor"

    print(f"\n🏆 Model Quality Assessment:")
    if quality_score == "excellent":
        print(f"   ✨ Excellent! Model achieved MSE < 0.05 on validation set")
    elif quality_score == "good":
        print(f"   ✅ Good! Model achieved MSE < 0.1 on validation set")
    elif quality_score == "fair":
        print(f"   ⚠️  Fair. Model achieved MSE < 0.2 on validation set")
    else:
        print(f"   ❌ Poor. Consider training longer or adjusting hyperparameters")

    print(f"\n💡 Note:")
    print(f"   - Task: Time series prediction (multi-frequency sine wave)")
    print(f"   - MSE Loss: Lower is better (perfect prediction = 0)")
    print(f"   - Training samples: 8,000 sequences")
    print(f"   - Validation samples: 2,000 sequences")
    print(f"   - Model is relatively small (for demonstration)")

    # Log final summary to W&B
    if use_wandb:
        wandb.log({
            "final/train_loss": final_loss,
            "final/best_train_loss": best_loss,
            "final/train_loss_reduction_pct": loss_reduction,
            "final/val_loss": final_val_loss,
            "final/best_val_loss": best_val_loss,
            "final/val_loss_reduction_pct": val_loss_reduction,
            "final/quality_score": quality_score,
            "final/epochs_completed": epoch + 1
        })

        # Create a summary table
        wandb.run.summary["best_train_loss"] = best_loss
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["final_train_loss"] = final_loss
        wandb.run.summary["final_val_loss"] = final_val_loss
        wandb.run.summary["train_loss_reduction"] = loss_reduction
        wandb.run.summary["val_loss_reduction"] = val_loss_reduction
        wandb.run.summary["quality"] = quality_score

        print(f"\n📊 W&B Summary logged")
        print(f"   - View results at: {wandb.run.url}")

        # Finish W&B run
        wandb.finish()
        print(f"   - W&B run finished successfully")

    print(f"\n{'='*80}")
    print("🎉 Enhanced LSTM Training Script Finished Successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
