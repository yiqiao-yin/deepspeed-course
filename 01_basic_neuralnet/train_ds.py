"""Train a simple linear model using DeepSpeed on dummy data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import sys
import os

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class SimpleModel(nn.Module):
    """
    A simple linear regression model: y = Wx + b
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        """
        return self.linear(x)


def get_data_loader(batch_size: int) -> DataLoader:
    """
    Generates a dummy dataset y = 2x + 1 and returns a DataLoader.
    True parameters: W = 2.0, b = 1.0
    """
    x_data = torch.randn(1000, 1)
    y_data = 2 * x_data + 1
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size)


def main() -> None:
    """
    Initializes the model with DeepSpeed, trains it over a dummy dataset,
    and prints loss at intervals.

    Optional: Set WANDB_API_KEY environment variable to enable W&B tracking.
    """
    print("=" * 80)
    print("🚀 Starting DeepSpeed Linear Regression Training")
    print("=" * 80)

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
        print(f"   - To install: pip install wandb")
    elif not WANDB_AVAILABLE:
        print(f"\n📊 Weights & Biases: Not installed")
        print(f"   - To enable tracking: pip install wandb")
        print(f"   - Then: export WANDB_API_KEY=your_api_key")

    # True parameters from synthetic data: y = 2x + 1
    TRUE_WEIGHT = 2.0
    TRUE_BIAS = 1.0

    print(f"\n📊 Dataset Information:")
    print(f"   - Synthetic data: y = {TRUE_WEIGHT}x + {TRUE_BIAS}")
    print(f"   - Training samples: 1000")
    print(f"   - True Weight (W): {TRUE_WEIGHT}")
    print(f"   - True Bias (b): {TRUE_BIAS}")

    model = SimpleModel()

    # Print initial parameters
    with torch.no_grad():
        init_weight = model.linear.weight.item()
        init_bias = model.linear.bias.item()

    print(f"\n🎲 Initial Model Parameters (random):")
    print(f"   - Weight: {init_weight:.6f}")
    print(f"   - Bias: {init_bias:.6f}")

    print(f"\n⚙️  Initializing DeepSpeed...")
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )
    print(f"✅ DeepSpeed initialized successfully")

    data_loader = get_data_loader(batch_size=model_engine.train_micro_batch_size_per_gpu())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n💻 Training Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Batch size: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"   - Total batches per epoch: {len(data_loader)}")
    print(f"   - Number of epochs: 30")

    model_dtype = next(model_engine.module.parameters()).dtype
    print(f"   - Model dtype: {model_dtype}")

    # Initialize W&B run if enabled
    if use_wandb:
        wandb.init(
            project="deepspeed-linear-regression",
            name="simple-linear-model",
            config={
                "model": "SimpleLinearRegression",
                "dataset": "synthetic",
                "true_weight": TRUE_WEIGHT,
                "true_bias": TRUE_BIAS,
                "epochs": 30,
                "batch_size": model_engine.train_micro_batch_size_per_gpu(),
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "framework": "DeepSpeed"
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: deepspeed-linear-regression")
        print(f"   - View at: {wandb.run.url}")

    print(f"\n{'='*80}")
    print("🏋️  Training Started...")
    print(f"{'='*80}\n")

    epoch_losses = []

    for epoch in range(30):
        epoch_loss_sum = 0.0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            outputs = model_engine(x_batch)
            loss = nn.functional.mse_loss(outputs, y_batch)

            model_engine.backward(loss)
            model_engine.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if step % 10 == 0:
                print(f"Epoch {epoch:2d}/{30} | Step {step:3d} | Loss: {loss.item():.6f}")

                # Log to W&B if enabled
                if use_wandb:
                    wandb.log({
                        "step_loss": loss.item(),
                        "epoch": epoch,
                        "step": step
                    })

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss_sum / num_batches
        epoch_losses.append(avg_epoch_loss)

        # Log epoch metrics to W&B
        if use_wandb:
            with torch.no_grad():
                current_weight = model_engine.module.linear.weight.item()
                current_bias = model_engine.module.linear.bias.item()
                weight_error = abs(current_weight - TRUE_WEIGHT)
                bias_error = abs(current_bias - TRUE_BIAS)

            wandb.log({
                "epoch": epoch,
                "epoch_avg_loss": avg_epoch_loss,
                "learned_weight": current_weight,
                "learned_bias": current_bias,
                "weight_error": weight_error,
                "bias_error": bias_error,
                "weight_error_pct": (weight_error / TRUE_WEIGHT) * 100,
                "bias_error_pct": (bias_error / abs(TRUE_BIAS)) * 100 if TRUE_BIAS != 0 else 0
            })

        # Print epoch summary
        if epoch % 5 == 0 or epoch == 29:
            print(f"\n📈 Epoch {epoch:2d} Summary: Avg Loss = {avg_epoch_loss:.6f}")

            # Show current parameter estimates
            with torch.no_grad():
                current_weight = model_engine.module.linear.weight.item()
                current_bias = model_engine.module.linear.bias.item()
                weight_error = abs(current_weight - TRUE_WEIGHT)
                bias_error = abs(current_bias - TRUE_BIAS)

            print(f"   Current Parameters: W = {current_weight:.6f}, b = {current_bias:.6f}")
            print(f"   Parameter Errors: ΔW = {weight_error:.6f}, Δb = {bias_error:.6f}\n")

    print(f"\n{'='*80}")
    print("✅ Training Completed!")
    print(f"{'='*80}\n")

    # Final results
    final_loss = epoch_losses[-1]
    initial_loss = epoch_losses[0]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"📊 Training Summary:")
    print(f"   - Initial Loss: {initial_loss:.6f}")
    print(f"   - Final Loss: {final_loss:.6f}")
    print(f"   - Loss Reduction: {loss_reduction:.2f}%")

    # Extract final learned parameters
    with torch.no_grad():
        learned_weight = model_engine.module.linear.weight.item()
        learned_bias = model_engine.module.linear.bias.item()

    print(f"\n🎯 Final Model Parameters:")
    print(f"   - Learned Weight: {learned_weight:.6f}")
    print(f"   - Learned Bias: {learned_bias:.6f}")

    print(f"\n🎓 Ground Truth Parameters:")
    print(f"   - True Weight: {TRUE_WEIGHT:.6f}")
    print(f"   - True Bias: {TRUE_BIAS:.6f}")

    # Calculate errors
    weight_error = abs(learned_weight - TRUE_WEIGHT)
    bias_error = abs(learned_bias - TRUE_BIAS)
    weight_error_pct = (weight_error / TRUE_WEIGHT) * 100
    bias_error_pct = (bias_error / abs(TRUE_BIAS)) * 100 if TRUE_BIAS != 0 else 0

    print(f"\n📏 Parameter Estimation Errors:")
    print(f"   - Weight Error: {weight_error:.6f} ({weight_error_pct:.2f}%)")
    print(f"   - Bias Error: {bias_error:.6f} ({bias_error_pct:.2f}%)")

    # Accuracy assessment
    quality_score = "excellent" if (weight_error < 0.01 and bias_error < 0.01) else \
                   "good" if (weight_error < 0.05 and bias_error < 0.05) else \
                   "fair" if (weight_error < 0.1 and bias_error < 0.1) else "poor"

    print(f"\n🏆 Model Quality Assessment:")
    if quality_score == "excellent":
        print(f"   ✨ Excellent! Parameters match ground truth within 1% error")
    elif quality_score == "good":
        print(f"   ✅ Good! Parameters are very close to ground truth")
    elif quality_score == "fair":
        print(f"   ⚠️  Fair. Parameters are reasonably close to ground truth")
    else:
        print(f"   ❌ Poor. Consider training longer or adjusting learning rate")

    # Log final summary to W&B
    if use_wandb:
        wandb.log({
            "final/loss": final_loss,
            "final/loss_reduction_pct": loss_reduction,
            "final/learned_weight": learned_weight,
            "final/learned_bias": learned_bias,
            "final/weight_error": weight_error,
            "final/bias_error": bias_error,
            "final/weight_error_pct": weight_error_pct,
            "final/bias_error_pct": bias_error_pct,
            "final/quality_score": quality_score
        })

        # Create a summary table
        wandb.run.summary["true_weight"] = TRUE_WEIGHT
        wandb.run.summary["true_bias"] = TRUE_BIAS
        wandb.run.summary["learned_weight"] = learned_weight
        wandb.run.summary["learned_bias"] = learned_bias
        wandb.run.summary["total_loss_reduction"] = loss_reduction
        wandb.run.summary["quality"] = quality_score

        print(f"\n📊 W&B Summary logged")
        print(f"   - View results at: {wandb.run.url}")

        # Finish W&B run
        wandb.finish()
        print(f"   - W&B run finished successfully")

    print(f"\n{'='*80}")
    print("🎉 Training Script Finished Successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
