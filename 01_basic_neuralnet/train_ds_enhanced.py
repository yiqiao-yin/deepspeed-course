"""Enhanced training script with improved convergence strategies for simple linear regression.

Improvements over train_ds.py:
1. Better weight initialization (Xavier/Glorot)
2. Learning rate warmup and decay
3. Gradient clipping with monitoring
4. Loss plateau detection and LR adjustment
5. Early stopping with patience
6. Gradient norm tracking
7. More frequent parameter monitoring
8. FP32 precision (FP16 causes numerical instability for simple problems)

IMPORTANT: This script uses ds_config_fp32.json instead of ds_config.json.
For simple linear regression, FP16's limited precision range can cause gradient
underflow, preventing any parameter updates. FP32 provides the numerical stability
needed for small gradients typical in simple problems like y = 2x + 1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import sys
import argparse
import os

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
    print("      ./tests/run_all.sh    # the full logic suite, no GPU, no downloads")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py gpus --min-vram 24")
    print("      uv run runpod/runpod_ctl.py run 01_basic_neuralnet")
    print("\n" + bar + "\n")
    sys.exit(1)


class SimpleModelEnhanced(nn.Module):
    """
    Enhanced linear regression model with better initialization.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

        # Xavier/Glorot initialization for better convergence
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.5)  # Initialize closer to true value (1.0)

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
    # Set seed for reproducibility
    torch.manual_seed(42)
    x_data = torch.randn(1000, 1)
    y_data = 2 * x_data + 1
    dataset = TensorDataset(x_data, y_data)
    # Don't shuffle for more stable gradients in simple linear regression
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_lr_schedule(epoch: int, initial_lr: float = 0.01, warmup_epochs: int = 10, total_epochs: int = 100) -> float:
    """
    Learning rate schedule with warmup and slow cosine decay.

    Args:
        epoch: Current epoch
        initial_lr: Initial learning rate (higher for faster convergence)
        warmup_epochs: Number of warmup epochs (more warmup for stability)
        total_epochs: Total training epochs

    Returns:
        Adjusted learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Slow cosine decay after warmup (keeps LR higher for longer)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


def parse_args() -> "argparse.Namespace":
    """
    Command-line options.

    Added so a CoreWeave user can validate the whole pipeline without burning
    a full allocation:

        sbatch run_deepspeed.sh --max-steps 20

    Both defaults preserve the previous behaviour exactly -- `--max-steps -1`
    means no cap, and `--epochs` defaults to what the script always used.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100).")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Stop after this many optimizer steps. -1 means "
                             "no cap. Used by the dry-run path; a handful of "
                             "steps proves the plumbing without training.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set by the deepspeed launcher; accepted so the "
                             "launcher's argument does not cause a parse error.")
    return parser.parse_known_args()[0]


def main() -> None:
    """
    Enhanced training with multiple convergence strategies.
    """
    args = parse_args()
    global_step = 0
    require_gpu()
    print("=" * 80)
    print("🚀 Starting ENHANCED DeepSpeed Linear Regression Training")
    print("=" * 80)
    print("\n✨ Enhancements in this version:")
    print("   1. Xavier/Glorot weight initialization")
    print("   2. Learning rate warmup (10 epochs)")
    print("   3. Slow cosine learning rate decay")
    print("   4. Gradient norm monitoring")
    print("   5. Loss plateau detection")
    print("   6. Early stopping with patience")
    print("   7. More frequent progress updates")
    print("   8. Higher learning rate (0.01) for faster convergence")
    print("   9. Longer training (100 epochs) for better convergence")
    print("  10. Non-shuffled data for stable gradients")

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

    # True parameters from synthetic data: y = 2x + 1
    TRUE_WEIGHT = 2.0
    TRUE_BIAS = 1.0

    print(f"\n📊 Dataset Information:")
    print(f"   - Synthetic data: y = {TRUE_WEIGHT}x + {TRUE_BIAS}")
    print(f"   - Training samples: 1000")
    print(f"   - True Weight (W): {TRUE_WEIGHT}")
    print(f"   - True Bias (b): {TRUE_BIAS}")

    model = SimpleModelEnhanced()

    # Print initial parameters
    with torch.no_grad():
        init_weight = model.linear.weight.item()
        init_bias = model.linear.bias.item()

    print(f"\n🎲 Initial Model Parameters (Xavier + bias=0.5):")
    print(f"   - Weight: {init_weight:.6f}")
    print(f"   - Bias: {init_bias:.6f}")
    print(f"   - Weight error from true: {abs(init_weight - TRUE_WEIGHT):.6f}")
    print(f"   - Bias error from true: {abs(init_bias - TRUE_BIAS):.6f}")

    print(f"\n⚙️  Initializing DeepSpeed...")
    print(f"   - Using FP32 precision (FP16 causes numerical instability for simple linear regression)")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_fp32.json"
    )
    print(f"✅ DeepSpeed initialized successfully")

    data_loader = get_data_loader(batch_size=model_engine.train_micro_batch_size_per_gpu())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n💻 Training Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Batch size: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"   - Total batches per epoch: {len(data_loader)}")
    print(f"   - Number of epochs: 100")
    print(f"   - Initial learning rate: 0.01")
    print(f"   - Warmup epochs: 10")
    print(f"   - LR schedule: Warmup → Slow Cosine decay")

    model_dtype = next(model_engine.module.parameters()).dtype
    print(f"   - Model dtype: {model_dtype}")

    # Initialize W&B run if enabled
    if use_wandb:
        wandb.init(
            project="deepspeed-linear-regression",
            name="enhanced-linear-model",
            config={
                "model": "EnhancedLinearRegression",
                "dataset": "synthetic",
                "true_weight": TRUE_WEIGHT,
                "true_bias": TRUE_BIAS,
                "epochs": 100,
                "batch_size": model_engine.train_micro_batch_size_per_gpu(),
                "optimizer": "Adam",
                "initial_lr": 0.01,
                "warmup_epochs": 10,
                "lr_schedule": "warmup_slow_cosine",
                "initialization": "xavier",
                "framework": "DeepSpeed",
                "enhancements": [
                    "xavier_init",
                    "lr_warmup",
                    "cosine_decay",
                    "gradient_clipping",
                    "early_stopping"
                ]
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: deepspeed-linear-regression")
        print(f"   - View at: {wandb.run.url}")

    print(f"\n{'='*80}")
    print("🏋️  Enhanced Training Started...")
    print(f"{'='*80}\n")

    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20  # Increased patience for 100 epochs
    min_improvement = 1e-7  # More sensitive to small improvements
    total_epochs = args.epochs

    for epoch in range(total_epochs):
        epoch_loss_sum = 0.0
        num_batches = 0
        epoch_grad_norms = []

        # Get learning rate for this epoch
        current_lr = get_lr_schedule(epoch, initial_lr=0.01, warmup_epochs=10, total_epochs=100)

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\n📚 Epoch {epoch:3d}/100 - Learning Rate: {current_lr:.6e}")

        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device).to(model_dtype)

            outputs = model_engine(x_batch)
            loss = nn.functional.mse_loss(outputs, y_batch)

            model_engine.backward(loss)

            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in model_engine.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)

            model_engine.step()
            global_step += 1

            # Dry-run cap. Breaks the inner loop here and the epoch loop just
            # below, so `--max-steps 20` stops after 20 optimizer steps rather
            # than finishing the epoch.
            if 0 < args.max_steps <= global_step:
                break

            epoch_loss_sum += loss.item()
            num_batches += 1

            if step % 5 == 0:  # More frequent updates
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

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss_sum / num_batches
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        epoch_losses.append(avg_epoch_loss)

        # Get current parameters
        with torch.no_grad():
            current_weight = model_engine.module.linear.weight.item()
            current_bias = model_engine.module.linear.bias.item()
            weight_error = abs(current_weight - TRUE_WEIGHT)
            bias_error = abs(current_bias - TRUE_BIAS)

        # Print epoch summary every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            print(f"\n📈 Epoch {epoch:3d} Summary:")
            print(f"   - Avg Loss: {avg_epoch_loss:.6f}")
            print(f"   - Avg Grad Norm: {avg_grad_norm:.6f}")
            print(f"   - Learning Rate: {current_lr:.6e}")
            print(f"   - Parameters: W={current_weight:.6f}, b={current_bias:.6f}")
            print(f"   - Errors: ΔW={weight_error:.6f}, Δb={bias_error:.6f}")

        # Check for improvement
        if avg_epoch_loss < best_loss - min_improvement:
            best_loss = avg_epoch_loss
            patience_counter = 0
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                print(f"   ✅ New best loss! Patience reset.")
        else:
            patience_counter += 1
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience_limit}")

        # Log epoch metrics to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_avg_loss": avg_epoch_loss,
                "epoch_avg_grad_norm": avg_grad_norm,
                "learning_rate": current_lr,
                "learned_weight": current_weight,
                "learned_bias": current_bias,
                "weight_error": weight_error,
                "bias_error": bias_error,
                "weight_error_pct": (weight_error / TRUE_WEIGHT) * 100,
                "bias_error_pct": (bias_error / abs(TRUE_BIAS)) * 100 if TRUE_BIAS != 0 else 0,
                "best_loss": best_loss,
                "patience": patience_counter
            })

        # Early stopping check
        if patience_counter >= patience_limit:
            print(f"\n🛑 Early stopping triggered! No improvement for {patience_limit} epochs.")
            print(f"   Best loss achieved: {best_loss:.6f}")
            break

        if 0 < args.max_steps <= global_step:
            print(f"\n[dry run] stopped at --max-steps {args.max_steps}")
            break

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
    print(f"   - Best Loss: {best_loss:.6f}")
    print(f"   - Loss Reduction: {loss_reduction:.2f}%")
    print(f"   - Epochs completed: {epoch + 1}")

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

    # Improvement analysis
    print(f"\n🔍 Improvement Analysis:")
    print(f"   - Initial weight error: {abs(init_weight - TRUE_WEIGHT):.6f}")
    print(f"   - Final weight error: {weight_error:.6f}")
    print(f"   - Weight improvement: {((abs(init_weight - TRUE_WEIGHT) - weight_error) / abs(init_weight - TRUE_WEIGHT) * 100):.2f}%")
    print(f"   - Initial bias error: {abs(init_bias - TRUE_BIAS):.6f}")
    print(f"   - Final bias error: {bias_error:.6f}")
    print(f"   - Bias improvement: {((abs(init_bias - TRUE_BIAS) - bias_error) / abs(init_bias - TRUE_BIAS) * 100):.2f}%")

    # Log final summary to W&B
    if use_wandb:
        wandb.log({
            "final/loss": final_loss,
            "final/best_loss": best_loss,
            "final/loss_reduction_pct": loss_reduction,
            "final/learned_weight": learned_weight,
            "final/learned_bias": learned_bias,
            "final/weight_error": weight_error,
            "final/bias_error": bias_error,
            "final/weight_error_pct": weight_error_pct,
            "final/bias_error_pct": bias_error_pct,
            "final/quality_score": quality_score,
            "final/epochs_completed": epoch + 1
        })

        # Create a summary table
        wandb.run.summary["true_weight"] = TRUE_WEIGHT
        wandb.run.summary["true_bias"] = TRUE_BIAS
        wandb.run.summary["learned_weight"] = learned_weight
        wandb.run.summary["learned_bias"] = learned_bias
        wandb.run.summary["total_loss_reduction"] = loss_reduction
        wandb.run.summary["quality"] = quality_score
        wandb.run.summary["best_loss"] = best_loss

        print(f"\n📊 W&B Summary logged")
        print(f"   - View results at: {wandb.run.url}")

        # Finish W&B run
        wandb.finish()
        print(f"   - W&B run finished successfully")

    print(f"\n{'='*80}")
    print("🎉 Enhanced Training Script Finished Successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
