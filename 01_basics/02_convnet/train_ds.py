"""Enhanced training script with improved convergence strategies for CNN classification.

Improvements over basic train_ds.py:
1. Better weight initialization (Kaiming/He for ReLU networks)
2. Learning rate warmup and decay
3. Gradient clipping with monitoring
4. Loss plateau detection and LR adjustment
5. Early stopping with patience
6. Gradient norm tracking
7. More frequent parameter monitoring
8. Training accuracy tracking
9. Comprehensive logging and W&B integration
10. Model quality assessment based on accuracy

This script trains a CNN on synthetic MNIST-like data (28x28 grayscale images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
              "https://download.pytorch.org/whl/cu128\n")
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
    print("      uv run runpod/runpod_ctl.py run 01_basics/01_neuralnet")
    print("\n" + bar + "\n")
    sys.exit(1)


class CNNModelEnhanced(nn.Module):
    """
    Enhanced CNN for MNIST-like classification with better initialization.
    Architecture: Conv(16) -> Pool -> Conv(32) -> Pool -> FC(128) -> FC(10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Kaiming/He initialization for ReLU networks (better for deep networks)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN.
        Input: [batch, 1, 28, 28]
        Output: [batch, 10] (logits)
        """
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 7, 7]
        x = torch.flatten(x, 1)               # [batch, 32*7*7 = 1568]
        x = F.relu(self.fc1(x))               # [batch, 128]
        x = self.fc2(x)                       # [batch, 10]
        return x


def get_data_loader(batch_size: int, num_samples: int = 10000) -> DataLoader:
    """
    Generates a random dataset that simulates MNIST:
    - 28x28 grayscale images (1 channel)
    - Integer labels 0-9

    Args:
        batch_size: Number of samples per batch
        num_samples: Total number of training samples

    Returns:
        DataLoader with synthetic MNIST-like data
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    x_data = torch.randn(num_samples, 1, 28, 28)
    y_data = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_lr_schedule(epoch: int, initial_lr: float = 0.001, warmup_epochs: int = 5, total_epochs: int = 50) -> float:
    """
    Learning rate schedule with warmup and cosine decay.

    Args:
        epoch: Current epoch
        initial_lr: Initial learning rate
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs

    Returns:
        Adjusted learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.

    Args:
        outputs: Model output logits [batch, num_classes]
        targets: Ground truth labels [batch]

    Returns:
        Accuracy as percentage (0-100)
    """
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return (correct / total) * 100.0


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
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50).")
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
    Enhanced CNN training with multiple convergence strategies.
    """
    args = parse_args()
    global_step = 0
    require_gpu()
    print("=" * 80)
    print("🚀 Starting ENHANCED DeepSpeed CNN Training")
    print("=" * 80)
    print("\n✨ Enhancements in this version:")
    print("   1. Kaiming/He weight initialization for ReLU networks")
    print("   2. Learning rate warmup (5 epochs)")
    print("   3. Cosine learning rate decay")
    print("   4. Gradient norm monitoring")
    print("   5. Loss plateau detection")
    print("   6. Early stopping with patience")
    print("   7. Training accuracy tracking")
    print("   8. More frequent progress updates")
    print("   9. Comprehensive logging with W&B support")
    print("  10. Model quality assessment")

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

    print(f"\n📊 Dataset Information:")
    print(f"   - Task: MNIST-like image classification")
    print(f"   - Image size: 28x28 grayscale")
    print(f"   - Number of classes: 10")
    print(f"   - Training samples: 10,000 (synthetic)")

    model = CNNModelEnhanced()

    print(f"\n🏗️  Model Architecture:")
    print(f"   - Conv1: 1 → 16 channels (5x5 kernel)")
    print(f"   - MaxPool: 2x2")
    print(f"   - Conv2: 16 → 32 channels (5x5 kernel)")
    print(f"   - MaxPool: 2x2")
    print(f"   - FC1: 1568 → 128")
    print(f"   - FC2: 128 → 10 (output)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")

    print(f"\n⚙️  Initializing DeepSpeed...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )
    print(f"✅ DeepSpeed initialized successfully")

    batch_size = model_engine.train_micro_batch_size_per_gpu()
    data_loader = get_data_loader(batch_size=batch_size, num_samples=10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n💻 Training Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Total batches per epoch: {len(data_loader)}")
    print(f"   - Number of epochs: 50")
    print(f"   - Initial learning rate: 0.001")
    print(f"   - Warmup epochs: 5")
    print(f"   - LR schedule: Warmup → Cosine decay")

    model_dtype = next(model_engine.module.parameters()).dtype
    print(f"   - Model dtype: {model_dtype}")

    # Initialize W&B run if enabled
    if use_wandb:
        wandb.init(
            project="deepspeed-cnn-mnist",
            name="enhanced-cnn-model",
            config={
                "model": "EnhancedCNN",
                "dataset": "synthetic-mnist",
                "num_classes": 10,
                "epochs": 50,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "initial_lr": 0.001,
                "warmup_epochs": 5,
                "lr_schedule": "warmup_cosine",
                "initialization": "kaiming",
                "framework": "DeepSpeed",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "enhancements": [
                    "kaiming_init",
                    "lr_warmup",
                    "cosine_decay",
                    "gradient_monitoring",
                    "early_stopping",
                    "accuracy_tracking"
                ]
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: deepspeed-cnn-mnist")
        print(f"   - View at: {wandb.run.url}")

    print(f"\n{'='*80}")
    print("🏋️  Enhanced Training Started...")
    print(f"{'='*80}\n")

    loss_fn = nn.CrossEntropyLoss()
    epoch_losses = []
    epoch_accuracies = []
    best_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    patience_limit = 15
    min_improvement = 1e-5
    total_epochs = args.epochs

    for epoch in range(total_epochs):
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        epoch_grad_norms = []

        # Get learning rate for this epoch
        current_lr = get_lr_schedule(epoch, initial_lr=0.001, warmup_epochs=5, total_epochs=total_epochs)

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\n📚 Epoch {epoch:3d}/{total_epochs} - Learning Rate: {current_lr:.6e}")

        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device).to(model_dtype)
            y_batch = y_batch.to(device)

            outputs = model_engine(x_batch)
            loss = loss_fn(outputs, y_batch)

            model_engine.backward(loss)

            # Compute gradient norm before stepping
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

            # Track metrics
            epoch_loss_sum += loss.item()
            batch_accuracy = calculate_accuracy(outputs, y_batch)
            batch_correct = int((batch_accuracy / 100.0) * y_batch.size(0))
            epoch_correct += batch_correct
            epoch_total += y_batch.size(0)
            num_batches += 1

            if step % 50 == 0:  # Update every 50 steps
                print(f"   Step {step:3d} | Loss: {loss.item():.6f} | Acc: {batch_accuracy:.2f}% | Grad Norm: {total_norm:.6f}")

                # Log to W&B if enabled
                if use_wandb:
                    wandb.log({
                        "step_loss": loss.item(),
                        "step_accuracy": batch_accuracy,
                        "gradient_norm": total_norm,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": step
                    })

        # Calculate average metrics for the epoch
        avg_epoch_loss = epoch_loss_sum / num_batches
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        epoch_accuracy = (epoch_correct / epoch_total) * 100.0
        epoch_losses.append(avg_epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Print epoch summary
        print(f"\n📈 Epoch {epoch:3d} Summary:")
        print(f"   - Avg Loss: {avg_epoch_loss:.6f}")
        print(f"   - Accuracy: {epoch_accuracy:.2f}%")
        print(f"   - Avg Grad Norm: {avg_grad_norm:.6f}")
        print(f"   - Learning Rate: {current_lr:.6e}")

        # Check for improvement
        if avg_epoch_loss < best_loss - min_improvement:
            best_loss = avg_epoch_loss
            patience_counter = 0
            print(f"   ✅ New best loss! Patience reset.")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience_limit}")

        # Track best accuracy
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            print(f"   🎯 New best accuracy: {best_accuracy:.2f}%")

        # Log epoch metrics to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_avg_loss": avg_epoch_loss,
                "epoch_accuracy": epoch_accuracy,
                "epoch_avg_grad_norm": avg_grad_norm,
                "learning_rate": current_lr,
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
                "patience": patience_counter
            })

        # Early stopping check
        if patience_counter >= patience_limit:
            print(f"\n🛑 Early stopping triggered! No improvement for {patience_limit} epochs.")
            print(f"   Best loss achieved: {best_loss:.6f}")
            print(f"   Best accuracy achieved: {best_accuracy:.2f}%")
            break

        if 0 < args.max_steps <= global_step:
            print(f"\n[dry run] stopped at --max-steps {args.max_steps}")
            break

    print(f"\n{'='*80}")
    print("✅ Training Completed!")
    print(f"{'='*80}\n")

    # Final results
    final_loss = epoch_losses[-1]
    final_accuracy = epoch_accuracies[-1]
    initial_loss = epoch_losses[0]
    initial_accuracy = epoch_accuracies[0]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    accuracy_gain = final_accuracy - initial_accuracy

    print(f"📊 Training Summary:")
    print(f"   - Initial Loss: {initial_loss:.6f}")
    print(f"   - Final Loss: {final_loss:.6f}")
    print(f"   - Best Loss: {best_loss:.6f}")
    print(f"   - Loss Reduction: {loss_reduction:.2f}%")
    print(f"   - Epochs completed: {epoch + 1}")

    print(f"\n🎯 Accuracy Metrics:")
    print(f"   - Initial Accuracy: {initial_accuracy:.2f}%")
    print(f"   - Final Accuracy: {final_accuracy:.2f}%")
    print(f"   - Best Accuracy: {best_accuracy:.2f}%")
    print(f"   - Accuracy Gain: {accuracy_gain:.2f}%")

    # Model quality assessment
    quality_score = "excellent" if best_accuracy >= 90 else \
                   "good" if best_accuracy >= 70 else \
                   "fair" if best_accuracy >= 50 else "poor"

    print(f"\n🏆 Model Quality Assessment:")
    if quality_score == "excellent":
        print(f"   ✨ Excellent! Model achieved ≥90% accuracy")
    elif quality_score == "good":
        print(f"   ✅ Good! Model achieved ≥70% accuracy")
    elif quality_score == "fair":
        print(f"   ⚠️  Fair. Model achieved ≥50% accuracy")
    else:
        print(f"   ❌ Poor. Consider training longer or adjusting hyperparameters")

    # Note about synthetic data
    print(f"\n💡 Note:")
    print(f"   - This is trained on random synthetic data (not real MNIST)")
    print(f"   - High accuracy on random data indicates the model is learning patterns")
    print(f"   - For real MNIST, accuracy should approach 98-99%")

    # Log final summary to W&B
    if use_wandb:
        wandb.log({
            "final/loss": final_loss,
            "final/best_loss": best_loss,
            "final/loss_reduction_pct": loss_reduction,
            "final/accuracy": final_accuracy,
            "final/best_accuracy": best_accuracy,
            "final/accuracy_gain": accuracy_gain,
            "final/quality_score": quality_score,
            "final/epochs_completed": epoch + 1
        })

        # Create a summary table
        wandb.run.summary["best_loss"] = best_loss
        wandb.run.summary["best_accuracy"] = best_accuracy
        wandb.run.summary["final_accuracy"] = final_accuracy
        wandb.run.summary["total_loss_reduction"] = loss_reduction
        wandb.run.summary["accuracy_gain"] = accuracy_gain
        wandb.run.summary["quality"] = quality_score

        print(f"\n📊 W&B Summary logged")
        print(f"   - View results at: {wandb.run.url}")

        # Finish W&B run
        wandb.finish()
        print(f"   - W&B run finished successfully")

    print(f"\n{'='*80}")
    print("🎉 Enhanced CNN Training Script Finished Successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
