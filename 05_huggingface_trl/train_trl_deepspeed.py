"""TRL Supervised Fine-Tuning with DeepSpeed for function calling.

This script fine-tunes Qwen/Qwen3-0.6B on a tool-augmented dataset
for function calling capabilities using TRL's SFTTrainer with DeepSpeed
distributed training support.

The training enables the model to:
1. Recognize when to call functions from user queries
2. Generate proper function call arguments
3. Process tool responses and continue conversations

Requirements:
    pip install uv
    uv init .
    uv add torch transformers trl datasets deepspeed wandb hf_transfer
"""

import json
import argparse
import os
import sys
from typing import Dict, Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
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
    print("\n  This example CANNOT run on CPU: it needs real GPU memory and")
    print("  downloads a large model. Examples 01-04 teach the same mechanics")
    print("  and do run on CPU.")
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


def load_tool_dataset(dataset_path: str = "tool_augmented_dataset.json") -> Dataset:
    """
    Load tool-augmented dataset from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        HuggingFace Dataset object
    """
    print(f"📂 Loading dataset from {dataset_path}...")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        print(f"   Please ensure the file exists in the current directory.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_dict(data)
    print(f"✅ Dataset loaded successfully")
    print(f"   - Number of examples: {len(dataset)}")
    print(f"   - Features: {list(dataset.features.keys())}")

    return dataset


def verify_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-0.6B"
) -> tuple:
    """
    Load and verify model and tokenizer before training.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n🤖 Loading model: {model_name}")
    print(f"   - This may take a few minutes on first run...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use FP32 for stability
        )
        print(f"✅ Model and tokenizer loaded successfully")

        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"\n📊 Model Information:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def get_training_arguments(
    output_dir: str = "./sft_qwen_model",
    use_wandb: bool = False, max_steps: int = -1) -> TrainingArguments:
    """
    Create training arguments for SFTTrainer.

    Args:
        output_dir: Directory to save model checkpoints
        use_wandb: Whether to use Weights & Biases tracking

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        # -1 means "ignore me, use epochs" — Trainer's own convention.
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # Use FP32 for numerical stability
        bf16=False,
        gradient_checkpointing=False,
        deepspeed="ds_config.json",  # DeepSpeed configuration
        report_to=["wandb"] if use_wandb else [],
        run_name="trl-qwen-function-calling" if use_wandb else None,
        logging_dir="./logs",
        remove_unused_columns=False,  # Keep all dataset columns
        dataloader_num_workers=2,
    )


def parse_args() -> "argparse.Namespace":
    """
    Command-line options.

    Added so a CoreWeave user can validate the pipeline without burning a full
    allocation. Defaults reproduce the previous behaviour exactly.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Stop after this many optimizer steps. -1 means no cap (use epochs) — HuggingFace Trainer's own convention, so the default preserves the previous behaviour exactly. This is what makes `sbatch run_deepspeed.sh --max-steps 20` a real dry run rather than a full job.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set by the deepspeed launcher; accepted so its "
                             "argument does not cause a parse error.")
    return parser.parse_known_args()[0]


def main() -> None:
    """
    Main training function for TRL supervised fine-tuning with DeepSpeed.
    """
    require_gpu()
    print("=" * 80)
    print("🚀 Starting TRL Supervised Fine-Tuning with DeepSpeed")
    print("=" * 80)
    print("\n📋 Training Configuration:")
    print("   - Model: Qwen/Qwen3-0.6B")
    print("   - Task: Function calling / Tool use")
    print("   - Trainer: TRL SFTTrainer")
    print("   - Framework: DeepSpeed")
    print("   - Dataset: tool_augmented_dataset.json")

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

    # Load dataset
    dataset = load_tool_dataset("tool_augmented_dataset.json")

    # Load model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    model, tokenizer = verify_model_and_tokenizer(model_name)

    # Get training arguments
    args = parse_args()
    training_args = get_training_arguments(
        max_steps=args.max_steps,
        output_dir="./sft_qwen_model",
        use_wandb=use_wandb
    )

    print(f"\n⚙️  Training Parameters:")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Warmup steps: {training_args.warmup_steps}")
    print(f"   - Logging steps: {training_args.logging_steps}")
    print(f"   - FP16: {training_args.fp16}")
    print(f"   - DeepSpeed config: {training_args.deepspeed}")

    # Initialize W&B run if enabled
    if use_wandb:
        wandb.init(
            project="trl-function-calling",
            name="qwen-sft-deepspeed",
            config={
                "model": model_name,
                "task": "function_calling",
                "dataset": "tool_augmented_dataset",
                "trainer": "TRL_SFTTrainer",
                "framework": "DeepSpeed",
                "num_train_samples": len(dataset),
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
            }
        )
        print(f"\n📈 W&B Run initialized: {wandb.run.name}")
        print(f"   - Project: trl-function-calling")
        print(f"   - View at: {wandb.run.url}")

    # Initialize SFTTrainer
    print(f"\n🎯 Initializing SFTTrainer with DeepSpeed...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"✅ SFTTrainer initialized successfully")

    # Training
    print(f"\n{'='*80}")
    print("🏋️  Training Started...")
    print(f"{'='*80}\n")

    try:
        train_result = trainer.train()

        print(f"\n{'='*80}")
        print("✅ Training Completed Successfully!")
        print(f"{'='*80}")

        # Print training summary
        print(f"\n📊 Training Summary:")
        print(f"   - Total runtime: {train_result.metrics.get('train_runtime', 0):.2f} seconds")
        print(f"   - Samples per second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
        print(f"   - Training loss: {train_result.metrics.get('train_loss', 0):.4f}")
        print(f"   - Global steps: {train_result.global_step}")

        # Save model
        print(f"\n💾 Saving model to {training_args.output_dir}...")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"✅ Model saved successfully")

        # Log final metrics to W&B
        if use_wandb:
            wandb.log({
                "final/train_loss": train_result.metrics.get('train_loss', 0),
                "final/train_runtime": train_result.metrics.get('train_runtime', 0),
                "final/global_step": train_result.global_step,
            })
            wandb.finish()
            print(f"\n📊 W&B run finished successfully")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        raise

    print(f"\n{'='*80}")
    print("🎉 TRL Training Script Finished Successfully!")
    print(f"{'='*80}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Load your model: AutoModelForCausalLM.from_pretrained('{training_args.output_dir}')")
    print(f"   2. Run inference with function calling prompts")
    print(f"   3. Test with timer and reminder examples")
    print(f"\n")


if __name__ == "__main__":
    main()
