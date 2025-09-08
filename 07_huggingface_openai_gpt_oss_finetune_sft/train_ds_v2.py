import subprocess
import sys
import os
import argparse
from pathlib import Path

def check_requirements():
    """Check if required files and dependencies exist."""
    required_files = ["train_ds.py", "ds_config.json"]
    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    # Check if deepspeed is installed
    try:
        subprocess.run(["deepspeed", "--help"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: DeepSpeed not found!")
        print("Please install DeepSpeed: pip install deepspeed")
        return False

    return True

def run_deepspeed_training(num_gpus=4, include_gpus=None, training_script="train_ds.py"):
    """Run DeepSpeed training with specified configuration."""

    # Build the deepspeed command
    cmd = ["deepspeed"]

    if include_gpus:
        cmd.extend(["--include", f"localhost:{include_gpus}"])
    else:
        cmd.extend([f"--num_gpus={num_gpus}"])

    cmd.append(training_script)

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        # Run the deepspeed command with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())

        # Wait for the process to complete
        return_code = process.poll()

        if return_code == 0:
            print("=" * 60)
            print("🎉 Training completed successfully!")
            return True
        else:
            print("=" * 60)
            print(f"❌ Training failed with return code: {return_code}")
            return False

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("🚀 Hello from proj!")
    print("Starting DeepSpeed training...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DeepSpeed training")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use (default: 4)")
    parser.add_argument("--include_gpus", type=str, default=None,
                       help="Specific GPUs to use (e.g., '0,1,2,3')")
    parser.add_argument("--script", type=str, default="train_ds.py",
                       help="Training script to run (default: train_ds.py)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token for pushing models to hub")
    parser.add_argument("--no_push_to_hub", action="store_true",
                       help="Disable pushing model to Hugging Face hub")
    parser.add_argument("--no_evaluation", action="store_true",
                       help="Skip evaluation after training")

    args = parser.parse_args()

    # Set environment variables for the training script
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("   ✅ HF Token provided")

    if args.no_push_to_hub:
        os.environ["PUSH_TO_HUB"] = "false"
        print("   ⚠️  Push to HF Hub: DISABLED")
    else:
        print("   📤 Push to HF Hub: ENABLED (if authenticated)")

    if args.no_evaluation:
        os.environ["RUN_EVALUATION"] = "false"
        print("   ⚠️  Evaluation: DISABLED")
    else:
        print("   🧪 Evaluation: ENABLED")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Print configuration
    print("\n📋 Configuration:")
    if args.include_gpus:
        print(f"   GPUs: {args.include_gpus}")
    else:
        print(f"   Number of GPUs: {args.num_gpus}")
    print(f"   Training script: {args.script}")
    print(f"   DeepSpeed config: ds_config.json")
    print()

    # Run training
    success = run_deepspeed_training(
        num_gpus=args.num_gpus,
        include_gpus=args.include_gpus,
        training_script=args.script
    )

    if success:
        print("✅ All done!")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()