import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Get Hugging Face Token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN not found in environment variables.")
    print("Please export your Hugging Face token with: export HF_TOKEN=your-token")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    hf_token = None

# Check for W&B API key
wandb_api_key = os.environ.get("WANDB_API_KEY")
use_wandb = False

if WANDB_AVAILABLE and wandb_api_key:
    try:
        wandb.login(key=wandb_api_key)
        use_wandb = True
        print("✅ Weights & Biases: Enabled")
    except Exception as e:
        print(f"⚠️  Weights & Biases: Login failed - {e}")
        use_wandb = False
elif not WANDB_AVAILABLE:
    print("📊 Weights & Biases: Not installed (optional)")
    print("   To enable tracking: pip install wandb or uv add wandb")
else:
    print("📊 Weights & Biases: Disabled (no API key found)")
    print("   To enable: export WANDB_API_KEY=your-api-key")

# Load the dataset
# dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = load_dataset("eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024", split="train")

# Download model and tokenizer
# model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "unsloth/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Initialize W&B run if enabled
if use_wandb:
    wandb.init(
        project="huggingface-deepspeed-finetuning",
        name=f"llama-3.2-3b-warren-buffett",
        config={
            "model": model_name,
            "dataset": "eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024",
            "batch_size": 8,
            "num_epochs": 50,
            "learning_rate": 2e-5,
            "deepspeed_config": "ds_config.json"
        }
    )
    print(f"📈 W&B Run initialized: {wandb.run.name}")
    print(f"   View at: {wandb.run.url}")

# Set Deepspeed configuration file path
ds_config_path = "ds_config.json"

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


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, the run gets as far as loading the model and then dies deep
    inside the training stack -- for this script, with
    "Your setup doesn't support bf16/gpu", which tells a newcomer nothing
    about what went wrong or what to do next. Worse, it happens AFTER the
    model download, so the reader has already waited.

    Set ALLOW_CPU=1 to bypass.
    """
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
        print("            You will also need bf16 disabled in the training")
        print("            config, or the trainer raises anyway.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before the run fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This example fine-tunes a HuggingFace LLM with DeepSpeed")
    print("  ZeRO. It downloads real weights and needs real GPU memory.")
    print("\n  Examples 01-04 teach the same mechanics and DO run on CPU.")
    print("\n  No GPU at all? These need none:")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("      ./tests/run_all.sh    # the full logic suite, no downloads")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 05_huggingface")
    print("      uv run runpod/runpod_ctl.py run 05_huggingface \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


_args = parse_args()
require_gpu()

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    # -1 means "ignore me, use epochs" — Trainer's own convention.
    max_steps=_args.max_steps,
    per_device_train_batch_size=8,  # adjust per your GPU memory
    gradient_accumulation_steps=1,
    num_train_epochs=50,
    learning_rate=2e-5,
    fp16=False,
    deepspeed=ds_config_path,
    logging_steps=10,
    save_strategy="no",
    report_to="wandb" if use_wandb else "none",  # Enable W&B reporting
)

# Define the formatting function
def format_instruction(sample):
    return {"text": f"Question: {sample['question']}\nAnswer: {sample['answer']}"}

# Preprocess dataset
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save model
# Define model and user IDs for Hugging Face Hub
model_id = "warren-buffett-letters-qna-r1-enhanced-1998-2024-finetuned-llama-3.2-3B-Instruct"
user_id = "eagle0504"
repo_id = f"{user_id}/{model_id}"

# Save the model and tokenizer locally
trainer.model.save_pretrained(repo_id)
tokenizer.save_pretrained(repo_id)

# Push to Hugging Face Hub
# trainer.model.push_to_hub(repo_id, use_auth_token=hf_token)
# tokenizer.push_to_hub(repo_id, use_auth_token=hf_token)

# Finish W&B run
if use_wandb:
    wandb.finish()
    print("✅ W&B run finished")


