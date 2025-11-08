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

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
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


