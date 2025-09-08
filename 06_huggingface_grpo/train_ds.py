"""
Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B using DeepSpeed on a CoT-enhanced GSM8K dataset.
Logs progress and saves the fine-tuned model locally in Hugging Face Hub-style format.
"""

import logging
import os

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading tokenizer and model...")

# Load model and tokenizer from Hugging Face Hub
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

logger.info("Loading dataset...")

# Load dataset from Hugging Face Datasets Hub
dataset = load_dataset(
    "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1"
)
train_dataset = dataset["train"]

logger.info("Formatting dataset...")

# Format each example with CoT reasoning
def format_example(example: dict) -> dict:
    """
    Formats an example to include CoT reasoning before the final answer.

    Args:
        example (dict): A dictionary containing 'question', 'cot', and 'answer'.

    Returns:
        dict: Formatted string as a new 'text' field.
    """
    prompt = (
        f"Question: {example['question']}\n"
        f"Let's think step by step.\n{example['cot']}\n"
        f"Answer: {example['answer']}"
    )
    return {"text": prompt}

train_dataset = train_dataset.map(format_example)

logger.info("Tokenizing dataset...")

# Tokenize each example with max length and padding
def tokenize_function(example: dict) -> dict:
    """
    Tokenizes the formatted example using the provided tokenizer.

    Args:
        example (dict): A dictionary containing a 'text' field.

    Returns:
        dict: Tokenized representation including input_ids and attention_mask.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)

logger.info("Setting PyTorch format...")

# Format dataset to PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

logger.info("Defining training arguments...")

# Define training hyperparameters and DeepSpeed config
training_args = TrainingArguments(
    output_dir="./finetuned-deepseek-cot",          # Output directory for checkpoints
    per_device_train_batch_size=2,                  # Batch size per GPU
    gradient_accumulation_steps=4,                  # Accumulate gradients over 4 steps
    num_train_epochs=3,                             # Number of full dataset passes
    logging_steps=50,                               # Log every 50 steps
    save_steps=500,                                 # Save every 500 steps
    save_total_limit=2,                             # Keep last 2 checkpoints
    fp16=True,                                      # Use half precision for speed
    deepspeed="ds_config_zero1.json",               # DeepSpeed config file
    report_to="none"                                # No external logging service
)

logger.info("Creating data collator...")

# Create data collator for language modeling (no masked LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

logger.info("Initializing Trainer...")

# Create Hugging Face Trainer for supervised fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

logger.info("Starting training...")

# Start the training loop
trainer.train()

logger.info("Saving model and tokenizer locally...")

# Define repo ID structure for local Hugging Face-style saving
model_id = "openai-gsm8k-enhanced-DeepSeek-R1-Distill-Qwen-1.5B"
user_id = "eagle0504"
repo_id = os.path.join(user_id, model_id)

# Create local folder and save model and tokenizer
os.makedirs(repo_id, exist_ok=True)
trainer.model.save_pretrained(repo_id)
tokenizer.save_pretrained(repo_id)

logger.info("Model and tokenizer saved to: %s", repo_id)

