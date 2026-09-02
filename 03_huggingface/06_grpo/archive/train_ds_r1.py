"""
Fine-tunes DeepSeek-R1-Distill-Qwen-1.5B using DeepSpeed on a CoT-enhanced GSM8K dataset.
Optimized for memory-limited environments (e.g., RunPod with 20GB disk).
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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading tokenizer and model...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

logger.info("Loading dataset...")
dataset = load_dataset(
    "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1"
)
train_dataset = dataset["train"]

def format_example(example):
    """Formats each example with reasoning and answer tags."""
    prompt = (
        f"Question: {example['question']}\n"
        f"<think>{example['cot']}</think>\n"
        f"<answer>{example['answer']}</answer>"
    )
    return {"text": prompt}

logger.info("Formatting dataset...")
train_dataset = train_dataset.map(format_example)

def tokenize_function(example):
    """Tokenizes the 'text' field."""
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )

logger.info("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)

max_len = max(len(sample["input_ids"]) for sample in train_dataset)
logger.info("Max tokenized input length: %d", max_len)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

logger.info("Defining training arguments...")
training_args = TrainingArguments(
    output_dir="./checkpoints-tiny",   # Smaller, cleaner save path
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    logging_steps=50,
    save_strategy="no",
    save_steps=1_000_000,              # Saves less often
    save_total_limit=0,                # Keeps only one checkpoint
    fp16=True,
    deepspeed="ds_config_zero1.json",
    report_to="none",                  # No logging clutter
    save_safetensors=True,             # Lighter + faster saves
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

logger.info("Starting training...")
trainer.train()

logger.info("Saving model...")
model_id = "openai-gsm8k-enhanced-deepseek-r1-distill-qwen-1.5b"
user_id = "eagle0504"
repo_id = os.path.join(user_id, model_id)
os.makedirs(repo_id, exist_ok=True)
trainer.model.save_pretrained(repo_id)
tokenizer.save_pretrained(repo_id)
logger.info("Model and tokenizer saved to: %s", repo_id)


