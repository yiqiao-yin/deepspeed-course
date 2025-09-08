import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Define your Hugging Face Token
hf_token = "xxx"
os.environ["HF_TOKEN"] = hf_token

# Load the dataset
# dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = load_dataset("eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024", split="train")

# Download model and tokenizer
# model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "unsloth/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Set Deepspeed configuration file path
ds_config_path = "ds_config_zero1.json"

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


