import logging
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import Dataset
import torch

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading tokenizer and model...")
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)

logger.info("Loading dataset...")
dataset = load_dataset("eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1")["train"]

# Format prompts
def format_example(example):
    if "<think>" in example['cot']:
        if "<response>" in example['cot']:
            prompt = (
                f"<instruction>This is a math problem.</instruction>"
                f"<question>{example['question']}</question>"
                f"{example['cot']}"
            )
        else:
            prompt = (
                f"<instruction>This is a math problem.</instruction>"
                f"<question>{example['question']}</question>"
                f"{example['cot']}"
                f"<response>{example['answer']}</response>"
            )
    else:
        prompt = (
            f"<instruction>This is a math problem.</instruction>"
            f"<question>{example['question']}</question>"
            f"<think>{example['cot']}</think>"
            f"<response>{example['answer']}</response>"
        )
    return prompt

logger.info("Formatting examples...")
formatted_texts = [format_example(e) for e in dataset]

# Dataset wrapper for GRPO
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

train_dataset = PromptDataset(formatted_texts)

# Reward function: encourage structured formatting
def reward_fn(samples: list[str]) -> list[float]:
    rewards = []
    for text in samples:
        text = text.strip().lower()
        score = 0.0
        for tag in ["<instruction>", "<question>", "<think>", "<response>"]:
            if tag in text and f"</{tag[1:]}" in text:
                score += 0.25
        rewards.append(score)
    return rewards

# GRPOTrainer setup
logger.info("Instantiating GRPOTrainer...")
training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=100,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
    
    # 👇 Add your DeepSpeed config here
    deepspeed="ds_config_zero1.json"  # or "ds_config_zero1.json"
)

logger.info("Starting GRPO training...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
logger.info("Starting training...")
trainer.train()

# Save the model
logger.info("Saving model...")
model_id = "openai-gsm8k-enhanced-deepseek-r1-distill-qwen-1.5b-grpo"
user_id = "eagle0504"
repo_id = os.path.join(user_id, model_id)
os.makedirs(repo_id, exist_ok=True)
trainer.model.save_pretrained(repo_id)
tokenizer.save_pretrained(repo_id)
logger.info("Model and tokenizer saved to: %s", repo_id)
