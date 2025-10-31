"""
Train a Qwen-based model on GSM8K using GRPOTrainer with a custom reward
function that promotes <think>...</think> tags and character diversity.

Memory-efficient version using LoRA for 8GB GPUs (RTX 3070, etc.)
"""

import logging
from typing import List

from datasets import load_dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reward_num_unique_chars(completions: List[str], **kwargs) -> List[float]:
    """
    Dummy reward function that counts unique characters in completions.

    Args:
        completions (List[str]): List of generated completions.

    Returns:
        List[float]: Reward scores for each completion.
    """
    return [float(len(set(c))) for c in completions]


def reward_has_think_tags(completions: List[str], **kwargs) -> List[float]:
    """
    Reward completions that contain <think>...</think> tags.

    Args:
        completions (List[str]): List of generated completions.

    Returns:
        List[float]: 1.0 if tags present, else 0.0
    """
    return [1.0 if "<think>" in c and "</think>" in c else 0.0 for c in completions]


def reward_combined(completions: List[str], **kwargs) -> List[float]:
    """
    Combine multiple reward functions with weighting.

    Args:
        completions (List[str]): List of generated completions.

    Returns:
        List[float]: Combined weighted reward scores.
    """
    reward_think = reward_has_think_tags(completions, **kwargs)
    reward_unique = reward_num_unique_chars(completions, **kwargs)

    alpha = 0.7  # weight for <think> tags
    beta = 0.3   # weight for character diversity

    return [
        alpha * r1 + beta * r2
        for r1, r2 in zip(reward_think, reward_unique)
    ]


def format_gsm8k_example(example: dict) -> dict:
    """
    Format GSM8K example into prompt/output format.

    Args:
        example (dict): A sample from the GSM8K dataset.

    Returns:
        dict: A dictionary with 'prompt' and 'output'.
    """
    prompt = f"Question: {example['question']}\n<think>{example['cot']}</think>\nAnswer:"
    output = example["answer"]
    return {"prompt": prompt, "output": output}


# def main() -> None:
#     """
#     Main function to train the model with GRPOTrainer and save the result.
#     """
#     logger.info("Loading dataset...")
#     dataset = load_dataset(
#         "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1",
#         split="train",
#     )

#     logger.info("Formatting dataset...")
#     dataset = dataset.map(format_gsm8k_example)
#     dataset = dataset.remove_columns(
#         [col for col in dataset.column_names if col not in ["prompt", "output"]]
#     )

#     logger.info("Initializing trainer...")
#     trainer = GRPOTrainer(
#         model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
#         reward_funcs=reward_combined,
#         train_dataset=dataset,
#     )

#     logger.info("Starting training...")
#     trainer.train()
#     logger.info("Training complete.")

#     # Save the trained model and tokenizer
#     save_path = "./grpo-trained-qwen-gsm8k"
#     logger.info("Saving model and tokenizer to %s", save_path)
#     trainer.model.save_pretrained(save_path)
#     trainer.tokenizer.save_pretrained(save_path)
#     logger.info("Model and tokenizer saved.")

def main() -> None:
    """
    Main function to train the model with GRPOTrainer and save the result.
    Memory-efficient version using LoRA for 8GB GPUs.
    """
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1",
        split="train",
    )

    logger.info("Formatting dataset...")
    dataset = dataset.map(format_gsm8k_example)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "output"]]
    )

    logger.info("Configuring LoRA for memory efficiency...")
    # LoRA configuration for 8GB GPUs
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank (higher = more parameters, more memory)
        lora_alpha=32,  # LoRA scaling factor
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        bias="none",
    )

    logger.info("Configuring training with DeepSpeed and LoRA...")
    # Create GRPO config with reduced batch size for 8GB GPUs
    grpo_config = GRPOConfig(
        output_dir="./grpo-trained-qwen-gsm8k-lora",
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # Reduced from 32 for 8GB GPUs
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        deepspeed="ds_config.json",
        fp16=True,
        max_grad_norm=1.0,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    logger.info("Initializing trainer with DeepSpeed and LoRA...")
    trainer = GRPOTrainer(
        model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
        reward_funcs=reward_combined,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,  # Enable LoRA
    )

    logger.info("Starting training with LoRA...")
    logger.info(f"Trainable parameters: {trainer.model.num_parameters(only_trainable=True):,}")
    logger.info(f"Total parameters: {trainer.model.num_parameters():,}")

    trainer.train()
    logger.info("Training complete.")

    # Save the trained LoRA adapter and tokenizer
    save_path = "./grpo-trained-qwen-gsm8k-lora"
    logger.info("Saving LoRA adapter and tokenizer to %s", save_path)
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)
    logger.info("LoRA adapter and tokenizer saved.")
    logger.info("To use the model, load both the base model and this LoRA adapter.")


if __name__ == "__main__":
    main()
