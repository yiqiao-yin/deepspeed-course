"""
Train a Qwen-based model on GSM8K using GRPOTrainer with a custom reward
function that promotes <think>...</think> tags and character diversity.
"""

import logging
from typing import List

from datasets import load_dataset
from trl import GRPOTrainer

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

    logger.info("Initializing trainer with DeepSpeed...")
    trainer = GRPOTrainer(
        model="eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
        reward_funcs=reward_combined,
        train_dataset=dataset,
        deepspeed="ds_config.json"
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # Save the trained model and tokenizer
    save_path = "./grpo-trained-qwen-gsm8k"
    logger.info("Saving model and tokenizer to %s", save_path)
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)
    logger.info("Model and tokenizer saved.")


if __name__ == "__main__":
    main()
