"""
Fine-tune a video-text-to-text model on exactly two cat videos, push both model and dataset to Hugging Face Hub.
Uses TRL's SFTTrainer + Accelerate for multi-GPU training.

Install:
pip install torch datasets transformers trl huggingface_hub accelerate
"""

import os
from typing import List
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, TrainingArguments
from trl import SFTTrainer
from huggingface_hub import HfApi, create_repo, upload_file

# Load Hugging Face credentials
HF_USER_ID = os.environ.get("HF_USER_ID", "eagle0504")
HF_TOKEN = os.environ.get("HF_TOKEN", "xxx")

if not HF_USER_ID or not HF_TOKEN:
    raise EnvironmentError("HF_USER_ID and HF_TOKEN must be set as environment variables.")

# ---- Dataset creation ----
def create_dataset_dict(video_urls: List[str]) -> DatasetDict:
    """
    Create DatasetDict with exactly two cat video samples.

    Args:
        video_urls (List[str]): Two .mp4 video URLs.

    Returns:
        DatasetDict: train/validation split.
    """
    if len(video_urls) != 2:
        raise ValueError("Exactly two video URLs are required.")

    # Text input: questions; output: cat description
    data = {
        "video": video_urls,
        "question": [
            "What is in this video?",
            "Can you describe what is happening?"
        ],
        "caption": [
            "There is a cat in the video.",
            "A cat is present in the scene."
        ]
    }

    dataset = Dataset.from_dict(data)
    split_data = dataset.train_test_split(test_size=0.5, seed=42)
    return DatasetDict({
        "train": split_data["train"],
        "validation": split_data["test"]
    })

# ---- Push dataset to HF Hub ----
def push_dataset_to_hub(dataset_dict: DatasetDict, repo_id: str, token: str):
    """
    Push dataset to Hugging Face Hub with README.md.
    """
    dataset_dict.push_to_hub(repo_id, token=token)

    readme_content = f"""# {repo_id}

This is a **tiny dataset** with exactly two cat videos.

- **Field `video`**: MP4 video URLs
- **Field `question`**: Input prompt/question
- **Field `caption`**: Target description

| video | question | caption |
|-------|----------|---------|
| sample1.mp4 | What is in this video? | There is a cat in the video. |
| sample2.mp4 | Can you describe what is happening? | A cat is present in the scene. |
"""
    upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    with open("README.md", "w") as f:
        f.write(readme_content)

# ---- Main training function ----
def main():
    # Video sources
    video_urls = [
        "https://huggingface.co/datasets/diffusion-datasets/sample-videos/resolve/main/sample1.mp4",
        "https://huggingface.co/datasets/diffusion-datasets/sample-videos/resolve/main/sample2.mp4"
    ]

    # Create dataset
    dataset_dict = create_dataset_dict(video_urls)
    print(dataset_dict)

    # Push dataset
    dataset_repo_id = f"{HF_USER_ID}/two-cat-videos"
    push_dataset_to_hub(dataset_dict, dataset_repo_id, HF_TOKEN)

    # Load model and processor
    model_name = "facebook/nllb-200-distilled-600M"
    processor = AutoProcessor.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

    # Preprocess
    def preprocess_fn(examples):
        return processor(
            text=examples["caption"],
            padding="max_length",
            truncation=True
        )

    tokenized_dataset = dataset_dict.map(preprocess_fn, batched=True)

    # Training args
    training_args = TrainingArguments(
        output_dir="./video_finetune",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=1,
        report_to=[],
        fp16=True
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=processor,
        args=training_args
    )

    # Train
    trainer.train()

    # Save + push model to HF Hub
    model_repo_id = f"{HF_USER_ID}/two-cat-video-model"
    create_repo(model_repo_id, private=False, exist_ok=True, token=HF_TOKEN)
    trainer.model.push_to_hub(model_repo_id, token=HF_TOKEN)
    processor.push_to_hub(model_repo_id, token=HF_TOKEN)

    # Add README.md for model
    model_readme = f"""# {model_repo_id}

Fine-tuned model on **two cat videos**.

- **Base model**: {model_name}
- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})
"""
    with open("README.md", "w") as f:
        f.write(model_readme)
    upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=model_repo_id,
        repo_type="model",
        token=HF_TOKEN
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPUs not available.")
    main()
