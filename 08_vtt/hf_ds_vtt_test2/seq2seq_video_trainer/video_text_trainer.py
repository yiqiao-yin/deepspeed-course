"""
Fine-tune a video-text-to-text model using DeepSpeed for multi-GPU training.

This script fine-tunes a model on exactly two cat videos and pushes both 
model and dataset to Hugging Face Hub. Uses TRL's SFTTrainer with DeepSpeed 
and Accelerate for efficient multi-GPU training.

Requirements:
    pip install torch datasets transformers trl huggingface_hub accelerate deepspeed

Environment Variables:
    HF_USER_ID: Hugging Face username
    HF_TOKEN: Hugging Face API token
"""

import os
import json
import time
from typing import List, Dict, Any
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments
)
from trl import SFTTrainer
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.errors import HfHubHTTPError


class RetryHandler:
    """Handle retries with exponential backoff for rate limiting."""
    
    @staticmethod
    def exponential_backoff_retry(
        func, 
        max_retries: int = 5, 
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for delay increase
            
        Returns:
            Function result if successful
            
        Raises:
            Exception: If all retries fail
        """
        delay = base_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except HfHubHTTPError as e:
                last_exception = e
                if e.response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries:
                        print(f"⏳ Rate limited. Waiting {delay:.1f}s before retry "
                              f"(attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                        continue
                    else:
                        print("❌ Max retries reached for rate limiting")
                        raise
                else:
                    # Non-rate-limiting error, don't retry
                    print(f"❌ HTTP Error {e.response.status_code}: {e}")
                    raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"⚠️  Error occurred: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                    continue
                else:
                    print("❌ Max retries reached")
                    raise
        
        raise last_exception


class VideoTextTrainer:
    """Video-text model trainer with DeepSpeed support."""
    
    def __init__(self, hf_user_id: str, hf_token: str):
        """
        Initialize the trainer.
        
        Args:
            hf_user_id: Hugging Face user ID
            hf_token: Hugging Face API token
        """
        self.hf_user_id = hf_user_id
        self.hf_token = hf_token
        self.processor = None
        self.retry_handler = RetryHandler()
        self.validate_credentials()
    
    def validate_credentials(self) -> None:
        """Validate Hugging Face credentials."""
        if not self.hf_user_id or not self.hf_token:
            raise EnvironmentError(
                "HF_USER_ID and HF_TOKEN must be set as environment variables."
            )
    
    def create_dataset_dict(self, video_urls: List[str]) -> DatasetDict:
        """
        Create DatasetDict with video samples.

        Args:
            video_urls: List of video URLs (expects 4 URLs)

        Returns:
            DatasetDict with train/validation split

        Raises:
            ValueError: If not exactly 4 video URLs provided
        """
        if len(video_urls) != 4:
            raise ValueError("Exactly four video URLs are required.")

        # Prepare training data with video-question-caption triplets
        data = {
            "video": video_urls,
            "question": [
                "What is in this video?",
                "Can you describe what is happening?",
                "What is in the video?",
                "Describe the video."
            ],
            "caption": [
                "There is a cat in the video.",
                "A cat is present in the scene.",
                ("A gentle breeze rustles the leaves and sways the grape "
                 "cluster softly."),
                ("A gentle breeze rustles the pages of open books on the "
                 "shelves, creating a soft whispering sound.")
            ]
        }

        dataset = Dataset.from_dict(data)
        split_data = dataset.train_test_split(test_size=0.5, seed=42)
        
        return DatasetDict({
            "train": split_data["train"],
            "validation": split_data["test"]
        })

    def create_dataset_readme(self, repo_id: str) -> str:
        """
        Create README content for dataset.
        
        Args:
            repo_id: Repository ID for the dataset
            
        Returns:
            README content as string
        """
        return f"""# {repo_id}

This is a **tiny dataset** with exactly four video samples for training.

- **Field `video`**: Video URLs (MP4/GIF format)
- **Field `question`**: Input prompt/question  
- **Field `caption`**: Target description

## Dataset Structure

| video | question | caption |
|-------|----------|---------|
| sample1.mp4 | What is in this video? | There is a cat in the video. |
| sample2.mp4 | Can you describe what is happening? | A cat is present in the scene. |
| sample3.gif | What is in the video? | A gentle breeze rustles the leaves... |
| sample4.gif | Describe the video. | A gentle breeze rustles the pages... |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```
"""

    def push_dataset_to_hub(
        self, 
        dataset_dict: DatasetDict, 
        repo_id: str
    ) -> None:
        """
        Push dataset to Hugging Face Hub with retry logic.

        Args:
            dataset_dict: Dataset to push
            repo_id: Repository ID for the dataset
        """
        print(f"📤 Pushing dataset to {repo_id} (with retry logic)...")
        
        # First, create the repository if it doesn't exist
        def create_dataset_repo():
            return create_repo(
                repo_id, 
                repo_type="dataset",
                private=False, 
                exist_ok=True, 
                token=self.hf_token
            )
        
        self.retry_handler.exponential_backoff_retry(create_dataset_repo)
        print("✅ Dataset repository created/verified")
        
        # Add delay to avoid immediate rate limiting
        time.sleep(2)
        
        # Push dataset with retry logic
        def push_dataset():
            return dataset_dict.push_to_hub(repo_id, token=self.hf_token)
        
        self.retry_handler.exponential_backoff_retry(push_dataset)
        print("✅ Dataset uploaded successfully")
        
        # Add delay before uploading README
        time.sleep(3)
        
        # Create and upload README with retry logic
        readme_content = self.create_dataset_readme(repo_id)
        
        with open("dataset_README.md", "w", encoding="utf-8") as file:
            file.write(readme_content)
        
        def upload_readme():
            return upload_file(
                path_or_fileobj="dataset_README.md",
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.hf_token
            )
        
        self.retry_handler.exponential_backoff_retry(upload_readme)
        print("✅ Dataset README uploaded successfully")

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for training.
        
        Args:
            examples: Batch of examples to preprocess
            
        Returns:
            Preprocessed examples
        """
        return self.processor(
            text=examples["caption"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    def create_model_readme(
        self, 
        model_repo_id: str, 
        dataset_repo_id: str, 
        base_model: str
    ) -> str:
        """
        Create README content for model.
        
        Args:
            model_repo_id: Repository ID for the model
            dataset_repo_id: Repository ID for the dataset
            base_model: Base model name
            
        Returns:
            README content as string
        """
        return f"""# {model_repo_id}

Fine-tuned model on **video-text dataset** using DeepSpeed.

## Model Details

- **Base model**: {base_model}
- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})
- **Training**: Multi-GPU with DeepSpeed ZeRO Stage 2
- **Task**: Video-text-to-text generation

## Usage

```python
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("{model_repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{model_repo_id}")

# Generate text from video
inputs = processor(text="What is in this video?", return_tensors="pt")
outputs = model.generate(**inputs)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

- DeepSpeed ZeRO Stage 2
- Mixed precision (BF16)
- AdamW optimizer
- Learning rate: 5e-5
"""

    def get_training_arguments(self, deepspeed_config_path: str) -> TrainingArguments:
        """
        Create training arguments with DeepSpeed configuration.
        
        Args:
            deepspeed_config_path: Path to DeepSpeed config file
            
        Returns:
            TrainingArguments configured for DeepSpeed
        """
        return TrainingArguments(
            output_dir="./video_finetune",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            learning_rate=5e-5,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=1,
            report_to=[],
            deepspeed=deepspeed_config_path,
            bf16=True,
            dataloader_pin_memory=False,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            weight_decay=0.01,
        )

    def push_model_to_hub_with_retry(
        self, 
        trainer: SFTTrainer, 
        model_repo_id: str,
        dataset_repo_id: str,
        base_model: str
    ) -> None:
        """
        Push trained model to HuggingFace Hub with retry logic.
        
        Args:
            trainer: Trained SFTTrainer instance
            model_repo_id: Repository ID for the model
            dataset_repo_id: Repository ID for the dataset  
            base_model: Base model name
        """
        print(f"💾 Pushing model to {model_repo_id} (with retry logic)...")
        
        # Create model repository
        def create_model_repo():
            return create_repo(
                model_repo_id, 
                private=False, 
                exist_ok=True, 
                token=self.hf_token
            )
        
        self.retry_handler.exponential_backoff_retry(create_model_repo)
        print("✅ Model repository created/verified")
        
        # Add delay to avoid rate limiting
        time.sleep(5)
        
        # Push model with retry logic
        def push_model():
            return trainer.model.push_to_hub(model_repo_id, token=self.hf_token)
        
        self.retry_handler.exponential_backoff_retry(push_model)
        print("✅ Model uploaded successfully")
        
        # Add delay before pushing processor
        time.sleep(3)
        
        # Push processor with retry logic
        def push_processor():
            return self.processor.push_to_hub(model_repo_id, token=self.hf_token)
        
        self.retry_handler.exponential_backoff_retry(push_processor)
        print("✅ Processor uploaded successfully")
        
        # Add delay before uploading README
        time.sleep(3)
        
        # Upload model README with retry logic
        model_readme = self.create_model_readme(
            model_repo_id, 
            dataset_repo_id, 
            base_model
        )
        
        with open("model_README.md", "w", encoding="utf-8") as file:
            file.write(model_readme)
        
        def upload_model_readme():
            return upload_file(
                path_or_fileobj="model_README.md",
                path_in_repo="README.md",
                repo_id=model_repo_id,
                repo_type="model",
                token=self.hf_token
            )
        
        self.retry_handler.exponential_backoff_retry(upload_model_readme)
        print("✅ Model README uploaded successfully")

    def train_model(
        self, 
        video_urls: List[str], 
        base_model: str = "facebook/nllb-200-distilled-600M",
        deepspeed_config_path: str = "ds_config.json"
    ) -> None:
        """
        Main training pipeline with proper rate limiting.
        
        Args:
            video_urls: List of video URLs for training
            base_model: Base model to fine-tune
            deepspeed_config_path: Path to DeepSpeed configuration
        """
        print("🚀 Starting video-text model training with DeepSpeed...")
        
        # Create and push dataset
        print("📊 Creating dataset...")
        dataset_dict = self.create_dataset_dict(video_urls)
        print(f"Dataset created with {len(dataset_dict['train'])} train samples")
        
        dataset_repo_id = f"{self.hf_user_id}/video-text-dataset"
        print(f"📤 Pushing dataset to {dataset_repo_id}...")
        self.push_dataset_to_hub(dataset_dict, dataset_repo_id)
        
        # Load model and processor
        print(f"🤖 Loading model: {base_model}")
        self.processor = AutoProcessor.from_pretrained(
            base_model, 
            use_auth_token=self.hf_token
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, 
            use_auth_token=self.hf_token
        )
        
        # Preprocess dataset
        print("🔄 Preprocessing dataset...")
        tokenized_dataset = dataset_dict.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        # Setup training
        training_args = self.get_training_arguments(deepspeed_config_path)
        
        print("🏋️ Initializing trainer with DeepSpeed...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            processing_class=self.processor,
            args=training_args,
        )

        # Train the model
        print("🎯 Starting training...")
        trainer.train()
        
        # Save and push model with retry logic
        model_repo_id = f"{self.hf_user_id}/video-text-model"
        self.push_model_to_hub_with_retry(
            trainer, 
            model_repo_id, 
            dataset_repo_id, 
            base_model
        )
        
        print("✅ Training completed successfully!")


def create_deepspeed_config(config_path: str = "ds_config.json") -> None:
    """
    Create DeepSpeed configuration file.
    
    Args:
        config_path: Path where to save the config file
    """
    config = {
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-5,
                "warmup_num_steps": 100
            }
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "memory_breakdown": False
    }
    
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    
    print(f"📝 DeepSpeed config saved to {config_path}")


def main() -> None:
    """Main execution function."""
    # Environment setup
    hf_user_id = os.environ.get("HF_USER_ID", "eagle0504")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable must be set!")
    
    # GPU availability check
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPUs not available.")
    
    print(f"🔥 Found {torch.cuda.device_count()} GPU(s)")
    
    # Video URLs for training
    video_urls = [
        ("https://huggingface.co/datasets/diffusion-datasets/"
         "sample-videos/resolve/main/sample1.mp4"),
        ("https://huggingface.co/datasets/diffusion-datasets/"
         "sample-videos/resolve/main/sample2.mp4"),
        "https://assets.rapidata.ai/hailuo-02_scene-motion_0059.gif",
        "https://assets.rapidata.ai/hailuo-02_scene-motion_0008.gif"
    ]
    
    # Create DeepSpeed config
    config_path = "ds_config.json"
    create_deepspeed_config(config_path)
    
    # Initialize trainer and start training
    trainer = VideoTextTrainer(hf_user_id, hf_token)
    trainer.train_model(
        video_urls=video_urls,
        deepspeed_config_path=config_path
    )


if __name__ == "__main__":
    # Set visible GPUs (adjust as needed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()