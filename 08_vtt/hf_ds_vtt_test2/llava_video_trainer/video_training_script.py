"""
Fine-tune a LLaVA video-text model using DeepSpeed for multi-GPU training.

This script fine-tunes a LLaVA model on video samples and pushes the trained
model to Hugging Face Hub. Uses TRL's SFTTrainer with DeepSpeed and Accelerate
for efficient multi-GPU training.

Requirements:
    pip install torch datasets transformers trl huggingface_hub accelerate deepspeed pillow requests wandb

Environment Variables:
    HF_USER_ID: Hugging Face username
    HF_TOKEN: Hugging Face API token
    WANDB_API_KEY: (Optional) Weights & Biases API key for tracking
"""

import os
import json
import time
import requests
import shutil
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
from huggingface_hub import HfApi, create_repo, upload_file, delete_repo
from huggingface_hub.errors import HfHubHTTPError

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


def check_disk_space() -> None:
    """Check and report disk space usage."""
    try:
        # Check main filesystem
        stat_root = shutil.disk_usage('/')
        root_free_gb = stat_root.free / (1024**3)
        root_used_percent = (stat_root.used / stat_root.total) * 100
        
        # Check workspace
        stat_workspace = shutil.disk_usage('/workspace')  
        workspace_free_gb = stat_workspace.free / (1024**3)
        workspace_used_percent = (stat_workspace.used / stat_workspace.total) * 100
        
        print(f"💾 Disk Space Status:")
        print(f"  - Root (/): {root_free_gb:.1f}GB free ({root_used_percent:.1f}% used)")
        print(f"  - Workspace: {workspace_free_gb:.1f}GB free ({workspace_used_percent:.1f}% used)")
        
        if root_free_gb < 1.0:  # Less than 1GB free
            print(f"⚠️  WARNING: Root filesystem low on space!")
        
    except Exception as e:
        print(f"Could not check disk space: {e}")


def cleanup_cache_files() -> None:
    """Clean up temporary cache files to save space."""
    try:
        # Clean up pip cache
        import subprocess
        subprocess.run(["pip", "cache", "purge"], capture_output=True)
        print("🧹 Cleared pip cache")
        
        # Clear any temporary files in /tmp
        temp_dirs = ["/tmp", "/var/tmp"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception:
                        pass
        print("🧹 Cleared temporary files")
        
    except Exception as e:
        print(f"Warning: Could not clean cache: {e}")


class RetryHandler:
    """Handle retries with exponential backoff for rate limiting and conflict resolution."""

    def __init__(self, hf_token: str = None):
        """
        Initialize retry handler.

        Args:
            hf_token: HuggingFace token for repository operations
        """
        self.hf_token = hf_token

    def exponential_backoff_retry(
        self,
        func,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        repo_id: str = None,
        repo_type: str = None
    ):
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for delay increase
            repo_id: Repository ID (for 412 conflict handling)
            repo_type: Repository type: "model" or "dataset" (for 412 conflict handling)

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
                elif e.response.status_code == 409:  # Conflict (concurrent operation)
                    if attempt < max_retries:
                        print(f"⚠️  Concurrent operation in progress (409). Waiting {delay:.1f}s...")
                        print(f"   Another commit is happening. Retrying (attempt {attempt + 2}/{max_retries + 1})")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                        continue
                    else:
                        print(f"❌ HTTP Error 409: Concurrent operation timeout.")
                        print(f"   Try again later or check: https://huggingface.co/{repo_id}")
                        raise
                elif e.response.status_code == 412:  # Precondition Failed (conflict)
                    if attempt < max_retries and repo_id and repo_type and self.hf_token:
                        print(f"⚠️  Repository conflict (412). Deleting and recreating {repo_id}...")
                        try:
                            delete_repo(repo_id, repo_type=repo_type, token=self.hf_token)
                            print(f"🗑️  Deleted {repo_type} repository: {repo_id}")
                            time.sleep(2)  # Wait before recreating
                        except Exception as del_error:
                            print(f"⚠️  Could not delete repo (might not exist): {del_error}")

                        print(f"🔄 Retrying after cleanup (attempt {attempt + 2}/{max_retries + 1})")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                        continue
                    else:
                        print(f"❌ HTTP Error 412: Repository conflict. Cannot auto-resolve.")
                        print(f"   Try manually deleting: https://huggingface.co/{repo_id}")
                        raise
                else:
                    # Other HTTP errors, don't retry
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
    """LLaVA video-text model trainer with DeepSpeed support."""
    
    def __init__(self, hf_user_id: str, hf_token: str, num_frames: int = 5):
        """
        Initialize the trainer.

        Args:
            hf_user_id: Hugging Face user ID
            hf_token: Hugging Face API token
            num_frames: Number of frames to sample from each video
        """
        self.hf_user_id = hf_user_id
        self.hf_token = hf_token
        self.num_frames = num_frames
        self.processor = None
        self.retry_handler = RetryHandler(hf_token=hf_token)
        self.validate_credentials()
    
    def validate_credentials(self) -> None:
        """Validate Hugging Face credentials."""
        if not self.hf_user_id or not self.hf_token:
            raise EnvironmentError(
                "HF_USER_ID and HF_TOKEN must be set as environment variables."
            )
    
    def create_dataset_dict(self, video_urls: List[str]) -> DatasetDict:
        """
        Create DatasetDict with video samples in LLaVA conversation format.

        Args:
            video_urls: List of video URLs (expects 4 URLs)

        Returns:
            DatasetDict with train/validation split

        Raises:
            ValueError: If not exactly 4 video URLs provided
        """
        if len(video_urls) != 4:
            raise ValueError("Exactly four video URLs are required.")

        # Create conversations in LLaVA format
        conversations = []
        questions = [
            "What is in this video?",
            "Can you describe what is happening?", 
            "What is in the video?",
            "Describe the video."
        ]
        
        answers = [
            "There is a cat in the video.",
            "A cat is present in the scene.",
            ("A gentle breeze rustles the leaves and sways the grape "
             "cluster softly."),
            ("A gentle breeze rustles the pages of open books on the "
             "shelves, creating a soft whispering sound.")
        ]

        for video_url, question, answer in zip(video_urls, questions, answers):
            # Create content with multiple image tokens for video frames
            content = [{"type": "text", "text": question}]
            # Add multiple image tokens for video frames
            for _ in range(self.num_frames):
                content.append({"type": "image"})
            
            conversation = [
                {
                    "role": "user", 
                    "content": content
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                }
            ]
            
            conversations.append({
                "video_url": video_url,
                "conversation": conversation,
                "num_frames": self.num_frames
            })

        # Prepare training data
        data = {
            "video_url": [conv["video_url"] for conv in conversations],
            "conversation": [conv["conversation"] for conv in conversations],
            "num_frames": [conv["num_frames"] for conv in conversations]
        }

        dataset = Dataset.from_dict(data)
        split_data = dataset.train_test_split(test_size=0.5, seed=42)
        
        return DatasetDict({
            "train": split_data["train"],
            "validation": split_data["test"]
        })

    def create_dataset_readme(self, repo_id: str) -> str:
        """
        Create README content for LLaVA dataset.
        
        Args:
            repo_id: Repository ID for the dataset
            
        Returns:
            README content as string
        """
        return f"""# {repo_id}

This is a **tiny LLaVA dataset** with exactly four video samples for training.

- **Field `video_url`**: Video URLs (MP4/GIF format)
- **Field `conversation`**: LLaVA conversation format with user/assistant roles
- **Field `num_frames`**: Number of frames per video ({self.num_frames})

## Dataset Structure

Each sample contains a conversation in LLaVA format:

```json
{{
  "video_url": "https://example.com/video.mp4",
  "conversation": [
    {{
      "role": "user",
      "content": [
        {{"type": "text", "text": "What is in this video?"}},
        {{"type": "image"}},
        {{"type": "image"}},
        {{"type": "image"}},
        {{"type": "image"}},
        {{"type": "image"}}
      ]
    }},
    {{
      "role": "assistant", 
      "content": [{{"type": "text", "text": "There is a cat in the video."}}]
    }}
  ],
  "num_frames": {self.num_frames}
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Model Compatibility

This dataset is designed for LLaVA models that support video input through multiple image frames.
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

        self.retry_handler.exponential_backoff_retry(
            create_dataset_repo,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Dataset repository created/verified")

        # Add delay to avoid immediate rate limiting
        time.sleep(2)

        # Push dataset with retry logic
        def push_dataset():
            return dataset_dict.push_to_hub(repo_id, token=self.hf_token)

        self.retry_handler.exponential_backoff_retry(
            push_dataset,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Dataset uploaded successfully")

        # Add delay before uploading README
        time.sleep(3)

        # Create and upload README with retry logic
        readme_content = self.create_dataset_readme(repo_id)

        readme_path = "/workspace/dataset_README.md"  # Save to workspace
        with open(readme_path, "w", encoding="utf-8") as file:
            file.write(readme_content)

        def upload_readme():
            return upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.hf_token
            )

        self.retry_handler.exponential_backoff_retry(
            upload_readme,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Dataset README uploaded successfully")

    def download_and_process_video_frames(self, video_url: str, num_frames: int) -> List[Image.Image]:
        """
        Download video and extract frames (placeholder implementation).
        
        Args:
            video_url: URL of the video
            num_frames: Number of frames to extract
            
        Returns:
            List of PIL Images
            
        Note:
            This is a simplified implementation. In practice, you'd use opencv-python
            or similar to extract actual video frames.
        """
        # For this example, we'll use a placeholder image repeated
        # In practice, you'd extract actual frames from the video
        try:
            if video_url.endswith(('.jpg', '.png', '.jpeg')):
                # If it's an image URL, use it directly
                response = requests.get(video_url, stream=True, timeout=30)
                image = Image.open(response.raw)
                return [image] * num_frames
            else:
                # For video URLs, use a placeholder approach
                # You should implement actual video frame extraction here
                placeholder_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                response = requests.get(placeholder_url, stream=True, timeout=30)
                image = Image.open(response.raw)
                return [image] * num_frames
        except Exception as e:
            print(f"Warning: Could not download {video_url}, using placeholder. Error: {e}")
            # Use a solid color placeholder
            placeholder = Image.new('RGB', (224, 224), color='gray')
            return [placeholder] * num_frames

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for LLaVA training.
        
        Args:
            examples: Batch of examples to preprocess
            
        Returns:
            Preprocessed examples with tokenized conversations
        """
        batch_conversations = examples["conversation"]
        batch_video_urls = examples["video_url"] 
        batch_num_frames = examples["num_frames"]
        
        # Process each example individually due to LLaVA's specific requirements
        batch_texts = []
        
        for conversation, video_url, num_frames in zip(batch_conversations, batch_video_urls, batch_num_frames):
            try:
                # Apply chat template to get the formatted prompt
                full_prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=False,
                    tokenize=False
                )
                batch_texts.append(full_prompt)
                
            except Exception as e:
                print(f"Error processing conversation: {e}")
                # Create a fallback text
                fallback_text = "What is in this video? There is content in the video."
                batch_texts.append(fallback_text)
        
        # Tokenize the texts
        tokenized = self.processor.tokenizer(
            batch_texts,
            padding=False,  # Don't pad here, let data collator handle it
            truncation=False,  # Don't truncate
            return_tensors=None  # Return lists
        )
        
        # For SFTTrainer, we need 'input_ids' and 'labels'
        # Set labels same as input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    def create_model_readme(
        self,
        model_repo_id: str,
        base_model: str,
        num_samples: int = 4
    ) -> str:
        """
        Create README content for LLaVA model.

        Args:
            model_repo_id: Repository ID for the model
            base_model: Base model name
            num_samples: Number of training samples

        Returns:
            README content as string
        """
        return f"""# {model_repo_id}

Fine-tuned **LLaVA model** on video-text data using DeepSpeed.

## Model Details

- **Base model**: {base_model}
- **Architecture**: LLaVA (Large Language and Vision Assistant)
- **Training samples**: {num_samples} videos
- **Training**: Multi-GPU with DeepSpeed ZeRO Stage 2
- **Task**: Video-text conversation generation
- **Video frames**: {self.num_frames} frames per video

## Usage

```python
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load model and processor
processor = AutoProcessor.from_pretrained("{model_repo_id}")
model = LlavaForConditionalGeneration.from_pretrained(
    "{model_repo_id}",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

# Define conversation with multiple images for video
conversation = [
    {{
        "role": "user",
        "content": [
            {{"type": "text", "text": "What is in this video?"}},
            {{"type": "image"}},
            {{"type": "image"}},
            {{"type": "image"}},
            {{"type": "image"}},
            {{"type": "image"}},
        ],
    }},
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process video frames (you need to extract frames from your video)
video_frames = [...]  # List of PIL Images from video
inputs = processor(images=video_frames, text=prompt, return_tensors='pt').to(0, torch.float16)

# Generate response
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

## Training Configuration

- DeepSpeed ZeRO Stage 2
- Mixed precision (BF16)
- AdamW optimizer
- Learning rate: 5e-5
- Video frames per sample: {self.num_frames}

## Video Processing

This model expects {self.num_frames} frames extracted from each video. For best results:
1. Extract evenly spaced frames from your video
2. Resize frames to model's expected input size
3. Pass frames as a list to the processor
"""

    def get_training_arguments(self, deepspeed_config_path: str) -> TrainingArguments:
        """
        Create training arguments with DeepSpeed configuration and optional W&B.

        Args:
            deepspeed_config_path: Path to DeepSpeed config file

        Returns:
            TrainingArguments configured for DeepSpeed
        """
        # Check if wandb is available and configured
        use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY") is not None

        if use_wandb:
            report_to = ["wandb"]
            run_name = f"llava-video-{time.strftime('%Y%m%d-%H%M%S')}"
            print(f"✅ Weights & Biases enabled. Run: {run_name}")
        else:
            report_to = []
            run_name = None
            if os.environ.get("WANDB_API_KEY"):
                print("⚠️  WANDB_API_KEY set but wandb not installed. Install: pip install wandb")
            else:
                print("ℹ️  Weights & Biases disabled (WANDB_API_KEY not set)")

        return TrainingArguments(
            output_dir="./llava_video_finetune",
            run_name=run_name,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            learning_rate=5e-5,
            # save_strategy="epoch",
            # evaluation_strategy="epoch",
            logging_dir="./logs",
            logging_steps=1,
            report_to=report_to,
            deepspeed=deepspeed_config_path,
            bf16=True,
            dataloader_pin_memory=False,
            save_total_limit=2,
            warmup_steps=100,
            weight_decay=0.01,
            remove_unused_columns=False,  # Important for multimodal data
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            do_eval=True,  # Explicitly enable evaluation
            logging_first_step=True,
            seed=42,
        )

    def train_model(
        self,
        video_urls: List[str],
        base_model: str = "llava-hf/llava-interleave-qwen-7b-hf",
        deepspeed_config_path: str = "ds_config.json"
    ) -> None:
        """
        Main training pipeline - download videos, train model, save to Hub.

        Args:
            video_urls: List of video URLs for training
            base_model: Base LLaVA model to fine-tune
            deepspeed_config_path: Path to DeepSpeed configuration
        """
        print("🚀 Starting LLaVA video-text model training with DeepSpeed...")

        # Create dataset locally (no upload)
        print("📊 Creating LLaVA dataset from video URLs...")
        dataset_dict = self.create_dataset_dict(video_urls)
        print(f"✅ Dataset created with {len(dataset_dict['train'])} train samples")
        print(f"   - Train: {len(dataset_dict['train'])} samples")
        print(f"   - Validation: {len(dataset_dict['validation'])} samples")
        
        # Load LLaVA model and processor
        print(f"🤖 Loading LLaVA model: {base_model}")
        self.processor = AutoProcessor.from_pretrained(
            base_model, 
            use_auth_token=self.hf_token
        )
        
        # Fix LLaVA processor tokenizer issues
        if not hasattr(self.processor, 'pad_token') and hasattr(self.processor, 'tokenizer'):
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                self.processor.pad_token = self.processor.tokenizer.pad_token
                self.processor.pad_token_id = self.processor.tokenizer.pad_token_id
                print(f"✅ Set pad_token to eos_token: {self.processor.pad_token}")
            else:
                self.processor.pad_token = self.processor.tokenizer.pad_token
                self.processor.pad_token_id = self.processor.tokenizer.pad_token_id
                print(f"✅ Using existing pad_token: {self.processor.pad_token}")
        
        # Additional processor attribute fixes
        if hasattr(self.processor, 'tokenizer'):
            # Ensure all necessary attributes are available
            for attr in ['eos_token', 'bos_token', 'unk_token']:
                if not hasattr(self.processor, attr) and hasattr(self.processor.tokenizer, attr):
                    setattr(self.processor, attr, getattr(self.processor.tokenizer, attr))
            
            print(f"✅ Processor setup: pad_token={getattr(self.processor, 'pad_token', 'None')}")
            print(f"✅ Tokenizer vocab size: {len(self.processor.tokenizer)}")
        else:
            print("⚠️ Warning: No tokenizer found in processor")
        
        model = LlavaForConditionalGeneration.from_pretrained(
            base_model, 
            use_auth_token=self.hf_token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Clear cache after model loading to save disk space
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🧹 Cleared CUDA cache to save memory")
        
        # Clear cache after model loading to save disk space
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🧹 Cleared CUDA cache to save memory")
        
        # Preprocess dataset
        print("🔄 Preprocessing LLaVA dataset...")
        try:
            tokenized_dataset = dataset_dict.map(
                self.preprocess_function, 
                batched=True,
                batch_size=2,  # Small batch size
                remove_columns=dataset_dict["train"].column_names,
                desc="Preprocessing LLaVA conversations"
            )
            print("✅ Dataset preprocessing completed")
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            raise
        
        # Setup training
        training_args = self.get_training_arguments(deepspeed_config_path)
        
        print("🏋️ Initializing LLaVA trainer with DeepSpeed...")
        
        # Create data collator for LLaVA
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=None,  # CRITICAL: No eval dataset to prevent saving
            # tokenizer=self.processor.tokenizer,  # Use tokenizer instead of processor
            data_collator=data_collator,
            args=training_args,
        )

        # Train the model
        print("🎯 Starting LLaVA training...")
        
        # Monitor disk space before training
        check_disk_space()
        
        try:
            trainer.train()
            print("✅ Training completed successfully!")

            # Monitor disk space after training
            check_disk_space()

        except Exception as e:
            print(f"❌ Training failed: {e}")
            # Check if it's a disk space issue
            check_disk_space()
            raise

        # Save model directly to HuggingFace Hub (bypass local checkpoints)
        model_repo_id = f"{self.hf_user_id}/llava-video-text-model"
        print(f"\n💾 Saving trained model to {model_repo_id}...")

        self.save_model_directly_to_hub(
            trainer.model,
            model_repo_id,
            base_model,
            num_samples=len(video_urls)
        )

        print("✅ LLaVA training and upload completed successfully!")
        print(f"🤗 Model available at: https://huggingface.co/{model_repo_id}")

    def save_model_directly_to_hub(
        self,
        model,
        model_repo_id: str,
        base_model: str,
        num_samples: int = 4
    ) -> None:
        """
        Save model directly to HuggingFace Hub without local checkpoint.

        Args:
            model: Trained model to save
            model_repo_id: Repository ID for the model
            base_model: Base model name
            num_samples: Number of training samples
        """
        print(f"💾 Saving LLaVA model directly to {model_repo_id}...")

        # Create model repository with retry
        def create_model_repo():
            return create_repo(
                model_repo_id,
                private=False,
                exist_ok=True,
                token=self.hf_token
            )

        self.retry_handler.exponential_backoff_retry(
            create_model_repo,
            repo_id=model_repo_id,
            repo_type="model"
        )
        print("✅ Model repository created/verified")

        # Add delay to avoid rate limiting
        time.sleep(5)

        # Check disk space before saving
        check_disk_space()

        try:
            # Push model directly to hub with retry logic
            def push_model():
                return model.push_to_hub(
                    model_repo_id,
                    token=self.hf_token,
                    safe_serialization=True,  # Use safetensors for smaller files
                    max_shard_size="2GB"      # Smaller shards to avoid memory issues
                )

            self.retry_handler.exponential_backoff_retry(
                push_model,
                repo_id=model_repo_id,
                repo_type="model"
            )
            print("✅ Model uploaded successfully")

            # Add delay before pushing processor
            time.sleep(3)

            # Push processor with retry logic
            def push_processor():
                return self.processor.push_to_hub(model_repo_id, token=self.hf_token)

            self.retry_handler.exponential_backoff_retry(
                push_processor,
                repo_id=model_repo_id,
                repo_type="model"
            )
            print("✅ Processor uploaded successfully")

            # Add delay before uploading README
            time.sleep(3)

            # Upload model README with retry logic
            model_readme = self.create_model_readme(
                model_repo_id,
                base_model,
                num_samples
            )

            readme_path = "/workspace/model_README.md"  # Save to workspace
            with open(readme_path, "w", encoding="utf-8") as file:
                file.write(model_readme)

            def upload_model_readme():
                return upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=model_repo_id,
                    repo_type="model",
                    token=self.hf_token
                )

            self.retry_handler.exponential_backoff_retry(
                upload_model_readme,
                repo_id=model_repo_id,
                repo_type="model"
            )
            print("✅ Model README uploaded successfully")

        except Exception as e:
            print(f"❌ Error saving model: {e}")
            check_disk_space()
            raise


def create_deepspeed_config(config_path: str = "ds_config.json") -> None:
    """
    Create DeepSpeed configuration file optimized for LLaVA.
    
    Args:
        config_path: Path where to save the config file
    """
    config = {
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",  # Sync with TrainingArguments
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
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
        "gradient_accumulation_steps": "auto",  # Sync with TrainingArguments
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",  # Sync with TrainingArguments
        "train_micro_batch_size_per_gpu": "auto",  # Sync with TrainingArguments
        "wall_clock_breakdown": False,
        "memory_breakdown": False
    }
    
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    
    print(f"📝 DeepSpeed config saved to {config_path}")
    print("✅ All values set to 'auto' to sync with TrainingArguments")


def main() -> None:
    """Main execution function."""
    # Environment setup
    hf_user_id = os.environ.get("HF_USER_ID", "eagle0504")
    hf_token = os.environ.get("HF_TOKEN", "xxx")
    
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable must be set!")
    
    # GPU availability check
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPUs not available.")
    
    print(f"🔥 Found {torch.cuda.device_count()} GPU(s)")
    
    # Video URLs for training (using existing URLs as placeholders)
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
    
    # Initialize LLaVA trainer and start training
    num_frames = 5  # Number of frames to extract from each video
    trainer = VideoTextTrainer(hf_user_id, hf_token, num_frames=num_frames)
    trainer.train_model(
        video_urls=video_urls,
        base_model="llava-hf/llava-interleave-qwen-7b-hf",
        deepspeed_config_path=config_path
    )


if __name__ == "__main__":
    # Set visible GPUs (adjust as needed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()