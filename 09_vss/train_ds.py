"""
Fine-tune LongCat-Flash-Omni for Video-Speech-to-Speech using LoRA, DeepSpeed, and TRL.

This script supports:
- Video (.mp4) + Audio (.wav/.mp3) inputs → Audio (.wav/.mp3) output
- LoRA for parameter-efficient fine-tuning
- DeepSpeed ZeRO for memory optimization
- Optional W&B tracking (set WANDB_API_KEY)
- Optional HuggingFace Hub upload (set HF_TOKEN)

Data Structure:
    data/
    ├── train/
    │   ├── 01/
    │   │   ├── in.mp4 (or in.MOV)
    │   │   ├── in.wav
    │   │   └── out.wav
    │   ├── 02/
    │   │   ├── in.mp4 (or in.MOV)
    │   │   ├── in.wav
    │   │   └── out.wav
    │   └── ...
    └── test/
        └── (same structure)

WARNING: LongCat-Flash-Omni is a 560B parameter model (27B activated).
This requires substantial hardware even with LoRA + DeepSpeed ZeRO-3.
Minimum: 8x H100 (80GB) or 8x H200 (141GB) for training.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login, HfApi
import torchaudio
import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_wandb() -> bool:
    """
    Setup Weights & Biases tracking (optional).
    Returns True if W&B is available and configured, False otherwise.
    """
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_api_key:
        logger.info("⚠️  W&B tracking disabled (WANDB_API_KEY not set)")
        logger.info("   To enable: export WANDB_API_KEY=your_key_here")
        logger.info("   Get key from: https://wandb.ai/authorize")
        return False

    try:
        import wandb
        wandb.login(key=wandb_api_key)
        logger.info("✅ W&B tracking enabled")
        return True
    except ImportError:
        logger.warning("⚠️  W&B not installed. Install with: uv add wandb")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize W&B: {e}")
        return False


def setup_huggingface_auth() -> bool:
    """
    Setup Hugging Face authentication (optional).
    Returns True if successful, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            logger.info("✅ Successfully authenticated with Hugging Face using token")
            return True
        except Exception as e:
            logger.warning(f"❌ Failed to authenticate with provided token: {e}")

    try:
        api = HfApi()
        user_info = api.whoami()
        if user_info:
            logger.info(f"✅ Already authenticated with Hugging Face as: {user_info['name']}")
            return True
    except Exception as e:
        logger.warning(f"⚠️  No valid Hugging Face authentication found: {e}")

    logger.info("ℹ️  To enable model pushing to HF Hub:")
    logger.info("   1. Get token from: https://huggingface.co/settings/tokens")
    logger.info("   2. Set environment variable: export HF_TOKEN=your_token_here")
    logger.info("   3. Or run: huggingface-cli login")

    return False


def load_video_frames(video_path: str, max_frames: int = 16) -> torch.Tensor:
    """
    Load and preprocess video frames.

    Args:
        video_path: Path to video file (.mp4)
        max_frames: Maximum number of frames to extract

    Returns:
        Tensor of shape (max_frames, C, H, W) with normalized frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            # Resize to standard size (e.g., 224x224)
            frame = torch.nn.functional.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            frames.append(frame)

    cap.release()

    # Stack frames: (num_frames, C, H, W)
    return torch.stack(frames) if frames else torch.zeros((max_frames, 3, 224, 224))


def load_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 30.0) -> torch.Tensor:
    """
    Load and preprocess audio file.

    Args:
        audio_path: Path to audio file (.wav or .mp3)
        sample_rate: Target sample rate
        max_duration: Maximum duration in seconds

    Returns:
        Tensor of shape (1, num_samples) with normalized audio
    """
    waveform, sr = torchaudio.load(audio_path)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Truncate or pad to max_duration
    max_samples = int(sample_rate * max_duration)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform


def load_dataset_from_folder(data_dir: str, split: str = "train") -> Dataset:
    """
    Load video-speech-to-speech dataset from folder structure.

    Args:
        data_dir: Root data directory
        split: "train" or "test"

    Returns:
        HuggingFace Dataset with video, input audio, and output audio
    """
    logger.info(f"Loading {split} data from: {data_dir}/{split}")

    data_path = Path(data_dir) / split
    samples = []

    # Iterate through sample folders (01, 02, 03, ...)
    for sample_folder in sorted(data_path.iterdir()):
        if not sample_folder.is_dir():
            continue

        # Find video file (try in.mp4 first, then in.MOV)
        video_file = None
        for video_name in ["in.mp4", "in.MOV"]:
            if (sample_folder / video_name).exists():
                video_file = sample_folder / video_name
                break

        # Find input audio (in.wav or in.mp3)
        input_audio_file = None
        for ext in [".wav", ".mp3"]:
            if (sample_folder / f"in{ext}").exists():
                input_audio_file = sample_folder / f"in{ext}"
                break

        # Find output audio (out.wav or out.mp3)
        output_audio_file = None
        for ext in [".wav", ".mp3"]:
            if (sample_folder / f"out{ext}").exists():
                output_audio_file = sample_folder / f"out{ext}"
                break

        # Validate all files exist
        if video_file is None:
            logger.warning(f"Missing video file (in.mp4 or in.MOV) in {sample_folder}")
            continue
        if input_audio_file is None:
            logger.warning(f"Missing input audio (in.wav or in.mp3) in {sample_folder}")
            continue
        if output_audio_file is None:
            logger.warning(f"Missing output audio (out.wav or out.mp3) in {sample_folder}")
            continue

        samples.append({
            "sample_id": sample_folder.name,
            "video_path": str(video_file),
            "input_audio_path": str(input_audio_file),
            "output_audio_path": str(output_audio_file),
        })

    logger.info(f"Loaded {len(samples)} samples from {split} set")

    if len(samples) == 0:
        raise ValueError(f"No valid samples found in {data_path}")

    return Dataset.from_list(samples)


def preprocess_function(examples: Dict) -> Dict:
    """
    Preprocess batch of examples: load video frames and audio waveforms.

    NOTE: In a real implementation, you would need to integrate this with
    LongCat-Flash-Omni's specific preprocessing pipeline for video/audio encoding.
    """
    batch = {
        "video_frames": [],
        "input_audio": [],
        "output_audio": [],
    }

    for video_path, input_audio_path, output_audio_path in zip(
        examples["video_path"],
        examples["input_audio_path"],
        examples["output_audio_path"]
    ):
        # Load video frames
        video_frames = load_video_frames(video_path)
        batch["video_frames"].append(video_frames)

        # Load input audio
        input_audio = load_audio(input_audio_path)
        batch["input_audio"].append(input_audio)

        # Load output audio (target)
        output_audio = load_audio(output_audio_path)
        batch["output_audio"].append(output_audio)

    return batch


def load_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load LongCat-Flash-Omni model with optimizations.

    WARNING: This is a 560B parameter model. Loading it requires significant resources.
    """
    logger.info(f"Loading base model: {model_name}")
    logger.info("⚠️  WARNING: LongCat-Flash-Omni is a 560B parameter model!")
    logger.info("   This requires 8x H100/H200 GPUs minimum for training.")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,  # Required for custom model code
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("   Make sure you have access to the model and sufficient resources.")
        raise


def apply_lora(model) -> PeftModel:
    """
    Apply LoRA PEFT to the model.

    For a 560B model, LoRA is essential to make training feasible.
    """
    logger.info("Applying LoRA to model...")

    # LoRA configuration for large multimodal models
    # Target attention and MLP layers
    lora_config = LoraConfig(
        r=32,  # Higher rank for large model
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        peft_model.print_trainable_parameters()

    logger.info("✅ LoRA applied successfully")
    return peft_model


def train(
    peft_model,
    train_dataset,
    output_dir: str,
    hf_user: str,
    push_to_hub: bool = True,
    use_wandb: bool = False,
) -> None:
    """
    Train the model using DeepSpeed and optionally push to HF Hub.
    """
    # Check HF authentication if push_to_hub is enabled
    can_push_to_hub = False
    if push_to_hub:
        can_push_to_hub = setup_huggingface_auth()
        if not can_push_to_hub:
            logger.warning("⚠️  Will train without pushing to HF Hub")

    # Determine reporting destination
    if use_wandb:
        report_to = ["wandb", "tensorboard"]
        logger.info("📊 Logging to: Weights & Biases + TensorBoard")
    else:
        report_to = ["tensorboard"]
        logger.info("📊 Logging to: TensorBoard only")

    # DeepSpeed-enabled training configuration
    training_args = TrainingArguments(
        # DeepSpeed configuration
        deepspeed="./ds_config.json",

        # Training hyperparameters
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Very small due to model size
        gradient_accumulation_steps=32,  # Large accumulation for effective batch size
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,
        dataloader_pin_memory=True,

        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        # Output settings
        overwrite_output_dir=True,
        report_to=report_to if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0 else [],

        # Hub settings
        push_to_hub=can_push_to_hub,
        hub_model_id=f"{hf_user}/{output_dir}" if can_push_to_hub else None,

        # DeepSpeed specific
        remove_unused_columns=False,

        # W&B settings
        run_name=f"longcat-vss-{os.getenv('USER', 'user')}" if use_wandb else None,
    )

    logger.info("Initializing Trainer...")
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Batch size per GPU: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total epochs: {training_args.num_train_epochs}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)

    # NOTE: In a real implementation, you would need a custom Trainer or data collator
    # that properly handles the multimodal inputs (video + audio) and outputs (audio)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        # data_collator=custom_collator,  # Would need custom collator for multimodal data
    )

    # Train the model
    logger.info("🚀 Starting training...")
    trainer.train()
    logger.info("✅ Training completed!")

    # Save the model (only on main process)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        trainer.save_model(training_args.output_dir)
        logger.info(f"✅ Model saved to {training_args.output_dir}")

        # Push to hub only if authentication is available
        if can_push_to_hub:
            try:
                trainer.push_to_hub()
                logger.info(f"✅ Model pushed to hub: {hf_user}/{output_dir}")
            except Exception as e:
                logger.warning(f"❌ Failed to push to hub: {e}")
        else:
            logger.info("ℹ️  Skipping hub upload (no authentication)")


def main() -> None:
    """Main entry point for fine-tuning."""
    model_name = "meituan-longcat/LongCat-Flash-Omni"
    output_dir = "longcat-flash-omni-vss-lora"
    hf_user = os.getenv("HF_USER", "your-username")
    data_dir = "./data"

    # Configuration options from environment variables
    push_to_hub = os.getenv("PUSH_TO_HUB", "true").lower() == "true"

    # Setup W&B (optional)
    use_wandb = setup_wandb()

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    # Print configuration (only on main process)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("🚀 LongCat-Flash-Omni Video-Speech-to-Speech Fine-tuning")
        logger.info("=" * 60)
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   HF User: {hf_user}")
        logger.info(f"   Data Directory: {data_dir}")
        logger.info(f"   Push to Hub: {push_to_hub}")
        logger.info(f"   W&B Tracking: {'Enabled' if use_wandb else 'Disabled'}")
        logger.info("=" * 60)
        logger.info("⚠️  WARNING: This model requires 8x H100/H200 GPUs minimum!")
        logger.info("=" * 60)

    # Load dataset
    try:
        train_dataset = load_dataset_from_folder(data_dir, split="train")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        logger.error(f"   Make sure data exists at: {data_dir}/train/")
        raise

    # Load and prepare model
    logger.info("Loading model (this may take several minutes)...")
    model = load_model(model_name)

    # Apply LoRA
    peft_model = apply_lora(model)

    # Train the model
    train(peft_model, train_dataset, output_dir, hf_user, push_to_hub, use_wandb)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
