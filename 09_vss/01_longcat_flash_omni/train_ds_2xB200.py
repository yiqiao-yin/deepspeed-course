"""
Fine-tune LongCat-Flash-Omni for Video-Speech-to-Speech using LoRA, DeepSpeed, and TRL.

OPTIMIZED FOR: 2x B200 (192GB VRAM each) with 3TB System RAM

This is a CONSERVATIVE configuration designed for 2x B200 GPUs with maximum safety:
- Ultra-aggressive CPU offloading
- Minimal GPU memory footprint
- LoRA training only (~200MB trainable vs 560B total)
- Maximum gradient checkpointing
- Small batch sizes with large gradient accumulation

Hardware Requirements:
- GPUs: 2x B200 (192GB each = 384GB total)
- System RAM: 3TB (for CPU offloading)
- Storage: 2TB+ (model weights ~1.1TB)

This script supports:
- Video (.mp4/.MOV) + Audio (.wav/.mp3) inputs → Audio (.wav/.mp3) output
- LoRA for parameter-efficient fine-tuning
- DeepSpeed ZeRO-3 with aggressive CPU offloading
- Optional W&B tracking (set WANDB_API_KEY)
- Optional HuggingFace Hub upload (set HF_TOKEN)

Data Structure:
    data/
    ├── train/
    │   ├── 01/
    │   │   ├── in.mp4 (or in.MOV)
    │   │   ├── in.wav
    │   │   └── out.wav
    │   └── ...
"""

import logging
import os
from pathlib import Path
from typing import Dict, List
import json
import gc

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


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    # Imported locally so this helper stays self-contained and can be copied
    # between example scripts unchanged. Some of those scripts do not import
    # os/sys at module scope, so these are not always redundant.
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \"torch_adam\": true and "
              "fp16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  DeepSpeed compiles fused CUDA kernels at startup. Without a CUDA")
    print("  toolkit it aborts with a confusing CUDA_HOME error from inside")
    print("  torch's extension loader, so this check stops first.")
    print("\n  This example CANNOT run on CPU: it needs real GPU memory and")
    print("  downloads a large model. Examples 01-04 teach the same mechanics")
    print("  and do run on CPU.")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  No GPU at all? These need none:")
    print("      ./tests/run_all.sh    # the full logic suite, no GPU, no downloads")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py gpus --min-vram 24")
    print("      uv run runpod/runpod_ctl.py run 01_basic_neuralnet")
    print("\n" + bar + "\n")
    sys.exit(1)


def setup_wandb() -> bool:
    """Setup Weights & Biases tracking (optional)."""
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
    """Setup Hugging Face authentication (optional)."""
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


def load_video_frames(video_path: str, max_frames: int = 8) -> torch.Tensor:
    """
    Load and preprocess video frames (REDUCED to 8 frames for memory).

    Args:
        video_path: Path to video file (.mp4 or .MOV)
        max_frames: Maximum number of frames (reduced from 16 to 8)

    Returns:
        Tensor of shape (max_frames, C, H, W) with normalized frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        logger.warning(f"No frames found in {video_path}")
        cap.release()
        return torch.zeros((max_frames, 3, 224, 224))

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            # Resize to standard size (224x224)
            frame = torch.nn.functional.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            frames.append(frame)

    cap.release()

    # Stack frames: (num_frames, C, H, W)
    if len(frames) < max_frames:
        # Pad with zeros if not enough frames
        while len(frames) < max_frames:
            frames.append(torch.zeros(3, 224, 224))

    return torch.stack(frames)


def load_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
    """
    Load and preprocess audio file (REDUCED to 10s for memory).

    Args:
        audio_path: Path to audio file (.wav or .mp3)
        sample_rate: Target sample rate
        max_duration: Maximum duration in seconds (reduced from 30s to 10s)

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
    """Load video-speech-to-speech dataset from folder structure."""
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


def load_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load LongCat-Flash-Omni model with conservative settings for 2x B200.
    """
    logger.info(f"Loading base model: {model_name}")
    logger.info("=" * 70)
    logger.info("⚠️  CONSERVATIVE MODE: Optimized for 2x B200 (384GB)")
    logger.info("   - Using aggressive CPU offloading")
    logger.info("   - Model will be slow but stable")
    logger.info("   - System RAM usage will be high (1-1.5TB)")
    logger.info("=" * 70)

    # Set environment variables for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_cache": False,  # Disable KV cache to save memory
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("✅ Model loaded successfully")

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("   Troubleshooting:")
        logger.error("   1. Check storage: df -h")
        logger.error("   2. Check model exists: https://huggingface.co/meituan-longcat/LongCat-Flash-Omni")
        logger.error("   3. Try: export HF_HUB_ENABLE_HF_TRANSFER=1")
        raise


def apply_lora(model) -> PeftModel:
    """
    Apply LoRA PEFT with CONSERVATIVE settings for 2x B200.

    Reduced rank from 32 to 16 for memory safety.
    """
    logger.info("Applying LoRA to model (CONSERVATIVE mode)...")

    # LoRA configuration - REDUCED rank for safety
    lora_config = LoraConfig(
        r=16,  # Reduced from 32 to 16 for memory
        lora_alpha=32,
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

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    return peft_model


def parse_args() -> "argparse.Namespace":
    """
    Command-line arguments.

    Kept deliberately small: this example is configured mostly through
    environment variables (HF_USER, VSS_DATA_DIR, PUSH_TO_HUB), and the one
    thing a launcher genuinely needs to override is how much work to do.

    `parse_known_args` rather than `parse_args` because the DeepSpeed launcher
    injects --local_rank, which is not ours to validate.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (-1 = train the full "
             "schedule). A cheap dry run on hardware that costs real money: "
             "--max-steps 5 proves the data, model and ZeRO config all work "
             "together without paying for a full epoch.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Injected by the DeepSpeed launcher; accepted and ignored here.",
    )
    return parser.parse_known_args()[0]


def train(
    peft_model,
    train_dataset,
    output_dir: str,
    hf_user: str,
    push_to_hub: bool = True,
    use_wandb: bool = False,
    max_steps: int = -1,
) -> None:
    """Train the model using DeepSpeed with CONSERVATIVE settings for 2x B200."""

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

    # CONSERVATIVE training configuration for 2x B200
    training_args = TrainingArguments(
        # DeepSpeed configuration (conservative)
        deepspeed="./ds_config_2xB200.json",

        # Training hyperparameters (ULTRA CONSERVATIVE)
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,      # Minimum
        gradient_accumulation_steps=64,     # INCREASED from 32 to 64
        learning_rate=5e-5,                 # Reduced from 1e-4
        warmup_ratio=0.05,                  # Slightly more warmup
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,                  # Aggressive gradient clipping

        # Memory optimization (MAXIMUM)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        bf16_full_eval=False,
        dataloader_pin_memory=False,        # Disabled to save memory
        dataloader_num_workers=0,           # No multiprocessing
        max_steps=max_steps,   # -1 = full schedule; see --max-steps

        # Logging and saving (verbose for monitoring)
        logging_steps=1,                    # Log every step
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,                 # Keep only 2 checkpoints
        load_best_model_at_end=False,

        # Output settings
        overwrite_output_dir=True,
        report_to=report_to if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0 else [],

        # Hub settings
        push_to_hub=can_push_to_hub,
        hub_model_id=f"{hf_user}/{output_dir}" if can_push_to_hub else None,
        hub_strategy="every_save",

        # DeepSpeed specific
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,

        # W&B settings
        run_name=f"longcat-vss-2xB200-{os.getenv('USER', 'user')}" if use_wandb else None,
    )

    logger.info("=" * 70)
    logger.info("🚀 CONSERVATIVE Training Configuration for 2x B200")
    logger.info("=" * 70)
    logger.info(f"  Batch size per GPU: {training_args.per_device_train_batch_size}")
    logger.info(f"  Number of GPUs: 2")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * 2 * training_args.gradient_accumulation_steps}")
    logger.info(f"  Total epochs: {training_args.num_train_epochs}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Dataset size: {len(train_dataset)} samples")
    logger.info(f"  Steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * 2)}")
    logger.info("=" * 70)
    logger.info("⚠️  MEMORY MONITORING:")
    logger.info("   Watch GPU: nvidia-smi -l 1")
    logger.info("   Watch RAM: watch -n 1 free -h")
    logger.info("=" * 70)

    # Initialize trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    logger.info("🚀 Starting training (this will be SLOW but STABLE)...")
    logger.info("   Expected: ~30-60 minutes for 1 epoch with 8 samples")

    try:
        trainer.train()
        logger.info("✅ Training completed successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("❌ Out of Memory Error!")
            logger.error("   Try:")
            logger.error("   1. Reduce gradient_accumulation_steps further")
            logger.error("   2. Check: watch -n 1 free -h (RAM usage)")
            logger.error("   3. Ensure no other processes using GPU/RAM")
        raise

    # Save the model (only on main process)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(f"💾 Saving model to {training_args.output_dir}...")
        trainer.save_model(training_args.output_dir)
        logger.info(f"✅ Model saved (~200MB LoRA adapters)")

        # Push to hub only if authentication is available
        if can_push_to_hub:
            try:
                logger.info(f"📤 Uploading to HuggingFace Hub: {hf_user}/{output_dir}")
                trainer.push_to_hub()
                logger.info(f"✅ Model uploaded to hub!")
                logger.info(f"   View at: https://huggingface.co/{hf_user}/{output_dir}")
            except Exception as e:
                logger.warning(f"❌ Failed to push to hub: {e}")
        else:
            logger.info("ℹ️  Skipping hub upload (no authentication)")


def main() -> None:
    """Main entry point for fine-tuning on 2x B200."""
    require_gpu()
    args = parse_args()
    model_name = "meituan-longcat/LongCat-Flash-Omni"
    output_dir = "longcat-flash-omni-vss-lora-2xB200"
    hf_user = os.getenv("HF_USER", "your-username")
    # The sample corpus lives one level up, at 09_vss/data/, because it is
    # SHARED by every subtopic in this folder -- 44 MB of real video+audio that
    # would otherwise be duplicated four times in git history. Override with
    # VSS_DATA_DIR if you keep your own corpus elsewhere.
    data_dir = os.getenv("VSS_DATA_DIR",
                         str(Path(__file__).resolve().parent.parent / "data"))

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
        logger.info("=" * 70)
        logger.info("🚀 LongCat-Flash-Omni VSS Fine-tuning - CONSERVATIVE MODE")
        logger.info("=" * 70)
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   HF User: {hf_user}")
        logger.info(f"   Data Directory: {data_dir}")
        logger.info(f"   Push to Hub: {push_to_hub}")
        logger.info(f"   W&B Tracking: {'Enabled' if use_wandb else 'Disabled'}")
        logger.info("=" * 70)
        logger.info("💪 HARDWARE: 2x B200 (384GB GPU + 3TB RAM)")
        logger.info("⚙️  STRATEGY: LoRA + ZeRO-3 + Aggressive CPU Offload")
        logger.info("⏱️  SPEED: Slow but stable (~30-60 min per epoch)")
        logger.info("💾 MEMORY: High RAM usage expected (1-1.5TB)")
        logger.info("=" * 70)

    # Load dataset
    try:
        train_dataset = load_dataset_from_folder(data_dir, split="train")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        logger.error(f"   Make sure data exists at: {data_dir}/train/")
        logger.error(f"   Expected structure: train/01/{{in.mp4, in.wav, out.wav}}")
        raise

    # Load and prepare model
    logger.info("Loading model (this may take 10-20 minutes for 1.1TB download)...")
    logger.info("   Tip: Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster download")
    model = load_model(model_name)

    # Apply LoRA
    peft_model = apply_lora(model)

    # Train the model
    train(peft_model, train_dataset, output_dir, hf_user, push_to_hub,
          use_wandb, max_steps=args.max_steps)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("=" * 70)
        logger.info("✅ Training completed successfully!")
        logger.info(f"💾 Model saved to: {output_dir}/")
        logger.info(f"📊 Logs saved to: tensorboard_logs/")
        if push_to_hub and setup_huggingface_auth():
            logger.info(f"📤 Model on HF Hub: https://huggingface.co/{hf_user}/{output_dir}")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
