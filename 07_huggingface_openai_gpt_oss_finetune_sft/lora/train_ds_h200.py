"""
Fine-tune OpenAI GPT-OSS-20B on multilingual reasoning using LoRA, TRL SFTTrainer, and DeepSpeed.

OPTIMIZED FOR: 2x H200 SXM (141GB VRAM each)

This configuration maximizes throughput on high-end datacenter GPUs by using larger batch sizes.

This script supports optional integrations:
- HuggingFace Hub: Set HF_TOKEN to push models to the hub
- Weights & Biases: Set WANDB_API_KEY to track experiments

Without these tokens, the script will train and save models locally.
"""

import logging
import os
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login, HfApi

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
        logger.warning("⚠️  W&B not installed. Install with: pip install wandb")
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


def load_data() -> List[Dict]:
    """Load the multilingual thinking dataset."""
    logger.info("Loading dataset: HuggingFaceH4/Multilingual-Thinking")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def init_tokenizer(model_name: str) -> AutoTokenizer:
    """Initialize the tokenizer."""
    logger.info(f"Initializing tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def preview_conversation(tokenizer: AutoTokenizer, messages: List[Dict]) -> None:
    """Print a formatted example conversation."""
    if torch.distributed.get_rank() == 0:  # Only print on main process
        conversation = tokenizer.apply_chat_template(messages, tokenize=False)
        logger.info("=== Example Conversation ===")
        print(conversation)
        logger.info("=" * 50)


def load_model(model_name: str) -> AutoModelForCausalLM:
    """Load the model with bfloat16 (no device_map for DeepSpeed)."""
    logger.info(f"Loading base model: {model_name}")
    # low_cpu_mem_usage=True is critical for large models to avoid OOM during loading
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "low_cpu_mem_usage": True,  # Load incrementally to reduce memory spikes
    }

    # Only log on main process
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        logger.info("Loading with low_cpu_mem_usage=True to prevent OOM")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info("✅ Model loaded successfully")
    return model


def apply_lora(model) -> PeftModel:
    """Apply LoRA PEFT to selected model layers."""
    logger.info("Applying LoRA to model...")

    # LoRA configuration targeting specific MLP expert layers
    lora_config = LoraConfig(
        r=16,  # Increased rank for better quality on large GPUs
        lora_alpha=32,
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        lora_dropout=0.0,  # Must be 0 when using target_parameters
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    if torch.distributed.get_rank() == 0:  # Only print on main process
        peft_model.print_trainable_parameters()

    logger.info("✅ LoRA applied successfully")
    return peft_model


def train(
    peft_model,
    tokenizer,
    dataset,
    output_dir: str,
    hf_user: str,
    push_to_hub: bool = True,
    use_wandb: bool = False,
) -> None:
    """Train and optionally push the model to the hub using DeepSpeed."""

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
    # Optimized for 2x H200 SXM (141GB VRAM each)
    training_args = SFTConfig(
        # DeepSpeed configuration
        deepspeed="./ds_config.json",

        # Training hyperparameters (optimized for H200 - MUCH larger batch sizes!)
        learning_rate=2e-4,
        num_train_epochs=10,
        per_device_train_batch_size=8,  # Increased from 2 to 8 (4x larger!)
        gradient_accumulation_steps=4,   # Reduced from 8 to 4 (faster updates)
        max_length=2048,

        # Optimization settings
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},

        # Memory optimization (can be more aggressive with H200)
        gradient_checkpointing=True,
        dataloader_pin_memory=True,  # Enabled for faster data loading

        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,  # Keep more checkpoints with large disk

        # Output settings
        output_dir=output_dir,
        overwrite_output_dir=True,
        report_to=report_to if torch.distributed.get_rank() == 0 else [],

        # Hub settings - only enable if authentication is available
        push_to_hub=can_push_to_hub,
        hub_model_id=f"{hf_user}/{output_dir}" if can_push_to_hub else None,

        # DeepSpeed specific settings
        remove_unused_columns=False,
        prediction_loss_only=True,

        # W&B settings (if enabled)
        run_name=f"gpt-oss-20b-multilingual-h200-{os.getenv('USER', 'user')}" if use_wandb else None,
    )

    logger.info("Initializing SFTTrainer...")
    logger.info("=" * 60)
    logger.info("Training Configuration (Optimized for H200):")
    logger.info(f"  Batch size per GPU: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * 2 * training_args.gradient_accumulation_steps}")
    logger.info(f"  Total epochs: {training_args.num_train_epochs}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train the model
    logger.info("🚀 Starting training on H200 GPUs...")
    trainer.train()
    logger.info("✅ Training completed!")

    # Save the model (only on main process)
    if torch.distributed.get_rank() == 0:
        trainer.save_model(training_args.output_dir)
        logger.info(f"✅ Model saved to {training_args.output_dir}")

        # Push to hub only if authentication is available
        if can_push_to_hub:
            try:
                trainer.push_to_hub(dataset_name=f"{hf_user}/{output_dir}")
                logger.info(f"✅ Model pushed to hub: {hf_user}/{output_dir}")
            except Exception as e:
                logger.warning(f"❌ Failed to push to hub: {e}")
        else:
            logger.info("ℹ️  Skipping hub upload (no authentication)")


def evaluate(output_dir: str, tokenizer, prompt: str, reasoning_lang: str) -> None:
    """Load the merged model and evaluate on a reasoning prompt (only on main process)."""
    if torch.distributed.get_rank() != 0:
        return

    logger.info("Starting post-training evaluation...")

    base_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    try:
        logger.info("Loading base model for evaluation...")
        base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **base_kwargs)
        model = PeftModel.from_pretrained(base_model, output_dir)
        model = model.merge_and_unload()

        messages = [
            {"role": "system", "content": f"reasoning language: {reasoning_lang}"},
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)
            response = tokenizer.batch_decode(output_ids)[0]

        logger.info("=== Evaluation Results ===")
        print(response)
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def main() -> None:
    """Main entry point for fine-tuning and evaluation."""
    model_name = "openai/gpt-oss-20b"
    output_dir = "gpt-oss-20b-multilingual-reasoner-lora-h200"
    hf_user = os.getenv("HF_USER", "eagle0504")

    # Configuration options from environment variables
    push_to_hub = os.getenv("PUSH_TO_HUB", "true").lower() == "true"
    run_evaluation = os.getenv("RUN_EVALUATION", "true").lower() == "true"

    # Setup W&B (optional)
    use_wandb = setup_wandb()

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    # Print configuration (only on main process)
    if torch.distributed.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("🚀 GPT-OSS-20B Fine-tuning with LoRA on H200 GPUs")
        logger.info("=" * 60)
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   HF User: {hf_user}")
        logger.info(f"   Push to Hub: {push_to_hub}")
        logger.info(f"   Run Evaluation: {run_evaluation}")
        logger.info(f"   W&B Tracking: {'Enabled' if use_wandb else 'Disabled'}")
        logger.info("=" * 60)
        logger.info("Hardware: 2x NVIDIA H200 SXM (141GB VRAM each)")
        logger.info("Optimizations: Larger batch sizes for faster training")
        logger.info("=" * 60)

    # Load data and tokenizer
    dataset = load_data()
    tokenizer = init_tokenizer(model_name)

    # Optional preview (only on main process)
    if len(dataset) > 0:
        preview_conversation(tokenizer, dataset[0]["messages"])

    # Load and prepare model
    model = load_model(model_name)

    # Apply LoRA
    peft_model = apply_lora(model)

    # Train the model
    train(peft_model, tokenizer, dataset, output_dir, hf_user, push_to_hub, use_wandb)

    # Evaluate (only on main process and if enabled)
    if run_evaluation:
        logger.info("Starting evaluation...")
        evaluate(
            output_dir=output_dir,
            tokenizer=tokenizer,
            prompt="¿Cuál es el capital de Australia?",
            reasoning_lang="German",
        )
    else:
        logger.info("Skipping evaluation (disabled)")

    if torch.distributed.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("✅ Training and evaluation completed!")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
