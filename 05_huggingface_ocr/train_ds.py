#!/usr/bin/env python3
"""
Minimal Vision-Language Model Fine-tuning with DeepSpeed and W&B Integration

Setup Instructions:
==================
1. Create virtual environment:
   python -m venv venv
   source venv/bin/activate

2. Install PyTorch with CUDA 11.8 (matching your system):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
3. Install other packages:
   pip install transformers datasets accelerate peft bitsandbytes
   pip install deepspeed wandb pillow

4. Set W&B API key:
   export WANDB_API_KEY=your_api_key_here

5. Run training with DeepSpeed (2 GPUs):
   deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora

6. Or run without DeepSpeed (single GPU):
   python train_ds.py --no-deepspeed --use-4bit --use-lora
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import Dataset, DataLoader
    import deepspeed
    from transformers import (
        AutoProcessor, 
        AutoModelForVision2Seq,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig
    )
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from PIL import Image
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    logger.error("Please install required packages")
    sys.exit(1)

# Import wandb and check if API key is set
try:
    import wandb
    WANDB_AVAILABLE = True
    if os.environ.get('WANDB_API_KEY'):
        logger.info("W&B API key found, tracking will be enabled")
    else:
        logger.warning("W&B installed but no API key found. Set WANDB_API_KEY environment variable")
        WANDB_AVAILABLE = False
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("W&B not available. Install with: pip install wandb")


class VisionDataset(Dataset):
    """Simple dataset wrapper for vision-language tasks."""
    
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # For Qwen2-VL, we need to format the messages properly
        messages = item.get("messages", [])
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Extract image from messages
        image = None
        for msg in messages:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content["type"] == "image" and "image" in content:
                        image = content["image"]
                        break
        
        # Process inputs
        if image is not None:
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Flatten batch dimension and prepare labels
            inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
            
            # For training, labels are the same as input_ids
            inputs["labels"] = inputs["input_ids"].clone()
            
            return inputs
        
        # Return empty tensors if no valid data
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.zeros(self.max_length, dtype=torch.long),
            "pixel_values": torch.zeros(3, 336, 336)
        }


def generate_deepspeed_config(output_path: str = "ds_config.json") -> Dict[str, Any]:
    """Generate minimal DeepSpeed configuration without CPU offload to avoid CUDA mismatch."""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto", 
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            # Disable CPU offload to avoid CUDA compilation issues
            "offload_optimizer": {
                "device": "none",  # Changed from "cpu" to "none"
            },
            "offload_param": {
                "device": "none",
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True
        },
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        # Use standard AdamW instead of CPU Adam
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
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
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,  # Disable forced CPU optimizer
        # W&B logging integration
        "wandb": {
            "enabled": WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY') is not None,
            "project": "vision-language-training",
            "group": "qwen2-vl-finetuning"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"DeepSpeed config saved to {output_path}")
    return config


def initialize_wandb(args, local_rank):
    """Initialize W&B with proper configuration."""
    if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY') and local_rank <= 0:
        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.wandb_run_name or 'qwen2vl'}_{timestamp}"
        
        # Initialize W&B
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            tags=args.wandb_tags.split(",") if args.wandb_tags else [],
            config={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
                "use_4bit": args.use_4bit,
                "use_lora": args.use_lora,
                "lora_r": args.lora_r if args.use_lora else None,
                "lora_alpha": args.lora_alpha if args.use_lora else None,
                "lora_dropout": args.lora_dropout if args.use_lora else None,
                "num_gpus": torch.cuda.device_count(),
                "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "mixed_precision": "fp16",
                "deepspeed": not args.no_deepspeed,
                "deepspeed_stage": 2 if not args.no_deepspeed else None,
            }
        )
        
        # Log system information
        if torch.cuda.is_available():
            wandb.log({
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "system/cuda_version": torch.version.cuda,
                "system/pytorch_version": torch.__version__,
            })
        
        logger.info(f"W&B run initialized: {run_name}")
        return True
    return False


def log_gpu_metrics(step=None):
    """Log GPU metrics to W&B."""
    if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
        gpu_metrics = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_metrics.update({
                    f"gpu_{i}/memory_allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    f"gpu_{i}/memory_reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                    f"gpu_{i}/memory_free_gb": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9,
                    f"gpu_{i}/utilization_percent": (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100,
                })
        
        if gpu_metrics:
            if step is not None:
                wandb.log(gpu_metrics, step=step)
            else:
                wandb.log(gpu_metrics)


def load_model_and_processor(args, local_rank):
    """Load model with optional quantization and LoRA."""
    logger.info(f"Loading model: {args.model_name}")
    
    # Configure quantization if requested
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Try to load processor
    try:
        from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
        processor = Qwen2VLProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        
        # Set image size constraints
        if hasattr(processor, 'image_processor'):
            processor.image_processor.size = {"height": 336, "width": 336}
    except:
        # Fallback to AutoProcessor
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
    
    # Determine device map
    if not args.no_deepspeed and local_rank != -1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"
    
    # Load model
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not args.use_4bit else None,
            device_map=device_map,
            trust_remote_code=True
        )
    except:
        # Fallback to auto model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not args.use_4bit else None,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # Prepare model for k-bit training if using quantization
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Applying LoRA configuration")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Log trainable parameters to W&B
        if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY') and local_rank <= 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            wandb.log({
                "model/trainable_params": trainable_params,
                "model/all_params": all_params,
                "model/trainable_percentage": 100 * trainable_params / all_params
            })
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Log initial GPU metrics
    log_gpu_metrics(step=0)
    
    return model, processor


def prepare_dataset(args, processor):
    """Load and prepare dataset."""
    logger.info(f"Preparing dataset with {args.max_samples} samples")
    
    # Create simple synthetic dataset for testing
    sample_data = []
    
    for i in range(min(10, args.max_samples)):
        # Create a simple white image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (336, 336), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Sample {i}", fill='black')
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.instruction},
                    {"type": "image", "image": img}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"This is sample response {i}"}
                ]
            },
        ]
        sample_data.append({"messages": conversation})
    
    logger.info(f"Prepared {len(sample_data)} samples")
    
    # Log dataset info to W&B
    if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
        wandb.log({
            "dataset/num_samples": len(sample_data),
            "dataset/max_length": args.max_length,
            "dataset/instruction": args.instruction
        })
    
    # Create PyTorch dataset
    train_dataset = VisionDataset(sample_data, processor, args.max_length)
    
    return train_dataset


class WandbCallback:
    """Custom callback for additional W&B logging."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional metrics to W&B."""
        if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY') and logs:
            # Log GPU metrics with current step
            log_gpu_metrics(step=state.global_step)
            
            # Log learning rate if available
            if "learning_rate" in logs:
                wandb.log({
                    "training/learning_rate": logs["learning_rate"]
                }, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch completion."""
        if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
            wandb.log({
                "training/epoch": state.epoch,
                "training/global_step": state.global_step
            })


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Minimal Vision-Language Model Training")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, 
                      default="Qwen/Qwen2-VL-2B-Instruct",
                      help="Model name from HuggingFace")
    parser.add_argument("--use-4bit", action="store_true",
                      help="Use 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true",
                      help="Use LoRA for efficient training")
    parser.add_argument("--lora-r", type=int, default=8,
                      help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                      help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                      help="LoRA dropout")
    
    # Dataset arguments
    parser.add_argument("--instruction", type=str,
                      default="Describe this image.",
                      help="Task instruction")
    parser.add_argument("--max-samples", type=int, default=10,
                      help="Max samples to use")
    parser.add_argument("--max-length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--num-epochs", type=int, default=1,
                      help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=10,
                      help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=2,
                      help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=50,
                      help="Save frequency")
    parser.add_argument("--output-dir", type=str, default="outputs",
                      help="Output directory")
    
    # W&B arguments
    parser.add_argument("--wandb-project", type=str, default="vision-language-training",
                      help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                      help="W&B run name (auto-generated if not specified)")
    parser.add_argument("--wandb-group", type=str, default="qwen2-vl",
                      help="W&B group name")
    parser.add_argument("--wandb-tags", type=str, default="vision,language,finetuning",
                      help="W&B tags (comma-separated)")
    
    # System arguments
    parser.add_argument("--no-deepspeed", action="store_true",
                      help="Disable DeepSpeed")
    
    # Handle both --local-rank and --local_rank
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Get local rank from environment if not provided
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Initialize W&B (only on main process)
    wandb_enabled = initialize_wandb(args, args.local_rank)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate DeepSpeed config if needed
    if not args.no_deepspeed:
        generate_deepspeed_config()
    
    # Setup distributed training if using DeepSpeed
    if not args.no_deepspeed and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        logger.info(f"Initialized DeepSpeed on rank {args.local_rank}")
    
    # Load model and processor
    model, processor = load_model_and_processor(args, args.local_rank)
    
    # Prepare dataset
    train_dataset = prepare_dataset(args, processor)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if wandb_enabled else "none",
        run_name=args.wandb_run_name,
        deepspeed="ds_config.json" if not args.no_deepspeed else None,
        local_rank=args.local_rank if not args.no_deepspeed else -1,
        gradient_checkpointing=True,
        logging_first_step=True,
        logging_nan_inf_filter=True,
    )
    
    # Create custom callback for additional W&B logging
    wandb_callback = WandbCallback() if wandb_enabled else None
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor,
        callbacks=[wandb_callback] if wandb_callback else [],
    )
    
    # Log initial state
    if wandb_enabled and args.local_rank <= 0:
        wandb.log({
            "training/started": True,
            "training/total_steps": len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation_steps)
        })
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log final metrics
    if wandb_enabled and args.local_rank <= 0:
        wandb.log({
            "training/completed": True,
            "training/final_loss": train_result.metrics.get("train_loss", 0),
            "training/total_runtime": train_result.metrics.get("train_runtime", 0),
            "training/samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "training/steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
        })
        
        # Log final GPU metrics
        log_gpu_metrics()
    
    # Save model (only on rank 0)
    if args.local_rank <= 0:
        logger.info("Saving model...")
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        
        # Log model artifacts to W&B
        if wandb_enabled:
            wandb.log_artifact(
                args.output_dir,
                type="model",
                name=f"{args.model_name.replace('/', '-')}-finetuned"
            )
    
    # Finish W&B run
    if wandb_enabled and args.local_rank <= 0:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()