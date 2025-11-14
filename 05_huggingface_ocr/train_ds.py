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

4. Set W&B API key (optional):
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
import traceback

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
    from PIL import Image, ImageDraw
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    logger.error("Please install required packages")
    sys.exit(1)

# Check for wandb
WANDB_AVAILABLE = False
try:
    import wandb
    if os.environ.get('WANDB_API_KEY'):
        WANDB_AVAILABLE = True
        logger.info("W&B API key found, tracking will be enabled")
    else:
        logger.warning("W&B installed but no API key found. Set WANDB_API_KEY environment variable")
except ImportError:
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
        try:
            item = self.data[idx]
            messages = item.get("messages", [])
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Extract image
            image = None
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            image = content["image"]
                            break
            
            if image is not None:
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                
                inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                return inputs
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            
        # Return default on error
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.zeros(self.max_length, dtype=torch.long),
            "pixel_values": torch.zeros(3, 336, 336)
        }


def generate_deepspeed_config(output_path: str = "ds_config.json") -> Dict[str, Any]:
    """Generate minimal DeepSpeed configuration."""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto", 
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none",
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
        "zero_force_ds_cpu_optimizer": False,
    }
    
    # Only save on main process
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"DeepSpeed config saved to {output_path}")
    
    return config


def load_model_and_processor(args, local_rank):
    """Load model with optional quantization and LoRA."""
    try:
        logger.info(f"Loading model: {args.model_name}")
        
        # Configure quantization
        bnb_config = None
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load processor
        try:
            from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
            processor = Qwen2VLProcessor.from_pretrained(
                args.model_name,
                trust_remote_code=True
            )
            ModelClass = Qwen2VLForConditionalGeneration
        except:
            processor = AutoProcessor.from_pretrained(
                args.model_name,
                trust_remote_code=True
            )
            ModelClass = AutoModelForCausalLM
        
        # Device map
        if not args.no_deepspeed and local_rank != -1:
            device_map = {"": local_rank}
        else:
            device_map = "auto"
        
        # Load model
        model = ModelClass.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not args.use_4bit else None,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Prepare for quantized training
        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
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
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully")
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise


def prepare_dataset(args, processor):
    """Prepare a simple synthetic dataset."""
    logger.info(f"Preparing dataset with {args.max_samples} samples")
    
    sample_data = []
    for i in range(min(10, args.max_samples)):
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
    return VisionDataset(sample_data, processor, args.max_length)


def main():
    """Main training function with better error handling."""
    parser = argparse.ArgumentParser(description="Vision-Language Model Training")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    
    # Dataset arguments
    parser.add_argument("--instruction", type=str, default="Describe this image.")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs")
    
    # W&B arguments
    parser.add_argument("--wandb-project", type=str, default="vision-language-training")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="qwen2-vl")
    parser.add_argument("--wandb-tags", type=str, default="vision,language,finetuning")
    
    # System arguments
    parser.add_argument("--no-deepspeed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Get local rank
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate DeepSpeed config
        if not args.no_deepspeed:
            generate_deepspeed_config()
        
        # Initialize distributed training
        if not args.no_deepspeed and args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            deepspeed.init_distributed()
            logger.info(f"DeepSpeed initialized on rank {args.local_rank}")
            
            # Synchronize all processes
            if dist.is_initialized():
                dist.barrier()
        
        # Load model and processor
        model, processor = load_model_and_processor(args, args.local_rank)
        
        # Synchronize after model loading
        if not args.no_deepspeed and dist.is_initialized():
            dist.barrier()
        
        # Prepare dataset
        train_dataset = prepare_dataset(args, processor)
        
        # Determine report_to setting for W&B
        report_to = "none"
        if WANDB_AVAILABLE and args.local_rank <= 0:
            report_to = "wandb"
            # Let transformers handle W&B initialization
            os.environ["WANDB_PROJECT"] = args.wandb_project
            if args.wandb_run_name:
                os.environ["WANDB_NAME"] = args.wandb_run_name
            os.environ["WANDB_TAGS"] = args.wandb_tags
        
        # Training arguments
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
            report_to=report_to,
            deepspeed="ds_config.json" if not args.no_deepspeed else None,
            local_rank=args.local_rank if not args.no_deepspeed else -1,
            gradient_checkpointing=True,
            ddp_timeout=1800,
        )
        
        # Initialize trainer (no custom callbacks)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=processor,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model (only on main process)
        if args.local_rank <= 0:
            logger.info("Saving model...")
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        
        # Clean shutdown
        if not args.no_deepspeed and dist.is_initialized():
            dist.destroy_process_group()
        
        sys.exit(1)


if __name__ == "__main__":
    main()