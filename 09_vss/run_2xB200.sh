#!/bin/bash
# Quick launch script for training on 2x B200 GPUs

set -e  # Exit on error

echo "=================================================="
echo "🚀 LongCat-Flash-Omni Training on 2x B200"
echo "=================================================="
echo ""

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -ne 2 ]; then
    echo "⚠️  WARNING: Expected 2 GPUs, found $GPU_COUNT"
    echo "   This script is optimized for 2x B200 GPUs"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check storage
echo ""
echo "Checking storage..."
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "✓ Available storage: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt 2000 ]; then
    echo "⚠️  WARNING: Less than 2TB available!"
    echo "   Model requires ~1.1TB for weights"
    echo "   You have: ${AVAILABLE_GB}GB"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Please increase storage to 2TB+ and try again"
        exit 1
    fi
fi

# Check system RAM
echo ""
echo "Checking system RAM..."
TOTAL_RAM_GB=$(free -g | awk 'NR==2 {print $2}')
echo "✓ Total RAM: ${TOTAL_RAM_GB}GB"

if [ "$TOTAL_RAM_GB" -lt 500 ]; then
    echo "⚠️  WARNING: Less than 512GB RAM detected!"
    echo "   Training may fail with insufficient RAM"
    echo "   Recommended: 1TB+ (you have ${TOTAL_RAM_GB}GB)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check data folder
echo ""
echo "Checking data folder..."
if [ ! -d "./data/train" ]; then
    echo "❌ ERROR: data/train folder not found!"
    echo "   Please create data/train/ with your samples"
    echo "   Structure: data/train/01/{in.mp4, in.wav, out.wav}"
    exit 1
fi

SAMPLE_COUNT=$(ls -1 ./data/train | wc -l)
echo "✓ Found $SAMPLE_COUNT training samples"

if [ "$SAMPLE_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No samples found in data/train/"
    exit 1
fi

# Check if ds_config_2xB200.json exists
echo ""
echo "Checking DeepSpeed config..."
if [ ! -f "./ds_config_2xB200.json" ]; then
    echo "❌ ERROR: ds_config_2xB200.json not found!"
    echo "   This file is required for 2x B200 training"
    exit 1
fi
echo "✓ DeepSpeed config found"

# Set environment variables for optimal performance
echo ""
echo "Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_P2P_LEVEL=NVL  # Enable NVLink for B200

# Optional: Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set - model won't upload to HuggingFace Hub"
    echo "   To enable: export HF_TOKEN=your_token"
else
    echo "✓ HF_TOKEN is set"
fi

# Optional: Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set - no W&B tracking"
    echo "   To enable: export WANDB_API_KEY=your_key"
else
    echo "✓ WANDB_API_KEY is set"
fi

echo ""
echo "=================================================="
echo "🚀 Starting Training"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - GPUs: 2x B200"
echo "  - LoRA rank: 16 (conservative)"
echo "  - Batch size: 1 per GPU"
echo "  - Gradient accumulation: 64"
echo "  - Effective batch size: 128"
echo "  - Dataset: $SAMPLE_COUNT samples"
echo "  - Expected time: ~30-60 min per epoch"
echo ""
echo "Monitoring:"
echo "  - GPU: watch -n 1 nvidia-smi"
echo "  - RAM: watch -n 1 free -h"
echo "  - Logs: tail -f tensorboard_logs/*"
echo ""
echo "=================================================="
echo ""

# Give user 5 seconds to cancel
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Run training
echo ""
echo "🚀 Launching DeepSpeed training..."
echo ""

deepspeed --num_gpus=2 train_ds_2xB200.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Training completed successfully!"
    echo "=================================================="
    echo ""
    echo "Output:"
    echo "  - Model: ./longcat-flash-omni-vss-lora-2xB200/"
    echo "  - Logs: ./tensorboard_logs/"
    echo ""
    if [ -n "$HF_TOKEN" ]; then
        echo "Model uploaded to HuggingFace Hub ✓"
    fi
else
    echo ""
    echo "=================================================="
    echo "❌ Training failed!"
    echo "=================================================="
    echo ""
    echo "Common issues:"
    echo "  1. Out of memory - try reducing gradient accumulation"
    echo "  2. Storage full - check df -h"
    echo "  3. Model download failed - check network connection"
    echo ""
    echo "Check logs above for error details"
    exit 1
fi
