#!/bin/bash

# DeepSpeed Multi-GPU Training Script
# Usage: ./run_training.sh [num_gpus]

set -e

# Configuration
NUM_GPUS=${1:-4}  # Default to 4 GPUs, can be overridden
SCRIPT_NAME="video_text_trainer.py"
CONFIG_NAME="ds_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting DeepSpeed Multi-GPU Training${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  - GPUs: ${NUM_GPUS}"
echo -e "  - Script: ${SCRIPT_NAME}"
echo -e "  - DeepSpeed Config: ${CONFIG_NAME}"
echo

# Check if required files exist
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo -e "${RED}Error: $SCRIPT_NAME not found!${NC}"
    exit 1
fi

if [[ ! -f "$CONFIG_NAME" ]]; then
    echo -e "${YELLOW}Warning: $CONFIG_NAME not found. Will be created by script.${NC}"
fi

# Check environment variables
if [[ -z "$HF_USER_ID" ]]; then
    echo -e "${RED}Error: HF_USER_ID environment variable not set!${NC}"
    echo "Please set it with: export HF_USER_ID=your_username"
    exit 1
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo -e "${RED}Error: HF_TOKEN environment variable not set!${NC}"
    echo "Please set it with: export HF_TOKEN=your_token"
    exit 1
fi

# Check if DeepSpeed is installed
if ! python -c "import deepspeed" 2>/dev/null; then
    echo -e "${RED}Error: DeepSpeed not installed!${NC}"
    echo "Install with: pip install deepspeed"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ $GPU_COUNT -lt $NUM_GPUS ]]; then
    echo -e "${YELLOW}Warning: Requested $NUM_GPUS GPUs but only $GPU_COUNT available.${NC}"
    echo -e "${YELLOW}Adjusting to use $GPU_COUNT GPUs.${NC}"
    NUM_GPUS=$GPU_COUNT
fi

# Create log directory
mkdir -p logs

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo -e "${GREEN}Using GPUs: $CUDA_VISIBLE_DEVICES${NC}"

# Run with DeepSpeed launcher
echo -e "${GREEN}🎯 Launching training...${NC}"
deepspeed --num_gpus=$NUM_GPUS $SCRIPT_NAME \
    --deepspeed $CONFIG_NAME \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Check if training completed successfully
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}📋 Logs saved to logs/ directory${NC}"
else
    echo -e "${RED}❌ Training failed! Check the logs for details.${NC}"
    exit 1
fi

# Optional: Clean up temporary files
# rm -f dataset_README.md model_README.md

echo -e "${GREEN}🎉 All done!${NC}"