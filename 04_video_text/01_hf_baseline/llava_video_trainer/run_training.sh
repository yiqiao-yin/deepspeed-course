#!/bin/bash

# DeepSpeed Multi-GPU Training Script for LLaVA Video Trainer
# Usage: ./run_training.sh [num_gpus]

set -e

# Configuration
NUM_GPUS=${1:-4}  # Default to 4 GPUs, can be overridden
SCRIPT_NAME="video_training_script.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting LLaVA Video Trainer with DeepSpeed${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  - GPUs: ${NUM_GPUS}"
echo -e "  - Script: ${SCRIPT_NAME}"
echo -e "  - Model: LLaVA (llava-hf/llava-interleave-qwen-7b-hf)"
echo -e "  - Config: Generated internally with 'auto' values"
echo -e "${BLUE}================================================${NC}"
echo

# Check if required files exist
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo -e "${RED}Error: $SCRIPT_NAME not found!${NC}"
    exit 1
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

# Check if required libraries are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
for lib in PIL requests transformers trl; do
    if ! python -c "import $lib" 2>/dev/null; then
        echo -e "${RED}Error: $lib not installed!${NC}"
        echo "Install with: pip install pillow requests transformers trl"
        exit 1
    fi
done
echo -e "${GREEN}✅ All dependencies installed${NC}"
echo

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ $GPU_COUNT -eq 0 ]]; then
    echo -e "${RED}Error: No GPUs detected!${NC}"
    exit 1
fi

if [[ $GPU_COUNT -lt $NUM_GPUS ]]; then
    echo -e "${YELLOW}Warning: Requested $NUM_GPUS GPUs but only $GPU_COUNT available.${NC}"
    echo -e "${YELLOW}Adjusting to use $GPU_COUNT GPUs.${NC}"
    NUM_GPUS=$GPU_COUNT
fi

# Display GPU info
echo -e "${GREEN}🖥️  GPU Information:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo -e "  GPU $line"
done
echo

# Check disk space
echo -e "${YELLOW}Checking disk space...${NC}"
ROOT_FREE=$(df -h / | awk 'NR==2 {print $4}')
echo -e "${GREEN}  Root filesystem: ${ROOT_FREE} available${NC}"
if command -v df &> /dev/null; then
    WORKSPACE_FREE=$(df -h /workspace 2>/dev/null | awk 'NR==2 {print $4}' || echo "N/A")
    if [[ "$WORKSPACE_FREE" != "N/A" ]]; then
        echo -e "${GREEN}  Workspace: ${WORKSPACE_FREE} available${NC}"
    fi
fi
echo

# Create log directory
mkdir -p logs

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo -e "${GREEN}Using GPUs: $CUDA_VISIBLE_DEVICES${NC}"
echo

# Important notes
echo -e "${BLUE}📝 Important Notes:${NC}"
echo -e "  - DeepSpeed config will be ${GREEN}generated automatically${NC}"
echo -e "  - Config uses ${GREEN}'auto'${NC} values synced with TrainingArguments"
echo -e "  - Model size: ${YELLOW}~14GB${NC} (7B parameters in FP16)"
echo -e "  - Expected GPU memory: ${YELLOW}~16-20GB per GPU${NC}"
echo -e "  - Video frames per sample: ${GREEN}5 frames${NC}"
echo -e "  - Training will ${GREEN}extract actual video frames${NC}"
echo -e "  - Disk space will be monitored during training"
echo -e "${BLUE}================================================${NC}"
echo

# Run with Python (script handles DeepSpeed initialization)
echo -e "${GREEN}🎯 Launching LLaVA training...${NC}"
echo -e "${YELLOW}This may take 10-15 minutes for 3 epochs on 4 samples${NC}"
echo

python $SCRIPT_NAME 2>&1 | tee logs/llava_training_$(date +%Y%m%d_%H%M%S).log

# Check if training completed successfully
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}📋 Logs saved to logs/ directory${NC}"
    echo -e "${GREEN}📦 Dataset: ${HF_USER_ID}/llava-video-text-dataset${NC}"
    echo -e "${GREEN}🤖 Model: ${HF_USER_ID}/llava-video-text-model${NC}"
    echo -e "${BLUE}================================================${NC}"
else
    echo
    echo -e "${RED}❌ Training failed! Check the logs for details.${NC}"
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  - Out of GPU memory: Reduce num_frames or batch size"
    echo -e "  - Disk space: Check output for disk space warnings"
    echo -e "  - Rate limiting: Script has retry logic, may need to wait"
    exit 1
fi

echo -e "${GREEN}🎉 All done!${NC}"
