#!/bin/bash
# SLURM batch script for Parallel Tempering MCMC with DeepSpeed
# This script runs Bayesian neural network inference using replica exchange MCMC
# across multiple GPUs with Weights & Biases tracking

#SBATCH --gres=gpu:2
# Request 2 GPUs for parallel tempering (each GPU runs one temperature replica)

#SBATCH --partition=h200-low
# Submit to the "h200-low" partition/queue

#SBATCH --time=02:00:00
# Maximum wall-clock time: 2 hours
# 2000 iterations takes ~10-15 minutes on 2 GPUs
# 10000 iterations takes ~45-60 minutes on 2 GPUs

#SBATCH --job-name=pt_mcmc
# Job name: Parallel Tempering MCMC

#SBATCH --ntasks-per-node=1
# Number of tasks per node (DeepSpeed handles multi-GPU internally)

#SBATCH --cpus-per-task=8
# Number of CPU cores (8 cores for efficient data loading and MCMC operations)

#SBATCH --mem=32G
# Total memory per node: 32 GB
# MCMC stores samples in memory during sampling

#SBATCH --output=logs/pt_mcmc_%j.out
# Standard output log

#SBATCH --error=logs/pt_mcmc_%j.err
# Standard error log

# Create logs directory if it doesn't exist
mkdir -p logs

# Export Weights & Biases API key for experiment tracking
# Get your API key from: https://wandb.ai/authorize
# Uncomment and add your key to enable W&B tracking
# export WANDB_API_KEY=<ENTER_KEY_HERE>

# If you don't want to use W&B, add --no_wandb flag to the deepspeed command below

# Activate Python virtual environment
# Option 1: Using uv (recommended for faster dependency management)
# Install: pip install uv && uv init . && uv add torch numpy deepspeed wandb matplotlib
# source .venv/bin/activate

# Option 2: Using traditional virtualenv
# Update this path to your actual virtual environment location
source ~/myenv/bin/activate

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Verify WANDB_API_KEY is set (optional)
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not set!"
    echo "   To enable W&B tracking:"
    echo "   1. Get API key from: https://wandb.ai/authorize"
    echo "   2. export WANDB_API_KEY=your_key"
    echo "   Or add --no_wandb flag to disable tracking"
    echo ""
fi

# Launch Parallel Tempering MCMC with DeepSpeed
# Configuration:
# - Model: 3-layer MLP (10 -> 50 -> 50 -> 1)
# - Algorithm: Parallel Tempering MCMC with Metropolis-Hastings
# - Temperatures: Geometric schedule from 1.0 to 10.0
# - Prior: Gaussian N(0, 1.0^2) on all weights
# - Likelihood: Gaussian N(f(X), 0.1^2)
# - Proposal: Gaussian random walk with std=0.01
# - Replica Exchange: Every 10 iterations
# - Output: MCMC samples saved to ./pt_samples/
echo ""
echo "🚀 Starting Parallel Tempering MCMC with DeepSpeed"
echo "   - GPUs: 2"
echo "   - Iterations: 2000"
echo "   - Burn-in: 500"
echo "   - Thinning: 5"
echo "   - Expected samples: ~300 per replica"
echo "   - Output: ./pt_samples/"
echo ""

deepspeed --num_gpus=2 parallel_tempering_mcmc.py \
    --num_iterations 2000 \
    --burn_in 500 \
    --thinning 5 \
    --swap_interval 10 \
    --proposal_std 0.01 \
    --prior_std 1.0 \
    --noise_std 0.1 \
    --batch_size 100 \
    --save_dir ./pt_samples \
    --wandb_project "parallel-tempering-mcmc" \
    --experiment_name "pt_mcmc_2gpu_slurm_${SLURM_JOB_ID}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ MCMC sampling completed successfully!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo ""
    echo "📊 Results saved to: ./pt_samples/"
    echo ""
    echo "📦 Sample files:"
    ls -lh pt_samples/samples_replica_*.pt 2>/dev/null || echo "   (No sample files found)"
    echo ""
    echo "💡 Next Steps:"
    echo "   1. Load samples: torch.load('pt_samples/samples_replica_0.pt')"
    echo "   2. Check W&B dashboard for diagnostics (if enabled)"
    echo "   3. Analyze acceptance rates and convergence"
    echo ""
    echo "🔍 Quick analysis:"
    python -c "
import torch
import os
if os.path.exists('pt_samples/samples_replica_0.pt'):
    s = torch.load('pt_samples/samples_replica_0.pt')
    print(f'   - Samples collected: {len(s[\"samples\"])}')
    print(f'   - Acceptance rate: {s[\"acceptance_rate\"]:.3f}')
    print(f'   - Swap rate: {s[\"swap_acceptance_rate\"]:.3f}')
    print(f'   - Temperature: {s[\"temperature\"]:.3f}')
else:
    print('   (Sample files not found)')
" 2>/dev/null
    echo ""
else
    echo ""
    echo "=================================================="
    echo "❌ MCMC sampling failed!"
    echo "End Time: $(date)"
    echo "=================================================="
    echo ""
    echo "🔍 Check logs for errors:"
    echo "   - Standard output: logs/pt_mcmc_${SLURM_JOB_ID}.out"
    echo "   - Standard error: logs/pt_mcmc_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "   1. Missing dependencies: uv add torch numpy deepspeed wandb matplotlib"
    echo "   2. GPU not available: Check CUDA_VISIBLE_DEVICES"
    echo "   3. WANDB_API_KEY not set: export WANDB_API_KEY=your_key or add --no_wandb"
    echo ""
    exit 1
fi
