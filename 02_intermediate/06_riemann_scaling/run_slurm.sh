#!/bin/bash
# =============================================================================
# Riemann-Siegel Z(t) precision/scaling benchmark -- SLURM batch script
# =============================================================================
#
# Named run_slurm.sh rather than run_deepspeed.sh on purpose: this lab calls
# no `deepspeed` launcher, because there is nothing to distribute. The whole
# workload is one (points x N) elementwise cosine and a reduction -- a single
# card saturates on memory bandwidth long before a second card would help, and
# wrapping it in a distributed launcher would be the cargo cult CLAUDE.md
# warns about. The benchmark is registered with launcher="python" in
# runpod/runpod_ctl.py for the same reason.
#
# Submit:   sbatch run_slurm.sh
# Watch:    squeue -u $USER   then   tail -f logs/riemann_<jobid>.out
# Cancel:   scancel <jobid>
# -----------------------------------------------------------------------------

#SBATCH --job-name=riemann_scaling      # shows up in squeue
#SBATCH --nodes=1                       # one node; see the note above
#SBATCH --ntasks-per-node=1             # one process -- no distribution
#SBATCH --gres=gpu:1                    # ONE GPU is the point of this lab
#SBATCH --cpus-per-task=8               # grid construction is cheap but not free
#SBATCH --mem=32G                       # the grid is float64 and host-resident
#SBATCH --time=01:00:00                 # full 8-height sweep measured ~25 min
#SBATCH --partition=h200-low            # adjust to your cluster's partitions
#SBATCH --output=logs/riemann_%j.out     # %j expands to the job id
#SBATCH --error=logs/riemann_%j.err

set -euo pipefail

# SLURM does not create the output directory, and a missing one makes the job
# fail before a single line is written -- with no log to explain why.
mkdir -p logs

echo "============================================================"
echo " Riemann-Siegel Z(t): precision reach and GPU scaling"
echo "============================================================"
echo " job      : ${SLURM_JOB_ID:-<interactive>}"
echo " node     : $(hostname)"
echo " started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# Credentials stay COMMENTED. An uncommented `export KEY=<ENTER_KEY_HERE>` is a
# bash syntax error -- '<' is a redirection operator -- so the script would die
# on that line and never reach the benchmark. See CLAUDE.md.
# export WANDB_API_KEY="your_key_here"

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader || echo "  (nvidia-smi unavailable)"

# --max-heights 8 sweeps t = 1e3 .. 1e10. Drop it to 5 for a quick smoke test;
# the script says so itself when the sweep is capped, so a short run is never
# mistaken for a failed one.
uv run benchmark_zeta.py --max-heights 8 --output-dir results

echo "============================================================"
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " results  : $(pwd)/results"
echo "============================================================"
