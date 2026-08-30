#!/usr/bin/env bash
# =============================================================================
# Is multi-GPU NCCL actually working on this machine?
#
#     bash tests/gpu/diagnose_nccl.sh [num_gpus]      # default 2
#
# Run this FIRST when a multi-GPU job hangs. A DeepSpeed run that stops with
#
#     Watchdog caught collective operation timeout: WorkNCCL(... OpType=BROADCAST
#
# tells you a collective never completed. It does not tell you whether the
# fault is in your script or in the box, and those have completely different
# fixes. Ten minutes of NCCL timeout looks identical either way.
#
# So this walks up from the smallest possible test:
#
#   1. topology      -- can the GPUs see each other at all?
#   2. bare NCCL     -- two processes, one all_reduce, no DeepSpeed involved
#   3. NCCL_P2P_DISABLE=1  -- the usual fix on virtualised/rented hosts where
#                             peer-to-peer is advertised but not functional
#   4. NCCL_SHM_DISABLE=1  -- the usual fix when /dev/shm is too small, which
#                             is common in containers
#
# If step 2 fails and step 3 passes, the box needs NCCL_P2P_DISABLE=1 and every
# script here is innocent. If step 2 passes, the box is fine and the fault is
# in the training script.
#
# Skips cleanly with exit 0 when fewer than 2 GPUs are visible.
# =============================================================================
set -uo pipefail

GPUS="${1:-2}"
PY_BIN="${PYTHON:-python}"
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN=python3

if ! "$PY_BIN" -c "import torch" 2>/dev/null; then
    echo "[skip] $PY_BIN cannot import torch. Exiting 0."; exit 0
fi
VISIBLE=$("$PY_BIN" -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$VISIBLE" -lt 2 ]; then
    echo "[skip] need >=2 GPUs, saw $VISIBLE. Exiting 0."; exit 0
fi

echo "=============================================================================="
echo "  NCCL diagnosis — $VISIBLE GPUs visible, testing with $GPUS"
echo "=============================================================================="

echo ""
echo "--- 1. topology -------------------------------------------------------------"
nvidia-smi topo -m 2>&1 | head -12
echo ""
echo "    /dev/shm: $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2" total, "$4" free"}')"
echo "    (a 64 MB /dev/shm is the container default and is a classic NCCL"
echo "     failure cause — NCCL falls back to shared memory between local ranks)"

# The smallest possible collective. If this hangs, nothing built on top can work.
cat > /tmp/_nccl_probe.py <<'PYEOF'
import os, sys, datetime, torch, torch.distributed as dist

# A short timeout is the entire point: the default is 10 minutes, which turns a
# diagnostic into a coffee break. 60s is far longer than a working all_reduce
# of 8 floats needs.
dist.init_process_group(
    backend="nccl", timeout=datetime.timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()

# Bind THIS rank to ITS OWN device before allocating anything. Skipping this is
# the single most common multi-GPU bug: torch.device("cuda") means cuda:0 for
# every rank, so all ranks pile onto GPU 0 and NCCL deadlocks.
torch.cuda.set_device(rank % torch.cuda.device_count())
t = torch.ones(8, device="cuda") * (rank + 1)
dist.all_reduce(t)
expected = sum(range(1, world + 1))
ok = bool((t == expected).all().item())
print(f"  rank {rank}/{world} on cuda:{torch.cuda.current_device()} "
      f"({torch.cuda.get_device_name(torch.cuda.current_device())}) "
      f"all_reduce -> {t[0].item():.0f} (want {expected}) {'OK' if ok else 'WRONG'}",
      flush=True)
dist.barrier()
dist.destroy_process_group()
sys.exit(0 if ok else 1)
PYEOF

echo ""
echo "--- 2. bare NCCL all_reduce, no DeepSpeed -----------------------------------"
timeout 150 "$PY_BIN" -m torch.distributed.run --nproc_per_node="$GPUS" \
    --master_port=29677 /tmp/_nccl_probe.py > /tmp/_probe_plain.txt 2>&1
RC_PLAIN=$?
grep -viE '^\[W|warn|^\s*$' /tmp/_probe_plain.txt | tail -6
echo "    plain rc=$RC_PLAIN"

echo ""
echo "--- 3. same, with NCCL_P2P_DISABLE=1 ----------------------------------------"
NCCL_P2P_DISABLE=1 timeout 150 "$PY_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" --master_port=29678 /tmp/_nccl_probe.py \
    > /tmp/_probe_RC_NOP2P.txt 2>&1
RC_NOP2P=$?
grep -viE '^\[W|warn|^\s*$' /tmp/_probe_RC_NOP2P.txt | tail -6
echo "    NCCL_P2P_DISABLE=1 rc=$RC_NOP2P"

echo ""
echo "--- 4. same, with NCCL_SHM_DISABLE=1 ----------------------------------------"
NCCL_SHM_DISABLE=1 timeout 150 "$PY_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" --master_port=29679 /tmp/_nccl_probe.py \
    > /tmp/_probe_RC_NOSHM.txt 2>&1
RC_NOSHM=$?
grep -viE '^\[W|warn|^\s*$' /tmp/_probe_RC_NOSHM.txt | tail -6
echo "    NCCL_SHM_DISABLE=1 rc=$RC_NOSHM"

echo ""
echo "=============================================================================="
echo "  VERDICT"
echo "=============================================================================="
if [ "$RC_PLAIN" -eq 0 ]; then
    echo "  NCCL works on this box with no special flags."
    echo "  => A DeepSpeed hang here is the SCRIPT's fault, not the machine's."
elif [ "$RC_NOP2P" -eq 0 ]; then
    echo "  Bare NCCL HANGS, but works with NCCL_P2P_DISABLE=1."
    echo "  => The machine advertises peer-to-peer it cannot actually do."
    echo "     Export NCCL_P2P_DISABLE=1 before every multi-GPU run here."
    echo "     The training scripts are innocent."
elif [ "$RC_NOSHM" -eq 0 ]; then
    echo "  Bare NCCL HANGS, but works with NCCL_SHM_DISABLE=1."
    echo "  => /dev/shm is too small for NCCL. Enlarge it, or export"
    echo "     NCCL_SHM_DISABLE=1 before every multi-GPU run here."
else
    echo "  NCCL fails in ALL three configurations."
    echo "  => Multi-GPU is broken on this machine irrespective of this course."
fi
echo "=============================================================================="
