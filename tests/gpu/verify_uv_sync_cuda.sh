#!/usr/bin/env bash
# =============================================================================
# Does `uv sync` produce a torch that can actually SEE this machine's GPU?
#
#     bash tests/gpu/verify_uv_sync_cuda.sh
#
# The examples pin torch through a committed uv.lock. Whatever CUDA build that
# lock resolves to must work against the DRIVER on the machine, and those move
# independently: a driver reporting "CUDA Version: 12.8" in nvidia-smi cannot
# run a wheel built for CUDA 13.0.
#
# This is the reader's actual first experience -- clone, cd, uv sync, run --
# so it is worth one explicit test rather than an assumption.
# =============================================================================
set -uo pipefail
# Mirror everything to VERIFY_LOG as well as stdout, so the RunPod driver
# can ship the file back the same way the other harnesses do.
LOG="${VERIFY_LOG:-/tmp/verify_uvsync.log}"
exec > >(tee "$LOG") 2>&1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "=============================================================================="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /'
echo "  nvidia-smi CUDA: $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
echo "=============================================================================="
cd "$HERE/01_basic_neuralnet" || exit 1
echo "--- uv sync (the locked versions) ---"
uv sync --frozen 2>&1 | tail -2
echo "--- what the lock gave us, and can it see the GPU? ---"
uv run python - <<'PY'
import torch
print(f"  torch            {torch.__version__}")
print(f"  built for CUDA   {torch.version.cuda}")
ok = torch.cuda.is_available()
print(f"  cuda.is_available  {ok}")
if ok:
    print(f"  device           {torch.cuda.get_device_name(0)}")
    x = (torch.ones(8, device='cuda') * 3).sum().item()
    print(f"  real kernel ran  sum={x} (want 24.0)")
    print("  VERDICT: locked torch works on this driver")
else:
    print("  VERDICT: locked torch CANNOT use this GPU -- driver too old for its CUDA")
PY
