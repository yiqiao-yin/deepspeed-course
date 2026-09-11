"""
Does each rank actually bind to its OWN GPU before the first collective?

    uv run deepspeed --num_gpus=2 tests/gpu/probe_device_binding.py

Run this when a multi-GPU job hangs in a barrier near startup, BEFORE spending
twenty minutes on a real training run. It answers one question in about ten
seconds and needs no dataset, no model and no download.

Why this exists
---------------
`01_basics/03_convnet_cifar10` was fixed so that only rank 0 downloads CIFAR-10
and the others wait on a barrier. The guard worked. The job then died twelve
minutes later *in that barrier*:

    WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
      ran for 721595 milliseconds before timing out

NCCL implements `barrier()` as an all-reduce of a one-element tensor, so it has
to choose a device for that tensor. torch chooses by checking, in order:

    1. barrier(device_ids=[...])
    2. the device bound at init_process_group
    3. CPU
    4. "the current device"

and on step 4, with nothing set, that is **cuda:0 on every rank**. torch's own
source warns this "may use default device 0, causing issues like hang or all
processes creating context on device 0". Both ranks post to the same GPU and
NCCL waits forever.

`deepspeed.initialize()` normally binds the device for you. Any code that runs
*before* initialize() -- a dataset download guard, for instance -- is therefore
exactly the window where this bites. `deepspeed.init_distributed()` does NOT
bind it; there is no `set_device` call anywhere in `deepspeed/comm/comm.py`.

Reading the result
------------------
Expected on a healthy 2-GPU box:

    RANK=0 LOCAL_RANK=0 current_device=0
    RANK=1 LOCAL_RANK=1 current_device=1
    RANK=0 passed the barrier
    RANK=1 passed the barrier

**Both ranks reporting current_device=0** means `set_device` is not taking, and
the fault is upstream of the barrier -- check that the launcher is exporting
LOCAL_RANK (DeepSpeed's launch.py sets RANK, LOCAL_RANK and WORLD_SIZE in each
child's environment; a launcher that only passes `--local_rank` through argv
would leave `os.environ["LOCAL_RANK"]` unset and every rank would read the "0"
default).

**Correct devices but a hang at the barrier** means the device binding is fine
and the problem is the interconnect rather than the script. Go to
`tests/gpu/diagnose_nccl.sh`, which walks up from topology to bare NCCL to
NCCL_P2P_DISABLE=1.

Note there is no timeout= on the barrier. torch 2.11 -- what every lab in this
repo locks -- accepts only group, async_op and device_ids; the per-call timeout
arrives in 2.13. Passing it raises TypeError, which is how this probe's own
first version failed. If this probe hangs rather than returning, that IS the
answer: the ranks are not rendezvousing, and NCCL's 600 s default is what you
are waiting out. Ctrl-C after a few seconds and go to diagnose_nccl.sh.
"""

import os
import sys


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("This probe needs CUDA. It is asking which GPU each rank binds "
              "to, which is not a question on a CPU-only box.")
        sys.exit(1)

    import deepspeed

    rank = os.environ.get("RANK", "<unset>")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        print("⚠️  LOCAL_RANK is not set in the environment. Every rank will "
              "read the '0' default and bind to cuda:0 — which is the bug this "
              "probe looks for. Are you running under the deepspeed launcher?")
    local_rank = int(local_rank_env or "0")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # The line under test. Must happen before ANY collective.
    torch.cuda.set_device(local_rank)

    deepspeed.init_distributed()

    print(f"RANK={rank} LOCAL_RANK={local_rank} "
          f"current_device={torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name(local_rank)})", flush=True)

    if world_size < 2:
        print("Only one rank — a single process cannot race itself. "
              "Re-run with --num_gpus=2 for this to mean anything.", flush=True)
        return

    torch.distributed.barrier(device_ids=[local_rank])
    print(f"RANK={rank} passed the barrier", flush=True)


if __name__ == "__main__":
    main()
