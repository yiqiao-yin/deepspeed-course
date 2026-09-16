#!/usr/bin/env python3
"""
Train a Mixture-of-Experts layer under DeepSpeed, and watch the router balance.

    # the comparison, on one GPU
    uv run deepspeed --num_gpus=1 train_moe_ds.py --balance bias

    # expert parallelism: the experts themselves live on different ranks
    uv run deepspeed --num_gpus=2 train_moe_ds.py --expert-parallel \\
        --deepspeed_config ds_config_ep.json

    # cheap pipeline check, no real training
    uv run deepspeed --num_gpus=1 train_moe_ds.py --max-steps 20

What this trains
----------------
A small stack of MoE blocks on the grouped synthetic task from `moe.py`: tokens
drawn from latent groups, each group with its own target transformation. The
group is recoverable from the token, so **expert specialisation is possible and
measurable** rather than decorative. `moe.py` explains why that matters and what
happens when it is not true.

The point is not the loss. The point is the three numbers printed beside it:

    max/min    busiest expert over quietest. What expert parallelism pays for.
    dead       experts that received no tokens at all.
    purity     did the router find the task's structure, or just spread evenly?

Two paths, and they are genuinely different code
------------------------------------------------
**Default (`--balance none|aux|bias`)** uses `MoELayer` from `moe.py` -- this
repository's own implementation, written from the DeepSeek-V3 paper. Every
expert lives on every rank; ZeRO shards optimizer state as usual. Use this to
see the *mechanism*: the routing, the balancing, the collapse.

**`--expert-parallel`** replaces it with `deepspeed.moe.layer.MoE`, where the
experts are **partitioned across ranks** and tokens are shipped to them with an
all-to-all. Use this to see the *systems consequence*: why anyone tolerates the
loss penalty that balancing costs.

They are not interchangeable and the script does not pretend they are. The
DeepSpeed layer brings its own gate (top-1/top-2, its own auxiliary loss,
capacity-based token dropping), so `--balance` does not apply to it.

Expert parallelism is not ZeRO, and the difference is the lesson
---------------------------------------------------------------
Both split a model across GPUs. They split different things:

    ZeRO     shards optimizer state, gradients, parameters of the SAME model.
             Every rank still runs every layer, on different DATA.
    EP       shards the EXPERTS. Every rank holds a different SUBSET of the
             model and runs it on tokens routed to it from every other rank.

So EP introduces a communication pattern ZeRO never has: an all-to-all in the
forward pass and another in the backward. And its cost is set by the *busiest*
expert, because every rank waits at that all-to-all. That is the whole reason
load balancing exists -- see the measured table in `moe.py`, where balancing
makes the model strictly worse and is worth it anyway.
"""

import argparse
import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    # Imported locally so this helper stays self-contained and can be copied
    # between example scripts unchanged.
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \"torch_adam\": true and "
              "fp16/bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This example is small enough to run on CPU, and the ROUTING")
    print("  lesson does not need a GPU at all. Two options, both better than")
    print("  renting hardware to learn what a router does:")
    print("\n      uv run moe.py        # the whole comparison, CPU, ~1 minute")
    print("                           # collapse, balancing, specialisation")
    print("\n      ALLOW_CPU=1 deepspeed --num_gpus=1 train_moe_ds.py --max-steps 20")
    print("      (ds_config.json also needs \"torch_adam\": true and fp16 off)")
    print("\n  What you CANNOT do on CPU is --expert-parallel: it partitions the")
    print("  experts across ranks and needs more than one real device.")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  No GPU at all? These need none:")
    print("      ./tests/run_all.sh    # the full logic suite, no GPU, no downloads")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 03_llms/11_moe")
    print("      uv run runpod/runpod_ctl.py run 03_llms/11_moe \\")
    print("          --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    Command-line options.

    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N into every worker's argv, and a strict parser exits 2 with
    "unrecognized arguments" before training starts. CONTRIBUTING.md §3.2.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--balance", default="bias", choices=["none", "aux", "bias"],
                   help="Load-balancing strategy for this repo's MoELayer. "
                        "Ignored under --expert-parallel, which uses "
                        "DeepSpeed's own gate. Default: bias (DeepSeek-V3).")
    p.add_argument("--expert-parallel", action="store_true",
                   help="Partition experts ACROSS RANKS with "
                        "deepspeed.moe.layer.MoE. Needs --num_gpus >= 2 and "
                        "ds_config_ep.json.")
    p.add_argument("--experts", type=int, default=16,
                   help="Routed experts per layer (default: 16).")
    p.add_argument("--top-k", type=int, default=2,
                   help="Experts activated per token. DeepSpeed's MoE layer "
                        "supports only 1 or 2. Default: 2.")
    p.add_argument("--layers", type=int, default=2,
                   help="MoE blocks to stack (default: 2).")
    p.add_argument("--groups", type=int, default=4,
                   help="Latent groups in the synthetic task (default: 4). "
                        "Fewer groups than experts is what makes collapse "
                        "possible -- see moe.py.")
    p.add_argument("--epochs", type=int, default=8,
                   help="Training epochs (default: 8).")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps. -1 means no cap "
                        "(use epochs) -- HuggingFace Trainer's own convention. "
                        "This is what makes `sbatch run_deepspeed.sh "
                        "--max-steps 20` a real dry run rather than a full job.")
    p.add_argument("--tokens", type=int, default=16384,
                   help="Synthetic tokens to generate (default: 16384).")
    p.add_argument("--deepspeed_config", default="ds_config.json",
                   help="DeepSpeed config. Use ds_config_ep.json with "
                        "--expert-parallel.")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted so its "
                        "argument does not cause a parse error.")
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    # Preflight BEFORE the heavy imports, or a CPU-only reader gets a CUDA
    # traceback from inside deepspeed's import chain instead of our message.
    require_gpu()

    import deepspeed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from moe import MoEConfig, MoELayer, routing_purity, synthetic_groups

    # Optional Weights & Biases. Soft by contract: no WANDB_API_KEY, no W&B,
    # and the script runs identically without the package installed.
    use_wandb = False
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(project="deepspeed-course-moe", config=vars(args))
            use_wandb = True
    except ImportError:
        pass

    # LOCAL_RANK from the ENVIRONMENT first. The deepspeed launcher sets both
    # the env var and --local_rank in argv; torchrun sets only the env var, and
    # falling back to argparse's -1 default would bind every rank to cuda:0.
    local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    bar = "=" * 78

    if args.expert_parallel and world_size < 2:
        # Fail loudly. Silently running EP on one rank would "work" and teach
        # exactly nothing, which is the failure mode this course exists to avoid.
        raise SystemExit(
            "\n--expert-parallel partitions experts ACROSS ranks and is "
            f"meaningless with WORLD_SIZE={world_size}.\n"
            "Run it as:  deepspeed --num_gpus=2 train_moe_ds.py "
            "--expert-parallel --deepspeed_config ds_config_ep.json\n")
    if args.expert_parallel and args.experts % world_size != 0:
        raise SystemExit(
            f"\n--experts ({args.experts}) must divide evenly across "
            f"{world_size} ranks under expert parallelism.\n")

    cfg = MoEConfig(n_routed=args.experts, top_k=args.top_k)

    class MoEStack(nn.Module):
        """Embed -> N MoE blocks -> project. Small on purpose."""

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList()
            self.ds_moe = args.expert_parallel
            for _ in range(args.layers):
                if args.expert_parallel:
                    from deepspeed.moe.layer import MoE as DeepSpeedMoE
                    expert = nn.Sequential(
                        nn.Linear(cfg.d_model, cfg.expert_hidden, bias=False),
                        nn.GELU(),
                        nn.Linear(cfg.expert_hidden, cfg.d_model, bias=False))
                    self.blocks.append(DeepSpeedMoE(
                        hidden_size=cfg.d_model,
                        expert=expert,
                        num_experts=args.experts,
                        ep_size=world_size,       # experts split over ALL ranks
                        k=args.top_k,
                        use_residual=False))
                else:
                    self.blocks.append(MoELayer(cfg, balance=args.balance))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            aux = torch.zeros((), device=x.device, dtype=x.dtype)
            for blk in self.blocks:
                if self.ds_moe:
                    # DeepSpeed's MoE returns (output, gate_loss, expert_counts).
                    x, l_aux, _ = blk(x)
                    aux = aux + l_aux
                else:
                    x = blk(x)
                    aux = aux + blk.aux_loss
            self.aux = aux
            return x

    model = MoEStack()

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        mode = ("expert-parallel (deepspeed.moe)" if args.expert_parallel
                else f"single-rank experts, balance={args.balance}")
        print(bar)
        print("  Mixture of Experts under DeepSpeed")
        print(bar)
        print(f"  mode            {mode}")
        print(f"  experts/layer   {args.experts}   top_k {args.top_k}   "
              f"layers {args.layers}")
        print(f"  world size      {world_size}")
        if args.expert_parallel:
            print(f"  experts/rank    {args.experts // world_size}"
                  f"   (partitioned, reached by all-to-all)")
        print(f"  parameters      {total:,}")
        print(bar, flush=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config)

    device = model_engine.device
    micro = model_engine.train_micro_batch_size_per_gpu()

    x, y, group = synthetic_groups(args.tokens, cfg, n_groups=args.groups,
                                   seed=1 + local_rank)
    x, y = x.to(device), y.to(device)

    global_step = 0
    stopped_early = False
    for epoch in range(args.epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for lo in range(0, x.shape[0] - micro + 1, micro):
            idx = perm[lo:lo + micro]
            xb, yb = x[idx], y[idx]
            pred = model_engine(xb.unsqueeze(0)).squeeze(0)
            loss = F.mse_loss(pred, yb) + model_engine.module.aux
            model_engine.backward(loss)
            model_engine.step()

            # The bias update is NOT gradient descent and must not be inside
            # backward(). It runs after the optimizer step, from the counts the
            # forward pass already recorded.
            if not args.expert_parallel:
                for blk in model_engine.module.blocks:
                    blk.update_bias()

            global_step += 1
            if is_main and global_step % 50 == 0:
                print(f"  step {global_step:>5} | loss {float(loss):.4f}",
                      flush=True)
            if use_wandb:
                wandb.log({"loss": float(loss), "step": global_step})
            if 0 < args.max_steps <= global_step:
                stopped_early = True
                break
        if stopped_early:
            break

    # EVERY rank runs the eval forward. Only rank 0 PRINTS it.
    #
    # This used to `return` early on non-zero ranks, which is fine for a layer
    # that does no communicating and fatal for one that does. Under
    # --expert-parallel the forward below is an ALL-TO-ALL: it needs every rank
    # present. With the others already gone, the remaining rank sat in a
    # collective until NCCL's watchdog aborted it -- about eleven minutes of
    # silence after a training run that had completed perfectly, then a
    # SIGABRT and a non-zero exit. Verified on two independent 2-GPU boxes.
    #
    # The failure is worse than a crash because the training SUCCEEDS first: a
    # learner watches 500 steps of falling loss, then eleven minutes of
    # nothing, then "failed". Guard the PRINTING, never the COLLECTIVE.
    xe, ye, ge = synthetic_groups(2048, cfg, n_groups=args.groups, seed=9999)
    xe, ye = xe.to(device), ye.to(device)
    model_engine.eval()
    with torch.no_grad():
        eval_loss = float(F.mse_loss(
            model_engine(xe.unsqueeze(0)).squeeze(0), ye))

    if is_main:
        print(f"\n{bar}")
        print("  Result")
        print(bar)
        print(f"  eval loss           {eval_loss:.4f}")

        if args.expert_parallel:
            # DeepSpeed's gate keeps its own statistics and does not expose this
            # module's counters. Saying so beats printing a zero and implying the
            # router was perfectly balanced.
            print("  expert utilisation  not reported under --expert-parallel:")
            print("                      DeepSpeed's gate owns the counters, not")
            print("                      this script. Use the default path (or")
            print("                      `uv run moe.py`) to see utilisation.")
        else:
            blk = model_engine.module.blocks[0]
            m = blk.balance_metrics()
            purity = routing_purity(blk, xe, ge.to(device), args.groups)
            mm = "inf" if m["maxmin"] == float("inf") else f"{m['maxmin']:.1f}"
            print(f"  balance strategy    {args.balance}")
            print(f"  entropy             {m['entropy']:.3f}   (1.0 = uniform)")
            print(f"  max/min load        {mm}")
            print(f"  dead experts        {m['dead']} of {args.experts}")
            print(f"  routing purity      {purity:.3f}   (task structure recovered)")

        # A capped run must not be reported as a failure. This is the path the lab
        # manifest offers by name, and a beginner who sees a bad number under a
        # success banner concludes they broke something.
        if stopped_early or args.epochs <= 2:
            cap = (f"--max-steps {args.max_steps}" if stopped_early
                   else f"{args.epochs} epoch(s)")
            print(f"\n  NOTE: this run was capped at {cap}. That is a PIPELINE")
            print(f"  smoke test -- success means DeepSpeed launched, every rank")
            print(f"  ran, and the loss moved. The routing numbers above need a")
            print(f"  full run to mean anything. Drop --max-steps for that.")

        print(f"\n  What to compare: run --balance none against --balance bias.")
        print(f"  Balancing makes the LOSS WORSE and the load EVEN. That trade is")
        print(f"  the entire topic -- see the measured table in moe.py.")
        print(bar)


    # Tear down on EVERY rank, together. Without the barrier a fast rank can
    # exit while a slow one is still inside a collective, which is the same
    # class of bug as the early `return` above -- just at shutdown instead of
    # at eval. destroy_process_group() also stops torch warning that the group
    # was never cleaned up.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if is_main and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
