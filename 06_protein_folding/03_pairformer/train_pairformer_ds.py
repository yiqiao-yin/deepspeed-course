#!/usr/bin/env python3
"""
Train an AlphaFold3 Pairformer trunk, and find the wall still there.

    deepspeed --num_gpus=1 train_pairformer_ds.py                    # baseline
    deepspeed --num_gpus=1 train_pairformer_ds.py --ds-evoformer-attn
    deepspeed --num_gpus=1 train_pairformer_ds.py --data nanofold --n-res 128

Read `06_protein_folding/02_evoformer` first. This script is deliberately the
same script with one substitution -- `PairformerStack` for `EvoformerStack` --
because a comparison between architectures is only worth anything when
everything else is held fixed. Same data, same loss, same metric, same
configs, same seeds.

What to expect
--------------
AlphaFold3 deleted the MSA representation from the trunk. `uv run
pairformer.py` prices that deletion: ~24% of per-block activation bytes at 384
residues, falling to ~11% at 1024 as the cubic term takes over.  Real, and
worth having.

And then **the wall is still there**, because the four triangle operations are
unchanged and the triangle attention logits are byte-identical to AF2's. The
saving is a constant; the asymptote is untouched.

So the interesting run is the same one as in 02_evoformer:

    --ds-evoformer-attn

The kernel's name says "Evoformer". It matters just as much to this trunk,
and noticing that is the point of having built both folders.

What "success" looks like
-------------------------
Precision at K over long-range pairs only (|i - j| >= 6), K = the number of
true contacts, on HELD-OUT chains, with the base rate printed beside it.
Identical to 02_evoformer so the two are comparable.

Hardware
--------
Declared 24 GB, 1 GPU. `--n-res` is still the knob that will OOM you, for
exactly the same reason it was before.
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
              "bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.")
        print("            --ds-evoformer-attn cannot work on CPU and will be "
              "ignored.\n")
        return

    print("\n" + "=" * 78)
    print("  NO GPU DETECTED -- this training run needs CUDA")
    print("=" * 78)
    print("""
  Why it stopped
      Training a Pairformer means materialising a triangle attention tensor
      that grows with the CUBE of the protein length. That is a GPU job, and
      DeepSpeed's optimizer needs CUDA to build at all.

  What you CAN do right now, on this machine, with no GPU
      uv run pairformer.py       the AF2-vs-AF3 comparison: what deleting the
                                 MSA representation bought (~24% of per-block
                                 activation bytes) and what it did NOT buy
                                 (the cubic term is byte-identical)
      uv run synthetic_msa.py    the data: coevolution measured directly, and
                                 the counterexample that carries no signal

      Both run on CPU in about a minute and are where most of the teaching is.
      From the repository root, ./tests/run_all.sh runs every logic test.

  How to get a GPU
      uv run runpod/runpod_ctl.py run 06_protein_folding/03_pairformer \\
          --collect --wait --terminate --yes

      24 GB is enough for the defaults. Confirm the pod is gone afterwards
      with `uv run runpod/runpod_ctl.py pods`.

  To step through this script on CPU anyway
      ALLOW_CPU=1 python train_pairformer_ds.py --max-steps 2
""")
    print("=" * 78 + "\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an AlphaFold3 Pairformer trunk for contact prediction."
    )
    p.add_argument("--data", choices=("synthetic", "nanofold"),
                   default="synthetic",
                   help="synthetic coevolving MSAs (default, no download) or "
                        "real MSAs from ChrisHayduk/nanofold-public (~1 GB)")
    p.add_argument("--n-res", type=int, default=48,
                   help="residues per chain. THIS is the cubic knob.")
    p.add_argument("--n-seq", type=int, default=64, help="alignment depth")
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--coupling", type=float, default=1.0,
                   help="synthetic only: 0.0 makes the data unlearnable on "
                        "purpose -- see synthetic_msa.py")
    p.add_argument("--train-chains", type=int, default=256)
    p.add_argument("--eval-chains", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total optimizer steps (for a cheap dry run)")
    p.add_argument("--ds-evoformer-attn", action="store_true",
                   help="use DeepSpeed's DS4Sci_EvoformerAttention kernel")
    p.add_argument("--activation-checkpointing", action="store_true",
                   help="recompute triangle activations in the backward pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1)
    # parse_known_args because the deepspeed launcher injects its own flags.
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    require_gpu()                      # FIRST -- before torch or deepspeed

    # Heavy imports live here, after the preflight. At module scope a CPU-only
    # reader gets a CUDA traceback before the message above ever runs.
    import json
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    import deepspeed

    from pairformer import PairformerConfig, PairformerStack
    from synthetic_msa import SyntheticConfig, SyntheticContactDataset, eval_mask

    try:
        import wandb
        HAVE_WANDB = True
    except ImportError:
        HAVE_WANDB = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0

    if torch.cuda.is_available():
        # Bind the device BEFORE any collective. NCCL implements barrier() as
        # an all-reduce of a one-element tensor and must pick a device; with
        # nothing bound it picks cuda:0 on every rank and hangs. deepspeed's
        # initialize() would do this for us, but the data guard below runs
        # first by design.
        torch.cuda.set_device(local_rank)

    def log(msg: str = "") -> None:
        if is_main:
            print(msg, flush=True)

    log("=" * 78)
    log("  PAIRFORMER CONTACT PREDICTION (AlphaFold3 trunk)")
    log("=" * 78)
    log(f"  data           : {args.data}")
    log(f"  residues       : {args.n_res}   (pair rep is N^2, triangle attn is N^3)")
    log(f"  alignment depth: {args.n_seq}")
    log(f"  blocks         : {args.n_blocks}")
    log(f"  world size     : {world_size}")
    log(f"  DS4Sci kernel  : {'requested' if args.ds_evoformer_attn else 'off'}")
    log("=" * 78)

    # ---------------------------------------------------------------- data
    if args.data == "synthetic":
        cfg = SyntheticConfig(
            n_res=args.n_res, n_seq=args.n_seq,
            n_contacts=max(4, args.n_res // 2), coupling=args.coupling,
        )
        train_ds = SyntheticContactDataset(args.train_chains, cfg, seed=args.seed)
        eval_ds = SyntheticContactDataset(args.eval_chains, cfg,
                                          seed=args.seed + 10_000)
        scored = torch.from_numpy(eval_mask(cfg)).float()
        if args.coupling < 1e-9:
            log("\n  NOTE: --coupling 0.0 generates data with NO signal, on")
            log("        purpose. A model that scores above the base rate on")
            log("        it is reading something it should not be.\n")
    else:
        # Only rank 0 fetches; the others wait. huggingface_hub does take .lock
        # files, so this is belt and braces -- but without it every rank
        # decodes the same shard, which is pure waste.
        from nanofold_data import NanofoldContactDataset, load_shards

        if is_main:
            shards = load_shards(n_shards=2)
        if world_size > 1:
            # No timeout= : torch 2.13 accepts one, torch 2.11 (what this lab
            # locks) raises TypeError. Verify against uv.lock, not the box.
            torch.distributed.barrier(device_ids=[local_rank])
        if not is_main:
            shards = load_shards(n_shards=2)

        full = NanofoldContactDataset(shards, n_res=args.n_res,
                                      n_seq=args.n_seq, min_sep=6)
        n_eval = min(args.eval_chains, max(1, len(full) // 5))
        eval_ds = torch.utils.data.Subset(full, range(n_eval))
        train_ds = torch.utils.data.Subset(full, range(n_eval, len(full)))
        scored = full.scored_mask()

    log(f"  train chains   : {len(train_ds)}")
    log(f"  held-out chains: {len(eval_ds)}")

    # --------------------------------------------------------------- model
    model = PairformerStack(
        PairformerConfig(n_blocks=args.n_blocks), n_tokens=23
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  parameters     : {n_params:,}")

    kernel_state = "off"
    if args.ds_evoformer_attn:
        kernel_state = _enable_ds_kernel(model, log)

    with open(args.deepspeed_config) as fh:
        ds_config = json.load(fh)

    # require_gpu() is not sufficient, and finding that out cost a run.
    #
    # torch.cuda.is_available() is True on any box with a driver, because the
    # PyTorch wheels ship their own CUDA runtime. DeepSpeed's FusedAdam is a
    # different matter: it JIT-COMPILES a CUDA extension, which needs nvcc and
    # CUDA_HOME. Without a toolkit the run dies inside
    # deepspeed.initialize() with
    #
    #     OSError: CUDA_HOME environment variable is not set.
    #
    # raised from torch/utils/cpp_extension.py -- which is exactly the
    # unhelpful, deep-in-the-stack message require_gpu() exists to prevent,
    # arriving after the data has been built and the model constructed.
    _check_cuda_toolkit(ds_config, log)

    if args.activation_checkpointing:
        ds_config.setdefault("activation_checkpointing", {})
        ds_config["activation_checkpointing"]["partition_activations"] = True

    engine, _, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_ds,
        config=ds_config,
    )
    device = engine.device
    scored = scored.to(device)

    # ------------------------------------------------------------ training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step = 0
    capped = False
    t0 = time.time()
    log("\n" + "-" * 78)
    for epoch in range(args.epochs):
        for msa, contacts, mask in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                capped = True
                break
            msa = msa.to(device)
            # Labels and loss stay in fp32 even under bf16 training: the loss
            # is a sum over N^2 pairs and reducing that in bf16 loses the
            # small gradients the long-range contacts live in.
            contacts = contacts.to(device).float()
            mask = mask.to(device).float()

            logits = engine(msa).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, contacts, weight=mask, reduction="sum"
            ) / mask.sum().clamp(min=1.0)

            engine.backward(loss)
            engine.step()
            step += 1

            if is_main and step % 25 == 0:
                log(f"  step {step:>5}   loss {loss.item():.4f}")
        if capped:
            break
    elapsed = time.time() - t0

    # ---------------------------------------------------------- evaluation
    engine.eval()
    precisions, base_rates = [], []
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=2)
    with torch.no_grad():
        for msa, contacts, mask in eval_loader:
            logits = engine(msa.to(device)).float().cpu()
            p, b = _precision_at_k(logits, contacts, mask)
            precisions.extend(p)
            base_rates.extend(b)
    prec = float(np.mean(precisions)) if precisions else 0.0
    base = float(np.mean(base_rates)) if base_rates else 0.0

    peak_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0.0)

    # Report what the kernel ACTUALLY did, not what we asked it to do.
    #
    # TriangleAttention falls back by setting self.ds_kernel = None the first
    # time a call raises -- which is the right behaviour, but it happens after
    # this status string was computed at wiring time. The first version
    # printed "DS4Sci kernel: on (4 modules)" on a box with no CUDA toolkit,
    # where the kernel cannot possibly have run. A verification harness that
    # reports success for a path that fell back does not lose information, it
    # manufactures confidence -- the same failure the RunPod harness shipped
    # four times (see POSTMORTEMS.md).
    if args.ds_evoformer_attn:
        from pairformer import TriangleAttention as _TA
        live = sum(1 for m in model.modules()
                   if isinstance(m, _TA) and m.ds_kernel is not None)
        total = sum(1 for m in model.modules() if isinstance(m, _TA))
        if kernel_state == "unavailable":
            pass                                    # already accurate
        elif live == 0:
            kernel_state = (f"FELL BACK (0/{total} modules) -- the kernel was "
                            "wired but every call raised")
        elif live < total:
            kernel_state = f"partial ({live}/{total} modules still using it)"
        else:
            kernel_state = f"on ({live}/{total} modules, used throughout)"


    log("\n" + "=" * 78)
    log("  RESULTS  (held-out chains, long-range pairs only)")
    log("=" * 78)
    log(f"  precision@K    : {prec:.3f}")
    log(f"  base rate      : {base:.3f}   <- what random guessing gets")
    log(f"  ratio          : {prec / base:.1f}x base rate" if base > 0 else "")
    log(f"  steps          : {step}")
    log(f"  wall clock     : {elapsed:.1f}s")
    if torch.cuda.is_available():
        log(f"  peak GPU memory: {peak_gb:.2f} GB   (DS4Sci kernel: {kernel_state})")

    # The short-run rule. Clawdeck offers a capped run by name, and printing a
    # bad verdict under a success banner is how a beginner concludes they broke
    # something that worked.
    if capped or (args.max_steps is not None and step <= args.max_steps):
        log("\n  THIS WAS A CAPPED RUN.")
        log(f"  --max-steps {args.max_steps} stops after {step} optimizer "
            f"steps, which is a smoke")
        log("  test of the plumbing, not a trained model. Do not read the")
        log("  precision above as a result. A real run is the default:")
        log("      deepspeed --num_gpus=1 train_pairformer_ds.py")
    elif args.coupling < 1e-9:
        log("\n  --coupling 0.0 was requested, so the data carried NO signal.")
        log("  A precision near the base rate is the CORRECT result here and")
        log("  means the model is honest. Anything higher is the bug.")
    elif prec > 3 * base:
        log(f"\n  The model is well clear of the base rate ({prec / base:.1f}x),")
        log("  so it is recovering contacts from coevolution rather than from")
        log("  residue indices.")
    else:
        log("\n  The model is near the base rate. Before tuning: check that")
        log("  --coupling is not 0, that --n-seq is not tiny (depth IS the")
        log("  evidence), and that the run was not capped.")

    log("\n  Next: compare peak memory with and without --ds-evoformer-attn,")
    log("        then try --deepspeed_config ds_config_z3.json and notice")
    log("        that ZeRO-3 does NOT help. That contrast is the topic.")
    log("=" * 78)

    if HAVE_WANDB and os.environ.get("WANDB_API_KEY") and is_main:
        wandb.init(project="deepspeed-course-pairformer", config=vars(args))
        wandb.log({"precision_at_k": prec, "base_rate": base,
                   "peak_gb": peak_gb, "steps": step})
        wandb.finish()

    if world_size > 1:
        # Tear down deliberately: a fast rank exiting while a slow one is still
        # inside a collective is the same bug as guarding a collective.
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


def _check_cuda_toolkit(ds_config: dict, log) -> None:
    """
    Fail fast, and usefully, when there is a GPU but no CUDA toolkit.

    Skipped when the config already asks for torch's own Adam, because that
    path compiles nothing and works fine without nvcc.
    """
    import shutil
    import sys

    wants_torch_adam = (
        ds_config.get("optimizer", {}).get("params", {}).get("torch_adam")
        is True
    )
    if wants_torch_adam or shutil.which("nvcc"):
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:                                       # noqa: BLE001
        CUDA_HOME = None
    if CUDA_HOME:
        return

    log("\n" + "=" * 78)
    log("  GPU FOUND, BUT NO CUDA TOOLKIT -- DeepSpeed cannot build FusedAdam")
    log("=" * 78)
    log("""
  Why this is not caught by the GPU check above
      torch.cuda.is_available() is True: the PyTorch wheels ship their own
      CUDA runtime, so tensors and training work fine. DeepSpeed's FusedAdam
      is different -- it JIT-COMPILES a CUDA extension, which needs `nvcc`
      and CUDA_HOME. Neither is present here.

      Left alone, this run would die inside deepspeed.initialize() with
      `OSError: CUDA_HOME environment variable is not set`, raised from
      torch/utils/cpp_extension.py, after the data was built and the model
      constructed.

  Two ways forward

      1. Use torch's optimizer instead of the fused one. Nothing is compiled,
         and for a ~100k-parameter trunk the speed difference is noise:

             "optimizer": { "type": "AdamW",
                            "params": { ..., "torch_adam": true } }

         Everything this lab teaches works on that path, including the
         memory comparisons.

      2. Install a CUDA toolkit matching your driver, then set CUDA_HOME.
         Required for --ds-evoformer-attn, which compiles CUTLASS and has
         no pure-PyTorch fallback worth the name.
""")
    log("=" * 78 + "\n")
    sys.exit(1)


def _enable_ds_kernel(model, log) -> str:
    """
    Swap triangle attention for DS4Sci_EvoformerAttention where possible.

    Returns a short status string. **Never raises.** The kernel needs
    DeepSpeed >= 0.10.3, CUDA >= 11.3, compute capability >= 7.0, fp16 or
    bf16, and it JIT-compiles CUTLASS on first call -- so there are several
    honest reasons it may be unavailable, and a lab that dies because an
    optional accelerator is missing is a worse lab.
    """
    try:
        from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
    except Exception as exc:                      # noqa: BLE001
        log(f"\n  [kernel] DS4Sci_EvoformerAttention unavailable: {exc}")
        log("  [kernel] falling back to the plain attention path. The run is")
        log("           correct, just heavier. This is not an error.\n")
        return "unavailable"

    from pairformer import TriangleAttention

    n = 0
    for module in model.modules():
        if isinstance(module, TriangleAttention):
            module.ds_kernel = DS4Sci_EvoformerAttention
            n += 1
    log(f"\n  [kernel] DS4Sci_EvoformerAttention wired into {n} "
        f"triangle attention modules.")
    log("  [kernel] First call JIT-compiles CUTLASS -- expect a slow start.\n")
    return f"on ({n} modules)"


def _precision_at_k(logits, contacts, mask):
    """Per-chain precision@K and base rate over scored long-range pairs."""
    import torch

    precs, bases = [], []
    n = logits.shape[-1]
    off_diag = ~torch.eye(n, dtype=torch.bool)
    for b in range(logits.shape[0]):
        tri = torch.triu(mask[b].bool() & off_diag, diagonal=1)
        labels = contacts[b][tri]
        k = int(labels.sum().item())
        if k == 0:
            continue
        order = torch.argsort(logits[b][tri], descending=True)
        precs.append(labels[order][:k].mean().item())
        bases.append(labels.mean().item())
    return precs, bases


if __name__ == "__main__":
    main()
