#!/usr/bin/env python3
"""
Refine a noisy backbone, with a guaranteed symmetry or a learned one.

    deepspeed --num_gpus=1 train_structure_ds.py --head ipa
    deepspeed --num_gpus=1 train_structure_ds.py --head diffusion
    deepspeed --num_gpus=1 train_structure_ds.py --head diffusion --augment
    deepspeed --num_gpus=1 train_structure_ds.py --data cath --head ipa

Read `uv run structure.py` first -- it prices the architectural choice on CPU
in a minute, and this script is the same choice under training.

The task
--------
Take a real backbone, perturb it, and ask the model to put it back. That is
the structure module's actual job inside AlphaFold (and literally the job of
AlphaFold3's diffusion head), scored with **FAPE** -- frame-aligned point
error, which compares predicted and true coordinates inside each residue's own
local frame and is therefore blind to the overall orientation.

The three runs worth making
---------------------------
Measured on 1 x RTX 3080 Ti, synthetic data, defaults, equivariance taken
from the TRAINED model:

    arm                           FAPE    vs do-nothing   equivariance
    --head ipa                  1.7972          +18.8%       1.086e-15
    --head diffusion            2.2785           -3.0%       5.075e-02
    --head diffusion --augment  2.2917           -3.6%       7.807e-03
    --head mlp                  2.2662           -2.4%       3.335e-01

Three things to read off that table:

1. **Only IPA learns the task.** The other heads do worse than leaving the
   noisy input alone. The model is given no sequence and no pair features
   here -- `s` and `z` are zeros -- so geometry is the ONLY signal, and
   geometry is precisely what IPA can read and ordinary attention cannot.
   On a full AlphaFold this gap would be smaller, because the trunk supplies
   features the other heads could use.
2. **Augmentation works**: 5.1e-02 -> 7.8e-03, a 6.5x reduction.
3. **And it is still not a guarantee** -- thirteen orders of magnitude above
   IPA, which never saw a rotation in training.

Every run prints its own equivariance error at the end, so the comparison is
something you measure rather than something you read.

Why AF3 made that trade
-----------------------
Generality, not carelessness. IPA needs a residue frame built from N, CA and
C atoms; ligands, ions and nucleic acids have no backbone to build one from.
Dropping the guarantee is what let AlphaFold3 treat everything as atoms.

Hardware
--------
Declared 24 GB, 1 GPU. This is the cheapest lab in the section -- there is no
cubic term here, so `--n-res` costs quadratic memory at worst.
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
      Refining 128-residue backbones with invariant point attention is a GPU
      job, and DeepSpeed's optimizer needs CUDA to build at all.

  What you CAN do right now, on this machine, with no GPU
      uv run structure.py        the comparison this folder exists for: IPA's
                                 SE(3) guarantee against AlphaFold3's learned
                                 symmetry, measured three ways
      uv run cath_data.py        the real backbones, and the chemistry check
                                 that says they are what they claim

      Both run on CPU in about a minute and are where most of the teaching is.
      From the repository root, ./tests/run_all.sh runs every logic test.

  How to get a GPU
      uv run runpod/runpod_ctl.py run 06_protein_folding/04_structure_module \\
          --collect --wait --terminate --yes

      24 GB is enough for the defaults. Confirm the pod is gone afterwards
      with `uv run runpod/runpod_ctl.py pods`.

  To step through this script on CPU anyway
      ALLOW_CPU=1 python train_structure_ds.py --max-steps 2
""")
    print("=" * 78 + "\n")
    sys.exit(1)




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine noisy backbones with IPA or an AF3-style head."
    )
    p.add_argument("--head", choices=("ipa", "diffusion", "mlp"), default="ipa",
                   help="ipa = AF2, equivariant by construction; diffusion = "
                        "AF3-style, must learn the symmetry; mlp = strawman")
    p.add_argument("--data", choices=("synthetic", "cath"), default="synthetic",
                   help="synthetic helix/strand backbones (default, no "
                        "download) or real CATH 4.3 chains (~240 MB)")
    p.add_argument("--augment", action="store_true",
                   help="random SE(3) on every example -- how AlphaFold3 "
                        "recovers the symmetry it gave up")
    p.add_argument("--n-res", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--noise", type=float, default=1.0,
                   help="Angstroms of Gaussian noise added to the backbone "
                        "the model must undo")
    p.add_argument("--train-chains", type=int, default=512)
    p.add_argument("--eval-chains", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total optimizer steps (for a cheap dry run)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1)
    # parse_known_args because the deepspeed launcher injects its own flags.
    args, _ = p.parse_known_args()
    return args


def _synthetic_backbones(n: int, n_res: int, seed: int):
    """
    Backbones built from alpha helices and beta strands.

    **The first version of this generator was a random walk**, and it was the
    `01_basics/02_convnet` bug wearing a lab coat. A random walk has no
    structure beyond its 3.8 A step, so there is nothing for a denoiser to
    denoise *toward* -- the true positions are themselves random. Measured,
    the model beat the do-nothing baseline by **+1.0%**, which is not a weak
    model, it is very nearly the information-theoretic ceiling for that data.
    The same lab on real CATH backbones managed **+33.1%**, which is what
    showed the task was fine and the data was not.

    So the segments here carry real secondary structure, which is exactly the
    local regularity a structure module exploits:

        alpha helix   rise 1.5 A/residue, radius 2.3 A, ~100 deg/residue
        beta strand   extended, ~3.3 A rise, near-linear

    Segments of random length and type are concatenated at random
    orientations. The result is not a folded protein -- `--data cath` is the
    real thing -- but it is locally predictable, which is what makes
    refinement a task rather than a coin flip.
    """
    import math

    import torch

    g = torch.Generator().manual_seed(seed)
    ca_all = torch.zeros(n, n_res, 3)

    for i in range(n):
        pts, pos, k = [], torch.zeros(3), 0
        # A random rotation to orient each segment independently.
        axis = torch.nn.functional.normalize(torch.randn(3, generator=g), dim=0)
        while k < n_res:
            helix = torch.rand(1, generator=g).item() < 0.6
            seg_len = int(torch.randint(5, 16, (1,), generator=g).item())
            seg_len = min(seg_len, n_res - k)

            local = torch.empty(seg_len, 3)
            for j in range(seg_len):
                if helix:
                    ang = math.radians(100.0) * j
                    local[j] = torch.tensor(
                        [2.3 * math.cos(ang), 2.3 * math.sin(ang), 1.5 * j])
                else:
                    # A strand, with the slight pleat real sheets have.
                    local[j] = torch.tensor(
                        [0.8 * ((-1.0) ** j), 0.0, 3.3 * j])

            # Random orientation per segment, then append at the running end.
            a = torch.randn(3, 3, generator=g)
            q, r = torch.linalg.qr(a)
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
            if torch.det(q) < 0:
                q[:, 0] *= -1
            seg = local @ q.T + pos
            pts.append(seg)
            pos = seg[-1] + torch.nn.functional.normalize(
                torch.randn(3, generator=g), dim=0) * 3.8
            k += seg_len
        ca_all[i] = torch.cat(pts, dim=0)[:n_res]

    # N and C must be placed in each residue's LOCAL frame, not at a fixed
    # global offset.
    #
    # The second version of this generator put them at constant offsets like
    # `ca + [-1.2, 0, 0]`. Every frame then pointed the same way regardless of
    # where the chain was going, so the frames carried no information about
    # local geometry and the model had nothing to attend with: measured
    # -0.7%, WORSE than doing nothing. Structure in the CA trace is not
    # enough -- the frames have to track it.
    t_vec = torch.zeros_like(ca_all)
    t_vec[:, :-1] = ca_all[:, 1:] - ca_all[:, :-1]
    t_vec[:, -1] = t_vec[:, -2]
    t_hat = torch.nn.functional.normalize(t_vec, dim=-1, eps=1e-8)

    prev = torch.zeros_like(ca_all)
    prev[:, 1:] = ca_all[:, 1:] - ca_all[:, :-1]
    prev[:, 0] = prev[:, 1]
    b_hat = torch.nn.functional.normalize(
        torch.cross(prev, t_vec, dim=-1), dim=-1, eps=1e-8)
    # Degenerate where the chain is locally straight; any perpendicular does.
    fallback = torch.nn.functional.normalize(
        torch.cross(t_hat, torch.randn(1, 1, 3, generator=g).expand_as(t_hat),
                    dim=-1), dim=-1, eps=1e-8)
    b_hat = torch.where(b_hat.norm(dim=-1, keepdim=True) < 0.1, fallback, b_hat)
    n_hat = torch.cross(b_hat, t_hat, dim=-1)

    jitter = lambda: torch.randn(n, n_res, 3, generator=g) * 0.08
    nx = ca_all - 1.46 * (0.52 * t_hat + 0.85 * n_hat) + jitter()
    cx = ca_all + 1.52 * (0.58 * t_hat - 0.49 * n_hat) + jitter()
    mask = torch.ones(n, n_res)
    return torch.utils.data.TensorDataset(nx, ca_all, cx, mask)


def main() -> None:
    args = parse_args()
    require_gpu()                      # FIRST -- before torch or deepspeed

    # Heavy imports live here, after the preflight.
    import json
    import time

    import numpy as np
    import torch
    import deepspeed

    from structure import (StructureConfig, StructureModule,
                           equivariance_error, fape_loss,
                           frames_from_backbone, random_se3)

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
        # Bind the device before any collective; see 02_evoformer for why.
        torch.cuda.set_device(local_rank)

    def log(msg: str = "") -> None:
        if is_main:
            print(msg, flush=True)

    log("=" * 78)
    log("  BACKBONE REFINEMENT  --  a guaranteed symmetry vs a learned one")
    log("=" * 78)
    log(f"  head           : {args.head}")
    log(f"  augmentation   : {'on' if args.augment else 'off'}")
    log(f"  data           : {args.data}")
    log(f"  noise          : {args.noise} A")
    log("=" * 78)

    # ---------------------------------------------------------------- data
    if args.data == "synthetic":
        train_ds = _synthetic_backbones(args.train_chains, args.n_res, args.seed)
        eval_ds = _synthetic_backbones(args.eval_chains, args.n_res,
                                       args.seed + 10_000)
    else:
        from cath_data import CathBackboneDataset, load_split

        if is_main:
            rows = load_split("train")
        if world_size > 1:
            # No timeout= : torch 2.11 (what this lab locks) rejects it.
            torch.distributed.barrier(device_ids=[local_rank])
        if not is_main:
            rows = load_split("train")
        train_ds = CathBackboneDataset(rows, limit=args.train_chains)
        eval_ds = CathBackboneDataset(load_split("validation"),
                                      limit=args.eval_chains)
        log(f"  CATH filtering : {train_ds.dropped_short} chains not exactly "
            f"128 residues, {train_ds.dropped_unresolved} below 90% resolved")
        args.n_res = train_ds[0][1].shape[0]

    log(f"  residues       : {args.n_res}")
    log(f"  train chains   : {len(train_ds)}")
    log(f"  held-out chains: {len(eval_ds)}")

    # --------------------------------------------------------------- model
    cfg = StructureConfig(n_blocks=args.n_blocks)
    model = StructureModule(cfg, args.head)
    log(f"  parameters     : {sum(p.numel() for p in model.parameters()):,}")

    with open(args.deepspeed_config) as fh:
        ds_config = json.load(fh)
    _check_cuda_toolkit(ds_config, log)

    engine, _, train_loader, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        training_data=train_ds, config=ds_config)
    device = engine.device

    def prepare(nx, ca, cx, augment: bool):
        """Noise the backbone, build frames, optionally reorient everything."""
        nx, ca, cx = nx.to(device), ca.to(device), cx.to(device)
        if augment:
            Rg, tg = random_se3(ca.shape[0], device=device, dtype=ca.dtype)

            def mv(x):
                return torch.einsum("bij,bnj->bni", Rg, x) + tg.unsqueeze(1)

            nx, ca, cx = mv(nx), mv(ca), mv(cx)
        R_true, t_true = frames_from_backbone(nx, ca, cx)
        noise = torch.randn_like(ca) * args.noise
        R_in, t_in = frames_from_backbone(nx + noise, ca + noise, cx + noise)
        return R_in.float(), t_in.float(), R_true.float(), t_true.float()

    def blank(b, n):
        return (torch.zeros(b, n, cfg.c_s, device=device),
                torch.zeros(b, n, n, cfg.c_z, device=device))

    # ------------------------------------------------------------ training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step, capped = 0, False
    t0 = time.time()
    log("\n" + "-" * 78)
    for _ in range(args.epochs):
        for nx, ca, cx, _mask in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                capped = True
                break
            R_in, t_in, R_true, t_true = prepare(nx, ca, cx, args.augment)
            s, z = blank(ca.shape[0], ca.shape[1])
            R_pred, t_pred = engine(s, z, R_in, t_in)
            loss = fape_loss(R_pred, t_pred, R_true, t_true)
            engine.backward(loss)
            engine.step()
            step += 1
            if is_main and step % 50 == 0:
                log(f"  step {step:>5}   FAPE {loss.item():.4f}")
        if capped:
            break
    elapsed = time.time() - t0

    # ---------------------------------------------------------- evaluation
    engine.eval()
    fapes, baselines = [], []
    with torch.no_grad():
        for nx, ca, cx, _m in torch.utils.data.DataLoader(eval_ds, batch_size=4):
            R_in, t_in, R_true, t_true = prepare(nx, ca, cx, False)
            s, z = blank(ca.shape[0], ca.shape[1])
            R_pred, t_pred = engine(s, z, R_in, t_in)
            fapes.append(fape_loss(R_pred, t_pred, R_true, t_true).item())
            # What you get by doing nothing: the noisy input, unrefined. Any
            # claim of success is relative to THIS, not to zero.
            baselines.append(fape_loss(R_in, t_in, R_true, t_true).item())
    fape = float(np.mean(fapes))
    baseline = float(np.mean(baselines))

    # The number this folder is actually about -- measured on the TRAINED
    # module, not a fresh one. Passing the trained model is what makes
    # --augment's effect visible; see the docstring of equivariance_error.
    import copy
    trained = copy.deepcopy(engine.module).float().cpu()
    equi = equivariance_error(args.head, seed=args.seed, n_res=16,
                              model=trained)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0.0)

    log("\n" + "=" * 78)
    log("  RESULTS  (held-out chains)")
    log("=" * 78)
    log(f"  FAPE           : {fape:.4f}")
    log(f"  do-nothing FAPE: {baseline:.4f}   <- the noisy input, unrefined")
    log(f"  improvement    : {100 * (baseline - fape) / baseline:+.1f}%")
    log(f"  steps          : {step}")
    log(f"  wall clock     : {elapsed:.1f}s")
    if torch.cuda.is_available():
        log(f"  peak GPU memory: {peak_gb:.2f} GB")
    log("")
    log(f"  SE(3) equivariance error ({args.head}): {equi:.3e}")
    if args.head == "ipa":
        log("  Guaranteed by construction. This head never saw a rotation in")
        log("  training and does not need to.")
    else:
        log("  NOT guaranteed -- this head has to learn the symmetry.")
        log(f"  Augmentation is {'ON' if args.augment else 'OFF'}. Compare "
            "--head ipa, which measures ~1e-16.")

    if capped:
        log("\n  THIS WAS A CAPPED RUN.")
        log(f"  --max-steps stopped it after {step} optimizer steps, which is")
        log("  a smoke test of the plumbing, not a trained model. Do not read")
        log("  the FAPE above as a result.")
        log("  The equivariance error IS still meaningful: for --head ipa it")
        log("  is a property of the arithmetic, not of training.")
    elif fape < baseline:
        log(f"\n  The model beat the do-nothing baseline by "
            f"{100 * (baseline - fape) / baseline:.1f}%.")
    else:
        log("\n  The model did NOT beat doing nothing. Before tuning: check")
        log("  that --noise is not so large the input carries no structure,")
        log("  and that the run was not capped.")
    log("=" * 78)

    if HAVE_WANDB and os.environ.get("WANDB_API_KEY") and is_main:
        wandb.init(project="deepspeed-course-structure", config=vars(args))
        wandb.log({"fape": fape, "baseline_fape": baseline,
                   "equivariance_error": equi, "peak_gb": peak_gb})
        wandb.finish()

    if world_size > 1:
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



if __name__ == "__main__":
    main()
