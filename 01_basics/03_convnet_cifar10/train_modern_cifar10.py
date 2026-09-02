#!/usr/bin/env python3
"""
CIFAR-10 past 90%: modern architectures on the same DeepSpeed plumbing.

    deepspeed --num_gpus=2 train_modern_cifar10.py --model cifarnet
    deepspeed --num_gpus=2 train_modern_cifar10.py --model resnet9 --epochs 30
    deepspeed --num_gpus=2 train_modern_cifar10.py --list-models     # no GPU

CoreWeave / SLURM:      sbatch run_deepspeed.sh --model cifarnet
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 01_basics/03_convnet_cifar10 \\
                            --dry-run --collect --wait --terminate --yes

Why this script exists
----------------------
`cifar10_deepspeed.py` in this folder reaches about 81%. Nothing is wrong with
it -- it is a two-conv-layer network, and 81% is what a two-conv-layer network
gets. But a reader who leaves this folder believing 81% is what CNNs do on
CIFAR-10 has learned something false, because the same dataset has been solved
to 94-96% by small networks for years, and to 99%+ by fine-tuning a pretrained
transformer.

So this trains three architectures that close that gap, on exactly the same
DeepSpeed setup, changing only the model and the recipe:

    resnet9     the recognisable residual net
    cifarnet    the CIFAR-10 speedrun architecture (arXiv:2404.00498)
    wrn_16_8    Wide ResNet 16-8 (arXiv:1605.07146)

What actually buys the accuracy
-------------------------------
Roughly in order of how much each is worth, which is NOT the order people
expect:

  1. Augmentation. flip + translate + cutout is worth several points on its
     own. The baseline uses none.
  2. Schedule. Warmup then cosine decay to zero, rather than a fixed LR.
  3. Label smoothing (0.2). Stops the network chasing confidence it is not
     rewarded for, and pairs with the logit scaling in the model.
  4. Test-time augmentation. Averaging the logits of an image and its mirror
     is worth a few tenths of a point for one extra forward pass.
  5. The architecture itself, last. Going from a good CNN to a better CNN is
     worth about a point; the four items above are worth ten.

Accuracy is reported, never asserted: the number a run reaches depends on
epochs, GPU count and the effective batch size, and printing a target this
script did not measure would be exactly the fabricated-output problem
CONTRIBUTING.md warns about.
"""

import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused optimizer and
    dies with `OSError: CUDA_HOME environment variable is not set` from deep
    inside torch's C++ extension loader -- which tells a newcomer nothing.

    Set ALLOW_CPU=1 to bypass. Training these models on a CPU is not
    realistic (hours per epoch), but the preflight should not be the thing
    that stops you experimenting.
    """
    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. From this folder:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            Expect this to be impractically slow.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  These are real convnets on 50,000 images. They need a GPU.")
    print("\n  No GPU at all? These need none:")
    print("      uv run modern_cifar_models.py     # the architectures, on CPU")
    print("      uv run ../../tests/test_modern_cifar.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 01_basics/03_convnet_cifar10 \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> "argparse.Namespace":
    """
    Command-line options.

    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N into every worker's argv and a strict parser exits 2 with
    "unrecognized arguments" before training starts. CONTRIBUTING.md section 3.2.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="cifarnet",
                   choices=["resnet9", "cifarnet", "wrn_16_8"])
    p.add_argument("--list-models", action="store_true",
                   help="Describe the architectures and exit. Needs no GPU.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = no cap). "
                        "The dry-run path: a handful of steps proves the "
                        "pipeline without training anything.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-GPU micro-batch. Must match "
                        "train_micro_batch_size_per_gpu in the DeepSpeed config.")
    p.add_argument("--lr", type=float, default=0.2,
                   help="Peak learning rate after warmup.")
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--translate", type=int, default=4,
                   help="Random translation in pixels (0 disables).")
    p.add_argument("--cutout", type=int, default=12,
                   help="Cutout square size in pixels (0 disables).")
    p.add_argument("--flip", default="alternating",
                   choices=["alternating", "random", "none"],
                   help="'alternating' is the derandomised variant from "
                        "arXiv:2404.00498 and is the default there.")
    p.add_argument("--tta", default="mirror", choices=["none", "mirror"],
                   help="Test-time augmentation. 'mirror' averages the logits "
                        "of each test image and its horizontal flip.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", default="ds_config_modern.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    # --list-models must work with no GPU and no torch import cost.
    if args.list_models:
        from modern_cifar_models import MODELS
        bar = "=" * 78
        print(bar)
        print("  Modern CIFAR-10 architectures in this folder")
        print(bar)
        for name, blurb in MODELS.items():
            print(f"  {name:<10} {blurb}")
        print(bar)
        print("  The baseline (cifar10_deepspeed.py) reaches ~81%.")
        print("  Published results for these designs are 94-96%; what YOUR run")
        print("  reaches depends on epochs, GPUs and batch size, so this script")
        print("  reports the number it measures rather than a number it hopes for.")
        print(bar)
        return

    require_gpu()

    # Heavy imports AFTER the preflight, so a CPU-only reader gets our message
    # rather than a CUDA traceback from inside deepspeed's import chain.
    import deepspeed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms

    from modern_cifar_models import (alternating_flip, build_model, count_params,
                                     cutout, init_whitening_conv,
                                     pad_and_random_crop, random_flip)

    torch.manual_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    bar = "=" * 78

    if is_main:
        print(bar)
        print(f"  CIFAR-10 — {args.model}")
        print(bar)

    # ---- data ---------------------------------------------------------------
    # Loaded once into memory as tensors and kept on the GPU. CIFAR-10 is 150 MB
    # in fp32; a DataLoader with worker processes would spend more time moving
    # it than the network spends computing on it.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def as_tensors(train: bool):
        ds = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=train, download=is_main,
            transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
        images, labels = next(iter(loader))
        return images, labels

    if world_size > 1:
        # Only rank 0 downloads; everyone else waits, or they race on the same
        # files and one of them reads a half-written archive.
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        if is_main:
            as_tensors(True), as_tensors(False)
        torch.distributed.barrier()

    train_x, train_y = as_tensors(True)
    test_x, test_y = as_tensors(False)

    device = torch.device(f"cuda:{max(args.local_rank, 0)}")
    torch.cuda.set_device(device)
    train_x = ((train_x - mean) / std).to(device)
    train_y = train_y.to(device)
    test_x = ((test_x - mean) / std).to(device)
    test_y = test_y.to(device)

    # ---- model --------------------------------------------------------------
    model = build_model(args.model)
    if args.model == "cifarnet":
        # The frozen first layer is initialised from the DATA, not randomly.
        # A subset is plenty -- the patch covariance of 5,000 images is
        # indistinguishable from that of 50,000.
        init_whitening_conv(model.whiten, train_x[:5000].cpu())
    n_params = count_params(model)

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        config=args.deepspeed)
    device = engine.device

    # Each rank trains on its own shard. Without this the launcher runs the same
    # work N times: DeepSpeed still all-reduces the gradients, they are simply
    # identical, so two GPUs cost twice as much and learn what one would.
    if world_size > 1:
        train_x, train_y = train_x[rank::world_size], train_y[rank::world_size]

    steps_per_epoch = max(1, len(train_x) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))

    if is_main:
        print(f"  parameters     {n_params:,}")
        print(f"  train / test   {len(train_x) * world_size:,} / {len(test_x):,}")
        print(f"  world size     {world_size}  ({len(train_x):,} images per rank)")
        print(f"  batch          {args.batch_size} per GPU")
        print(f"  epochs         {args.epochs}  ({total_steps:,} steps)")
        print(f"  augment        flip={args.flip} translate={args.translate} "
              f"cutout={args.cutout}")
        print(f"  label smooth   {args.label_smoothing}   TTA: {args.tta}")
        print(bar)

    def lr_at(step: int) -> float:
        """Linear warmup, then cosine decay to zero.

        Decaying to ZERO rather than to a floor matters more than the shape:
        the last few epochs at a tiny learning rate are where most of the final
        accuracy is consolidated.
        """
        import math
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    @torch.no_grad()
    def evaluate(tta: str) -> float:
        engine.eval()
        correct = 0
        for i in range(0, len(test_x), 1000):
            xb = test_x[i:i + 1000].to(engine.module.head.weight.dtype)
            logits = engine(xb).float()
            if tta == "mirror":
                # Average the LOGITS of the image and its mirror. Averaging
                # probabilities instead would weight a confidently-wrong view
                # more than it deserves.
                logits = logits + engine(xb.flip(-1)).float()
            correct += (logits.argmax(1) == test_y[i:i + 1000]).sum().item()
        engine.train()
        return correct / len(test_x)

    # ---- train --------------------------------------------------------------
    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_x), device=device)
        running, seen = 0.0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb, yb = train_x[idx], train_y[idx]

            if args.flip == "alternating":
                xb = alternating_flip(xb, epoch)
            elif args.flip == "random":
                xb = random_flip(xb)
            xb = pad_and_random_crop(xb, args.translate)
            xb = cutout(xb, args.cutout)

            for group in optimizer.param_groups:
                group["lr"] = lr_at(global_step)

            logits = engine(xb.to(engine.module.head.weight.dtype))
            loss = F.cross_entropy(logits.float(), yb,
                                   label_smoothing=args.label_smoothing)
            engine.backward(loss)
            engine.step()

            running += loss.item() * len(xb)
            seen += len(xb)
            global_step += 1
            if 0 < args.max_steps <= global_step:
                stop = True
                break

        if is_main:
            acc = evaluate("none")
            print(f"  epoch {epoch + 1:>3}/{args.epochs}  loss {running / max(1, seen):.4f}  "
                  f"lr {lr_at(global_step):.4f}  test {acc:.2%}")
        if stop:
            if is_main:
                print(f"\n  [dry run] stopped at --max-steps {args.max_steps}")
            break

    # ---- report -------------------------------------------------------------
    if is_main:
        plain = evaluate("none")
        print(bar)
        print(f"  FINAL   {args.model}   test accuracy {plain:.2%}")
        if args.tta == "mirror":
            with_tta = evaluate("mirror")
            print(f"          with mirror TTA           {with_tta:.2%}"
                  f"   ({with_tta - plain:+.2%} for one extra forward pass)")
        print(f"          baseline in this folder   ~81%  (cifar10_deepspeed.py)")
        print(bar)
        if args.max_steps > 0:
            print("  This was a capped run — the accuracy above is meaningless.")
            print("  Drop --max-steps for a real number.")
            print(bar)


if __name__ == "__main__":
    main()
