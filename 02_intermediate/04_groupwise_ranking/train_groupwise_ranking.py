#!/usr/bin/env python3
"""
Groupwise ranking with DeepSpeed: score documents in context, not alone.

    deepspeed --num_gpus=2 train_groupwise_ranking.py --model gsf
    deepspeed --num_gpus=2 train_groupwise_ranking.py --model all --epochs 40
    uv run train_groupwise_ranking.py --list-models          # no GPU needed

CoreWeave / SLURM:      sbatch run_deepspeed.sh --model all
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 02_intermediate/04_groupwise_ranking \\
                            --dry-run --collect --wait --terminate --yes

What this example is about
--------------------------
The previous folder changed the LOSS and held the scoring function fixed: one
document in, one score out. This folder does the opposite. The loss is held
fixed at ListNet for every model, and what changes is how much of the
candidate list the scorer is allowed to look at:

    pointwise    f(document)                       the baseline, no context
    gsf          f(document, other) averaged       Ai et al. 2019
    setrank      self-attention over the set       Pang et al. 2020

Because only the architecture varies, the comparison is controlled.

Two properties decide whether a groupwise model is right or broken, and
`groupwise.py` measures both rather than asserting them:

    context sensitivity      does a document's score change when its
                             NEIGHBOURS change? Exactly 0 for pointwise.
    permutation equivariance shuffle the candidates and the scores must
                             permute identically. If they do not, the model
                             is reading candidate ORDER -- which at training
                             time is usually the label order. That is a leak
                             that looks like a result.

The first draft of the GSF here failed the second test (error 1.5e-01) because
it grouped documents by rotating the list, which depends on absolute position.
The property test caught it. That is why the tests assert properties.

Why DeepSpeed for a model this small
------------------------------------
Not for memory. The lesson is that groupwise scoring is O(L^2) in the list
length -- GSF enumerates every ordered pair, SetRank attends over the whole
set -- so the memory knob is --list-len, not the parameter count, and a query
CANNOT be split across devices without changing the computation. Data
parallelism here shards queries. That constraint is the point.
"""

import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused optimizer and
    dies with `OSError: CUDA_HOME environment variable is not set` from deep
    inside torch's C++ extension loader -- which tells a newcomer nothing.

    Set ALLOW_CPU=1 to bypass. As in the previous folder that is a REASONABLE
    thing to do here: the models are tiny and the CPU path finishes in a
    couple of minutes. `groupwise.py` needs no GPU at all.
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
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing on CPU.")
        print("            This example is small enough that CPU is viable.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This example is SMALL. You have three good options without a GPU:")
    print("      uv run groupwise.py                  # models + the two property checks")
    print("      uv run ../../tests/test_groupwise_ranking.py")
    print("      ALLOW_CPU=1 uv run train_groupwise_ranking.py --model all")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 02_intermediate/04_groupwise_ranking \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> "argparse.Namespace":
    """
    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N into every worker's argv, and a strict parser exits 2 with
    "unrecognized arguments" before training starts. CONTRIBUTING.md §3.2.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gsf",
                   help="pointwise | gsf | setrank | all")
    p.add_argument("--list-models", action="store_true",
                   help="Describe the architectures and exit. Needs no GPU.")
    p.add_argument("--task", default="duplicate",
                   choices=["duplicate", "redundancy"],
                   help="'duplicate' plants near-identical documents and demotes "
                        "the twin -- a context-free scorer provably cannot solve "
                        "it. 'redundancy' is the softer, more realistic version "
                        "where context helps but is not required.")
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--list-len", type=int, default=12,
                   help="Documents per query. Groupwise scoring is O(L^2) in "
                        "this, so it is THE memory knob here.")
    p.add_argument("--features", type=int, default=16)
    p.add_argument("--duplicates", type=int, default=3,
                   help="Near-duplicate pairs planted per list (--task duplicate). "
                        "Must satisfy 2*duplicates <= list-len.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = no cap). "
                        "The dry-run path.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="QUERIES per step, not documents. Must match "
                        "train_micro_batch_size_per_gpu in the DeepSpeed config.")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ndcg-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


class _PlainEngine:
    """
    A DeepSpeedEngine-shaped wrapper around a plain model and optimizer.

    Exists so the training loop is written ONCE. The alternative -- branching
    on `if launched_distributed` inside the loop -- means the CPU path and the
    GPU path are different code, and the one nobody runs is the one that rots.
    """

    def __init__(self, model, optimizer):
        self.module = model
        self._model = model
        self._optimizer = optimizer

    def __call__(self, *a, **k):
        return self._model(*a, **k)

    def backward(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

    def step(self):
        self._optimizer.step()

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from groupwise import (MODELS, build_model, context_sensitivity,
                           duplicate_ranking_data,
                           permutation_equivariance_error,
                           redundancy_ranking_data)
    from ranking_metrics import listnet_loss, ndcg

    bar = "=" * 78
    if args.list_models:
        print(bar)
        print("  Scoring architectures in this folder")
        print(bar)
        for name, (_, blurb) in MODELS.items():
            print(f"  {name:<12} {blurb}")
        print(bar)
        print("  They share the ListNet loss and differ ONLY in the scorer, so")
        print("  `--model all` is a controlled comparison.")
        print("  Changing the OBJECTIVE instead is the previous folder:")
        print("      ../03_learning_to_rank/")
        print(bar)
        return

    require_gpu()

    import torch

    torch.manual_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    # Use DeepSpeed only when a launcher actually started us. Run under plain
    # `python` and deepspeed.initialize() finds no rank environment, falls back
    # to MPI discovery, and dies with `ModuleNotFoundError: No module named
    # 'mpi4py'` -- an error that says nothing about the real problem and would
    # make the documented ALLOW_CPU=1 path impossible.
    launched_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
        or getattr(args, "local_rank", -1) >= 0
    )
    if launched_distributed:
        import deepspeed

    models = list(MODELS) if args.model == "all" else [m.strip() for m in args.model.split(",")]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Choose from: {list(MODELS)}")

    if args.task == "duplicate" and 2 * args.duplicates > args.list_len:
        raise SystemExit(
            f"--duplicates {args.duplicates} needs a list of at least "
            f"{2 * args.duplicates} documents, but --list-len is {args.list_len}. "
            "The pairs must be disjoint.")

    # ---- data --------------------------------------------------------------
    # Both generators take a fixed task_seed, so train and test share the same
    # hidden utility direction. Drawing a new one per call makes them unrelated
    # tasks and training then makes the metric WORSE -- a bug this course has
    # already shipped once, in the previous folder's generator.
    if args.task == "duplicate":
        x_tr, y_tr = duplicate_ranking_data(
            args.queries, args.list_len, args.features,
            n_duplicates=args.duplicates, seed=args.seed)
        x_te, y_te = duplicate_ranking_data(
            max(256, args.queries // 4), args.list_len, args.features,
            n_duplicates=args.duplicates, seed=args.seed + 9973)
    else:
        x_tr, y_tr = redundancy_ranking_data(
            args.queries, args.list_len, args.features, seed=args.seed)
        x_te, y_te = redundancy_ranking_data(
            max(256, args.queries // 4), args.list_len, args.features,
            seed=args.seed + 9973)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)

    if is_main:
        print(bar)
        print("  Groupwise Ranking")
        print(bar)
        print(f"  task          {args.task}")
        print(f"  train / test  {len(x_tr)} / {len(x_te)} queries")
        print(f"  list length   {args.list_len} documents  ({args.features} features each)")
        print(f"  models        {', '.join(models)}")
        print(f"  loss          listnet (held FIXED across models)")
        print(f"  world size    {world_size}")
        print(bar)

    results = {}
    for name in models:
        torch.manual_seed(args.seed)          # same init policy for every model
        model = build_model(name, args.features)

        if launched_distributed:
            engine, optimizer, _, _ = deepspeed.initialize(
                args=args, model=model, model_parameters=model.parameters(),
                config=args.deepspeed)
        else:
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            engine = _PlainEngine(model, optimizer)

        # Shard QUERIES across ranks, never the documents inside a query. A
        # groupwise scorer whose group spans devices is a DIFFERENT function --
        # every document would be scored against a subset of its real
        # candidates, silently.
        xs, ys = x_tr, y_tr
        if world_size > 1:
            xs, ys = x_tr[rank::world_size], y_tr[rank::world_size]

        steps_per_epoch = max(1, len(xs) // args.batch_size)
        global_step, stop = 0, False
        for epoch in range(args.epochs):
            perm = torch.randperm(len(xs), device=device)
            for i in range(steps_per_epoch):
                idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                scores = engine(xs[idx])
                loss = listnet_loss(scores.float(), ys[idx])
                engine.backward(loss)
                engine.step()
                global_step += 1
                if 0 < args.max_steps <= global_step:
                    stop = True
                    break
            if stop:
                break

        engine.eval()
        with torch.no_grad():
            s = engine(x_te).float()
            score = ndcg(s, y_te, args.ndcg_k).mean().item()
            # Report the two properties alongside the metric. A groupwise model
            # with context sensitivity ~0 has collapsed to pointwise and its
            # NDCG is not evidence for groupwise scoring; one that is not
            # equivariant may simply be reading candidate order.
            ctx = context_sensitivity(engine.module, x_te[:64].float())
            perm_err = permutation_equivariance_error(engine.module,
                                                      x_te[:64].float())
        engine.train()
        results[name] = dict(ndcg=score, context=ctx, perm_err=perm_err,
                             params=sum(p.numel() for p in engine.module.parameters()))
        if is_main:
            print(f"  {name:<12} NDCG@{args.ndcg_k} {score:.4f}   "
                  f"context {ctx:.6f}   perm_err {perm_err:.2e}"
                  + ("   [dry run — meaningless]" if args.max_steps > 0 else ""))

    if is_main:
        # An untrained baseline, because a ranking number without one is
        # unreadable: random ordering on a 12-document list already scores
        # around 0.47 NDCG here, and every model above must beat that to mean
        # anything at all.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            base = build_model("pointwise", args.features).to(device).eval()
            base_ndcg = ndcg(base(x_te), y_te, args.ndcg_k).mean().item()
        print(bar)
        print(f"  untrained baseline   NDCG@{args.ndcg_k} {base_ndcg:.4f}")

        leaky = [n for n, r in results.items() if r["perm_err"] > 1e-4]
        if leaky:
            print(f"\n  WARNING: {', '.join(leaky)} is not permutation-equivariant.")
            print("  Its scores depend on the ORDER candidates arrive in, which")
            print("  during training is usually the label order. Treat its NDCG")
            print("  as unproven, not as a result.")

        if len(results) > 1:
            best = max(results, key=lambda k: results[k]["ndcg"])
            print(f"  best                 {best} ({results[best]['ndcg']:.4f})")
            if "pointwise" in results and best != "pointwise":
                gain = results[best]["ndcg"] - results["pointwise"]["ndcg"]
                print(f"  gain over pointwise  {gain:+.4f}")
            print()
            print("  Read the parameter counts next to the scores. On this data")
            print("  the SMALLEST groupwise model tends to win: SetRank has ~10x")
            print("  the parameters of GSF and overfits a few thousand queries.")
            print("  More context capacity is not free.")
        print(bar)


if __name__ == "__main__":
    main()
