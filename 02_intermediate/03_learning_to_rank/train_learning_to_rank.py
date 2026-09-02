#!/usr/bin/env python3
"""
Learning to Rank with DeepSpeed: four objectives, one data pipeline.

    deepspeed --num_gpus=2 train_learning_to_rank.py --method lambdarank
    deepspeed --num_gpus=2 train_learning_to_rank.py --method all --epochs 20
    uv run train_learning_to_rank.py --list-methods          # no GPU needed

CoreWeave / SLURM:      sbatch run_deepspeed.sh --method all
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 02_intermediate/03_learning_to_rank \\
                            --dry-run --collect --wait --terminate --yes

What this example is about
--------------------------
Ranking is not classification, and the difference lives in the LOSS. All four
methods here share the same scoring network -- features in, one score out,
applied to each document independently -- and differ only in what they
compare:

    pointwise    a document against its own label       regression
    ranknet      a document against another             pairwise
    lambdarank   the same pairs, weighted by |ΔNDCG|    pairwise, metric-aware
    listnet      the whole list at once                 listwise

Because everything except the loss is shared, `--method all` is a real
controlled comparison rather than four runs that differ in a dozen ways.

`ranking_losses.py` holds the objectives and the metrics and runs on CPU;
read it first. The honest headline from its own measurements: on synthetic
data these four land within ~0.005 NDCG of each other, while training at all
moves NDCG from ~0.62 to ~0.99. The objective is not the lever people expect
it to be until the data is hard.

Why DeepSpeed for a model this small
------------------------------------
Not for memory -- the network is a few thousand parameters. The interesting
part is that ranking is a LISTWISE problem: the unit of data is a query with
its whole candidate list, so a "batch" is a batch of lists, and data
parallelism shards queries rather than documents. Splitting a query's list
across GPUs would break every objective above, because a pair or a softmax
that spans devices is not the same computation. That constraint is the
lesson, and it is why the sharding here is explicit.
"""

import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused optimizer and
    dies with `OSError: CUDA_HOME environment variable is not set` from deep
    inside torch's C++ extension loader -- which tells a newcomer nothing.

    Set ALLOW_CPU=1 to bypass. Unusually for this course that is a REASONABLE
    thing to do here: the model is tiny, and the CPU path finishes in under a
    minute. `ranking_losses.py` needs no GPU at all.
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
    print("      uv run ranking_losses.py             # objectives + metrics + a demo")
    print("      uv run ../../tests/test_ranking_losses.py")
    print("      ALLOW_CPU=1 uv run train_learning_to_rank.py --method all")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 02_intermediate/03_learning_to_rank \\")
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
    p.add_argument("--method", default="lambdarank",
                   help="pointwise | ranknet | lambdarank | listnet | all")
    p.add_argument("--list-methods", action="store_true",
                   help="Describe the objectives and exit. Needs no GPU.")
    p.add_argument("--source", default="synthetic", choices=["synthetic", "hf"],
                   help="'synthetic' generates graded-relevance data with numpy: "
                        "no download, reproducible, and enough to compare the "
                        "objectives. 'hf' uses a real reranking corpus "
                        "(see --dataset), encoded with a sentence model.")
    p.add_argument("--dataset", default="mteb/scidocs-reranking",
                   help="HuggingFace reranking dataset when --source hf.")
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence encoder used to featurise documents for "
                        "--source hf.")
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--list-len", type=int, default=16,
                   help="Documents per query. Pairwise losses are O(L^2) in "
                        "this, so it is the memory knob that matters.")
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--noise", type=float, default=2.0,
                   help="Label noise for --source synthetic. At 0 every method "
                        "reaches NDCG 1.0 and the comparison says nothing.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = no cap). "
                        "The dry-run path.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="QUERIES per step, not documents. Must match "
                        "train_micro_batch_size_per_gpu in the DeepSpeed config.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ndcg-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


def hf_ranking_data(args, torch):
    """
    Build ranking lists from a real reranking corpus.

    The corpus gives (query, positive[], negative[]); documents are encoded
    with a sentence model and the query embedding is concatenated as context,
    so each document's feature vector says something about the PAIR rather
    than the document alone.

    Labels here are binary, not graded, which matters: NDCG's exponential gain
    has nothing to bite on, and the four objectives get closer together than
    they would on a graded corpus like MSLR. That is a property of the data,
    not of the methods, and it is why the synthetic generator produces grades
    0-4 by default.
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"  loading {args.dataset} and encoding with {args.encoder}")
    ds = load_dataset(args.dataset, split="test")
    encoder = SentenceTransformer(args.encoder)

    feats, labels = [], []
    half = max(1, args.list_len // 2)
    for row in ds:
        pos = list(row["positive"])[:half]
        neg = list(row["negative"])[:args.list_len - len(pos)]
        docs = pos + neg
        if len(docs) < args.list_len:
            continue
        emb = encoder.encode(docs, convert_to_numpy=True,
                             show_progress_bar=False)
        q = encoder.encode([row["query"]], convert_to_numpy=True,
                           show_progress_bar=False)[0]
        # document embedding, elementwise product with the query, and the dot
        # product: the standard cheap interaction features.
        import numpy as np
        inter = np.concatenate(
            [emb, emb * q[None, :], (emb @ q)[:, None]], axis=-1)
        feats.append(inter[: args.list_len])
        labels.append([1.0] * len(pos) + [0.0] * (args.list_len - len(pos)))
        if len(feats) >= args.queries:
            break

    if not feats:
        raise SystemExit(
            f"{args.dataset} produced no lists of length {args.list_len}. "
            "Lower --list-len or pick another reranking dataset.")
    import numpy as np
    x = torch.from_numpy(np.stack(feats).astype("float32"))
    y = torch.tensor(labels, dtype=torch.float32)
    print(f"  built {len(x)} lists, {x.shape[-1]} features per document")
    return x, y


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
    from ranking_losses import (LOSSES, RankingMLP, average_precision, mrr,
                                ndcg, synthetic_ranking_data)

    bar = "=" * 78
    if args.list_methods:
        print(bar)
        print("  Learning-to-rank objectives in this folder")
        print(bar)
        for name, (_, blurb) in LOSSES.items():
            print(f"  {name:<12} {blurb}")
        print(bar)
        print("  They share a scoring network and differ ONLY in the loss, so")
        print("  `--method all` is a controlled comparison.")
        print("  Scoring documents IN CONTEXT is the next folder:")
        print("      ../04_groupwise_ranking/")
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
    # make the documented ALLOW_CPU=1 path impossible. The deepspeed and
    # torchrun launchers both export LOCAL_RANK and WORLD_SIZE, and deepspeed
    # additionally passes --local_rank.
    launched_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
        or getattr(args, "local_rank", -1) >= 0
    )
    if launched_distributed:
        import deepspeed

    methods = list(LOSSES) if args.method == "all" else [m.strip() for m in args.method.split(",")]
    unknown = [m for m in methods if m not in LOSSES]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Choose from: {list(LOSSES)}")

    # ---- data --------------------------------------------------------------
    if args.source == "hf":
        x, y = hf_ranking_data(args, torch)
        n_features = x.shape[-1]
        split = int(len(x) * 0.8)
        x_tr, y_tr, x_te, y_te = x[:split], y[:split], x[split:], y[split:]
    else:
        n_features = args.features
        x_tr, y_tr = synthetic_ranking_data(
            args.queries, args.list_len, n_features, args.noise, seed=args.seed)
        x_te, y_te = synthetic_ranking_data(
            max(256, args.queries // 4), args.list_len, n_features, args.noise,
            seed=args.seed + 9973)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)

    if is_main:
        print(bar)
        print("  Learning to Rank")
        print(bar)
        print(f"  source        {args.source}")
        print(f"  train / test  {len(x_tr)} / {len(x_te)} queries")
        print(f"  list length   {args.list_len} documents  ({n_features} features each)")
        print(f"  methods       {', '.join(methods)}")
        print(f"  world size    {world_size}")
        print(bar)

    results = {}
    for method in methods:
        loss_fn = LOSSES[method][0]
        torch.manual_seed(args.seed)          # same init for every method
        model = RankingMLP(n_features=n_features)

        if launched_distributed:
            engine, optimizer, _, _ = deepspeed.initialize(
                args=args, model=model, model_parameters=model.parameters(),
                config=args.deepspeed)
        else:
            # Plain PyTorch. Same model, same losses, same metrics -- only the
            # engine differs, so a CPU run and a launcher run are comparable.
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            engine = _PlainEngine(model, optimizer)

        # Shard QUERIES across ranks, never the documents inside a query. A
        # pairwise loss or a listwise softmax that spans devices is a different
        # computation; splitting a list would silently change the objective.
        xs, ys = x_tr, y_tr
        if world_size > 1:
            xs, ys = x_tr[rank::world_size], y_tr[rank::world_size]

        steps_per_epoch = max(1, len(xs) // args.batch_size)
        global_step, stop = 0, False
        for epoch in range(args.epochs):
            perm = torch.randperm(len(xs), device=device)
            total = 0.0
            for i in range(steps_per_epoch):
                idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                scores = engine(xs[idx].to(engine.module.net[0].weight.dtype))
                loss = loss_fn(scores.float(), ys[idx])
                engine.backward(loss)
                engine.step()
                total += loss.item()
                global_step += 1
                if 0 < args.max_steps <= global_step:
                    stop = True
                    break
            if stop:
                break

        engine.eval()
        with torch.no_grad():
            s = engine(x_te.to(engine.module.net[0].weight.dtype)).float()
            m = dict(ndcg=ndcg(s, y_te, args.ndcg_k).mean().item(),
                     mrr=mrr(s, y_te).mean().item(),
                     map=average_precision(s, y_te).mean().item())
        engine.train()
        results[method] = m
        if is_main:
            print(f"  {method:<12} NDCG@{args.ndcg_k} {m['ndcg']:.4f}   "
                  f"MRR {m['mrr']:.4f}   MAP {m['map']:.4f}"
                  + ("   [dry run — meaningless]" if args.max_steps > 0 else ""))

    if is_main:
        # An untrained baseline, because a ranking number without one is
        # unreadable: random ordering on a 16-document list already scores
        # around 0.6 NDCG, and every method above must beat that to mean
        # anything at all.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            base = RankingMLP(n_features=n_features).to(device).eval()
            base_ndcg = ndcg(base(x_te), y_te, args.ndcg_k).mean().item()
        print(bar)
        print(f"  untrained baseline   NDCG@{args.ndcg_k} {base_ndcg:.4f}")
        best = max(results, key=lambda k: results[k]["ndcg"])
        spread = (max(r["ndcg"] for r in results.values())
                  - min(r["ndcg"] for r in results.values()))
        print(f"  best                 {best} ({results[best]['ndcg']:.4f})")
        print(f"  spread across methods {spread:.4f}")
        if len(results) > 1 and spread < 0.02:
            print("\n  That spread is small on purpose-built synthetic data, and")
            print("  saying so is the point: the objective matters far less than")
            print("  training at all. Published listwise gains come from real")
            print("  corpora with ties, position bias and long lists.")
        print(bar)


if __name__ == "__main__":
    main()
