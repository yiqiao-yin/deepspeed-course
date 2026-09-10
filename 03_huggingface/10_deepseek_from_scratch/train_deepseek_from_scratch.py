#!/usr/bin/env python3
"""
MLA vs GQA vs MHA: train all three and measure what the cache costs.

    deepspeed --num_gpus=1 train_deepseek_from_scratch.py --variant all
    deepspeed --num_gpus=1 train_deepseek_from_scratch.py --variant mla --epochs 20
    uv run mla.py                       # the architecture alone, no GPU
    uv run train_deepseek_from_scratch.py --list-variants   # no GPU

CoreWeave / SLURM:      sbatch run_deepspeed.sh --variant all
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 03_huggingface/10_deepseek_from_scratch \\
                            --dry-run --collect --wait --terminate --yes

What this example is about
--------------------------
`mla.py` shows that Multi-head Latent Attention caches far less per token. That
is arithmetic, and arithmetic alone cannot tell you what the compression COSTS.
This script trains the same tiny language model three times, changing only the
attention module, and reports quality beside cache size.

The task is **induction**: each sequence contains a pattern that recurs later,
and predicting it requires attending to the earlier occurrence. That is
deliberate. A bag-of-words task would be solved without using attention at all,
and every variant would tie — which would look like a result and be nothing of
the kind.

Why this is in a DeepSpeed course
---------------------------------
The KV cache is an inference cost, not a training one, so ZeRO does not touch
it. That is exactly the point worth seeing: ZeRO shards what the model *is*,
and MLA shrinks what the model must *remember at serving time*. Two different
memory problems, and a reader who has only met ZeRO tends to assume it covers
both.

The models here are a few million parameters, so ZeRO stage 0 is the honest
setting. See `ds_config.json`, which says so.

Reference: DeepSeek-AI, *DeepSeek-V2* (arXiv:2405.04434), §2.1. This is an
independent implementation from the paper; no third-party code is vendored.
"""

import argparse
import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused optimizer and
    dies with `OSError: CUDA_HOME environment variable is not set` from deep
    inside torch's C++ extension loader -- which tells a newcomer nothing.

    ALLOW_CPU=1 bypasses it, and here that is a REASONABLE thing to do: the
    models are small and a short run finishes on a laptop. `mla.py` needs no
    GPU at all and is where the architecture actually lives.
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
    print("\n  This example is SMALL. Three good options without a GPU:")
    print("      uv run mla.py                # the architecture + the cache table")
    print("      uv run ../../tests/test_mla.py")
    print("      ALLOW_CPU=1 uv run train_deepseek_from_scratch.py --variant all")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 03_huggingface/10_deepseek_from_scratch \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N into every worker's argv, and a strict parser exits 2 with
    "unrecognized arguments" before training starts. CONTRIBUTING.md section 3.2.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="all",
                   help="mha | gqa | mla | all")
    p.add_argument("--list-variants", action="store_true",
                   help="Describe the three and exit. Needs no GPU.")
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length. The induction pattern is planted "
                        "early and queried late, so this is also the recall "
                        "distance being tested.")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2, help="GQA only.")
    p.add_argument("--kv-lora-rank", type=int, default=32, help="MLA only.")
    p.add_argument("--train-seqs", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = no cap). "
                        "The dry-run path.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


MARK = 1          # reserved token that announces a pattern; never random


def induction_data(n_seqs: int, seq_len: int, vocab: int, seed: int):
    """
    Sequences where one position REQUIRES looking back to an earlier one.

    Layout, with random filler everywhere else:

        ... [MARK] a b c d ......... [MARK] a b c d
                   ^ first sighting           ^ predictable from the first

    After the second MARK, every pattern token is recoverable by finding the
    earlier MARK and copying what followed. That is an induction head, and it
    is the one thing a change to ATTENTION should affect. A task solvable from
    token frequencies alone would let all three variants tie, and the tie would
    mean nothing.

    Returns (x, y, mask), and BOTH the mask and the marker are load-bearing:

    * **The mask.** Most positions are uniform random and unpredictable by
      construction, so a loss averaged over all of them sits at chance however
      good the model is. The first version of this file did that and reported
      4.147 against a 4.159 floor for all three variants -- a number that looks
      like a result and is not one. Loss is computed on masked positions only,
      for training AND evaluation; without masked training the induction
      gradient is drowned by ~95% noise and the model learns almost nothing
      (measured: 4.8% accuracy after 20 epochs, versus 27% with it).
    * **The marker.** Without it the model must match a bare random token,
      which collides by chance at this vocabulary size and makes the first
      pattern position ambiguous. Measured: 4.8% accuracy without MARK, 26%
      with, under identical training.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randint(2, vocab, (n_seqs, seq_len), generator=g)
    mask = torch.zeros(n_seqs, seq_len, dtype=torch.bool)

    pat_len = 4
    for i in range(n_seqs):
        pat = torch.randint(2, vocab, (pat_len,), generator=g)
        early = torch.randint(1, seq_len // 3, (1,), generator=g).item()
        late = torch.randint(2 * seq_len // 3, seq_len - pat_len - 2, (1,),
                             generator=g).item()
        x[i, early] = MARK
        x[i, early + 1:early + 1 + pat_len] = pat
        x[i, late] = MARK
        x[i, late + 1:late + 1 + pat_len] = pat
        mask[i, late:late + pat_len] = True

    # Next-token prediction: labels are inputs shifted by one, mask with them.
    return (x[:, :-1].contiguous(), x[:, 1:].contiguous(),
            mask[:, 1:].contiguous())


class TinyLM:
    """Namespace holder; the module is built in main() once torch is imported."""


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from mla import VARIANTS, AttnConfig, build

    bar = "=" * 78
    if args.list_variants:
        print(bar)
        print("  Attention variants in this folder")
        print(bar)
        for name, (_, blurb) in VARIANTS.items():
            print(f"  {name:<6} {blurb}")
        print(bar)
        print("  They share every other dimension, so `--variant all` is a")
        print("  controlled comparison. The architecture itself is in mla.py,")
        print("  which runs on CPU in seconds.")
        print(bar)
        return

    require_gpu()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    # Use DeepSpeed only when a launcher actually started us. Run under plain
    # `python` and deepspeed.initialize() finds no rank environment, falls back
    # to MPI discovery, and dies with `ModuleNotFoundError: No module named
    # 'mpi4py'` -- an error that says nothing about the real problem and would
    # make the documented ALLOW_CPU=1 path impossible.
    launched = (os.environ.get("LOCAL_RANK") is not None
                or os.environ.get("WORLD_SIZE") is not None
                or getattr(args, "local_rank", -1) >= 0)
    if launched:
        import deepspeed

    variants = list(VARIANTS) if args.variant == "all" else \
        [v.strip() for v in args.variant.split(",")]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Choose from {list(VARIANTS)}")

    cfg = AttnConfig(hidden_size=args.hidden, n_heads=args.heads,
                     head_dim=args.hidden // args.heads,
                     n_kv_heads=args.kv_heads,
                     kv_lora_rank=args.kv_lora_rank,
                     q_lora_rank=max(32, args.hidden // 2),
                     qk_rope_head_dim=max(8, (args.hidden // args.heads) // 4),
                     max_seq_len=args.seq_len)

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.attn = build(name, cfg)
            self.n1 = nn.RMSNorm(cfg.hidden_size)
            self.n2 = nn.RMSNorm(cfg.hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size), nn.GELU(),
                nn.Linear(4 * cfg.hidden_size, cfg.hidden_size))

        def forward(self, h):
            h = h + self.attn(self.n1(h))
            return h + self.mlp(self.n2(h))

    class LM(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.emb = nn.Embedding(args.vocab, cfg.hidden_size)
            self.blocks = nn.ModuleList([Block(name) for _ in range(args.layers)])
            self.norm = nn.RMSNorm(cfg.hidden_size)
            self.head = nn.Linear(cfg.hidden_size, args.vocab, bias=False)

        def forward(self, idx):
            h = self.emb(idx)
            for b in self.blocks:
                h = b(h)
            return self.head(self.norm(h))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr, m_tr = induction_data(args.train_seqs, args.seq_len, args.vocab,
                                      args.seed)
    x_te, y_te, m_te = induction_data(max(256, args.train_seqs // 8),
                                      args.seq_len, args.vocab, args.seed + 9973)
    x_tr, y_tr, m_tr = x_tr.to(device), y_tr.to(device), m_tr.to(device)
    x_te, y_te, m_te = x_te.to(device), y_te.to(device), m_te.to(device)

    if is_main:
        print(bar)
        print("  MLA vs GQA vs MHA — quality beside cache size")
        print(bar)
        print(f"  task          induction recall over {args.seq_len} tokens")
        print(f"  train / test  {len(x_tr)} / {len(x_te)} sequences")
        print(f"  model         {args.layers} layers, hidden {args.hidden}, "
              f"{args.heads} heads")
        print(f"  variants      {', '.join(variants)}")
        print(f"  world size    {world_size}")
        print(bar)

    results = {}
    for name in variants:
        torch.manual_seed(args.seed)          # same init for every variant
        model = LM(name)
        cache = model.blocks[0].attn.cache_per_token()

        if launched:
            engine, optimizer, _, _ = deepspeed.initialize(
                args=args, model=model, model_parameters=model.parameters(),
                config=args.deepspeed)
            step = lambda loss: (engine.backward(loss), engine.step())
            fwd = engine
        else:
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            def step(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            fwd = model

        xs, ys, ms = x_tr, y_tr, m_tr
        if world_size > 1:
            xs, ys, ms = (x_tr[rank::world_size], y_tr[rank::world_size],
                          m_tr[rank::world_size])

        spe = max(1, len(xs) // args.batch_size)
        gstep, stop = 0, False
        for _ in range(args.epochs):
            perm = torch.randperm(len(xs), device=device)
            for i in range(spe):
                idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                logits = fwd(xs[idx])
                # Masked: only the induction positions carry signal. Training
                # on all of them drowns the gradient in ~95% noise.
                sel = ms[idx].reshape(-1)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, args.vocab).float()[sel],
                    ys[idx].reshape(-1)[sel])
                step(loss)
                gstep += 1
                if 0 < args.max_steps <= gstep:
                    stop = True
                    break
            if stop:
                break

        (fwd if not launched else engine).eval()
        with torch.no_grad():
            logits = fwd(x_te)
            # Scored ONLY on the induction positions. Averaging over the whole
            # sequence would include the uniformly random majority and pin the
            # number at chance regardless of the model.
            sel = m_te.reshape(-1)
            lg, tgt = logits.reshape(-1, args.vocab).float()[sel], y_te.reshape(-1)[sel]
            val = nn.functional.cross_entropy(lg, tgt).item()
            acc = (lg.argmax(-1) == tgt).float().mean().item() * 100
        (fwd if not launched else engine).train()

        params = sum(p.numel() for p in model.parameters())
        results[name] = dict(val=val, acc=acc, cache=cache, params=params)
        if is_main:
            print(f"  {name:<5} loss {val:6.4f}  acc {acc:5.1f}%  "
                  f"cache/token {cache:>6,}  params {params:>9,}"
                  + ("   [dry run — meaningless]" if args.max_steps > 0 else ""))

    if is_main and len(results) > 1:
        import math
        print(bar)
        base = results.get("mha")
        for name, r in results.items():
            if base:
                print(f"  {name:<5} cache {base['cache'] / r['cache']:>5.1f}x smaller "
                      f"than MHA   acc {r['acc'] - base['acc']:+5.1f} points")
        print()
        print(f"  Chance is {math.log(args.vocab):.3f} nats / "
              f"{100 / args.vocab:.1f}% on a {args.vocab}-token vocabulary.")
        print("  A variant near that has learned nothing, and its cache")
        print("  advantage would be meaningless -- read the accuracy first.")
        print()
        print("  The cache column is EXACT — it is arithmetic, not a measurement.")
        print("  The loss column is one small model on one synthetic task; read")
        print("  it as 'does compression cost accuracy here', not as a ranking.")
        print(bar)


if __name__ == "__main__":
    main()
