"""
Forecasting mean reversion with a LANGUAGE MODEL over quantized values.

THE IDEA, AND WHY IT IS NOT A STRETCH
-------------------------------------
A language model predicts the next word, and a word is an index into a finite
dictionary. The mean-reversion signal delta-bar = P - MA is continuous, but
bounded in practice -- prices do not deviate from their own moving average
without limit.

So bin it. Slice the range into B levels, replace each value by its bin index,
and the series becomes a sequence of tokens over a vocabulary of size B. Every
tool built for language now applies unchanged: attention, cross-entropy,
sampling, pretraining.

This is what WaveNet did to raw audio in 2016 (256 mu-law levels, categorical
softmax) and what Chronos does to time series in 2024 (scale, quantize, train
a T5 with cross-entropy, sample for probabilistic forecasts).

WHAT IT BUYS THAT REGRESSION DOES NOT
--------------------------------------
1. **A full predictive distribution, for free.** A softmax over B bins IS a
   distribution. §9 of the write-up recommends "predict a distribution, not a
   point" -- this delivers it as a by-product rather than as extra machinery.
   Sample it for intervals; read its entropy for a confidence estimate.

2. **Heavy tails stop dominating.** MSE is quadratic, so a few crash days own
   the gradient. Cross-entropy is bounded per example. The write-up suggests
   Huber loss for this; tokenization is a stronger version of the same move.

3. **Entropy is a signal in itself.** A point forecast cannot tell you when the
   model is unsure. log2(B) bits means "no idea"; a sharp distribution means
   the model thinks it knows something. That is more actionable than one RMSE
   for the whole test period.

THE THING YOU MUST GET RIGHT
-----------------------------
**Scale each window before quantizing.** Measured on this exact data:

    bin edges fitted on raw train values
        test range [-50.46, 32.97] vs train [-21.07, 30.44]
        3.57% of test values CLIP to an end bin
        error floor 2.2003 -- and adding bins does NOT help, because the
        residual is clipping, not resolution (4-bit 2.53 -> 12-bit 2.16, flat)

    per-window scaling first (Chronos's "scaling and quantization")
        0.01% clipping
        error floor 0.1028      <- 21x better

Without the scaling step this idea fails on financial data, and it fails for
the reason §7 of the write-up already gives: the process is non-stationary, so
the test period leaves the training range. Scaling makes the vocabulary
describe *shape relative to local context* instead of absolute level.

RUNNING IT
----------
    uv run train_token_lm.py --floor-only          # the diagnostic, no training
    uv run train_token_lm.py --bits 8 --horizon 20
    uv run train_token_lm.py --no-scale            # watch it degrade

CoreWeave / SLURM:      MODEL=tokenlm sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 04_intermediate_rnn_stock_data \\
                            --dry-run --collect --wait --terminate --yes

References:
- van den Oord et al. "WaveNet." 2016. https://arxiv.org/abs/1609.03499
- Ansari et al. "Chronos: Learning the Language of Time Series." 2024.
  https://arxiv.org/abs/2403.07815
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            This model is small — CPU is viable here.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This model is small (~200k parameters), so unlike most of this")
    print("  course it genuinely runs on CPU:")
    print("      ALLOW_CPU=1 uv run train_token_lm.py --bits 8")
    print("\n  The QUANTIZATION diagnostic needs no GPU and no training — and")
    print("  you should run it first, because it can rule the idea out:")
    print("      uv run train_token_lm.py --floor-only")
    print("      uv run 04_intermediate_rnn_stock_data/tokenize_series.py")
    print("      uv run tests/test_ts_forecasting.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 04_intermediate_rnn_stock_data \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def build_token_lm(vocab_size, seq_len, d_model=64, n_heads=4, depth=2):
    """
    A small decoder-only transformer over value tokens.

    Deliberately the plainest possible language model — embedding, positional
    embedding, causal transformer stack, linear head to vocab. If the idea
    works, it should work here; if it needs 700M parameters to work, that is
    worth knowing too.
    """
    import torch
    import torch.nn as nn

    class TokenLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1,
            )
            self.blocks = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(d_model, vocab_size)
            # CAUSAL mask. Here it is genuinely required, unlike the many-to-one
            # attention in attention_layers.py: this model is trained with a
            # next-token loss at EVERY position, so position t predicting t+1
            # must not be allowed to look at t+1.
            self.register_buffer(
                "mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            )

        def forward(self, tokens):
            h = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
            h = self.blocks(h, mask=self.mask[: tokens.shape[1], : tokens.shape[1]])
            return self.head(h)                    # (B, T, vocab)

    return TokenLM()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=8,
                        help="Vocabulary is 2**bits levels. 8 = 256.")
    parser.add_argument("--scheme", default="uniform",
                        choices=["uniform", "quantile"])
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip per-window scaling. Expect clipping and a "
                             "much worse floor — that is the demonstration.")
    parser.add_argument("--floor-only", action="store_true",
                        help="Report the quantization floor and exit. Run this "
                             "FIRST: it can rule the whole idea out in seconds.")
    parser.add_argument("--soft-sigma", type=float, default=1.0,
                        help="Spread the target over neighbouring bins to "
                             "restore ordinality. 0 = plain one-hot.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deepspeed",
        default="train_rnn_stock_data_config.json",
        help="DeepSpeed config. The default is the one that actually "
             "ships in this folder -- the previous default named a "
             "ds_config.json that does not exist here, so the "
             "DeepSpeed path silently never ran.")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if not args.floor_only:
        require_gpu()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tokenize_series import (Quantizer, distribution_stats,
                                 expected_value, scale_windows, soft_targets)
    from train_modern_ts import load_series

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = 2 ** args.bits
    bar = "=" * 78
    print(bar)
    print(f"  Token language model over quantized delta-bar — {args.ticker}")
    print(bar)
    print(f"  vocabulary  {vocab} levels ({args.bits}-bit, {args.scheme})")
    print(f"  scaling     {'OFF (expect clipping)' if args.no_scale else 'per-window (Chronos-style)'}")

    values = torch.tensor(load_series(args.ticker, args.start, args.end).flatten(),
                          dtype=torch.float32)
    L = args.seq_len
    W = torch.stack([values[i:i + L + 1] for i in range(len(values) - L - 1)])
    n = len(W)
    tr, te = W[: int(n * 0.70)], W[int(n * 0.85):]
    print(f"  windows     {len(tr)} train / {len(te)} test")

    def prep(w):
        """Scale (or not), then quantize. Returns tokens + inverse-transform."""
        if args.no_scale:
            return w, torch.zeros(len(w), 1), torch.ones(len(w), 1)
        return scale_windows(w)

    tr_s, _, _ = prep(tr)
    te_s, te_off, te_scale = prep(te)

    q = Quantizer.fit(tr_s.flatten(), vocab, args.scheme)
    clip = q.clip_rate(te_s.flatten())

    # The floor, in ORIGINAL units — undo the scaling before measuring.
    recon = q.decode(q.encode(te_s)) * te_scale + te_off
    floor = float(torch.sqrt(((recon - te) ** 2).mean()))

    # Persistence on the last step, for reference.
    naive = float(torch.sqrt(((te[:, -1] - te[:, -2]) ** 2).mean()))

    print(bar)
    print(f"  clip rate         {clip * 100:.2f}%   "
          f"{'(values pinned to an end bin)' if clip > 0.005 else ''}")
    print(f"  quantization floor {floor:.4f}   "
          "<- no token model can beat this")
    print(f"  persistence RMSE   {naive:.4f}")
    print(f"  headroom           {naive / floor:.1f}x")
    if floor > naive * 0.5:
        print()
        print("  WARNING: the floor is within 2x of the bar. Resolution, not")
        print("  modelling, is the binding constraint — add bits or enable")
        print("  per-window scaling before training anything.")
    print(bar)

    if args.floor_only:
        print("  --floor-only: stopping before training, as intended.")
        return

    tr_tok, te_tok = q.encode(tr_s), q.encode(te_s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_token_lm(vocab, L).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model  {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    x_tr = tr_tok[:, :-1].long().to(device)
    y_tr = tr_tok[:, 1:].long().to(device)

    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(x_tr[idx])
            if args.soft_sigma > 0:
                # Ordinal-aware target: being one bin off should cost less than
                # being two hundred off. Plain cross-entropy cannot express that.
                tgt = soft_targets(y_tr[idx], vocab, args.soft_sigma)
                loss = -(tgt * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            else:
                loss = F.cross_entropy(logits.reshape(-1, vocab),
                                       y_tr[idx].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); step += 1
            if 0 < args.max_steps <= step:
                break
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>3}  loss {total / max(1, len(x_tr) // args.batch_size):.4f}")
        if 0 < args.max_steps <= step:
            break

    model.eval()
    with torch.no_grad():
        logits = model(te_tok[:, :-1].long().to(device))[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1).cpu()

    # Point forecast = EXPECTATION over bins, which minimises squared error.
    # argmax would give the mode, which differs whenever the distribution is
    # skewed or bimodal.
    pred_scaled = expected_value(probs, q)
    pred = pred_scaled * te_scale.squeeze(-1) + te_off.squeeze(-1)
    true = te[:, -1]
    rmse = float(torch.sqrt(((pred - true) ** 2).mean()))
    mean_, std_, ent = distribution_stats(probs, q)

    print(bar)
    print(f"  RMSE               {rmse:.4f}")
    print(f"  persistence RMSE   {naive:.4f}")
    print(f"  Theil U2           {rmse / naive:.4f}   "
          f"{'BEATS persistence' if rmse < naive else 'loses to persistence'}")
    print(f"  quantization floor {floor:.4f}")
    print()
    print(f"  mean entropy       {ent:.2f} bits of {args.bits}   "
          f"({'confident' if ent < args.bits * 0.6 else 'near-uniform: no idea'})")
    print(f"  mean pred. std     {std_:.4f} (scaled units)")
    print(bar)
    print("  Entropy is what regression cannot give you: it says WHEN the")
    print("  model thinks it knows something, rather than one number for the")
    print("  whole test period.")


if __name__ == "__main__":
    main()
