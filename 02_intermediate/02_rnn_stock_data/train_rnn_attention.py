"""
Attention over RNN hidden states for time-series forecasting — six architectures.

WHAT THIS ADDS TO THE FOLDER
----------------------------
`train_rnn_stock_data_ds.py` ends its forward pass with `out[:, -1, :]`: sixty
days go in, the last hidden state is kept, the other fifty-nine are discarded.
Everything the sequence learned has to survive by being squeezed through one
fixed-width vector.

That is the bottleneck attention was invented to remove. This script keeps the
same data pipeline, the same split-then-scale discipline and the same metrics,
and varies only the architecture — so the comparison is honest.

    --model rnn         the existing baseline: ReLU RNN, out[:, -1, :]
    --model lstm        gated recurrence, still last-state pooling
    --model lstm_attn   LSTM + additive (Bahdanau) attention over all 60 states
    --model lstm_mha    LSTM + multi-head self-attention, then mean-pool
    --model darnn       dual-stage attention (Qin et al. 2017): features, then time
    --model dlinear     NO recurrence at all — the linear baseline that may win

WHY dlinear IS IN THE LIST
--------------------------
Zeng et al. (2022) compared transformer forecasters against a one-layer linear
model across nine datasets. The linear model won, often by a wide margin. Their
argument is uncomfortable and specific: self-attention is permutation-invariant,
so positional encoding is patching back the ordering that a time series *is*.

This folder already refuses to report an RMSE without the persistence baseline
next to it. `--model dlinear` is the same discipline applied one level up: an
architecture you have not compared against a trivial alternative is an
architecture you cannot claim anything about.

WHAT TO EXPECT, HONESTLY
------------------------
The target here is a moving-average deviation, which is smooth by construction,
so persistence is already hard to beat (see §5 of the write-up). Attention is
not likely to change that. What it buys you is **interpretability**: the
attention weights say which of the sixty days the model used, and on this task
you can plot them and look.

Read **Theil U**, not RMSE. U < 1 means the model beats persistence; U >= 1
means a one-line baseline does as well as your distributed network.

MEMORY
------
Trivial — all six models are under ~100k parameters and the dataset is ~2,400
sequences. This is a DeepSpeed *mechanics* example, not a scale one. It runs on
one small GPU, and `--model dlinear` runs on a CPU in seconds.

RUNNING IT
----------
CoreWeave / SLURM:      MODEL=lstm_attn sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 02_intermediate/02_rnn_stock_data \\
                            --dry-run --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install deepspeed yfinance pandas scikit-learn matplotlib

References:
- Bahdanau et al. (2014). https://arxiv.org/abs/1409.0473
- Qin et al. "DA-RNN." IJCAI 2017. https://arxiv.org/abs/1704.02971
- Lim et al. "Temporal Fusion Transformers." IJF 2021. https://arxiv.org/abs/1912.09363
- Zeng et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
  https://arxiv.org/abs/2205.13504
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "rnn":       "ReLU RNN + out[:, -1, :] — the existing baseline",
    "lstm":      "LSTM + out[:, -1, :] — gating, still one state",
    "lstm_attn": "LSTM + additive attention over all 60 states",
    "lstm_mha":  "LSTM + multi-head self-attention, mean-pooled",
    "darnn":     "Dual-stage attention: features, then time (Qin 2017)",
    "dlinear":   "No recurrence. Decomposition + one linear layer (Zeng 2022)",
}


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
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            These models are tiny — CPU is genuinely viable here.")
        print("            ds_config also needs \"torch_adam\": true and fp16")
        print("            disabled, or DeepSpeed will still build CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  These models are TINY (<100k parameters), so unlike most of this")
    print("  course this example really can run on CPU:")
    print("      ALLOW_CPU=1 uv run train_rnn_attention.py --model dlinear")
    print("\n  The attention MECHANISMS need no GPU and no download at all:")
    print("      uv run 02_intermediate/02_rnn_stock_data/attention_layers.py")
    print("      uv run tests/test_attention_layers.py     # 49 checks")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 02_intermediate/02_rnn_stock_data \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def build_model(name: str, input_size: int, hidden_size: int,
                num_layers: int, seq_len: int):
    """
    Construct one of the six architectures.

    Every model takes (B, T, input_size) and returns (B, 1), so they are
    interchangeable in the training loop and the comparison is like-for-like.
    """
    import torch
    import torch.nn as nn

    from attention_layers import (InputAttention, TemporalAttention,
                                  series_decomposition)

    class LastState(nn.Module):
        """RNN or LSTM with `out[:, -1, :]` — the existing behaviour."""

        def __init__(self, cell: str):
            super().__init__()
            Cell = nn.RNN if cell == "rnn" else nn.LSTM
            kwargs = dict(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
            if cell == "rnn":
                kwargs["nonlinearity"] = "relu"
            self.rnn = Cell(**kwargs)
            self.fc = nn.Linear(hidden_size, 1)
            _init_recurrent(self.rnn)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    class LSTMAttention(nn.Module):
        """
        LSTM whose 60 hidden states are pooled by attention, not discarded.

        The only change from `LastState('lstm')` is the pooling. That isolation
        is deliberate: if this beats the LSTM, attention is why.
        """

        def __init__(self, kind: str = "additive"):
            super().__init__()
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
            self.attn = TemporalAttention(hidden_size, kind=kind)
            self.fc = nn.Linear(hidden_size, 1)
            _init_recurrent(self.rnn)
            self.last_weights = None

        def forward(self, x):
            out, _ = self.rnn(x)
            # No causal mask: every one of the 60 days is already in the past
            # relative to the day being predicted. See attention_layers.py.
            context, weights = self.attn(out)
            # Kept for plotting — the interpretable output, and the main reason
            # to prefer this over a transformer on a task this small.
            self.last_weights = weights.detach()
            return self.fc(context)

    class LSTMMultiHead(nn.Module):
        """LSTM + multi-head self-attention over its outputs, then mean-pool."""

        def __init__(self, n_heads: int = 4):
            super().__init__()
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
            # hidden_size must divide by n_heads; fall back to 1 head if not,
            # rather than failing at runtime with a shape error.
            heads = n_heads if hidden_size % n_heads == 0 else 1
            self.mha = nn.MultiheadAttention(hidden_size, heads,
                                             batch_first=True)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)
            _init_recurrent(self.rnn)

        def forward(self, x):
            out, _ = self.rnn(x)
            attended, _ = self.mha(out, out, out)
            # Residual + norm, as in a transformer block: without the residual
            # the LSTM's representation is replaced rather than refined, and on
            # 2,400 samples that throws away the part that already works.
            out = self.norm(out + attended)
            return self.fc(out.mean(dim=1))

    class DARNN(nn.Module):
        """
        Dual-stage attention (Qin et al. 2017): features first, then time.

        Stage 1 is a no-op while `input_size == 1` — softmax over one feature
        is identically 1. It is wired up so that adding volume, realized
        volatility and the individual delta_p is a flag change, not a rewrite.
        """

        def __init__(self):
            super().__init__()
            self.input_attn = InputAttention(input_size, hidden_size)
            self.encoder = nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True)
            self.temporal_attn = TemporalAttention(hidden_size, kind="additive")
            self.fc = nn.Linear(hidden_size, 1)
            _init_recurrent(self.encoder)

        def forward(self, x):
            b, t, _ = x.shape
            h = x.new_zeros(b, hidden_size)
            weighted = []
            for step in range(t):
                x_t, _ = self.input_attn(x[:, step, :], h)
                weighted.append(x_t)
            out, (hn, _) = self.encoder(torch.stack(weighted, dim=1))
            context, _ = self.temporal_attn(out)
            return self.fc(context)

    class DLinear(nn.Module):
        """
        Decomposition + one linear layer per component. No recurrence at all.

        Roughly 120 parameters at seq_len=60. If this matches the attention
        models, that is the result — and it is the one worth reporting.
        """

        def __init__(self, kernel_size: int = 25):
            super().__init__()
            self.kernel_size = kernel_size
            self.seasonal = nn.Linear(seq_len * input_size, 1)
            self.trend = nn.Linear(seq_len * input_size, 1)

        def forward(self, x):
            seasonal, trend = series_decomposition(x, self.kernel_size)
            return (self.seasonal(seasonal.flatten(1))
                    + self.trend(trend.flatten(1)))

    def _init_recurrent(module):
        """
        Orthogonal recurrent init — the same choice as the existing trainer.

        Every singular value equals 1, which bounds the recurrent Jacobian's
        top singular value at 1 and so prevents gradient EXPLOSION over 60
        steps. It does not prevent vanishing: for a ReLU RNN the Jacobian is
        D_t W_hh with D_t a 0/1 mask, and ||D_t W_hh|| <= 1.
        """
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    builders = {
        "rnn": lambda: LastState("rnn"),
        "lstm": lambda: LastState("lstm"),
        "lstm_attn": LSTMAttention,
        "lstm_mha": LSTMMultiHead,
        "darnn": DARNN,
        "dlinear": DLinear,
    }
    return builders[name]()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="lstm_attn", choices=sorted(MODELS))
    parser.add_argument("--list-models", action="store_true",
                        help="Print the architecture table and exit. No GPU.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap optimizer steps; the dry-run path uses this.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--deepspeed",
        default="train_rnn_stock_data_config.json",
        help="DeepSpeed config. The default is the one that actually "
             "ships in this folder -- the previous default named a "
             "ds_config.json that does not exist here, so the "
             "DeepSpeed path silently never ran.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeded so the architecture comparison is "
                             "reproducible. A table of results that cannot be "
                             "regenerated is not evidence.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set by the deepspeed launcher.")
    args = parser.parse_args()

    if args.list_models:
        bar = "=" * 76
        print(bar)
        print("  Architectures — same data, same split, same metrics")
        print(bar)
        for name, note in MODELS.items():
            print(f"  {name:<11} {note}")
        print(bar)
        print("  Read THEIL U, not RMSE. The target is a moving-average")
        print("  deviation and therefore smooth by construction, so")
        print("  persistence is already hard to beat. U >= 1 means a one-line")
        print("  baseline does as well as your network.")
        print()
        print("  The mechanisms run on CPU with no download:")
        print("      uv run attention_layers.py")
        return

    require_gpu()

    # Imported AFTER the preflight so a missing GPU produces our message
    # rather than a CUDA error from inside torch's import chain.
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import yfinance as yf
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    bar = "=" * 76
    print(bar)
    print(f"  {args.model} — {MODELS[args.model]}")
    print(bar)

    # Seed everything BEFORE any weight is created. Comparing architectures on
    # ~2,400 samples without this measures the initialisation lottery as much
    # as the architecture.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data prep is duplicated from train_rnn_stock_data_ds.py rather than
    # imported, deliberately and for two reasons. The repo's convention is that
    # folders are self-contained (see CONTRIBUTING.md §1) — but concretely,
    # that module imports `deepspeed` at file scope, so importing it would drag
    # a CUDA-toolkit dependency into the CPU path this script advertises. A
    # reader running `ALLOW_CPU=1 ... --model dlinear` would get a
    # ModuleNotFoundError instead of a forecast.
    print(f"  downloading {args.ticker} {args.start} -> {args.end}")
    data = yf.download(args.ticker, start=args.start, end=args.end,
                       progress=False, auto_adjust=True)
    if data.empty:
        raise SystemExit(
            f"yfinance returned no rows for {args.ticker}. Check the ticker "
            "and that this machine has network egress — compute nodes on a "
            "cluster usually do not."
        )

    close = data["Close"].squeeze()
    df = pd.DataFrame({"Close": close})
    for period in [14, 26, 50, 100, 200]:
        df[f"delta_{period}"] = close - close.rolling(window=period).mean()
    df["avg_delta"] = df[[f"delta_{p}" for p in [14, 26, 50, 100, 200]]].mean(axis=1)
    df = df.dropna()
    print(f"  {len(df)} usable rows after the 200-day warm-up")

    values = df["avg_delta"].values.reshape(-1, 1)

    # SPLIT FIRST, THEN FIT. Fitting the scaler on the full series leaks the
    # test set's min and max into training — the bug section 5 of the write-up
    # is about, and the one tests/test_stock_leakage.py guards.
    n = len(values)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_s = scaler.fit_transform(values[:train_end])
    test_s = scaler.transform(values[val_end:])

    def windows(series):
        xs, ys = [], []
        for i in range(len(series) - args.seq_len):
            xs.append(series[i:i + args.seq_len])
            ys.append(series[i + args.seq_len])
        return (torch.tensor(np.array(xs), dtype=torch.float32),
                torch.tensor(np.array(ys), dtype=torch.float32))

    x_train, y_train = windows(train_s)
    x_test, y_test = windows(test_s)
    print(f"  train {len(x_train)} windows   test {len(x_test)} windows")

    model = build_model(args.model, 1, args.hidden_size,
                        args.num_layers, args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters {n_params:,}")
    print(bar)

    engine = None
    # Use DeepSpeed only when we were actually LAUNCHED by it.
    #
    # The presence of a config file is not evidence of that, and treating it as
    # evidence breaks the plain `python train_rnn_attention.py` path: DeepSpeed
    # finds no rank environment, falls back to MPI discovery, and dies with
    # `ModuleNotFoundError: No module named 'mpi4py'` -- an error that says
    # nothing about the real problem. The `deepspeed`/`torchrun` launchers both
    # export LOCAL_RANK and WORLD_SIZE, and the deepspeed launcher additionally
    # passes --local_rank, so any of the three is a reliable signal.
    launched_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
        or getattr(args, "local_rank", -1) >= 0
    )
    if launched_distributed and os.path.exists(args.deepspeed) and torch.cuda.is_available():
        import deepspeed
        engine, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model, model_parameters=model.parameters(),
            config=args.deepspeed,
        )
        device = engine.device
        engine_dtype = next(engine.module.parameters()).dtype
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.MSELoss()
    x_train, y_train = x_train.to(device), y_train.to(device)
    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(x_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            if engine is not None:
                # DeepSpeedEngine has no .dtype -- ask the wrapped module,
                # the same way train_rnn_stock_data_ds.py does. With fp16
                # enabled the parameters are half, and feeding fp32 inputs
                # to a half model raises before the first backward.
                loss = criterion(engine(xb.to(engine_dtype)).float(), yb)
                engine.backward(loss); engine.step()
            else:
                loss = criterion(model(xb), yb)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item(); step += 1
            if 0 < args.max_steps <= step:
                break
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>3}  loss {total / max(1, len(x_train) // args.batch_size):.6f}")
        if 0 < args.max_steps <= step:
            print(f"  stopped at --max-steps {args.max_steps}")
            break

    # ---- evaluation, in ORIGINAL units, against persistence ----------------
    net = engine.module if engine is not None else model
    net.eval()
    with torch.no_grad():
        pred = net(x_test.to(device).to(next(net.parameters()).dtype))
    pred = scaler.inverse_transform(pred.float().cpu().numpy().reshape(-1, 1))
    true = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    naive_rmse = float(np.sqrt(mean_squared_error(true[1:], true[:-1])))
    theil_u = rmse / naive_rmse if naive_rmse > 0 else float("inf")
    da = float(np.mean(
        np.sign(pred[1:, 0] - true[:-1, 0]) == np.sign(true[1:, 0] - true[:-1, 0])
    ))

    print()
    print(bar)
    print(f"  {args.model}  ({n_params:,} parameters)")
    print(bar)
    print(f"  Model RMSE            {rmse:.4f}")
    print(f"  Naive RMSE            {naive_rmse:.4f}   (predict tomorrow = today)")
    print(f"  Theil U               {theil_u:.4f}   "
          f"{'BEATS persistence' if theil_u < 1 else 'NO BETTER than persistence'}")
    print(f"  Directional accuracy  {da:.4f}   "
          f"{'better' if da > 0.5 else 'no better'} than a coin flip")
    print(bar)
    print("  Theil U is the number that matters. The target is a")
    print("  moving-average deviation and therefore smooth by construction,")
    print("  so a low RMSE mostly measures that smoothness, not skill.")

    if getattr(net, "last_weights", None) is not None:
        w = net.last_weights.float().mean(0).cpu().numpy()
        print()
        print("  Mean attention over the 60-day window (oldest -> newest):")
        buckets = np.array_split(w, 10)
        for i, b in enumerate(buckets):
            days = f"t-{60 - i * 6}..t-{60 - (i + 1) * 6 + 1}"
            print(f"    {days:>12}  {b.sum():.4f}  {'#' * int(b.sum() * 120)}")
        print("  This is what attention buys on a task this small: you can see")
        print("  which days the model used, and check it against intuition.")


if __name__ == "__main__":
    main()
