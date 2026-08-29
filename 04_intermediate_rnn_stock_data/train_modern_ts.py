"""
Modern time-series architectures for mean-reversion forecasting, across horizons.

THE PROBLEM WITH THE PREVIOUS EXPERIMENT
----------------------------------------
`train_rnn_attention.py` compared six architectures at horizon H=1 and every
one of them lost to persistence, with more parameters doing monotonically
worse. That is a real result and it is a result about the SETUP, not about the
architectures:

    the target delta-bar = P - MA is smooth by construction,
    so at H=1 the best predictor is very nearly "yesterday's value",
    and there is almost nothing left for a model to add.

Persistence is not near-optimal at every horizon, though. Its error grows with
H while a model that has actually learned the mean-reversion dynamics degrades
more slowly -- because reverting toward a moving average is exactly the kind of
structure that shows up over weeks rather than overnight.

So this script sweeps the horizon, and reports Theil U at each one. **The
question is not "which architecture wins" but "at what horizon does anything
beat persistence at all".** That is a question with a falsifiable answer, which
"is our model good?" is not.

THE ARCHITECTURES
-----------------
None of these is an RNN. They are what moved the field between 2019 and 2024.

    persistence   yhat(t+h) = y(t) for all h. Zero parameters. The bar.
    dlinear       decomposition + one linear layer per component (Zeng 2022)
    nbeats        doubly-residual stack, interpretable basis (Oreshkin 2020)
    tcn           dilated causal convolutions (Bai 2018)
    patchtst      patching + transformer encoder (Nie 2023)
    timemixer     multi-scale decomposition + MLP mixing (Wang 2024)

WHAT THE LITERATURE SAYS, AND WHY IT IS NOT A RANKING
-----------------------------------------------------
Zeng et al. (2022) showed a linear model beating every transformer of its day.
PatchTST (2023) answered with patching. TimeMixer (2024) answered that. Then a
2025 position paper surveyed the exchange and concluded there are **no
champions**: the models are close and the rankings move with the hyperparameter
search.

Which is the honest frame for this script. It is not here to crown a winner. It
is here so you can run six genuinely different inductive biases against
persistence on YOUR series, cheaply, and find out whether any of them earns its
parameters.

RUNNING IT
----------
    uv run train_modern_ts.py --list-models          # no GPU
    uv run train_modern_ts.py --model nbeats --horizon 20
    uv run train_modern_ts.py --sweep                # all models x all horizons

CoreWeave / SLURM:      MODEL=nbeats sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 04_intermediate_rnn_stock_data \\
                            --dry-run --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    uv pip install deepspeed yfinance pandas scikit-learn

References:
- Oreshkin et al. "N-BEATS." ICLR 2020. https://arxiv.org/abs/1905.10437
- Bai et al. "Generic Convolutional and Recurrent Networks." 2018.
  https://arxiv.org/abs/1803.01271
- Nie et al. "A Time Series is Worth 64 Words." ICLR 2023.
  https://arxiv.org/abs/2211.14730
- Wang et al. "TimeMixer." ICLR 2024. https://arxiv.org/abs/2405.14616
- Zeng et al. "Are Transformers Effective...?" AAAI 2023.
  https://arxiv.org/abs/2205.13504
- "There are no Champions in Long-Term Time Series Forecasting." 2025.
  https://arxiv.org/abs/2502.14045
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "persistence": "yhat(t+h) = y(t). Zero parameters. The bar to clear.",
    "dlinear":     "Decomposition + one linear layer each (Zeng 2022)",
    "nbeats":      "Doubly-residual stack, interpretable basis (Oreshkin 2020)",
    "tcn":         "Dilated causal convolutions (Bai 2018)",
    "patchtst":    "Patching + transformer encoder (Nie 2023)",
    "timemixer":   "Multi-scale decomposition + MLP mixing (Wang 2024)",
}
HORIZONS = (1, 5, 10, 20)


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
        print("            These models are small — CPU is viable here.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  These models are SMALL (a few hundred to ~100k parameters), so")
    print("  unlike most of this course they genuinely run on CPU:")
    print("      ALLOW_CPU=1 uv run train_modern_ts.py --sweep")
    print("\n  The primitives need no GPU and no download at all:")
    print("      uv run 04_intermediate_rnn_stock_data/modern_ts_layers.py")
    print("      uv run tests/test_modern_ts.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 04_intermediate_rnn_stock_data \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def build_model(name, seq_len, horizon, hidden=128):
    """Every model maps (B, L) -> (B, H), so the comparison is like-for-like."""
    import torch
    import torch.nn as nn

    from modern_ts_layers import (dilated_receptive_field, n_patches,
                                  patchify, seasonality_basis, trend_basis)
    from attention_layers import series_decomposition

    class DLinear(nn.Module):
        def __init__(self, kernel_size=25):
            super().__init__()
            self.kernel_size = kernel_size
            self.seasonal = nn.Linear(seq_len, horizon)
            self.trend = nn.Linear(seq_len, horizon)

        def forward(self, x):
            s, t = series_decomposition(x.unsqueeze(-1), self.kernel_size)
            return self.seasonal(s.squeeze(-1)) + self.trend(t.squeeze(-1))

    class NBeatsBlock(nn.Module):
        """One block: an MLP emits COEFFICIENTS on a fixed basis, not values."""

        def __init__(self, basis_b, basis_f):
            super().__init__()
            self.register_buffer("basis_b", basis_b)
            self.register_buffer("basis_f", basis_f)
            self.mlp = nn.Sequential(
                nn.Linear(seq_len, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.theta_b = nn.Linear(hidden, basis_b.shape[0])
            self.theta_f = nn.Linear(hidden, basis_f.shape[0])

        def forward(self, x):
            h = self.mlp(x)
            return self.theta_b(h) @ self.basis_b, self.theta_f(h) @ self.basis_f

    class NBeats(nn.Module):
        """
        Trend stack then seasonality stack, with the residual flowing between.

        The second stack sees only what the first could not explain, so the
        two specialise without being told to.
        """

        def __init__(self, degree=3, harmonics=4):
            super().__init__()
            self.blocks = nn.ModuleList([
                NBeatsBlock(trend_basis(seq_len, degree),
                            trend_basis(horizon, degree)),
                NBeatsBlock(seasonality_basis(seq_len, harmonics),
                            seasonality_basis(horizon, harmonics)),
            ])

        def forward(self, x):
            residual, forecast = x, 0
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast

    class TCN(nn.Module):
        """
        Dilated causal convolutions. Depth is chosen so the receptive field
        actually covers the window — otherwise early days are invisible and
        nothing warns you.
        """

        def __init__(self, channels=32, kernel_size=3):
            super().__init__()
            layers, n_layers, in_ch = [], 1, 1
            while dilated_receptive_field(kernel_size, n_layers) < seq_len:
                n_layers += 1
            self.n_layers = n_layers
            self.receptive_field = dilated_receptive_field(kernel_size, n_layers)
            for i in range(n_layers):
                d = 2 ** i
                layers += [
                    nn.ConstantPad1d((d * (kernel_size - 1), 0), 0.0),
                    nn.Conv1d(in_ch, channels, kernel_size, dilation=d),
                    nn.ReLU(),
                ]
                in_ch = channels
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(channels, horizon)

        def forward(self, x):
            out = self.net(x.unsqueeze(1))          # (B, C, L)
            return self.head(out[:, :, -1])         # last position only

    class PatchTST(nn.Module):
        """Patching + a small transformer encoder over the patches."""

        def __init__(self, patch_len=16, stride=8, d_model=64, n_heads=4,
                     depth=2):
            super().__init__()
            self.patch_len, self.stride = patch_len, stride
            n_p = n_patches(seq_len, patch_len, stride)
            self.embed = nn.Linear(patch_len, d_model)
            # Learned positional embedding. Attention is permutation-invariant,
            # so without this the model cannot tell which patch came first —
            # the exact criticism Zeng et al. levelled at transformers here.
            self.pos = nn.Parameter(torch.zeros(1, n_p, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 2,
                batch_first=True, dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(n_p * d_model, horizon)

        def forward(self, x):
            p = patchify(x, self.patch_len, self.stride)      # (B, n_p, PL)
            z = self.encoder(self.embed(p) + self.pos)
            return self.head(z.flatten(1))

    class TimeMixer(nn.Module):
        """Mix across resolutions, then project. All MLP, no attention."""

        def __init__(self, scales=(1, 2, 4)):
            super().__init__()
            self.scales = scales
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Linear(seq_len // s, hidden), nn.GELU(),
                              nn.Linear(hidden, hidden))
                for s in scales
            ])
            self.head = nn.Linear(hidden * len(scales), horizon)

        def forward(self, x):
            import torch.nn.functional as F
            feats = []
            for s, branch in zip(self.scales, self.branches):
                v = x if s == 1 else F.avg_pool1d(
                    x.unsqueeze(1), kernel_size=s, stride=s).squeeze(1)
                feats.append(branch(v))
            return self.head(torch.cat(feats, dim=-1))

    return {"dlinear": DLinear, "nbeats": NBeats, "tcn": TCN,
            "patchtst": PatchTST, "timemixer": TimeMixer}[name]()


def load_series(ticker, start, end):
    """Close -> deviations from 5 moving averages -> their mean. Causal."""
    import pandas as pd
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, progress=False,
                       auto_adjust=True)
    if data.empty:
        raise SystemExit(
            f"yfinance returned no rows for {ticker}. Check the ticker and "
            "that this machine has network egress — cluster compute nodes "
            "usually do not."
        )
    close = data["Close"].squeeze()
    periods = [14, 26, 50, 100, 200]
    df = pd.DataFrame({f"d{p}": close - close.rolling(p).mean()
                       for p in periods}).dropna()
    return df.mean(axis=1).values.reshape(-1, 1)


def run_one(model_name, horizon, values, args):
    """Train and evaluate one (model, horizon) pair. Returns a result dict."""
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # SPLIT FIRST, THEN FIT — the §5 rule. Fitting on the full series leaks the
    # test period's min/max into training.
    n = len(values)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(values[:train_end])
    test_s = scaler.transform(values[val_end:])

    def windows(series):
        xs, ys = [], []
        for i in range(len(series) - args.seq_len - horizon + 1):
            xs.append(series[i:i + args.seq_len, 0])
            ys.append(series[i + args.seq_len:i + args.seq_len + horizon, 0])
        return (torch.tensor(np.array(xs), dtype=torch.float32),
                torch.tensor(np.array(ys), dtype=torch.float32))

    x_tr, y_tr = windows(train_s)
    x_te, y_te = windows(test_s)

    # Persistence: repeat the last observed value across the whole horizon.
    # This is the bar, and it costs nothing to compute.
    naive = x_te[:, -1:].repeat(1, horizon)
    inv = lambda a: scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)
    naive_rmse = float(np.sqrt(((inv(naive.numpy()) - inv(y_te.numpy())) ** 2).mean()))

    if model_name == "persistence":
        return dict(model=model_name, horizon=horizon, params=0,
                    rmse=naive_rmse, theil_u=1.0, naive_rmse=naive_rmse)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, args.seq_len, horizon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    crit = nn.MSELoss()
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

    for _ in range(args.epochs):
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = crit(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_te.to(device)).cpu().numpy()
    rmse = float(np.sqrt(((inv(pred) - inv(y_te.numpy())) ** 2).mean()))

    return dict(model=model_name, horizon=horizon, params=n_params,
                rmse=rmse, theil_u=rmse / naive_rmse, naive_rmse=naive_rmse)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nbeats", choices=sorted(MODELS))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--sweep", action="store_true",
                        help="Every model at every horizon. The real experiment.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-09-01")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Unused here; accepted so the dry-run path works.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if args.list_models:
        bar = "=" * 78
        print(bar); print("  Modern time-series architectures"); print(bar)
        for k, v in MODELS.items():
            print(f"  {k:<13} {v}")
        print(bar)
        print("  Sweep horizons with --sweep. The question is not which model")
        print("  wins, it is AT WHAT HORIZON anything beats persistence.")
        return

    require_gpu()
    import numpy as np  # noqa: F401  (used by run_one)

    bar = "=" * 78
    print(bar)
    print(f"  Mean-reversion forecasting — {args.ticker}")
    print(bar)
    values = load_series(args.ticker, args.start, args.end)
    print(f"  {len(values)} usable days after the 200-day warm-up")
    print(f"  lookback {args.seq_len} days, seed {args.seed}, "
          f"{args.epochs} epochs")

    if not args.sweep:
        res = run_one(args.model, args.horizon, values, args)
        print(bar)
        print(f"  {res['model']}  H={res['horizon']}  "
              f"{res['params']:,} params")
        print(f"  RMSE        {res['rmse']:.4f}")
        print(f"  Naive RMSE  {res['naive_rmse']:.4f}")
        print(f"  Theil U2    {res['theil_u']:.4f}  "
              f"{'BEATS persistence' if res['theil_u'] < 1 else 'loses to persistence'}")
        print(bar)
        return

    print(bar)
    print("  Theil U2 by horizon  (< 1.0 beats persistence)")
    print(bar)
    header = "  " + f"{'model':<13}" + "".join(f"{'H=' + str(h):>10}" for h in HORIZONS)
    print(header); print("  " + "-" * (13 + 10 * len(HORIZONS)))

    for name in MODELS:
        cells = []
        for h in HORIZONS:
            try:
                res = run_one(name, h, values, args)
                cells.append(f"{res['theil_u']:>10.4f}")
            except Exception as exc:                    # noqa: BLE001
                cells.append(f"{'ERR':>10}")
                print(f"    [{name} H={h}] {type(exc).__name__}: {exc}")
        print(f"  {name:<13}" + "".join(cells))

    print(bar)
    print("  Read DOWN each column: does anything drop below 1.0 as H grows?")
    print("  Read ACROSS each row: how does this model degrade with horizon?")
    print("  Persistence is 1.0 by definition — it IS the denominator.")


if __name__ == "__main__":
    main()
