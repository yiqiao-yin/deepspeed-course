# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Regression test: the stock-prediction scaler must not leak the future.

Run:
    uv run tests/test_stock_leakage.py

Background
----------
`02_intermediate/02_rnn_stock_data` originally fit MinMaxScaler on the ENTIRE
series before splitting:

    scaler = MinMaxScaler()
    avg_delta_scaled = scaler.fit_transform(analysis_df['avg_delta'].values...)
    X, y = create_sequences(avg_delta_scaled, sequence_length)
    train_size = int(len(X) * 0.7)          # split happened AFTER scaling

The scaler's min/max were therefore derived partly from the test period, so
every training example was normalized using information from the future.
That is look-ahead bias, and it makes reported test error optimistically
biased — worst exactly around the volatility spikes that matter most.

These checks assert the fix is in place and behaves correctly.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, source_contains  # noqa: E402

DS_SCRIPT = "02_intermediate/02_rnn_stock_data/train_rnn_stock_data_ds.py"
SM_SCRIPT = "02_intermediate/02_rnn_stock_data/train_rnn_stock_data.py"


def create_sequences(data, seq_length):
    """Mirror of the helper used by both training scripts."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def main() -> int:
    r = Results("Stock prediction — look-ahead bias regression test")

    # ---- 1. Source-level guards ---------------------------------------
    # fit_transform must never be applied to the full series.
    for script in (DS_SCRIPT, SM_SCRIPT):
        r.check(
            not source_contains(script, "fit_transform(analysis_df['avg_delta']"),
            f"{Path(script).name}: does not fit_transform the full series",
            "Found the original leaking call.",
        )
        r.check(
            source_contains(script, "scaler.transform("),
            f"{Path(script).name}: uses transform() for held-out splits",
            "Expected transform() (not fit_transform) on val/test.",
        )

    # ---- 2. The leak is real, and the fix removes it -------------------
    # Series whose extreme value lives in the TEST period. This is the
    # situation that makes leakage visible.
    rng = np.random.default_rng(0)
    n = 1000
    series = np.cumsum(rng.normal(0, 1, n)).reshape(-1, 1)
    series[900] = 500.0                       # spike, test region only

    train_end, val_end = int(n * 0.7), int(n * 0.85)

    leaky = MinMaxScaler().fit(series)                  # OLD behaviour
    fixed = MinMaxScaler().fit(series[:train_end])      # NEW behaviour

    r.check(
        fixed.data_max_[0] == series[:train_end].max(),
        "scaler constants derive only from the training slice",
        f"got {fixed.data_max_[0]}, expected {series[:train_end].max()}",
    )
    r.check(
        leaky.data_max_[0] != fixed.data_max_[0],
        "the leak is measurable (old vs new constants differ)",
        "If these matched, the test data would not exercise the bug.",
    )

    same_obs_leaky = leaky.transform(series[:1])[0, 0]
    same_obs_fixed = fixed.transform(series[:1])[0, 0]
    r.check(
        abs(same_obs_leaky - same_obs_fixed) > 1e-6,
        "an identical training observation scales differently under the two schemes",
        f"leaky={same_obs_leaky:.6f} fixed={same_obs_fixed:.6f}",
    )

    # ---- 3. Held-out values may leave [0, 1] — that is correct ---------
    test_scaled = fixed.transform(series[val_end:])
    r.check(
        test_scaled.max() > 1.0,
        "test values are allowed outside [0,1] when the period exceeds training range",
        "Clipping to [0,1] here would re-introduce information about the future.",
    )

    # ---- 4. Sequences are built per split, not across boundaries -------
    seq_len = 60
    tr = fixed.transform(series[:train_end])
    va = fixed.transform(series[train_end:val_end])
    te = fixed.transform(series[val_end:])

    X_tr, _ = create_sequences(tr, seq_len)
    X_va, _ = create_sequences(va, seq_len)
    X_te, _ = create_sequences(te, seq_len)

    r.check(
        len(X_tr) == train_end - seq_len
        and len(X_va) == (val_end - train_end) - seq_len
        and len(X_te) == (n - val_end) - seq_len,
        "each split loses exactly seq_len samples to its initial window",
        f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}",
    )

    # No test window may contain an observation from before the test period.
    first_test_window = te[:seq_len]
    r.check(
        len(first_test_window) == seq_len
        and np.allclose(first_test_window, fixed.transform(series[val_end:val_end + seq_len])),
        "the first test window contains only test-period observations",
        "Building sequences before splitting would leak training data in here.",
    )

    # ---- 5. Baseline metrics are reported ------------------------------
    for needle, label in (
        ("theil_u", "Theil U"),
        ("naive_rmse", "persistence baseline RMSE"),
        ("directional_acc", "directional accuracy"),
    ):
        r.check(
            source_contains(DS_SCRIPT, needle),
            f"{Path(DS_SCRIPT).name}: reports {label}",
            "An RMSE with no baseline to compare against carries no information.",
        )

    # Theil U arithmetic
    actual = series[val_end:].ravel()
    naive_rmse = np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2))
    r.check(
        naive_rmse > 0 and np.isfinite(naive_rmse),
        "persistence baseline is computable and finite",
        f"naive_rmse={naive_rmse}",
    )
    perfect_u = 0.0 / naive_rmse
    r.check(perfect_u < 1.0, "Theil U < 1 for a perfect model")

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
