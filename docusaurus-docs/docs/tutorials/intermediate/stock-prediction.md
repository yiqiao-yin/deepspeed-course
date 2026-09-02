---
sidebar_position: 2
---

# Stock Price Prediction

A recurrent network on real market data — and a case study in the methodological traps that make financial machine learning uniquely easy to get wrong. The modelling is straightforward; **evaluating it honestly is the hard part**, and this page spends most of its length there.

:::danger Not investment advice
This is a pedagogical example for distributed training. Nothing here is a trading strategy, and the evaluation section explains at length why results that look good on this task usually are not. Do not trade on it.
:::

## 1. What Is Actually Being Predicted

An important detail that the framing "stock price prediction" obscures: the model does **not** predict price. It predicts a **mean-reversion signal**.

Write $P(t)$ for the closing price on trading day $t$, and let

$$
\mathcal{P} = \{14,\; 26,\; 50,\; 100,\; 200\}
$$

be the set of moving-average periods. For each $p \in \mathcal{P}$, define the deviation of price from its own moving average:

$$
\delta_p(t) \;=\; P(t) - \mathrm{MA}_p(t),
\qquad
\mathrm{MA}_p(t) \;=\; \frac{1}{p}\sum_{i=0}^{p-1} P(t-i)
$$

Both are **causal**: $\mathrm{MA}_p(t)$ uses days $t-p+1$ through $t$ and nothing later. Averaging across horizons gives the target,

$$
y(t) \;\equiv\; \bar\delta(t) \;=\; \frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}} \delta_p(t)
$$

The alias $y(t)$ is introduced here because the rest of this page uses it for the generic forecasting notation; $y$ and $\bar\delta$ are the same quantity throughout.

The model observes $y(t-59), \dots, y(t)$ — 60 trading days — and predicts $\hat y(t+1)$.

**This choice is defensible and worth understanding.** Raw price is close to a random walk with a strong unit root: the best predictor of tomorrow's price is today's price, and a model trained on levels learns exactly that while appearing accurate. By contrast $\bar\delta$ is a *detrended* quantity, roughly stationary and genuinely mean-reverting — prices do tend to return toward their moving averages. Modelling a stationary transformation instead of a non-stationary level is the right instinct.

:::warning But it also builds in strong autocorrelation
$\mathrm{MA}_{200}$ changes by at most $\left(P(t) - P(t-200)\right)/200$ per day. Consecutive targets $\bar\delta(t)$ and $\bar\delta(t+1)$ are therefore **highly correlated by construction**, independent of any real predictability.

The consequence: a model that simply outputs its most recent input, $\hat{\bar\delta}(t+1) = \bar\delta(t)$, will score extremely well on RMSE. Low error on this task is largely evidence that the target is smooth, not that the model learned market dynamics. §5 makes this concrete.
:::

```mermaid
flowchart TB
    RAW["Daily close prices<br/>AAPL, 2015-01-01 to 2025-09-01"]
    MA["Moving averages<br/>14, 26, 50, 100, 200 days<br/>causal, backward-looking only"]
    DELTA["Deviations<br/>delta_p = Close - MA_p"]
    AVG["Target: avg over p<br/>a stationary mean-reversion signal"]
    SEQ["Sliding windows<br/>60 days in, 1 day out"]
    SPLIT["Chronological split<br/>70 / 15 / 15"]

    RAW --> MA --> DELTA --> AVG --> SEQ --> SPLIT

    classDef deep fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef base fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    classDef steel fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    class RAW base
    class MA,DELTA steel
    class AVG,SEQ base
    class SPLIT bright
```

## 2. Quick Start

```bash
cd 02_intermediate/02_rnn_stock_data

# SLURM (CoreWeave / HPC)
sbatch run_deepspeed.sh

# Direct execution
deepspeed --num_gpus=2 train_rnn_stock_data_ds.py

# Single-machine version, no DeepSpeed
python train_rnn_stock_data.py
```

Data is fetched at runtime via `yfinance`, so the machine needs network access — a common failure on air-gapped HPC compute nodes. If `yfinance` cannot reach Yahoo, download the data on a login node and cache it to disk first.

## 3. Configuration as Implemented

Taken from `train_rnn_stock_data_ds.py`:

| Parameter | Value |
|---|---|
| Ticker | `AAPL` |
| Date range | 2015-01-01 → 2025-09-01 |
| MA periods | 14, 26, 50, 100, 200 |
| Sequence length | **60** trading days |
| Hidden size | **50** |
| Layers | 2 |
| Cell type | `nn.RNN` with `nonlinearity='relu'` |
| Input size | 1 (univariate — $\bar\delta$ only) |
| Epochs | 50 |
| Loss | `nn.MSELoss` |
| Split | 70 / 15 / 15, **chronological** |
| Seed | 42 |

Two things done right that are worth calling out, because they are commonly done wrong.

**The split is chronological, not shuffled.** `X[:train], X[train:train+val], X[train+val:]` with no shuffling. Random-splitting a time series is catastrophic: it places examples from *after* the test period into training, and the model interpolates rather than forecasts. Reported accuracy becomes fiction. Chronological splitting is the only defensible choice, and this code gets it right.

**The moving averages are causal.** `rolling(window=p).mean()` looks strictly backward. A centred rolling mean would embed future prices in each feature — a subtle and devastating leak.

## 4. The Model

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="relu",
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                         device=x.device, dtype=x.dtype)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])       # last timestep only
```

This is a **vanilla RNN, not an LSTM** — and with a ReLU nonlinearity, which is an aggressive choice. Recall from [the RNN gradient analysis](/docs/tutorials/basic/rnn#mathematical-analysis) that stability is governed by $\left(\gamma\,\sigma_{\max}\right)^{T}$. For $\tanh$, $\gamma = 1$ and activations are bounded to $(-1,1)$. **ReLU is unbounded**, so both activations and gradients can grow without limit over 60 timesteps.

Two design choices make it viable:

### Orthogonal recurrent initialization

```python
if 'weight_ih' in name:
    nn.init.xavier_uniform_(param.data)      # input-to-hidden
elif 'weight_hh' in name:
    nn.init.orthogonal_(param.data)          # hidden-to-hidden
```

An orthogonal matrix has **every singular value equal to 1**. That places $\sigma_{\max}(\mathbf{W}_{hh}) = 1$ exactly — the knife-edge where gradients neither vanish nor explode — and, because orthogonal matrices are normal, $\|\mathbf{W}_{hh}^n\| = 1$ for all $n$, so there is no transient-amplification loophole either. It is the single most important line in this model, and it is the standard companion to ReLU recurrence (Le et al., 2015).

This holds only at initialization; training moves $\mathbf{W}_{hh}$ off the orthogonal manifold. Hence:

### Gradient clipping

```json
{ "gradient_clipping": 1.0 }
```

Non-negotiable for a ReLU RNN. Orthogonal init sets a good starting point; clipping bounds the damage thereafter.

**Only the last timestep is used.** `out[:, -1, :]` discards the other 59 hidden states — a many-to-one architecture. The intermediate states still matter, because the last one is a function of all of them, but no loss is applied to them.

## 5. Look-Ahead Bias: Why the Scaler Must Be Fit After the Split

The most important thing on this page.

:::tip Fixed in the repository
This example **used to** contain the bug described below. It has been corrected —
the code now splits first and fits the scaler on the training slice only, and
`tests/test_stock_leakage.py` guards against a regression. The section is kept
because the mistake is subtle, extremely common in financial ML, and worth being
able to recognize in other people's code.

```bash
uv run tests/test_stock_leakage.py
```
:::

The original implementation was:

```python
# BEFORE — leaks the future:
scaler = MinMaxScaler(feature_range=(0, 1))
avg_delta_scaled = scaler.fit_transform(analysis_df['avg_delta'].values.reshape(-1, 1))

X, y = create_sequences(avg_delta_scaled, sequence_length)

train_size = int(len(X) * 0.7)          # <-- split happens AFTER scaling
```

`MinMaxScaler.fit` computes

$$
x_{\text{scaled}} = \frac{x - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})}
$$

over **the entire series, including the test period.** The transform applied to training data therefore depends on the maximum and minimum of data from the future. That is **look-ahead bias** — a form of data leakage.

Why it matters concretely: if AAPL's largest deviation from its moving averages over the whole decade occurs in 2024 (test period), then every training example from 2016 has been normalized using a constant that could not have been known in 2016. The model implicitly learns the scale of future volatility. Reported test error is optimistically biased, and the gap can be large precisely when it matters most — around regime changes and volatility spikes.

**The fix, now in the repository** — fit on training data only, then apply that fitted transform to validation and test:

```python
# Split the raw series FIRST
n = len(avg_delta)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(avg_delta[:train_end].reshape(-1, 1))   # fit here only
val_scaled   = scaler.transform(avg_delta[train_end:val_end].reshape(-1, 1))  # transform only
test_scaled  = scaler.transform(avg_delta[val_end:].reshape(-1, 1))           # transform only

# Then build sequences within each split independently
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_val,   y_val   = create_sequences(val_scaled,   sequence_length)
X_test,  y_test  = create_sequences(test_scaled,  sequence_length)
```

:::note Two further consequences of fixing it properly
**Test values will fall outside $[0,1]$.** If the test period exceeds the training range, scaled values exceed 1. That is correct and expected — it is the honest representation of encountering conditions not seen in training. `MinMaxScaler` is in fact a poor choice here for exactly this reason; `StandardScaler`, or scaling by training-set volatility, degrades more gracefully. Being surprised by out-of-range test values *is the point*.

**Sequences must not straddle a split boundary.** Building sequences before splitting means the first ~60 test windows contain training-period observations. Constructing sequences within each split, as above, avoids it at the cost of losing `sequence_length` samples per boundary.
:::

### The baseline, now reported

:::tip Also fixed
The script previously reported a bare RMSE with nothing to compare it against.
It now also reports the persistence baseline, Theil U, and directional accuracy.
:::

The script reports test RMSE after inverting the scaling:

```python
test_predict_inv = scaler.inverse_transform(test_predict)
test_actual_inv  = scaler.inverse_transform(test_actual)
test_rmse = np.sqrt(mean_squared_error(test_actual_inv, test_predict_inv))
```

Inverting to original units before computing the metric is correct — RMSE in scaled units is uninterpretable.

But **an RMSE with nothing to compare it to carries no information.** For any time-series forecast the mandatory reference is the naive persistence predictor:

$$
\hat y_{\text{naive}}(t+1) \;=\; y(t)
$$

(recall from §1 that $y \equiv \bar\delta$ — this is the mean-reversion signal, not the price)

Given §1's observation that $\bar\delta$ is smooth by construction, persistence will be *hard to beat*. Always report the ratio — the Theil U statistic:

$$
U_2 \;=\; \frac{\mathrm{RMSE}_{\text{model}}}{\mathrm{RMSE}_{\text{naive}}}
$$

The subscript matters: Theil defined two statistics, and "Theil's U" unqualified is ambiguous. $U_1$ is his *inequality coefficient*, a normalised error bounded in $[0,1]$; $U_2$ is the ratio above. This page always means $U_2$.

$U_2 < 1$ means the model adds value; $U_2 \ge 1$ means a one-line baseline does as well as your distributed RNN. **A large fraction of published financial deep-learning results fail this test when it is applied.**

```python
naive_pred = test_actual_inv[:-1]          # predict tomorrow = today
naive_true = test_actual_inv[1:]
rmse_naive = np.sqrt(mean_squared_error(naive_true, naive_pred))
print(f"Model RMSE: {test_rmse:.4f}")
print(f"Naive RMSE: {rmse_naive:.4f}")
print(f"Theil U2:   {test_rmse / rmse_naive:.4f}   (<1 means the model helps)")
```

:::tip For a trading signal, RMSE is the wrong metric anyway
Profit depends on **direction**, not magnitude. A model with excellent RMSE that systematically gets the sign wrong at turning points loses money. Report directional accuracy alongside RMSE:

$$
\mathrm{DA} \;=\; \frac{1}{n}\sum_{t}\mathbb{1}\Big[\operatorname{sign}\big(\hat y(t+1) - y(t)\big) \;=\; \operatorname{sign}\big(y(t+1) - y(t)\big)\Big]
$$

and compare against 50%.

Two caveats, both easy to miss:

**Persistence scores zero, not "undefined."** The persistence forecast gives $\hat y(t+1) - y(t) = 0$ exactly, and $\operatorname{sign}(0) = 0$, which never equals $\pm 1$. So the baseline that is *hardest to beat on RMSE* scores $\mathrm{DA} = 0$ — a direct demonstration that low RMSE and useful signal are different things.

**This is the direction of $\bar\delta$, not of price.** Since $\bar\delta = P - \overline{\mathrm{MA}}$, a rising $\bar\delta$ means price rose *relative to its own moving average* — which is compatible with the price falling, if the average falls faster. Converting a $\bar\delta$ forecast into a price or return forecast requires modelling the moving average's own motion, and this example does not do that.
:::

## 6. DeepSpeed Configuration

From `train_rnn_stock_data_config.json`:

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 1e-5
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 100
    }
  },
  "gradient_clipping": 1.0,
  "fp16": { "enabled": true, "loss_scale": 0, "loss_scale_window": 1000, "hysteresis": 2, "min_loss_scale": 1 },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

The batch invariant resolves as $64 = 32 \times 1 \times 2$ — **this config assumes exactly 2 GPUs**, matching `run_deepspeed.sh`. Change `--num_gpus` and it aborts.

| Setting | Note |
|---|---|
| `loss_scale: 0` | Enables **dynamic** loss scaling — a fixed scale would be `> 0`. Correct for FP16 |
| `hysteresis: 2` | Consecutive overflows tolerated before the scale is reduced |
| `WarmupLR`, 100 steps | Adam's second-moment estimate is unreliable early; see [bias correction](/docs/tutorials/basic/neural-network#52-momentum-and-adam) |
| `weight_decay: 1e-5` | Mild; the model is small |
| ZeRO Stage 2 | Essentially free relative to DDP — though see the caveat below |

:::note Be honest about what DeepSpeed buys here
This model is roughly $2\times 50 \times (50 + 1 + 1) \approx 5{,}200$ parameters in the RNN plus 51 in the head. Model states are $16\Psi \approx 84$ KB. **ZeRO Stage 2 partitions nothing worth partitioning, and CPU offload would be pure overhead.**

The example is configured this way to demonstrate the mechanics on a realistic data pipeline, not because it needs the memory. At this scale, communication overhead means two GPUs are probably *slower* than one. Per [ZeRO Stages](/docs/getting-started/deepspeed-zero-stages), Stage 2 costs nothing extra in bandwidth, so it is a harmless default — but do not infer that it is helping.

Where distributed training genuinely helps a workload like this is **many models rather than a big one**: sweeping tickers, sequence lengths, or seeds in parallel. That is embarrassingly parallel and does not need ZeRO at all.
:::

## 7. Why This Task Is Hard

Beyond the mechanics, financial forecasting has structural properties that defeat the standard ML playbook.

**Low signal-to-noise.** Daily returns are dominated by noise. A model explaining 1–2% of variance can be economically valuable — but $R^2 = 0.02$ looks like failure by the standards of any other domain, so the usual "keep tuning until the metric looks good" loop pushes you directly into overfitting.

**Non-stationarity.** The data-generating process changes. A model fitted on 2015–2021 is fitted partly on a zero-rate regime that no longer exists. This violates the i.i.d. assumption behind essentially all generalization theory, and it is why a single train/test split is weak evidence — **walk-forward validation** (repeatedly retrain on an expanding window, test on the next block) is the appropriate protocol.

**Reflexivity.** If a predictive pattern is discovered and traded, the trading removes it. Unlike image classification, where cats do not adapt, the target here responds to being modelled.

**Multiple-comparisons bias.** Trying 100 architectures and reporting the best gives a result that is, with high probability, noise. Bailey et al. (2014) formalize this as the "deflated Sharpe ratio": a backtest's apparent quality must be discounted by the number of configurations tried. Since the count is rarely reported, most published backtests cannot be assessed.

```mermaid
flowchart TB
    subgraph TRAPS["Why financial ML results usually do not replicate"]
        direction TB
        LEAK["Look-ahead bias<br/>scaler or features fitted<br/>on future data — section 5"]
        SHUF["Shuffled splits<br/>test data leaks into training"]
        BASE["No naive baseline<br/>RMSE reported with no reference"]
        OVER["Multiple comparisons<br/>best of N configs reported as if it were the only one"]
        NONS["Non-stationarity<br/>one split cannot detect regime change"]
    end

    FIX["Chronological splits, fit transforms on train only,<br/>always report Theil U vs persistence,<br/>walk-forward validation, disclose the search"]

    LEAK --> FIX
    SHUF --> FIX
    BASE --> FIX
    OVER --> FIX
    NONS --> FIX

    classDef deep fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef dark fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    class LEAK,SHUF,BASE,OVER,NONS dark
    class FIX bright
    class TRAPS deep
```

## 8. Attention After the RNN

The model in §4 ends with:

```python
out, _ = self.rnn(x)
out = self.fc(out[:, -1, :])      # 59 hidden states discarded
```

Sixty days go in, the last hidden state is kept, the other fifty-nine are thrown away. Everything the sequence learned has to survive by being squeezed through one fixed-width vector.

**That is exactly the bottleneck attention was invented to remove.** Bahdanau et al. (2014) raised the identical complaint about translation — one context vector is not enough — and the fix transfers directly. Instead of taking the last state, take a *learned weighted average* of all of them:

$$
c \;=\; \sum_{t=1}^{T}\alpha_t\, h_t,
\qquad
\alpha \;=\; \operatorname{softmax}(e),
\qquad
\sum_{t}\alpha_t = 1
$$

where the scores $e_t$ come from either

$$
e_t \;=\; v^\top \tanh\!\big(W_q q + W_k h_t\big)
\quad\text{(additive, Bahdanau)}
\qquad\text{or}\qquad
e_t \;=\; \frac{q^\top h_t}{\sqrt{d}}
\quad\text{(scaled dot-product, Luong)}
$$

with $q = h_T$ as the query: *given where I ended up, which earlier states should I revisit?*

:::tip Attention **contains** the current model
Put all the mass on $t = T$ and $c = h_T$ exactly — you recover `out[:, -1, :]`. So this is a strict generalisation, and it cannot be worse in representational terms. What you pay is parameters, and on ~2,400 training samples that turns out to be the binding constraint. `tests/test_attention_layers.py` asserts the equivalence numerically.
:::

### The $1/\sqrt{d}$ is not cosmetic

For $q, h$ with unit-variance components, $q^\top h$ has variance $d$. The softmax of scores that large is effectively one-hot, its gradient is ~0, and attention stops learning before it starts. Measured over 400 random draws with 16 keys (uniform would be 0.0625):

| $d$ | max weight, unscaled | max weight, scaled |
|---|---|---|
| 8 | 0.5605 | 0.2402 |
| 32 | 0.7872 | 0.2472 |
| 128 | 0.8965 | 0.2539 |
| 512 | 0.9412 | 0.2450 |

Reproduce with `uv run 02_intermediate/02_rnn_stock_data/attention_layers.py`.

### Causal masking: needed less often than you would think

:::warning Not needed here — and applying it anyway loses information
The model reads a window of 60 **past** days and emits one number for day 61. Every element is already historical, so window-position 10 may legitimately attend to window-position 50. Masking would discard real information for no reason.

It becomes **required** the moment you add per-timestep losses, autoregressive multi-step decoding, or an encoder–decoder. Then it is the in-model analogue of the scaler leak in §5: the future contaminating the past, producing excellent metrics and a worthless model.
:::

### Three architectures, and a fourth that is not an RNN

[`train_rnn_attention.py`](https://github.com/yiqiao-yin/deepspeed-course/blob/main/02_intermediate/02_rnn_stock_data/train_rnn_attention.py) implements six models behind one `--model` flag, sharing the data pipeline, the split-then-scale discipline and the metrics — so only the architecture varies.

| `--model` | What it is |
|---|---|
| `rnn` | the §4 baseline: ReLU RNN, `out[:, -1, :]` |
| `lstm` | gated recurrence, still last-state pooling |
| `lstm_attn` | **LSTM + additive attention** over all 60 states |
| `lstm_mha` | LSTM + multi-head self-attention, residual, mean-pooled |
| `darnn` | **DA-RNN** (Qin et al., 2017): attention over features, *then* over time |
| `dlinear` | **no recurrence at all** — decomposition + one linear layer |

**DA-RNN** is the one purpose-built for this problem. Its stage 1 attends across *input series* at each timestep, stage 2 across time. Stage 1 is a no-op here because `input_size=1` — softmax over one feature is identically 1 — and it becomes useful the moment you add the volume, realized volatility and individual $\delta_p$ that §1 already computes and discards.

**Temporal Fusion Transformer** (Lim et al., 2021) is the production-grade version of this idea: LSTM encoder for local structure, interpretable multi-head attention for long-range, gating to prune. It is not implemented here because at 2,400 samples it would be badly over-parameterised, and the measurement below explains why that matters.

### Measured: attention does not help on this task

Seed 42, 40 epochs, CPU, no DeepSpeed. Persistence RMSE is **3.9843**.

| Model | Params | RMSE | **Theil $U_2$** | Directional acc. |
|---|---|---|---|---|
| `dlinear` | **122** | 4.1248 | 1.0352 | 0.4936 |
| `rnn` | 7,801 | 4.3364 | 1.0884 | 0.4776 |
| `lstm` | 31,051 | 4.1195 | **1.0339** | 0.5128 |
| `lstm_attn` | 37,515 | 5.2640 | 1.3212 | 0.4840 |
| `darnn` | 40,843 | 6.7146 | 1.6853 | 0.5192 |
| `lstm_mha` | 41,351 | 9.0232 | 2.2647 | 0.5385 |

Two things to read off this, and neither is flattering:

**Every model loses to persistence.** $U_2 > 1$ across the board. The one-line baseline `predict tomorrow = today` beats all six, which is what §1 predicted: the target is a moving-average deviation and therefore smooth by construction, so persistence is very close to optimal.

**More capacity is monotonically worse.** Order the table by parameter count and $U_2$ rises almost perfectly with it — 122 params at 1.035, 41k params at 2.265. On ~2,400 training sequences the attention models have ample capacity to memorise and none of the data needed to generalise. A 122-parameter linear model is competitive with an LSTM and beats every attention variant.

:::danger This is the result, and reporting it is the point
Attention is a real improvement to the *architecture* — it removes a genuine bottleneck, and on translation or long-horizon multivariate forecasting it earns its place. It does not help **here**, and the reasons are specific: a smooth target, a tiny dataset, and a baseline that is already near-optimal.

Had this page reported "we added attention and RMSE improved" without the persistence column, every number in it would have been true and the conclusion would have been wrong. That is the same failure mode §5 is about, one level up.
:::

### What attention *does* buy on this task

Interpretability. The weights $\alpha_t$ say which of the sixty days the model used, and unlike most settings you can look at them:

```bash
uv run 02_intermediate/02_rnn_stock_data/train_rnn_attention.py --model lstm_attn
# ... prints the mean attention mass over the 60-day window, oldest to newest
```

If the mass concentrates on the last few days, the model has rediscovered persistence — which, given the table above, is the correct thing for it to do.

### If you want attention to actually help here

In rough order of expected value:

1. **Add features.** `input_size=1` is the binding constraint. Volume, realized volatility and the individual $\delta_p$ are already computed and discarded, and DA-RNN's stage-1 attention only becomes meaningful once there is more than one series to choose between.
2. **Add tickers.** ~2,400 sequences from one stock cannot support 40k parameters. Cross-sectional training across hundreds of names regularises far more effectively than any architectural choice.
3. **Lengthen the horizon.** Attention pays off when long-range dependencies exist. Predicting $t+1$ from a smooth series has almost none; predicting $t+20$ has more.
4. **Then** revisit the architecture.

### CPU-runnable

The mechanisms need no GPU and no download:

```bash
uv run 02_intermediate/02_rnn_stock_data/attention_layers.py   # the demos above
uv run tests/test_attention_layers.py                       # 49 checks
```

The tests assert properties rather than shapes — weights form a distribution, masking precedes the softmax (zeroing after leaves the denominator contaminated), one-hot attention reproduces `out[:, -1, :]` exactly, and the decomposition reconstructs its input.

## 9. Improving the Model

Ordered by expected value, not by novelty.

| Change | Why |
|---|---|
| ~~Fix the scaler leak (§5)~~ | **Done** — split-then-fit, with a regression test |
| ~~Add the persistence baseline and Theil U~~ | **Done** — reported alongside RMSE |
| **Walk-forward validation** | One chronological split cannot distinguish skill from a favourable regime |
| Multivariate input | `input_size=1` uses only $\bar\delta$. Volume, realized volatility, and the individual $\delta_p$ are already computed and discarded |
| Swap `nn.RNN` → `nn.LSTM`/`nn.GRU` | Gating handles 60-step dependencies far better; see [LSTM](/docs/tutorials/basic/rnn#long-short-term-memory-lstm) |
| Predict a distribution, not a point | Quantile regression or a Gaussian head gives calibrated intervals — far more useful than a point estimate under this much noise. See [Bayesian NNs](/docs/tutorials/intermediate/bayesian-nn) |
| Multiple tickers | ~2,400 usable samples from one stock is very little data. Cross-sectional training regularizes strongly |
| Huber loss instead of MSE | Financial data is heavy-tailed; MSE's quadratic outlier sensitivity lets a few crash days dominate. See [loss choice](/docs/tutorials/basic/neural-network#32-robust-regression-alternatives) |

## 10. Troubleshooting

**`yfinance` fails on a compute node.** Compute nodes are usually air-gapped. Download on a login node, cache to disk (`analysis_df.to_parquet(...)`), and load from disk in the job.

**Loss goes `NaN`.** ReLU RNN over 60 steps. Confirm `gradient_clipping: 1.0` is active and that orthogonal init is applied to `weight_hh`; try `nonlinearity='tanh'`, or switch to `nn.LSTM`. Alternatively use BF16, which removes the FP16 overflow path.

**Batch-size assertion at startup.** The config is fixed at 2 GPUs ($64 = 32\times1\times2$).

**Test RMSE much worse than validation.** Expected under non-stationarity — the test block is the most recent and most distributionally distant period. Evidence for walk-forward validation, not necessarily a bug.

**Suspiciously excellent results.** Check §5 first. Near-perfect fit on financial data is almost always leakage.

## Next Steps

- [Basic RNN](/docs/tutorials/basic/rnn) — the recurrence, gradient dynamics, and LSTM gating behind this model
- [Bayesian Neural Networks](/docs/tutorials/intermediate/bayesian-nn) — predictive uncertainty, which this task badly needs
- [DeepSpeed ZeRO Stages](/docs/getting-started/deepspeed-zero-stages) — when partitioning actually pays

## References

**Time series and forecasting**

1. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. [Free online](https://otexts.com/fpp3/) — baselines, evaluation, and why persistence is the reference.
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
3. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3). — ML methods repeatedly losing to simple statistical baselines in the M4 competition.
4. Theil, H. (1966). *Applied Economic Forecasting*. North-Holland. — the U statistic.

**Financial machine learning methodology**

5. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — the standard reference on leakage, purged cross-validation, and backtest overfitting.
6. Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS*, 61(5), 458–471.
7. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383–417.
8. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4). — the general treatment of the §5 bug.

**Recurrent models**

9. Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize Recurrent Networks of Rectified Linear Units. [arXiv:1504.00941](https://arxiv.org/abs/1504.00941) — ReLU RNNs and orthogonal/identity initialization.
10. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear networks. *ICLR 2014*. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120) — why orthogonal initialization works.
11. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. *ICML 2013*. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
12. Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191. — probabilistic RNN forecasting done properly.

**Attention for time series (§8)**

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Luong, M.-T., Pham, H., & Manning, C. (2015). Effective Approaches to Attention-based Neural Machine Translation. [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Qin, Y., et al. (2017). A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. *IJCAI 2017*. [arXiv:1704.02971](https://arxiv.org/abs/1704.02971)
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
