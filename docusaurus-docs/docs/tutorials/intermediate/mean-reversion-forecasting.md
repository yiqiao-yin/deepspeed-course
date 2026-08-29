---
sidebar_position: 3
---

# Forecasting Mean Reversion: Beyond RNNs

[Stock Price Prediction](./stock-prediction.md) establishes what is actually being modelled — not price, but the **mean-reversion signal**

$$
y(t) \;\equiv\; \bar\delta(t) \;=\; \frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}}\big(P(t) - \mathrm{MA}_p(t)\big)
$$

and §8 of that page measures six RNN and attention architectures against it. **Every one loses to persistence, and more parameters is monotonically worse.**

This page takes that result seriously and asks the follow-up: if attention over an RNN is not the answer, what is? Two directions, both measured.

**Example folder:** [`04_intermediate_rnn_stock_data/`](https://github.com/yiqiao-yin/deepspeed-course/tree/main/04_intermediate_rnn_stock_data) — `modern_ts_layers.py`, `tokenize_series.py`, `train_modern_ts.py`, `train_token_lm.py`

## 1. Why the Previous Result Was About the Setup, Not the Models

At horizon $H=1$, the target is a moving-average deviation and therefore smooth by construction. Persistence — $\hat y(t+1) = y(t)$ — is very nearly optimal, and there is almost nothing left for a model to add.

That is a fact about the **problem**, not about deep learning. Two things follow:

1. **Lengthen the horizon.** Persistence degrades as $H$ grows; a model that has learned the reversion dynamics should degrade more slowly, because reverting toward a moving average is structure that plays out over weeks rather than overnight.
2. **Stop borrowing architectures from language.** The four primitives below were built for series.

:::tip The question worth asking
Not *"which architecture wins?"* — that has no stable answer. But *"at what horizon does anything beat persistence at all?"* That is falsifiable.
:::

## 2. Four Primitives That Are Not RNNs

`modern_ts_layers.py` implements each as its actual mechanism. All CPU-runnable:

```bash
uv run 04_intermediate_rnn_stock_data/modern_ts_layers.py
```

### N-BEATS — the constraint is the point

*Oreshkin et al., ICLR 2020.* Blocks emit **coefficients on a fixed basis**, not free-form values:

$$
\hat y = \theta^\top B, \qquad B_{i,t} = (t/H)^i \ \text{(trend)}
\quad\text{or}\quad
B = [\cos 2\pi k t,\; \sin 2\pi k t]_{k=1}^{K} \ \text{(seasonality)}
$$

so $\theta$ is directly readable — level, slope, curvature. A low-degree polynomial *cannot* represent a wiggle, so a trend block is forced to model trend and leave the rest to the residual.

The residual flow is what makes the blocks specialise:

$$
r_0 = x, \qquad r_n = r_{n-1} - \mathrm{backcast}_n, \qquad \hat y = \sum_n \mathrm{forecast}_n
$$

Block 2 never sees the raw series — only what block 1 failed to explain.

:::note The bases are complementary, and the test checks it
Every seasonality row integrates to ~0 over the horizon, so it cannot express a level or trend. The trend level row sums to $H$, so it can. If either could do the other's job the decomposition would be meaningless.
:::

### PatchTST — the idea that made transformers work on series

*Nie et al., ICLR 2023 — "A Time Series is Worth 64 Words".* Point-wise attention treats each timestep as a token, which is wrong in a stateable way: a single timestep carries almost no semantic content, whereas a word does. Attention between two individual days is mostly attention between two noise samples.

Patching fixes three things at once. Measured, at a 60-day lookback:

| patch_len | stride | tokens | attention cost vs point-wise |
|---|---|---|---|
| 1 | 1 | 60 | 1.000× |
| 8 | 8 | 7 | 0.014× |
| 16 | 8 | 6 | 0.010× |
| 16 | 16 | 3 | 0.003× |

Local semantics survive, cost falls quadratically, and the saved budget buys a longer history.

### TCN — check the receptive field before you train

*Bai, Kolter & Koltun, 2018.* Dilated causal convolutions reach back

$$
R = 1 + (k-1)\frac{b^{L}-1}{b-1}
$$

which grows **exponentially** in depth:

| layers | k=2 | k=3 | k=5 |
|---|---|---|---|
| 2 | 4 | 7 | 13 |
| 4 | 16 | 31 | 61 |
| 6 | 64 | 127 | 253 |
| 8 | 256 | 511 | 1021 |

:::danger If R < your lookback, part of the window is structurally invisible
With a 60-day window you need $k=3$ and 5 layers ($R = 63$). Four layers gives $R = 31$ — the model **cannot** use the first 29 days, and nothing tells you. Training proceeds and the loss falls.
:::

Bai et al.'s conclusion was that convolutions should be the *default* starting point for sequence modelling, not RNNs.

### Causal convolution — verify, do not assume

An ordinary `conv1d` with `padding=k//2` is **centred**: the output at $t$ depends on inputs at $t+1 \dots t+k/2$. For a forecaster that is look-ahead bias hidden inside a layer — the same class of bug as the scaler leak in [§5](./stock-prediction.md), and just as silent.

The test perturbs a future input and asserts no earlier output moves, and separately checks the measured receptive field against the formula rather than against itself.

## 3. Treating the Signal as a Language

Here is the second direction, and it is a genuinely different idea.

A language model predicts the next word, and a word is just an index into a finite dictionary. $\bar\delta$ is continuous — but bounded in practice, because prices do not deviate from their own moving average without limit.

**So bin it.** Slice the range into $B$ levels, replace each value with its bin index, and the series becomes a sequence of tokens over a vocabulary of size $B$. Every tool built for language now applies unchanged.

This is not a stretched analogy. It is what real systems do:

| System | What it quantized |
|---|---|
| **WaveNet** (2016) | raw audio → 256 μ-law levels, categorical softmax |
| **Chronos** (2024) | time series → fixed vocabulary, T5 + cross-entropy, sampling for probabilistic forecasts |

### What it buys

**A full predictive distribution, for free.** A softmax over $B$ bins *is* a distribution. [§9](./stock-prediction.md) already recommends "predict a distribution, not a point" — this delivers it as a by-product. Sample for intervals; read the entropy for confidence.

**Heavy tails stop dominating.** MSE is quadratic, so a few crash days own the gradient. Cross-entropy is bounded per example.

**Entropy is a signal regression cannot give you.** $\log_2 B$ bits means "no idea"; a sharp distribution means the model thinks it knows something. That is more actionable than one RMSE for the whole period.

### The floor you must compute first

With $B$ bins you can never predict better than half a bin width. Run the diagnostic **before** building anything:

```bash
uv run 04_intermediate_rnn_stock_data/train_token_lm.py --floor-only
```

### The result that matters — and it is not about resolution

On real AAPL $\bar\delta$, 2015–2025, bin edges fitted on the train split:

```
train range   [-21.07, 30.44]
test range    [-50.46, 32.97]     <- the 2022 drawdown
```

| bits | bins | uniform floor | quantile floor | clip rate |
|---|---|---|---|---|
| 4 | 16 | 2.5331 | 4.8970 | 3.49% |
| 6 | 64 | 2.2199 | 3.2039 | 3.49% |
| 8 | 256 | 2.1736 | 2.3790 | 3.49% |
| 10 | 1024 | 2.1639 | 2.2087 | 3.49% |
| 12 | 4096 | 2.1616 | 2.1742 | 3.49% |

:::danger Sixteen times the vocabulary buys nothing
The floor moves 2.53 → 2.16 from 4-bit to 12-bit and then **stops**. The residual is not bin width — it is **clipping**. 3.49% of test values fall outside the fitted range and pin to an end bin, and no amount of resolution fixes a value that is off the scale entirely.

This is [§7's non-stationarity](./stock-prediction.md) showing up as a concrete number.
:::

### The fix: scale before you quantize

This is the "scaling" half of Chronos's *"scaling and quantization"*, and it is not optional on financial data. Normalise each window by its own mean and mean-absolute-deviation before binning, so the vocabulary describes **shape relative to local context** rather than absolute level:

| | clip rate | error floor | headroom vs persistence |
|---|---|---|---|
| raw values | 3.57% | 2.2003 | 1.7× |
| **per-window scaling** | **0.01%** | **0.1028** | **37.3×** |

A **21× improvement in the floor**, from one preprocessing step. `train_token_lm.py --floor-only` prints a warning when the floor is within 2× of the bar, because at that point resolution — not modelling — is the binding constraint.

### Ordinality: what cross-entropy throws away

Cross-entropy treats bin 5 and bin 6 as exactly as different as bin 5 and bin 200. For language that blindness is correct — "cat" and "cats" being adjacent in the vocabulary means nothing. For a quantized real number it discards the single most useful piece of structure the labels have.

Replacing the one-hot target with a Gaussian over neighbouring bins,

$$
q_j \;\propto\; \exp\!\left(-\frac{(j - j^{*})^2}{2\sigma^2}\right)
$$

makes the loss distance-aware again. It is label smoothing *with a metric*.

### Measured

Seed 42, 30 epochs, 8-bit uniform, per-window scaling, $\sigma=1$:

```
RMSE               4.0990
persistence RMSE   3.8267
Theil U2           1.0712   loses to persistence
quantization floor 0.1027

mean entropy       5.15 bits of 8   (near-uniform: no idea)
```

**It loses — but it loses by less than any attention variant** (1.0712 vs 1.3212 for `lstm_attn`, 2.2647 for `lstm_mha`), with the quantization floor 37× below the bar, so resolution is nowhere near the constraint.

And the entropy reading is the genuinely new information: **5.15 of 8 bits is near-uniform.** The model is correctly reporting that it does not know. A regression head cannot say that — it emits a number with the same confident face whether it has learned something or nothing.

## 4. Honest Summary

| Direction | Result |
|---|---|
| Attention over an RNN ([§8](./stock-prediction.md)) | loses; more params monotonically worse |
| Modern TS architectures (§2) | `train_modern_ts.py --sweep` |
| Value tokenization (§3) | loses by less; gives calibrated uncertainty |

:::note There are no champions
A 2025 position paper surveyed the DLinear → PatchTST → TimeMixer exchange and concluded the models are close and the rankings move with the hyperparameter search. Treat every leaderboard in this area accordingly — including this page's.

The durable content here is not a ranking. It is: **compute the floor before you build**, **scale before you quantize**, **check the receptive field**, and **always report Theil $U_2$ against persistence.**
:::

## 5. Running It

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install deepspeed yfinance pandas scikit-learn
```

**CPU, no download** — the primitives and the diagnostic:

```bash
uv run 04_intermediate_rnn_stock_data/modern_ts_layers.py
uv run 04_intermediate_rnn_stock_data/tokenize_series.py
uv run tests/test_ts_forecasting.py          # 74 checks
```

**CoreWeave / SLURM:**

```bash
cd 04_intermediate_rnn_stock_data
MODEL=nbeats sbatch run_deepspeed.sh
sbatch run_deepspeed.sh --max-steps 20        # cheap dry run
```

**RunPod** — creates the pod and shuts it down:

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py run 04_intermediate_rnn_stock_data \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods       # confirm: "Nothing is billing."
```

## References

1. Oreshkin, B. N., et al. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*. [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
3. Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
4. Wang, S., et al. (2024). TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. *ICLR 2024*. [arXiv:2405.14616](https://arxiv.org/abs/2405.14616)
5. Zeng, A., et al. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
6. van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. [arXiv:1609.03499](https://arxiv.org/abs/1609.03499)
7. Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
8. Position: There are no Champions in Long-Term Time Series Forecasting (2025). [arXiv:2502.14045](https://arxiv.org/abs/2502.14045)
