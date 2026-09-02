# Intermediate Examples

Still runnable on modest hardware, but the modelling questions get harder and
the honest answers get less comfortable.

## Topics

| Folder | What it is |
|---|---|
| [`01_bayesian_neuralnet/`](01_bayesian_neuralnet/) | Parallel-tempering MCMC — uncertainty, not point estimates. |
| [`02_rnn_stock_data/`](02_rnn_stock_data/) | Time-series forecasting, including why most of these models lose to persistence. |

Each folder is self-contained and follows the same six-file contract (`CONTRIBUTING.md`):
a training script, a DeepSpeed config, a launcher, a README, a `pyproject.toml` and a
committed `uv.lock`. So:

```bash
cd 02_intermediate/01_bayesian_neuralnet
uv sync
```

works from a fresh clone with no other setup.
