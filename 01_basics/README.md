# Basic Examples

The mechanics of DeepSpeed on problems small enough to run end to end on one
machine in minutes. Every example here is fully runnable: synthetic or small
data, under a million parameters, one or two GPUs.

## Topics

| Folder | What it is |
|---|---|
| [`01_neuralnet/`](01_neuralnet/) | Fitting `y = 2x + 1` with two parameters. The smallest thing that is still a real DeepSpeed run. |
| [`02_convnet/`](02_convnet/) | MNIST CNN — the first example with a real dataset. |
| [`03_convnet_cifar10/`](03_convnet_cifar10/) | CIFAR-10, including a documented failure-and-recovery case study and three modern architectures that reach 93%. |
| [`04_rnn/`](04_rnn/) | An LSTM on sequence data. |

Each folder is self-contained and follows the same six-file contract (`CONTRIBUTING.md`):
a training script, a DeepSpeed config, a launcher, a README, a `pyproject.toml` and a
committed `uv.lock`. So:

```bash
cd 01_basics/01_neuralnet
uv sync
```

works from a fresh clone with no other setup.
