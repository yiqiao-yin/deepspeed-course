#!/usr/bin/env python3
"""
Dynamics models for the cold-gas car, at sizes from 10k to 100M.

    uv run dynamics.py                  # the size ladder and its memory
    uv run dynamics.py --sizes tiny,xl  # two of them

WHAT THE MODEL LEARNS
---------------------
Given the 41-number observation and the 10 valve commands, predict
the NEXT observation. It is trained on the residual, not the state:

    o_{t+1} = o_t + f(o_t, u_t)

because at 100 Hz consecutive observations differ by very little and
a model asked for the absolute state spends all its capacity
learning the identity. Predicting the delta makes the identity free
and the dynamics the only thing left to fit. This is the standard
trick and it is worth a factor of several in error.

WHY THERE IS A SIZE LADDER
---------------------------
This topic has said five times that a 13k-parameter policy has
nothing for ZeRO to shard, and that is true of the POLICY. It says
nothing about a dynamics model, which has a harder job: the policy
only has to pick an action, the model has to reproduce real-gas
blowdown, valve lag, contact and terrain.

So the lab asks the question properly, with a bar written down
first: **a model 1000x larger must cut multi-step prediction error
by at least half, or the sharding machinery is not earning its
place.** The ladder spans 10k to ~100M so the answer is measured on
this card rather than asserted.

THE ARCHITECTURES
-----------------
`mlp`  residual MLP over a single (obs, act) pair. No history, so it
       cannot represent anything the observation does not carry --
       which includes the fault, deliberately.
`gru`  the same, with a recurrent state across the episode. Can in
       principle infer a stuck nozzle from the mismatch between
       commanded and realised motion.

Both scale by width and depth through the same table, so a size
comparison is a comparison.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# (hidden, layers) per rung. Chosen so each step is roughly 8-10x the
# last in parameters, which makes "1000x larger" three rungs rather
# than an awkward interpolation.
SIZES: dict[str, tuple[int, int]] = {
    "tiny":   (32, 2),
    "small":  (96, 2),
    "medium": (256, 3),
    "large":  (768, 4),
    "xl":     (2048, 6),
    "xxl":    (4096, 8),
}


class ResidualMLP(nn.Module):
    """Predict the state DELTA from (obs, act)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        dims = [obs_dim + act_dim] + [hidden] * layers
        body: list[nn.Module] = []
        for a, b in zip(dims, dims[1:]):
            body += [nn.Linear(a, b), nn.SiLU()]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(hidden, obs_dim)
        # Zero the head so an untrained model predicts NO CHANGE
        # rather than noise. That is the right prior at 100 Hz and it
        # makes the first epochs stable; without it the rollout
        # diverges before the model has learned anything and the
        # early curve is all explosion.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, act: torch.Tensor,
                state=None):
        h = self.body(torch.cat([obs, act], dim=-1))
        return obs + self.head(h), None


class ResidualGRU(nn.Module):
    """
    The same, with memory across the episode.

    A stuck nozzle is not in the observation. It is visible only as a
    persistent disagreement between what was commanded and how the
    vehicle moved, which is a fact about the HISTORY. An MLP cannot
    represent it at all; a recurrent model can in principle.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.inp = nn.Linear(obs_dim + act_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=max(1, layers - 1),
                          batch_first=True)
        self.head = nn.Linear(hidden, obs_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, state=None):
        # obs/act are (B, T, D) when training on sequences and
        # (B, D) when stepping a rollout. One path, two shapes.
        squeeze = obs.dim() == 2
        if squeeze:
            obs = obs.unsqueeze(1)
            act = act.unsqueeze(1)
        h = F.silu(self.inp(torch.cat([obs, act], dim=-1)))
        y, state = self.rnn(h, state)
        out = obs + self.head(y)
        return (out.squeeze(1) if squeeze else out), state


def build(kind: str, size: str, obs_dim: int, act_dim: int) -> nn.Module:
    if size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}, got {size!r}")
    hidden, layers = SIZES[size]
    cls = {"mlp": ResidualMLP, "gru": ResidualGRU}
    if kind not in cls:
        raise ValueError(f"kind must be mlp or gru, got {kind!r}")
    return cls[kind](obs_dim, act_dim, hidden, layers)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def memory_table(obs_dim: int = 41, act_dim: int = 10) -> list[dict]:
    """
    Weights plus Adam state per card, which is what decides whether
    a model fits before any activation is allocated.

    Adam keeps two fp32 moments beside the fp32 parameter and the
    gradient, so the floor is 16 bytes per parameter. ZeRO-2 shards
    the gradient and the optimizer state (12 of those 16); ZeRO-3
    shards the parameter too. Activations are NOT in this table and
    are not small -- topic 7 predicted 7.3 GB for a LoRA arm and
    measured 11.5.
    """
    rows = []
    for size in SIZES:
        for kind in ("mlp", "gru"):
            m = build(kind, size, obs_dim, act_dim)
            n = n_params(m)
            full = n * 16 / 1e9                     # param+grad+2 moments
            rows.append(dict(size=size, kind=kind, params=n,
                             gb_1card=full,
                             gb_zero2=n * 4 / 1e9 + n * 12 / 1e9 / 8,
                             gb_zero3=full / 8))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="")
    a, _ = ap.parse_known_args()
    want = set(a.sizes.split(",")) if a.sizes else set(SIZES)

    print("=" * 78)
    print("  DYNAMICS MODEL SIZE LADDER")
    print("=" * 78)
    print("  Weights + Adam state per card. Activations are NOT included")
    print("  and are not small -- topic 7 predicted 7.3 GB for a LoRA arm")
    print("  and measured 11.5. ZeRO shards parameters, gradients and")
    print("  optimizer state; it never shards activations.\n")
    print(f"  {'size':<8}{'kind':<6}{'params':>14}{'1 card':>10}"
          f"{'ZeRO-2 x8':>11}{'ZeRO-3 x8':>11}")
    for r in memory_table():
        if r["size"] not in want:
            continue
        print(f"  {r['size']:<8}{r['kind']:<6}{r['params']:>14,}"
              f"{r['gb_1card']:>9.2f}G{r['gb_zero2']:>10.2f}G"
              f"{r['gb_zero3']:>10.2f}G")
    print("\n  The bar, written down before the measurement: a model 1000x")
    print("  larger must cut multi-step prediction error by at least HALF,")
    print("  or the sharding machinery is not earning its place here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
