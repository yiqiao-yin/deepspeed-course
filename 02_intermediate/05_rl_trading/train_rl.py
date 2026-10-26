#!/usr/bin/env python3
"""
Train a trading policy by reward instead of by loss: PPO or GRPO.

    uv run train_rl.py --method ppo                 # a critic learns the baseline
    uv run train_rl.py --method grpo                # the BASKET is the baseline
    uv run train_rl.py --method grpo --neutral      # dollar-neutral book
    uv run train_rl.py --dry-run                    # ~1 min, proves it assembles

ONE SCRIPT, TWO METHODS, AND THE DIFFERENCE IS ONE LINE
---------------------------------------------------------
PPO and GRPO share everything here -- the same network, the same
clipped surrogate, the same data, the same budget. They differ in
where the ADVANTAGE baseline comes from, which is the only thing
that distinguishes them, and keeping them in one file is what
makes that visible. `03_llms/05_dpo` carries six objectives in one
trainer for exactly this reason.

    PPO    a second network learns V(s) and the advantage is
           r + gamma*V(s') - V(s). The baseline is fitted, and
           every bit of it has to be learned from the same noisy
           reward the policy is learning from.

    GRPO   no critic at all. Each sample is scored against the
           MEAN OF ITS GROUP.

WHY GRPO FITS THIS PROBLEM UNUSUALLY WELL
-------------------------------------------
GRPO was built for language models, where a "group" is several
sampled answers to one prompt and the baseline is their mean
score. Transplanted here, the natural group is **every sector on
the same day**, and then something nice happens:

    advantage[i,t] = r[i,t] - mean over the basket on day t

which is, to within a normalisation, the EXCESS RETURN of sector
i over an equal-weighted market. The baseline is not approximated
by a network -- it IS the market, exactly, for free, every day.
A policy trained on it optimises market-relative performance
without anyone writing a benchmark into the reward.

That is long-short equity falling out of the algorithm rather
than being bolted on, and it is the reason this lab exists.

WHAT THE POLICY ACTUALLY DOES
-------------------------------
State: the eight features for that sector today, PLUS the position
currently held in it. The position has to be in the state or the
policy cannot know that holding is free and flipping is not --
see `market.py`.

Action: a target weight in [-1, +1] per sector, squashed by tanh.
Reward: `w * r_next - cost * |w - w_prev|`, exactly `market.pnl`.

THE REWARD IS THE P&L. There is no shaping, no auxiliary
prediction loss, no risk penalty bolted on. Every result in this
lab is what falls out of optimising money after costs, which is
the only way the comparison against a forecast trained on squared
error means anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent


def require_cpu_or_gpu(width: int) -> None:
    """
    No unconditional `require_gpu()`, deliberately.

    The headline policy is ~5k parameters over nine assets and runs
    faster on a CPU than on a card -- the batch never fills one.
    Topic 7 measured a policy this size at 1.7x SLOWER on a GPU. A
    guard that can never fire is decoration.

    `--width` above 512 builds the cross-sectional model that
    `capacity.py` sizes, and that one genuinely wants a card.
    """
    if width <= 512:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    raise SystemExit(
        f"\n  --width {width} builds the cross-sectional model, which "
        f"wants a GPU.\n\n"
        f"  What runs right now on this machine:\n"
        f"      uv run train_rl.py --method grpo\n"
        f"      uv run baselines.py\n"
        f"      uv run market.py --cost-curve\n"
        f"      uv run capacity.py\n\n"
        f"  Or set ALLOW_CPU=1 to try it anyway.\n")


def build(obs_dim: int, width: int, critic: bool):
    import numpy as np
    import torch
    import torch.nn as nn

    def body():
        return nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(),
                             nn.Linear(width, width), nn.Tanh())

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi_body = body()
            self.mu = nn.Linear(width, 1)
            self.v_body = body() if critic else None
            self.v_head = nn.Linear(width, 1) if critic else None
            self.log_std = nn.Parameter(torch.full((1,), -0.7))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.zeros_(m.bias)
            # Small final gain: start near w = 0, i.e. FLAT. Starting
            # at a random large position means the first thousand
            # updates are spent paying to unwind it.
            nn.init.orthogonal_(self.mu.weight, 0.01)

        def dist(self, x):
            return torch.distributions.Normal(torch.tanh(self.mu(self.pi_body(x))),
                                              self.log_std.exp())

        def value(self, x):
            return self.v_head(self.v_body(x)).squeeze(-1)

        def n_params(self):
            return sum(p.numel() for p in self.parameters())

    return Policy()


def rollout(net, X, y, cost, device):
    """
    One pass over the training window, carrying the position forward.

    Sequential and not shuffled: the position at t depends on the
    action at t-1, which is the entire reason this is a decision
    problem rather than a regression. Shuffling here would silently
    destroy the cost structure and every arm would look better.
    """
    import numpy as np
    import torch

    T, N, F = X.shape
    pos = torch.zeros(N, device=device)
    S, A, LP, R = [], [], [], []
    with torch.no_grad():
        for t in range(T):
            s = torch.cat([torch.as_tensor(X[t], device=device),
                           pos.unsqueeze(1)], dim=1)
            d = net.dist(s)
            a = d.sample()
            lp = d.log_prob(a).sum(-1)
            new = a.squeeze(-1).clamp(-1.0, 1.0)
            r = new * torch.as_tensor(y[t], device=device) \
                - (new - pos).abs() * cost
            S.append(s); A.append(a); LP.append(lp); R.append(r)
            pos = new
    return (torch.stack(S), torch.stack(A), torch.stack(LP),
            torch.stack(R))


def advantages(method, net, S, R, gamma, lam):
    """
    The one line that separates the two methods.

    GRPO: subtract the basket mean on each day and divide by its
    spread. No critic, no bootstrapping, no gamma -- the baseline
    is the market and it is exact.

    PPO: GAE over a learned value function, per asset, along time.
    """
    import torch

    T, N = R.shape
    if method == "grpo":
        # THE GROUP IS THE DAY. Subtracting the cross-sectional mean
        # makes the advantage an excess return over an equal-weighted
        # market; dividing by the spread puts quiet days and violent
        # days on the same scale.
        centred = R - R.mean(dim=1, keepdim=True)
        return centred / (R.std(dim=1, keepdim=True) + 1e-8), None

    with torch.no_grad():
        V = net.value(S.reshape(T * N, -1)).reshape(T, N)
    adv = torch.zeros_like(R)
    last = torch.zeros(N, device=R.device)
    for t in reversed(range(T)):
        nxt = V[t + 1] if t + 1 < T else torch.zeros(N, device=R.device)
        delta = R[t] + gamma * nxt - V[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return (adv - adv.mean()) / (adv.std() + 1e-8), adv + V


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="grpo", choices=("ppo", "grpo"))
    ap.add_argument("--name", default="")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neutral", action="store_true",
                    help="subtract the daily mean position, making the "
                         "book dollar-neutral: it can only earn from the "
                         "RELATIVE call, never from market direction")
    ap.add_argument("--deepspeed", action="store_true")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args, _ = ap.parse_known_args()

    require_cpu_or_gpu(args.width)

    import numpy as np
    import torch

    import data as D
    import market as M

    if args.dry_run:
        args.iters = 4
        args.epochs = 1

    name = args.name or f"{args.method}{'_neutral' if args.neutral else ''}_s{args.seed}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    X, y, dates, names, _ = D.panel("sectors")
    (Xtr, ytr, dtr, _), (Xte, yte, dte, _) = D.split(X, y, dates)
    T, N, F = Xtr.shape
    net = build(F + 1, args.width, critic=(args.method == "ppo")).to(device)

    print("=" * 80)
    print(f"  TRAIN — {name}: {args.method.upper()} on "
          f"{N} sectors, reward = P&L after {args.cost * 1e4:.0f} bps")
    print("=" * 80)
    print(f"  policy   {net.n_params():,} params, width {args.width}, "
          f"{'with a critic' if args.method == 'ppo' else 'NO critic'}")
    if args.method == "grpo":
        print(f"  baseline the basket mean on each day — which IS the "
              f"equal-weight market")
    print(f"  train    {T} days {dtr[0].date()}..{dtr[-1].date()}")
    print(f"  test     {len(dte)} days {dte[0].date()}..{dte[-1].date()} "
          f"(held out)")

    engine = None
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    if args.deepspeed:
        import os

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        import deepspeed

        cfg = json.loads((HERE / "ds_config.json").read_text())
        cfg["train_micro_batch_size_per_gpu"] = args.batch
        # A TORCH optimiser is handed in rather than named in the
        # config: DeepSpeed's fused Adam is JIT-compiled with nvcc,
        # and the PyPI CUDA wheels ship only ptxas.
        engine, opt, _, _ = deepspeed.initialize(
            model=net, optimizer=opt, config=cfg)
        print(f"  deepspeed ZeRO stage "
              f"{cfg['zero_optimization']['stage']}")

    history, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        S, A, LP, R = rollout(net, Xtr, ytr, args.cost, device)
        adv, ret = advantages(args.method, net, S, R, args.gamma, args.lam)
        n = T * N
        Sf, Af = S.reshape(n, -1), A.reshape(n, -1)
        LPf, ADVf = LP.reshape(n), adv.reshape(n)
        RETf = ret.reshape(n) if ret is not None else None

        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for k in range(0, n, args.batch):
                b = perm[k:k + args.batch]
                d = net.dist(Sf[b])
                lp = d.log_prob(Af[b]).sum(-1)
                ratio = (lp - LPf[b]).exp()
                pl = -torch.min(
                    ratio * ADVf[b],
                    ratio.clamp(1 - args.clip, 1 + args.clip) * ADVf[b]).mean()
                loss = pl - args.ent_coef * d.entropy().sum(-1).mean()
                if RETf is not None:
                    loss = loss + 0.5 * ((net.value(Sf[b]) - RETf[b]) ** 2).mean()
                if engine is not None:
                    engine.backward(loss); engine.step()
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()

        if it % 20 == 0 or it == args.iters:
            pos = evaluate_positions(net, Xtr, device, args.neutral)
            s = M.stats(M.book(pos, ytr, args.cost))
            history.append({"iter": it, "train_ann": s["ann"],
                            "train_sharpe": s["sharpe"],
                            "turnover": M.turnover(pos)})
            print(f"  it {it:>4}/{args.iters}  train {s['ann']:>7.1%}/yr  "
                  f"sharpe {s['sharpe']:>5.2f}  "
                  f"turnover {M.turnover(pos):.3f}", flush=True)

    # --- held out -----------------------------------------------------
    pos = evaluate_positions(net, Xte, device, args.neutral)
    M.assert_causal(lambda Z: evaluate_positions(net, Z, device, args.neutral),
                    Xte)
    daily = M.book(pos, yte, args.cost)
    s = M.stats(daily)
    sd = (engine.module if engine is not None else net).state_dict()
    torch.save({"model": sd, "obs_dim": F + 1, "width": args.width,
                "method": args.method, "neutral": args.neutral}, out / "policy.pt")
    np.save(out / "positions.npy", pos)
    summary = {"name": name, "method": args.method, "neutral": args.neutral,
               "seed": args.seed, "params": net.n_params(),
               "cost": args.cost, "iters": args.iters,
               "wall_seconds": time.time() - t0, "deepspeed": args.deepspeed,
               "test": s, "turnover": M.turnover(pos), "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  wrote {out}/policy.pt, positions.npy, summary.json")
    print(f"  HELD OUT  {s['ann']:+.1%}/yr   sharpe {s['sharpe']:.2f}   "
          f"maxDD {s['maxdd']:.1%}   turnover {M.turnover(pos):.3f}")
    print(f"  (buy-and-hold on the same window: "
          f"{M.stats(M.book(np.ones_like(yte), yte, args.cost))['ann']:+.1%}/yr)")
    if args.dry_run:
        print("\n  DRY RUN — 4 iterations, not a result.")
    return 0


def evaluate_positions(net, X, device, neutral=False):
    """Deterministic: the mean action, with the position carried forward."""
    import numpy as np
    import torch

    T, N, _ = X.shape
    pos = torch.zeros(N, device=device)
    out = []
    with torch.no_grad():
        for t in range(T):
            s = torch.cat([torch.as_tensor(X[t], device=device),
                           pos.unsqueeze(1)], dim=1)
            pos = torch.tanh(net.mu(net.pi_body(s))).squeeze(-1).clamp(-1, 1)
            out.append(pos.cpu().numpy().copy())
    P = np.asarray(out)
    if neutral:
        # Dollar-neutral: the book holds no net market exposure, so
        # it can only earn from being RIGHT ABOUT WHICH SECTOR, never
        # from the market going up. This is the strictest test of
        # whether the group baseline learned anything relative.
        P = P - P.mean(axis=1, keepdims=True)
    return P


if __name__ == "__main__":
    raise SystemExit(main())
