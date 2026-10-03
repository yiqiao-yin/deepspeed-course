#!/usr/bin/env python3
"""
Train the hopper to get past an obstacle of unpredictable height.

    uv run train_ppo.py --dry-run                 # 30s, proves it assembles
    uv run train_ppo.py                           # ~5 min on CPU
    uv run train_ppo.py --device cuda             # works; see below
    uv run train_ppo.py --blind-to-height         # the ablation

ON CPU VERSUS GPU, HONESTLY
---------------------------
This runs on either and the README reports both. **CPU is normally faster
here**, and saying so is more useful than pretending otherwise: the policy
is ~10k parameters, so the matrix multiplications are trivial, and the real
cost is MuJoCo stepping sixteen environments on the CPU regardless. Moving a
10k-parameter forward pass to a GPU adds transfer latency to the one part
that was never the bottleneck.

`--device cuda` is supported, measured, and reported rather than
recommended. The GPU story in `07_physical_ai` is real from lab 2 onward,
where the policy is a vision-language-action model that does not fit on one
card. Using a distributed launcher here, on 10k parameters, would be the
cargo cult this repository has a rule about.

WHAT GETS WRITTEN
-----------------
`runs/<name>/` gets `curve.csv` (every evaluation), `policy.pt`, and
`summary.json`. The figures on the book page are generated from those files
by `make_figures.py`, so a plot can never show something the run did not
produce.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def preflight() -> None:
    """
    Check the simulator imports before anything slow happens.

    There is deliberately no `require_gpu()` here, and the reason is the
    same one that keeps a DeepSpeed launcher out of this lab: the
    environment runs on a CPU, the policy is 10k parameters, and a guard
    that can never legitimately fire is decoration. `03_llms/01_llm_fine-
    tuning/analyze_kimi_k3.py` carries the same note for the same reason.

    What CAN go wrong is MuJoCo failing to import, so that is what is
    checked, with the fix rather than a traceback.
    """
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("=" * 72)
        print("  MuJoCo is not installed — this lab cannot simulate anything")
        print("=" * 72)
        print("\n  Fix:   uv sync")
        print("\n  MuJoCo is pip-installable and needs no licence since "
              "DeepMind")
        print("  released it. No GPU and no display are required: training")
        print("  uses state observations, and rendering is only needed for")
        print("  the optional video in make_figures.py.\n")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=400_000)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="hard cap on environment steps; -1 means use "
                         "--total-steps. A capped run is a smoke test, "
                         "not a result.")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny run that proves the pipeline assembles")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-to-height", action="store_true",
                    help="zero the obstacle-height input: the ablation the "
                         "test suite uses to prove the robot is looking")
    ap.add_argument("--name", default=None)
    args, _ = ap.parse_known_args()

    if args.dry_run:
        args.total_steps, args.n_envs, args.rollout = 8_192, 4, 128
    if args.max_steps > 0:
        args.total_steps = min(args.total_steps, args.max_steps)

    preflight()
    run(args)


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from obstacle_env import ACT_DIM, OBS_DIM, ObstacleHopper, rollout, \
        random_policy, WALKABLE_MAX, BOX_HEIGHT_MIN, BOX_HEIGHT_MAX
    from ppo import ActorCritic, RunningNorm, clipped_policy_loss, \
        compute_gae, pick_device

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    name = args.name or ("blind" if args.blind_to_height else "seeing")
    out = Path(__file__).parent / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    envs = [ObstacleHopper(seed=args.seed + i,
                           blind_to_height=args.blind_to_height)
            for i in range(args.n_envs)]
    net = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(OBS_DIM)

    print("=" * 74)
    print(f"  Obstacle hopper — PPO   [{'BLIND to height' if args.blind_to_height else 'sees the obstacle'}]")
    print("=" * 74)
    print(f"  device {device}   envs {args.n_envs}   "
          f"policy {net.n_params():,} params")
    print(f"  budget {args.total_steps:,} env steps"
          f"{'   (DRY RUN — a smoke test, not a result)' if args.dry_run else ''}")
    print(f"  box height ~ U({BOX_HEIGHT_MIN}, {BOX_HEIGHT_MAX}) m, "
          f"redrawn every episode")
    print()

    # ---- the baseline, measured before training, never assumed ----------
    rng = np.random.default_rng(args.seed)
    base_env = ObstacleHopper(seed=args.seed,
                              blind_to_height=args.blind_to_height)
    base = [rollout(base_env, random_policy(rng), seed=1000 + s)
            for s in range(20)]
    base_ret = float(np.mean([b["return"] for b in base]))
    base_clear = sum(b["cleared"] for b in base) / len(base)
    print(f"  BASELINE (random policy, 20 episodes): "
          f"return {base_ret:.1f}, cleared {base_clear:.0%}")
    print()

    obs = np.stack([e.reset(seed=args.seed + i)[0]
                    for i, e in enumerate(envs)])
    norm.update(obs)

    history: list[dict] = []
    steps_done = 0
    t0 = time.time()
    iteration = 0

    while steps_done < args.total_steps:
        iteration += 1
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = \
            [], [], [], [], [], []

        for _ in range(args.rollout):
            nobs = norm(obs)
            with torch.no_grad():
                t = torch.as_tensor(nobs, dtype=torch.float32, device=device)
                action, logp, value = net.act(t)
            a = action.cpu().numpy()

            nxt, rew, done = [], [], []
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(a[i])
                if term or trunc:
                    o, _ = e.reset()
                nxt.append(o)
                rew.append(r)
                done.append(float(term or trunc))

            buf_obs.append(nobs)
            buf_act.append(a)
            buf_logp.append(logp.cpu().numpy())
            buf_val.append(value.cpu().numpy())
            buf_rew.append(rew)
            buf_done.append(done)

            obs = np.stack(nxt)
            norm.update(obs)
            steps_done += args.n_envs

        with torch.no_grad():
            last_v = net.value(torch.as_tensor(norm(obs), dtype=torch.float32,
                                               device=device))

        to_t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32,
                                         device=device)
        adv, ret = compute_gae(to_t(buf_rew), to_t(buf_val), to_t(buf_done),
                               last_v, args.gamma, args.lam)

        b_obs = to_t(buf_obs).reshape(-1, OBS_DIM)
        b_act = to_t(buf_act).reshape(-1, ACT_DIM)
        b_logp = to_t(buf_logp).reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        mb = max(1, n // args.minibatches)
        for _ in range(args.epochs):
            for idx in torch.randperm(n, device=device).split(mb):
                logp, ent, value = net.evaluate(b_obs[idx], b_act[idx])
                pi_loss, stats = clipped_policy_loss(
                    logp, b_logp[idx], b_adv[idx], args.clip)
                v_loss = ((value - b_ret[idx]) ** 2).mean()
                loss = (pi_loss + args.vf_coef * v_loss
                        - args.ent_coef * ent.mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if iteration % 2 == 0 or steps_done >= args.total_steps:
            ev = evaluate(net, norm, device, args.blind_to_height, args.seed)
            ev.update(step=steps_done, seconds=round(time.time() - t0, 1),
                      clip_fraction=round(stats["clip_fraction"], 4))
            history.append(ev)
            print(f"  {steps_done:>8,} steps  "
                  f"return {ev['return']:7.1f}  "
                  f"max_x {ev['max_x']:5.2f}m  "
                  f"cleared {ev['cleared_all']:5.0%}  "
                  f"(low {ev['cleared_low']:4.0%} / high {ev['cleared_high']:4.0%})  "
                  f"{ev['seconds']:5.0f}s")

    final = history[-1]
    summary = {
        "name": name, "device": device, "blind_to_height": args.blind_to_height,
        "total_steps": steps_done, "dry_run": args.dry_run,
        "seed": args.seed, "params": net.n_params(),
        "wall_seconds": round(time.time() - t0, 1),
        "baseline_return": round(base_ret, 2),
        "baseline_cleared": base_clear,
        "final": final,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "curve.csv").open("w") as fh:
        cols = list(history[0])
        fh.write(",".join(cols) + "\n")
        for h in history:
            fh.write(",".join(str(h[c]) for c in cols) + "\n")
    torch.save({"model": net.state_dict(), "norm": norm.state_dict()},
               out / "policy.pt")

    print()
    print("=" * 74)
    if args.dry_run:
        print("  DRY RUN COMPLETE — this is a smoke test, not a result.")
        print(f"  It proves the pipeline assembles on {device}. For a real")
        print("  run drop --dry-run; it takes about five minutes.")
    else:
        print(f"  trained {steps_done:,} steps in "
              f"{summary['wall_seconds']:.0f}s on {device}")
        print(f"  BASELINE  return {base_ret:7.1f}   cleared {base_clear:.0%}")
        print(f"  TRAINED   return {final['return']:7.1f}   "
              f"cleared {final['cleared_all']:.0%}")
        print(f"            low boxes  {final['cleared_low']:.0%}   "
              f"high boxes {final['cleared_high']:.0%}")
    print(f"  wrote {out.relative_to(Path(__file__).parent)}/"
          "{curve.csv, policy.pt, summary.json}")
    print("=" * 74)


def evaluate(net, norm, device: str, blind: bool, seed: int,
             episodes: int = 12) -> dict:
    """
    Score the current policy, split by obstacle height.

    The split is the whole point. A single clear-rate averages over a task
    whose difficulty varies by an order of magnitude, and a policy that
    walks over every low box and fails every high one scores the same as
    one that is mediocre everywhere. Those are different policies and the
    lab is about telling them apart.
    """
    import numpy as np
    import torch

    from obstacle_env import ObstacleHopper, rollout, WALKABLE_MAX, \
        BOX_HEIGHT_MIN, BOX_HEIGHT_MAX

    def policy(o):
        with torch.no_grad():
            t = torch.as_tensor(norm(o), dtype=torch.float32,
                                device=device).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).cpu().numpy()

    rows = []
    for band, (lo, hi) in (("low", (BOX_HEIGHT_MIN, WALKABLE_MAX)),
                           ("high", (0.12, BOX_HEIGHT_MAX))):
        env = ObstacleHopper(seed=seed, height_range=(lo, hi),
                             blind_to_height=blind)
        rows += [(band, rollout(env, policy, seed=5000 + i))
                 for i in range(episodes // 2)]

    allr = [r for _, r in rows]
    low = [r for b, r in rows if b == "low"]
    high = [r for b, r in rows if b == "high"]
    return {
        "return": round(float(np.mean([r["return"] for r in allr])), 2),
        "max_x": round(float(np.mean([r["max_x"] for r in allr])), 3),
        "cleared_all": round(sum(r["cleared"] for r in allr) / len(allr), 3),
        "cleared_low": round(sum(r["cleared"] for r in low) / len(low), 3),
        "cleared_high": round(sum(r["cleared"] for r in high) / len(high), 3),
    }


if __name__ == "__main__":
    main()
