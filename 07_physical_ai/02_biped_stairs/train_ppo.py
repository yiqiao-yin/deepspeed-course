#!/usr/bin/env python3
"""
Train one cell of the 2x2, or sweep all four.

    uv run train_ppo.py --dry-run                   # 30 s smoke test
    uv run train_ppo.py --cell 2leg_free            # one cell
    uv run train_ppo.py --sweep --seeds 3           # all four, 3 seeds
    uv run train_ppo.py --cell 2leg_free --device cuda

THE EXPERIMENT
--------------
Four robots, one staircase, everything else identical:

    1leg_locked   1leg_free
    2leg_locked   2leg_free

The question is which switch costs more -- the extra limb, or the removed
constraint. The prediction from lab 1, and from the passive-stability
probe in `morphology.py`, is that the torso dominates and it is not close.

ON CPU VERSUS GPU
-----------------
Same as lab 1 and for the same reason: the policy is ~12k parameters, so
the forward pass is not the work. MuJoCo stepping is, and it happens on
the CPU either way. `--device auto` resolves to CPU; `--device cuda` is
supported and measured rather than recommended. The README reports both.

A sweep is four cells x N seeds of independent processes, so the useful
parallelism is `--sweep --jobs 8`, not a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def preflight() -> None:
    """Fail with a fix, not a traceback. No require_gpu(): see the README."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("=" * 72)
        print("  MuJoCo is not installed — nothing can be simulated")
        print("=" * 72)
        print("\n  Fix:   uv sync\n")
        print("  No GPU and no display are needed: training uses state")
        print("  observations, and rendering is a separate optional script.\n")
        sys.exit(1)


def main() -> None:
    from morphology import CELLS

    names = [c[0] for c in CELLS]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", choices=names, default="2leg_free")
    ap.add_argument("--sweep", action="store_true",
                    help="run every cell x every seed, then stop")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per cell. THREE IS THE MINIMUM that means "
                         "anything; lab 1's whole finding was that one run "
                         "supports either conclusion.")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel processes for --sweep")
    ap.add_argument("--total-steps", type=int, default=800_000)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
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
    args, _ = ap.parse_known_args()

    if args.dry_run:
        args.total_steps, args.n_envs, args.rollout = 8_192, 4, 128
    if args.max_steps > 0:
        args.total_steps = min(args.total_steps, args.max_steps)

    preflight()
    if args.sweep:
        sweep(args, names)
    else:
        run(args, args.cell, args.seed)


def sweep(args: argparse.Namespace, names: list[str]) -> None:
    """
    Fan the 2x2 out across processes.

    Independent runs, so this is embarrassingly parallel and the only
    sensible way to spend a multi-core machine on it. Each child is the
    same script with --cell and --seed pinned.
    """
    jobs = [(c, s) for c in names for s in range(args.seeds)]
    print("=" * 74)
    print(f"  SWEEP: {len(names)} cells x {args.seeds} seeds = "
          f"{len(jobs)} runs, {args.jobs} at a time")
    print(f"  {args.total_steps:,} steps each")
    print("=" * 74)

    t0 = time.time()
    running: list = []
    todo = list(jobs)
    while todo or running:
        while todo and len(running) < args.jobs:
            cell, seed = todo.pop(0)
            cmd = [sys.executable, str(HERE / "train_ppo.py"),
                   "--cell", cell, "--seed", str(seed),
                   "--total-steps", str(args.total_steps),
                   "--device", "cpu", "--quiet-child"]
            running.append(((cell, seed),
                            subprocess.Popen(cmd, cwd=HERE,
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE)))
            print(f"  started  {cell}  seed {seed}")
        time.sleep(1.0)
        for item in list(running):
            (cell, seed), proc = item
            if proc.poll() is not None:
                running.remove(item)
                ok = proc.returncode == 0
                print(f"  {'done    ' if ok else 'FAILED  '} {cell}  "
                      f"seed {seed}"
                      f"{'' if ok else '  ' + proc.stderr.read().decode()[-200:]}")

    print(f"\n  sweep finished in {(time.time() - t0) / 60:.1f} min")
    summarise()


def summarise() -> None:
    """Print the 2x2 as a table, straight from the written summaries."""
    import numpy as np

    from morphology import CELLS

    runs = HERE / "runs"
    print()
    print("=" * 74)
    print("  RESULTS — mean over seeds (individual seeds in brackets)")
    print("=" * 74)
    print(f"  {'cell':<14}{'baseline':>10}{'return':>10}"
          f"{'steps up':>10}{'top':>7}   per-seed return")
    for name, _, _ in CELLS:
        ds = sorted(runs.glob(f"{name}_s*/summary.json"))
        if not ds:
            print(f"  {name:<14}  (not run)")
            continue
        js = [json.loads(d.read_text()) for d in ds]
        ret = [j["final"]["return"] for j in js]
        print(f"  {name:<14}{np.mean([j['baseline_return'] for j in js]):>10.0f}"
              f"{np.mean(ret):>10.0f}"
              f"{np.mean([j['final']['climbed'] for j in js]):>10.2f}"
              f"{np.mean([j['final']['at_top'] for j in js]):>6.0%}   "
              f"{[round(r) for r in ret]}")
    print()
    print("  Read the spread before the means. Lab 1's finding was that one")
    print("  run per arm supports either conclusion.")


def run(args: argparse.Namespace, cell: str, seed: int) -> None:
    import numpy as np
    import torch

    from morphology import CELLS
    from ppo import (ActorCritic, RunningNorm, clipped_policy_loss,
                     compute_gae, pick_device)
    from stairs_env import BipedStairs, random_policy, rollout

    legs, locked = next((l, t) for n, l, t in CELLS if n == cell)
    device = pick_device(args.device)
    quiet = "--quiet-child" in sys.argv
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = HERE / "runs" / f"{cell}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)

    mk = lambda i: BipedStairs(legs=legs, locked_torso=locked, seed=seed + i)
    envs = [mk(i) for i in range(args.n_envs)]
    o_dim, a_dim = envs[0].obs_dim, envs[0].act_dim
    net = ActorCritic(o_dim, a_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(o_dim)

    if not quiet:
        print("=" * 74)
        print(f"  {cell}   seed {seed}   "
              f"{legs} leg(s), torso {'LOCKED' if locked else 'FREE'}")
        print("=" * 74)
        print(f"  device {device}   obs {o_dim}   act {a_dim}   "
              f"policy {net.n_params():,} params")
        print(f"  budget {args.total_steps:,} env steps"
              f"{'   (DRY RUN — a smoke test, not a result)' if args.dry_run else ''}")

    rng = np.random.default_rng(seed)
    base_env = mk(0)
    base = [rollout(base_env, random_policy(rng, a_dim), seed=1000 + s)
            for s in range(20)]
    base_ret = float(np.mean([b["return"] for b in base]))
    if not quiet:
        print(f"  BASELINE (random, 20 eps): return {base_ret:.1f}, "
              f"top {sum(b['at_top'] for b in base)}/20\n")

    obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    norm.update(obs)
    history: list[dict] = []
    done_steps, it, t0 = 0, 0, time.time()

    while done_steps < args.total_steps:
        it += 1
        B = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        for _ in range(args.rollout):
            nobs = norm(obs)
            with torch.no_grad():
                t = torch.as_tensor(nobs, dtype=torch.float32, device=device)
                action, logp, value = net.act(t)
            a = action.cpu().numpy()
            nxt, rew, dn = [], [], []
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(a[i])
                if term or trunc:
                    o, _ = e.reset()
                nxt.append(o); rew.append(r); dn.append(float(term or trunc))
            B["obs"].append(nobs); B["act"].append(a)
            B["logp"].append(logp.cpu().numpy())
            B["val"].append(value.cpu().numpy())
            B["rew"].append(rew); B["done"].append(dn)
            obs = np.stack(nxt); norm.update(obs); done_steps += args.n_envs

        with torch.no_grad():
            last_v = net.value(torch.as_tensor(norm(obs), dtype=torch.float32,
                                               device=device))
        T = lambda k: torch.as_tensor(np.array(B[k]), dtype=torch.float32,
                                      device=device)
        adv, ret = compute_gae(T("rew"), T("val"), T("done"), last_v,
                               args.gamma, args.lam)
        b_obs = T("obs").reshape(-1, o_dim)
        b_act = T("act").reshape(-1, a_dim)
        b_logp, b_adv, b_ret = T("logp").reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        mb = max(1, n // args.minibatches)
        for _ in range(args.epochs):
            for idx in torch.randperm(n, device=device).split(mb):
                logp, ent, value = net.evaluate(b_obs[idx], b_act[idx])
                pi_loss, _ = clipped_policy_loss(logp, b_logp[idx],
                                                 b_adv[idx], args.clip)
                loss = (pi_loss + args.vf_coef * ((value - b_ret[idx]) ** 2).mean()
                        - args.ent_coef * ent.mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if it % 3 == 0 or done_steps >= args.total_steps:
            ev = evaluate(net, norm, device, legs, locked, seed)
            ev.update(step=done_steps, seconds=round(time.time() - t0, 1))
            history.append(ev)
            if not quiet:
                print(f"  {done_steps:>8,}  return {ev['return']:7.1f}  "
                      f"max_x {ev['max_x']:5.2f}  climbed {ev['climbed']:.2f}"
                      f"  top {ev['at_top']:4.0%}  {ev['seconds']:5.0f}s")

    final = history[-1]
    (out / "summary.json").write_text(json.dumps({
        "cell": cell, "legs": legs, "locked_torso": locked, "seed": seed,
        "device": device, "total_steps": done_steps, "dry_run": args.dry_run,
        "params": net.n_params(), "obs_dim": o_dim, "act_dim": a_dim,
        "wall_seconds": round(time.time() - t0, 1),
        "baseline_return": round(base_ret, 2), "final": final,
    }, indent=2))
    with (out / "curve.csv").open("w") as fh:
        cols = list(history[0])
        fh.write(",".join(cols) + "\n")
        for h in history:
            fh.write(",".join(str(h[c]) for c in cols) + "\n")
    torch.save({"model": net.state_dict(), "norm": norm.state_dict(),
                "legs": legs, "locked": locked}, out / "policy.pt")

    if not quiet:
        print(f"\n  BASELINE return {base_ret:7.1f}")
        print(f"  TRAINED  return {final['return']:7.1f}   "
              f"climbed {final['climbed']:.2f}/3   top {final['at_top']:.0%}")
        if args.dry_run:
            print("\n  DRY RUN — a smoke test, not a result.")


def evaluate(net, norm, device, legs, locked, seed, episodes=10) -> dict:
    """Score the deterministic policy; report behaviour, not just return."""
    import numpy as np
    import torch

    from stairs_env import BipedStairs, rollout

    def policy(o):
        with torch.no_grad():
            t = torch.as_tensor(norm(o), dtype=torch.float32,
                                device=device).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).cpu().numpy()

    env = BipedStairs(legs=legs, locked_torso=locked, seed=seed)
    rs = [rollout(env, policy, seed=5000 + i) for i in range(episodes)]
    ends = [r["ended_by"] for r in rs]
    return {
        "return": round(float(np.mean([r["return"] for r in rs])), 2),
        "max_x": round(float(np.mean([r["max_x"] for r in rs])), 3),
        "climbed": round(float(np.mean([r["climbed"] for r in rs])), 3),
        "at_top": round(sum(r["at_top"] for r in rs) / len(rs), 3),
        "tipped_frac": round(ends.count("tipped") / len(ends), 3),
        "collapsed_frac": round(ends.count("collapsed") / len(ends), 3),
    }


if __name__ == "__main__":
    main()
