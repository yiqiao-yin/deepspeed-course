#!/usr/bin/env python3
"""
Train a cold-gas car to land on a marked spot, by PPO.

    uv run train_target.py --dry-run                      # 40 s, assembles
    uv run train_target.py --sense blind --name blind_s0
    uv run train_target.py --sense patch --seed 1 --name patch_s1

THE PROBLEM, STATED THE WAY RL NEEDS IT
----------------------------------------
GOAL             arrive on the wheels within 4 m of a marked spot,
                 on ground gentle enough to survive: under 2.5 m/s
                 sink, under 25 degrees of tilt, under 1.2 rad/s of
                 spin. Close but broken does not count, and intact
                 but elsewhere does not count.

DEGREES OF FREEDOM
                 6 for the vehicle (3 position, 3 attitude) plus 4
                 wheel spins. The policy commands N valve openings in
                 [0, 1] at 100 Hz, where N is the rig's nozzle count.

CONSTRAINTS      propellant is finite (297 kg, about 7 s of hover)
                 and thrust fades as the tank cools; nozzles push and
                 never pull; 25 ms of valve lag and 10 ms of transport
                 delay; the clock; anything but a wheel touching the
                 ground ends the episode. The ground is badlands --
                 packed boulder fields, not a parking lot.

WHAT THE THREE ARMS VARY, AND WHY THAT IS THE LAB
--------------------------------------------------
The arms differ in ONE thing: how much of the ground the policy is
allowed to see. Everything else -- rig, terrain, reward, budget,
seeds -- is held fixed.

    blind   the mark and the vehicle's own state. No terrain at all.
    slope   adds the two-component ground gradient under the car.
    patch   adds a 5 x 5 height patch spanning 8 m, centred on the car.

The question is whether terrain knowledge is worth anything to a
lander. The hand-written controller says no -- see the README's
divert-budget sweep, where the best use of perfect terrain knowledge
turned out to be ignoring it. That is a statement about ONE
controller. These arms ask whether a learned policy finds value in
the same information, which is a different question and the only one
PPO can answer.

A FAIR COMPARISON NEEDS THE ARMS TO DIFFER ONLY IN THE INPUT
-------------------------------------------------------------
Each sense changes obs_dim, so the three policies are different
networks and cannot share weights. They do share seeds, reward,
terrain presets and step budget. The evaluation runs on HELD-OUT
seeds (900_000+) that no training episode ever saw.

CPU, AND THE REASON IS MEASURED
--------------------------------
~13k parameters. Topic 7 measured a policy this size running 1.7x
SLOWER on a GPU, because the rollout batch never fills a card. What
pays is the worker pool.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent


def require_cpu_or_gpu() -> None:
    """
    No `require_gpu()` here, DELIBERATELY, and this comment is the
    justification the contract asks for.

    This lab trains a 13k-parameter MLP against MuJoCo on the CPU,
    measured FASTER than a GPU for exactly this shape of work. A
    guard that can never fire is decoration, and the same cargo-cult
    objection applies as to a distributed launcher with nothing to
    distribute.

    What this lab does need to check is cores, because the whole
    speed story is the worker pool.
    """
    import os

    n = os.cpu_count() or 1
    if n < 2:
        print(f"  note: {n} core visible. This lab's speed comes from a "
              f"process pool;\n        it will still run, serially, and "
              f"take about {16 / max(n, 1):.0f}x longer.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--sense", default="slope",
                    choices=("blind", "slope", "patch"),
                    help="how much of the ground the policy may see. "
                         "This is the ONLY thing the three arms vary.")
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--total-steps", type=int, default=4_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="passed INTO the env too: the shaping "
                         "term is gamma * Phi(s') - Phi(s) and "
                         "the two must agree")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.002)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--eval-every", type=int, default=15,
                    help="iterations between evaluations. Topic 7 measured "
                         "in-training evaluation at ~75%% of wall clock when "
                         "it stayed serial; raising this from 3 to 15 was "
                         "worth 1.6x on its own.")
    ap.add_argument("--eval-episodes", type=int, default=48)
    ap.add_argument("--init-from", default="",
                    help="warm start from another run's policy.pt. Must be "
                         "the SAME sense: each sense has its own obs_dim.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args, _ = ap.parse_known_args()

    require_cpu_or_gpu()

    # Heavy imports AFTER the preflight, per the contract.
    import numpy as np
    import torch

    import thrusters as TH
    from ppo import (ActorCritic, RunningNorm, clipped_policy_loss,
                     compute_gae, hover_bias, pick_device)
    from vecenv import ParallelEnvs

    if args.dry_run:
        args.total_steps = 40_000
        args.n_envs = min(args.n_envs, 4)
        args.eval_every = 2
        args.eval_episodes = 4

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)        # the pool owns the cores, not BLAS

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    env_kw = dict(rig_name=args.rig, preset=args.preset,
                  sense=args.sense, world=args.world, gamma=args.gamma)
    envs = ParallelEnvs(n_workers=args.n_envs, per_worker=1,
                        base_seed=args.seed * 1000, **env_kw)
    obs_dim, act_dim = envs.obs_dim, envs.act_dim
    bias = hover_bias(envs.rig)

    net = ActorCritic(obs_dim, act_dim, init_bias=bias).to(device)
    norm = RunningNorm(obs_dim)

    warm = ""
    if args.init_from:
        ck = torch.load(Path(args.init_from) / "policy.pt",
                        map_location=device, weights_only=False)
        if ck["obs_dim"] != obs_dim or ck["act_dim"] != act_dim:
            raise ValueError(
                f"cannot warm start: {args.init_from} is "
                f"{ck['obs_dim']}x{ck['act_dim']}, this run is "
                f"{obs_dim}x{act_dim}. Each SENSE has its own "
                f"observation width, so a policy cannot cross arms — "
                f"which is exactly what keeps the comparison honest.")
        net.load_state_dict(ck["model"])
        norm.load_state_dict(ck["norm"])
        warm = f" (warm start from {Path(args.init_from).name})"
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    audit = TH.audit(args.rig, envs.rig)
    print("=" * 78)
    print(f"  TRAIN — {args.name}: rig {args.rig} ({act_dim} nozzles), "
          f"sense {args.sense.upper()}, terrain {args.preset}")
    print("=" * 78)
    print(f"  obs {obs_dim}  act {act_dim}  policy {net.n_params():,} params "
          f"on {device}{warm}")
    print(f"  rig   L/W {audit['lift_to_weight']:.2f}, rank {audit['rank']}/6")
    print(f"  pool  {args.n_envs} workers x 1 env, horizon {args.horizon}")
    print(f"  steps {args.total_steps:,}  "
          f"({args.total_steps / (args.n_envs * args.horizon):.0f} iterations)")
    if audit["rank"] < 6:
        print("  WARNING: rank < 6 — some wrench direction is unreachable at "
              "any command.\n           The policy cannot learn around that; "
              "it is a hardware fact.")

    obs = envs.reset(args.seed)
    norm.update(obs)
    history: list[dict] = []
    t_start = time.time()
    step = 0
    it = 0
    recent: list[str] = []

    n_iters = max(1, args.total_steps // (args.n_envs * args.horizon))
    while it < n_iters:
        it += 1
        # --- rollout ------------------------------------------------
        O = np.zeros((args.horizon, args.n_envs, obs_dim), dtype=np.float32)
        A = np.zeros((args.horizon, args.n_envs, act_dim), dtype=np.float32)
        LP = np.zeros((args.horizon, args.n_envs), dtype=np.float32)
        R = np.zeros((args.horizon, args.n_envs), dtype=np.float32)
        D = np.zeros((args.horizon, args.n_envs), dtype=np.float32)
        V = np.zeros((args.horizon, args.n_envs), dtype=np.float32)

        for t in range(args.horizon):
            no = norm(obs).astype(np.float32)
            with torch.no_grad():
                ot = torch.as_tensor(no, device=device)
                a, lp, v = net.act(ot)
            an = a.cpu().numpy()
            O[t], A[t], LP[t], V[t] = no, an, lp.cpu().numpy(), v.cpu().numpy()
            obs, rew, done, info = envs.step(np.clip(an, 0.0, 1.0))
            R[t], D[t] = rew, done
            norm.update(obs)
            recent.extend(x for x in info if x)
            step += args.n_envs

        with torch.no_grad():
            last_v = net.value(torch.as_tensor(norm(obs).astype(np.float32),
                                               device=device))
        adv, ret = compute_gae(torch.as_tensor(R, device=device),
                               torch.as_tensor(V, device=device),
                               torch.as_tensor(D, device=device),
                               last_v, args.gamma, args.lam)

        # --- update -------------------------------------------------
        b_obs = torch.as_tensor(O.reshape(-1, obs_dim), device=device)
        b_act = torch.as_tensor(A.reshape(-1, act_dim), device=device)
        b_lp = torch.as_tensor(LP.reshape(-1), device=device)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        mb = n // args.minibatches
        stats = {}
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for k in range(args.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                lp, ent, v = net.evaluate(b_obs[idx], b_act[idx])
                pl, stats = clipped_policy_loss(lp, b_lp[idx], b_adv[idx],
                                                args.clip)
                vl = ((v - b_ret[idx]) ** 2).mean()
                loss = pl + args.vf_coef * vl - args.ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.max_grad_norm)
                opt.step()

        # --- report -------------------------------------------------
        if it % args.eval_every == 0 or it == n_iters:
            tally: dict[str, int] = {}
            misses: list[float] = []
            for o in recent[-400:]:
                name, _, miss = o.partition("|")
                tally[name] = tally.get(name, 0) + 1
                if miss:
                    misses.append(float(miss))
            tot = max(sum(tally.values()), 1)
            ok = tally.get("on_target", 0) / tot
            crash = tally.get("crash", 0) / tot
            el = time.time() - t_start
            row = {"iter": it, "step": step, "on_target": ok,
                   "crash": crash,
                   "hard": tally.get("landed_hard", 0) / tot,
                   "off": tally.get("landed_off", 0) / tot,
                   "timeout": tally.get("timeout", 0) / tot,
                   "miss": float(np.mean(misses)) if misses else float("nan"),
                   "episodes": tot, "seconds": el,
                   "steps_per_s": step / max(el, 1e-9),
                   "approx_kl": stats.get("approx_kl", 0.0),
                   "clip_fraction": stats.get("clip_fraction", 0.0),
                   "return": float(R.sum(0).mean())}
            history.append(row)
            print(f"  it {it:>4}/{n_iters}  step {step:>9,}  "
                  f"on-target {ok:>5.1%}  miss {row['miss']:>5.2f} m  "
                  f"crash {crash:>5.1%}  "
                  f"ret {row['return']:>7.1f}  "
                  f"{row['steps_per_s']:>6,.0f} st/s  "
                  f"kl {row['approx_kl']:+.4f}", flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))

    envs.close()

    # --- final evaluation on HELD-OUT seeds -------------------------
    from evaluate import evaluate

    torch.save({"model": net.state_dict(), "norm": norm.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim,
                "bias": bias}, out / "policy.pt")
    res = evaluate(out, episodes=args.eval_episodes, seed0=900_000,
                   **env_kw) if not args.dry_run else {}

    summary = {"name": args.name, "rig": args.rig, "preset": args.preset,
               "sense": args.sense,
               "total_steps": step, "n_envs": args.n_envs,
               "seed": args.seed, "device": device,
               "obs_dim": obs_dim, "act_dim": act_dim,
               "params": net.n_params(),
               "wall_seconds": time.time() - t_start,
               "rig_audit": {k: v for k, v in audit.items()
                             if k not in ("names", "throat_mm2")},
               "init_from": args.init_from,
               "world": args.world,
               "eval": res, "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  wrote {out}/policy.pt and summary.json")
    print(f"  wall {summary['wall_seconds'] / 60:.1f} min "
          f"for {step:,} steps "
          f"({step / summary['wall_seconds']:,.0f} steps/s)")
    if res:
        print(f"  HELD-OUT: on target {res['rate']:.0%} "
              f"({res['landed_ok']}/{res['episodes']}), "
              f"miss {res['miss']:.2f} m, sink {res['sink']:.2f} m/s, "
              f"gas {res['gas']:.0f} kg")
    if args.dry_run:
        print("\n  DRY RUN — 40k steps, not a result. A real run is "
              f"{4_000_000:,} steps and takes roughly 50 minutes on 16 "
              "workers.")


if __name__ == "__main__":
    main()
