#!/usr/bin/env python3
"""
Train a navigating biped by PPO.

    uv run train_nav.py --flat --name flat_s0      # the gate: can it walk to B?
    uv run train_nav.py --mode privileged --name priv_s0
    uv run train_nav.py --mode blind      --name blind_s0
    uv run train_nav.py --dry-run                  # 30 s, proves it assembles

THREE ARMS, AND WHY THE ORDER MATTERS
-------------------------------------
`--flat` comes first and is not part of the experiment. It removes the
terrain entirely and asks only: can this body walk to a point and turn
to face it? If that fails, nothing measured on rough ground means
anything, because the failure would be locomotion rather than
navigation. Lab 3 spent four designs discovering too late that a task
was broken rather than hard; this one checks first.

Then the arms that are the experiment:

    blind        proprioception and the goal bearing. Must collide with
                 a ridge to learn it is there.
    privileged   plus an 11x11 patch of ground height ahead. An oracle.
    depth        plus a 64x64 depth image -- what a real robot has.

WHAT IS MEASURED
----------------
Two numbers, because reaching B is not the whole story:

    arrival      did it get within 0.8 m of B
    efficiency   the ORACLE's route length over the distance actually
                 walked. A robot that reaches B having wandered three
                 times as far has solved the task in a sense nobody
                 wants, and only this number says so.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-steps", type=int, default=1_500_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", default="blind",
                    choices=("blind", "privileged", "depth"))
    ap.add_argument("--flat", action="store_true",
                    help="no terrain at all — the locomotion gate")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--goal-range", type=float, default=None,
                    help="cap how far B is placed, in metres. The arrival "
                         "reward is unreachable early on a 15 m goal, so the "
                         "policy never sees it; a nearer goal gives it "
                         "something to climb toward.")
    ap.add_argument("--eval-episodes", type=int, default=12)
    a, _ = ap.parse_known_args()
    if a.dry_run:
        a.total_steps, a.n_envs, a.rollout = 8192, 4, 128
    run(a)


def evaluate(net, norm, dev, mode, flat, episodes, seed,
             goal_range=None) -> dict:
    import numpy as np
    import torch

    from nav_env import NavWorld

    env = NavWorld(mode=mode, flat=flat, seed=seed,
                   goal_range=goal_range)
    arrived = fell = 0
    effs, lefts = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=5000 + i)
        while True:
            with torch.no_grad():
                t = torch.as_tensor(norm(obs), dtype=torch.float32,
                                    device=dev).unsqueeze(0)
                act = net.distribution(t).mean.squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        arrived += bool(info["arrived"])
        fell += bool(info["fell"])
        lefts.append(info["to_goal"])
        if info["arrived"] and info["travelled"] > 0.1:
            effs.append(min(info["route"] / info["travelled"], 1.0))
    return {"arrived": arrived / episodes, "fell": fell / episodes,
            "to_goal": float(np.mean(lefts)),
            "efficiency": float(np.mean(effs)) if effs else 0.0}


def run(a) -> None:
    import numpy as np
    import torch

    from nav_env import NavWorld, random_policy, rollout
    from ppo import (ActorCritic, RunningNorm, clipped_policy_loss,
                     compute_gae, pick_device)

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = HERE / "runs" / (a.name or f"nav_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)

    envs = [NavWorld(mode=a.mode, flat=a.flat, seed=a.seed + i,
                     goal_range=a.goal_range)
            for i in range(a.n_envs)]
    od, ad = envs[0].obs_dim, envs[0].act_dim
    net = ActorCritic(od, ad).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    norm = RunningNorm(od)

    if not a.quiet:
        print("=" * 74)
        print(f"  {a.mode.upper()}{'  (FLAT — locomotion gate)' if a.flat else ''}"
              f"   seed {a.seed}   device {dev}")
        print("=" * 74)
        print(f"  obs {od}  act {ad}  policy {net.n_params():,} params")
        print(f"  budget {a.total_steps:,} steps"
              f"{'   (DRY RUN)' if a.dry_run else ''}\n")

    rng = np.random.default_rng(a.seed)
    base = [rollout(NavWorld(mode=a.mode, flat=a.flat, seed=a.seed),
                    random_policy(rng), seed=900 + i) for i in range(5)]
    base_ret = float(np.mean([b["return"] for b in base]))
    if not a.quiet:
        print(f"  BASELINE random: return {base_ret:.1f}, "
              f"arrived {sum(b['arrived'] for b in base)}/5\n")

    obs = np.stack([e.reset(seed=a.seed + i)[0] for i, e in enumerate(envs)])
    norm.update(obs)
    hist, done, it, t0 = [], 0, 0, time.time()
    while done < a.total_steps:
        it += 1
        B = {k: [] for k in "oalrdv"}
        for _ in range(a.rollout):
            no = norm(obs)
            with torch.no_grad():
                t = torch.as_tensor(no, dtype=torch.float32, device=dev)
                act, lp, v = net.act(t)
            an = act.cpu().numpy()
            nx, rw, dn = [], [], []
            for i, e in enumerate(envs):
                o2, r, te, tr, _ = e.step(an[i])
                if te or tr:
                    o2, _ = e.reset()
                nx.append(o2); rw.append(r); dn.append(float(te or tr))
            B['o'].append(no); B['a'].append(an)
            B['l'].append(lp.cpu().numpy()); B['v'].append(v.cpu().numpy())
            B['r'].append(rw); B['d'].append(dn)
            obs = np.stack(nx); norm.update(obs); done += a.n_envs
        with torch.no_grad():
            lv = net.value(torch.as_tensor(norm(obs), dtype=torch.float32,
                                           device=dev))
        T = lambda k: torch.as_tensor(np.array(B[k]), dtype=torch.float32,
                                      device=dev)
        adv, ret = compute_gae(T('r'), T('v'), T('d'), lv, 0.99, 0.95)
        bo = T('o').reshape(-1, od); ba = T('a').reshape(-1, ad)
        bl = T('l').reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        n = bo.shape[0]
        for _ in range(a.epochs):
            for idx in torch.randperm(n, device=dev).split(
                    max(1, n // a.minibatches)):
                lp, ent, v = net.evaluate(bo[idx], ba[idx])
                loss, _ = clipped_policy_loss(lp, bl[idx], badv[idx], 0.2)
                loss = loss + 0.5 * ((v - bret[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if it % 3 == 0 or done >= a.total_steps:
            ev = evaluate(net, norm, dev, a.mode, a.flat,
                          a.eval_episodes, a.seed, a.goal_range)
            ev.update(step=done, seconds=round(time.time() - t0, 1))
            hist.append(ev)
            if not a.quiet:
                print(f"  {done:>9,}  arrived {ev['arrived']:5.0%}  "
                      f"fell {ev['fell']:5.0%}  left {ev['to_goal']:5.1f} m  "
                      f"eff {ev['efficiency']:4.0%}  {ev['seconds']:5.0f}s")

    f = hist[-1]
    (out / "summary.json").write_text(json.dumps(
        {"mode": a.mode, "flat": a.flat, "seed": a.seed, "device": dev,
         "obs_dim": od, "params": net.n_params(),
         "total_steps": a.total_steps, "baseline_return": round(base_ret, 2),
         "wall_seconds": round(time.time() - t0, 1), "final": f}, indent=2))
    with (out / "curve.csv").open("w") as fh:
        cols = list(hist[0]); fh.write(",".join(cols) + "\n")
        for h in hist:
            fh.write(",".join(str(h[c]) for c in cols) + "\n")
    torch.save({"model": net.state_dict(), "norm": norm.state_dict(),
                "obs_dim": od, "mode": a.mode, "flat": a.flat},
               out / "policy.pt")

    if not a.quiet:
        print(f"\n  FINAL  arrived {f['arrived']:.0%}  "
              f"efficiency {f['efficiency']:.0%}  "
              f"mean distance left {f['to_goal']:.1f} m")
        if a.total_steps < 200_000:
            print(f"\n  NOTE: CAPPED at {a.total_steps:,} steps "
                  f"({a.total_steps/1_500_000:.1%} of a real run). Scoring at")
            print("  or near the random baseline here is EXPECTED, not a")
            print("  failure. For the published numbers drop --dry-run.")


if __name__ == "__main__":
    main()
