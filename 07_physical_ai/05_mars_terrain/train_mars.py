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
                    choices=("blind", "privileged", "depth", "padded"))
    ap.add_argument("--flat", action="store_true",
                    help="no terrain at all — the locomotion gate")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=3,
                    help="iterations between in-training evaluations. "
                         "Each one costs ~32k SERIAL env steps, which "
                         "once parallel training arrived was 3x the "
                         "wall clock of the training it measured -- "
                         "about 75%% of the run. Raise it when "
                         "--workers is high.")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes to split the environments across. "
                         "1 keeps the original serial loop. The envs "
                         "were stepped in a Python for-loop on one "
                         "core, so a 20-core box ran at 5% of itself.")
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--hidden", type=int, default=64,
                    help="policy width. 64 was sized for lab 4, a task "
                         "with no balance component.")
    ap.add_argument("--upright-w", type=float, default=0.0,
                    help="weight on the posture term, as a fraction of "
                         "the alive bonus")
    ap.add_argument("--relief-scale", type=float, default=1.0,
                    help="terrain curriculum: 0.0 is a plane, 1.0 is "
                         "Mars as generated")
    ap.add_argument("--init-from", default=None, metavar="RUN",
                    help="warm-start from another run's policy. The "
                         "CURRICULUM: learn to balance on flat ground "
                         "first, then carry that into rough terrain.")
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

    from mars_env import NavWorld

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
            effs.append(env.geodesic_len / max(info["travelled"], 1e-6))
    return {"arrived": arrived / episodes, "fell": fell / episodes,
            "to_goal": float(np.mean(lefts)),
            "efficiency": float(np.mean(effs)) if effs else 0.0}


def run(a) -> None:
    import numpy as np
    import torch

    from mars_env import NavWorld, random_policy, rollout
    from ppo import (ActorCritic, RunningNorm, clipped_policy_loss,
                     compute_gae, pick_device)

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = HERE / "runs" / (a.name or f"nav_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)

    # Parallel or serial, decided by --workers. The serial path is
    # kept rather than deleted: `vecenv.py` asserts the two produce
    # bit-identical observations and rewards, and that assertion needs
    # something to compare against.
    vec = None
    if a.workers > 1:
        from vecenv import ParallelEnvs

        per = max(1, a.n_envs // a.workers)
        vec = ParallelEnvs(n_workers=a.workers, per_worker=per,
                           base_seed=a.seed, mode=a.mode, flat=a.flat,
                           relief_scale=a.relief_scale,
                           upright_w=a.upright_w, goal_range=a.goal_range)
        a.n_envs = vec.n
        od, ad = vec.obs_dim, vec.act_dim
        envs = None
    else:
        envs = [NavWorld(mode=a.mode, flat=a.flat,
                         relief_scale=a.relief_scale, upright_w=a.upright_w,
                         seed=a.seed + i, goal_range=a.goal_range)
                for i in range(a.n_envs)]
        od, ad = envs[0].obs_dim, envs[0].act_dim
    net = ActorCritic(od, ad, hidden=a.hidden).to(dev)
    if a.init_from:
        # WARM START -- the curriculum step.
        #
        # Freeing the torso turned this into balance AND navigation at
        # once, and learning both from scratch stalled: across 900k
        # steps on rough ground, arrival stayed at 0%, distance-to-goal
        # moved 17 m -> 13.6 m and stopped, and the fall rate got WORSE
        # (75% -> 83%). The policy was not converging on either skill.
        #
        # Flat ground is the same observation space and a far easier
        # balance problem, so a flat-trained policy is a legal and much
        # better starting point than random weights. The observation
        # width is asserted to match rather than assumed -- a silent
        # shape mismatch here would train from a scrambled prior and
        # look merely like a bad seed.
        import json as _json
        src = HERE / "runs" / a.init_from
        meta = _json.loads((src / "summary.json").read_text())
        if meta.get("hidden", 64) != a.hidden:
            raise SystemExit(
                f"\n  --init-from {a.init_from} was trained at hidden="
                f"{meta.get('hidden', 64)}, this run asks for "
                f"{a.hidden}.\n  A wider policy cannot inherit a "
                f"narrower one's weights; train its own flat stage "
                f"first.\n")
        if meta["obs_dim"] != od:
            raise SystemExit(
                f"\n  --init-from {a.init_from} has obs_dim "
                f"{meta['obs_dim']}, this run needs {od}.\n"
                f"  Refusing to warm-start from a different observation "
                f"space.\n")
        ck = torch.load(src / "policy.pt", weights_only=False)
        net.load_state_dict(ck["model"])
        norm_init = ck["norm"]
        print(f"  warm-started from {a.init_from} "
              f"(obs_dim {meta['obs_dim']}, {meta['total_steps']:,} steps)")
    else:
        norm_init = None
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    norm = RunningNorm(od)
    if norm_init is not None:
        # Carry the observation statistics too. Starting a warm policy
        # behind a cold normaliser feeds it inputs on a different scale
        # than it was trained on, which undoes the warm start.
        norm.load_state_dict(norm_init)

    if not a.quiet:
        print("=" * 74)
        print(f"  {a.mode.upper()}{'  (FLAT — locomotion gate)' if a.flat else ''}"
              f"   seed {a.seed}   device {dev}")
        print("=" * 74)
        print(f"  obs {od}  act {ad}  policy {net.n_params():,} params")
        print(f"  budget {a.total_steps:,} steps"
              f"{'   (DRY RUN)' if a.dry_run else ''}\n")

    rng = np.random.default_rng(a.seed)
    base_env = NavWorld(mode=a.mode, flat=a.flat,
                        relief_scale=a.relief_scale,
                        upright_w=a.upright_w, seed=a.seed)
    base = [rollout(base_env, random_policy(rng, base_env.act_dim),
                    seed=900 + i) for i in range(5)]
    base_ret = float(np.mean([b["return"] for b in base]))
    if not a.quiet:
        print(f"  BASELINE random: return {base_ret:.1f}, "
              f"arrived {sum(b['arrived'] for b in base)}/5\n")

    obs = (vec.reset(seed=a.seed) if vec is not None
           else np.stack([e.reset(seed=a.seed + i)[0]
                          for i, e in enumerate(envs)]))
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
            if vec is not None:
                nx, rw, dn = vec.step(an)
            else:
                nx_, rw, dn = [], [], []
                for i, e in enumerate(envs):
                    o2, r, te, tr, _ = e.step(an[i])
                    if te or tr:
                        o2, _ = e.reset()
                    nx_.append(o2); rw.append(r); dn.append(float(te or tr))
                nx = np.stack(nx_)
            B['o'].append(no); B['a'].append(an)
            B['l'].append(lp.cpu().numpy()); B['v'].append(v.cpu().numpy())
            B['r'].append(rw); B['d'].append(dn)
            obs = nx; norm.update(obs); done += a.n_envs
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

        if it % a.eval_every == 0 or done >= a.total_steps:
            ev = evaluate(net, norm, dev, a.mode, a.flat,
                          a.eval_episodes, a.seed, a.goal_range)
            ev.update(step=done, seconds=round(time.time() - t0, 1))
            hist.append(ev)
            if not a.quiet:
                print(f"  {done:>9,}  arrived {ev['arrived']:5.0%}  "
                      f"fell {ev['fell']:5.0%}  left {ev['to_goal']:5.1f} m  "
                      f"eff {ev['efficiency']:4.0%}  {ev['seconds']:5.0f}s",
                      flush=True)

            # CHECKPOINT AS WE GO, not only at the end.
            #
            # A run used to write `policy.pt` once, after the final
            # step. Six large-world runs were killed at 94% of their
            # budget when the session holding them ended, and all of
            # it -- four and a half hours of training that had already
            # reached 42% arrival -- was unrecoverable, because nothing
            # had been written yet. The logs survived and the weights
            # did not, which is exactly backwards.
            #
            # `summary.json` is still written only on completion, so it
            # stays the marker of a FINISHED run and nothing downstream
            # mistakes a partial checkpoint for a result.
            torch.save({"model": net.state_dict(),
                        "norm": norm.state_dict()}, out / "policy.pt")
            (out / "progress.json").write_text(json.dumps(
                {"step": done, "total": a.total_steps, "latest": ev,
                 "mode": a.mode, "seed": a.seed,
                 "goal_range": a.goal_range,
         "workers": a.workers, "n_envs": a.n_envs,
         "eval_every": a.eval_every,
         "relief_scale": a.relief_scale,
         "hidden": a.hidden,
         "upright_w": a.upright_w,
                 "world": "mars-rover",
                 "complete": False}, indent=2))

    f = hist[-1]
    (out / "summary.json").write_text(json.dumps(
        # `goal_range` is part of the TASK, not a training detail, and
        # a run that does not record it cannot be scored or filmed
        # afterwards. It was omitted, so `render.py` built its
        # environment without one and every animation filmed goals at
        # the map's own endpoints -- a harder task than any policy here
        # was trained on, which is why the clips showed the robot
        # stranded 14 m out while the evaluator reported 42% arrival.
        # The figure and the number were describing different worlds.
        {"mode": a.mode, "flat": a.flat, "seed": a.seed, "device": dev,
         "goal_range": a.goal_range,
         "workers": a.workers, "n_envs": a.n_envs,
         "eval_every": a.eval_every,
         "relief_scale": a.relief_scale,
         "hidden": a.hidden,
         "upright_w": a.upright_w,
         # Which WORLD this policy was trained in. Same reasoning as
         # goal_range: a run that does not record its task cannot be
         # scored or filmed afterwards without guessing.
         "world": "mars-rover",
         "arena_m": __import__("mars").preset("rover")["extent"],
         "max_steps": __import__("mars_env").MAX_STEPS,
         "obs_dim": od, "act_dim": ad, "params": net.n_params(),
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
