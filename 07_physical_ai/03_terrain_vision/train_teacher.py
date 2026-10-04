#!/usr/bin/env python3
"""
Train the privileged teacher: one policy, all four terrains.

    uv run train_teacher.py --dry-run
    uv run train_teacher.py --total-steps 800000

The teacher is handed the terrain descriptor directly, so it needs no
rendering and trains at full physics speed. Whether it can solve all four
terrains at all is the question that decides if this lab is viable --
there is no point rendering 40,000 depth frames to imitate a teacher that
falls down every staircase.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=800_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-privileged", action="store_true",
                    help="withhold the terrain descriptor: the BLIND control")
    ap.add_argument("--terrain", default=None,
                    choices=("flat", "up", "down", "hop"),
                    help="train a SPECIALIST on one terrain. One policy "
                         "serving all four scored 62/0/100 on hollow "
                         "ascent across seeds -- too unreliable to distil "
                         "from. Specialists each find their own gait.")
    a, _ = ap.parse_known_args()
    if a.dry_run:
        a.total_steps, a.n_envs, a.rollout = 8192, 4, 128
    run(a)


def run(a) -> None:
    import numpy as np, torch
    from ppo import ActorCritic, RunningNorm, clipped_policy_loss, compute_gae, pick_device
    from vision_env import TerrainWorld, rollout, random_policy
    from terrain import KINDS

    dev = pick_device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = HERE / "runs" / (a.name or f"teacher_s{a.seed}"); out.mkdir(parents=True, exist_ok=True)

    priv = not a.no_privileged
    envs = [TerrainWorld(kind=a.terrain, privileged=priv, seed=a.seed + i)
            for i in range(a.n_envs)]
    od, ad = envs[0].obs_dim, envs[0].act_dim
    net = ActorCritic(od, ad).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    norm = RunningNorm(od)

    if not a.quiet:
        print("=" * 74)
        print(f"  {'TEACHER (privileged)' if priv else 'BLIND (proprioception only)'}   seed {a.seed}   device {dev}")
        print("=" * 74)
        print(f"  obs {od}  act {ad}  policy {net.n_params():,} params")
        print(f"  budget {a.total_steps:,} steps"
              f"{'   (DRY RUN)' if a.dry_run else ''}\n")

    rng = np.random.default_rng(a.seed)
    base_kinds = [a.terrain] if a.terrain else list(KINDS)
    base = [rollout(TerrainWorld(kind=k, seed=a.seed), random_policy(rng), seed=900+i)
            for k in base_kinds for i in range(5)]
    base_ret = float(np.mean([b["return"] for b in base]))
    if not a.quiet:
        print(f"  BASELINE random: return {base_ret:.1f}, "
              f"past event {sum(b['past_event'] for b in base)}/20\n")

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
            an = act.cpu().numpy(); nx, rw, dn = [], [], []
            for i, e in enumerate(envs):
                o2, r, te, tr, _ = e.step(an[i])
                if te or tr: o2, _ = e.reset()
                nx.append(o2); rw.append(r); dn.append(float(te or tr))
            B['o'].append(no); B['a'].append(an); B['l'].append(lp.cpu().numpy())
            B['v'].append(v.cpu().numpy()); B['r'].append(rw); B['d'].append(dn)
            obs = np.stack(nx); norm.update(obs); done += a.n_envs
        with torch.no_grad():
            lv = net.value(torch.as_tensor(norm(obs), dtype=torch.float32, device=dev))
        T = lambda k: torch.as_tensor(np.array(B[k]), dtype=torch.float32, device=dev)
        adv, ret = compute_gae(T('r'), T('v'), T('d'), lv, 0.99, 0.95)
        bo = T('o').reshape(-1, od); ba = T('a').reshape(-1, ad)
        bl = T('l').reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8); n = bo.shape[0]
        for _ in range(a.epochs):
            for idx in torch.randperm(n, device=dev).split(max(1, n // a.minibatches)):
                lp, ent, v = net.evaluate(bo[idx], ba[idx])
                loss, _ = clipped_policy_loss(lp, bl[idx], badv[idx], 0.2)
                loss = loss + 0.5 * ((v - bret[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

        if it % 3 == 0 or done >= a.total_steps:
            ev = evaluate(net, norm, dev, a.seed, priv=priv, only=a.terrain)
            ev.update(step=done, seconds=round(time.time() - t0, 1))
            hist.append(ev)
            if not a.quiet:
                per = "  ".join(f"{k}{ev['past_'+k]:4.0%}" for k in KINDS)
                print(f"  {done:>8,}  return {ev['return']:7.1f}  "
                      f"past {ev['past_all']:5.0%}   {per}   {ev['seconds']:5.0f}s")

    f = hist[-1]
    (out / "summary.json").write_text(json.dumps(
        {"seed": a.seed, "device": dev, "total_steps": done, "terrain": a.terrain,
         "dry_run": a.dry_run, "params": net.n_params(), "obs_dim": od,
         "baseline_return": round(base_ret, 2),
         "wall_seconds": round(time.time() - t0, 1), "final": f}, indent=2))
    with (out / "curve.csv").open("w") as fh:
        cols = list(hist[0]); fh.write(",".join(cols) + "\n")
        for h in hist: fh.write(",".join(str(h[c]) for c in cols) + "\n")
    torch.save({"model": net.state_dict(), "norm": norm.state_dict()}, out / "policy.pt")
    if not a.quiet:
        print(f"\n  BASELINE {base_ret:7.1f}   TRAINED {f['return']:7.1f}")
        print(f"  past the event: " + "  ".join(f"{k} {f['past_'+k]:.0%}" for k in KINDS))
        # Say so when the run was CAPPED. A dry run stops at 8k steps and
        # scores at or below the random baseline, because 8k steps is
        # roughly 1/200th of what the task needs -- that is the expected
        # result, not a regression. This repository has twice shipped a
        # short run whose summary read like a failure, and a reader who
        # cannot tell the two apart will go looking for a bug that is
        # not there.
        if a.total_steps < 200_000:
            print(f"\n  NOTE: this run was CAPPED at {a.total_steps:,} steps "
                  f"({a.total_steps / 1_503_232:.1%} of a real one).")
            print("  Scoring at or below the random baseline here is the")
            print("  EXPECTED outcome, not a failure. The terrains are first")
            print("  cleared at ~200k steps and the curve is still rising at")
            print("  1.5M. For the published numbers:")
            print("      uv run train_teacher.py --name v3_priv_s0")


def evaluate(net, norm, dev, seed, per_kind=8, priv=True, only=None) -> dict:
    import numpy as np, torch
    from vision_env import TerrainWorld, rollout
    from terrain import KINDS

    def pol(o):
        with torch.no_grad():
            t = torch.as_tensor(norm(o), dtype=torch.float32, device=dev).unsqueeze(0)
            return net.distribution(t).mean.squeeze(0).cpu().numpy()

    out, allr = {}, []
    for k in ([only] if only else KINDS):
        env = TerrainWorld(kind=k, seed=seed, privileged=priv)
        rs = [rollout(env, pol, seed=7000 + i) for i in range(per_kind)]
        out[f"past_{k}"] = round(sum(r["past_event"] for r in rs) / len(rs), 3)
        out[f"maxx_{k}"] = round(float(np.mean([r["max_x"] for r in rs])), 3)
        allr += rs
    for k in KINDS:                       # keep the schema stable
        out.setdefault(f"past_{k}", 0.0)
        out.setdefault(f"maxx_{k}", 0.0)
    out["return"] = round(float(np.mean([r["return"] for r in allr])), 2)
    out["past_all"] = round(sum(r["past_event"] for r in allr) / len(allr), 3)
    return out


if __name__ == "__main__":
    main()
