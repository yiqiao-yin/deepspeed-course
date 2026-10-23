#!/usr/bin/env python3
"""
Roll out the vehicle and record what the physics did.

    uv run collect.py --episodes 400 --out data/train.npz
    uv run collect.py --episodes 100 --seed0 500000 --out data/val.npz

WHY THE DATA IS MIXED
---------------------
A dynamics model learned only from a good policy's trajectories sees
only the states a good policy visits -- upright, descending, valves
near hover. It will then be asked to predict a tumble it has never
seen, which is exactly the regime a planner would want it for.

So the mix is deliberate: trained policies for the realistic states,
a scripted controller for the competent-but-different ones, and a
random policy for the violent ones nothing else reaches. The share
of each is recorded in the file, because a prediction error quoted
without the distribution it was measured on is not a number.

WHAT IS RECORDED
----------------
Per step: the 41-number observation, the 10 valve commands, and the
NEXT observation. Episodes are stored whole, with their boundaries,
because the lab's metric is multi-step rollout error and that needs
contiguous sequences rather than shuffled transitions.

The fault is recorded too. It is not an input to the model -- the
model sees what the policy saw -- but it lets the error be broken
down by whether the vehicle was broken at the time, which turns out
to matter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def require_cpu_or_gpu() -> None:
    """
    No `require_gpu()` here, deliberately: collection is MuJoCo on
    the CPU and a guard that can never fire is decoration. Training
    the dynamics model is the part that wants a card, and that is
    `train_dyn.py`.
    """
    import os

    n = os.cpu_count() or 1
    if n < 2:
        print(f"  note: {n} core visible; collection is serial MuJoCo "
              f"and will take about {16 / max(n, 1):.0f}x longer.")


def policy_sources(rig: str) -> list[tuple[str, str]]:
    """
    (label, path) for every trained policy that fits this vehicle.

    Searched rather than hardcoded: the earlier labs' run directories
    are gitignored, so a fresh clone has none of them and the mix
    falls back to scripted and random. That is a weaker dataset and
    the header says so rather than pretending otherwise.
    """
    out = []
    root = HERE.parent
    for lab in ("05_thruster_failure", "01_aerial_stabilization"):
        for p in sorted((root / lab).glob("runs/*/policy.pt")):
            out.append((f"{lab.split('_')[0]}:{p.parent.name}", str(p.parent)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rig", default="reveal")
    ap.add_argument("--preset", default="rolling")
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--fail-p", type=float, default=0.5)
    ap.add_argument("--out", default="data/train.npz")
    a, _ = ap.parse_known_args()

    require_cpu_or_gpu()

    import torch

    from fail_env import FailEnv, MAX_STEPS

    env = FailEnv(a.rig, a.preset, world=a.world, fail_p=a.fail_p)
    srcs = policy_sources(a.rig)
    nets = []
    from evaluate import load_policy

    skipped = []
    for label, rd in srcs:
        # The obs width is read from the CHECKPOINT, not guessed from
        # the module tree. An earlier version poked at `net.net[0]`,
        # which does not exist on ActorCritic, so every policy raised
        # and a bare `except: continue` turned that into "0 trained
        # policies usable" -- a silently weaker dataset with no error
        # anywhere. Reasons are collected and printed instead.
        try:
            import torch as _t

            ck = _t.load(Path(rd) / "policy.pt", map_location="cpu",
                         weights_only=False)
            if ck["obs_dim"] != env.obs_dim or ck["act_dim"] != env.act_dim:
                skipped.append(f"{label}: {ck['obs_dim']}x{ck['act_dim']}")
                continue
            net, norm = load_policy(rd)
            nets.append((label, net, norm))
        except Exception as exc:
            skipped.append(f"{label}: {type(exc).__name__} {exc}")

    print("=" * 78)
    print("  COLLECT — rollouts for the dynamics model")
    print("=" * 78)
    print(f"  rig {a.rig}  world {a.world}  obs {env.obs_dim}  "
          f"act {env.act_dim}  fail_p {a.fail_p}")
    print(f"  {len(nets)} trained policies usable"
          + (f": {', '.join(n for n, _, _ in nets[:4])}"
             f"{' ...' if len(nets) > 4 else ''}" if nets else
             "  (none found — the mix falls back to scripted + random,\n"
             "   which is a WEAKER dataset; earlier labs' runs/ are "
             "gitignored)"))

    rng = np.random.default_rng(a.seed0)
    OBS, ACT, NXT, EPS, FAULTY = [], [], [], [], []
    kinds = {"trained": 0, "scripted": 0, "random": 0}

    for ep in range(a.episodes):
        # The mix. Random gets a full third on purpose: it is the only
        # source of the violent states, and a model that has never
        # seen one cannot warn a planner away from it.
        r = rng.uniform()
        if nets and r < 0.45:
            kind = "trained"
            label, net, norm = nets[int(rng.integers(0, len(nets)))]
        elif r < 0.67:
            kind = "scripted"
        else:
            kind = "random"
        kinds[kind] += 1

        o = env.reset(a.seed0 + ep)
        n0 = len(OBS)
        hover = env.hover_action()
        while True:
            if kind == "trained":
                with torch.no_grad():
                    x = torch.as_tensor(norm(o), dtype=torch.float32
                                        ).unsqueeze(0)
                    u = net.distribution(x).mean.squeeze(0).numpy()
            elif kind == "scripted":
                # Hover command plus a slow wander: competent, and
                # visiting states no trained policy bothers with.
                u = hover + 0.25 * rng.normal(size=env.act_dim)
            else:
                u = rng.uniform(0.0, 1.0, env.act_dim)
            u = np.clip(u, 0.0, 1.0)

            o2, _r, done, _ = env.step(u)
            OBS.append(o)
            ACT.append(u)
            NXT.append(o2)
            FAULTY.append(1.0 if env.fault_active else 0.0)
            o = o2
            if done:
                break
        EPS.append((n0, len(OBS)))
        if (ep + 1) % 50 == 0:
            print(f"    {ep + 1}/{a.episodes} episodes, "
                  f"{len(OBS):,} transitions", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.asarray(OBS, dtype=np.float32),
        act=np.asarray(ACT, dtype=np.float32),
        nxt=np.asarray(NXT, dtype=np.float32),
        faulty=np.asarray(FAULTY, dtype=np.float32),
        ep_start=np.asarray([s for s, _ in EPS], dtype=np.int64),
        ep_end=np.asarray([e for _, e in EPS], dtype=np.int64),
        # The distribution travels WITH the data. An error quoted
        # without it is not a number.
        mix=np.asarray([kinds["trained"], kinds["scripted"],
                        kinds["random"]], dtype=np.int64),
        rig=a.rig, world=a.world, fail_p=a.fail_p, seed0=a.seed0,
        max_steps=MAX_STEPS)

    n = len(OBS)
    print(f"\n  wrote {out}  —  {n:,} transitions over {len(EPS)} episodes")
    print(f"  mix: {kinds['trained']} trained, {kinds['scripted']} scripted, "
          f"{kinds['random']} random")
    print(f"  {np.mean(FAULTY):.0%} of transitions had a broken nozzle")
    print(f"  mean episode length {n / len(EPS):.0f} steps "
          f"of a {MAX_STEPS} step clock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
