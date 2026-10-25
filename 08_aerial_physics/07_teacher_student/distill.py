#!/usr/bin/env python3
"""
Teach the student what the teacher knows, using only what the student can see.

    uv run distill.py --teacher runs/oracle_s0 --method bc     --name bc_s0
    uv run distill.py --teacher runs/oracle_s0 --method dagger --name dagger_s0
    uv run distill.py --teacher runs/oracle_s0 --method dagger --deepspeed

WHAT THIS SCRIPT IS FOR
------------------------
`train_policy.py` trains three arms by PPO, each from a scalar
reward. This one trains two more by SUPERVISED REGRESSION against a
teacher that has already solved the task with information no
vehicle could carry.

    bc        behaviour cloning. Fly the TEACHER, record what it saw
              through the student's sensor and what it did, fit the
              student to that. One pass, no interaction.
    dagger    the same, iterated, with the STUDENT flying and the
              teacher relabelling the states the student actually
              reaches.

THE WHOLE DIFFERENCE IS WHOSE STATE DISTRIBUTION YOU FIT ON
------------------------------------------------------------
Behaviour cloning minimises

    E over states drawn from the TEACHER's trajectories
        || student(o(s)) - teacher(s) ||^2

and is then deployed on states drawn from the STUDENT's
trajectories. Those are not the same distribution, and the
mismatch is self-amplifying: a small error moves the vehicle
somewhere the teacher never went, where the student was never
trained, where its error is larger, which moves it further. Ross
and Bagnell's result is that this compounds QUADRATICALLY in the
horizon -- regret O(eps * T^2) against O(eps * T) for a method
that trains on its own distribution.

This course has already measured that mathematics once, in a
different costume. Lab 6 fed a dynamics model its own predictions
and watched the error grow with the rollout length rather than
staying at the one-step value. Same phenomenon, same cause: a
model evaluated on the distribution its own mistakes create.
T here is 600 steps, so the gap between T and T^2 is not subtle.

DAgger closes it by making the training distribution the student's
own. Round i flies the student, asks the teacher what IT would
have done at each state visited -- which requires the teacher's
privileged view of a state the teacher did not choose, and is
exactly what `env.obs_for("oracle")` exists for -- and appends to
the dataset. The student is refit on the union.

THE MIXING SCHEDULE
-------------------
Round 0 is driven entirely by the teacher, so it IS behaviour
cloning. Later rounds hand over: with probability beta_i the
teacher drives, otherwise the student does, and beta decays
geometrically. Labels always come from the teacher regardless of
who is driving -- it is the DRIVER that changes, never the
labeller.

    beta_i = decay ^ i

THE COMPARISON IS BUDGET-MATCHED, AND THAT MATTERS
---------------------------------------------------
BC and DAgger are given the SAME number of labelled transitions
and the SAME number of gradient steps. Otherwise "DAgger wins"
would be indistinguishable from "DAgger saw more data", which is
the kind of uncontrolled comparison this repository has published
before and had to withdraw. BC simply spends its whole budget on
round 0.

WHY THERE IS A DEEPSPEED CONFIG HERE AND NOT IN THE PPO SCRIPT
---------------------------------------------------------------
`train_policy.py` deliberately runs plain python: a 33k-parameter
policy has nothing for ZeRO to shard, and topic 7 measured a
policy that size running 1.7x SLOWER on a GPU than on the CPU.
Using a distributed launcher there would be cargo cult.

Distillation is a different shape of problem. It is supervised,
the dataset is large and fixed, the batches are dense, and the
student can be made as big as you like -- which is the first
moment in topic 8 where the question "should this be sharded?"
has a real answer rather than a rhetorical one. `--deepspeed`
takes that path and `--size` scales the student so the memory
table has something to say. The honest result, whichever way it
falls, is reported by `compare.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent

# Student widths. The small one MATCHES the PPO arms exactly --
# 64x64 tanh -- because the headline comparison is between methods
# and a bigger student would confound it with capacity. The larger
# rungs exist so the memory question can be asked at all.
SIZES: dict[str, dict] = {
    "match":  dict(hidden=64,   layers=2),    # identical to the PPO arms
    "wide":   dict(hidden=512,  layers=3),
    "deep":   dict(hidden=1024, layers=6),
    "xl":     dict(hidden=4096, layers=8),
}


CPU_OK = ("match", "wide")


def require_cpu_or_gpu(size: str) -> None:
    """
    A SIZE-AWARE guard, and the size-awareness is the point.

    At `--size match` the student is ~17k parameters and the work is
    MuJoCo rollouts with a torch forward per step -- CPU-bound, and
    measured faster on a CPU than a GPU for exactly this shape. A
    guard that fired there would make the lab's headline arm
    unrunnable on a CPU-only box while Clawdeck still advertised it
    as "Runs now, no GPU needed", which is a defect this repo has
    shipped six times. So the guard asks which student was asked
    for and stops only the ones that really cannot run.

    IT ASKS TORCH, NOT THE FILESYSTEM. The first version tested
    `shutil.which("nvidia-smi")`, which is present on any box that
    has ever had a driver installed -- including one where the
    contract's own Reader A test, `CUDA_VISIBLE_DEVICES=""`, has
    hidden every card. That version walked straight past the guard
    and started a six-hour run. `torch.cuda.is_available()` is the
    question actually being asked.

    ALLOW_CPU=1 overrides it, per the contract.
    """
    import os

    if size in CPU_OK or os.environ.get("ALLOW_CPU") == "1":
        n = os.cpu_count() or 1
        if n < 2:
            print(f"  note: {n} core visible. Rollout collection is "
                  f"serial per round; it will still run.")
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    print("=" * 74)
    print("  NO GPU VISIBLE")
    print("=" * 74)
    print(f"  --size {size} builds a student large enough to want a card,")
    print("  and that is the only arm in this lab that does.")
    print()
    print("  What you can run right now, on this machine:")
    print("      uv run distill.py --method dagger --size match")
    print("      uv run rangefinder.py --scene --cost")
    print("      uv run teach_env.py --gap")
    print("      uv run compare.py")
    print()
    print("  `match` is the HEADLINE arm, not a consolation prize: it is")
    print("  the same 64x64 network the PPO arms use, which is what makes")
    print("  the method comparison a comparison.")
    print()
    print("  To rent a card:")
    print("      uv run ../../runpod/runpod_ctl.py run \\")
    print("          08_aerial_physics/07_teacher_student --wait --terminate")
    print()
    print("  Or set ALLOW_CPU=1 to try it anyway.")
    print("=" * 74)
    raise SystemExit(2)


def build_student(obs_dim: int, act_dim: int, size: str, bias=None):
    """
    The student network.

    At `--size match` this is the SAME architecture as the PPO
    arms, deliberately: two 64-wide tanh layers and a linear head,
    initialised at hover. Only the input width differs, because the
    student reads 144 beams where the teacher reads 25 heights.
    Anything else and the comparison would be measuring capacity.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    if size not in SIZES:
        raise ValueError(f"unknown --size {size!r}; have {sorted(SIZES)}")
    spec = SIZES[size]
    h, n_layers = spec["hidden"], spec["layers"]

    layers: list[nn.Module] = []
    d = obs_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    head = nn.Linear(d, act_dim)
    net = nn.Sequential(*layers, head)

    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(head.weight, 0.01)
    if bias is not None:
        with torch.no_grad():
            head.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return net


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="runs/oracle_s0",
                    help="a run directory from train_policy.py --sense oracle")
    ap.add_argument("--method", default="dagger", choices=("bc", "dagger"))
    ap.add_argument("--name", default="")
    ap.add_argument("--size", default="match", choices=sorted(SIZES))
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--world", default="earth", choices=("earth", "mars"))
    ap.add_argument("--iters", type=int, default=10,
                    help="DAgger rounds. --method bc forces 1.")
    ap.add_argument("--episodes", type=int, default=600,
                    help="TOTAL labelled episodes, split across rounds. "
                         "Identical for bc and dagger -- the comparison "
                         "is budget-matched on purpose.")
    ap.add_argument("--decay", type=float, default=0.55,
                    help="beta_i = decay ** i, the chance the TEACHER drives")
    ap.add_argument("--epochs", type=int, default=12,
                    help="passes over the dataset per round")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=120,
                    help="held-out episodes for the FINAL score")
    ap.add_argument("--round-eval", type=int, default=40,
                    help="episodes for the per-round progress read. Small "
                         "on purpose: it is a curve, not a result, and at "
                         "10 rounds it would otherwise cost more than the "
                         "training it is reporting on.")
    ap.add_argument("--deepspeed", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args, _ = ap.parse_known_args()

    require_cpu_or_gpu(args.size)

    # Heavy imports AFTER the preflight, per the contract.
    import numpy as np
    import torch

    import thrusters as TH
    from ppo import ActorCritic, RunningNorm, hover_bias, pick_device
    from teach_env import TeachEnv

    if args.dry_run:
        args.episodes = 24
        args.iters = min(args.iters, 3)
        args.epochs = 2
        args.eval_episodes = 8
        args.round_eval = 4

    name = args.name or f"{args.method}_s{args.seed}"
    if args.method == "bc":
        # Behaviour cloning IS DAgger's round 0 with the teacher
        # driving throughout. Forcing one round here rather than
        # writing a second code path means the two arms cannot
        # drift apart in some detail nobody is watching.
        args.iters = 1
        args.decay = 1.0

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) // 2))

    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    # --- the teacher ------------------------------------------------
    tdir = Path(args.teacher)
    ck = torch.load(tdir / "policy.pt", map_location=device,
                    weights_only=False)
    teacher = ActorCritic(ck["obs_dim"], ck["act_dim"],
                          init_bias=ck.get("bias")).to(device)
    teacher.load_state_dict(ck["model"])
    teacher.eval()
    tnorm = RunningNorm(ck["obs_dim"])
    tnorm.load_state_dict(ck["norm"])

    tsum = json.loads((tdir / "summary.json").read_text())
    if tsum["sense"] != "oracle":
        raise ValueError(
            f"{tdir} was trained with sense={tsum['sense']!r}, not "
            f"'oracle'. Distilling from a teacher that only had the "
            f"student's own sensor would teach nothing the student "
            f"could not have learned alone, and the lab's claim "
            f"would be vacuous.")
    t_rate = tsum.get("eval", {}).get("rate")

    env_kw = dict(rig_name=args.rig, preset=args.preset, world=args.world)
    probe = TeachEnv(sense="onboard", seed=args.seed, **env_kw)
    s_obs_dim, act_dim = probe.obs_dim, probe.act_dim
    if probe.obs_for("oracle").shape[0] != ck["obs_dim"]:
        raise ValueError(
            f"teacher expects {ck['obs_dim']} inputs but this rig's "
            f"oracle view is {probe.obs_for('oracle').shape[0]} wide — "
            f"the teacher was trained on a different rig or preset.")
    bias = hover_bias(probe.rig)

    student = build_student(s_obs_dim, act_dim, args.size, bias).to(device)
    snorm = RunningNorm(s_obs_dim)
    n_params = sum(p.numel() for p in student.parameters())

    print("=" * 78)
    print(f"  DISTILL — {name}: {args.method.upper()} from "
          f"{tdir.name}, student size {args.size}")
    print("=" * 78)
    print(f"  teacher  {ck['obs_dim']:>4} in, {teacher.n_params():,} params"
          + (f", held-out on-target {t_rate:.0%}" if t_rate is not None else ""))
    print(f"  student  {s_obs_dim:>4} in, {n_params:,} params on {device}")
    print(f"  budget   {args.episodes} episodes over {args.iters} round(s), "
          f"{args.epochs} epochs each")
    print(f"  beta     teacher drives with p = {args.decay}^i"
          + ("  (BC: the teacher drives throughout)"
             if args.method == "bc" else ""))

    engine = None
    if args.deepspeed:
        import os

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29518")
        import deepspeed

        ds_cfg = json.loads((HERE / "ds_config.json").read_text())
        ds_cfg["train_micro_batch_size_per_gpu"] = args.batch
        # A TORCH optimizer is supplied rather than named in the
        # config. DeepSpeed's fused Adam is JIT-compiled and needs a
        # CUDA toolkit (`nvcc`); the PyPI wheels ship only `ptxas`,
        # so naming it in the JSON fails at initialize() on a box
        # that trains perfectly well. Lab 6 hit this first.
        torch_opt = torch.optim.Adam(student.parameters(), lr=args.lr)
        engine, opt, _, _ = deepspeed.initialize(
            model=student, optimizer=torch_opt, config=ds_cfg)
        print(f"  deepspeed ZeRO stage "
              f"{ds_cfg['zero_optimization']['stage']}, "
              f"micro-batch {args.batch}")
    else:
        opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    # --- the rounds -------------------------------------------------
    from vecenv import ParallelEnvs

    per_round = max(1, args.episodes // args.iters)
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    history: list[dict] = []
    t0 = time.time()

    for i in range(args.iters):
        beta = args.decay ** i
        sx, sy, drove = _collect(
            teacher, tnorm, student, snorm, env_kw, args.seed,
            per_round, beta, device, args.n_envs, engine)
        X.append(sx)
        Y.append(sy)
        bigX = np.concatenate(X)
        bigY = np.concatenate(Y)
        snorm.update(sx)

        loss = _fit(student, snorm, bigX, bigY, args.epochs, args.batch,
                    device, opt, engine)

        res = _rollout_eval(student, snorm, env_kw, args.round_eval,
                            device, engine)
        row = {"round": i, "beta": beta, "pairs": int(len(bigX)),
               "new_pairs": int(len(sx)), "teacher_drove": drove,
               "loss": loss, "on_target": res["rate"],
               "miss": res["miss"], "crash": res["crash"],
               "seconds": time.time() - t0}
        history.append(row)
        print(f"  round {i:>2}/{args.iters}  beta {beta:>5.3f}  "
              f"pairs {len(bigX):>8,}  loss {loss:.5f}  "
              f"on-target {res['rate']:>5.1%}  miss {res['miss']:>5.2f} m  "
              f"crash {res['crash']:>5.1%}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))

    final = _rollout_eval(student, snorm, env_kw, args.eval_episodes,
                          device, engine, seed0=900_000)
    sd = (engine.module if engine is not None else student).state_dict()
    torch.save({"model": sd, "norm": snorm.state_dict(),
                "obs_dim": s_obs_dim, "act_dim": act_dim,
                "size": args.size, "bias": bias,
                "arch": "student_mlp"}, out / "policy.pt")

    summary = {"name": name, "method": args.method, "size": args.size,
               "teacher": str(tdir), "teacher_rate": t_rate,
               "rig": args.rig, "preset": args.preset, "world": args.world,
               "seed": args.seed, "device": device, "deepspeed": args.deepspeed,
               "obs_dim": s_obs_dim, "act_dim": act_dim, "params": n_params,
               "episodes": args.episodes, "iters": args.iters,
               "epochs": args.epochs, "decay": args.decay,
               "wall_seconds": time.time() - t0,
               "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9
                               if device == "cuda" else 0.0),
               "eval": final, "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "episodes.json").write_text(json.dumps(
        {"seeds": final["seeds"], "on_target": final["on_target"],
         "outcome": final["outcome"], "miss": final["per_miss"]}))
    print(f"\n  wrote {out}/policy.pt, summary.json and episodes.json")
    print(f"  HELD-OUT: on target {final['rate']:.0%} "
          f"({final['hits']}/{final['episodes']}), miss {final['miss']:.2f} m,"
          f" crash {final['crash']:.0%}")
    if t_rate:
        print(f"  teacher was {t_rate:.0%}; the student retains "
              f"{final['rate'] / t_rate:.0%} of it")
    if args.dry_run:
        print("\n  DRY RUN — a fraction of the budget, not a result.")
    return 0


def _teacher_action(teacher, tnorm, env, device):
    """What the teacher WOULD do here, deterministically.

    The mean of the policy's Gaussian, not a sample. A sampled
    label would inject the teacher's exploration noise into the
    regression target, which the student would then have to average
    out -- costing data for no benefit, since the behaviour being
    taught is the mean.
    """
    import numpy as np
    import torch

    o = tnorm(env.obs_for("oracle")).astype(np.float32)
    with torch.no_grad():
        mu = teacher.mu(teacher.pi_body(torch.as_tensor(o, device=device)))
    return mu.cpu().numpy()


def _student_action(student, snorm, obs, device, engine=None):
    import numpy as np
    import torch

    net = engine.module if engine is not None else student
    o = snorm(obs).astype(np.float32)
    with torch.no_grad():
        return net(torch.as_tensor(o, device=device)).cpu().numpy()


def _collect(teacher, tnorm, student, snorm, env_kw, seed, episodes,
             beta, device, n_envs, engine):
    """
    One DAgger round. Returns (student observations, teacher labels).

    Serial rather than pooled, and the reason is the relabelling:
    the teacher has to be queried at every state the student
    visits, which means a torch forward pass interleaved with each
    env step. Shipping the policy into 16 worker processes to do
    that would need the weights broadcast every round. The round is
    bounded by episodes rather than by steps, so it stays short.
    """
    import numpy as np

    from teach_env import TeachEnv

    env = TeachEnv(sense="onboard", seed=seed, **env_kw)
    rng = np.random.default_rng(seed * 7919 + int(beta * 1e6))
    xs, ys = [], []
    teacher_steps = total = 0
    for ep in range(episodes):
        obs = env.reset(seed * 100_000 + ep)
        done = False
        while not done:
            # The LABEL is always the teacher's, whoever drives.
            lab = _teacher_action(teacher, tnorm, env, device)
            xs.append(obs.copy())
            ys.append(lab)
            if rng.random() < beta:
                act = lab
                teacher_steps += 1
            else:
                act = _student_action(student, snorm, obs, device, engine)
            total += 1
            obs, _rew, done, _info = env.step(np.clip(act, 0.0, 1.0))
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            teacher_steps / max(total, 1))


def _fit(student, snorm, X, Y, epochs, batch, device, opt, engine):
    """Least squares from the student's view to the teacher's action."""
    import numpy as np
    import torch

    net = engine.module if engine is not None else student
    Xn = torch.as_tensor(snorm(X).astype(np.float32), device=device)
    Yt = torch.as_tensor(Y, device=device)
    n = len(Xn)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            pred = (engine(Xn[idx]) if engine is not None else net(Xn[idx]))
            loss = ((pred - Yt[idx]) ** 2).mean()
            if engine is not None:
                engine.backward(loss)
                engine.step()
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            last = float(loss)
    return last


def _rollout_eval(student, snorm, env_kw, episodes, device, engine,
                  seed0=700_000):
    """Fly the student alone and count landings."""
    import numpy as np

    from teach_env import TeachEnv

    env = TeachEnv(sense="onboard", seed=0, **env_kw)
    hits = crashes = 0
    misses, seeds, per_hit, outcomes = [], [], [], []
    for ep in range(episodes):
        obs = env.reset(seed0 + ep)
        done = False
        while not done:
            a = _student_action(student, snorm, obs, device, engine)
            obs, _r, done, _i = env.step(np.clip(a, 0.0, 1.0))
        hits += env.outcome == "on_target"
        crashes += env.outcome == "crash"
        misses.append(env.miss())
        seeds.append(seed0 + ep)
        per_hit.append(env.outcome == "on_target")
        outcomes.append(env.outcome)
    return {"episodes": episodes, "hits": hits,
            "rate": hits / max(episodes, 1),
            "crash": crashes / max(episodes, 1),
            "miss": float(np.mean(misses)),
            "seeds": seeds, "on_target": per_hit,
            "outcome": outcomes, "per_miss": misses}


if __name__ == "__main__":
    raise SystemExit(main())
