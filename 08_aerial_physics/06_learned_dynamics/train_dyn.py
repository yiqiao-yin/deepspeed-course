#!/usr/bin/env python3
"""
Fit a dynamics model to the cold-gas car, and see whether size helps.

    uv run train_dyn.py --size small --kind mlp
    uv run train_dyn.py --size xxl --kind gru --deepspeed
    uv run train_dyn.py --ladder                 # every rung, in order

WHAT IS MEASURED
----------------
One-step error is the easy metric and the misleading one. At 100 Hz
a model that simply predicts "no change" already scores well on it,
because consecutive observations barely differ. The metric that
matters for anything downstream is OPEN-LOOP ROLLOUT ERROR: feed the
model its own prediction for H steps and see how far it has drifted
from the real trajectory.

Both are reported, and the gap between them is the point.

THE BAR, WRITTEN DOWN FIRST
----------------------------
A model 1000x larger must cut 50-step rollout error by at least half,
or the sharding machinery is not earning its place in this topic.
Topic 7 lab 6 did the same thing with GPU physics -- "5x the CPU pool
or it is not worth renting" -- and got a clean FAILS, which was the
finding. The bar is printed before the result either way.

WHY THIS LAB MAY USE DEEPSPEED WHEN THE OTHER FIVE DO NOT
----------------------------------------------------------
Labs 1 to 5 train a ~14k-parameter policy. There is genuinely nothing
to shard and saying so five times was correct. The dynamics model is
a different object: the largest rung here is 705M parameters, which
needs 11.3 GB of weights and Adam state before a single activation,
against the 16 GB on the card this was measured on.

So `--deepspeed` is offered, and whether it is WORTH offering is the
thing being measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


# Rungs that genuinely fit on a CPU in a reasonable time. Anything
# above `medium` is 1.8 GB or more of weights and Adam state and will
# crawl without a card.
CPU_OK = ("tiny", "small", "medium")


def require_gpu(size: str = "xxl") -> None:
    """
    A REAL guard, the first in topic 8 -- and SIZE-AWARE, which is
    the part that matters for the manifest.

    Labs 1 to 5 deliberately carry no guard: a 14k-parameter MLP
    against MuJoCo is faster on the CPU and a guard that can never
    fire is decoration. The big rungs here genuinely need a card.

    But a guard that fires for EVERY size would make the small rungs
    unrunnable on a CPU-only box while Clawdeck still advertised
    them as "Runs now, no GPU needed" -- exactly the manifest defect
    this repo already shipped six times. So the guard asks which
    rung was requested and only stops the ones that really cannot
    run. ALLOW_CPU=1 still overrides it for the rest.
    """
    import os

    if size in CPU_OK or os.environ.get("ALLOW_CPU") == "1":
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
    print(f"  The {', '.join(CPU_OK)} rungs run fine on a CPU and are")
    print("  not blocked by this guard at all:")
    print("      uv run train_dyn.py --size small --kind mlp")
    print()
    print("  The large ones do not. `xl` and `xxl` need 2 to 11 GB of")
    print("  weights and Adam state before any activation -- see")
    print("      uv run dynamics.py")
    print()
    print("  To rent a card:")
    print("      uv run runpod/runpod_ctl.py run "
          "08_aerial_physics/06_learned_dynamics \\")
    print("          --dry-run --collect --wait --terminate --yes")
    raise SystemExit(2)


def load_split(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rollout_error(model, obs, act, starts, ends, horizon, device,
                  n_traj=64, seed=0):
    """
    Open-loop error: feed the model its OWN prediction for `horizon`
    steps and measure the drift from the truth.

    This is the number that decides whether a model is useful to a
    planner. One-step error flatters every model at 100 Hz because
    "predict no change" is nearly right.
    """
    import torch

    rng = np.random.default_rng(seed)
    usable = [(s, e) for s, e in zip(starts, ends) if e - s > horizon + 1]
    if not usable:
        return float("nan")
    pick = rng.choice(len(usable), size=min(n_traj, len(usable)),
                      replace=False)
    errs = []
    model.eval()
    with torch.no_grad():
        for i in pick:
            s, _e = usable[i]
            o = torch.as_tensor(obs[s:s + 1], device=device)
            state = None
            for h in range(horizon):
                u = torch.as_tensor(act[s + h:s + h + 1], device=device)
                o, state = model(o, u, state)
            truth = torch.as_tensor(obs[s + horizon:s + horizon + 1],
                                    device=device)
            errs.append(float(torch.linalg.norm(o - truth)))
    model.train()
    return float(np.mean(errs))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="small")
    ap.add_argument("--kind", default="mlp", choices=("mlp", "gru"))
    ap.add_argument("--train", default="data/train.npz")
    ap.add_argument("--val", default="data/val.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seq", type=int, default=32,
                    help="sequence length for the recurrent model")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=50,
                    help="open-loop rollout length for the headline metric")
    ap.add_argument("--deepspeed", action="store_true",
                    help="train under DeepSpeed ZeRO. Whether this is "
                         "WORTH doing is what the lab measures.")
    ap.add_argument("--zero-stage", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="runs")
    args, _ = ap.parse_known_args()

    # GATED AT THE CALL SITE, not inside the function.
    #
    # Hiding the size test behind an early return made the guard look
    # unconditional to tests/test_clawdeck_manifest.py, which reads
    # the AST and cannot see into the callee. It flagged the CPU
    # command as reaching a preflight that always fires -- correctly,
    # on the evidence available to it. Writing the condition here
    # makes the gating true for the checker and obvious to a reader.
    if args.size not in CPU_OK:
        require_gpu(args.size)

    import torch

    import dynamics as DY

    dev = (args.device if args.device != "auto"
           else ("cuda" if torch.cuda.is_available() else "cpu"))
    tr = load_split(HERE / args.train)
    va = load_split(HERE / args.val)
    obs_dim = tr["obs"].shape[1]
    act_dim = tr["act"].shape[1]

    model = DY.build(args.kind, args.size, obs_dim, act_dim).to(dev)
    n_par = DY.n_params(model)
    name = args.name or f"{args.kind}_{args.size}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DYNAMICS — {name}: {args.kind} {args.size}, {n_par:,} parameters")
    print("=" * 78)
    mix = tr["mix"]
    print(f"  train {len(tr['obs']):,} transitions over {len(tr['ep_start'])} "
          f"episodes  (mix {mix[0]}/{mix[1]}/{mix[2]} "
          f"trained/scripted/random)")
    print(f"  val   {len(va['obs']):,} transitions, DISJOINT seeds")
    print(f"  device {dev}"
          + (f", DeepSpeed ZeRO-{args.zero_stage}" if args.deepspeed else ""))
    print(f"  weights + Adam state: {n_par * 16 / 1e9:.2f} GB on one card")

    engine = None
    if args.deepspeed:
        # SET THE RENDEZVOUS VARS BEFORE IMPORTING DEEPSPEED.
        #
        # Without them `deepspeed.initialize` falls through to MPI
        # discovery and dies on a missing mpi4py, which reads like a
        # dependency problem and is not one -- it is a single-process
        # run with no launcher, so there is no rendezvous to discover.
        # Same family as topic 7's rule that LOCAL_RANK comes from the
        # environment and not from argv.
        import os

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")

        import deepspeed

        # SUPPLY THE OPTIMISER, do not let DeepSpeed build one.
        #
        # Naming "Adam" in the config makes DeepSpeed construct its
        # own fused Adam, which it JIT-compiles with nvcc at first
        # use. There is no CUDA toolkit on this box -- the PyPI
        # `nvidia-cuda-nvcc` wheels ship ptxas and not nvcc -- so
        # that compile dies with "CUDA_HOME environment variable is
        # not set", which reads like a misconfiguration and is
        # really a missing toolchain. The repo already carries this
        # lesson for DS4Sci_EvoformerAttention.
        #
        # Handing in a torch optimiser keeps ZeRO's partitioning and
        # skips the kernel build entirely.
        torch_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # The config is a FILE, as in every other DeepSpeed example in
        # this course, so the ZeRO settings are inspectable without
        # reading Python. The two values the CLI owns are overridden
        # on top of it.
        ds_cfg = json.loads((HERE / "ds_config.json").read_text())
        ds_cfg.pop("_comment", None)
        ds_cfg["train_micro_batch_size_per_gpu"] = args.batch
        ds_cfg["zero_optimization"]["stage"] = args.zero_stage
        engine, opt, _, _ = deepspeed.initialize(
            model=model, optimizer=torch_opt, config=ds_cfg)
        model = engine
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Normalise inputs and the TARGET DELTA separately. The delta is
    # three orders of magnitude smaller than the state at 100 Hz, so
    # a loss on raw deltas is numerically a loss on nothing.
    o_mu = tr["obs"].mean(0, keepdims=True)
    o_sd = tr["obs"].std(0, keepdims=True) + 1e-6
    d_tr = tr["nxt"] - tr["obs"]
    d_sd = d_tr.std(0, keepdims=True) + 1e-8

    O = torch.as_tensor(tr["obs"], device=dev)
    A = torch.as_tensor(tr["act"], device=dev)
    N = torch.as_tensor(tr["nxt"], device=dev)
    W = torch.as_tensor(1.0 / d_sd, device=dev)

    n = len(O)
    hist = []
    t0 = time.time()
    peak0 = (torch.cuda.max_memory_allocated() / 1e9
             if dev == "cuda" else 0.0)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for i in range(0, n - args.batch, args.batch):
            idx = perm[i:i + args.batch]
            pred, _ = model(O[idx], A[idx], None)
            # Weighted by the inverse per-channel spread of the true
            # delta, so a channel that barely moves cannot dominate
            # and one that moves a lot cannot be ignored.
            loss = (((pred - N[idx]) * W) ** 2).mean()
            if engine is not None:
                engine.backward(loss)
                engine.step()
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters()], 1.0)
                opt.step()
            tot += float(loss)
            nb += 1

        one = float(tot / max(nb, 1))
        roll = rollout_error(model, va["obs"], va["act"],
                             va["ep_start"], va["ep_end"],
                             args.horizon, dev)
        hist.append({"epoch": ep + 1, "train_loss": one,
                     "rollout_err": roll,
                     "seconds": time.time() - t0})
        print(f"  epoch {ep + 1:>3}/{args.epochs}  loss {one:.5f}  "
              f"{args.horizon}-step rollout err {roll:8.3f}  "
              f"{time.time() - t0:6.0f}s", flush=True)

    peak = (torch.cuda.max_memory_allocated() / 1e9
            if dev == "cuda" else 0.0)
    short = rollout_error(model, va["obs"], va["act"], va["ep_start"],
                          va["ep_end"], 1, dev)
    summary = {
        "name": name, "kind": args.kind, "size": args.size, "params": n_par,
        "device": dev, "deepspeed": args.deepspeed,
        "zero_stage": args.zero_stage if args.deepspeed else None,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "horizon": args.horizon,
        "one_step_err": short,
        "rollout_err": hist[-1]["rollout_err"],
        "best_rollout_err": min(h["rollout_err"] for h in hist),
        "peak_gpu_gb": peak, "wall_seconds": time.time() - t0,
        "train_mix": [int(x) for x in mix],
        "history": hist,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"model": (engine.module if engine is not None
                          else model).state_dict(),
                "kind": args.kind, "size": args.size,
                "obs_dim": obs_dim, "act_dim": act_dim},
               out / "model.pt")

    print(f"\n  1-step error      {short:.4f}")
    print(f"  {args.horizon}-step error     {summary['rollout_err']:.3f}"
          f"   (best {summary['best_rollout_err']:.3f})")
    if dev == "cuda":
        print(f"  peak GPU          {peak:.2f} GB")
    print(f"  wall              {summary['wall_seconds'] / 60:.1f} min")
    print(f"  wrote {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
