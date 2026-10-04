#!/usr/bin/env python3
"""
Train the vision student by behaviour cloning, and check that it looks.

    uv run train_student.py                  # train + evaluate + probe
    uv run train_student.py --device cuda
    uv run train_student.py --bench          # CPU vs GPU, nothing else

THE EXPERIMENT
--------------
Three policies on the same three terrains:

    privileged   handed the terrain type and geometry. Cannot be deployed.
    blind        proprioception only. Deployable, and measurably worse:
                 71 points behind on ascent, 46 on descent.
    VISION       proprioception plus a 64x64 depth image.

The question is how much of that gap depth recovers. The first two are
measured; this script produces the third.

WHY THIS IS THE FIRST GPU-RELEVANT STEP IN 07_physical_ai
---------------------------------------------------------
Labs 1 and 2 ran policies of 10k and 12k parameters, and both measured
CPU as FASTER than a GPU, because the work was MuJoCo stepping rather
than arithmetic. A convolutional encoder over depth is roughly a hundred
times larger, and this is supervised training over a fixed dataset -- big
batches, no simulator in the loop. If a GPU ever wins in this category it
wins here, and `--bench` measures it rather than assuming.

THE THREE CHECKS
----------------
A vision policy that scores well may still be ignoring its camera. Both
previous labs in this category shipped an information ablation that came
back null, so the burden of proof is high:

  1. BLANK IMAGE at evaluation. If performance holds, the student is
     reading proprioception and the camera is decoration.
  2. LINEAR PROBE on the frozen encoder, predicting terrain type. If the
     encoder sees, terrain should be nearly linearly separable in it.
  3. PER-TERRAIN results. `flat` is solvable blind, so an average hides
     everything -- the question is only about `up` and `down`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


class VisionPolicy:
    """Built lazily so `--help` does not import torch."""


def build(torch, nn, proprio_dim: int, act_dim: int, width: int = 32):
    class Net(nn.Module):
        """
        A small conv encoder over depth, concatenated with proprioception.

        Deliberately small. The point is not a big model -- it is the
        first model in this category where the arithmetic is non-trivial,
        and keeping it small keeps the CPU/GPU comparison honest rather
        than stacked in the GPU's favour.
        """

        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, width, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            )
            self.feat = width * 2 * 4
            self.head = nn.Sequential(
                nn.Linear(self.feat + proprio_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, act_dim),
            )

        def encode(self, depth):
            return self.enc(depth.unsqueeze(1))

        def forward(self, depth, proprio):
            return self.head(torch.cat([self.encode(depth), proprio], dim=-1))

    return Net()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/bc.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto", action="store_true",
                    help="produce the prerequisites if they are missing: "
                         "train a teacher, then render the dataset, then "
                         "train the student. Neither `runs/` nor the "
                         "rendered dataset is committed (both are "
                         "reproducible), so this is what makes the lab "
                         "runnable on a fresh pod in one command.")
    ap.add_argument("--eval-only", action="store_true",
                    help="re-run the probe and the rollouts against a "
                         "checkpoint that already exists, and rewrite its "
                         "summary.json. Used when the WORLD changes but the "
                         "policy does not -- retraining would also work but "
                         "would risk the published numbers describing "
                         "different weights than the animations do.")
    ap.add_argument("--bench", action="store_true",
                    help="time CPU against GPU on this model and stop")
    ap.add_argument("--tag", default=None,
                    help="run directory name; used to keep an UNDERTRAINED "
                         "checkpoint beside the finished one, so the "
                         "before/after animation compares two real "
                         "policies rather than a policy against noise")
    ap.add_argument("--episodes", type=int, default=12,
                    help="evaluation episodes per terrain")
    a, _ = ap.parse_known_args()
    run(a)


def pick(torch, want: str) -> str:
    if want == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if want in ("cuda", "auto"):
        try:
            torch.zeros(8, device="cuda") @ torch.zeros(8, 8, device="cuda")
            torch.cuda.synchronize()
            return "cuda"
        except Exception:                                      # noqa: BLE001
            return "cpu"
    return "cpu"


def ensure_prerequisites(a) -> None:
    """
    Train a teacher and render the dataset, if they are not already here.

    Shelling out rather than importing, so each stage is exactly the
    command the README documents -- a reader who runs them by hand gets
    the same artifacts, and there is no second code path that can drift
    from the first.
    """
    import subprocess
    import sys as _sys

    if not any((HERE / "runs").glob("v3_priv_*/summary.json")):
        print("  no teacher found -- training one (~13 min, CPU)")
        subprocess.run([_sys.executable, str(HERE / "train_teacher.py"),
                        "--name", "v3_priv_s0", "--quiet"], check=True)
    if not (HERE / a.data).exists():
        print(f"  no {a.data} -- rendering it (~9 min)")
        subprocess.run([_sys.executable, str(HERE / "collect.py"),
                        "--episodes", "120"], check=True)


def missing_data(a) -> None:
    print(f"""
  {a.data} does not exist.

  It is 23 MB of rendered depth frames, and it is NOT committed --
  `collect.py` reproduces it in about nine minutes, so it belongs with
  runs/ rather than in git history.

  Produce it, with a teacher to render from:

      uv run train_teacher.py --name v3_priv_s0
      uv run collect.py --episodes 120

  or have this script do both for you:

      uv run train_student.py --auto --device cuda
""", file=sys.stderr)
    sys.exit(1)


def run(a) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    if a.auto:
        ensure_prerequisites(a)
    if not (HERE / a.data).exists():
        missing_data(a)

    d = np.load(HERE / a.data)
    depth, proprio, action, terrain = (d["depth"], d["proprio"],
                                       d["action"], d["terrain"])
    n = len(depth)
    print("=" * 74)
    print(f"  dataset {n:,} frames   depth {depth.shape[1:]}   "
          f"proprio {proprio.shape[1]}   actions {action.shape[1]}")
    print("=" * 74)

    if a.bench:
        bench(torch, nn, a, depth, proprio, action)
        return

    dev = pick(torch, a.device)
    torch.manual_seed(a.seed)
    out = HERE / "runs" / (a.tag or "student")
    net = build(torch, nn, proprio.shape[1], action.shape[1], a.width).to(dev)

    if a.eval_only:
        ck = torch.load(out / "student.pt", weights_only=False)
        net.load_state_dict(ck["model"])
        print(f"  device {dev}   loaded {out.name}/student.pt   "
              f"{sum(p.numel() for p in net.parameters()):,} parameters")
        probe(torch, nn, net, dev, depth, terrain)
        rolls = evaluate(torch, net, dev, a.episodes)
        prev = json.loads((out / "summary.json").read_text())
        prev.update({"rollout": rolls, "eval_episodes": a.episodes})
        (out / "summary.json").write_text(json.dumps(prev, indent=2))
        return

    print(f"  device {dev}   student {sum(p.numel() for p in net.parameters()):,} "
          f"parameters  (the PPO policies were ~12k)")

    # A held-out split, because training loss on a BC dataset tells you
    # nothing about whether the thing will drive a robot.
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(n)
    cut = int(0.9 * n)
    tr, va = idx[:cut], idx[cut:]
    T = lambda x, i: torch.as_tensor(x[i], dtype=torch.float32, device=dev)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(tr), a.batch):
            b = tr[perm[s:s + a.batch]]
            loss = ((net(T(depth, b), T(proprio, b)) - T(action, b)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = ((net(T(depth, va), T(proprio, va)) - T(action, va)) ** 2).mean().item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}  train {tot/len(tr):.4f}  val {vl:.4f}"
                  f"  {time.time()-t0:5.0f}s")

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "width": a.width,
                "proprio_dim": proprio.shape[1]}, out / "student.pt")

    probe(torch, nn, net, dev, depth, terrain)
    rolls = evaluate(torch, net, dev, a.episodes)
    (out / "summary.json").write_text(json.dumps(
        {"device": dev, "frames": int(n), "epochs": a.epochs,
         "params": sum(p.numel() for p in net.parameters()),
         "val_mse": round(vl, 5), "rollout": rolls,
         "eval_episodes": a.episodes}, indent=2))


def probe(torch, nn, net, dev, depth, terrain) -> None:
    """
    Check 2: is terrain linearly decodable from the frozen encoder?

    A student can score well while ignoring its camera. If the encoder
    has learned to see, a single linear layer on its features should
    recover which terrain the frame came from.
    """
    import numpy as np

    net.eval()
    with torch.no_grad():
        feats = []
        for s in range(0, len(depth), 512):
            x = torch.as_tensor(depth[s:s + 512], dtype=torch.float32, device=dev)
            feats.append(net.encode(x).cpu())
        F = torch.cat(feats)
    y = torch.as_tensor(terrain)
    cut = int(0.8 * len(F))
    lin = nn.Linear(F.shape[1], int(y.max()) + 1)
    o = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(400):
        loss = nn.functional.cross_entropy(lin(F[:cut]), y[:cut])
        o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():
        acc = (lin(F[cut:]).argmax(1) == y[cut:]).float().mean().item()
    chance = float(np.bincount(terrain).max() / len(terrain))
    print()
    print(f"  LINEAR PROBE on the frozen encoder: terrain accuracy "
          f"{acc:.1%}  (chance {chance:.1%})")
    print(f"  -> the encoder {'HAS' if acc > chance + 0.15 else 'has NOT'} "
          f"learned to distinguish the terrains")


def evaluate(torch, net, dev, episodes: int) -> dict:
    """Checks 1 and 3: per terrain, with and without a working camera."""
    import numpy as np

    from terrain import KINDS
    from vision_env import TerrainWorld

    net.eval()
    out = {}
    for blank in (False, True):
        tag = "blank" if blank else "vision"
        out[tag] = {}
        for k in KINDS:
            ok = 0
            for i in range(episodes):
                env = TerrainWorld(kind=k, privileged=False, depth=True,
                                   seed=0)
                obs, _ = env.reset(seed=8000 + i)
                while True:
                    img = (np.zeros((64, 64), np.float32) if blank
                           else env.depth())
                    with torch.no_grad():
                        act = net(
                            torch.as_tensor(img, device=dev).unsqueeze(0),
                            torch.as_tensor(obs[:15], dtype=torch.float32,
                                            device=dev).unsqueeze(0),
                        ).squeeze(0).cpu().numpy()
                    obs, _, term, trunc, info = env.step(act)
                    if term or trunc:
                        break
                ok += bool(info["past_event"])
            out[tag][k] = round(ok / episodes, 3)
    print()
    print(f"  {'terrain':<8}{'with camera':>14}{'BLANK image':>14}")
    for k in KINDS:
        print(f"  {k:<8}{out['vision'][k]:>13.0%}{out['blank'][k]:>14.0%}")
    drop = (np.mean(list(out["vision"].values()))
            - np.mean(list(out["blank"].values())))
    print(f"  blanking the camera costs {drop:+.0%} on average")
    print(f"  -> the policy {'USES' if drop > 0.1 else 'does NOT use'} "
          f"its camera")
    return out


def bench(torch, nn, a, depth, proprio, action) -> None:
    """
    Time one training epoch on CPU against GPU.

    Labs 1 and 2 both found CPU faster, because the work was physics. Here
    there is no simulator in the loop and the model is ~100x larger, so
    this is where the answer should flip -- and it is worth measuring
    rather than asserting, since the last two measurements went the other
    way.
    """
    import numpy as np

    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            print("  cuda unavailable"); continue
        torch.manual_seed(0)
        net = build(torch, nn, proprio.shape[1], action.shape[1], a.width).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        T = lambda x, i: torch.as_tensor(x[i], dtype=torch.float32, device=dev)
        idx = np.arange(min(len(depth), 8192))
        for _ in range(2):                          # warm up
            b = idx[:a.batch]
            ((net(T(depth, b), T(proprio, b)) - T(action, b)) ** 2).mean().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for s in range(0, len(idx), a.batch):
            b = idx[s:s + a.batch]
            loss = ((net(T(depth, b), T(proprio, b)) - T(action, b)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  {dev:<5} {len(idx)/dt:8.0f} samples/s   "
              f"({dt:.2f}s for {len(idx):,} samples, batch {a.batch})")


if __name__ == "__main__":
    main()
