#!/usr/bin/env python3
"""
Fly two arms down the same launch, side by side, and watch them diverge.

    uv run render_arms.py --left runs/oracle_s0 --right runs/onboard_s0
    uv run render_arms.py --left runs/oracle_s0 --right runs/dagger_s0 \
        --out ts-teacher-vs-dagger
    uv run render_arms.py --landings runs/dagger_s0

WHY SIDE BY SIDE AND NOT TWO FILMS
-----------------------------------
Both panels are the SAME held-out seed: same terrain, same mark,
same launch attitude, same starting propellant. Two separate
clips of two separate flights would show two outcomes and prove
nothing about which arm caused them -- on terrain whose
per-spot success varies from 8% to 83%, the easiest way to make
any policy look good is to film it on an easy spot.

Stepping them in lockstep from one seed is the visual form of
the paired comparison `compare.py` does numerically, and it is
the same reason that script refuses to report unless every arm
flew the identical seed list.

THE HUD CARRIES THE TWO NUMBERS THE ARMS DISAGREE ABOUT
---------------------------------------------------------
Height above ground and lowest-corner clearance. The teacher is
simply told both; the student is reading them out of 144 beams
to about 0.7 m. Showing the TRUE value on both panels, with the
student's own estimate beside it where one exists, is what makes
the divergence legible as a perception failure rather than as
bad luck.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docusaurus-docs" / "static" / "img" / "aerial"
BACKENDS = ("egl", "osmesa", "glfw")

BG = (11, 15, 20)
PANEL = (16, 22, 30)
INK = (236, 242, 248)
DIM = (150, 170, 186)
FAINT = (104, 126, 146)
GOOD = (108, 206, 148)
WARN = (226, 176, 92)
BAD = (228, 108, 108)
JET = (126, 198, 248)
PINK = (226, 140, 200)

OUTCOME_COL = {"on_target": GOOD, "landed_off": WARN, "landed_hard": WARN,
               "crash": BAD, "timeout": BAD}


def pick_backend() -> str:
    """Find a backend that can RENDER, not merely import. See lab 6."""
    import importlib
    import sys

    for b in BACKENDS:
        os.environ["MUJOCO_GL"] = b
        for mod in [m for m in list(sys.modules) if m.startswith("mujoco")]:
            del sys.modules[mod]
        try:
            mujoco = importlib.import_module("mujoco")
            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
            rr = mujoco.Renderer(m, height=8, width=8)
            rr.update_scene(mujoco.MjData(m))
            rr.render()
            rr.close()
            print(f"  [render] MUJOCO_GL={b}")
            return b
        except Exception:
            continue
    raise SystemExit(
        "  No OpenGL backend available. Everything this lab MEASURES\n"
        "  needs none:  uv run compare.py")


def save_gif(frames, path, ms=70, colors=160, hold_ms=1500, holds=None):
    """
    Write a GIF that actually animates.

    `optimize=True` merges consecutive identical frames and rolls
    their durations together, so a file can report a frame count
    it does not have and look like a still. Lab 6 shipped three
    figures that way. Verified by reading the file back.

    `holds` is a set of frame indices that should linger -- the
    end of each episode, so a reader can read the outcome before
    the next launch starts. A LINGER IS A DURATION, NOT A REPEATED
    FRAME. The first version of this function appended six copies
    of the final frame and the guard below rejected its own
    output, correctly: duplicates are precisely what the optimiser
    collapses, and the frame count then stops describing the file.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [f.convert("P", palette=Image.ADAPTIVE, colors=colors,
                     dither=Image.Dither.NONE) for f in frames]
    holds = holds or set()
    durations = [(hold_ms if i in holds else ms) for i in range(len(pil))]
    durations[-1] = hold_ms
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=durations, loop=0, optimize=False)
    kept = getattr(Image.open(path), "n_frames", 1)
    if kept != len(frames):
        raise RuntimeError(
            f"{path.name}: asked for {len(frames)} frames, the file has "
            f"{kept}. Consecutive frames were merged -- it would look "
            f"frozen.")
    print(f"  {path.name}  ({kept} frames @ {ms} ms + {hold_ms} ms hold, "
          f"{path.stat().st_size / 1024 / 1024:.2f} MB)")


def load_any(run_dir: Path, device="cpu"):
    """
    Load either a PPO ActorCritic or a distilled student.

    They are different objects -- the PPO arms carry a value head
    and a log-std, the students are a bare regression net -- so the
    checkpoint is tagged and the caller gets back one uniform
    `act(obs) -> action` closure either way. Without the tag this
    would be a `try: load as A except: load as B`, which silently
    picks the wrong one when a key happens to match.
    """
    import torch

    from ppo import ActorCritic, RunningNorm

    ck = torch.load(run_dir / "policy.pt", map_location=device,
                    weights_only=False)
    norm = RunningNorm(ck["obs_dim"])
    norm.load_state_dict(ck["norm"])

    if ck.get("arch") == "student_mlp":
        from distill import build_student

        net = build_student(ck["obs_dim"], ck["act_dim"], ck["size"],
                            ck.get("bias"))
        net.load_state_dict(ck["model"])
        net.eval()

        def act(o):
            with torch.no_grad():
                x = torch.as_tensor(norm(o), dtype=torch.float32)
                return net(x).numpy()
        kind = f"distilled student ({ck['size']})"
    else:
        net = ActorCritic(ck["obs_dim"], ck["act_dim"],
                          init_bias=ck.get("bias"))
        net.load_state_dict(ck["model"])
        net.eval()

        def act(o):
            with torch.no_grad():
                x = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
                return net.distribution(x).mean.squeeze(0).numpy()
        kind = "PPO policy"
    return act, ck["obs_dim"], kind


def sense_of(run_dir: Path) -> str:
    import json

    s = json.loads((run_dir / "summary.json").read_text())
    # The distilled arms have no `sense` key -- they are always
    # flown onboard, which is the entire point of distilling them.
    return s.get("sense", "onboard")


def label_of(run_dir: Path) -> str:
    import json

    s = json.loads((run_dir / "summary.json").read_text())
    sense = s.get("sense")
    rate = s.get("eval", {}).get("rate")
    if sense == "oracle":
        what = "TEACHER  -  told its exact height"
    elif sense == "deaf":
        what = "DEAF  -  no height information"
    elif s.get("method") == "dagger":
        what = "STUDENT, DAgger  -  reads 144 beams"
    elif s.get("method") == "bc":
        what = "STUDENT, cloned  -  reads 144 beams"
    else:
        what = "DIRECT RL  -  reads 144 beams"
    tail = f"   held-out {rate:.0%}" if rate is not None else ""
    return f"{what}{tail}"


def panel_hud(img, env, step, n_steps, title, est=None, outcome=None):
    from PIL import ImageDraw

    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 46], fill=(8, 12, 18, 230))
    d.text((10, 5), title, fill=INK)
    d.text((10, 22), f"t {step / 100:5.2f}s   alt {env.altitude():5.2f} m"
                     f"   clear {env.clearance():5.2f} m"
                     f"   miss {env.miss():5.2f} m", fill=DIM)
    if est is not None:
        d.text((10, 34), f"      the beams say {est:4.2f} m", fill=PINK)
    tilt = env.tilt_deg()
    col = GOOD if tilt < 15 else (WARN if tilt < 25 else BAD)
    d.text((W - 170, 22), f"tilt {tilt:4.1f}  "
                          f"sink {max(0.0, -env.vel()[2]):4.1f}", fill=col)
    if outcome:
        oc = OUTCOME_COL.get(outcome, DIM)
        d.rectangle([0, H - 30, W, H], fill=(8, 12, 18, 235))
        d.text((10, H - 22), outcome.replace("_", " ").upper(), fill=oc)
    bw = int((W - 20) * step / max(n_steps, 1))
    d.rectangle([10, H - 4, 10 + bw, H - 2], fill=col)
    return img


def film_pair(left: Path, right: Path, episodes: int, seed0: int,
              width: int, height: int, every: int, out_name: str,
              rig: str, preset: str, colors: int = 96) -> None:
    import mujoco
    from PIL import Image, ImageDraw

    from teach_env import TeachEnv

    pw = width // 2
    cams, envs, acts, renderers, titles = [], [], [], [], []
    for rd in (left, right):
        a, odim, _kind = load_any(rd)
        env = TeachEnv(rig, preset, sense=sense_of(rd), seed=0)
        if env.obs_dim != odim:
            raise ValueError(
                f"{rd.name} expects {odim} inputs, this env gives "
                f"{env.obs_dim}. Wrong rig or preset.")
        envs.append(env)
        acts.append(a)
        titles.append(label_of(rd))
        r = mujoco.Renderer(env.model, height=height - 0, width=pw)
        renderers.append(r)
        c = mujoco.MjvCamera()
        c.elevation = -21.0
        cams.append(c)

    frames, holds = [], set()
    for ep in range(episodes):
        obs = [e.reset(seed0 + ep) for e in envs]
        done = [False, False]
        oc = [None, None]
        n_steps = envs[0].max_steps
        for k in range(n_steps):
            if k % every == 0:
                tiles = []
                for i, (env, r, c) in enumerate(zip(envs, renderers, cams)):
                    p = env.pos()
                    c.lookat[:] = [p[0], p[1], max(p[2] * 0.75, 0.4)]
                    c.distance = 12.0 + 1.0 * max(env.altitude() - 2.0, 0.0)
                    c.azimuth = 130.0 + 14.0 * math.sin(k / 100.0)
                    r.update_scene(env.data, c)
                    tile = Image.fromarray(r.render()).convert("RGB")
                    tiles.append(panel_hud(tile, env, k + 1, n_steps,
                                           titles[i], outcome=oc[i]))
                img = Image.new("RGB", (width, height), BG)
                img.paste(tiles[0], (0, 0))
                img.paste(tiles[1], (pw, 0))
                d = ImageDraw.Draw(img)
                d.line([pw, 0, pw, height], fill=(60, 74, 90), width=2)
                d.text((pw - 92, 52),
                       f"one launch, seed {seed0 + ep}", fill=FAINT)
                frames.append(img)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                a = np.clip(acts[i](obs[i]), 0.0, 1.0)
                obs[i], _r, dd, _ = env.step(a)
                if dd:
                    done[i] = True
                    oc[i] = env.outcome
            if all(done):
                # Let the settled result LINGER, so a reader sees
                # which outcome each arm reached rather than the
                # film cutting at the instant of contact. Marked as
                # a long duration on this one frame; appending
                # copies would be merged away.
                holds.add(len(frames) - 1)
                break
        print(f"    seed {seed0 + ep}: {oc[0]} | {oc[1]}")
    for r in renderers:
        r.close()
    save_gif(frames, OUT / f"{out_name}.gif", ms=65, holds=holds,
             colors=colors)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="runs/oracle_s0")
    ap.add_argument("--right", default="runs/onboard_s0")
    ap.add_argument("--out", default="ts-teacher-vs-student")
    ap.add_argument("--rig", default="dense")
    ap.add_argument("--preset", default="badlands")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--colors", type=int, default=96,
                    help="GIF palette size. These films are two rendered "
                         "panels per frame and the file grows fast: the "
                         "first cut, 4 episodes at every=3 and 160 colours, "
                         "came out at 50 MB, which is too heavy to link let "
                         "alone embed. 96 colours at every=5 lands near 12.")
    a, _ = ap.parse_known_args()

    pick_backend()
    import sys

    sys.path.insert(0, str(HERE))
    film_pair(HERE / a.left, HERE / a.right, a.episodes, a.seed0,
              a.width, a.height, a.every, a.out, a.rig, a.preset,
              a.colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
