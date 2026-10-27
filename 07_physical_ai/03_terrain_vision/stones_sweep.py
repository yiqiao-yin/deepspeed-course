#!/usr/bin/env python3
"""
PART 2: when does information actually matter?

    uv run stones_sweep.py --difficulty      # the inverted-U, ~45 min
    uv run stones_sweep.py --fov             # wider camera = better?
    uv run stones_sweep.py --all

WHY THIS EXISTS
---------------
Part 1 of this lab measured a camera against no camera on one task and
got a confusing answer: large on ascent, exactly zero on flat and on
descent. The natural reading -- "vision sometimes helps" -- is not very
useful, and it is not what is going on.

The reason is that a task has to be in a particular band before any
extra information can pay. A pilot sweep over stepping-stone difficulty
makes that explicit (specialists, 400k steps, one seed):

    stone 0.40 gap 0.12+   privileged 100%   blind 100%   gap   0
    stone 0.36 gap 0.16+   privileged  75%   blind  25%   gap +50
    stone 0.30 gap 0.22+   privileged   0%   blind   0%   gap   0

Too easy and proprioception suffices. Too hard and nothing works --
that is a broken terrain, not a discrimination, which this lab has
already shipped once with an overhead ceiling at 0% for both arms. In
between, information is decisive.

So the question is not "does vision help" but "is this task in the band
where it can". This script measures the band.

STEPPING STONES RATHER THAN STAIRS
----------------------------------
A staircase is CONTINUOUS: the ground is always somewhere under the
foot, so a blind policy can sweep, touch and react. Sparse footholds
remove that affordance -- between the stones there is nothing to feel,
and a foot placed into a void gets no second chance. This is the
standard benchmark for exteroception being necessary rather than merely
useful (Agarwal et al., CoRL 2022; Miki et al., Science Robotics 2022).

AND A DOSE-RESPONSE, NOT JUST ON/OFF
------------------------------------
`--fov` sweeps the camera's field of view. An on/off comparison can
always be explained by "the network with more inputs trained better".
If performance instead RISES with how much of the world the camera can
see, that is much harder to explain any other way.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys_path_hint = None

HERE = Path(__file__).parent
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE))
from terrain import STONE_KINDS  # noqa: E402

# Found by the pilot above. `medium` is the only cell where the oracle
# solves the task and the blind arm does not, which is the only place a
# sensor comparison can say anything.
DIFFICULTIES = {
    "easy":   dict(STONE_TOP="0.40", STONE_GAP_MIN="0.12", STONE_GAP_MAX="0.20"),
    "medium": dict(STONE_TOP="0.36", STONE_GAP_MIN="0.16", STONE_GAP_MAX="0.26"),
    "hard":   dict(STONE_TOP="0.30", STONE_GAP_MIN="0.22", STONE_GAP_MAX="0.34"),
}

# Degrees. 45 is the default the rest of the lab uses.
FOVS = (25, 45, 70, 100)


def sh(cmd: list[str], env_extra: dict | None = None) -> None:
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "glfw"))
    env.update(env_extra or {})
    subprocess.run(cmd, check=True, env=env, cwd=HERE)


def final(run: str, key: str = "past_stones") -> float | None:
    f = RUNS / run / "summary.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["final"].get(key)


def difficulty_sweep(a) -> dict:
    """
    The inverted-U: privileged against blind, at three difficulties.

    Specialists on `stones` only, so the number is not diluted by three
    other terrains that are solved either way.
    """
    out = {}
    for name, cfg in DIFFICULTIES.items():
        out[name] = {"config": cfg, "privileged": [], "blind": []}
        for seed in range(a.seeds):
            for arm, flag in (("privileged", []), ("blind", ["--no-privileged"])):
                run = f"st_{name}_{arm}_s{seed}"
                if not (RUNS / run / "summary.json").exists():
                    print(f"  training {run}")
                    sh([sys.executable, "train_teacher.py", "--stones",
                        "--terrain", "stones", "--total-steps", str(a.steps),
                        "--name", run, "--seed", str(seed), "--quiet"]
                       + flag, cfg)
                # The arm has to be CHECKED, not assumed. The first
                # version of this loop computed `flag` and never passed
                # it, so all 18 runs trained privileged and 9 were
                # labelled blind. Three difficulties came back with the
                # two arms EXACTLY equal -- 83/83, 75/75, 17/17 -- and
                # that impossible coincidence is the only thing that
                # caught it. A quieter bug would have been published.
                want = 15 if arm == "blind" else len(STONE_KINDS) + 17
                got = json.loads((RUNS / run / "summary.json").read_text())["obs_dim"]
                assert got == want, (
                    f"{run}: obs_dim {got}, expected {want} for a "
                    f"{arm} arm -- the arm flag did not take effect")
                out[name][arm].append(final(run))

    print()
    print("=" * 74)
    print(f"  DIFFICULTY SWEEP — stepping stones, {a.seeds} seeds, "
          f"{a.steps:,} steps")
    print("=" * 74)
    print(f"  {'difficulty':<10}{'stone/gap':<16}{'privileged':>12}{'blind':>10}{'gap':>8}")
    for name, r in out.items():
        import statistics as st
        p = st.mean(r["privileged"]); b = st.mean(r["blind"])
        geom = f"{r['config']['STONE_TOP']}/{r['config']['STONE_GAP_MIN']}+"
        print(f"  {name:<10}{geom:<16}{p:>11.0%}{b:>10.0%}{p - b:>+8.0%}")
    print()
    print("  Information pays only in the middle. Too easy and feeling the")
    print("  ground suffices; too hard and nothing works, which is a broken")
    print("  terrain rather than a discrimination.")
    return out


def focus(a) -> dict:
    """
    The medium cell only, with enough seeds and enough steps.

    The three-difficulty sweep at 400k steps measured convergence noise
    rather than the information effect: five of its six medium runs
    were still climbing when training stopped, and one was falling.
    Quarter-by-quarter they read 0/9/41/67, 0/0/19/30, 0/8/98/95 for
    the privileged arm -- those are not converged policies, they are
    snapshots taken at arbitrary points on a rising curve.

    So spend the compute where it can answer something: one cell, more
    seeds, and long enough that the curve has flattened.
    """
    import statistics as st

    cfg = DIFFICULTIES["medium"]
    out = {"config": cfg, "steps": a.steps, "privileged": [], "blind": []}
    for seed in range(a.seeds):
        for arm, flag in (("privileged", []), ("blind", ["--no-privileged"])):
            run = f"stf_{arm}_s{seed}"
            if not (RUNS / run / "summary.json").exists():
                print(f"  training {run}  ({a.steps:,} steps)")
                sh([sys.executable, "train_teacher.py", "--stones",
                    "--terrain", "stones", "--total-steps", str(a.steps),
                    "--name", run, "--seed", str(seed), "--quiet"] + flag, cfg)
            want = 15 if arm == "blind" else len(STONE_KINDS) + 17
            got = json.loads((RUNS / run / "summary.json").read_text())["obs_dim"]
            assert got == want, f"{run}: obs_dim {got}, expected {want}"
            out[arm].append(final(run))

    p_, b_ = out["privileged"], out["blind"]
    print()
    print("=" * 74)
    print(f"  MEDIUM CELL — {a.seeds} seeds, {a.steps:,} steps")
    print("=" * 74)
    print(f"  privileged  {st.mean(p_):5.0%}   {[f'{v:.0%}' for v in p_]}")
    print(f"  blind       {st.mean(b_):5.0%}   {[f'{v:.0%}' for v in b_]}")
    diffs = [x - y for x, y in zip(p_, b_)]
    print(f"  gap         {st.mean(diffs):+5.0%}   per seed {[f'{d:+.0%}' for d in diffs]}")
    print(f"  all seeds agree on sign: {all(d > 0 for d in diffs) or all(d < 0 for d in diffs)}")
    return out


def fov_sweep(a) -> dict:
    """
    Wider camera, better policy?

    Run at the `medium` difficulty, because that is the only cell where
    the sensor can matter at all -- measuring field of view where the
    task is already solved blind would report zero for every width and
    mean nothing.

    Each field of view needs its OWN dataset: the depth image is what
    changes, so the frames have to be re-rendered. That is the expensive
    part, ~9 min each.
    """
    cfg = DIFFICULTIES["medium"]
    teacher = f"st_medium_privileged_s0"
    if not (RUNS / teacher / "summary.json").exists():
        print(f"  need {teacher} first; run --difficulty")
        return {}

    out = {}
    for fov in FOVS:
        data = f"data/stones_fov{fov}.npz"
        tag = f"st_fov{fov}"
        if not (HERE / data).exists():
            print(f"  rendering {data}")
            sh([sys.executable, "collect.py", "--teacher", teacher,
                "--episodes", str(a.episodes), "--out", data,
                "--fovy", str(fov), "--kinds", "stones"], cfg)
        if not (RUNS / tag / "summary.json").exists():
            print(f"  training {tag}")
            sh([sys.executable, "train_student.py", "--data", data,
                "--tag", tag, "--device", a.device, "--episodes", "24",
                "--eval-kind", "stones", "--fovy", str(fov)], cfg)
        s = json.loads((RUNS / tag / "summary.json").read_text())
        out[str(fov)] = s["rollout"]["vision"].get("stones")

    print()
    print("=" * 74)
    print("  FIELD-OF-VIEW SWEEP — medium difficulty, stepping stones")
    print("=" * 74)
    for fov, rate in out.items():
        bar = "#" * int(round((rate or 0) * 40))
        print(f"  {fov:>4}deg  {rate:>5.0%}  {bar}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--difficulty", action="store_true")
    ap.add_argument("--focus", action="store_true",
                    help="the medium cell only, more seeds, longer runs")
    ap.add_argument("--fov", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    a, _ = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    result = {}
    if a.all or a.difficulty:
        result["difficulty"] = difficulty_sweep(a)
    if a.focus:
        result["focus"] = focus(a)
    if a.all or a.fov:
        result["fov"] = fov_sweep(a)
    if not result:
        ap.print_help()
        return 1

    f = RUNS / "stones.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    prev.update(result)
    f.write_text(json.dumps(prev, indent=2))
    print(f"\n  wrote {f.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
