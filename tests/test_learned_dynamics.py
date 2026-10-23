#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "torch==2.11.0"]
# ///
"""
Topic 8 Lab 6 — a scaling claim is only as good as its null and its optimiser.

This lab's claim is that a dynamics model for the cold-gas car does
not improve with size, so the one model in topic 8 big enough to
need ZeRO does not deserve it. Several things would make that claim
look true while being an artifact, and each is pinned here.

  1. THE NULL MUST BE MEASURED, NOT ASSUMED. At 100 Hz "predict no
     change" is already a good one-step predictor. Quoting 0.026
     without the null's 0.219 beside it turns a 1.4x edge into
     apparent mastery.

  2. ONE-STEP ERROR IS NOT PREDICTION. The headline metric feeds the
     model its OWN output, so errors compound. A rollout that
     secretly used ground truth at each step would make every model
     look excellent and the lab's finding would evaporate.

  3. THE MODEL PREDICTS A DELTA AND STARTS AT ZERO. An untrained
     model must predict NO CHANGE, which is the right prior at this
     rate and what keeps early epochs from diverging.

  4. A SCALING LADDER MUST ACTUALLY SCALE, and every rung must see
     the same budget. A ladder whose big rungs also got more epochs
     is not a scaling comparison.

Properties, not shapes. No GPU, no training, no downloads.

    uv run tests/test_learned_dynamics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "08_aerial_physics" / "06_learned_dynamics"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import torch

    import dynamics as DY

    print("=" * 76)
    print("  TOPIC 8 LAB 6 — learned dynamics")
    print("=" * 76)

    OBS, ACT = 41, 10

    # ------------------------------------------------------------------
    # 1. An untrained model predicts NO CHANGE.
    # ------------------------------------------------------------------
    print("\n  The right prior at 100 Hz is 'nothing happens'")
    for kind in ("mlp", "gru"):
        m = DY.build(kind, "small", OBS, ACT)
        o = torch.randn(4, OBS)
        u = torch.rand(4, ACT)
        with torch.no_grad():
            p, _ = m(o, u, None)
        check(f"{kind}: an untrained model returns the state unchanged",
              torch.allclose(p, o, atol=1e-6),
              f"max |delta| {float((p - o).abs().max()):.2e} — the head is "
              f"zero-initialised, so the first epochs cannot diverge")

    # It must not STAY the identity once the head has weights.
    m = DY.build("mlp", "small", OBS, ACT)
    with torch.no_grad():
        m.head.weight.normal_(0.0, 0.1)
        p, _ = m(o, u, None)
    check("...but it is not hard-wired to the identity",
          not torch.allclose(p, o, atol=1e-6),
          "or the model could never learn anything")

    # ------------------------------------------------------------------
    # 2. The ladder actually scales, and the rungs are ordered.
    # ------------------------------------------------------------------
    print("\n  The ladder must span enough to test the bar")
    sizes = list(DY.SIZES)
    counts = {k: DY.n_params(DY.build("mlp", k, OBS, ACT)) for k in sizes}
    ordered = [counts[k] for k in sizes]
    check("every rung is larger than the one below it",
          all(a < b for a, b in zip(ordered, ordered[1:])),
          " < ".join(f"{c:,}" for c in ordered))
    span = ordered[-1] / ordered[0]
    check("the ladder spans at least 1000x, or the bar is untestable",
          span >= 1000.0, f"{span:,.0f}x from {ordered[0]:,} to {ordered[-1]:,}")
    check("an unknown size is REJECTED, not silently defaulted",
          _raises(lambda: DY.build("mlp", "enormous", OBS, ACT)))
    check("an unknown architecture is REJECTED too",
          _raises(lambda: DY.build("transformer", "small", OBS, ACT)))

    # ------------------------------------------------------------------
    # 3. The memory table is arithmetic, and says what it excludes.
    # ------------------------------------------------------------------
    print("\n  Weights plus Adam state, per card")
    rows = {(r["kind"], r["size"]): r for r in DY.memory_table(OBS, ACT)}
    big = rows[("gru", "xxl")]
    check("Adam's floor is 16 bytes per parameter",
          abs(big["gb_1card"] - big["params"] * 16 / 1e9) < 1e-9,
          f"{big['params']:,} params -> {big['gb_1card']:.2f} GB "
          f"(param + grad + two moments, fp32)")
    check("ZeRO-3 across 8 cards is one eighth of that",
          abs(big["gb_zero3"] - big["gb_1card"] / 8) < 1e-9,
          f"{big['gb_zero3']:.2f} GB")
    check("ZeRO-2 shards the gradient and moments but NOT the parameter",
          big["gb_zero2"] > big["gb_zero3"],
          f"{big['gb_zero2']:.2f} GB vs {big['gb_zero3']:.2f} GB — the "
          f"parameter copy stays whole on every card")
    check("the largest rung would not fit a 16 GB card with activations",
          big["gb_1card"] > 10.0,
          f"{big['gb_1card']:.2f} GB before a single activation; measured "
          f"peak was 15.97 GB of 16.38")

    # ------------------------------------------------------------------
    # 4. The rollout metric must feed the model its OWN output.
    # ------------------------------------------------------------------
    print("\n  The headline metric must compound its own error")
    src = (LAB / "train_dyn.py").read_text()
    fn = src[src.index("def rollout_error"):src.index("def main(")]
    check("the rollout loop reassigns the state from the prediction",
          "o, state = model(o, u, state)" in fn,
          "feeding ground truth back each step would make every model "
          "look excellent and erase the lab's finding")
    # ASK THE AST, not the text. The first version counted the
    # substring "truth" and expected 2; the docstring says the word
    # as well, so a correct implementation failed. A substring is
    # not a fact about the program.
    import ast

    tree = ast.parse(src)
    rollout = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "rollout_error")
    inner = [n for n in ast.walk(rollout)
             if isinstance(n, ast.For)
             and any(isinstance(t, ast.Name) and t.id == "h"
                     for t in ast.walk(n.target))]
    assigned_inside = {
        t.id
        for loop in inner
        for n in ast.walk(loop)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    check("...and never reads the truth INSIDE the horizon loop",
          "truth" not in assigned_inside and bool(inner),
          f"the step loop assigns {sorted(assigned_inside)} — ground "
          f"truth entering here would silently reset the compounding")
    check("the horizon is a parameter, not a magic number",
          "horizon" in fn)

    # A tiny end-to-end check that compounding actually happens: a
    # model with a constant non-zero delta must drift linearly.
    class Drift(torch.nn.Module):
        def forward(self, obs, act, state=None):
            return obs + 0.1, None

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_dyn", LAB / "train_dyn.py")
    td = importlib.util.module_from_spec(spec)
    sys.modules["train_dyn"] = td
    spec.loader.exec_module(td)

    T = 60
    obs = np.zeros((T, OBS), dtype=np.float32)
    act = np.zeros((T, ACT), dtype=np.float32)
    e10 = td.rollout_error(Drift(), obs, act, np.array([0]), np.array([T]),
                           10, "cpu", n_traj=1)
    e20 = td.rollout_error(Drift(), obs, act, np.array([0]), np.array([T]),
                           20, "cpu", n_traj=1)
    check("a constant-drift model's error GROWS with the horizon",
          e20 > e10 * 1.8,
          f"{e10:.2f} at 10 steps, {e20:.2f} at 20 — compounding, "
          f"not a fixed offset")

    # ------------------------------------------------------------------
    # 5. The published numbers, recomputed from the shipped runs.
    # ------------------------------------------------------------------
    print("\n  The published claim, against the runs on disk")
    import json

    runs = []
    for d in sorted((LAB / "runs").glob("*/")):
        s = d / "summary.json"
        if s.exists():
            runs.append(json.loads(s.read_text()))
    if not runs:
        print("    (no runs on disk — runs/ is gitignored; skipping)")
    else:
        base = [r for r in runs
                if abs(r["lr"] - 1e-3) < 1e-12 and not r["deepspeed"]
                and r["kind"] == "mlp"]
        base.sort(key=lambda r: r["params"])
        if len(base) >= 2:
            grew = base[-1]["params"] / base[0]["params"]
            cut = base[0]["rollout_err"] / base[-1]["rollout_err"]
            check("the ladder grew by at least 1000x", grew >= 1000.0,
                  f"{grew:,.0f}x")
            check("the BAR (1000x must halve the error) is NOT met "
                  "— this lab's finding",
                  cut < 2.0,
                  f"error moved {base[0]['rollout_err']:.3f} -> "
                  f"{base[-1]['rollout_err']:.3f} ({cut:.2f}x)")
            best = min(base, key=lambda r: r["rollout_err"])
            check("...and the best rung is a SMALL one",
                  best["params"] < 1e6,
                  f"{best['params']:,} params at "
                  f"{best['rollout_err']:.3f}")
        ds = [r for r in runs if r["deepspeed"]]
        plain = [r for r in runs if r["name"] == "mlp_xxl_lr1e-4"]
        if ds and plain:
            check("ZeRO at world size 1 did NOT save memory",
                  ds[0]["peak_gpu_gb"] > plain[0]["peak_gpu_gb"],
                  f"{ds[0]['peak_gpu_gb']:.2f} GB vs "
                  f"{plain[0]['peak_gpu_gb']:.2f} GB — there is no second "
                  f"card to partition with")
            check("...and did not change the accuracy either",
                  abs(ds[0]["rollout_err"] - plain[0]["rollout_err"]) < 0.2,
                  f"{ds[0]['rollout_err']:.3f} vs "
                  f"{plain[0]['rollout_err']:.3f} — ZeRO is a memory "
                  f"optimisation, not a numerical one")

    print("\n" + "=" * 76)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 76)
    return 1 if FAILURES else 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
