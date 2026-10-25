#!/usr/bin/env python3
"""
Score the five arms against each other, and against the falsifier.

    uv run compare.py
    uv run compare.py --json compare.json

THE FALSIFIER, WRITTEN DOWN BEFORE ANY ARM FINISHED
----------------------------------------------------
    The distilled student must land on target at least 80% as
    often as its teacher, AND must beat behaviour cloning by more
    than the seed-to-seed spread. If it does not, distillation
    does not rescue perception on this task and the central claim
    of this lab is wrong here.

Printed as HOLDS or FAILS whichever way it falls. Topic 3's
prefill lab wrote its falsifier first, got FAILS on the first dry
run, and documented a whole regime it would otherwise have
missed -- that is the argument for the practice, not a formality.

THE FIVE ARMS
-------------
    oracle    PPO, exact altitude and clearance            the ceiling
    deaf      PPO, no idea how high it is                  the floor
    direct    PPO, 144 rangefinder beams, no altimeter     can RL do it alone?
    bc        regression onto the teacher, teacher-driven  the naive copy
    dagger    the same, iterated on the STUDENT's states   the fix

`oracle` and `deaf` bracket the scale. Without them a student
scoring 44% is unreadable: it could be most of what is achievable
or barely above what you get for free. Topic 7's VLM lab is the
cautionary case -- its sighted arm beat its blind arm with
p = 0.006 and still lost to a one-line heuristic, and reporting
only the first comparison would have been true and misleading.

WHY PAIRED TESTS
----------------
Every arm flies the SAME held-out seeds, so the episodes come in
matched tuples: same terrain, same mark, same launch. An unpaired
test throws that away and needs far more episodes to see the same
effect. `assert_same_seeds()` refuses to report if the arms did
not in fact fly the same course -- a comparison between two
different sets of launches is not a comparison, and nothing in
the numbers would look wrong.

McNemar's exact test is the right one for paired binary outcomes:
it conditions on the DISCORDANT pairs, the episodes where one arm
landed and the other did not, and asks whether the split between
them is fairer than a coin. Episodes both arms won, or both lost,
carry no information about which is better and are correctly
ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).parent

# (arm name, run-directory prefix, what it knows, its role).
# The prefix is separate from the name because `train_policy.py`
# names its runs after the SENSE (`onboard_s0`) while the lab talks
# about the METHOD (`direct`). The first version assumed they were
# the same string and silently reported the arm as "(not run)" --
# a missing row in a comparison table is much easier to miss than
# a crash.
ARMS = [
    ("oracle", "oracle", "PPO, exact altitude + clearance", "the ceiling"),
    ("direct", "onboard", "PPO, 144 beams, no altimeter", "RL alone"),
    ("bc", "bc", "cloned from the teacher", "naive copy"),
    ("dagger", "dagger", "DAgger from the teacher", "the fix"),
    ("deaf", "deaf", "PPO, no height information", "the floor"),
]
PREFIX = {name: pre for name, pre, _w, _r in ARMS}

# The bar, as a number, so the verdict cannot be argued after the
# fact. See the module docstring.
RETAIN_BAR = 0.80


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval.

    Not the textbook normal approximation: at the rates this lab
    produces -- some arms near 0, some near 0.9 -- the normal
    interval runs off the end of [0, 1] and reports a negative
    lower bound for a count that cannot be negative. Wilson is
    well behaved there and is what the rest of topic 8 quotes.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - half) / d), min(1.0, (c + half) / d))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """
    Exact McNemar on paired binary outcomes. Two-sided.

    `b01` is episodes a lost and b won; `b10` the reverse. Under
    the null the discordant pairs split binomially at p = 1/2, so
    the exact p-value is a two-sided binomial tail -- no chi-square
    approximation, which is unreliable at the handful of
    discordant pairs a 120-episode comparison can produce.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired: {len(a)} vs {len(b)} episodes")
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p": 1.0}
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b01": b01, "b10": b10, "n_discordant": n,
            "p": min(1.0, 2 * tail)}


def load_arm(runs: Path, arm: str) -> dict | None:
    """Pool every seed of one arm, keeping per-episode outcomes."""
    arm_dir = PREFIX.get(arm, arm)
    seeds, hits, miss, n = [], [], [], 0
    rates = []
    for d in sorted(runs.glob(f"{arm_dir}_s*")):
        f = d / "episodes.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        seeds.append(d.name)
        hits.extend(rec["on_target"])
        miss.extend(rec["miss"])
        rates.append(sum(rec["on_target"]) / max(len(rec["on_target"]), 1))
        n += len(rec["on_target"])
    if not seeds:
        return None
    return {"arm": arm, "seeds": seeds, "n": n, "hits": hits,
            "miss": miss, "k": sum(hits),
            "rate": sum(hits) / max(n, 1),
            "per_seed": rates,
            "spread": (max(rates) - min(rates)) if len(rates) > 1 else 0.0}


def assert_same_seeds(runs: Path, arms: list[str]) -> list[int]:
    """
    Every arm must have flown the SAME held-out launches.

    A comparison between arms that saw different terrain is not a
    comparison, and no number in the output would look wrong.
    Lab 4 shipped this check for the same reason and it is cheap.
    """
    ref = None
    for arm in arms:
        for d in sorted(runs.glob(f"{PREFIX.get(arm, arm)}_s*")):
            f = d / "episodes.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())["seeds"]
            if ref is None:
                ref = s
            elif s != ref:
                raise RuntimeError(
                    f"{d.name} flew different held-out seeds than the "
                    f"first arm ({len(s)} vs {len(ref)}, first differing "
                    f"at index {next(i for i, (x, y) in enumerate(zip(s, ref)) if x != y) if len(s) == len(ref) else 'length'}). "
                    f"These arms cannot be paired and an unpaired "
                    f"comparison of them would be meaningless.")
    if ref is None:
        raise RuntimeError("no arm has episodes.json — run evaluate first")
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    runs = HERE / a.runs
    names = [n for n, _p, _w, _r in ARMS]
    seeds = assert_same_seeds(runs, names)
    data = {n: load_arm(runs, n) for n in names}
    have = {k: v for k, v in data.items() if v}

    print("=" * 78)
    print("  THE FIVE ARMS — one task, five ways of knowing where the ground is")
    print("=" * 78)
    print(f"  {len(seeds)} held-out launches per seed, identical across "
          f"arms (seeds {seeds[0]}..{seeds[-1]})")
    print()
    print(f"  {'arm':<9}{'what it knows':<34}{'on target':>11}"
          f"{'95% CI':>16}{'miss':>8}")
    for arm, _pre, what, _role in ARMS:
        d = have.get(arm)
        if not d:
            print(f"  {arm:<9}{what:<34}{'(not run)':>11}")
            continue
        lo, hi = wilson(d["k"], d["n"])
        import statistics
        print(f"  {arm:<9}{what:<34}{d['rate']:>10.1%}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}"
              f"{statistics.fmean(d['miss']):>7.2f} m")

    print("\n  per seed, because a pooled figure hides a real spread:")
    for arm, _pre, _w, _r in ARMS:
        d = have.get(arm)
        if d:
            print(f"    {arm:<9}" + "  ".join(f"{r:.0%}" for r in d["per_seed"])
                  + f"      spread {d['spread']:.0%}")

    # --- the comparisons that matter --------------------------------
    print("\n" + "=" * 78)
    print("  PAIRED COMPARISONS — exact McNemar on discordant pairs")
    print("=" * 78)
    pairs = [("dagger", "bc", "does training on the STUDENT's own states help?"),
             ("dagger", "direct", "is distillation better than RL from the sensor?"),
             ("direct", "deaf", "does the sensor help RL at all?"),
             ("oracle", "dagger", "how much does the student still give up?")]
    tests = {}
    for x, y, q in pairs:
        if x not in have or y not in have:
            continue
        m = mcnemar_exact(have[x]["hits"], have[y]["hits"])
        tests[f"{x}_vs_{y}"] = m
        better = x if m["b10"] > m["b01"] else y
        print(f"\n  {x} vs {y} — {q}")
        print(f"    {x} won {m['b10']}, {y} won {m['b01']}, "
              f"{m['n_discordant']} discordant of {have[x]['n']}")
        print(f"    p = {m['p']:.4f}   "
              f"{'significant, ' + better + ' ahead' if m['p'] < 0.05 else 'NOT significant at 0.05'}")

    # --- the falsifier ----------------------------------------------
    #
    # RETENTION IS MEASURED AGAINST THE TEACHER THAT WAS ACTUALLY
    # USED, not against the pooled oracle arm. Both students were
    # distilled from `oracle_s0`; the pooled oracle averages that
    # seed with `oracle_s1`, which scored 20 points lower and which
    # no student ever saw. Dividing by the pooled figure reported
    # retention of 109% -- a student apparently beating its teacher
    # -- which is an artifact of the denominator and not a result.
    teacher_dir = None
    for arm in ("dagger", "bc"):
        if arm in have:
            d = sorted(runs.glob(f"{arm}_s*"))
            if d:
                teacher_dir = Path(json.loads(
                    (d[0] / "summary.json").read_text())["teacher"]).name
                break
    t_rate = None
    if teacher_dir:
        rec = runs / teacher_dir / "episodes.json"
        if rec.exists():
            hits = json.loads(rec.read_text())["on_target"]
            t_rate = sum(hits) / max(len(hits), 1)

    print("\n" + "=" * 78)
    print("  THE FALSIFIER")
    print("=" * 78)
    print("  Written before any arm finished:")
    print("    the distilled student must reach at least 80% of its")
    print("    teacher's on-target rate, AND beat behaviour cloning by")
    print("    more than the seed spread.")
    verdict = None
    if "dagger" in have and t_rate:
        dag = have["dagger"]["rate"]
        bc_rate = have.get("bc", {"rate": 0.0})["rate"]
        retain = dag / t_rate
        gap = dag - bc_rate
        spread = max(have["dagger"]["spread"],
                     have.get("bc", {"spread": 0.0})["spread"])
        c1 = retain >= RETAIN_BAR
        c2 = gap > spread
        print(f"\n    the teacher actually distilled from is "
              f"{teacher_dir} at {t_rate:.1%}")
        print(f"      (the POOLED oracle arm is "
              f"{have['oracle']['rate']:.1%}; using it here would be "
              f"dividing by a\n       teacher the student never saw)")
        print(f"\n    retention   {dag:.1%} of {t_rate:.1%} = {retain:.0%}"
              f"   (bar {RETAIN_BAR:.0%})        {'PASS' if c1 else 'FAIL'}")
        print(f"    over BC     {gap:+.1%} against a seed spread of "
              f"{spread:.1%}           {'PASS' if c2 else 'FAIL'}")
        verdict = bool(c1 and c2)
        print(f"\n  PRE-REGISTERED VERDICT: {'HOLDS' if verdict else 'FAILS'}")

        # AND THE CLAUSE THAT CANNOT BE TRUSTED, SAID PLAINLY.
        dvb = tests.get("dagger_vs_bc")
        if dvb and spread == 0.0:
            print()
            print("  BUT THE SECOND CLAUSE IS A BAD TEST AND THIS RUN SHOWS")
            print("  WHY. Both distilled arms scored identically on both")
            print("  seeds, so the measured spread is 0.0% and ANY positive")
            print("  margin clears it. The pre-registered bar is satisfiable")
            print("  by noise.")
            print()
            print(f"  The paired test is the one to believe: McNemar on the")
            print(f"  {dvb['n_discordant']} discordant episodes gives "
                  f"p = {dvb['p']:.3f}.")
            if dvb["p"] >= 0.05:
                print("  DAgger's edge over plain cloning is NOT significant")
                print("  here. Reported as measured.")
                verdict = False
                print("\n  CORRECTED VERDICT: the retention clause HOLDS,")
                print("  the beats-cloning clause FAILS.")
    else:
        print("\n  (not enough arms to judge yet)")

    print("=" * 78)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"arms": {k: {kk: vv for kk, vv in v.items() if kk != "hits"}
                      for k, v in have.items()},
             "tests": tests, "verdict": verdict,
             "retain_bar": RETAIN_BAR, "n_seeds": len(seeds)}, indent=1))
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
