#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.0", "numpy>=1.24", "mpmath>=1.3"]
# ///
"""
02_intermediate/06_riemann_scaling -- properties, not shapes.

CPU only, no GPU, no downloads.

Every bug this lab could ship would RUN FINE and be quietly wrong: a Z(t)
missing its remainder term still returns plausible oscillating floats, and
a reach table off by a factor of two still prints a tidy table. A shape
assertion passes on all of them. So the checks here are:

  1. Z(t) against mpmath's INDEPENDENT siegelz implementation. This is the
     one that matters -- it is the difference between "a function that
     oscillates" and "the Riemann-Siegel Z function". Scored by the ORDER
     of agreement (error * t^0.75 bounded), because zeta_core truncates
     after the first remainder correction C0 and therefore converges as
     O(t^-3/4) BY DESIGN. A flat tolerance would either demand a different
     algorithm or, loosened until it passed, assert nothing.
  2. |Z(t)| == |zeta(1/2 + it)|, the identity the whole lab rests on.
  3. The remainder term is load-bearing -- deleting it must measurably
     break Z. A test that passes with the term removed proves nothing.
  4. The reach claim: float32 must be shown UNREACHABLE at t=1e10 and fine
     at t=1e3, with the crossing where the README says it is.
  5. Every number published in the README is recomputed here, so the prose
     cannot drift away from the code.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _srcload import REPO_ROOT, Results  # noqa: E402

LAB = REPO_ROOT / "02_intermediate" / "06_riemann_scaling"
sys.path.insert(0, str(LAB))

import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from zeta_core import (  # noqa: E402
    TAU, count_sign_changes, mean_zero_spacing, resolvable, terms_at, theta,
    z_values,
)


def main() -> int:
    r = Results("02_intermediate/06_riemann_scaling")

    # ---- 1. Z(t) against an independent implementation --------------------
    # mpmath.siegelz is written by someone else from the same mathematics,
    # so agreeing with it exercises theta, the main sum, the term count and
    # the remainder all at once.
    mpmath.mp.dps = 30
    probes = [50.0, 200.0, 1000.0, 5000.0, 20_000.0, 100_000.0, 1e6]
    errs = {}
    for t in probes:
        mine = float(z_values(np.array([t]), torch.float64, "cpu")[0])
        errs[t] = abs(mine - float(mpmath.siegelz(t)))

    # The ORDER of agreement, not a fixed tolerance.
    #
    # zeta_core truncates the Riemann-Siegel expansion after the first
    # remainder correction C0, for which theory predicts an error of
    # O(t^-3/4). So demanding a flat 1e-6 everywhere would be demanding a
    # different algorithm, and loosening the bound until it passes would
    # assert nothing at all. Pinning the SCALING tests the thing that can
    # actually be wrong: get C0 wrong or drop it and the error degrades to
    # O(t^-1/4), which this bound rejects by three orders of magnitude at
    # the top of the range.
    scaled = {t: e * t ** 0.75 for t, e in errs.items()}
    worst_scaled = max(scaled.values())
    r.check(worst_scaled < 0.1,
            f"Z(t) error scales as t^-3/4 as the C0 truncation predicts "
            f"(worst err*t^0.75 = {worst_scaled:.4f})",
            "A Z that oscillates is not necessarily THE Z.")

    # And in absolute terms over the range the lab actually measures.
    worst_abs = max(e for t, e in errs.items() if t >= 1000)
    r.check(worst_abs < 1e-3,
            f"Z(t) matches mpmath.siegelz to 1e-3 for t >= 1e3 "
            f"(worst {worst_abs:.2e})")

    # The error must genuinely SHRINK with height, or "asymptotic" is a
    # word rather than a property.
    r.check(errs[1e6] < errs[1000.0] < errs[50.0],
            f"the error falls monotonically with t "
            f"({errs[50.0]:.1e} -> {errs[1000.0]:.1e} -> {errs[1e6]:.1e})")

    # ---- 2. the identity the lab rests on ---------------------------------
    # |Z(t)| = |zeta(1/2 + it)| is WHY sign changes of Z locate zeros of
    # zeta. If this fails, counting crossings measures nothing.
    # Scored by the SAME t^-3/4 criterion as above rather than a flat
    # tolerance. A fixed bound here is arbitrary -- 1e-3 failed at t=400 by
    # 3%, and the honest fix is the criterion the method actually obeys,
    # not a number chosen because it passes.
    worst_id = 0.0
    for t in [400.4, 5000.5, 50_000.5]:
        mine = abs(float(z_values(np.array([t]), torch.float64, "cpu")[0]))
        ref = abs(complex(mpmath.zeta(mpmath.mpc(0.5, t))))
        worst_id = max(worst_id, abs(mine - ref) * t ** 0.75)
    r.check(worst_id < 0.1,
            f"|Z(t)| == |zeta(1/2+it)| to the C0 order "
            f"(worst err*t^0.75 = {worst_id:.4f})",
            "This identity is WHY sign changes of Z locate zeros of zeta.")

    # ---- 3. known zeros are bracketed -------------------------------------
    # The first few nontrivial zeros are among the best-known constants in
    # analytic number theory. Z must change sign across each.
    known = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    bracketed = 0
    for z0 in known:
        pair = z_values(np.array([z0 - 0.01, z0 + 0.01]), torch.float64, "cpu")
        if float(pair[0]) * float(pair[1]) < 0:
            bracketed += 1
    r.check(bracketed == len(known),
            f"all {len(known)} known low zeros are bracketed by a sign change")

    # ---- 4. the remainder term is load-bearing ----------------------------
    # Watch the check fail before trusting it: if Z were accurate WITHOUT
    # the Riemann-Siegel remainder, check 1 would be proving nothing about
    # it. Recompute the main sum alone and confirm it is measurably worse.
    t = 1000.5
    tt = torch.tensor([t], dtype=torch.float64)
    n_terms = int(math.sqrt(t / TAU))
    n = torch.arange(1, n_terms + 1, dtype=torch.float64)
    main_only = float(2.0 * (torch.cos(theta(tt) - tt * torch.log(n))
                             * n.rsqrt()).sum())
    ref = float(mpmath.siegelz(t))
    full = float(z_values(np.array([t]), torch.float64, "cpu")[0])
    r.check(abs(main_only - ref) > 100 * abs(full - ref),
            f"dropping the remainder makes Z measurably wrong "
            f"({abs(main_only - ref):.2e} vs {abs(full - ref):.2e})",
            "If this passes without the remainder, check 1 is vacuous.")

    # ---- 5. the reach claim, which is the lab's headline ------------------
    r.check(resolvable(1e3, torch.float32) > 1000,
            "float32 richly resolves t=1e3")
    r.check(resolvable(1e10, torch.float32) < 1.0,
            "float32 CANNOT resolve t=1e10 -- under one point per zero",
            "This is the lab's central claim.")
    r.check(resolvable(1e10, torch.float64) > 1e4,
            "float64 still resolves t=1e10 comfortably")

    ulp32 = float(np.spacing(np.float32(1e10)))
    gap = mean_zero_spacing(1e10)
    r.check(abs(ulp32 - 1024.0) < 1e-9,
            f"float32 ULP at t=1e10 is exactly 1024 (README quotes 1024)")
    r.check(abs(gap - 0.2965) < 5e-4,
            f"mean zero spacing at t=1e10 is {gap:.4f} (README quotes 0.2965)")

    # The crossing the reach figure annotates.
    cross = next(t for t in np.logspace(5, 9, 4000)
                 if np.spacing(np.float32(t)) > mean_zero_spacing(t))
    r.check(1e6 < cross < 1e7,
            f"float32 runs out between 1e6 and 1e7 (found {cross:.2e}; "
            "README says ~4e6)")

    # ---- 6. term count and spacing formulas -------------------------------
    for t, want in [(1e3, 12), (1e6, 398), (1e10, 39894)]:
        r.check(terms_at(t) == want,
                f"N({t:.0e}) == {want:,} terms (README table)")
    r.check(terms_at(4e6) // terms_at(1e6) == 2,
            "term count grows as sqrt(t): 4x the height is 2x the terms")

    # ---- 7. sign counting ------------------------------------------------
    # A counter that is merely plausible is the easiest thing here to get
    # wrong, so give it inputs whose answers are known by inspection.
    r.check(count_sign_changes(torch.tensor([1.0, -1.0, 1.0, -1.0])) == 3,
            "count_sign_changes: alternating signs")
    r.check(count_sign_changes(torch.tensor([1.0, 2.0, 3.0])) == 0,
            "count_sign_changes: no crossing")
    r.check(count_sign_changes(torch.tensor([1.0, 0.0, -1.0])) == 0,
            "count_sign_changes: an exact zero is not double-counted")

    # ---- 8. README numbers must match the code ----------------------------
    # A published claim and the configuration it was measured in are one
    # indivisible fact. These pin the prose to something recomputable.
    readme = (LAB / "README.md").read_text()
    for token in ("1024", "0.2965", "39,894", "33.84", "1.00", "2.39",
                  "1.47", "18.282", "9.7"):
        r.check(token in readme, f"README still quotes {token}")
    r.check("**Baseline:**" in readme and "**Budget:**" in readme,
            "README carries the Tier-3 baseline and budget lines")
    r.check("Rao" in readme,
            "README credits Ashwin Rao, whose notebook this builds on")

    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
