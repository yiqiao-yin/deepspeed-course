"""
Riemann-Siegel Z(t) on a GPU, in whatever precision you ask for.

Z(t) is real-valued and |Z(t)| = |zeta(1/2 + it)|, so the zeros of zeta on
the critical line are exactly the sign changes of Z. That turns "find the
zeros" into "evaluate a function on a grid and look for crossings", which
is an embarrassingly parallel workload -- and the reason this lab exists.

The cost per evaluation is the thing to notice:

    N(t) = floor(sqrt(t / 2pi))     terms in the main sum

so t = 1e6 needs 398 terms, t = 1e10 needs 39,894, and the work grows as
sqrt(t). Evaluating thousands of grid points at once is a (points x N)
elementwise cosine followed by a reduction.

This module is deliberately free of experiment logic so the same Z can be
reused by the benchmark, the figures and the tests.
"""

from __future__ import annotations

import math

import torch

# 2*pi shows up in almost every formula below.
TAU = 2.0 * math.pi


def theta(t: torch.Tensor) -> torch.Tensor:
    """
    Riemann-Siegel theta, via its asymptotic expansion.

    theta(t) = arg(Gamma(1/4 + it/2)) - (t/2) log(pi)

    Computing that directly needs a complex log-gamma. The expansion below
    is accurate to O(t^-5) and is what every practical implementation uses:

        theta(t) ~ (t/2) log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760 t^3)

    The 1/(48t) and 7/(5760 t^3) corrections matter only at small t; by
    t = 1000 they are below 1e-5 and 1e-12 respectively.
    """
    return (
        (t / 2.0) * torch.log(t / TAU)
        - t / 2.0
        - math.pi / 8.0
        + 1.0 / (48.0 * t)
        + 7.0 / (5760.0 * t ** 3)
    )


def terms_at(t: float) -> int:
    """How many terms the main sum needs at height t. Grows as sqrt(t)."""
    return int(math.sqrt(t / TAU))


def mean_zero_spacing(t: float) -> float:
    """
    Average gap between consecutive zeros at height t.

    From the Riemann-von Mangoldt formula, the density of zeros near t is
    log(t/2pi) / 2pi, so the mean spacing is its reciprocal. At t = 1e6
    that is 0.52; at t = 1e10 it is 0.30. It shrinks, but only
    logarithmically -- which is why the float32 wall below is so abrupt by
    comparison.
    """
    return TAU / math.log(t / TAU)


def z_values(
    grid,
    dtype: torch.dtype,
    device: str,
    chunk: int = 4096,
) -> torch.Tensor:
    """
    Evaluate Z(t) over `grid` in `dtype`, returning float64 on the CPU.

    The grid is built in float64 and cast down per chunk, so the ONLY
    difference between arms is the precision of the arithmetic -- and of
    the input itself, which turns out to be the dominant effect. See
    `resolvable()`.

    Returned as float64 regardless of the working dtype so that sign
    changes are counted identically across arms; the comparison must not
    depend on how the result is stored.
    """
    out = []
    for start in range(0, len(grid), chunk):
        block = grid[start : start + chunk]
        t = torch.as_tensor(block, device=device, dtype=torch.float64).to(dtype)

        # Each point sums a different number of terms. Build the widest and
        # mask the tail rather than looping -- the mask costs one multiply
        # and keeps the whole chunk a single kernel.
        n_terms = torch.floor(torch.sqrt(t / TAU))
        widest = int(n_terms.max().item())
        n = torch.arange(1, widest + 1, device=device, dtype=dtype)

        angle = theta(t).unsqueeze(1) - t.unsqueeze(1) * torch.log(n).unsqueeze(0)
        keep = (n.unsqueeze(0) <= n_terms.unsqueeze(1)).to(dtype)
        main = 2.0 * ((torch.cos(angle) * n.rsqrt().unsqueeze(0)) * keep).sum(dim=1)

        # First Riemann-Siegel remainder term. Without it the error is
        # O(t^-1/4) -- large enough to move a sign change.
        frac = torch.sqrt(t / TAU) - n_terms
        c0 = torch.cos(TAU * (frac * frac - frac - 1.0 / 16.0)) / torch.cos(TAU * frac)
        remainder = torch.pow(-1.0, n_terms - 1) * torch.pow(t / TAU, -0.25) * c0

        out.append((main + remainder).double().cpu())
    return torch.cat(out)


def count_sign_changes(z: torch.Tensor) -> int:
    """Zeros found on this grid. Each sign change brackets one zero."""
    return int((z[:-1] * z[1:] < 0).sum())


def resolvable(t: float, dtype: torch.dtype) -> float:
    """
    How many representable values of `dtype` fit between adjacent zeros.

    This is the quantity that actually decides whether a precision can
    work at a given height, and it is NOT about rounding error in the sum.
    A float has a fixed number of mantissa bits, so the gap between
    consecutive representable numbers (the ULP) grows with magnitude. Once
    the ULP exceeds the mean zero spacing, distinct grid points collapse
    onto the same value and the function is sampled at one t however many
    points you ask for.

    Below 1.0 the height is unreachable in that dtype, no matter how good
    the algorithm is.
    """
    import numpy as np

    as_np = {torch.float32: np.float32, torch.float64: np.float64}[dtype]
    ulp = float(np.spacing(as_np(t)))
    return mean_zero_spacing(t) / ulp


# ---------------------------------------------------------------------------
# The explicit formula.
#
# Everything above answers "where are the zeros". This answers "why does
# anyone want them", and it is pure mathematics -- no GPU, no drawing. It
# lives here rather than in make_figures.py so the tests can exercise it
# without importing matplotlib, and so the figure script stays a figure
# script.
# ---------------------------------------------------------------------------
import numpy as np

def find_zeros(n_want: int, t_max: float = 460.0, step: float = 0.02):
    """
    The first `n_want` zero heights, found with THIS LAB's own Z(t).

    Sign change, then bisection -- the same "evaluate on a grid and look for
    crossings" the whole lab is built on, which is the point of using our Z
    here rather than importing a table. Both the scan and the bisection are
    vectorised: one call for the entire grid, then one call per bisection
    step for ALL brackets at once, rather than 200 x 50 single-point calls.
    """
    grid = np.arange(10.0, t_max, step)
    z = z_values(grid, torch.float64, "cpu").numpy()
    idx = np.nonzero(z[:-1] * z[1:] < 0)[0][:n_want]

    lo, hi = grid[idx], grid[idx + 1]
    f_lo = z[idx]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f_mid = z_values(mid, torch.float64, "cpu").numpy()
        left = f_lo * f_mid < 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi)


def psi_exact(xs, x_max):
    """
    The prime staircase, weighted: psi(x) = sum of log p over prime POWERS
    p^k <= x.

    Why psi and not pi(x): same information, far tidier formula. Counting
    each prime as 1 gives an explicit formula cluttered with Mobius terms;
    counting it as log p gives the clean one used below. The staircase still
    jumps at exactly the primes (and their powers) -- only the step heights
    differ.
    """
    sieve = np.ones(int(x_max) + 1, bool)
    sieve[:2] = False
    for i in range(2, int(x_max ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False

    jumps = []
    for p in np.nonzero(sieve)[0]:
        q = int(p)
        while q <= x_max:
            jumps.append((q, math.log(int(p))))
            q *= int(p)

    out = np.zeros_like(xs)
    for pos, amt in sorted(jumps):
        out[xs >= pos] += amt
    return out


def psi_from_zeros(xs, ts):
    """
    Riemann's explicit formula -- an EQUALITY, not an approximation:

        psi(x) = x - SUM x^rho/rho - log(2pi) - (1/2)log(1 - x^-2)
                      rho

    Each zero rho = 1/2 + i*t contributes one WAVE. Its real part sets the
    amplitude (x^(1/2), i.e. sqrt(x)) and its imaginary part the frequency.
    Zeros are conjugate pairs, so a pair contributes 2*Re(x^rho/rho) and the
    result is real.

    This is why the zeros matter at all: they are not a curiosity about a
    complex function, they are the frequencies of the primes. Truncating the
    sum is the only approximation here.
    """
    out = xs - math.log(TAU) - 0.5 * np.log1p(-xs ** -2.0)
    if len(ts):
        rho = 0.5 + 1j * np.asarray(ts)
        out = out - 2.0 * np.real(np.exp(rho * np.log(xs)[:, None]) / rho).sum(axis=1)
    return out
