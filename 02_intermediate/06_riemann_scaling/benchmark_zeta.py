"""
Where precision runs out on the critical line, and what fp64 costs to buy.

Two questions, measured rather than assumed:

  1. REACH.  At what height t does each precision stop being able to find
     the zeros of zeta at all? The answer is not "the error gets worse" --
     it is a wall, and it arrives earlier than rounding analysis suggests.

  2. PRICE.  fp64 is the slow path on a consumer GPU. How slow, for THIS
     workload, against GPU fp32 and against a CPU doing the same fp64?

Mathematical basis: Jeffrey Lagarias and, for the framing of RH as
something you can compute with, Ashwin Rao's notebook "Understanding the
Riemann Hypothesis" (Stanford). Credited in full in the README. This lab
contributes no new mathematics; it asks what the arithmetic costs.

Usage
-----
    uv run python benchmark_zeta.py --plan                 # CPU, no GPU
    uv run python benchmark_zeta.py --max-heights 4        # quick GPU run
    uv run python benchmark_zeta.py                        # full sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Heights swept by both experiments. 1e3 is comfortably inside every
# precision; 1e10 is far outside float32. The interesting transition sits
# between 1e5 and 1e7 and is sampled by every decade in between.
HEIGHTS = [10.0 ** m for m in range(3, 11)]

# Zeros per window. 400 is enough that a handful of missed or spurious
# crossings is unambiguous rather than noise, and small enough that the
# largest height still runs in seconds.
ZEROS_PER_WINDOW = 400
POINTS_PER_WINDOW = 20_000


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Checks `torch.cuda.is_available()` rather than whether `nvidia-smi`
    runs: the latter reports the DRIVER, so `CUDA_VISIBLE_DEVICES=""`
    sails past it and silently benchmarks the CPU, which for this lab
    would report timings from entirely different hardware under a GPU
    heading.

    torch is imported here, not at module scope, so a reader without a
    GPU gets this message rather than a CUDA traceback.

    Set ALLOW_CPU=1 to bypass -- the reach experiment is still meaningful
    on a CPU, only the throughput numbers become uninteresting.
    """
    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. From this folder:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU. ALLOW_CPU=1 set, continuing on CPU.")
        print("            The reach experiment is unaffected; the")
        print("            throughput comparison will have no GPU rows.\n")
        return

    print("\n[preflight] No CUDA GPU visible, and this lab measures GPU")
    print("            throughput, so the numbers would be meaningless.\n")
    print("  What you can still do, with no GPU at all:")
    print("      uv run python benchmark_zeta.py --plan")
    print("        the precision-reach table, which is pure arithmetic")
    print("      ALLOW_CPU=1 uv run python benchmark_zeta.py --max-heights 4")
    print("        the full experiment on CPU, minus the GPU timings\n")
    print("  To rent a GPU (24 GB is ample; this lab is not memory-bound):")
    print("      uv run runpod/runpod_ctl.py recommend 02_intermediate/06_riemann_scaling")
    print("      uv run runpod/runpod_ctl.py run 02_intermediate/06_riemann_scaling \\")
    print("           --wait --collect --terminate --yes\n")
    sys.exit(1)


def plan() -> dict:
    """
    The reach table. Pure float arithmetic -- no GPU, no torch kernels.

    This is the part of the lab that needs no hardware, and it is also
    the part with the sharper finding, so it is what --plan prints.
    """
    import numpy as np

    from zeta_core import mean_zero_spacing, terms_at

    rows = []
    for t in HEIGHTS:
        spacing = mean_zero_spacing(t)
        row = {"height": t, "terms": terms_at(t), "mean_zero_spacing": spacing}
        for name, np_type in (("float32", np.float32), ("float64", np.float64)):
            ulp = float(np.spacing(np_type(t)))
            row[name] = {"ulp": ulp, "points_per_zero": spacing / ulp}
        rows.append(row)
    return {"reach": rows}


def print_plan(data: dict) -> None:
    print("\n  PRECISION REACH -- can the dtype even represent the grid?")
    print("  A float's spacing (ULP) grows with magnitude. Once the ULP")
    print("  exceeds the gap between zeros, distinct grid points collapse")
    print("  onto one value and the zeros become invisible.\n")
    print(f"  {'height':>9} {'terms':>8} {'zero gap':>10} "
          f"{'fp32 ULP':>11} {'fp32 pts/zero':>14} {'fp64 pts/zero':>14}")
    for r in data["reach"]:
        f32 = r["float32"]["points_per_zero"]
        mark = "" if f32 > 10 else ("  marginal" if f32 > 1 else "  UNREACHABLE")
        print(f"  {r['height']:>9.0e} {r['terms']:>8,} "
              f"{r['mean_zero_spacing']:>10.4f} {r['float32']['ulp']:>11.4g} "
              f"{f32:>14.1f} {r['float64']['points_per_zero']:>14.3e}{mark}")
    print("\n  PASS: reach table computed. GPU measurements: NOT PERFORMED.")
    print("  Run without --plan on a CUDA machine for zero counts and timings.\n")


def experiment_zeros(heights, device) -> list[dict]:
    """Count zeros at each height in each precision. float64 is the reference."""
    import numpy as np
    import torch

    from zeta_core import count_sign_changes, mean_zero_spacing, terms_at, z_values

    rows = []
    for t0 in heights:
        spacing = mean_zero_spacing(t0)
        grid = np.linspace(t0, t0 + ZEROS_PER_WINDOW * spacing, POINTS_PER_WINDOW)

        z64 = z_values(grid, torch.float64, device)
        z32 = z_values(grid, torch.float32, device)
        n64, n32 = count_sign_changes(z64), count_sign_changes(z32)

        rows.append({
            "height": t0,
            "terms": terms_at(t0),
            "window_zeros_expected": ZEROS_PER_WINDOW,
            "zeros_float64": n64,
            "zeros_float32": n32,
            "difference": n32 - n64,
            "max_abs_deviation": float((z64 - z32).abs().max()),
        })
        r = rows[-1]
        print(f"  {t0:>9.0e} {r['terms']:>8,} {n64:>11,} {n32:>11,} "
              f"{r['difference']:>+7,} {r['max_abs_deviation']:>12.3e}")
    return rows


def experiment_throughput(heights, device) -> list[dict]:
    """
    Time the same work in GPU fp64, GPU fp32 and CPU fp64.

    Three repeats, minimum taken. One timed call is a sample, not a
    measurement, and the minimum is the one least polluted by other load
    on the machine.
    """
    import numpy as np
    import torch

    from zeta_core import mean_zero_spacing, terms_at, z_values

    def timed(fn, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        return best

    rows = []
    for t0 in heights:
        spacing = mean_zero_spacing(t0)
        grid = np.linspace(t0, t0 + 200 * spacing, 8_000)

        if device == "cuda":
            z_values(grid[:512], torch.float64, device)   # warm the kernels

        g64 = timed(lambda: z_values(grid, torch.float64, device)) if device == "cuda" else None
        g32 = timed(lambda: z_values(grid, torch.float32, device)) if device == "cuda" else None
        c64 = timed(lambda: z_values(grid, torch.float64, "cpu"), repeats=2)

        row = {
            "height": t0, "terms": terms_at(t0), "points": len(grid),
            "gpu_float64_s": g64, "gpu_float32_s": g32, "cpu_float64_s": c64,
            "gpu_over_cpu_float64": (c64 / g64) if g64 else None,
            "float32_speedup_on_gpu": (g64 / g32) if g64 and g32 else None,
        }
        rows.append(row)
        if g64:
            print(f"  {t0:>9.0e} {row['terms']:>8,} {g64:>11.4f} {g32:>11.4f} "
                  f"{c64:>11.4f} {row['gpu_over_cpu_float64']:>11.2f}x "
                  f"{row['float32_speedup_on_gpu']:>10.2f}x")
        else:
            print(f"  {t0:>9.0e} {row['terms']:>8,} {'-':>11} {'-':>11} {c64:>11.4f}")
    return rows


def matmul_reference(device) -> dict | None:
    """
    fp32 vs fp64 on a matmul, for contrast.

    Included because the headline "consumer GPUs do fp64 at 1/N of fp32"
    is a statement about matmul throughput, and this lab's workload is
    not a matmul. Measuring both in one run is what makes the difference
    legible instead of arguable.
    """
    import torch

    if device != "cuda":
        return None

    def bench(dtype, n=3072, iters=20):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        return 2 * n ** 3 / ((time.perf_counter() - start) / iters) / 1e12

    f32, f64 = bench(torch.float32), bench(torch.float64)
    return {"matmul_fp32_tflops": f32, "matmul_fp64_tflops": f64,
            "matmul_fp32_over_fp64": f32 / f64}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true",
                    help="CPU-only reach table; performs no GPU measurement")
    ap.add_argument("--max-heights", type=int, default=len(HEIGHTS),
                    help="use only the first N heights (cheap dry run)")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    data = plan()
    if args.plan:
        print_plan(data)
        return

    require_gpu()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heights = HEIGHTS[: args.max_heights]

    print(f"\n  device: {device}"
          f"{'  ' + torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    print(f"  torch {torch.__version__}   heights: {len(heights)}\n")

    print_plan(data)

    print("  ZERO COUNTS -- float64 is the reference; float32 is the arm\n")
    print(f"  {'height':>9} {'terms':>8} {'fp64 zeros':>11} {'fp32 zeros':>11} "
          f"{'diff':>7} {'max|dZ|':>12}")
    data["zeros"] = experiment_zeros(heights, device)

    print("\n  THROUGHPUT -- same work, three ways\n")
    print(f"  {'height':>9} {'terms':>8} {'GPU fp64':>11} {'GPU fp32':>11} "
          f"{'CPU fp64':>11} {'gpu/cpu':>12} {'fp32 gain':>11}")
    data["throughput"] = experiment_throughput(heights, device)

    data["matmul_reference"] = matmul_reference(device)
    data["environment"] = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (torch.cuda.get_device_name(0).replace(" ", "_") if device == "cuda" else "cpu")
    path = out_dir / f"{tag}_{stamp}.json"
    path.write_text(json.dumps(data, indent=2))

    # A short run is a short run, not a failure. Say which happened.
    if args.max_heights < len(HEIGHTS):
        print(f"\n  NOTE: capped at {args.max_heights} of {len(HEIGHTS)} heights")
        print("        (--max-heights). A full sweep reaches t = 1e10, where")
        print("        float32 finds no zeros at all.")
    print(f"\n  wrote {path}\n")


if __name__ == "__main__":
    main()
