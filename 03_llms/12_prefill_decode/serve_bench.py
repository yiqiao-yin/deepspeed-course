#!/usr/bin/env python3
"""
Measure prefill and decode on a real GPU, then feed the scheduler model.

`scheduler.py` simulates three serving policies over a cost model. This
script supplies that cost model's constants from an actual forward pass of
an actual Qwen, and separately demonstrates the head-of-line stall end to
end so the simulation is not the only evidence.

    uv run serve_bench.py --calibrate          # fit the cost model
    uv run serve_bench.py --demo               # show the stall for real
    uv run serve_bench.py --calibrate --repeats 9

WHY A COST MODEL AT ALL
-----------------------
The interesting regime is tens to hundreds of concurrent streams, which no
single 16 GB card can hold. Measuring the per-token costs takes seconds;
extrapolating them through a scheduler costs nothing. The split is also
honest about which half can be wrong: a bad constant makes the numbers
wrong, a bad scheduler makes the conclusion wrong, and the second is
policed by `tests/test_prefill_decode.py` without any hardware.

ON REPORTING LATENCY HONESTLY
-----------------------------
Every number here is a MEDIAN OF REPEATS AFTER WARMUP, and the spread is
printed next to it. Latency on a GPU is not a scalar: the first call pays
kernel autotuning and allocator growth, clocks drift under thermal load,
and a laptop GPU throttles. A single timed call is a sample from a
distribution with a long left-truncated tail, and quoting it as "the"
latency is how benchmark folklore gets made.

The run therefore prints `n`, the warmup count, and p50/min/max. If the
spread is wide the reader can see that rather than trust a point estimate.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Checks `torch.cuda.is_available()` rather than merely whether
    `nvidia-smi` runs. The first version used nvidia-smi, which reports the
    DRIVER -- so `CUDA_VISIBLE_DEVICES="" python serve_bench.py` sailed past
    the guard and silently benchmarked the CPU, reporting constants that
    describe entirely different hardware. A preflight that cannot fail in
    the situation it exists for is decoration.

    torch is imported here rather than at module scope. transformers and the
    model stay inside the functions below, so a reader without a GPU gets
    this message instead of a CUDA traceback.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU; ALLOW_CPU=1 set, continuing on CPU.")
        print("            The constants will describe a CPU, which is a")
        print("            DIFFERENT machine with a different answer. The")
        print("            README's conclusions do not transfer.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before this measures the wrong thing")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  What this script needs a GPU for: timing prefill and decode")
    print("  on Qwen3-0.6B to fit the cost model scheduler.py simulates.")
    print("\n  WHAT YOU CAN STILL RUN, with no GPU and no download:")
    print("      uv run --no-project python scheduler.py")
    print("      uv run ../../tests/test_prefill_decode.py")
    print("\n  The scheduling BEHAVIOUR -- head-of-line blocking, what")
    print("  chunking bounds, what it costs -- is entirely CPU-testable.")
    print("  Only the constants need hardware.")
    print("\n  To rent one:")
    print("      uv run ../../runpod/runpod_ctl.py run \\")
    print("          03_llms/12_prefill_decode --dry-run --collect \\")
    print("          --wait --terminate --yes")
    print("\n  To force a slow CPU run anyway:")
    print("      ALLOW_CPU=1 uv run serve_bench.py --calibrate")
    print(bar + "\n")
    sys.exit(1)


MODEL = "Qwen/Qwen3-0.6B"


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, repeats: int, warmup: int) -> tuple[float, float, float]:
    """
    Median, min, max of `fn` over `repeats`, after `warmup` untimed calls.

    Returns the median rather than the mean: one thermal stall or one
    allocator growth event drags a mean and leaves a median alone.
    """
    import time
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples), max(samples)


def load(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()
    return tok, model


def calibrate(args: argparse.Namespace) -> None:
    """
    Fit the five constants of `scheduler.CostModel` on this machine.

    Each is isolated by DIFFERENCING two measurements rather than measuring
    it directly, because no single timed call contains only one term.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(device)
    print(f"model   {MODEL}  on {device}")
    if device == "cuda":
        print(f"gpu     {torch.cuda.get_device_name(0)}")
    print(f"method  median of {args.repeats} timed calls "
          f"after {args.warmup} warmup\n")

    def prefill_of(n: int):
        ids = torch.randint(0, 1000, (1, n), device=device)
        def run():
            with torch.no_grad():
                model(ids, use_cache=True)
        return run

    rows = []
    for n in (128, 512, 1024, 2048):
        med, lo, hi = _time(prefill_of(n), args.repeats, args.warmup)
        rows.append((n, med))
        spread = (hi - lo) / med * 100 if med else 0
        print(f"prefill {n:>5} tok   {med * 1e3:8.2f} ms   "
              f"[{lo * 1e3:.2f}, {hi * 1e3:.2f}]  spread {spread:5.1f}%")

    # Two points separate the per-token slope from the fixed pass overhead;
    # the quadratic term is small at these lengths and folded into the slope,
    # which the README states rather than hiding.
    (n1, t1), (n2, t2) = rows[0], rows[-1]
    per_token = (t2 - t1) / (n2 - n1)
    overhead = max(t1 - per_token * n1, 0.0)

    # Decode: one step with a short cache, then with a long one.
    def decode_at(ctx: int):
        ids = torch.randint(0, 1000, (1, ctx), device=device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = torch.randint(0, 1000, (1, 1), device=device)
        def run():
            with torch.no_grad():
                model(nxt, past_key_values=past, use_cache=False)
        return run

    d_short, lo_s, hi_s = _time(decode_at(128), args.repeats, args.warmup)
    d_long, lo_l, hi_l = _time(decode_at(2048), args.repeats, args.warmup)
    print(f"\ndecode @  128 kv   {d_short * 1e3:8.2f} ms   "
          f"[{lo_s * 1e3:.2f}, {hi_s * 1e3:.2f}]")
    print(f"decode @ 2048 kv   {d_long * 1e3:8.2f} ms   "
          f"[{lo_l * 1e3:.2f}, {hi_l * 1e3:.2f}]")

    per_kv = (d_long - d_short) / (2048 - 128)
    base = max(d_short - per_kv * 128, 0.0)

    # A negative marginal cost is not a small number, it is a wrong one:
    # more cache cannot be faster to read. When this fires it means the KV
    # term is below the noise floor of this machine, not that it is zero --
    # and publishing the fitted value would put a negative constant into a
    # teaching cost model.
    #
    # It fired on first run. On an RTX 3080 Ti Laptop at 0.6B the decode
    # step is FLAT at ~36 ms from 128 to 16,384 cached tokens, a 128x range,
    # while run-to-run spread is +-22%. At that scale the model is
    # kernel-launch bound, not bandwidth bound, which is the opposite of the
    # usual summary of decode and worth knowing before quoting the usual
    # summary.
    # Sign alone is the wrong test. A NEGATIVE fit is obviously impossible,
    # but a small POSITIVE one drawn from the same noise is equally
    # meaningless -- and the first version of this guard passed it happily,
    # so consecutive runs of this script disagreed about whether decode is
    # bandwidth-bound. The honest test is whether the effect exceeds the
    # measurement's own scatter: if the two decode medians differ by less
    # than the spread of either, the term is UNRESOLVED at this sample size,
    # whichever way the sign fell.
    noise = max(hi_s - lo_s, hi_l - lo_l)
    signal = abs(d_long - d_short)
    if per_kv <= 0 or signal < noise:
        print()
        print("!" * 70)
        print("MEASUREMENT REFUSED: decode_per_kv_token came out "
              f"{per_kv:.3e} s/token.")
        print(f"  signal (|{d_long * 1e3:.1f} - {d_short * 1e3:.1f}| = "
              f"{signal * 1e3:.1f} ms) vs noise ({noise * 1e3:.1f} ms)")
        if per_kv <= 0:
            print("A negative marginal cost is impossible -- more KV cannot")
            print("be cheaper to read.")
        else:
            print("The fitted effect is smaller than the run-to-run scatter,")
            print("so it is indistinguishable from zero at this sample size.")
        print("Reported as 0 rather than fitted.")
        print()
        print("What that means here: decode is LAUNCH-bound, not")
        print("bandwidth-bound, at this model size and batch. The usual")
        print("'decode is memory-bandwidth bound' holds for large models,")
        print("large batches and fused kernels -- not automatically.")
        print("!" * 70)
        per_kv = 0.0
        base = d_short

    print("\n" + "=" * 70)
    print("CostModel(")
    print(f"    prefill_pass_overhead={overhead:.3e},")
    print(f"    prefill_per_token={per_token:.3e},")
    print(f"    decode_step_base={base:.3e},")
    print(f"    decode_per_kv_token={per_kv:.3e},")
    print(")")
    print("=" * 70)

    ratio = base / per_token if per_token else 0.0
    print(f"\nOne decode step costs about as much as prefilling "
          f"{ratio:.0f} tokens.")
    print("That ratio is the topic: one token of output costs what a")
    print("thousand tokens of input cost, because decode cannot use the")
    print("parallelism prefill saturates.")
    advise(overhead, per_token, base)

    if per_kv > 0:
        print("Here the cost grows with cache size, so decode is")
        print("bandwidth-bound as usually described.")
    else:
        print("NOTE: on THIS machine the cost did not grow with cache size,")
        print("so decode is launch-bound rather than bandwidth-bound. The")
        print("textbook phrasing is a claim about large models and fused")
        print("kernels, and it does not survive being measured here.")


def advise(overhead: float, per_token: float, decode_step: float) -> None:
    """
    Turn the measured constants into a chunk size you can actually set.

    The chunk stops being worth shrinking once it is cheaper than the
    decode iteration it rides with -- below that the stall is floored by
    the decode step and every further split is pure TTFT cost. Setting the
    two equal:

        overhead + per_token * C  =  decode_step
        C* = (decode_step - overhead) / per_token

    C* is directly the knob a real serving stack exposes. In vLLM it is
    `max_num_batched_tokens` (with `--enable-chunked-prefill`), and vLLM's
    own tuning guidance describes exactly the tradeoff measured here:
    smaller values give better inter-token latency because fewer prefills
    interrupt decodes, larger values give better time-to-first-token.
    What this gives you is a way to pick it from your own hardware instead
    of inheriting a default.

    A NEGATIVE OR TINY C* IS THE INTERESTING CASE. It means the fixed cost
    of a forward pass has eaten the decode step, so there is no chunk small
    enough to help and chunking cannot pay on this machine at all. That is
    a diagnosis, not a failure -- it is the signal to reach for
    disaggregation instead.
    """
    print()
    print("-" * 70)
    print("WHAT TO SET, from the constants above")
    print("-" * 70)
    if per_token <= 0:
        print("  per-token cost did not resolve; cannot advise.")
        return

    c_star = (decode_step - overhead) / per_token
    frac = overhead / decode_step if decode_step else float("inf")
    print(f"  C* = (decode_step - pass_overhead) / prefill_per_token")
    print(f"     = ({decode_step * 1e3:.2f} - {overhead * 1e3:.2f}) ms / "
          f"{per_token * 1e6:.1f} us")
    print(f"     = {c_star:.0f} tokens")
    print()
    print(f"  pass overhead is {frac * 100:.0f}% of one decode step")
    if c_star < 256:
        print("  -> chunking has almost no room here. The fixed cost of a")
        print("     forward pass has eaten the decode step. Prefer")
        print("     DISAGGREGATION; chunking will cost TTFT and buy little.")
    else:
        print(f"  -> set vLLM --max-num-batched-tokens near {int(c_star)}")
        print("     (with --enable-chunked-prefill). Smaller hurts TTFT")
        print("     without improving the stall; larger lengthens the stall.")
    print()
    print("  Re-measure on YOUR serving hardware. These constants describe")
    print("  this machine and do not transfer.")


def demo(args: argparse.Namespace) -> None:
    """
    Show the head-of-line stall on real hardware, against a baseline.

    BASELINE: the same two streams decoding with NO long prompt arriving.
    Without it a stall figure means nothing -- there is no way to tell a
    blocked step from an ordinary one. This is the comparison the lab's
    claim rests on, so it is measured rather than assumed.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(device)
    n = args.prompt_len

    stream = torch.randint(0, 1000, (2, 64), device=device)
    with torch.no_grad():
        past = model(stream, use_cache=True).past_key_values
    nxt = torch.randint(0, 1000, (2, 1), device=device)

    def decode_step():
        with torch.no_grad():
            model(nxt, past_key_values=past, use_cache=False)

    undisturbed, lo, hi = _time(decode_step, args.repeats, args.warmup)

    long_ids = torch.randint(0, 1000, (1, n), device=device)

    def blocked():
        with torch.no_grad():
            model(long_ids, use_cache=True)
        decode_step()

    blocked_t, blo, bhi = _time(blocked, args.repeats, args.warmup)

    chunk = args.chunk
    def chunked():
        with torch.no_grad():
            kv = None
            for i in range(0, n, chunk):
                out = model(long_ids[:, i:i + chunk], past_key_values=kv,
                            use_cache=True)
                kv = out.past_key_values
                decode_step()

    worst_chunk = None
    import time
    for _ in range(args.warmup):
        chunked()
    _sync()
    gaps = []
    for _ in range(args.repeats):
        with torch.no_grad():
            kv = None
            for i in range(0, n, chunk):
                t0 = time.perf_counter()
                out = model(long_ids[:, i:i + chunk], past_key_values=kv,
                            use_cache=True)
                kv = out.past_key_values
                decode_step()
                _sync()
                gaps.append(time.perf_counter() - t0)
    worst_chunk = max(gaps)

    print("=" * 74)
    print(f"Head-of-line blocking, measured. prompt={n}, chunk={chunk}")
    print(f"median of {args.repeats} repeats after {args.warmup} warmup")
    print("=" * 74)
    print(f"BASELINE  decode step, nothing else running   "
          f"{undisturbed * 1e3:9.2f} ms   [{lo * 1e3:.1f}, {hi * 1e3:.1f}]")
    print(f"shared    decode step behind a full prefill   "
          f"{blocked_t * 1e3:9.2f} ms   [{blo * 1e3:.1f}, {bhi * 1e3:.1f}]")
    print(f"chunked   worst gap with chunked prefill      "
          f"{worst_chunk * 1e3:9.2f} ms")
    print()
    print(f"shared  stalls the stream {blocked_t / undisturbed:5.1f}x "
          f"its undisturbed step")
    print(f"chunked stalls it         {worst_chunk / undisturbed:5.1f}x")
    print()
    print("FALSIFIER: if chunked's worst gap were not below shared's, "
          "chunking")
    print("           would not help on this hardware and the lab's claim "
          "would")
    print("           be wrong here. Observed: "
          f"{'HOLDS' if worst_chunk < blocked_t else 'FAILS'}.")
    print()
    print("SCOPE: this measures UNFUSED chunking -- the chunk and the decode")
    print("       run as two forward passes. A real stack (Sarathi-Serve,")
    print("       vLLM) puts them in ONE batched iteration, so the decode")
    print("       rides along for free. These numbers are therefore an")
    print("       UPPER BOUND on the stall; scheduler.py models the fused")
    print("       version and reports lower. Do not compare the two")
    print("       directly without reading both.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true",
                    help="fit the scheduler's cost model on this GPU")
    ap.add_argument("--demo", action="store_true",
                    help="measure the head-of-line stall end to end")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=7,
                    help="timed calls per measurement (median is reported)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed calls first; the first is always slowest")
    ap.add_argument("--dry-run", action="store_true",
                    help="smallest possible run: 3 repeats, 1 warmup, short "
                         "prompt. Proves the model loads and the harness "
                         "works without spending a cluster allocation.")
    args, _ = ap.parse_known_args()

    if not (args.calibrate or args.demo):
        ap.error("choose --calibrate or --demo")
    if args.dry_run:
        # Cut REPEATS, not prompt length. The claim is about long prompts,
        # and the first --dry-run capped the prompt at 512 -- two chunks --
        # where chunking cannot help and the falsifier printed FAILS. A
        # smoke test that reports the headline claim as false is worse than
        # no smoke test.
        args.repeats, args.warmup = 3, 1
        args.prompt_len = min(args.prompt_len, 2048)
    if args.repeats < 3:
        ap.error("--repeats below 3 cannot show a spread; use at least 3")

    require_gpu()
    if args.calibrate:
        calibrate(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
