#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Prefill/decode scheduling: the properties, not the numbers.

The numbers in this lab depend on a GPU. The BEHAVIOUR does not, and the
behaviour is where a scheduler goes quietly wrong -- it still runs, every
request still finishes, and the conclusion is backwards. These checks assert
relationships that must hold on any hardware.

Two defects were caught writing it, both of which produced plausible output:

  * the head-of-line scenario had the live streams FINISHING 22 ms before
    the long prompt arrived, so no policy could interfere with any other.
    All three reported an identical stall and the lab's entire claim
    evaluated to "makes no difference". The scenario now refuses to build
    itself when nothing can collide, and `test_the_scenario_is_not_vacuous`
    pins that.

  * the cost model priced the same physical work two ways -- full prefill
    charged n^2 attention, the chunk path charged n^2/2 -- which made a
    chunked prompt CHEAPER than the same prompt in one pass. It flattered
    chunking in precisely the direction the lab argues, which is the worst
    possible direction for an error to point.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "03_llms" / "12_prefill_decode"))
sys.path.insert(0, str(Path(__file__).parent))

from _srcload import Results  # noqa: E402
from scheduler import (CostModel, Request, head_of_line_scenario,  # noqa: E402
                       max_decode_stall, simulate)

COST = CostModel()
POLICIES = ("shared", "chunked", "disaggregated")


def test_work_is_conserved(r: Results) -> None:
    """
    A scheduler moves work. It cannot delete it.

    Every policy must prefill exactly the same number of tokens and emit
    exactly the requested outputs. Without this, a "fast" policy that drops
    a chunk or short-changes a stream would look like the winner -- and it
    would win on every latency metric in the lab.
    """
    reqs = head_of_line_scenario(4096, COST)
    want_prefill = sum(x.prompt_len for x in reqs)
    want_tokens = {x.rid: x.output_len for x in reqs}

    for policy in POLICIES:
        t = simulate(policy, reqs, COST, chunk=512)
        r.check(t.prefill_tokens_processed == want_prefill,
                f"{policy}: prefills every prompt token ({want_prefill})",
                f"processed {t.prefill_tokens_processed}")
        emitted = {x.rid: len(x.token_times) for x in t.requests}
        r.check(emitted == want_tokens,
                f"{policy}: every request emits exactly its output_len",
                f"{emitted} != {want_tokens}")
        r.check(all(x.done_at is not None for x in t.requests),
                f"{policy}: every request completes")


def test_chunking_bounds_the_stall_by_chunk_not_prompt(r: Results) -> None:
    """
    THE claim of the lab, and the reason two prompt lengths are tested.

    Under `shared` the decode stall is the whole prefill, so it grows with
    the prompt. Under `chunked` it is one chunk, so it barely moves. A test
    at a single prompt length passes on an implementation with no chunking
    in it at all -- the same trap `test_tmrope.py` documents for frame
    rates.
    """
    stalls = {}
    for policy in POLICIES:
        for n in (1024, 8192):
            t = simulate(policy, head_of_line_scenario(n, COST), COST,
                         chunk=512)
            stalls[(policy, n)] = max_decode_stall(t)

    shared_growth = stalls[("shared", 8192)] / stalls[("shared", 1024)]
    chunk_growth = stalls[("chunked", 8192)] / stalls[("chunked", 1024)]

    r.check(shared_growth > 8.0,
            "shared: the stall grows with the prompt (8x prompt -> >8x stall)",
            f"grew only {shared_growth:.1f}x -- is the prefill blocking at all?")
    r.check(chunk_growth < 4.0,
            "chunked: the stall does NOT track the prompt length",
            f"grew {chunk_growth:.1f}x -- chunking is not bounding anything")
    r.check(shared_growth > 3 * chunk_growth,
            "the two policies scale differently, not just differ",
            f"shared {shared_growth:.1f}x vs chunked {chunk_growth:.1f}x")
    r.check(stalls[("chunked", 8192)] < stalls[("shared", 8192)] / 5,
            "at 8192 tokens chunking cuts the stall by at least 5x",
            f"{stalls[('chunked', 8192)]:.3f}s vs "
            f"{stalls[('shared', 8192)]:.3f}s")


def test_chunking_is_not_free(r: Results) -> None:
    """
    The cost must be visible, or the model is wrong.

    Chunking does not reduce attention arithmetic -- counted in causal
    query-key pairs it is identical either way. What it adds is one extra
    weight-streaming pass per chunk, so m chunks cost exactly
    `(m-1) * prefill_pass_overhead` more than one pass. Asserting the exact
    identity rather than "it is bigger" is what makes this a check on the
    model instead of a tautology.
    """
    n, chunk = 8192, 512
    m = n // chunk
    one_pass = COST.prefill(n)
    chunked = sum(COST.prefill_chunk(chunk, k * chunk) for k in range(m))
    extra = chunked - one_pass

    r.check(extra > 0, "chunked prefill costs MORE than a single pass",
            f"chunked {chunked:.4f}s vs one pass {one_pass:.4f}s")
    r.check(abs(extra - (m - 1) * COST.prefill_pass_overhead) < 1e-9,
            f"the excess is exactly (m-1) weight-streaming passes "
            f"({m - 1} x {COST.prefill_pass_overhead * 1e3:.1f}ms)",
            f"excess was {extra:.6f}s, expected "
            f"{(m - 1) * COST.prefill_pass_overhead:.6f}s")

    smaller = sum(COST.prefill_chunk(256, k * 256) for k in range(n // 256))
    r.check(smaller > chunked,
            "smaller chunks cost more still -- the knob has two ends",
            "halving the chunk should double the pass overhead")


def test_the_chunk_knob_behaves(r: Results) -> None:
    """
    Smaller chunks, smaller stall -- monotonically, and it saturates.

    This replaces a check that asserted `chunk >= prompt_len` reproduces
    `shared` exactly. It does not, and the code was right: Sarathi-Serve
    contributes TWO separable things, chunking *and* stall-free
    co-scheduling, and the second still applies when there is only one
    chunk. The decodes ride along with the single prefill iteration, so the
    limit of `chunked` is "shared with piggybacked decode", measurably
    better than `shared`.

    Worth keeping as a lesson: a limit check is only as good as the limit
    you believe in.
    """
    n = 2048
    reqs = head_of_line_scenario(n, COST)
    stalls = [(c, max_decode_stall(simulate("chunked", reqs, COST, chunk=c)))
              for c in (128, 256, 512, 1024, 2048)]

    rising = all(a[1] < b[1] for a, b in zip(stalls, stalls[1:]))
    r.check(rising,
            "stall rises monotonically with chunk size -- the knob works",
            f"not monotonic: {[(c, round(v, 4)) for c, v in stalls]}")

    saturated = max_decode_stall(simulate("chunked", reqs, COST, chunk=10 ** 6))
    r.check(abs(saturated - stalls[-1][1]) < 1e-9,
            "chunk >= prompt_len saturates: one chunk is one chunk",
            f"{saturated:.6f} vs {stalls[-1][1]:.6f}")

    shared = max_decode_stall(simulate("shared", reqs, COST))
    r.check(saturated < shared,
            "even at one chunk, co-scheduling beats shared",
            f"chunked-at-limit {saturated:.4f}s should be below shared "
            f"{shared:.4f}s -- the decodes still ride along")


def test_disaggregation_removes_interference_and_charges_for_it(
        r: Results) -> None:
    """
    Separate pools cannot interfere -- and that is not free either.

    The stall should collapse to a plain decode step, because nothing else
    runs on the decode pool. The bill arrives as GPU-seconds: a second pool
    plus a KV transfer. A model where disaggregation were free everywhere
    would recommend it unconditionally, which is not what the papers found.
    """
    reqs = head_of_line_scenario(8192, COST)
    shared = simulate("shared", reqs, COST)
    dis = simulate("disaggregated", reqs, COST)

    bare_step = COST.decode_step(64 * 2, 2)
    r.check(max_decode_stall(dis) < bare_step * 2,
            "disaggregated: the stall collapses to a plain decode step",
            f"stall {max_decode_stall(dis):.4f}s vs bare step "
            f"{bare_step:.4f}s")
    r.check(dis.gpu_seconds > shared.gpu_seconds,
            "disaggregated buys that with more GPU-seconds",
            f"{dis.gpu_seconds:.2f}s vs shared {shared.gpu_seconds:.2f}s -- "
            "if it were cheaper on every axis the model would be wrong")


def test_the_scenario_is_not_vacuous(r: Results) -> None:
    """
    The guard that would have caught the first version of this lab.

    `head_of_line_scenario` must refuse to build a case where the live
    streams finish before the long prompt arrives. That configuration runs
    fine, produces plausible latencies for all three policies, and compares
    nothing.
    """
    try:
        head_of_line_scenario(8192, COST, output_len=4, interrupt_at=5.0)
        raised = False
    except ValueError as exc:
        raised = "vacuous" in str(exc)
    r.check(raised,
            "a scenario where nothing can collide is REFUSED",
            "it returned requests instead, which is how the lab first "
            "measured 'chunking makes no difference'")

    # And the default must actually overlap, not merely avoid raising.
    t = simulate("shared", head_of_line_scenario(8192, COST), COST)
    bare = COST.decode_step(64 * 2, 2)
    r.check(max_decode_stall(t) > 10 * bare,
            "the default scenario really does stall the incumbents",
            f"max stall {max_decode_stall(t):.4f}s is not much more than "
            f"one decode step {bare:.4f}s -- nothing is being blocked")


def test_the_timing_diagram_is_generated_from_the_trace(r: Results) -> None:
    """
    The docs diagram must describe the run the tables describe.

    A hand-drawn schematic can illustrate a policy the code does not
    implement and nothing catches the disagreement, so the gantt in the
    book is emitted from `Trace.events`. These checks are on OUR generator,
    not on mermaid: mermaid's gantt grammar turned out to accept
    `dateFormat qqq` and non-numeric dates without complaint, so "it
    parses" is close to no evidence at all. What can be checked is that the
    bars are coherent and that nothing is dropped silently.
    """
    from scheduler import to_mermaid_gantt

    reqs = head_of_line_scenario(2048, COST)
    traces = [simulate(p, reqs, COST, chunk=512) for p in POLICIES]

    r.check(all(t.events for t in traces),
            "every policy records what the GPU actually did",
            "an empty event log draws an empty diagram")

    # A single worker cannot run two things at once. Checking NON-OVERLAP
    # per resource replaces a check that the global event list is in
    # chronological order -- which it is not for `disaggregated`, and the
    # code is right: two pools run concurrently, so one global ordering is
    # meaningless. The KV transfer is a third resource (the link), and it
    # legitimately overlaps the prefill GPU moving on to the next request.
    def resource(kind: str, who: str) -> str:
        if kind == "transfer":
            return "link"
        return "prefill-pool" if who.startswith("pool-P") else "worker"

    for t in traces:
        r.check(all(a < b for a, b, _, _ in t.events),
                f"{t.policy}: every event ends after it starts")
        buckets: dict[str, list] = {}
        for a, b, kind, who in t.events:
            buckets.setdefault(resource(kind, who), []).append((a, b))
        clashes = []
        for name, iv in buckets.items():
            iv.sort()
            clashes += [f"{name} {x}/{y}" for x, y in zip(iv, iv[1:])
                        if y[0] < x[1] - 1e-9]
        r.check(not clashes,
                f"{t.policy}: no resource runs two things at once",
                f"double-booked: {clashes[:2]}")

    # The event log must account for the GPU time the summary reports.
    for t in traces:
        busy = sum(b - a for a, b, _, _ in t.events)
        r.check(abs(busy - t.gpu_seconds) < 1e-6,
                f"{t.policy}: the bars account for all {t.gpu_seconds:.2f}s "
                f"of GPU time",
                f"bars total {busy:.4f}s but the trace reports "
                f"{t.gpu_seconds:.4f}s -- the diagram is hiding work")

    text = to_mermaid_gantt(traces, max_bars=4)
    r.check(text.startswith("gantt"), "the emitter produces a gantt")
    r.check(all(f"section {t.policy}" in text for t in traces),
            "all three policies appear as sections")
    r.check("more" in text,
            "truncation is LABELLED, not silent",
            "with max_bars=4 and ~100 events the chart must say what it "
            "dropped; a chart that quietly omits half a timeline lies "
            "about where the time went")


def test_bad_input_is_refused(r: Results) -> None:
    """Fail loudly: a misconfigured run must not quietly return a number."""
    reqs = [Request("a", 0.0, 128, 8)]
    for kwargs, why in [
        (dict(policy="turbo"), "an unknown policy"),
        (dict(chunk=0), "a zero chunk size"),
        (dict(chunk=-5), "a negative chunk size"),
    ]:
        call = dict(policy="chunked", requests=reqs, cost=COST)
        call.update(kwargs)
        try:
            simulate(**call)
            ok = False
        except ValueError:
            ok = True
        r.check(ok, f"{why} is rejected")

    try:
        simulate("shared", [Request("a", 0.0, 0, 8)], COST)
        ok = False
    except ValueError:
        ok = True
    r.check(ok, "a zero-length prompt is rejected")


def main() -> int:
    r = Results("Prefill/decode scheduling properties")
    test_work_is_conserved(r)
    test_chunking_bounds_the_stall_by_chunk_not_prompt(r)
    test_chunking_is_not_free(r)
    test_the_chunk_knob_behaves(r)
    test_disaggregation_removes_interference_and_charges_for_it(r)
    test_the_scenario_is_not_vacuous(r)
    test_the_timing_diagram_is_generated_from_the_trace(r)
    test_bad_input_is_refused(r)
    return r.finish()


if __name__ == "__main__":
    raise SystemExit(main())
