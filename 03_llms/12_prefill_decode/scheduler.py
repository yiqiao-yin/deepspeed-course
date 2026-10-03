#!/usr/bin/env python3
"""
Three ways to schedule prefill and decode on the same GPU, as a cost model.

THE PROBLEM
-----------
An LLM serving request has two phases that want opposite things from hardware:

  PREFILL  reads the whole prompt at once. Every token is available, so the
           attention and MLP work becomes big dense GEMMs. It SATURATES
           compute and is over in one forward pass.

  DECODE   emits one token at a time. Each step does a tiny amount of
           arithmetic but must stream the entire KV cache and every weight
           out of HBM. It is MEMORY-BANDWIDTH bound and leaves the compute
           units mostly idle.

Put them on one worker and they interfere. A long prompt's prefill occupies
the GPU for a long, indivisible stretch, and every live decode stream stalls
behind it -- classic head-of-line blocking. The user watching a stream does
not see "someone else's prompt is long"; they see their own tokens stop.

THE THREE POLICIES
------------------
  shared        one worker, first-come-first-served. A prefill runs to
                completion while decodes wait. Simplest; worst tail latency.

  chunked       split the prefill into chunks of `chunk` tokens and run one
                chunk per iteration ALONGSIDE the active decodes
                (Sarathi-Serve, Agrawal et al., OSDI 2024). Decode never
                waits longer than one chunk.

  disaggregated run prefill and decode on separate pools and ship the KV
                cache between them (DistServe, Zhong et al., OSDI 2024;
                Splitwise, Patel et al., ISCA 2024). No interference at all,
                paid for with a cache transfer and a second pool.

WHAT THIS MODULE IS, AND IS NOT
-------------------------------
It is a discrete-event simulator over a COST MODEL, not a kernel benchmark.
It runs on CPU in milliseconds, which is what makes the behaviour testable:
`serve_bench.py` measures the three cost constants on a real GPU with a real
Qwen, and this module then answers "what happens at 200 concurrent streams"
without renting 200 streams' worth of hardware.

The division matters because the two parts fail differently. A wrong constant
makes the numbers wrong; a wrong scheduler makes the CONCLUSION wrong, and
only the second is a teaching error. The properties in
`tests/test_prefill_decode.py` police the second.

THE INVARIANT THAT MATTERS
--------------------------
Chunking bounds the decode stall by the CHUNK SIZE, not by the prompt
length. That is the whole claim, and it is why the test checks two prompt
lengths rather than one: under `shared` the stall grows with the prompt,
under `chunked` it does not move. A single-length test passes on an
implementation that has no chunking at all.

References
----------
Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with
    Sarathi-Serve", OSDI 2024. https://arxiv.org/abs/2403.02310
Zhong et al., "DistServe: Disaggregating Prefill and Decoding for
    Goodput-optimized LLM Serving", OSDI 2024. https://arxiv.org/abs/2401.09670
Kwon et al., "Efficient Memory Management for LLM Serving with
    PagedAttention", SOSP 2023. https://arxiv.org/abs/2309.06180
Yu et al., "Orca: A Distributed Serving System for Transformer-Based
    Generative Models", OSDI 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# The cost model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Seconds per unit of work, as measured by `serve_bench.py --calibrate`.

    The defaults are a PLACEHOLDER shape, not a measurement of any GPU. They
    exist so the simulator runs without hardware; every published number in
    the README comes from calibrated constants, and the README says which
    GPU they came from. Inventing plausible-looking defaults and then
    quoting them would be exactly the failure this repository keeps
    rediscovering.

    prefill_per_token
        Compute-bound and roughly linear in tokens at these lengths.
    prefill_attn_per_token2
        The quadratic attention term. Small until prompts get long, and the
        reason a 8k prompt is not 8x a 1k prompt.
    decode_step_base
        Fixed cost of one decode iteration: weight streaming, kernel launch.
        Paid once per iteration no matter how many sequences are batched,
        which is the entire reason batching decode works.
    decode_per_kv_token
        Marginal cost of one more cached token in the batch. This is the
        memory-bandwidth term.
    kv_transfer_per_token
        Cost of shipping one token of KV between pools. Only `disaggregated`
        pays it.
    """

    prefill_pass_overhead: float = 3e-3
    prefill_per_token: float = 60e-6
    prefill_attn_per_pair: float = 16e-9
    decode_step_base: float = 9e-3
    decode_per_kv_token: float = 1.2e-6
    kv_transfer_per_token: float = 4e-6

    def prefill_chunk(self, chunk: int, already: int) -> float:
        """
        Cost of one prefill chunk, given `already` tokens already cached.

        Three terms, and getting the split right took two attempts:

        1. a fixed per-PASS overhead -- every forward pass streams the whole
           weight matrix out of HBM whether it carries one token or a
           thousand. This is why chunking costs more: m chunks pay it m
           times.
        2. a linear per-token term, the MLP and projections.
        3. the attention term, counted in query-key PAIRS under a causal
           mask. Chunk `[already, already+chunk)` attends to everything
           before it plus itself: `chunk*already + chunk*(chunk+1)/2`.

        The first version charged attention as `chunk * (already + chunk)`
        and defined full prefill as `n * n`, which made a chunked prompt
        measurably CHEAPER than the same prompt in one pass -- the model
        disagreed with itself about identical physical work, and it
        flattered chunking in exactly the direction the lab is arguing.
        With pairs counted correctly the attention cost is IDENTICAL either
        way, which is the honest result: chunking does not change the
        arithmetic, it changes how many times you pay to read the weights.
        """
        if chunk <= 0:
            return 0.0
        pairs = chunk * already + chunk * (chunk + 1) / 2
        return (self.prefill_pass_overhead
                + self.prefill_per_token * chunk
                + self.prefill_attn_per_pair * pairs)

    def prefill(self, n_tokens: int) -> float:
        """
        Cost of prefilling `n_tokens` in a single pass.

        Defined as the one-chunk case so the two can never drift apart.
        """
        return self.prefill_chunk(n_tokens, 0)

    def decode_step(self, kv_tokens: int, batch: int) -> float:
        """One decode iteration over a batch holding `kv_tokens` total."""
        if batch <= 0:
            return 0.0
        return self.decode_step_base + self.decode_per_kv_token * kv_tokens


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """One inference request."""

    rid: str
    arrival: float
    prompt_len: int
    output_len: int

    # filled in by the simulator
    first_token_at: float | None = None
    done_at: float | None = None
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        """Time to first token, from arrival. The prefill-side metric."""
        if self.first_token_at is None:
            raise ValueError(f"{self.rid} never produced a token")
        return self.first_token_at - self.arrival

    def itls(self) -> list[float]:
        """
        Inter-token latencies. The decode-side metric a reader actually feels.

        TTFT is how long until text appears; ITL is whether it then flows or
        stutters. Head-of-line blocking shows up here, not in the mean.
        """
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


@dataclass
class Trace:
    """What a policy did, and what it cost."""

    policy: str
    requests: list[Request]
    makespan: float
    prefill_tokens_processed: int
    decode_steps: int
    gpu_seconds: float
    # (start, end, kind, who) for every iteration the GPU ran. Recorded so
    # the timing diagram in the docs is GENERATED from a trace rather than
    # drawn by hand -- a hand-drawn schematic can illustrate a policy the
    # code does not implement, which is the diagram equivalent of a stale
    # measurement.
    events: list[tuple[float, float, str, str]] = field(default_factory=list)

    def ttfts(self) -> list[float]:
        return [r.ttft for r in self.requests]

    def all_itls(self) -> list[float]:
        out: list[float] = []
        for r in self.requests:
            out.extend(r.itls())
        return out

    def p(self, values: list[float], q: float) -> float:
        """
        Percentile by nearest-rank. No numpy, so the module stays dependency
        free and the arithmetic is inspectable.
        """
        if not values:
            raise ValueError("no values to take a percentile of")
        s = sorted(values)
        k = max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))
        return s[k]

    def summary(self) -> dict[str, float]:
        itls = self.all_itls()
        return {
            "ttft_p50": self.p(self.ttfts(), 50),
            "ttft_p99": self.p(self.ttfts(), 99),
            "itl_p50": self.p(itls, 50) if itls else 0.0,
            "itl_p99": self.p(itls, 99) if itls else 0.0,
            "makespan": self.makespan,
            "gpu_seconds": self.gpu_seconds,
        }


# ---------------------------------------------------------------------------
# The policies
# ---------------------------------------------------------------------------

def _active(reqs: list[Request], now: float) -> list[Request]:
    return [r for r in reqs if r.arrival <= now and r.done_at is None]


def simulate(policy: str, requests: list[Request], cost: CostModel,
             chunk: int = 512) -> Trace:
    """
    Run `requests` under `policy` and return what happened.

    Policies: "shared", "chunked", "disaggregated".

    The three share a clock and a cost model so their outputs are
    comparable; that comparability is the point of putting them in one
    function rather than three.
    """
    if policy not in {"shared", "chunked", "disaggregated"}:
        raise ValueError(f"unknown policy {policy!r}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")

    reqs = [Request(r.rid, r.arrival, r.prompt_len, r.output_len)
            for r in requests]
    for r in reqs:
        if r.prompt_len <= 0 or r.output_len <= 0:
            raise ValueError(f"{r.rid}: prompt and output must be positive")

    if policy == "disaggregated":
        return _disaggregated(reqs, cost)
    return _colocated(reqs, cost, chunk, chunked=(policy == "chunked"))


def _colocated(reqs: list[Request], cost: CostModel, chunk: int,
               chunked: bool) -> Trace:
    """
    One worker runs both phases. `chunked` decides whether a prefill may be
    interleaved with the decodes or must run to completion.
    """
    now = 0.0
    gpu = 0.0
    prefill_done: dict[str, int] = {r.rid: 0 for r in reqs}
    emitted: dict[str, int] = {r.rid: 0 for r in reqs}
    kv: dict[str, int] = {r.rid: 0 for r in reqs}
    steps = 0
    tokens_prefilled = 0
    events: list[tuple[float, float, str, str]] = []

    pending = sorted(reqs, key=lambda r: (r.arrival, r.rid))

    while any(r.done_at is None for r in reqs):
        live = _active(reqs, now)
        if not live:
            # Idle until the next arrival rather than spinning.
            future = [r.arrival for r in pending if r.arrival > now
                      and r.done_at is None]
            if not future:
                break
            now = min(future)
            continue

        needs_prefill = [r for r in live if prefill_done[r.rid] < r.prompt_len]
        decoding = [r for r in live if prefill_done[r.rid] >= r.prompt_len]

        if needs_prefill:
            victim = needs_prefill[0]
            remaining = victim.prompt_len - prefill_done[victim.rid]
            take = min(chunk, remaining) if chunked else remaining

            dt = cost.prefill_chunk(take, prefill_done[victim.rid])
            if chunked and decoding:
                # Sarathi-Serve's stall-free schedule: the chunk and the
                # decodes share one iteration, so the decodes advance.
                kv_total = sum(kv[r.rid] for r in decoding)
                dt = max(dt, cost.decode_step(kv_total, len(decoding)))
                t0 = now
                now += dt
                gpu += dt
                steps += 1
                events.append((t0, now, "fused",
                               f"chunk+{len(decoding)} decodes"))
                for r in decoding:
                    emitted[r.rid] += 1
                    kv[r.rid] += 1
                    r.token_times.append(now)
                    if r.first_token_at is None:
                        r.first_token_at = now
                    if emitted[r.rid] >= r.output_len:
                        r.done_at = now
            else:
                t0 = now
                now += dt
                gpu += dt
                events.append((t0, now, "prefill", victim.rid))

            prefill_done[victim.rid] += take
            tokens_prefilled += take
            kv[victim.rid] = prefill_done[victim.rid]
            continue

        # Pure decode iteration over everything that is ready.
        kv_total = sum(kv[r.rid] for r in decoding)
        dt = cost.decode_step(kv_total, len(decoding))
        t0 = now
        now += dt
        gpu += dt
        steps += 1
        events.append((t0, now, "decode", f"{len(decoding)} streams"))
        for r in decoding:
            emitted[r.rid] += 1
            kv[r.rid] += 1
            r.token_times.append(now)
            if r.first_token_at is None:
                r.first_token_at = now
            if emitted[r.rid] >= r.output_len:
                r.done_at = now

    return Trace("chunked" if chunked else "shared", reqs, now,
                 tokens_prefilled, steps, gpu, events)


def _disaggregated(reqs: list[Request], cost: CostModel) -> Trace:
    """
    Separate pools. Prefill never competes with decode for the same GPU.

    Modelled as two independent single-worker queues joined by a KV
    transfer. That is the honest minimum: a real deployment sizes the pools
    independently, which is DistServe's actual contribution, but modelling
    one worker each already shows the interference disappearing.
    """
    prefill_free = 0.0
    tokens_prefilled = 0
    gpu = 0.0
    events: list[tuple[float, float, str, str]] = []
    ready: list[tuple[float, Request]] = []

    for r in sorted(reqs, key=lambda r: (r.arrival, r.rid)):
        start = max(prefill_free, r.arrival)
        dt = cost.prefill(r.prompt_len)
        prefill_free = start + dt
        gpu += dt
        tokens_prefilled += r.prompt_len
        events.append((start, prefill_free, "prefill", f"pool-P {r.rid}"))

        transfer = cost.kv_transfer_per_token * r.prompt_len
        gpu += transfer
        events.append((prefill_free, prefill_free + transfer, "transfer",
                       f"KV {r.rid}"))
        ready.append((prefill_free + transfer, r))

    # Decode pool: one worker, continuous batching.
    now = 0.0
    steps = 0
    emitted: dict[str, int] = {r.rid: 0 for r in reqs}
    kv: dict[str, int] = {r.rid: r.prompt_len for r in reqs}
    arrive = dict((r.rid, t) for t, r in ready)

    while any(r.done_at is None for r in reqs):
        batch = [r for r in reqs
                 if r.done_at is None and arrive[r.rid] <= now]
        if not batch:
            nxt = [arrive[r.rid] for r in reqs if r.done_at is None]
            if not nxt:
                break
            now = min(nxt)
            continue

        kv_total = sum(kv[r.rid] for r in batch)
        dt = cost.decode_step(kv_total, len(batch))
        t0 = now
        now += dt
        gpu += dt
        steps += 1
        events.append((t0, now, "decode", f"pool-D {len(batch)} streams"))
        for r in batch:
            emitted[r.rid] += 1
            kv[r.rid] += 1
            r.token_times.append(now)
            if r.first_token_at is None:
                r.first_token_at = now
            if emitted[r.rid] >= r.output_len:
                r.done_at = now

    return Trace("disaggregated", reqs, max(now, prefill_free),
                 tokens_prefilled, steps, gpu, events)


# ---------------------------------------------------------------------------
# The scenario the whole lab is about
# ---------------------------------------------------------------------------

def head_of_line_scenario(prompt_len: int, cost: CostModel,
                          n_streams: int = 2, output_len: int = 96,
                          interrupt_at: float = 0.10) -> list[Request]:
    """
    Live streams, then one long prompt arrives mid-flight.

    This is the picture the lab exists to explain: `n_streams` users are
    already receiving tokens when a long prompt lands. Under `shared` their
    output stops dead for the whole prefill. The only variable that should
    change the size of that stall is `prompt_len`, which is exactly what
    the tests exploit.

    **It refuses to build a scenario where nothing collides**, and that
    guard is here because the first version of this function shipped one.
    With `output_len=24` the streams finished at 0.228 s and the prompt
    arrived at 0.25 s, so there was no interference to observe -- yet all
    three policies ran, produced plausible latencies, and reported an
    IDENTICAL maximum stall. The conclusion a reader would have drawn,
    "chunked prefill changes nothing", was an artifact of a scenario that
    never exercised the thing being compared.

    A simulator that can be misconfigured into measuring nothing must say
    so rather than return a number.
    """
    if n_streams < 1:
        raise ValueError(f"need at least one live stream, got {n_streams}")

    stream_prompt = 64
    reqs = [Request(f"stream{i}", 0.0, stream_prompt, output_len)
            for i in range(n_streams)]
    reqs.append(Request("longprompt", interrupt_at, prompt_len, output_len))

    # Will the incumbents still be streaming when the prompt lands? Estimate
    # with the cheapest possible decode step; a real step only costs more,
    # so this under-estimates the lifetime and errs toward raising.
    step = cost.decode_step(stream_prompt * n_streams, n_streams)
    lifetime = output_len * step
    if lifetime <= interrupt_at * 1.5:
        raise ValueError(
            f"vacuous scenario: {n_streams} streams finish after about "
            f"{lifetime:.3f}s but the long prompt does not arrive until "
            f"{interrupt_at:.3f}s, so no stall can occur and every policy "
            f"would score the same. Raise output_len (now {output_len}) or "
            f"lower interrupt_at.")
    return reqs


def max_decode_stall(trace: Trace) -> float:
    """
    The longest gap any already-streaming request saw between its tokens.

    This is the number a user feels, and the one the three policies actually
    differ on. Requests named `stream*` are the incumbents; the long prompt
    is excluded because its own first gap is not a stall.
    """
    gaps = [g for r in trace.requests if r.rid.startswith("stream")
            for g in r.itls()]
    return max(gaps) if gaps else 0.0


def to_mermaid_gantt(traces: list[Trace], max_bars: int = 14) -> str:
    """
    Render traces as a mermaid gantt — GENERATED, not drawn.

    The point of generating it is that a hand-drawn schematic can illustrate
    a policy the code does not implement, and nothing would catch the
    disagreement. Here the bars come from `Trace.events`, so the picture is
    a view of the same run the tables report.

    Consecutive iterations of the same kind are coalesced into one bar,
    because a real trace has ~110 events and a legible diagram has a dozen.
    `max_bars` then truncates, and the truncation is LABELLED rather than
    silent — a chart that quietly drops half a timeline is a chart that
    lies about where the time went.

    Colour via gantt's own tags, since gantt ignores classDef:
    `crit` for prefill (the thing that blocks), `active` for decode,
    `done` for a KV transfer.
    """
    TAG = {"prefill": "crit", "decode": "active", "fused": "active",
           "transfer": "done"}

    out = ["gantt",
           "    title Prefill and decode on one timeline (milliseconds)",
           "    dateFormat x",
           "    axisFormat %L"]

    for t in traces:
        # Coalesce runs of the same kind.
        runs: list[list] = []
        for start, end, kind, _who in t.events:
            if runs and runs[-1][2] == kind and abs(runs[-1][1] - start) < 1e-9:
                runs[-1][1] = end
                runs[-1][3] += 1
            else:
                runs.append([start, end, kind, 1])

        shown, dropped = runs[:max_bars], max(0, len(runs) - max_bars)
        out.append(f"    section {t.policy}")
        for i, (start, end, kind, n) in enumerate(shown):
            ms0, ms1 = int(round(start * 1000)), int(round(end * 1000))
            if ms1 <= ms0:
                ms1 = ms0 + 1          # gantt refuses a zero-width bar
            label = kind if n == 1 else f"{kind} x{n}"
            out.append(f"    {label} :{TAG[kind]}, {t.policy[:3]}{i}, "
                       f"{ms0}, {ms1}")
        if dropped:
            last = shown[-1][1] if shown else 0.0
            ms0 = int(round(last * 1000))
            out.append(f"    ...{dropped} more :done, {t.policy[:3]}x, "
                       f"{ms0}, {ms0 + 1}")
    return "\n".join(out)


def main() -> None:
    """Print the comparison the README quotes, with the baseline first."""
    cost = CostModel()
    print("=" * 78)
    print("Prefill/decode scheduling — PLACEHOLDER constants, not a measurement")
    print("Run `serve_bench.py --calibrate` for numbers from your own GPU.")
    print("=" * 78)

    for prompt_len in (1024, 8192):
        print(f"\nLong prompt = {prompt_len} tokens, 2 live streams")
        print(f"{'policy':<16}{'TTFT p99':>10}{'ITL p99':>10}"
              f"{'max stall':>11}{'GPU s':>9}")
        print("-" * 56)
        for policy in ("shared", "chunked", "disaggregated"):
            t = simulate(policy, head_of_line_scenario(prompt_len, cost),
                         cost, chunk=512)
            s = t.summary()
            print(f"{policy:<16}{s['ttft_p99']:>9.3f}s{s['itl_p99']:>9.3f}s"
                  f"{max_decode_stall(t):>10.3f}s{s['gpu_seconds']:>8.2f}s")

    print("\nBaseline is `shared` — the policy everyone starts with.")
    print("The claim under test: chunking bounds the stall by CHUNK SIZE,")
    print("not by prompt length. Compare the two blocks above.")


if __name__ == "__main__":
    main()
