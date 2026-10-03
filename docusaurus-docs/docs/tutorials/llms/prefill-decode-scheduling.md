---
sidebar_position: 18
---

# Prefill and decode: when to split them

One inference request has two phases that want opposite things from a GPU,
and almost every serving decision follows from that single fact.

**Prefill** reads the whole prompt at once. Every token is available
simultaneously, so attention and the MLP become large dense GEMMs. It
saturates compute and finishes in one forward pass.

**Decode** emits one token at a time. Each step does a trivial amount of
arithmetic but must stream every weight — and the entire KV cache — out of
memory. It cannot use the parallelism prefill feasts on.

Put them on one worker and they interfere. A long prompt's prefill is a
long, indivisible block of GPU time, and every live stream stalls behind it.
The user does not experience this as *"someone else sent a long prompt"*.
They experience their own tokens stopping.

### 1. Shared worker — the baseline

One GPU, first-come-first-served. The prompt's prefill occupies it
completely, and the two live streams stop dead for the whole duration.
That red-flagged break is the entire problem.

```mermaid
flowchart LR
    subgraph REQ["Long prompt arrives"]
        direction LR
        p1["P"] --- p2["P"] --- p3["P"] --- p4["P"] --- p5["P"] --- p6["P"]
    end

    subgraph LIVE["Live streams, already decoding"]
        direction TB
        sA["A"]
        sB["B"]
    end

    subgraph GPU["1 GPU — one job at a time"]
        direction TB
        subgraph BATCH["Prefill batch — all 6 tokens, indivisible"]
            direction LR
            b1["P"] --- b2["P"] --- b3["P"]
            b4["P"] --- b5["P"] --- b6["P"]
        end
        subgraph KVC["KV cache (local)"]
            direction TB
            kP["P rows — being written"]
            kA["A rows"]
            kB["B rows"]
        end
        BATCH --> KVC
    end

    subgraph OUT["Output"]
        direction TB
        oN["New request: nothing yet"]
        oA["A A A A A"]
        oB["B B B B B"]
    end

    p6 --> BATCH
    sA --x|"BLOCKED"| GPU
    sB --x|"BLOCKED"| GPU
    kA --> oA
    kB --> oB
    kP --> oN

    classDef deep   fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef dark   fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef base   fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    classDef steel  fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff

    class REQ,LIVE,GPU,OUT,KVC deep
    class p1,p2,p3,p4,p5,p6,oN,oA,oB base
    class kP,kA,kB steel
    class BATCH bright
    class b1,b2,b3,b4,b5,b6 bright
    class sA,sB base
```

### 2. Chunked prefill — Sarathi-Serve

The prompt is split, and each chunk shares **one batched iteration** with
the live decodes. Nobody waits longer than a single chunk. The prefill's
own first token arrives later, which is the trade.

```mermaid
flowchart LR
    subgraph REQ2["Long prompt, split into chunks"]
        direction TB
        subgraph CK1["chunk 1"]
            direction LR
            c1a["P"] --- c1b["P"]
        end
        subgraph CK2["chunk 2"]
            direction LR
            c2a["P"] --- c2b["P"]
        end
        subgraph CK3["chunk 3"]
            direction LR
            c3a["P"] --- c3b["P"]
        end
    end

    subgraph LIVE2["Live streams, never stop"]
        direction TB
        sA2["A"]
        sB2["B"]
    end

    subgraph GPU2["1 GPU"]
        direction TB
        subgraph MIX["ONE batched iteration — this is the whole idea"]
            direction LR
            mA["A decode"]
            mB["B decode"]
            mP["one P chunk"]
        end
        subgraph KVC2["KV cache (local)"]
            direction TB
            k2P["P rows — grow one chunk at a time"]
            k2A["A rows"]
            k2B["B rows"]
        end
        MIX --> KVC2
    end

    subgraph OUT2["Output"]
        direction TB
        o2N["New request: P P P — arrives later"]
        o2A["A A A … A — small gaps, no stall"]
        o2B["B B B … B"]
    end

    CK1 --> mP
    CK2 --> mP
    CK3 --> mP
    sA2 --> mA
    sB2 --> mB
    k2P --> o2N
    k2A --> o2A
    k2B --> o2B

    classDef deep   fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef dark   fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef base   fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    classDef steel  fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff

    class REQ2,LIVE2,GPU2,OUT2,KVC2 deep
    class CK1,CK2,CK3 dark
    class c1a,c1b,c2a,c2b,c3a,c3b,sA2,sB2,o2N,o2A,o2B base
    class k2P,k2A,k2B steel
    class MIX bright
    class mA,mB,mP bright
```

### 3. Separate prefill and decode — DistServe

Two pools. The prefill GPU and the decode GPU cannot interfere by
construction, and the KV cache is copied between them. The decode pool is
never interrupted, so the streams never stutter — paid for with a second
pool and a transfer.

```mermaid
flowchart LR
    subgraph REQ3["Long prompt"]
        direction LR
        q1["P"] --- q2["P"] --- q3["P"] --- q4["P"] --- q5["P"] --- q6["P"]
    end

    subgraph LIVE3["Live streams"]
        direction TB
        sA3["A"]
        sB3["B"]
    end

    subgraph POOLP["PREFILL POOL — sized for compute"]
        direction TB
        pf3["Prefill — all 6 tokens at once"]
        kc3["KV cache, freshly built"]
        pf3 --> kc3
    end

    subgraph POOLD["DECODE POOL — sized for bandwidth, never interrupted"]
        direction TB
        d3A["A rows"]
        d3B["B rows"]
        d3P["P rows — arrived by copy"]
        dec3["Decode loop"]
        d3A --> dec3
        d3B --> dec3
        d3P --> dec3
    end

    subgraph OUT3["Output"]
        direction TB
        o3N["New request: starts after the copy"]
        o3A["A A A A A A — one more than shared"]
        o3B["B B B B B B"]
    end

    q6 --> pf3
    kc3 ==>|"KV copy over the interconnect"| d3P
    sA3 --> d3A
    sB3 --> d3B
    dec3 --> o3N
    dec3 --> o3A
    dec3 --> o3B

    classDef deep   fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef dark   fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef base   fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    classDef steel  fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff

    class REQ3,LIVE3,OUT3 deep
    class POOLP,POOLD deep
    class q1,q2,q3,q4,q5,q6,sA3,sB3,o3N,o3A,o3B base
    class kc3,d3A,d3B,d3P steel
    class pf3 base
    class dec3 bright
```

## The same run, on one timeline

The diagram above is a schematic. This one is **generated from an actual
trace** — every bar is an iteration the simulator ran, at the constants
measured on the GPU. A 2048-token prompt arrives 100 ms into two live
streams:

```mermaid
gantt
    title Prefill and decode on one timeline (milliseconds)
    dateFormat x
    axisFormat %L
    section shared
    prefill x2 :crit, sha0, 0, 71
    decode :active, sha1, 71, 109
    prefill :crit, sha2, 109, 255
    decode x96 :active, sha3, 255, 3930
    section chunked
    prefill :crit, chu0, 0, 35
    fused :active, chu1, 35, 74
    decode :active, chu2, 74, 112
    fused x4 :active, chu3, 112, 353
    decode x96 :active, chu4, 353, 4029
    section disaggregated
    prefill :crit, dis0, 0, 35
    transfer :done, dis1, 35, 36
    prefill :crit, dis2, 35, 71
    transfer :done, dis3, 71, 72
    prefill :crit, dis4, 100, 245
    transfer :done, dis5, 245, 254
    decode x102 :active, dis6, 36, 3941
```

Read the `shared` row first. The long red bar from 109 ms to 255 ms is one
indivisible prefill, and the decode bars stop for its entire duration —
that gap *is* the head-of-line stall, drawn to scale.

`chunked` breaks the same prefill into `fused` iterations that each carry
the decodes along, so no single bar is long. `disaggregated` has two
resources: prefill bars on one pool, and a decode bar that runs
**continuously from 36 ms** because nothing on its pool ever interrupts it.

:::note Why generate it rather than draw it
A hand-drawn schematic can illustrate a policy the code does not implement,
and nothing would catch the disagreement — the diagram equivalent of a stale
measurement. These bars come from `Trace.events`, the same run the tables
report, and `tests/test_prefill_decode.py` asserts the bars account for
every GPU-second the trace claims and that no resource is ever double-booked.
Consecutive identical iterations are coalesced (`x96`) and any truncation is
labelled, because a chart that silently drops half a timeline lies about
where the time went.
:::

## The falsifier, stated first

A claim that cannot lose is not a finding. Before measuring anything, this
lab commits to a sentence that the experiment is allowed to refute:

:::warning The claim under test
**If chunked prefill's worst decode gap is not below the shared worker's,
chunking does not help on this hardware and the central claim is wrong
here.**
:::

`serve_bench.py --demo` prints `HOLDS` or `FAILS` against exactly that
sentence. It printed `FAILS` the first time, for a reason worth keeping —
see [below](#the-regime-where-chunking-stops-paying).

## The stall is real, measured against a baseline

RTX 3080 Ti Laptop (16 GB), Qwen3-0.6B bf16, prompt 4096, chunk 256.
**Median of 7 timed repeats after 3 warmup calls.**

| | decode step | vs baseline |
|---|---|---|
| **BASELINE** — nothing else running | **39.6 ms** `[33.1, 44.6]` | 1.0× |
| `shared` — behind a full prefill | 310.3 ms `[303.7, 313.4]` | **7.8×** |
| `chunked` — worst gap, chunk 256 | 115.7 ms | 2.9× |

The baseline row is what makes the other two mean anything. Without a
measurement of an *undisturbed* decode step there is no way to distinguish a
blocked step from an ordinary one, and "310 ms" is just a number.

Notice which column is noisy. `shared` is stable to ±1.5% because a
4096-token prefill dominates it. The **baseline** swings ±14%, because a
single 40 ms decode step is mostly launch overhead and scheduler jitter. The
*ratio* inherits that noisy denominator and wanders between about 4× and 8×
across runs while the absolute numbers barely move. Quote the absolutes.

## On this hardware, the ranking inverts

Feeding the measured constants into the scheduler, at an 8192-token prompt
against two live streams:

| chunk | TTFT p99 | max stall | TTFT cost | stall gain |
|---|---|---|---|---|
| `shared` (baseline) | 0.534 s | 0.524 s | 1.00× | 1.0× |
| 128 | 2.547 s | 0.039 s | **4.77×** | 13.4× |
| 256 | 1.526 s | 0.046 s | 2.86× | 11.4× |
| 512 | 1.015 s | 0.060 s | 1.90× | 8.7× |
| 1024 | 0.760 s | 0.089 s | 1.42× | 5.9× |
| 2048 | 0.632 s | 0.145 s | 1.18× | 3.6× |

**The stall gain saturates; the TTFT cost does not.** The stall is
approaching a floor of *one decode step* (38 ms) — once a chunk is cheaper
than the decode riding with it, shrinking it again cannot help. TTFT has no
floor; it grows linearly in the number of passes.

> **The smallest chunk worth using is the one whose prefill cost is about one
> decode step.** Here a 256-token chunk costs 46 ms against a 38 ms decode
> step, so 256–512 is the sensible range and 128 is past the knee.

**And on this GPU, disaggregation beats chunking outright.** The fixed cost
of a forward pass is 31.9 ms; thirty-two of them is **0.989 s of pure
overhead, twice the 0.486 s prefill being split.** Disaggregation reaches a
*better* stall (0.038 s) for a 10% TTFT cost instead of 186%, paying
instead in GPU-seconds — 4.88 against 4.27.

That inverts the usual recommendation, and it does **not** contradict
Sarathi-Serve. The overhead that makes chunking expensive is *launch*
overhead: large on a laptop GPU running eager PyTorch at batch 1, small on a
datacenter GPU with fused kernels and a big batch. A scheduling result is
scoped to the cost model it was measured under, and this one names its
model.

## A textbook claim that did not survive measurement

"Decode is memory-bandwidth bound" is the standard one-liner. On this setup
it is simply false:

| KV cache | decode step | spread |
|---|---|---|
| 128 | 36.07 ms | ±21% |
| 1,024 | 36.21 ms | ±36% |
| 4,096 | 35.37 ms | ±22% |
| 16,384 | 35.78 ms | ±22% |

**Flat across a 128× range**, with run-to-run spread far larger than any
trend. Fitting a marginal cost per cached token over two points returned a
**negative** number — which is impossible, since more cache cannot be
cheaper to read.

The calibrator detects that and refuses to publish the constant rather than
putting a negative value into a teaching cost model. It reproduced on an
independent run after the `transformers` pin changed.

At 0.6B parameters, batch 1, eager attention, this model is
**kernel-launch bound**. The bandwidth story is a claim about large models,
large batches and fused kernels — worth measuring before quoting.

## The regime where chunking stops paying

The first `--dry-run` capped the prompt at 512 tokens — two chunks — and
printed `FAILS` against the falsifier. That was not a bug in the lab. With a
31.9 ms per-pass overhead and only two passes to amortise it over, unfused
chunking genuinely is slower than letting the prefill run.

> Chunking pays only when the prompt is many chunks long. At two chunks the
> overhead is the whole story.

`--dry-run` now shortens the *repeats* rather than the prompt, because the
prompt length is the variable the claim is about. A smoke test that reports
the headline claim as false is worse than no smoke test.

:::note What the measurement does not model
The benchmark times **unfused** chunking: the chunk and the decode run as two
separate forward passes. A real serving stack batches them into one
iteration, so the decode rides along for free — that is Sarathi-Serve's
"stall-free" half, a *separate idea* from chunking itself. The measured
numbers are therefore an **upper bound** on the chunked stall, and the
simulator models the fused version and reports lower. They are not directly
comparable, and the tool says so in its own output.
:::

## Choosing between them

These are three different designs, not three attempts at the same thing, and
each is the right answer somewhere. The measurements above rank them *on one
laptop GPU*; the ranking is a property of that machine's cost model, not of
the policies.

| | hardware | what it costs you | best when |
|---|---|---|---|
| **Shared** | 1 GPU | tail latency: a stream stalls for the whole prefill | prompts are short, or nobody is watching latency — batch jobs, offline eval, overnight runs |
| **Chunked** | 1 GPU **+ a config flag** | TTFT for the *new* request, one pass overhead per chunk | long prompts and no budget for a second pool |
| **Disaggregated** | **2+ pools + interconnect** | GPU-seconds, a KV transfer, operational complexity | latency SLAs matter and the hardware exists |

Three things decide it in practice, and only the third is about latency:

**Chunked is a config change; disaggregated is an architecture change.** In
vLLM chunking is a flag. Disaggregation means standing up two fleets, moving
KV across an interconnect, and operating them. That asymmetry usually settles
the question before any benchmark does.

**Disaggregation's real prize is not the stall — it is independent sizing.**
Prefill is compute-bound and decode is bandwidth- (or, as measured here,
launch-) bound. Once they are separate pools you can scale them to different
ratios and even buy different hardware for each. No amount of clever
scheduling on one GPU gives you that, and it is the argument DistServe
actually makes.

**Shared is not a strawman.** It has the best throughput per GPU, because
nothing is ever fragmented or copied. If your workload is offline, it wins
outright. The measurements here are about *interactive* serving, which is a
different objective function.

:::tip What the measurement on this page does and does not say
It says: on an RTX 3080 Ti Laptop running eager PyTorch at batch 1, the
per-forward-pass overhead is 81% of a decode step, which leaves chunking
almost nothing to work with and makes disaggregation the better trade.

It does **not** say chunking is a bad technique. Sarathi-Serve's results are
from datacenter GPUs with fused kernels and large batches, where that
overhead term nearly vanishes — exactly the regime where chunking should
win, and the regime most readers deploy in. The formula in the next section
exists so you can tell which regime you are in.
:::

## Taking this into your own code

Running the lab tells you something about *this* GPU. The part that
transfers is the **method and the formula**, not the constants — and three
pieces are meant to be lifted.

### 1. The chunk-size rule maps onto a knob you already have

The lab's result is not "use chunk 512". It is a formula:

$$
C^{*} = \frac{t_{\text{decode step}} - t_{\text{pass overhead}}}{t_{\text{per token}}}
$$

A chunk stops being worth shrinking once it is cheaper than the decode
iteration it rides with — below that the stall is floored by the decode
step and every further split is pure TTFT cost.

**$C^{*}$ is directly `max_num_batched_tokens` in vLLM** (with
`--enable-chunked-prefill`). vLLM's own
[tuning guidance](https://docs.vllm.ai/en/stable/configuration/optimization/)
describes exactly the tradeoff measured here — smaller values give better
inter-token latency because fewer prefills interrupt decodes, larger values
give better time-to-first-token. What this lab adds is a way to *compute* a
starting point from your hardware instead of inheriting a default.

`serve_bench.py --calibrate` prints it:

```
C* = (decode_step - pass_overhead) / prefill_per_token
   = (35.73 - 29.03) ms / 55.2 us
   = 121 tokens

pass overhead is 81% of one decode step
-> chunking has almost no room here. Prefer DISAGGREGATION.
```

**A tiny or negative $C^{*}$ is the useful case.** It means the fixed cost of
a forward pass has eaten the decode step, so no chunk is small enough to
help. That is a diagnosis, not a failure — it is the signal to reach for
disaggregation, and it is exactly what this laptop reports.

As a sanity check: plugging in figures typical of a datacenter GPU with
fused kernels — ~1 ms overhead, ~10 µs/token, ~10 ms decode step, *all
illustrative and not measured here* — gives $C^{*} \approx 900$, the same
order as vLLM's defaults. The rule lands in the right place on hardware
where chunking is known to work.

### 2. `scheduler.py` drops into anything

It imports `math` and `dataclasses`. **No torch, no transformers, no
numpy** — copy the file and it runs:

```python
from scheduler import CostModel, Request, simulate

cost = CostModel(                      # YOUR constants, from --calibrate
    prefill_pass_overhead=1.0e-3,
    prefill_per_token=1.0e-5,
    decode_step_base=1.0e-2,
)
load = [Request(f"r{i}", arrival=i * 0.05, prompt_len=4096, output_len=256)
        for i in range(64)]

for policy in ("shared", "chunked", "disaggregated"):
    t = simulate(policy, load, cost, chunk=900)
    print(policy, t.summary())
```

That answers "what happens at 64 concurrent streams" without renting 64
streams' worth of hardware — which is the point of having a cost model at
all. Measure cheap, extrapolate free.

### 3. The measurement discipline is the most portable part

Three habits from `serve_bench.py` that are worth more than the numbers:

- **Median of repeats after warmup, with the spread printed.** One timed
  call is a sample, not a measurement.
- **Difference two measurements to isolate a term.** No single timed call
  contains only one cost; the per-token slope and the fixed overhead come
  from subtracting a short run from a long one.
- **Refuse a fit that the noise cannot support.** Compare the effect
  against the measurement's own scatter, *not* against zero. This lab's
  first guard only rejected negative values, so consecutive runs disagreed
  about whether decode is bandwidth-bound — a small positive number drawn
  from the same noise sailed through.

### What does NOT transfer

The constants, and therefore the conclusion. On this laptop disaggregation
beats chunking; on an H100 with fused kernels and a large batch the
overhead term shrinks and chunking wins. **Re-measure on your serving
hardware.** The lab's job is to make that cheap, not to hand you an answer.

## Why this is in a DeepSpeed course

ZeRO shards what the model *is*. Token compression shrinks what it *looks
at*. STAR memory bounds what it *retains*. This page is about none of those:
the model fits, the memory is fine, and the problem is **when each piece of
work runs**.

That makes it the scheduling member of the same family — and the one where
the reflex ("it's slow, get a bigger box") helps least. Nothing here is
fixed by more VRAM.

It is also the clearest case in the course of a result that **reverses with
hardware**. The same code, the same model, the same policies, run on an
H100 with fused kernels, would favour chunking. The lab's job is to make you
measure your own constants rather than inherit someone else's conclusion.

## Run it

```bash
cd 03_llms/12_prefill_decode
uv sync

# No GPU, no download. The scheduling BEHAVIOUR is all here.
uv run --no-project python scheduler.py
uv run ../../tests/test_prefill_decode.py      # 27 property checks

# With a GPU
uv run serve_bench.py --calibrate --repeats 9 --warmup 4
uv run serve_bench.py --demo --prompt-len 4096 --chunk 256
```

The split between the two files is deliberate: a wrong constant makes the
*numbers* wrong, a wrong scheduler makes the *conclusion* wrong, and only
the second is a teaching error. The property tests police the second without
any hardware.

## References

- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670), OSDI 2024
- Kwon et al., [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023
- Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models, OSDI 2022
