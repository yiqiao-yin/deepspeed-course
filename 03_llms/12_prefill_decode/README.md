# Prefill and Decode: when to split them

**Baseline:** an undisturbed decode step with nothing else running, 39.6 ms. Without it there is no way to tell a blocked step from an ordinary one.

**Budget:** median of 7 timed repeats after 3 warmup calls, prompt 4096, chunk 256, on one RTX 3080 Ti Laptop. Spreads are printed beside every number because the ratio is far noisier than the absolutes.

One request, two phases that want opposite hardware. **Prefill** reads the
whole prompt at once, so every token is available and the work becomes big
dense GEMMs — it saturates compute and finishes in one pass. **Decode**
emits one token at a time, doing almost no arithmetic while streaming the
weights and the whole KV cache out of memory.

Run both on one worker and they interfere. A long prompt's prefill is a
long, indivisible stretch of GPU time, and every live stream stalls behind
it. The user does not see *"someone else sent a long prompt"*. They see
their own tokens stop.

This lab builds three scheduling policies, measures the real cost of each
phase on a GPU, and asks which policy is worth it. **The measured answer
contradicts the obvious one**, which is the reason the lab exists.

| policy | idea | paper |
|---|---|---|
| `shared` | one worker, first-come-first-served. The baseline. | — |
| `chunked` | split the prefill, run a chunk per iteration *alongside* the decodes | [Sarathi-Serve, OSDI '24](https://arxiv.org/abs/2403.02310) |
| `disaggregated` | separate prefill and decode pools, ship the KV cache | [DistServe, OSDI '24](https://arxiv.org/abs/2401.09670) |

## What is here

| File | Role |
|---|---|
| `scheduler.py` | the three policies as a discrete-event simulator over a cost model. **Runs on CPU in milliseconds**, no model download |
| `serve_bench.py` | measures the cost model's constants on a real GPU with Qwen3-0.6B, and demonstrates the stall end to end |
| `run_deepspeed.sh` | SLURM wrapper for the measurement |

The split is deliberate. A wrong constant makes the *numbers* wrong; a wrong
scheduler makes the *conclusion* wrong. Only the second is a teaching error,
and `tests/test_prefill_decode.py` polices it with 27 property checks that
need no hardware.

## The falsifier

Stated before running, because a claim you cannot lose is not a finding:

> **If chunked prefill's worst decode gap is not below `shared`'s, chunking
> does not help on this hardware and the central claim of this lab is wrong
> here.**

`serve_bench.py --demo` prints `HOLDS` or `FAILS` against that sentence.

## Measured: the stall is real

RTX 3080 Ti Laptop (16 GB), Qwen3-0.6B bf16, prompt 4096, chunk 256,
**median of 7 timed repeats after 3 warmup**:

| | decode step | vs baseline |
|---|---|---|
| **BASELINE** — nothing else running | **39.6 ms** `[33.1, 44.6]` | 1.0× |
| `shared` — behind a full prefill | 310.3 ms `[303.7, 313.4]` | **7.8×** |
| `chunked` — worst gap, chunk 256 | 115.7 ms | 2.9× |

Note which column is noisy. `shared` is stable to ±1.5% because it is
dominated by a 4096-token prefill; the **baseline** swings ±14% because a
single 40 ms decode step is mostly launch overhead and scheduler jitter.
The *ratio* inherits the noisy denominator, so across repeated runs it
moves between roughly 4× and 8× while the absolute numbers barely move.
Quote the absolutes.

The baseline row is the one that makes the others mean anything. Without it
there is no way to tell a blocked step from an ordinary one.

**Scope: this measures *unfused* chunking.** The chunk and the decode run as
two separate forward passes. A real serving stack puts them in one batched
iteration, so the decode rides along for free — that is Sarathi-Serve's
"stall-free" half, and it is a separate idea from chunking itself. These
measurements are therefore an **upper bound** on the chunked stall;
`scheduler.py` models the fused version and reports lower. The two are not
directly comparable, and the demo says so in its own output.

## Measured: the obvious fix is the wrong one here

Feeding those constants into the simulator, at an 8192-token prompt against
two live streams:

| chunk | TTFT p99 | max stall | TTFT cost | stall gain |
|---|---|---|---|---|
| `shared` (baseline) | 0.534 s | 0.524 s | 1.00× | 1.0× |
| 128 | 2.547 s | 0.039 s | **4.77×** | 13.4× |
| 256 | 1.526 s | 0.046 s | 2.86× | 11.4× |
| 512 | 1.015 s | 0.060 s | 1.90× | 8.7× |
| 1024 | 0.760 s | 0.089 s | 1.42× | 5.9× |
| 2048 | 0.632 s | 0.145 s | 1.18× | 3.6× |

Two things fall out, and neither is in the infographic version of this topic.

**The stall gain saturates while the TTFT cost does not.** Going from chunk
512 to 128 buys 8.7× → 13.4× on the stall and costs 1.90× → 4.77× on TTFT.
The stall is approaching its floor — *one decode step*, 38 ms — because once
a chunk is cheaper than the decode it rides with, shrinking it again cannot
help. The TTFT cost has no such floor; it grows linearly in the number of
passes. The useful rule:

> the smallest chunk worth using is the one whose prefill cost is about one
> decode step. Here a 256-token chunk costs 46 ms against a 38 ms decode
> step, so 256–512 is the sensible range and 128 is past the knee.

**And on this GPU, disaggregation beats chunking outright.** A 31.9 ms fixed
cost per forward pass × 32 chunks is **0.989 s of pure overhead — twice the
0.486 s prefill it is splitting.** Disaggregation gets a *better* stall
(0.038 s) for a TTFT of 0.587 s against shared's 0.534 s — a 10% TTFT cost
rather than 90%. It pays instead in GPU-seconds, 4.88 against 4.27.

That is the opposite of the usual recommendation, and it is not a
contradiction of Sarathi-Serve. It is a statement about *this* hardware: the
per-pass overhead that makes chunking expensive is launch overhead, which is
large on a laptop GPU running eager PyTorch at batch 1 and small on a
datacenter GPU with fused kernels. **A scheduling result is scoped to the
cost model it was measured under.**

## The textbook claim that did not survive measurement

"Decode is memory-bandwidth bound" is the standard summary. On this setup it
is false, and the calibrator refuses to pretend otherwise:

| KV cache | decode step | spread |
|---|---|---|
| 128 | 36.07 ms | ±21% |
| 1,024 | 36.21 ms | ±36% |
| 4,096 | 35.37 ms | ±22% |
| 16,384 | 35.78 ms | ±22% |

**Flat across a 128× range**, with run-to-run spread far larger than any
trend. A least-squares fit over two points returned a *negative* marginal
cost per cached token — impossible, since more cache cannot be cheaper to
read. `serve_bench.py` detects that and refuses to publish the constant
rather than putting a negative number into a teaching cost model.

At 0.6B, batch 1, eager attention, this model is **kernel-launch bound**. The
bandwidth story is a claim about large models, large batches and fused
kernels. It is worth measuring before quoting.

### A regime where chunking loses

The first `--dry-run` capped the prompt at 512 tokens — two chunks — and
printed `FAILS` against the falsifier above. That was not a bug in the lab:
with a 28.4 ms per-pass overhead and only two passes to amortise it over,
unfused chunking really is slower than letting the prefill run. The useful
form of the rule:

> chunking pays only when the prompt is many chunks long. At two chunks the
> overhead is the whole story.

`--dry-run` now shortens the *repeats* rather than the prompt, because the
prompt length is the variable the claim is about. A smoke test that reports
the headline claim as false is worse than no smoke test.

## Run it

```bash
cd 03_llms/12_prefill_decode
uv sync

# no GPU, no download — the scheduling behaviour is all here
uv run --no-project python scheduler.py
uv run ../../tests/test_prefill_decode.py

# with a GPU
uv run serve_bench.py --calibrate --repeats 9 --warmup 4
uv run serve_bench.py --demo --prompt-len 4096 --chunk 256
```

### Environment & Local Testing

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B` — 0.6B params, 28 layers, GQA 16Q/8KV, 32k ctx |
| Download | ~1.2 GB |
| GPU | 1 × 16 GB is plenty; verified on RTX 3080 Ti Laptop |
| Without a GPU | `scheduler.py` and the full test suite run on CPU in under a second |

`ALLOW_CPU=1` forces the benchmark onto CPU. It will be slow and the
constants will describe a CPU, which is a different machine with a different
answer — the conclusions above do not transfer.

### Renting a GPU (RunPod)

```bash
# --dry-run first: 3 repeats, short prompt. Proves the model loads and the
# harness works before spending anything.
uv run runpod/runpod_ctl.py run 03_llms/12_prefill_decode --dry-run --yes

# the real run. --collect pulls the logs back, --wait blocks until it
# finishes, and --terminate destroys the pod afterwards.
uv run runpod/runpod_ctl.py run 03_llms/12_prefill_decode \
    --collect --wait --terminate --yes
```

**Confirm the pod is gone with `uv run runpod/runpod_ctl.py pods`.** An
abandoned pod bills until terminated, and `--terminate` is driven from your
machine in a `finally` — if your laptop dies mid-run the in-pod watchdog is
the backstop, not a guarantee. Check.

### Why there is no `deepspeed` launcher here

This is **inference**. There is no optimizer, no gradient, nothing to shard
across ranks — a single process owns the model and the scheduler is the
subject. Using a distributed launcher where there is nothing to distribute
is cargo cult, so this lab is registered with `launcher="python"`, joining
the five existing exceptions listed in `CLAUDE.md`.

## Reading on

- `03_llms/11_moe` — the other place where *where the work goes* matters more
  than how much of it there is
- `04_video_text/03_token_compression` — shrinking what the model looks at,
  the same bargain in a different currency
- [PagedAttention / vLLM, SOSP '23](https://arxiv.org/abs/2309.06180) — how
  the KV cache is actually stored once you are serving many of these
