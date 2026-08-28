"""
Answer questions about a live video stream, in constant memory.

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER SCRIPT IN 08_vtt
------------------------------------------------------------
Offline video inference is a batch job: you have the whole clip, you sample
frames, you build one sequence, you answer. Every technique in
`../02_token_compression/` optimises that shape — and every one of them
shrinks cost by a constant FACTOR. Halve the tokens and a two-hour video is
still twice a one-hour video. For any fixed compression ratio there exists a
video long enough to OOM you.

Streaming is a different problem with a strictly harder constraint:

    frames arrive forever, answers are needed DURING the stream,
    and memory must not grow.

Not "must grow slowly". Must not grow. A system whose per-frame cost creeps
upward does not degrade gracefully at hour six — it dies.

THE THREE THINGS THAT MUST BE TRUE
----------------------------------
1. WRITE IS O(1). Absorbing frame 1,000,000 costs the same as frame 1. If
   ingestion ever touches all previously seen frames, you have a batch system
   wearing a streaming costume.

2. READ IS BOUNDED. The context handed to the LLM has a fixed ceiling, so
   inference latency is flat and predictable. This is what makes real-time
   answering possible at all.

3. DECOUPLED CLOCKS. Ingestion runs at the camera's rate; questions arrive at
   the user's rate. Neither blocks the other. A question at t=500 is answered
   from whatever memory holds at t=500 — the stream does not pause, and frames
   are not buffered up waiting for the model.

Point 3 is the one that gets designed away. If answering blocks ingestion you
drop frames, and the memory you are so carefully maintaining is now a memory
of a video that did not happen.

WHAT THIS SCRIPT DOES
---------------------
Drives `star_memory.StarMemory` over a synthetic or real stream, answers
queries at fixed intervals, and reports per-frame latency plus memory at each
checkpoint — so you can SEE the flat line rather than take it on faith.

With `--model`, it wires the memory into a real Qwen2.5-VL for actual
generation. Without it, the memory mechanics run on CPU with no download,
which is the version you should read first.

RUNNING IT
----------
CPU, no model, no download:
    uv run 08_vtt/03_streaming_memory/stream_infer.py --frames 5000

CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 08_vtt/03_streaming_memory \
                            --collect --wait --terminate --yes

Reference: Zhang et al. "Flash-VStream: Memory-Based Real-Time Understanding
for Long Video Streams." https://arxiv.org/abs/2406.08085
"""

import argparse
import os
import sys
import time
from typing import Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from star_memory import StarConfig, StarMemory  # noqa: E402


def synthetic_stream(
    n_frames: int,
    tokens_per_frame: int = 64,
    dim: int = 1152,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """
    An endless-feeling stream with real temporal structure.

    Structure, not noise, deliberately. The content drifts slowly (a scene
    changing over minutes) with occasional abrupt shifts (a cut). That gives
    the temporal clustering genuine events to find. Pure noise would let a
    completely broken consolidation step look identical to a working one —
    every centroid would be equally meaningless and every test would pass.
    """
    generator = torch.Generator().manual_seed(seed)
    scene = torch.randn(dim, generator=generator)

    for i in range(n_frames):
        # A hard cut every ~500 frames.
        if i % 500 == 0 and i > 0:
            scene = torch.randn(dim, generator=generator)
        # Slow drift within a scene.
        scene = scene + torch.randn(dim, generator=generator) * 0.01
        yield scene.expand(tokens_per_frame, dim) + torch.randn(
            tokens_per_frame, dim, generator=generator
        ) * 0.1


def video_file_stream(
    path: str, dim: int = 1152, target_fps: float = 2.0
) -> Iterator[torch.Tensor]:
    """
    Decode a real video file into a frame-feature stream.

    Uses a fixed random projection of raw pixels rather than a vision encoder,
    so this path needs no model download. That is enough to exercise the
    memory mechanics honestly — the projection is deterministic and preserves
    relative distances well enough for clustering (Johnson-Lindenstrauss).
    Swap in a real encoder when you care about the answers rather than the
    memory behaviour.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / target_fps)))

    projection = torch.randn(
        64 * 3, dim, generator=torch.Generator().manual_seed(0)
    )

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                small = cv2.resize(frame, (8, 8))
                flat = torch.from_numpy(small).float().reshape(-1) / 255.0
                # 64 tokens of an 8x8 grid, each a projected pixel triple.
                patches = flat.reshape(64, 3)
                padded = torch.zeros(64, 64 * 3)
                padded[:, :3] = patches
                yield padded @ projection
            idx += 1
    finally:
        cap.release()


def answer_query(
    memory: StarMemory,
    question: str,
    model: Optional[object] = None,
    processor: Optional[object] = None,
) -> str:
    """
    Answer from the bounded memory, WITHOUT touching the raw stream.

    That restriction is the whole contract. It would be easy — and completely
    wrong — to re-read the original frames when a question arrives: the answer
    would be better and the system would no longer be O(1), because you would
    be storing every frame in order to re-read it. If answering ever needs
    more than `memory.read()`, the memory design has failed and should be
    fixed rather than worked around.

    Without a model this returns a structural summary, which is genuinely the
    useful thing while you are debugging: it tells you what the memory HOLDS,
    which is the question you actually have at that stage.
    """
    context = memory.read()

    if model is None or processor is None:
        return (
            f"[no model] context {context.shape[0]} tokens from "
            f"{memory.frames_seen:,} frames | "
            f"temporal clusters {memory.temporal.shape[0]} | "
            f"largest event {memory.temporal_w.max().item():.0f} frames"
        )

    # Real generation path: the memory tokens ARE the visual context, injected
    # as inputs_embeds so no re-encoding happens.
    inputs = processor(text=question, return_tensors="pt")
    embed_layer = model.get_input_embeddings()
    text_embeds = embed_layer(inputs["input_ids"].to(model.device))
    visual = context.unsqueeze(0).to(model.device, dtype=text_embeds.dtype)

    combined = torch.cat([visual, text_embeds], dim=1)
    with torch.no_grad():
        out = model.generate(inputs_embeds=combined, max_new_tokens=64)
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5000,
                        help="Stream length. Raise it — that is the demo.")
    parser.add_argument("--video", default=None,
                        help="Real video file. Omit for the synthetic stream.")
    parser.add_argument("--query-every", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=1152)
    parser.add_argument("--model", default=None,
                        help="Optional Qwen2.5-VL for real generation. "
                             "Omit to run the memory mechanics on CPU.")
    parser.add_argument("--n-temporal", type=int, default=25)
    parser.add_argument("--n-abstract", type=int, default=25)
    args = parser.parse_args()

    model = processor = None
    if args.model:
        # Only this path needs a GPU; the memory mechanics do not.
        if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
            print("\n[preflight] --model needs a GPU. Drop --model to run the")
            print("            memory mechanics on CPU, or set ALLOW_CPU=1.\n")
            sys.exit(1)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto"
        )

    config = StarConfig(n_temporal=args.n_temporal, n_abstract=args.n_abstract)
    memory = StarMemory(dim=args.dim, config=config)

    bar = "=" * 78
    print(bar)
    print("  Streaming video understanding — constant memory, unbounded stream")
    print(bar)
    print(f"  stream          {args.video or 'synthetic'}")
    print(f"  frames          {args.frames:,}")
    print(f"  context ceiling {memory.max_context_tokens()} tokens")
    print(f"  model           {args.model or 'none (memory mechanics only)'}")
    print(bar)

    stream = (
        video_file_stream(args.video, dim=args.dim)
        if args.video
        else synthetic_stream(args.frames, dim=args.dim)
    )

    write_times = []
    started = time.time()

    for i, frame in enumerate(stream, start=1):
        t0 = time.perf_counter()
        memory.write(frame)
        write_times.append(time.perf_counter() - t0)

        if i % args.query_every == 0:
            recent = sum(write_times[-args.query_every:]) / args.query_every
            first = sum(write_times[:100]) / min(100, len(write_times))
            answer = answer_query(
                memory, "What has happened in this video so far?",
                model, processor,
            )
            print(f"\n  frame {i:>8,}  |  write {recent * 1000:.3f} ms "
                  f"(first 100 frames: {first * 1000:.3f} ms)")
            print(f"  {answer}")

        if args.frames and i >= args.frames:
            break

    elapsed = time.time() - started
    print("\n" + bar)
    print(f"  {memory.frames_seen:,} frames in {elapsed:.1f}s "
          f"({memory.frames_seen / elapsed:.0f} frames/s)")
    print(f"  final context: {memory.read().shape[0]} tokens")

    naive = memory.frames_seen * 64
    print(f"  a naive system would hold {naive:,} tokens "
          f"— {naive / memory.read().shape[0]:,.0f}x more")
    print(bar)
    print("\n  The write time and the context size are FLAT. Everything else")
    print("  in this course trades memory for something; this trades away the")
    print("  ability to recall the distant past in detail, and buys the")
    print("  ability to run forever.")


if __name__ == "__main__":
    main()
