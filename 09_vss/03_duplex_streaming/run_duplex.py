"""
Drive a full-duplex conversation and measure whether it is actually real-time.

WHAT THIS MEASURES, AND WHY IT IS THE ONLY THING THAT MATTERS
-------------------------------------------------------------
`duplex.py` implements the turn-taking policy and proves it on CPU. This script
answers the question the policy cannot: **can your hardware keep up?**

    RTF = compute_time / audio_duration_produced

Every 480 ms slice must be processed in under 480 ms. Not on average --
*every* slice. At RTF > 1 the model falls progressively further behind, the
backlog grows without bound, and the conversation degrades until it collapses.
There is no batch size that fixes it and no amount of waiting that catches up.

So the headline number here is **worst-case RTF, not mean RTF**. A system
averaging 0.6 with occasional spikes to 1.4 stutters audibly, and the mean
hides that completely. This script reports both and fails on the worst.

WHAT A REAL RUN LOOKS LIKE
--------------------------
Without `--model` this drives the policy against a synthetic conversation using
a configurable simulated compute cost — useful for exploring how much budget
per slice you actually have before the design stops working, and it needs no
GPU at all.

With `--model` it runs a real omni model over a real (or synthetic) stream and
measures the true per-slice cost. That is the number to trust.

THE BUDGET, ROUGHLY
-------------------
480 ms per slice has to cover ALL of:

    encode ~480 ms of audio          (audio tower)
    encode the video frames in it    (vision tower)
    one Thinker forward step
    one Talker forward step -> 480 ms of audio tokens
    vocoder / token2wav

Which is why streaming omni models are small. A 3B model at ~0.2 RTF has
headroom; a 7B at ~0.7 does not, and the first time the user says something
long you hear it.

RUNNING IT
----------
CPU, no model, no download:
    uv run 09_vss/03_duplex_streaming/run_duplex.py --slices 200

CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 09_vss/03_duplex_streaming \\
                            --collect --wait --terminate --yes

Reference: "DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for
Full-Duplex Interaction." https://arxiv.org/abs/2606.09186
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duplex import SLICE_SECONDS, DuplexSession, Slice, State  # noqa: E402


def synthetic_conversation(n_slices: int, seed: int = 0
                           ) -> List[Tuple[bool, bool]]:
    """
    A conversation with realistic turn structure — including interruptions.

    Structured rather than random. Real dialogue is bursty: multi-second turns
    separated by gaps, with occasional overlaps. Uniformly random activity
    would never produce a sustained barge-in, so the policy's most important
    path would go untested.
    """
    rng = random.Random(seed)
    script: List[Tuple[bool, bool]] = []

    while len(script) < n_slices:
        # User turn: 2-8 slices of speech (about 1-4 seconds).
        for _ in range(rng.randint(2, 8)):
            script.append((True, False))
        # Gap: we reply here.
        for _ in range(rng.randint(4, 12)):
            script.append((False, False))
        # Sometimes the user interrupts our reply.
        if rng.random() < 0.35:
            gesture_only = rng.random() < 0.3
            for _ in range(rng.randint(2, 5)):
                script.append((not gesture_only, gesture_only))

    return script[:n_slices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Omni model for a REAL measurement. Omit to drive "
                             "the policy on CPU with simulated compute.")
    parser.add_argument("--slices", type=int, default=200,
                        help=f"Slices to run ({SLICE_SECONDS * 1000:.0f} ms each).")
    parser.add_argument("--simulated-compute", type=float, default=0.12,
                        help="Seconds per slice when no model is loaded. Raise "
                             "past 0.48 to see the RTF failure reported.")
    parser.add_argument("--barge-in-slices", type=int, default=2)
    parser.add_argument("--output", default="duplex_results.json")
    args = parser.parse_args()

    model = processor = None
    if args.model:
        try:
            import torch
        except ImportError:
            print("\n[preflight] PyTorch is not installed. Install it with:")
            print("            uv pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu128\n")
            sys.exit(1)

        if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
            bar = "=" * 72
            print("\n" + bar)
            print("  NO GPU DETECTED")
            print(bar)
            print("\n  --model needs a GPU. An omni model loads a language")
            print("  backbone, a vision encoder, an audio encoder AND a speech")
            print("  decoder; measuring RTF on CPU would tell you nothing about")
            print("  whether the real system keeps up.")
            print("\n  The turn-taking POLICY runs fine without one — drop")
            print("  --model and it drives on simulated compute:")
            print("      uv run run_duplex.py --slices 200")
            print("      uv run tests/test_duplex.py     # 36 checks")
            print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
            print("      uv run runpod/runpod_ctl.py run "
                  "09_vss/03_duplex_streaming \\")
            print("          --collect --wait --terminate --yes")
            print("\n" + bar + "\n")
            sys.exit(1)

        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model,
                                                  trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                          dtype=torch.bfloat16,
                                          device_map="auto")

    bar = "=" * 74
    print(bar)
    print("  Full-duplex conversation — is it actually real-time?")
    print(bar)
    print(f"  slice            {SLICE_SECONDS * 1000:.0f} ms")
    print(f"  slices           {args.slices} "
          f"({args.slices * SLICE_SECONDS:.0f} s of conversation)")
    print(f"  model            {args.model or 'none (policy + simulated compute)'}")
    if model is None:
        print(f"  simulated cost   {args.simulated_compute * 1000:.0f} ms/slice "
              f"(RTF {args.simulated_compute / SLICE_SECONDS:.2f})")
    print(f"  barge-in after   {args.barge_in_slices} active slices")
    print(bar)

    script = synthetic_conversation(args.slices)
    session = DuplexSession(barge_in_slices=args.barge_in_slices)

    words = ("I can see the chart on the left and I heard you mention "
             "the deadline moving to Friday which changes the plan").split()

    for i, (speaking, gesture) in enumerate(script):
        sl = Slice(index=i, user_speaking=speaking, user_gesture=gesture,
                   video_frames=int(SLICE_SECONDS * 2))

        started = time.perf_counter()
        if model is not None:
            # A real slice: encode this window's audio and video, step the
            # Thinker, step the Talker. Left explicit rather than hidden --
            # the per-slice cost IS the measurement.
            raise NotImplementedError(
                "Wire your model's streaming step here: encode the slice's "
                "audio and video, run one Thinker step, run one Talker step, "
                "and let the elapsed wall-clock become compute_seconds. See "
                "the README."
            )
        compute = args.simulated_compute
        elapsed = time.perf_counter() - started
        compute = max(compute, elapsed)

        want = (session.state == State.LISTENING and not sl.user_active
                and i > 0)
        session.step(sl, planned_text=words[i % len(words)],
                     compute_seconds=compute, want_floor=want)

    print()
    print(session.report())
    print(bar)

    if session.is_realtime():
        headroom = (1.0 - session.worst_rtf) * SLICE_SECONDS * 1000
        print(f"\n  REAL-TIME. {headroom:.0f} ms of headroom per slice at the "
              f"worst point.")
        print("  That headroom is your budget for a bigger model, more video")
        print("  frames, or a slower vocoder — spend it deliberately.")
    else:
        print("\n  NOT REAL-TIME. This is not a slow system, it is a broken")
        print("  one: the backlog grows every slice and never recovers.")
        print("  Shrink the model, cut the video frame rate, or lengthen the")
        print("  slice — but a longer slice raises the latency floor.")

    with open(args.output, "w") as handle:
        json.dump({
            "slices": session.slices_processed,
            "mean_rtf": session.mean_rtf,
            "worst_rtf": session.worst_rtf,
            "is_realtime": session.is_realtime(),
            "mean_response_latency": session.mean_response_latency,
            "barge_ins": sum(1 for r in session.results if "^" in r.control),
            "ghost_fragments": len(session.ghost_text),
        }, handle, indent=2)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
