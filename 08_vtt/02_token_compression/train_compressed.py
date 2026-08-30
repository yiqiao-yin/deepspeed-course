"""
Train Qwen2.5-VL with visual token compression, and MEASURE what it bought.

THE POINT OF THIS SCRIPT
------------------------
`token_compression.py` next door implements the algorithms and proves them
correct on CPU. Correct is not the same as useful. This script answers the
only question that decides whether you ship compression:

    At a fixed VRAM budget, how many more frames can I actually fit,
    and what did it cost me in loss?

It runs a real DeepSpeed training loop with compression on and off, records
`torch.cuda.max_memory_allocated()` for each, and prints the comparison. No
estimates. The estimator in `TokenBudget` is useful for planning and it is not
evidence.

WHY MEASURING IS NOT OPTIONAL
-----------------------------
Three ways the predicted win fails to materialise, all common:

  * You cut tokens 2x and memory barely moves, because at your batch size the
    optimizer states dominated — that is a ZeRO problem, and the fix is one
    directory up in the course, not here.
  * You cut tokens 2x and step time barely moves, because at your sequence
    length the MLP (linear in N) dominated attention (quadratic in N). The
    quadratic intuition only pays off once you are far enough along the curve.
  * You cut tokens 4x and the loss degrades, because the compression ratio was
    tuned on a benchmark whose videos are more static than yours.

Each of those looks identical from the outside: "compression didn't help."
Only measurement separates them, and the fix differs in every case.

RELATION TO ZeRO
----------------
ZeRO shards what the model IS across GPUs — parameters, gradients, optimizer
states — and pays in inter-GPU communication. Compression shrinks what the
model LOOKS AT and pays in fidelity. They address disjoint terms of the memory
equation and compose cleanly. If you are OOMing, the first question is always
which term dominates, because optimising the other one is free effort.

RUNNING IT
----------
CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 08_vtt/02_token_compression \
                            --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install deepspeed transformers accelerate peft opencv-python-headless
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from token_compression import (  # noqa: E402
    TokenBudget,
    bipartite_soft_matching,
    count_visual_tokens,
    dycoke_temporal_merge,
    merge_wavg,
)


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            Memory numbers will be MEANINGLESS on CPU — the "
              "whole point of")
        print("            this script is torch.cuda.max_memory_allocated().\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  This script MEASURES GPU memory, so it needs a GPU by")
    print("  definition. There is nothing meaningful to report on CPU.")
    print("\n  The compression ALGORITHMS themselves run fine on CPU:")
    print("      uv run 08_vtt/02_token_compression/token_compression.py")
    print("      uv run tests/test_token_compression.py   # 30 checks")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 08_vtt/02_token_compression \\")
    print("          --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def compress_video_tokens(
    video_tokens: "Any",
    strategy: str,
    keep_ratio: float,
    num_frames: int,
) -> "Any":
    """
    Apply a compression strategy to a flat (B, T*N, C) block of visual tokens.

    WHERE THIS SITS IN THE MODEL

    Between the vision tower and the language model. That placement is a
    deliberate compromise:

      - Inside the vision tower (true ToMe) is more effective — you save on
        the encoder's own layers too — but requires patching its forward pass,
        which breaks on every upstream release.
      - Between the towers, as here, is architecture-agnostic and survives
        library upgrades. You lose the encoder-side saving and keep the LLM-
        side saving, which is the larger of the two anyway since the LLM has
        far more layers.

    The frontier implementations patch the tower. Start here, measure, and
    only go deeper if the numbers justify the maintenance.

    Args:
        video_tokens: (B, T*N, C) visual tokens from the vision encoder.
        strategy: "none", "tome", "dycoke", or "both".
        keep_ratio: Target fraction to retain.
        num_frames: T, needed to un-flatten for the temporal strategies.

    Returns:
        (B, M, C) with M <= T*N.
    """
    import torch

    if strategy == "none" or keep_ratio >= 1.0:
        return video_tokens

    b_sz, total, chan = video_tokens.shape
    per_frame = total // num_frames

    if strategy in ("dycoke", "both"):
        # Temporal first. It is the cheaper filter — one dot product per
        # position, no N-by-N matrix — and it removes whole-frame redundancy
        # that spatial merging would otherwise waste its budget rediscovering
        # once per frame.
        frames = video_tokens[:, : num_frames * per_frame].reshape(
            b_sz, num_frames, per_frame, chan
        )
        _, mask = dycoke_temporal_merge(frames, window=4,
                                        similarity_threshold=0.9)
        flat_mask = mask.reshape(b_sz, -1)
        # Keep the batch rectangular: take the same NUMBER of tokens from each
        # sample, chosen by the mask. Ragged batches would force padding, and
        # padding gives back exactly what compression just saved.
        n_keep = max(1, int(flat_mask[0].sum().item()))
        idx = flat_mask.float().topk(n_keep, dim=-1).indices.sort(dim=-1).values
        video_tokens = video_tokens.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, chan)
        )

    if strategy in ("tome", "both"):
        # Spatial merging on whatever survived. We use the token features as
        # the similarity metric because we are outside the attention block and
        # have no keys here; inside the tower you would pass the keys, which
        # Bolya et al. show works better.
        current = video_tokens.shape[1]
        target = max(1, int(total * keep_ratio))
        size = None
        # ToMe removes at most half the tokens per round, so iterate.
        while current > target:
            r = min(current - target, current // 2)
            if r <= 0:
                break
            merge, _ = bipartite_soft_matching(video_tokens, r=r)
            video_tokens, size = merge_wavg(merge, video_tokens, size)
            current = video_tokens.shape[1]

    return video_tokens


def measure_run(
    strategy: str,
    keep_ratio: float,
    num_frames: int,
    hidden: int,
    steps: int,
    batch: int,
    config_path: str = "ds_config.json",
) -> Dict[str, Any]:
    """
    Run a few steps through a stand-in transformer block and record peak VRAM.

    WHY A STAND-IN AND NOT THE REAL MODEL

    We want to isolate ONE variable: how peak memory responds to sequence
    length. Loading Qwen2.5-VL adds several confounds — weight residency,
    ZeRO's sharding schedule, the vision tower's own activations — all of
    which are large, none of which are what compression changes. The block
    here has the same quadratic attention and linear MLP as the real thing, so
    the SHAPE of the curve is faithful even though the constant is not.

    Read the ratios, not the absolute numbers. The absolute numbers for your
    model come from running your model.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    per_frame = 256
    total_tokens = num_frames * per_frame

    device = "cuda"
    block = torch.nn.TransformerEncoderLayer(
        d_model=hidden,
        nhead=16,
        dim_feedforward=hidden * 4,
        batch_first=True,
        dtype=torch.bfloat16,
    ).to(device)

    # Run the step through DeepSpeed rather than a bare torch optimizer.
    # This is not ceremony: ZeRO changes where the optimizer states live, and
    # the whole question this script answers is "which memory term dominates?"
    # Measuring compression against a plain AdamW would attribute ZeRO's
    # saving to compression and vice versa. Same engine as the real trainer,
    # so the ratios transfer.
    engine = None
    if os.path.exists(config_path):
        try:
            import deepspeed
            engine, _, _, _ = deepspeed.initialize(
                model=block,
                model_parameters=block.parameters(),
                config=config_path,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  [warn] DeepSpeed init failed ({exc}); "
                  "falling back to torch AdamW. Memory numbers will not "
                  "include ZeRO's effect.")

    if engine is None:
        opt = torch.optim.AdamW(block.parameters(), lr=1e-4)

    compressed_tokens = total_tokens
    losses = []

    for _ in range(steps):
        tokens = torch.randn(
            batch, total_tokens, hidden, device=device, dtype=torch.bfloat16
        )
        tokens = compress_video_tokens(tokens, strategy, keep_ratio, num_frames)
        compressed_tokens = tokens.shape[1]

        if engine is not None:
            out = engine(tokens)
            loss = out.float().pow(2).mean()
            engine.backward(loss)
            engine.step()
        else:
            out = block(tokens)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    peak = torch.cuda.max_memory_allocated() / 1e9
    budget = TokenBudget(total_tokens, compressed_tokens, hidden_size=hidden)

    return {
        "strategy": strategy,
        "keep_ratio_requested": keep_ratio,
        "keep_ratio_actual": budget.keep_ratio,
        "tokens": compressed_tokens,
        "peak_vram_gb": peak,
        "mean_loss": sum(losses) / len(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--output", default="compression_results.json")
    parser.add_argument("--deepspeed", default="ds_config.json",
                        help="ZeRO config. Omit the file to fall back to "
                             "a plain torch AdamW.")
    # parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    # --local_rank=N into every worker's argv, and a strict parser exits 2
    # with "unrecognized arguments" before training starts -- breaking the
    # exact command this example documents. CONTRIBUTING.md section 3.2.
    args = parser.parse_known_args()[0]
    require_gpu()
    import torch

    bar = "=" * 74
    print(bar)
    print("  Visual token compression — MEASURED, not estimated")
    print(bar)
    print(f"  device        {torch.cuda.get_device_name(0)}")
    print(f"  frames        {args.frames}")
    print(f"  visual tokens {count_visual_tokens(args.frames):,} uncompressed")
    print(f"  hidden        {args.hidden}")
    print(bar)

    results = []
    for strategy in ("none", "dycoke", "tome", "both"):
        res = measure_run(
            strategy, args.keep_ratio, args.frames,
            args.hidden, args.steps, args.batch, args.deepspeed,
        )
        results.append(res)
        print(f"  {strategy:<8} {res['tokens']:>7,} tokens  "
              f"({res['keep_ratio_actual']:>5.1%} kept)  "
              f"peak {res['peak_vram_gb']:>6.2f} GB")

    baseline = results[0]
    print(bar)
    print("  Relative to no compression:")
    for res in results[1:]:
        saving = 1 - res["peak_vram_gb"] / baseline["peak_vram_gb"]
        # The headline number: at a fixed VRAM budget, this is how many more
        # frames the same card can now hold.
        frame_gain = baseline["peak_vram_gb"] / res["peak_vram_gb"]
        print(f"  {res['strategy']:<8} {saving:>6.1%} less peak VRAM   "
              f"-> ~{frame_gain:.2f}x the frames in the same budget")
    print(bar)

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"  wrote {args.output}")
    print("\n  Next: ../03_streaming_memory/ — when the video has no length "
          "at all,\n  a constant FACTOR is not enough and you need a constant "
          "BOUND.")


if __name__ == "__main__":
    main()
