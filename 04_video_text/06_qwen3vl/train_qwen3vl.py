#!/usr/bin/env python3
"""
Qwen3-VL LoRA fine-tuning — and a memory probe that answers the OOM question.

    uv run verify_arch.py                              # no GPU, 8 seconds
    uv run deepspeed --num_gpus=1 train_qwen3vl.py     # needs a 48 GB card
    uv run deepspeed --num_gpus=1 train_qwen3vl.py --probe-only

Qwen3-VL is the successor to the model in `02_qwen25vl`, and it is not a version
bump: it adds **DeepStack** (visual features injected at LLM layers 8, 16 and 24
rather than only at the input) and **interleaved MRoPE**, and it is 2.3x the
weights. `verify_arch.py` shows all of that on the meta device in 8 seconds,
with no GPU and no download. Start there.

What this costs, measured on one A40 (47.7 GB)
----------------------------------------------
Not estimated. Measured by this script, `--sweep 4,8,16,24`:

    frames   visual tokens   peak VRAM   % of card
         4             276      22.1 GB        46%
         8             540      25.3 GB        53%
        16           1,068      31.6 GB        66%
        24           1,596      37.9 GB        79%

which fits a straight line almost exactly:

    peak = 18.8 GB floor + 11.97 GB per 1,000 visual tokens

At ~67 visual tokens per 224x224 frame, that gives a usable ceiling of roughly
**33 frames on a 48 GB card**. A 24 GB card is **not viable**: the 18.8 GB floor
alone nearly fills it, and four frames already need 22.1 GB.

The floor is the bf16 weights (17.5 GB) plus fragmentation. It does not shrink
with sequence length, which is why this lab wants one big card rather than two
small ones.

Why the sweep asserts instead of just printing
----------------------------------------------
The first version of this sweep printed a flat 14.2 GB at 4, 8, 16 AND 24
frames and exited 0. It had passed the frames as `videos=`, so the processor
resampled every clip to its default frame rate and handed the model the same
128-token input four times. The table looked like a memory ceiling and was one
measurement repeated.

So the sweep now **raises** if the sequence length does not change across the
sweep. A flat column across a 6x change in input is not a result.
"""

import argparse
import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. From this folder:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return
    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  Qwen3-VL-8B is 17.5 GB of bf16 weights before a single visual")
    print("  token exists. This example cannot run on CPU.")
    print("\n  The ARCHITECTURE can be inspected with no GPU and no download:")
    print("      uv run verify_arch.py     # meta device, ~2 seconds")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py run 04_video_text/06_qwen3vl \\")
    print("          --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """parse_known_args: the launcher injects --local_rank into argv."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--frames", type=int, default=8,
                   help="Frames per clip for the training run.")
    p.add_argument("--sweep", default="4,8,16,24",
                   help="Frame counts to probe for peak VRAM. The point of "
                        "this build: find where it stops fitting.")
    p.add_argument("--probe-only", action="store_true",
                   help="Run the memory sweep and exit, no training.")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    require_gpu()

    import deepspeed
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    torch.cuda.set_device(local_rank)
    bar = "=" * 78

    def say(*a):
        if is_main:
            print(*a, flush=True)

    say(bar)
    say("  Qwen3-VL LoRA — memory probe")
    say(bar)
    props = torch.cuda.get_device_properties(local_rank)
    say(f"  device          {props.name}  {props.total_memory/1e9:.1f} GB")
    say(f"  world size      {world}")
    say(f"  model           {args.model}")

    # The DeepSpeed config must EXIST before from_pretrained, or zero.Init
    # never fires and every rank materialises the whole model. The first run of
    # this probe did exactly that: peak after load was 17.6 GB per rank -- the
    # entire 8.77 B model, unsharded, on both A40s. CLAUDE.md documents this
    # trap; holding HfDeepSpeedConfig in a live variable is what springs it.
    import json
    from transformers.integrations import HfDeepSpeedConfig
    ds_cfg = json.load(open("ds_config.json"))
    dschf = HfDeepSpeedConfig(ds_cfg)          # must stay alive: do not inline

    say("\n  loading weights (bf16, under zero.Init) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)

    # Under ZeRO-3, p.numel() LIES. zero.Init partitions each parameter, so
    # the local tensor is a placeholder and numel() reports 0 -- summing it
    # gives a 8.77 B model a parameter count of zero. The true size is carried
    # on p.ds_numel. This probe found out by dividing by it.
    #
    # Worth keeping as the lab's first lesson about ZeRO-3: the moment
    # zero.Init fires, every naive bit of parameter accounting in your script
    # silently changes meaning. Here it raised. It could as easily have
    # printed "0.00 B" under a success banner.
    def numel(prm) -> int:
        return getattr(prm, "ds_numel", None) or prm.numel()

    tot = sum(numel(p) for p in model.parameters())
    vis = sum(numel(p) for n, p in model.named_parameters() if "visual" in n)
    if tot == 0:
        raise SystemExit("\n  Parameter count is zero even via ds_numel -- "
                         "something is very wrong with the model load.\n")
    say(f"  parameters      {tot/1e9:.2f} B   vision {vis/1e9:.2f} B "
        f"({100*vis/tot:.1f}%)")
    say(f"  (counted via ds_numel: under ZeRO-3, numel() would report 0)")

    # ---- the naming trap, asserted rather than assumed --------------------
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    hits = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets]
    in_vision = [n for n in hits if "visual" in n]
    say(f"\n  LoRA targets {targets}")
    say(f"    resolve to {len(hits)} modules, {len(in_vision)} of them in the "
        f"vision tower")
    if in_vision:
        say("    NOTE: the vision tower IS being adapted.")
    else:
        say("    The vision tower is FROZEN -- not by choice but because its")
        say("    attention is a fused `qkv` that these names cannot match.")

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    if is_main:
        model.print_trainable_parameters()

    model_engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=[p for p in model.parameters()
                                       if p.requires_grad],
        config="ds_config.json")
    device = model_engine.device

    after_load = torch.cuda.max_memory_allocated(local_rank) / 1e9
    say(f"\n  peak after load+init   {after_load:5.1f} GB  "
        f"({100*after_load/(props.total_memory/1e9):.0f}% of the card)")

    # ---- the sweep: where does it stop fitting? ---------------------------
    def one_step(n_frames: int) -> tuple:
        """Build a synthetic clip of n_frames, run fwd+bwd, return (peak, tokens)."""
        torch.cuda.reset_peak_memory_stats(local_rank)
        # 224x224 frames; the processor's 2x2 merger yields ~64 visual tokens
        # per frame at this resolution.
        # PIL images, one per frame, passed as IMAGES not as a video.
        #
        # Passing them as `videos=` let the processor resample to its default
        # fps=24 and silently collapse every frame count to the SAME 128-token
        # input -- so the first sweep reported a flat 14.2 GB at 4, 8, 16 and 24
        # frames and looked like a result. It was measuring one input four
        # times. The processor even warned ("no video metadata was provided"),
        # under a progress bar, and the run exited 0.
        from PIL import Image
        import numpy as np
        frames = [Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(n_frames)]
        content = [{"type": "image"} for _ in range(n_frames)]
        content.append({"type": "text", "text": "Describe this clip."})
        msgs = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        batch = processor(text=[text], images=frames, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        out = model_engine(**batch)
        model_engine.backward(out.loss)
        model_engine.step()
        n_tok = int(batch["input_ids"].shape[1])
        return torch.cuda.max_memory_allocated(local_rank) / 1e9, n_tok, float(out.loss)

    say(f"\n{bar}")
    say("  MEMORY SWEEP — frames -> peak VRAM. This is the OOM boundary.")
    say(bar)
    say(f"  {'frames':>7} {'seq tokens':>11} {'peak GB':>9} {'% card':>7}  status")
    total_gb = props.total_memory / 1e9
    seen_tokens = []
    for n in [int(x) for x in args.sweep.split(",") if x.strip()]:
        try:
            peak, ntok, loss = one_step(n)
            seen_tokens.append(ntok)
            say(f"  {n:>7} {ntok:>11,} {peak:>9.1f} "
                f"{100*peak/total_gb:>6.0f}%  ok (loss {loss:.3f})")
        except torch.cuda.OutOfMemoryError:
            say(f"  {n:>7} {'-':>11} {'-':>9} {'-':>7}  OOM  <-- boundary")
            torch.cuda.empty_cache()
            break
        except Exception as exc:                          # noqa: BLE001
            say(f"  {n:>7}  FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            torch.cuda.empty_cache()
            break

    # A sweep whose input never changed is not a sweep. Fail LOUDLY rather
    # than print a flat column that reads like a memory ceiling.
    if len(set(seen_tokens)) <= 1 and len(seen_tokens) > 1:
        raise SystemExit(
            f"\n  SWEEP INVALID: sequence length was {seen_tokens[0]} at every "
            f"frame count {seen_tokens}.\n  The processor collapsed every input "
            f"to the same thing, so the peak-memory column\n  measures one input "
            f"repeated and means nothing. Fix the input before\n  reading any "
            f"number above.\n")

    if not args.probe_only:
        say(f"\n{bar}")
        say(f"  TRAINING — {args.max_steps} steps at {args.frames} frames")
        say(bar)
        for step in range(args.max_steps):
            try:
                peak, ntok, loss = one_step(args.frames)
                if step % 2 == 0:
                    say(f"  step {step:>3} | loss {loss:7.4f} | peak {peak:5.1f} GB")
            except torch.cuda.OutOfMemoryError:
                say(f"  step {step:>3} | OOM at {args.frames} frames")
                break

    say(f"\n{bar}")
    say("  Probe complete.")
    say(bar)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
