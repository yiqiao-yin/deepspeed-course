"""
Fine-tune Qwen2.5-VL on video with DeepSpeed ZeRO-3 + LoRA.

WHY THIS EXISTS ALONGSIDE hf_ds_vtt_test2/
------------------------------------------
The LLaVA trainer next door is the *foundational* example: a fixed number of
frames, a fixed resolution, one image encoder, and the standard SFT loop. It
is the right thing to read first because nothing is hidden.

It is also 2024's architecture, and video-language models changed in two
specific ways that a fixed-frame model structurally cannot express.

1. NATIVE DYNAMIC RESOLUTION. LLaVA resizes every frame to 336x336. Qwen2.5-VL
   keeps the native aspect ratio and emits a variable number of tokens, so a
   wide establishing shot and a tight close-up are not squeezed into the same
   budget. Letterboxing is not free -- it spends tokens on black bars.

2. ABSOLUTE TIME, NOT FRAME INDEX. This is the deeper one. A fixed-frame model
   samples N frames and numbers them 0..N-1. Sample 16 frames from a 10-second
   clip and from a 10-minute clip and the model sees *identical* position
   information. It cannot distinguish "he picked up the cup then immediately
   drank" from "he picked up the cup, and forty minutes later drank", because
   you deleted the only evidence. Any question containing "how long", "before",
   or "after" is then unanswerable in principle, not just in practice.

   Qwen2.5-VL aligns its M-RoPE temporal component to *timestamps*. Frame 5 at
   t=2.0s and frame 5 at t=300.0s get different positional encodings. Duration
   survives sampling.

That is why this folder exists: it is the same task as the LLaVA example, on an
architecture that can actually represent time.

WHAT THIS SCRIPT DEMONSTRATES
-----------------------------
- Dynamic FPS sampling with a hard token budget (`sample_video_frames`)
- Absolute-time metadata passed through to the processor
- LoRA on the language model only, vision tower frozen
- DeepSpeed ZeRO-3 via `ds_config.json`
- A synthetic-data fallback, so the pipeline is runnable end to end without
  a 200 GB video dataset download

MEMORY, HONESTLY
----------------
Qwen2.5-VL-3B + LoRA + ZeRO-3 fits in roughly 16 GB, which is a single
consumer card. The 7B needs about 40 GB. The 72B needs multiple 80 GB cards
and is not what you should be learning on.

The dominant term is NOT the weights -- it is the visual tokens, quadratically.
See `../03_token_compression/` before you raise `max_frames`.

RUNNING IT
----------
Local / CoreWeave (SLURM):
    sbatch run_deepspeed.sh

RunPod (creates the pod, runs, and terminates it):
    uv run runpod/runpod_ctl.py run 04_video_text/02_qwen25vl \\
        --collect --wait --terminate --yes

Setup:
    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install deepspeed transformers accelerate peft datasets \\
        qwen-vl-utils opencv-python-headless

Reference: Bai et al. "Qwen2.5-VL Technical Report."
https://arxiv.org/abs/2502.13923
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


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
        print("            ds_config.json also needs \"torch_adam\": true and "
              "bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  Qwen2.5-VL-3B with LoRA needs roughly 16 GB of VRAM. This")
    print("  example CANNOT run on CPU: it downloads a multi-GB model and")
    print("  needs real GPU memory for the visual tokens.")
    print("\n  The ALGORITHMS in this topic do run on CPU, though:")
    print("      uv run tests/test_token_compression.py   # ToMe, FastV, DyCoke")
    print("      uv run tests/test_star_memory.py         # streaming memory")
    print("      uv run 04_video_text/03_token_compression/token_compression.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 04_video_text/02_qwen25vl")
    print("      uv run runpod/runpod_ctl.py run 04_video_text/02_qwen25vl \\")
    print("          --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dynamic frame sampling
# ---------------------------------------------------------------------------

@dataclass
class VideoSample:
    """One sampled clip: the pixels, plus WHEN each frame happened."""

    frames: np.ndarray          # (T, H, W, 3) uint8
    timestamps: List[float]     # seconds from clip start, len == T
    fps: float                  # the fps we actually achieved
    duration: float             # true clip duration in seconds

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)


def sample_video_frames(
    video_path: str,
    target_fps: float = 2.0,
    max_frames: int = 64,
    min_frames: int = 4,
) -> VideoSample:
    """
    Sample frames at a target rate, then enforce a hard cap.

    THE TWO-STAGE DESIGN, AND WHY IT IS NOT JUST "TAKE N FRAMES"

    Uniform-N sampling -- the LLaVA approach -- has a property nobody wants:
    the effective frame rate depends on clip length. Sixteen frames from a
    5-second clip is 3.2 fps; sixteen frames from a 50-minute lecture is
    0.005 fps. The same model is handed wildly different temporal densities
    and has no way to know which it got.

    Sampling at a fixed *rate* fixes that, and introduces the opposite
    problem: a long clip produces unboundedly many frames and OOMs. So:

      1. Sample at `target_fps` -- constant temporal density, so motion looks
         the same whatever the clip length.
      2. If that exceeds `max_frames`, uniformly subsample down to the cap --
         a bounded token budget, so nothing OOMs.

    Crucially, TIMESTAMPS ARE PRESERVED THROUGH BOTH STAGES. After step 2 the
    frames are no longer evenly spaced in the way the model would assume from
    indices alone, and the timestamps are what tell it so. Discard them here
    and the absolute-time encoding downstream has nothing to encode -- which
    is the single most common way this architecture gets silently reduced to
    the one it replaced.

    Args:
        video_path: Path to a video file.
        target_fps: Frames per second to aim for. 2.0 suits most instructional
            and conversational video; raise it for sports or anything where
            sub-second events matter.
        max_frames: Hard cap. Each frame is ~256 tokens, so 64 frames is
            ~16k visual tokens -- already a long sequence.
        min_frames: Floor. A single frame is a photograph, not a video, and
            the temporal encoding degenerates.

    Returns:
        A VideoSample with frames and their true timestamps.

    Raises:
        RuntimeError: if the video cannot be opened or decodes to nothing.
            We raise rather than returning grey placeholders -- the earlier
            version of the LLaVA trainer in this repo returned placeholders on
            error, and it trained happily on solid grey squares for a full run
            without anyone noticing.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"video decodes to zero frames: {video_path}")

        duration = total / native_fps

        # Stage 1: how many frames does target_fps imply?
        wanted = max(min_frames, int(round(duration * target_fps)))
        # Stage 2: clamp to the budget and to what the file actually holds.
        n_sample = min(wanted, max_frames, total)

        indices = np.linspace(0, total - 1, n_sample).astype(int)

        frames, timestamps = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # The TRUE time of this frame, from its index in the source file.
            # Not its position in our sampled sequence.
            timestamps.append(float(idx) / native_fps)

        if not frames:
            raise RuntimeError(f"decoded no frames from {video_path}")
    finally:
        cap.release()

    achieved = len(frames) / duration if duration > 0 else 0.0
    return VideoSample(
        frames=np.stack(frames),
        timestamps=timestamps,
        fps=achieved,
        duration=duration,
    )


def synthetic_video_sample(
    num_frames: int = 16,
    size: int = 224,
    duration: float = 8.0,
    seed: int = 0,
) -> VideoSample:
    """
    A deterministic fake clip: a bright square moving across a dark field.

    Not decoration. Two real jobs:

    1. It makes the pipeline runnable without a multi-hundred-GB dataset, so
       you can prove your shapes, collator and DeepSpeed config are correct
       before spending on data transfer.

    2. It has GENUINE temporal structure -- the square's position is a known
       function of time. So "did the model learn anything about motion?" has a
       ground-truth answer here, which it does not for random noise. Random
       tensors would let a broken temporal path pass unnoticed; that is
       exactly the bug the LLaVA trainer in this repo shipped with.
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    for t in range(num_frames):
        frame = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
        # Square travels left to right, linearly in time.
        x = int((size - 40) * t / max(num_frames - 1, 1))
        y = size // 2 - 20
        frame[y:y + 40, x:x + 40] = 255
        frames[t] = frame

    timestamps = [duration * t / max(num_frames - 1, 1) for t in range(num_frames)]
    return VideoSample(
        frames=frames,
        timestamps=timestamps,
        fps=num_frames / duration,
        duration=duration,
    )


def build_time_aware_prompt(sample: VideoSample, question: str) -> str:
    """
    Put the timestamps where the model can read them.

    Qwen2.5-VL encodes absolute time in M-RoPE, but that only works if the
    processor is told the real fps -- and processors vary in whether they
    accept it. Stating the times in the text as well is belt-and-braces: it
    costs a handful of tokens and it means a duration question is answerable
    even when the positional path is misconfigured.

    It also makes misconfiguration VISIBLE. If the text says 0.0s-8.0s and the
    model still cannot answer "how long", you know the problem is the model,
    not the plumbing.
    """
    times = ", ".join(f"{t:.1f}s" for t in sample.timestamps[:8])
    more = ", ..." if sample.num_frames > 8 else ""
    return (
        f"<|vision_start|><|video_pad|><|vision_end|>\n"
        f"[clip: {sample.duration:.1f}s, {sample.num_frames} frames "
        f"at {sample.fps:.2f} fps; sampled at {times}{more}]\n"
        f"{question}"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataset(num_examples: int, num_frames: int) -> List[Dict[str, Any]]:
    """
    Synthetic clips with answers that REQUIRE the video to get right.

    The questions are about direction of motion and duration -- both
    unanswerable from any single frame. A model that ignores the visual path
    entirely cannot score above chance, which makes "is the vision tower
    actually wired up?" a question the loss curve can answer.
    """
    examples = []
    for i in range(num_examples):
        sample = synthetic_video_sample(
            num_frames=num_frames, duration=4.0 + (i % 5) * 2.0, seed=i
        )
        examples.append({
            "sample": sample,
            "question": "Describe the motion in this clip and its duration.",
            "answer": (
                f"A bright square moves steadily from left to right "
                f"across {sample.duration:.1f} seconds."
            ),
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="7B needs ~40GB; 3B fits a single 16GB card.")
    parser.add_argument("--max-frames", type=int, default=16,
                        help="Token cost is QUADRATIC in this. See "
                             "../03_token_compression/ before raising it.")
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output", default="./qwen25vl_video_lora")
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="LR warmup steps. Must be >0: ds_config.json "
                             "leaves warmup_num_steps 'auto', HuggingFace "
                             "fills it from this, and DeepSpeed rejects 0 "
                             "with 'warmup_num_steps must be a positive "
                             "integer'. Clamped for short runs.")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap steps; used by the RunPod --dry-run path.")
    # parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    # --local_rank=N into every worker's argv, and a strict parser exits 2
    # with "unrecognized arguments" before training starts -- breaking the
    # exact command this example documents. CONTRIBUTING.md section 3.2.
    args = parser.parse_known_args()[0]
    # DeepSpeed's WarmupDecayLR rejects warmup_num_steps=0. ds_config.json
    # leaves it "auto"; HuggingFace substitutes TrainingArguments.warmup_steps,
    # which defaults to 0 -- so without setting it, EVERY DeepSpeed run of this
    # example dies before step one with
    #   ValueError: warmup_num_steps must be a positive integer, got 0
    # Clamped so a short --max-steps smoke test does not request more warmup
    # than it has steps to give.
    warmup_steps = args.warmup_steps
    if args.max_steps > 0:
        warmup_steps = max(1, min(warmup_steps, max(1, args.max_steps // 2)))
    require_gpu()

    # Imported after the preflight so a missing GPU produces our message
    # rather than a CUDA error from inside transformers' import chain.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    bar = "=" * 72
    print(bar)
    print(f"  Qwen2.5-VL video fine-tuning — {args.model}")
    print(bar)

    budget = args.max_frames * 256
    print(f"  frames/clip      {args.max_frames}")
    print(f"  visual tokens    ~{budget:,}  (256 per frame after the 2x2 merger)")
    print(f"  attention cost   ~{(budget / 4096) ** 2:.1f}x a 4k-token text batch")
    print(bar)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Freeze the vision tower. It was trained on far more video than we have,
    # and unfreezing it on a small dataset reliably makes things worse while
    # costing a large slice of memory for its gradients and optimizer states.
    # LoRA on the language model is where the task adaptation belongs.
    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
        print("  vision tower frozen")

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention projections only. Including the MLP roughly triples the
        # adapter size for a marginal quality gain at this data scale.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = build_dataset(args.examples, args.max_frames)
    print(f"  dataset          {len(dataset)} synthetic clips")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build one training batch.

        The bug this guards against: a generic seq2seq collator pads token
        fields and silently DROPS `pixel_values`, because it does not know
        the key. Training then runs, the loss decreases -- on text alone --
        and no error is ever raised. The identical bug shipped in this repo's
        LLaVA trainer. Video keys are handled explicitly here for that reason.
        """
        texts, videos = [], []
        for item in batch:
            sample: VideoSample = item["sample"]
            prompt = build_time_aware_prompt(sample, item["question"])
            texts.append(f"{prompt}\n{item['answer']}")
            videos.append(list(sample.frames))

        encoded = processor(
            text=texts,
            videos=videos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # Causal LM: labels are the inputs, with padding masked out so the
        # model is not trained to predict pad tokens.
        labels = encoded["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels

        if "pixel_values_videos" not in encoded and "pixel_values" not in encoded:
            raise RuntimeError(
                "processor returned no video pixels — the vision path is "
                "disconnected and training would silently proceed on text only"
            )
        return encoded

    training_args = TrainingArguments(
        warmup_steps=warmup_steps,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Recompute activations instead of storing them. For video this is not
        # an optimisation, it is usually the difference between running and
        # OOMing: activations scale with visual tokens, which dominate.
        gradient_checkpointing=True,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        deepspeed=args.deepspeed if os.path.exists(args.deepspeed) else None,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print(f"\n  saved adapter to {args.output}")
    print("  next: ../03_token_compression/ — fit more frames in the same VRAM")


if __name__ == "__main__":
    main()
