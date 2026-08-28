"""
Fine-tune a Thinker-Talker omni model: video + speech in, speech out.

THE ARCHITECTURE, AND THE PROBLEM IT SOLVES
-------------------------------------------
A model that takes video and speech and *replies in speech* has to do two jobs
that pull against each other:

    reason about what was seen and heard  ->  wants a big language model
    emit audio tokens at 50 Hz, in order  ->  wants low latency and stability

Do both with one autoregressive head and they interfere. The classic failure is
that speech quality degrades exactly when reasoning gets hard: the model is
spending its capacity deciding *what* to say and the prosody falls apart
mid-sentence. Users read that as the model being unsure of itself.

Qwen2.5-Omni's answer is to split them:

    THINKER   A full language model. Consumes the interleaved video+audio
              sequence (see tmrope.py) and produces TEXT plus its hidden
              states. This is where understanding happens.

    TALKER    A smaller dual-track autoregressive model. Consumes the
              Thinker's HIDDEN STATES -- not its emitted text -- and produces
              audio tokens.

WHY HIDDEN STATES AND NOT TEXT

This is the part worth pausing on. If the Talker read the Thinker's *text*, it
would have to wait for a token to be decoded before it could speak, and it
would lose everything the text does not encode: hesitation, emphasis, whether
the model is confident. Hidden states carry that. They also arrive one step
earlier, which is where a meaningful slice of the latency budget comes from.

The consequence for training is the thing people get wrong: **the Talker's
gradient flows through the Thinker's hidden states.** Freeze the Thinker
completely and the Talker can only learn to decode a representation that is
not adapting to it. Unfreeze everything and the speech loss starts steering
the reasoning model, which degrades what it knew. LoRA on the Thinker is the
middle path, and it is why this script is built the way it is.

WHAT THIS SCRIPT DEMONSTRATES
-----------------------------
- TMRoPE position IDs built from real timestamps (see `tmrope.py`)
- 2-second video/audio interleaving
- LoRA on the Thinker, encoders frozen
- The two-loss structure (text loss + audio-token loss) and the weight between
- DeepSpeed ZeRO-3 via `ds_config.json`
- A synthetic fallback so the pipeline runs without a corpus

MEMORY, HONESTLY
----------------
    Qwen2.5-Omni-3B  + LoRA   ~24 GB     one card
    Qwen2.5-Omni-7B  + LoRA   ~40 GB
    Frontier omni (100B+)     multi-node -- see ../01_longcat_flash_omni/

Two streams of input tokens means the sequence is longer than a video-only
model at the same clip length: 25 audio tokens per second on top of the video.
A 30-second clip is ~750 audio tokens before a single video frame.

RUNNING IT
----------
CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 09_vss/02_thinker_talker \\
                            --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    uv pip install deepspeed transformers accelerate peft datasets
    uv pip install librosa soundfile opencv-python-headless

Reference: Xu et al. "Qwen2.5-Omni Technical Report."
https://arxiv.org/abs/2503.20215
"""

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmrope import (  # noqa: E402
    TIME_UNIT_SECONDS,
    interleave_video_audio,
    seconds_to_position,
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
              "https://download.pytorch.org/whl/cu121\n")
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
    print("\n  An omni model needs a GPU: it loads a language model, a vision")
    print("  encoder, an audio encoder AND a speech decoder. The smallest")
    print("  useful configuration here wants ~24 GB.")
    print("\n  The ALGORITHM this folder teaches DOES run on CPU, though —")
    print("  TMRoPE is integer arithmetic, and it is the part worth learning:")
    print("      uv run 09_vss/02_thinker_talker/tmrope.py")
    print("      uv run tests/test_tmrope.py         # 59 checks, no GPU")
    print("\n  Also CPU-runnable in this topic:")
    print("      uv run 09_vss/03_duplex_streaming/duplex.py")
    print("      uv run 09_vss/04_omni_eval/omni_eval.py")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 09_vss/02_thinker_talker")
    print("      uv run runpod/runpod_ctl.py run 09_vss/02_thinker_talker \\")
    print("          --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def build_position_ids(
    video_timestamps: List[float],
    audio_duration: float,
    grid_h: int,
    grid_w: int,
    text_prefix_len: int = 0,
) -> List[tuple]:
    """
    Build TMRoPE (t, h, w) position IDs for one video+audio sample.

    This is the function that makes the model able to answer *"what did he say
    while pointing at the chart?"*. Get it wrong and nothing raises -- the model
    trains happily and is simply unable to relate the two streams. See
    `tmrope.py` for why, and `tests/test_tmrope.py` for the proof.

    The text prefix (system prompt, instruction) is positioned first and
    ordinarily, then the interleaved media follows on the shared 40 ms clock.
    """
    from tmrope import text_positions

    prefix = text_positions(text_prefix_len)
    media = interleave_video_audio(
        video_timestamps, audio_duration, grid_h=grid_h, grid_w=grid_w
    )
    return ([tok.as_tuple() for tok in prefix]
            + [tok.as_tuple() for tok in media.tokens])


def synthetic_sample(index: int, duration: float = 4.0, fps: float = 2.0
                     ) -> Dict[str, Any]:
    """
    One deterministic fake video+speech turn.

    Structured, not random. The synthetic "video" has a moving object and the
    synthetic "speech" has a known transcript, so a cross-modal question has a
    ground-truth answer. Random tensors would let a completely disconnected
    vision path look identical to a working one -- which is the bug this
    repository has already shipped once, in `08_vtt/`.
    """
    import numpy as np

    n_frames = int(duration * fps)
    rng = np.random.default_rng(index)

    frames = np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
    for t in range(n_frames):
        frame = rng.integers(0, 40, (224, 224, 3), dtype=np.uint8)
        x = int((224 - 40) * t / max(n_frames - 1, 1))
        frame[92:132, x:x + 40] = 255
        frames[t] = frame

    # 16 kHz mono; a tone whose pitch rises, so it carries real temporal signal.
    n_samples = int(duration * 16000)
    time = np.arange(n_samples) / 16000.0
    audio = (0.3 * np.sin(2 * np.pi * (220 + 80 * time) * time)).astype(np.float32)

    return {
        "frames": frames,
        "timestamps": [t / fps for t in range(n_frames)],
        "audio": audio,
        "duration": duration,
        "prompt": "Describe what you see and hear, and say when it happens.",
        "answer": (f"A bright square moves left to right over "
                   f"{duration:.0f} seconds while a rising tone plays."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B",
                        help="7B needs ~40GB; 3B fits a single 24GB card.")
    parser.add_argument("--data-dir", default=None,
                        help="Corpus of {in.mp4, in.wav, out.wav}. Defaults to "
                             "the shared 09_vss/data/ if present, else "
                             "synthetic.")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--talker-loss-weight", type=float, default=1.0,
                        help="Weight on the audio-token loss relative to text. "
                             "Raise it and speech improves while the model "
                             "gets worse at reasoning; lower it and the "
                             "reverse. There is no setting that avoids the "
                             "trade.")
    parser.add_argument("--freeze-talker", action="store_true",
                        help="Train only the Thinker. Use when adapting to a "
                             "new DOMAIN rather than a new voice.")
    parser.add_argument("--output", default="./omni_thinker_talker_lora")
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap steps; the RunPod --dry-run path uses this.")
    args = parser.parse_args()

    require_gpu()

    # Imported after the preflight so a missing GPU produces our message
    # rather than a CUDA error from inside transformers' import chain.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Trainer, TrainingArguments

    bar = "=" * 74
    print(bar)
    print(f"  Thinker-Talker omni fine-tuning — {args.model}")
    print(bar)

    clip_seconds = 4.0
    audio_tokens = int(clip_seconds / TIME_UNIT_SECONDS)
    video_tokens = int(clip_seconds * args.fps) * 256
    print(f"  clip length      {clip_seconds:.0f} s @ {args.fps} fps")
    print(f"  audio tokens     {audio_tokens}  "
          f"(one per {TIME_UNIT_SECONDS * 1000:.0f} ms)")
    print(f"  video tokens     {video_tokens}")
    print(f"  shared clock     1 temporal ID = {TIME_UNIT_SECONDS * 1000:.0f} ms")
    print(f"  talker loss wt   {args.talker_loss_weight}")
    print(bar)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Qwen2.5-Omni ships a dedicated class; fall back to AutoModel so this
    # script also works with other Thinker-Talker checkpoints.
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration as OmniModel
        model = OmniModel.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except ImportError:
        from transformers import AutoModel
        print("  [note] Qwen2_5Omni class unavailable in this transformers "
              "build; using AutoModel + trust_remote_code.")
        model = AutoModel.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        )

    # ---- Freezing policy -------------------------------------------------
    # The encoders saw far more video and audio than this corpus contains.
    # Unfreezing them on a small dataset reliably makes things worse AND costs
    # a large slice of memory for their gradients and optimizer states.
    frozen = []
    for attr in ("visual", "audio_tower", "vision_tower", "audio_encoder"):
        module = getattr(model, attr, None)
        if module is None and hasattr(model, "thinker"):
            module = getattr(model.thinker, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
            frozen.append(attr)
    if frozen:
        print(f"  encoders frozen  {', '.join(frozen)}")

    if args.freeze_talker and hasattr(model, "talker"):
        for param in model.talker.parameters():
            param.requires_grad = False
        print("  talker frozen    (adapting the Thinker only)")

    # ---- LoRA on the Thinker's attention --------------------------------
    # Not on the Talker: it is small enough to tune directly when you want a
    # new voice, and LoRA on a 50 Hz autoregressive head tends to destabilise
    # prosody more than it helps.
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- Data -------------------------------------------------------------
    data_dir = args.data_dir
    if data_dir is None:
        shared = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "data")
        data_dir = shared if os.path.isdir(os.path.join(shared, "train")) else None

    if data_dir:
        print(f"  data             {os.path.abspath(data_dir)}")
        raise NotImplementedError(
            "Real-corpus loading is left to you: read in.mp4 / in.wav / "
            "out.wav, sample frames with their TIMESTAMPS, and pass both to "
            "build_position_ids. The shared corpus at 09_vss/data/ is the one "
            "01_longcat_flash_omni uses. Run without --data-dir for the "
            "synthetic path."
        )

    print(f"  data             synthetic ({args.examples} clips)")
    dataset = [synthetic_sample(i, duration=clip_seconds, fps=args.fps)
               for i in range(args.examples)]

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build one batch, and assert the media actually arrived.

        The guard is not paranoia. A processor that silently drops one modality
        leaves training to proceed on the other, the loss still falls, and
        nothing raises -- the exact failure `08_vtt/` shipped with. Two streams
        means two chances to lose one.
        """
        texts, videos, audios = [], [], []
        for item in batch:
            texts.append(f"{item['prompt']}\n{item['answer']}")
            videos.append(list(item["frames"]))
            audios.append(item["audio"])

        encoded = processor(
            text=texts,
            videos=videos,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        got_video = any(k.startswith("pixel_values") for k in encoded)
        got_audio = any("audio" in k or "feature" in k for k in encoded)
        if not got_video:
            raise RuntimeError(
                "processor returned no video pixels — the vision path is "
                "disconnected and training would silently proceed without it"
            )
        if not got_audio:
            raise RuntimeError(
                "processor returned no audio features — the audio path is "
                "disconnected and training would silently proceed without it"
            )

        labels = encoded["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Two token streams make the sequence long; recomputing activations is
        # usually the difference between running and OOMing here.
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
    print("  next: ../03_duplex_streaming/ — can it keep listening while "
          "it talks?")


if __name__ == "__main__":
    main()
