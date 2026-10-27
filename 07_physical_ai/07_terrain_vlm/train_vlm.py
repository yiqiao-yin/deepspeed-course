#!/usr/bin/env python3
"""
Fine-tune a VLM to read Martian terrain, and measure whether sight helps.

    # the control first, because it is the cheap one and it bounds the rest
    uv run python train_vlm.py --lora --blind --name blind

    # LoRA on one card. Note: PLAIN python, no `deepspeed` launcher --
    # there is nothing to distribute, and pretending otherwise is the
    # exact cargo cult this lab's capacity table argues against.
    uv run python train_vlm.py --lora --name lora3b

    # full fine-tune, which is the regime that NEEDS ZeRO-3
    uv run deepspeed --num_gpus=8 train_vlm.py --name full7b \
        --model Qwen/Qwen2.5-VL-7B-Instruct --deepspeed ds_config.json

THE QUESTION
------------
Lab 5 part 3 found that its robot told climbable landforms from
impassable ones about 3:1 -- and that a policy fed six constant zeros
did exactly the same. The discrimination came from having a BODY, not
from perception: you learn a scarp is impassable by walking into one.

This lab asks whether SEEING beats that. The bar is therefore not
chance; it is lab 5's measured 3:1, and a model that cannot clear it
has not earned its camera.

THE THREE ARMS, AND WHY THE CONTROL IS NOT OPTIONAL
---------------------------------------------------
    blind   identical pipeline, uniform grey image -- same token
            count, same everything, zero terrain information
    lora    LoRA r=16 on a 3B VLM, one card
    full    full fine-tune of a 7B VLM, ZeRO-3 across 8

`--blind` is the arm that makes the others mean anything, and it is
lab 5's `padded` control transplanted: zero information at identical
dimensionality. Without it, "the VLM scores 85%" is unreadable,
because this dataset's majority class is already 52% and a model that
learned nothing but the prior would look like it was seeing.

WHY LoRA DOES NOT USE ZeRO, AND FULL FINE-TUNING DOES
------------------------------------------------------
Measured by `verify_arch.py`, weights + optimizer, per card:

                        1 card    8 cards
    7B LoRA r=16        15.3 GB    1.9 GB      <- already fits
    7B full fine-tune  119.7 GB   15.0 GB      <- only fits sharded

Running LoRA under ZeRO-3 would shard 15.3 GB that already fitted,
and this course has concluded twice by measurement that reaching for
hardware before checking is the wrong move -- lab 5 found the GPU
1.7x SLOWER for its policy update, lab 6 found GPU physics at 1.50x
against a 5x bar. So the LoRA arm runs plain, and ZeRO-3 is reserved
for the regime where it is load-bearing rather than decorative. It is
the first lab in `07_physical_ai` where DeepSpeed earns its place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam
    kernel and dies with `OSError: CUDA_HOME environment variable is
    not set` from deep inside torch's extension loader, which tells a
    newcomer nothing about what went wrong or what to do next.

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
        return

    print("\n" + "=" * 72)
    print("  NO GPU DETECTED — this script fine-tunes a 3-8B vision model")
    print("=" * 72)
    print("  What you can still run on this machine, right now:\n")
    print("      uv run make_dataset.py --check     # the labels and their"
          " properties")
    print("      uv run verify_arch.py              # where ZeRO-3 becomes"
          " necessary")
    print("      uv run ../../tests/test_terrain_vlm.py\n")
    print("  To rent a GPU for the real thing:\n")
    print("      cd ../../runpod && uv run runpod_ctl.py run \\")
    print("          07_physical_ai/07_terrain_vlm --dry-run --collect \\")
    print("          --wait --terminate --yes\n")
    print("  Set ALLOW_CPU=1 to continue anyway (it will be very slow).")
    print("=" * 72 + "\n")
    sys.exit(1)


def load_split(data: Path, split: str, limit: int | None = None) -> list:
    path = data / f"{split}.jsonl"
    if not path.exists():
        print(f"\n  No dataset at {path}. Build it first:\n\n"
              f"      uv run make_dataset.py --seeds 400\n")
        sys.exit(1)
    rows = [json.loads(l) for l in path.open()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="run")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA r=16 on one card; no ZeRO (see the docstring)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--blind", action="store_true",
                    help="THE CONTROL: uniform grey image, zero terrain "
                         "information at identical token count")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=10,
                    help="must be >0: ds_config leaves warmup_num_steps "
                         "'auto', HuggingFace fills it from this, and "
                         "DeepSpeed rejects 0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap steps; used by the RunPod --dry-run path")
    ap.add_argument("--deepspeed", default=None,
                    help="path to a ds_config; omit for the LoRA arm")
    ap.add_argument("--output", default="runs")
    # parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    # --local_rank=N into every worker's argv, and a strict parser
    # exits 2 with "unrecognized arguments" before training starts.
    a = ap.parse_known_args()[0]

    warmup = a.warmup_steps
    if a.max_steps > 0:
        warmup = max(1, min(warmup, max(1, a.max_steps // 2)))

    require_gpu()

    # Heavy imports AFTER the preflight, so a CPU-only reader gets the
    # message above rather than a CUDA traceback from transformers.
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import AutoProcessor, Trainer, TrainingArguments

    import transformers
    auto = getattr(transformers, "AutoModelForImageTextToText", None) \
        or getattr(transformers, "AutoModelForVision2Seq")

    data = Path(a.data)
    train_rows = load_split(data, "train", a.limit)
    val_rows = load_split(data, "val", a.limit)

    bar = "=" * 72
    print(bar)
    print(f"  Terrain VLM — {a.model}")
    print(bar)
    arm = "blind (CONTROL)" if a.blind else ("LoRA" if a.lora else "full FT")
    print(f"  arm              {arm}")
    print(f"  train / val      {len(train_rows)} / {len(val_rows)}")
    frac = sum(r["answer"] == "AROUND" for r in val_rows) / max(len(val_rows), 1)
    print(f"  val majority     {max(frac, 1-frac):.1%}  <- beat this first")
    print(f"  lab 5 baseline   ~75%  (contact-only discrimination, 3:1)")
    print(f"  ZeRO             {'off — LoRA fits one card' if not a.deepspeed else a.deepspeed}")
    print(bar)

    processor = AutoProcessor.from_pretrained(a.model)

    # ZeRO-3 TRAP: under stage 3 the DeepSpeed config must exist BEFORE
    # from_pretrained, or `zero.Init` never fires and the full model is
    # materialised on every rank. The symptom is an OOM whose requested
    # allocation is trivially small on a card with gigabytes spare.
    # Building TrainingArguments first is what arms it.
    targs = TrainingArguments(
        output_dir=str(Path(a.output) / a.name),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        warmup_steps=warmup,
        max_steps=a.max_steps,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=["wandb"] if (WANDB_AVAILABLE and
                                os.environ.get("WANDB_API_KEY")) else [],
        deepspeed=a.deepspeed,
        gradient_checkpointing=not a.lora,
        remove_unused_columns=False,
    )

    model = auto.from_pretrained(a.model, dtype=torch.bfloat16,
                                 attn_implementation="sdpa")

    if a.lora:
        from peft import LoraConfig, get_peft_model

        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                p.requires_grad = False
            print("  vision tower frozen (LoRA arm)")
        model = get_peft_model(model, LoraConfig(
            r=a.lora_rank, lora_alpha=a.lora_rank * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.print_trainable_parameters()

    GREY = Image.new("RGB", (448, 336), (128, 128, 128))

    class Terrain(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        """
        One training batch, and an ASSERTION that pixels arrived.

        A collator that silently drops `pixel_values` produces a run
        that trains, converges, and has never seen an image -- which
        this repository has shipped before. The check is cheap and the
        failure is invisible without it.
        """
        texts, prompts, images = [], [], []
        for r in batch:
            img = GREY if a.blind else Image.open(data / r["image"]).convert("RGB")
            images.append(img)
            user = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
            prompts.append(processor.apply_chat_template(
                user, tokenize=False, add_generation_prompt=True))
            texts.append(processor.apply_chat_template(
                user + [{"role": "assistant", "content": [
                    {"type": "text", "text": r["answer"]}]}],
                tokenize=False))
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True)
        assert "pixel_values" in enc and enc["pixel_values"].numel() > 0, (
            "no pixel_values in the batch — the vision tower would be "
            "receiving nothing and the run would train happily on text alone")

        # Supervise the ANSWER ONLY.
        #
        # Masking just the padding makes the model spend its capacity
        # learning to reproduce a prompt that is identical on every
        # example -- a 30-token instruction whose loss drowns the one
        # token that carries the task. The first smoke run showed it:
        # train_loss 15.7, almost all of it the prompt.
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for i, p in enumerate(prompts):
            n_prompt = len(processor(text=[p], images=[images[i]],
                                     return_tensors="pt")["input_ids"][0])
            labels[i, :n_prompt] = -100
        assert (labels != -100).any(), (
            "every label masked — nothing to learn from this batch")
        enc["labels"] = labels
        return enc

    trainer = Trainer(model=model, args=targs,
                      train_dataset=Terrain(train_rows),
                      data_collator=collate)
    result = trainer.train()

    out = Path(a.output) / a.name
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    (out / "summary.json").write_text(json.dumps({
        "model": a.model, "arm": arm, "blind": a.blind, "lora": a.lora,
        "deepspeed": a.deepspeed, "epochs": a.epochs,
        "train_n": len(train_rows), "val_n": len(val_rows),
        "max_steps": a.max_steps,
        "val_majority": max(frac, 1 - frac),
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
    }, indent=2))
    print(f"\n  saved {out}")
    if a.max_steps > 0:
        print(f"  NOTE: capped at {a.max_steps} steps — this is a smoke "
              f"test, not a result.\n  A real run is "
              f"{a.epochs} epochs over {len(train_rows)} samples.")
    print(f"  next:  uv run evaluate.py --run {a.name}")


if __name__ == "__main__":
    main()
