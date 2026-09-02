"""
GPU validation of the LLaVA video vision path.

    uv run --with torch --with transformers --with accelerate \
           --with opencv-python-headless --with pillow \
           tests/gpu/validate_llava_vision_path.py

REQUIRES A GPU and downloads a ~1 GB model. This is NOT part of the CPU test
suite and does not run in CI — see tests/gpu/README.md.

What this covers that the CPU tests cannot
------------------------------------------
The CPU suite (tests/test_video_frames.py) checks structure: that frames are
distinct, that the batch dimension is unwrapped, that the collator keeps
pixel_values. It cannot check that the assembled batch is something a real
LLaVA model will accept, because that requires the model.

This script drives the ACTUAL shipped preprocess_function and LlavaVideoCollator
against a real model and asserts the whole path:

    video file -> extract_frames_from_file -> processor -> preprocess_function
               -> LlavaVideoCollator -> model.forward -> loss.backward

The decisive assertion is the last one: perturbing pixel_values must CHANGE the
loss. If it does not, the visual input is not reaching the model — which is
precisely the failure the original placeholder extractor produced, and which a
finite loss alone would not reveal.

It found a real bug on first run: HuggingFace processors return token fields
with a batch dimension, so input_ids is [[t0, t1, ...]]. preprocess_function
appended that unwrapped, making every sequence a length-1 list containing a
list; the collator then padded everything to length 1.

Model choice
------------
Uses llava-interleave-qwen-0.5b-hf — the smallest member of the same family as
the trainer's default (llava-interleave-qwen-7b-hf). Same processor class, same
architecture, same API contract, roughly 1/15 the memory. Validating the
contract does not require the large model.
"""

import sys
from pathlib import Path

# Heavy imports (cv2, PIL, transformers) are deliberately deferred into main()
# so the no-GPU skip path works on a machine that has none of them installed.
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tests"))
from _srcload import load_function  # noqa: E402

SCRIPT = REPO / "04_video_text/01_hf_baseline/llava_video_trainer/video_training_script.py"
MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"
NUM_FRAMES = 4

passed = failed = 0


def check(cond, label, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")
        if detail:
            print(f"        {detail}")
    return cond


def make_ramp_video(path: Path, n: int = 80, size: int = 128) -> None:
    """A video whose frame i is a solid colour encoding i — so distinct frames are provable."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for i in range(n):
        frame = np.zeros((size, size, 3), np.uint8)
        frame[:, :, 2] = min(i * 3, 255)           # BGR: channel 2 is red
        writer.write(frame)
    writer.release()


def main() -> int:
    print("=" * 72)
    print("LLaVA video vision path — GPU validation")
    print("=" * 72)

    if not torch.cuda.is_available():
        print("\n  SKIP: no CUDA device visible. This script requires a GPU.")
        print("  The structural guards run on CPU via:  uv run tests/test_video_frames.py")
        return 0

    print(f"  device: {torch.cuda.get_device_name(0)}")

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    video = Path("_gpu_val_video.mp4")
    try:
        make_ramp_video(video)

        # ---- the real shipped functions --------------------------------
        extract = load_function(
            SCRIPT, "extract_frames_from_file", class_name="VideoTextTrainer",
            extra_globals={"Image": Image, "List": list},
        )
        preprocess = load_function(
            SCRIPT, "preprocess_function", class_name="VideoTextTrainer",
            extra_globals={"List": list, "Dict": dict, "Any": object},
        )
        collate = load_function(
            SCRIPT, "__call__", class_name="LlavaVideoCollator",
            extra_globals={"torch": torch, "List": list, "Dict": dict, "Any": object},
        )

        frames = extract(str(video), NUM_FRAMES)
        reds = [int(np.array(f)[0, 0, 0]) for f in frames]
        check(len(frames) == NUM_FRAMES, f"extracted {NUM_FRAMES} frames")
        check(len(set(reds)) == NUM_FRAMES, "frames are distinct", f"reds={reds}")

        print(f"\n  loading {MODEL_ID} ...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to("cuda")
        model.config.use_cache = False
        print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")

        class FakeTrainer:
            """Supplies only what preprocess_function touches."""
            def __init__(self):
                self.processor = processor
                self.num_frames = NUM_FRAMES

            def download_and_process_video_frames(self, url, n):
                return extract(url, n)

        class FakeCollator:
            label_pad_token_id = -100
            pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        content = [{"type": "image"} for _ in range(NUM_FRAMES)]
        content.append({"type": "text", "text": "What happens in this video?"})
        conversation = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "The frame turns red."}]},
        ]

        # datasets.map(batched=True) hands columns as lists — mimic that exactly.
        out = preprocess(FakeTrainer(), {
            "conversation": [conversation, conversation],
            "video_url": [str(video), str(video)],
            "num_frames": [NUM_FRAMES, NUM_FRAMES],
        })

        tok0 = out["input_ids"][0]
        check(len(out["input_ids"]) == 2, "preprocess returned 2 examples")
        check(all(isinstance(t, int) for t in tok0),
              "input_ids is FLAT (processor batch dim unwrapped)",
              f"first={tok0[:5]} — nested lists pad every sequence to length 1")
        check(len(tok0) > 100, f"realistic sequence length ({len(tok0)} tokens)",
              "length 1 means the batch dimension was not unwrapped")
        check(len(out["attention_mask"][0]) == len(tok0), "attention_mask length matches")
        check(len(out["labels"][0]) == len(tok0), "labels length matches")
        check(np.array(out["pixel_values"][0]).shape[0] == NUM_FRAMES,
              f"pixel_values holds {NUM_FRAMES} frames per example")

        features = [
            {k: (out[k][i] if k != "pixel_values" else torch.tensor(np.array(out[k][i])))
             for k in ("input_ids", "attention_mask", "labels", "pixel_values")}
            for i in range(2)
        ]
        batch = collate(FakeCollator(), features)

        check("pixel_values" in batch, "collator RETAINED pixel_values",
              "DataCollatorForSeq2Seq would silently drop them")
        check(batch["input_ids"].dim() == 2,
              f"batched input_ids is 2-D {tuple(batch['input_ids'].shape)}")
        check(batch["pixel_values"].shape[0] == 2 * NUM_FRAMES,
              f"frames concatenated across batch = {2 * NUM_FRAMES}",
              f"got {tuple(batch['pixel_values'].shape)}")

        batch = {k: v.to("cuda") for k, v in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(torch.float32)

        result = model(**batch)
        check(torch.isfinite(result.loss), f"forward: finite loss = {result.loss.item():.4f}")

        result.loss.backward()
        gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum()
                               for p in model.parameters() if p.grad is not None)).item()
        check(np.isfinite(gnorm), f"backward: finite grad norm = {gnorm:.4f}")

        vision = [n for n, p in model.named_parameters()
                  if p.grad is not None and ("vision" in n or "multi_modal" in n)]
        check(len(vision) > 0, f"vision tower received gradients ({len(vision)} tensors)")

        # ---- the decisive check ---------------------------------------
        model.zero_grad()
        with torch.no_grad():
            loss_a = model(**batch).loss.item()
            perturbed = dict(batch)
            perturbed["pixel_values"] = torch.randn_like(batch["pixel_values"])
            loss_b = model(**perturbed).loss.item()

        check(abs(loss_a - loss_b) > 1e-4,
              f"pixels AFFECT the loss ({loss_a:.4f} -> {loss_b:.4f}, "
              f"d={abs(loss_a - loss_b):.4f})",
              "An unchanged loss means the visual input never reaches the model.")
    finally:
        video.unlink(missing_ok=True)

    print(f"\n  {passed}/{passed + failed} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
