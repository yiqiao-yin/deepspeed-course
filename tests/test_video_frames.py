# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "opencv-python-headless", "pillow"]
# ///
"""
Regression test: video frame extraction must produce REAL, DISTINCT frames.

Run:
    uv run tests/test_video_frames.py

Background
----------
`04_video_text/.../video_training_script.py` originally shipped a placeholder:

    # For this example, we'll use a placeholder image repeated
    ...
    return [image] * num_frames

Every "video" therefore became N copies of one still image — a fixed COCO
photograph, or a solid grey square on error. Training ran and the loss
decreased, but there was ZERO temporal signal, so the model could not learn
anything about motion or change. The function was additionally never called
by preprocess_function, so no pixels reached the model at all.

This test loads the ACTUAL shipped function via AST extraction (no torch or
transformers needed) and verifies it decodes genuinely different frames.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, load_function, source_contains  # noqa: E402

SCRIPT = "04_video_text/01_hf_baseline/llava_video_trainer/video_training_script.py"


def make_ramp_video(path: Path, n_frames: int = 100, size: int = 64) -> None:
    """Write a video whose frame i is a solid colour encoding i in the red channel."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size)
    )
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), np.uint8)
        frame[:, :, 2] = min(i * 2, 255)      # BGR: index 2 is RED
        writer.write(frame)
    writer.release()


def main() -> int:
    r = Results("Video frame extraction — temporal signal regression test")

    # ---- 1. Source-level guards ---------------------------------------
    r.check(
        not source_contains(SCRIPT, "return [image] * num_frames\n            else:"),
        "placeholder 'repeat one image' fallback is gone",
    )
    r.check(
        source_contains(SCRIPT, "cv2.COLOR_BGR2RGB"),
        "converts BGR to RGB",
        "OpenCV decodes BGR; the vision encoder expects RGB. Skipping this "
        "silently degrades accuracy rather than raising.",
    )
    r.check(
        source_contains(SCRIPT, "self.download_and_process_video_frames(video_url, num_frames)"),
        "preprocess_function actually CALLS the extractor",
        "Previously the extractor was defined but never invoked, so no pixels "
        "reached the model.",
    )
    r.check(
        source_contains(SCRIPT, "pixel_values"),
        "pixel_values are produced and carried through to the batch",
    )
    r.check(
        source_contains(SCRIPT, "class LlavaVideoCollator"),
        "a collator exists that batches pixel_values",
        "DataCollatorForSeq2Seq silently drops non-token keys.",
    )

    # ---- 2. Behavioural test against the real shipped function ---------
    extract = load_function(
        SCRIPT,
        "extract_frames_from_file",
        class_name="VideoTextTrainer",
        extra_globals={"Image": Image, "List": list},
    )

    tmp = Path("_test_ramp_video.mp4")
    try:
        make_ramp_video(tmp)

        for n in (5, 8, 16):
            frames = extract(str(tmp), n)
            reds = [int(np.array(f)[0, 0, 0]) for f in frames]
            r.check(
                len(frames) == n,
                f"num_frames={n}: returns exactly {n} frames",
                f"got {len(frames)}",
            )
            r.check(
                len(set(reds)) == n,
                f"num_frames={n}: all frames are DISTINCT (real temporal sampling)",
                f"red values {reds} — duplicates mean placeholder behaviour is back",
            )
            r.check(
                reds == sorted(reds),
                f"num_frames={n}: frames come back in temporal order",
                f"red values {reds}",
            )

        # Frames must span the clip, not cluster at the start.
        frames = extract(str(tmp), 5)
        reds = [int(np.array(f)[0, 0, 0]) for f in frames]
        r.check(
            reds[0] < 10 and reds[-1] > 180,
            "sampling spans the whole clip (first ~0, last ~198)",
            f"first={reds[0]} last={reds[-1]}",
        )

        # BGR->RGB: our video is pure red. In RGB, channel 0 must dominate.
        px = np.array(extract(str(tmp), 2)[1])[0, 0]
        r.check(
            px[0] > 150 and px[1] < 10 and px[2] < 10,
            "colour channels are correct after BGR->RGB conversion",
            f"pixel={tuple(int(v) for v in px)} — if channel 2 dominated, the "
            f"conversion is missing",
        )

        # Single frame
        r.check(len(extract(str(tmp), 1)) == 1, "num_frames=1 works")

        # ---- 3. preprocess_function unwraps the processor batch dim ----
        # HuggingFace processors return token fields WITH a batch dimension
        # even for a single text: input_ids is [[t0, t1, ...]]. Failing to
        # unwrap it made every sequence a length-1 list containing a list, so
        # the collator padded everything to length 1 and silently destroyed
        # the input. Caught only by running on a GPU; guarded here on CPU.
        preprocess = load_function(
            SCRIPT, "preprocess_function", class_name="VideoTextTrainer",
            extra_globals={"List": list, "Dict": dict, "Any": object},
        )

        class FakeProcessor:
            """Mimics the nested-output shape of a real HF processor."""
            def apply_chat_template(self, conversation, **kw):
                return "<image>" * NUM_FAKE_FRAMES + " describe"

            def __call__(self, images, text, **kw):
                return {
                    "input_ids": [[101, 102, 103, 104, 105]],      # NESTED
                    "attention_mask": [[1, 1, 1, 1, 1]],           # NESTED
                    "pixel_values": [[0.0]] * len(images),         # per-frame
                }

        NUM_FAKE_FRAMES = 3

        class FakeTrainer:
            processor = FakeProcessor()

            def download_and_process_video_frames(self, url, n):
                return [Image.new("RGB", (8, 8))] * n

        result = preprocess(FakeTrainer(), {
            "conversation": [[], []],
            "video_url": ["a.mp4", "b.mp4"],
            "num_frames": [NUM_FAKE_FRAMES, NUM_FAKE_FRAMES],
        })

        ids0 = result["input_ids"][0]
        r.check(
            all(isinstance(t, int) for t in ids0),
            "preprocess_function flattens the processor's batch dimension",
            f"got {ids0!r} — nested lists here pad every sequence to length 1",
        )
        r.check(len(ids0) == 5, f"sequence keeps all 5 tokens (got {len(ids0)})")
        r.check(
            len(result["attention_mask"][0]) == len(ids0),
            "attention_mask is flattened and length-matched",
        )
        r.check(
            len(result["labels"][0]) == len(ids0),
            "labels are length-matched to input_ids",
        )
        r.check(
            len(result["pixel_values"][0]) == NUM_FAKE_FRAMES,
            "pixel_values keeps one entry per frame (no batch dim to strip)",
        )

        # ---- 4. Failures must RAISE, not return placeholders ----------
        for bad, label in (
            ("does_not_exist.mp4", "missing file"),
            (__file__, "a non-video file"),
        ):
            try:
                extract(bad, 4)
                r.check(False, f"{label} raises instead of returning placeholders",
                        "Returned frames for invalid input — silent degradation.")
            except Exception as exc:
                r.check(
                    isinstance(exc, (ValueError, ImportError)),
                    f"{label} raises {type(exc).__name__}",
                )
    finally:
        if tmp.exists():
            tmp.unlink()

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
