# GPU Validation Scripts

These require **a real GPU and a model download**. They are deliberately kept
out of `tests/run_all.sh` and out of CI, which are CPU-only and must stay fast.

```bash
uv run --with torch --with transformers --with accelerate \
       --with opencv-python-headless --with pillow \
       tests/gpu/validate_llava_vision_path.py
```

Each script **skips cleanly with exit code 0** when no CUDA device is visible,
so running it on a laptop is harmless.

## Why this tier exists

The CPU suite in `tests/` and these scripts answer different questions.

| | `tests/*.py` (CPU) | `tests/gpu/*.py` |
|---|---|---|
| Requires | nothing but `uv` | GPU + model download |
| Runs in CI | **yes**, every push | no |
| Answers | *Is the code structurally correct?* | *Does a real model accept what we build?* |
| Speed | seconds | minutes |

Structural checks cannot tell you whether an assembled batch is something the
model will actually consume. That gap is not hypothetical — it is exactly where
the bug below was hiding.

## `validate_llava_vision_path.py`

Drives the real `preprocess_function` and `LlavaVideoCollator` from
`08_vtt/hf_ds_vtt_test2/llava_video_trainer/video_training_script.py` end to end:

```
video file -> extract_frames_from_file -> processor -> preprocess_function
           -> LlavaVideoCollator -> model.forward -> loss.backward
```

The decisive assertion is the last one: **perturbing `pixel_values` must change
the loss.** A finite loss proves the tensors were shaped acceptably; only this
proves the pixels are actually being read. It is the direct test for the failure
the original placeholder extractor produced, where training ran happily on
data with no visual signal at all.

Uses `llava-interleave-qwen-0.5b-hf`, the smallest model in the same family as
the trainer's `llava-interleave-qwen-7b-hf` default — same processor class, same
architecture, same API contract, about 1/15 the memory. Validating a contract
does not require the large model.

### The bug it found

On its first run it failed, and the failure was real. HuggingFace processors
return token fields **with a batch dimension**:

```python
processed = processor(images=frames, text=prompt, return_tensors=None)
processed["input_ids"]        # [[t0, t1, ..., t2937]]  -- note the nesting
```

`preprocess_function` appended that without unwrapping, so each "example" became
a length-1 list containing a list. `LlavaVideoCollator` then computed
`max(len(f["input_ids"]))` as **1** and padded every sequence to a single token.

This is a nasty class of bug because nothing raises: shapes broadcast, the
forward pass returns a finite loss, and training appears to proceed. Only
asserting on the *sequence length* and on *whether pixels change the loss*
catches it.

The fix unwraps the batch dimension, and
`tests/test_video_frames.py` now guards it **on CPU** with a fake processor that
reproduces the nested shape — so the regression is caught in CI, without a GPU.

## Adding a GPU script

- Skip with exit 0 when `torch.cuda.is_available()` is false.
- Load the code under test from source via `tests/_srcload.py`, so the script
  exercises the shipped implementation rather than a copy.
- Prefer the smallest model that exercises the same API contract.
- Assert on *behaviour* (does the input change the output?), not only on shapes.
- If the behaviour can be checked structurally, add a CPU guard in `tests/` too —
  that is what actually protects the repository day to day.
