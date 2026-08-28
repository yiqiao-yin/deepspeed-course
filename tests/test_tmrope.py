# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: TMRoPE must put video and audio on the SAME clock.

Run:
    uv run tests/test_tmrope.py

Why this suite exists
---------------------
Cross-modal position alignment fails silently. Get it wrong and the model
trains, the loss falls, and the only symptom is that questions requiring audio
and video to line up -- *"what did he say while pointing at the whiteboard?"* --
are answered at chance. That is indistinguishable from an undertrained model,
so you will spend a week on the learning rate.

Position assignment is integer arithmetic, so it can be *proved* on a laptop
instead. That is the entire reason `tmrope.py` contains no tensors.

The trap this pins down
-----------------------
Numbering video frames by INDEX rather than by timestamp is the obvious
approach, and it has a genuinely nasty property: **at exactly 25 fps it is
correct.** 25 fps is 40 ms per frame, which is exactly the temporal tick, so
index and time coincide and drift is zero.

Test on 25 fps footage and everything works. Ship it. Then someone feeds it
2 fps and by the one-minute mark video and audio positions describing the same
instant are ~1,380 IDs apart, and nothing raises. This suite asserts both
halves: that the coincidence is real, and that it does not generalise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "09_vss" / "02_thinker_talker"))

from tmrope import (  # noqa: E402
    CHUNK_SECONDS,
    TIME_UNIT_SECONDS,
    InterleavedSequence,
    audio_positions,
    image_positions,
    interleave_video_audio,
    naive_index_positions,
    position_to_seconds,
    seconds_to_position,
    text_positions,
    video_positions,
)


def test_the_shared_clock(r: Results) -> None:
    """One temporal ID is 40 ms, for every modality, without exception."""
    r.check(abs(TIME_UNIT_SECONDS - 0.04) < 1e-9,
            "the time unit is 40 ms",
            f"got {TIME_UNIT_SECONDS}")

    r.check(seconds_to_position(0.0) == 0, "t=0s is position 0")
    r.check(seconds_to_position(0.04) == 1, "t=40ms is position 1")
    r.check(seconds_to_position(1.0) == 25, "t=1s is position 25")
    r.check(seconds_to_position(60.0) == 1500, "t=60s is position 1500")

    # Round, do not truncate. A consistent sub-tick bias applied to one stream
    # and not the other is exactly the skew this design exists to prevent.
    r.check(seconds_to_position(0.039) == 1,
            "rounds to nearest (39ms -> 1, not 0)",
            f"got {seconds_to_position(0.039)}")
    r.check(seconds_to_position(0.021) == 1,
            "rounds up past the half-tick",
            f"got {seconds_to_position(0.021)}")

    # Round trip must be exact on tick boundaries.
    for pos in (0, 1, 25, 1500):
        r.check(seconds_to_position(position_to_seconds(pos)) == pos,
                f"position {pos} round-trips through seconds")

    try:
        seconds_to_position(-1.0)
        caught = False
    except ValueError:
        caught = True
    r.check(caught, "negative time is rejected")


def test_video_audio_alignment(r: Results) -> None:
    """
    THE load-bearing property: same instant -> same temporal ID, cross-modally.

    If this ever fails, the model cannot connect a gesture to the word it
    accompanied, and no amount of training will fix it.
    """
    for moment in (0.0, 0.5, 1.0, 2.5, 10.0, 137.44):
        vid = video_positions([moment], grid_h=1, grid_w=1)[0]
        aud = audio_positions(1, start_seconds=moment)[0]
        r.check(vid.t == aud.t,
                f"video and audio at t={moment}s share a temporal ID",
                f"video t={vid.t}, audio t={aud.t} — misalignment here means "
                "co-occurrence is unrepresentable")

    # Frame rate must not change the encoding of a given moment.
    ids = []
    for fps in (1, 2, 5, 10, 25):
        stamps = [i / fps for i in range(int(4 * fps))]
        nearest = min(stamps, key=lambda s: abs(s - 2.0))
        ids.append(video_positions([nearest], 1, 1)[0].t)
    r.check(len(set(ids)) == 1 and ids[0] == 50,
            "t=2.00s is position 50 at every sampling rate tried",
            f"got {ids} across 1/2/5/10/25 fps — sampling rate must be a "
            "resolution choice, not a semantic one")


def test_naive_indexing_is_wrong(r: Results) -> None:
    """
    Frame-index positions drift from the audio clock — except at 25 fps.

    Both halves matter. Without the first, the correct implementation is
    unmotivated. Without the second, you would not know why the bug survives
    testing.
    """
    # The coincidence: 25 fps IS 40 ms per frame, so index == tick.
    stamps_25 = [i / 25.0 for i in range(250)]
    naive_25 = naive_index_positions(stamps_25, 1, 1)
    correct_25 = video_positions(stamps_25, 1, 1)
    r.check(all(n.t == c.t for n, c in zip(naive_25, correct_25)),
            "at exactly 25 fps, naive indexing coincidentally MATCHES",
            "This is why the bug survives testing — and why testing at a "
            "single frame rate proves nothing.")

    # Everywhere else it drifts, and the drift grows without bound.
    for fps, elapsed, expected_min_drift in [(2, 10.0, 200),
                                             (2, 60.0, 1300),
                                             (1, 60.0, 1400)]:
        stamps = [i / fps for i in range(int(elapsed * fps) + 1)]
        naive = naive_index_positions(stamps, 1, 1)[-1]
        correct = video_positions(stamps, 1, 1)[-1]
        audio = audio_positions(1, start_seconds=stamps[-1])[0]

        r.check(correct.t == audio.t,
                f"{fps} fps @ {elapsed}s: correct positions still match audio")
        drift = abs(naive.t - audio.t)
        r.check(drift >= expected_min_drift,
                f"{fps} fps @ {elapsed}s: naive drifts {drift} positions from audio",
                f"expected at least {expected_min_drift}")

    # Drift must GROW with elapsed time — a fixed offset would be survivable.
    stamps_short = [i / 2.0 for i in range(21)]     # 10 s
    stamps_long = [i / 2.0 for i in range(121)]     # 60 s
    d_short = abs(naive_index_positions(stamps_short, 1, 1)[-1].t
                  - video_positions(stamps_short, 1, 1)[-1].t)
    d_long = abs(naive_index_positions(stamps_long, 1, 1)[-1].t
                 - video_positions(stamps_long, 1, 1)[-1].t)
    r.check(d_long > d_short * 3,
            "drift grows with clip length (unbounded, not a fixed offset)",
            f"10s drift {d_short}, 60s drift {d_long}")


def test_per_modality_assignment(r: Results) -> None:
    """Each modality follows its own rule, and text degenerates to 1-D RoPE."""
    text = text_positions(5)
    r.check(all(tok.t == tok.h == tok.w for tok in text),
            "text: all three components equal (1-D RoPE equivalent)",
            "A clever scheme here would break the backbone's pretrained "
            "text ability.")
    r.check([tok.t for tok in text] == [0, 1, 2, 3, 4],
            "text increments by one per token")
    r.check([tok.t for tok in text_positions(3, start=10)] == [10, 11, 12],
            "text respects a start offset")

    audio = audio_positions(5, start_seconds=0.0)
    r.check(all(tok.t == tok.h == tok.w for tok in audio),
            "audio: h and w pinned to t (sound has no spatial extent)")
    r.check([tok.t for tok in audio] == [0, 1, 2, 3, 4],
            "audio: one position per 40 ms frame")

    # Offsets must compose — this is what makes chunked audio work.
    offset = audio_positions(3, start_seconds=2.0)
    r.check([tok.t for tok in offset] == [50, 51, 52],
            "audio derives position from TIME, so chunk offsets compose",
            f"got {[t.t for t in offset]} — deriving from the frame index "
            "instead silently restarts the clock at each chunk boundary")

    img = image_positions(2, 3, t=7)
    r.check(len(img) == 6, "image emits grid_h x grid_w tokens")
    r.check(all(tok.t == 7 for tok in img),
            "image: constant temporal ID (it happens at one instant)")
    r.check(sorted((tok.h, tok.w) for tok in img)
            == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            "image: spatial IDs cover the grid exactly once")

    vid = video_positions([0.0, 1.0], grid_h=2, grid_w=2)
    r.check(len(vid) == 8, "video emits frames x grid tokens")
    r.check({tok.t for tok in vid} == {0, 25},
            "video: temporal ID per frame from its timestamp",
            f"got {sorted({t.t for t in vid})}")

    for bad in ((0, 2), (2, 0)):
        try:
            image_positions(*bad)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects a degenerate {bad[0]}x{bad[1]} grid")


def test_interleaving(r: Results) -> None:
    """
    Chunking must bound the sequence distance between co-occurring tokens.

    Aligned positions are necessary and not sufficient: tokens can share a
    temporal ID while sitting 10,000 apart in the sequence, and attention
    still has to span that.
    """
    stamps = [i / 2.0 for i in range(12)]        # 6 s at 2 fps
    seq = interleave_video_audio(stamps, audio_duration=6.0,
                                 grid_h=2, grid_w=2)

    r.check(len(seq.chunk_boundaries) == 3,
            "a 6 s clip yields 3 chunks of 2 s",
            f"got {len(seq.chunk_boundaries)}")

    # Within each chunk: video must come before audio.
    for i, start in enumerate(seq.chunk_boundaries):
        stop = (seq.chunk_boundaries[i + 1]
                if i + 1 < len(seq.chunk_boundaries) else len(seq))
        mods = [t.modality for t in seq.tokens[start:stop]]
        if "video" in mods and "audio" in mods:
            r.check(max(j for j, m in enumerate(mods) if m == "video")
                    < min(j for j, m in enumerate(mods) if m == "audio"),
                    f"chunk {i}: all video precedes all audio")

    # Chunk n must cover exactly [2n, 2n+2) seconds.
    for i, start in enumerate(seq.chunk_boundaries):
        stop = (seq.chunk_boundaries[i + 1]
                if i + 1 < len(seq.chunk_boundaries) else len(seq))
        times = [t.t for t in seq.tokens[start:stop]]
        lo, hi = seconds_to_position(i * 2.0), seconds_to_position((i + 1) * 2.0)
        r.check(all(lo <= t < hi for t in times),
                f"chunk {i} covers exactly [{i * 2.0}s, {(i + 1) * 2.0}s)",
                f"positions ranged {min(times)}..{max(times)}, "
                f"expected [{lo}, {hi})")

    # The payoff: chunking must beat a flat layout, measurably.
    flat = InterleavedSequence()
    flat.tokens = video_positions(stamps, 2, 2) + audio_positions(150, 0.0)
    flat.chunk_boundaries = [0]

    chunked_gap = seq.max_cross_modal_distance()
    flat_gap = flat.max_cross_modal_distance()
    r.check(chunked_gap < flat_gap,
            "chunking shrinks the worst-case cross-modal gap",
            f"chunked {chunked_gap} vs flat {flat_gap}")
    r.check(chunked_gap <= 60,
            "the chunked gap stays bounded by roughly one chunk",
            f"got {chunked_gap} tokens")

    # Alignment must SURVIVE interleaving — the reordering must not renumber.
    mismatches = [tok for tok in seq.tokens
                  if tok.modality in ("video", "audio")
                  and tok.t != seconds_to_position(tok.time)]
    r.check(not mismatches,
            "every token's position still equals its real time after interleaving",
            f"{len(mismatches)} tokens were renumbered by the reordering")


def test_irregular_and_edge_cases(r: Results) -> None:
    """Irregular sampling is the normal case, not the exception."""
    # Deliberately uneven: a scene cut, a dropped frame, a burst.
    stamps = [0.0, 0.1, 0.15, 1.9, 2.0, 5.5]
    vid = video_positions(stamps, 1, 1)
    r.check([t.t for t in vid] == [seconds_to_position(s) for s in stamps],
            "irregularly spaced frames each map to their own true time",
            f"got {[t.t for t in vid]}")

    # Unsorted input must not corrupt the layout.
    seq = interleave_video_audio([2.5, 0.5, 1.0], audio_duration=4.0,
                                 grid_h=1, grid_w=1)
    vids = [t.t for t in seq.tokens if t.modality == "video"]
    r.check(vids == sorted(vids),
            "unsorted timestamps are ordered before layout",
            f"got {vids}")

    # Video with no audio, and audio with no video, must both work.
    only_video = interleave_video_audio([0.0, 1.0], audio_duration=0.0,
                                        grid_h=1, grid_w=1)
    r.check(all(t.modality == "video" for t in only_video.tokens)
            and len(only_video) == 2,
            "silent video produces video tokens only",
            f"got {len(only_video)} tokens")

    only_audio = interleave_video_audio([], audio_duration=2.0,
                                        grid_h=1, grid_w=1)
    r.check(all(t.modality == "audio" for t in only_audio.tokens)
            and len(only_audio) == 50,
            "audio with no video produces 50 frames for 2 s",
            f"got {len(only_audio)} tokens")

    r.check(len(interleave_video_audio([], 0.0)) == 0,
            "an empty clip produces an empty sequence")

    for bad_kwargs, label in [
        ({"audio_duration": -1.0}, "negative audio duration"),
        ({"audio_duration": 1.0, "chunk_seconds": 0}, "zero chunk size"),
    ]:
        try:
            interleave_video_audio([0.0], **bad_kwargs)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects {label}")

    r.check(abs(CHUNK_SECONDS - 2.0) < 1e-9,
            "the interleave chunk is 2 s (roughly one spoken clause)")


def main() -> int:
    r = Results("TMRoPE — video and audio on one clock")
    test_the_shared_clock(r)
    test_video_audio_alignment(r)
    test_naive_indexing_is_wrong(r)
    test_per_modality_assignment(r)
    test_interleaving(r)
    test_irregular_and_edge_cases(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
