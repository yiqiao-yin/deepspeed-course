"""
TMRoPE — putting video and audio on the same clock.

THE PROBLEM THIS SOLVES
-----------------------
A video-speech-to-speech model receives two streams that disagree about what
time it is.

    video   1-25 frames per second, irregular, whatever the sampler gave you
    audio   16,000 samples per second, or ~50 encoder frames per second

Concatenate them into one sequence and the transformer sees a flat list of
tokens with no idea which video frame goes with which sound. Ask it *"what did
he say while pointing at the whiteboard?"* and it cannot answer, because the
information that pointing and saying happened **at the same moment** was never
in the input. Not lost during training -- never encoded at all.

This is the defining problem of the omni-modal family, and it is not the same
problem as `04_video_text/`. There, time only had to be represented *within* one
stream. Here two streams have to agree.

THE FIX: ONE SHARED CLOCK, MEASURED IN 40 ms TICKS
--------------------------------------------------
Qwen2.5-Omni's TMRoPE (Time-aligned Multimodal RoPE) extends M-RoPE's
three-component position -- (temporal, height, width) -- with a single rule
that does all the work:

    **one temporal position ID == 40 milliseconds of real time**

for *every* modality. Not one ID per token, not one ID per frame. One ID per
40 ms of wall-clock time.

The consequence is the whole point: a video frame at t=1.00s and an audio frame
at t=1.00s both receive temporal position 25. They are *the same position* to
the attention mechanism, so co-occurrence is expressed in the position encoding
rather than left for the model to infer.

Per-modality assignment follows from that one rule:

    text     t = h = w, all equal and incrementing.
             TMRoPE degenerates to ordinary 1-D RoPE. Nothing special happens
             to text, which is what you want.

    audio    t = h = w, one ID per 40 ms frame. Audio has no spatial extent,
             so the height and width components carry no information and are
             pinned to the temporal value.

    image    t constant (a photograph happens at one instant), h and w vary
             across the patch grid.

    video    t derived from each frame's ACTUAL TIMESTAMP, not its index.
             h and w vary across the patch grid.

That last line is the one people get wrong. See `naive_index_positions`.

THE 2-SECOND INTERLEAVE
-----------------------
Sharing a clock is necessary and not sufficient. If you lay out the whole video
and then the whole audio, correctly-numbered positions are still 10,000 tokens
apart in the sequence, and attention has to reach across the entire span to
connect a gesture to the word it accompanied.

So TMRoPE also chunks the sequence by real time: every 2 seconds, visual tokens
first, then the audio tokens for that same 2 seconds. Co-occurring content ends
up physically adjacent as well as identically numbered.

    [ video 0-2s ][ audio 0-2s ][ video 2-4s ][ audio 2-4s ][ video 4-6s ]...

WHY THIS FILE IS PURE INTEGER ARITHMETIC
----------------------------------------
Because it can be, and because that makes it provable. There is no model here,
no GPU, no download -- position assignment is arithmetic, so it is verified on
a laptop by `tests/test_tmrope.py` rather than hoped about on rented hardware.

The failure mode is worth stating plainly: get this wrong and **nothing
crashes**. The model trains, the loss falls, and it quietly cannot answer any
question that depends on audio and video lining up. That is indistinguishable
from a model that is merely undertrained, which is why it is worth an
arithmetic proof.

Reference: Xu et al. "Qwen2.5-Omni Technical Report."
https://arxiv.org/abs/2503.20215
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# The quantum. Every modality is measured against this, and it is the single
# constant that makes cross-modal synchronization work.
TIME_UNIT_SECONDS = 0.04          # 40 ms per temporal position ID
CHUNK_SECONDS = 2.0               # interleave granularity


def seconds_to_position(seconds: float) -> int:
    """
    Convert wall-clock time to a temporal position ID.

    The entire scheme rests on this one function being applied identically to
    every modality. Call it for a video frame and for an audio frame at the
    same instant and you get the same integer -- which is what "aligned" means
    here, concretely.

    Rounds rather than truncates. Truncation biases every timestamp
    systematically earlier by up to a full tick, and a consistent 40 ms skew
    between two streams is exactly the error this design exists to prevent.
    """
    if seconds < 0:
        raise ValueError(f"time cannot be negative: {seconds}")
    return int(round(seconds / TIME_UNIT_SECONDS))


def position_to_seconds(position: int) -> float:
    """Inverse of `seconds_to_position`, for reporting and debugging."""
    return position * TIME_UNIT_SECONDS


@dataclass
class PositionedToken:
    """
    One token and where it sits in (time, height, width).

    `modality` and `time` are carried for inspection and testing only -- a real
    implementation passes just the three integers to the rotary embedding. They
    are here because a position scheme you cannot inspect is a position scheme
    you cannot debug, and the bugs are silent.
    """

    modality: str          # "text" | "audio" | "image" | "video"
    t: int                 # temporal position ID  (40 ms per unit)
    h: int                 # height position
    w: int                 # width position
    time: float = 0.0      # real time in seconds, for inspection

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.t, self.h, self.w)


# ---------------------------------------------------------------------------
# Per-modality position assignment
# ---------------------------------------------------------------------------

def text_positions(n_tokens: int, start: int = 0) -> List[PositionedToken]:
    """
    Text: all three components equal, incrementing by one per token.

    With t == h == w, the three-component rotary embedding collapses to the
    ordinary 1-D case. That is deliberate. Text has no spatial extent and no
    intrinsic timestamp, and a scheme that did something clever here would
    break every pretrained text capability the backbone arrived with.

    Note that text positions are counted in TOKENS, not in 40 ms ticks -- text
    does not happen at a time. The shared clock governs the streams that do.
    """
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
    return [
        PositionedToken("text", start + i, start + i, start + i)
        for i in range(n_tokens)
    ]


def audio_positions(
    n_frames: int, start_seconds: float = 0.0
) -> List[PositionedToken]:
    """
    Audio: one position ID per 40 ms frame, all three components equal.

    Audio encoders in this family emit one frame per 40 ms after pooling, so
    the frame index and the tick index coincide -- but we derive the position
    from TIME anyway, via `seconds_to_position`, so that `start_seconds`
    composes correctly when audio is chunked and interleaved. Deriving from
    the frame index instead works right up until the first chunk boundary and
    then silently restarts the clock at zero.

    h and w are pinned to t because sound has no spatial extent here. (Models
    doing spatial audio would use them; this family does not.)
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")

    tokens = []
    for i in range(n_frames):
        t = seconds_to_position(start_seconds + i * TIME_UNIT_SECONDS)
        tokens.append(
            PositionedToken("audio", t, t, t,
                            time=start_seconds + i * TIME_UNIT_SECONDS)
        )
    return tokens


def image_positions(
    grid_h: int, grid_w: int, t: int = 0
) -> List[PositionedToken]:
    """
    Image: one constant temporal ID, spatial IDs across the patch grid.

    A photograph happens at a single instant, so every patch shares one
    temporal position. The h and w components are what let attention reason
    about *where* in the frame something is -- the same mechanism a
    vision-language model uses, unchanged.
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"grid must be at least 1x1, got {grid_h}x{grid_w}")

    return [
        PositionedToken("image", t, h, w, time=position_to_seconds(t))
        for h in range(grid_h)
        for w in range(grid_w)
    ]


def video_positions(
    timestamps: Sequence[float], grid_h: int, grid_w: int
) -> List[PositionedToken]:
    """
    Video: temporal ID from each frame's REAL TIMESTAMP, spatial IDs from the grid.

    This is where variable frame rates stop being a problem. Sample at 2 fps
    and frames land 50 ticks apart; sample at 25 fps and they land 2 ticks
    apart. Either way, the frame showing 3.00 seconds gets temporal position
    75 -- and so does the audio at 3.00 seconds.

    Sampling rate becomes a *resolution* choice rather than a *semantic* one,
    which is the property that lets you drop frames to save memory without
    lying to the model about when things happened.
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"grid must be at least 1x1, got {grid_h}x{grid_w}")

    tokens = []
    for stamp in timestamps:
        t = seconds_to_position(stamp)
        for h in range(grid_h):
            for w in range(grid_w):
                tokens.append(PositionedToken("video", t, h, w, time=stamp))
    return tokens


def naive_index_positions(
    timestamps: Sequence[float], grid_h: int, grid_w: int
) -> List[PositionedToken]:
    """
    The WRONG way, kept so the difference can be measured rather than asserted.

    Number video frames 0, 1, 2, ... by index -- the obvious approach, and the
    one a fixed-frame architecture forces on you. Two things break:

    1. **Frame rate changes the meaning.** At 2 fps, frame 4 is t=4. At 25 fps,
       frame 4 is also t=4. But the first is two seconds into the clip and the
       second is 160 milliseconds in. Same encoding, different reality.

    2. **The clocks diverge.** Audio still ticks at 40 ms. Video now ticks at
       "one frame." Unless your video happens to run at exactly 25 fps, the
       two streams drift apart, and the drift grows without bound over the
       clip. By the end of a long recording, video and audio positions
       describing the same instant can be thousands of IDs apart.

    `tests/test_tmrope.py` measures that drift explicitly, because "this is
    wrong" is a weaker claim than "at 2 fps this is 1,150 positions wrong by
    the one-minute mark."
    """
    tokens = []
    for index, stamp in enumerate(timestamps):
        for h in range(grid_h):
            for w in range(grid_w):
                tokens.append(PositionedToken("video", index, h, w, time=stamp))
    return tokens


# ---------------------------------------------------------------------------
# The 2-second interleave
# ---------------------------------------------------------------------------

@dataclass
class InterleavedSequence:
    """The laid-out sequence, plus enough bookkeeping to check it."""

    tokens: List[PositionedToken] = field(default_factory=list)
    chunk_boundaries: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    def position_ids(self) -> List[Tuple[int, int, int]]:
        """The (t, h, w) triples a rotary embedding actually consumes."""
        return [tok.as_tuple() for tok in self.tokens]

    def max_cross_modal_distance(self) -> int:
        """
        Worst-case sequence gap between a video token and an audio token that
        describe the same 40 ms instant.

        The number the interleave exists to shrink. Positions can be perfectly
        aligned while the tokens sit 10,000 apart in the sequence, and
        attention still has to span that gap. Chunking bounds it.
        """
        by_time: dict = {}
        for index, tok in enumerate(self.tokens):
            if tok.modality in ("video", "audio"):
                by_time.setdefault(tok.t, {}).setdefault(tok.modality, []).append(index)

        worst = 0
        for _, modalities in by_time.items():
            if len(modalities) < 2:
                continue
            video_idx = modalities.get("video", [])
            audio_idx = modalities.get("audio", [])
            if video_idx and audio_idx:
                worst = max(worst,
                            max(audio_idx) - min(video_idx),
                            max(video_idx) - min(audio_idx))
        return worst


def interleave_video_audio(
    video_timestamps: Sequence[float],
    audio_duration: float,
    grid_h: int = 2,
    grid_w: int = 2,
    chunk_seconds: float = CHUNK_SECONDS,
) -> InterleavedSequence:
    """
    Lay video and audio out in 2-second chunks: visual first, audio second.

    WHY VISUAL FIRST

    Within a chunk the order is a convention, and the convention is load-bearing
    only in that it must be *consistent* between training and inference. Video
    first matches the reference implementation. What matters is that both
    modalities for a given 2 seconds sit together before either moves on.

    WHY 2 SECONDS

    A trade-off with a floor and a ceiling. Smaller chunks put co-occurring
    tokens closer together, which is what you want, but they fragment both
    streams into many small runs and cost you the local coherence within each
    modality. Larger chunks preserve that coherence and let the cross-modal gap
    grow back. Two seconds is roughly the span of a spoken clause or a single
    gesture -- the natural unit of co-occurrence in conversation.

    Args:
        video_timestamps: Seconds from clip start for each sampled frame.
            Need not be evenly spaced; that is the point.
        audio_duration: Total audio length in seconds.
        grid_h, grid_w: Patch grid per video frame. Small values keep the
            demonstration readable; real models use 16x16 or larger.
        chunk_seconds: Interleave granularity.

    Returns:
        An `InterleavedSequence` whose positions are already aligned across
        modalities and whose tokens are ordered by chunk.
    """
    if audio_duration < 0:
        raise ValueError(f"audio_duration must be >= 0, got {audio_duration}")
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be > 0, got {chunk_seconds}")

    video_timestamps = sorted(video_timestamps)
    total = max(audio_duration,
                video_timestamps[-1] if video_timestamps else 0.0)

    sequence = InterleavedSequence()
    n_chunks = max(1, int(total / chunk_seconds) + (1 if total % chunk_seconds else 0))

    for chunk in range(n_chunks):
        start = chunk * chunk_seconds
        stop = start + chunk_seconds
        sequence.chunk_boundaries.append(len(sequence.tokens))

        # --- visual first ---
        frames_here = [ts for ts in video_timestamps if start <= ts < stop]
        if frames_here:
            sequence.tokens.extend(
                video_positions(frames_here, grid_h, grid_w)
            )

        # --- then the audio for the SAME window ---
        audio_stop = min(stop, audio_duration)
        if audio_stop > start:
            n_frames = int(round((audio_stop - start) / TIME_UNIT_SECONDS))
            sequence.tokens.extend(audio_positions(n_frames, start_seconds=start))

    return sequence


def describe(sequence: InterleavedSequence, max_rows: int = 12) -> str:
    """Render the layout so a human can see the interleave rather than trust it."""
    lines = [
        f"  {len(sequence)} tokens in {len(sequence.chunk_boundaries)} chunks",
        f"  worst-case video<->audio sequence gap: "
        f"{sequence.max_cross_modal_distance()} tokens",
        "",
        "  chunk  span        layout",
    ]

    for i, start in enumerate(sequence.chunk_boundaries):
        stop = (sequence.chunk_boundaries[i + 1]
                if i + 1 < len(sequence.chunk_boundaries) else len(sequence))
        chunk_tokens = sequence.tokens[start:stop]
        if not chunk_tokens:
            continue
        n_video = sum(1 for t in chunk_tokens if t.modality == "video")
        n_audio = sum(1 for t in chunk_tokens if t.modality == "audio")
        t_lo = min(t.t for t in chunk_tokens)
        t_hi = max(t.t for t in chunk_tokens)
        lines.append(
            f"  {i:>5}  t={t_lo:>4}-{t_hi:<4}  "
            f"video x{n_video:<4} audio x{n_audio:<4}"
            f"  ({position_to_seconds(t_lo):.2f}s - "
            f"{position_to_seconds(t_hi):.2f}s)"
        )
        if i + 1 >= max_rows:
            lines.append(f"  ...    ({len(sequence.chunk_boundaries) - max_rows} more)")
            break

    return "\n".join(lines)


if __name__ == "__main__":
    bar = "=" * 74
    print(bar)
    print("  TMRoPE — one clock, two streams")
    print(bar)
    print(f"  time unit      {TIME_UNIT_SECONDS * 1000:.0f} ms per temporal position ID")
    print(f"  chunk size     {CHUNK_SECONDS:.0f} s")
    print()

    # --- the property that matters -------------------------------------
    print("  Same instant, different modalities, same temporal position:")
    for moment in (0.0, 1.0, 2.5, 10.0):
        v = seconds_to_position(moment)
        a = seconds_to_position(moment)
        print(f"    t={moment:>5.2f}s   video -> {v:<5} audio -> {a:<5}"
              f"  {'ALIGNED' if v == a else 'MISALIGNED'}")

    # --- frame rate independence ---------------------------------------
    print()
    print("  The same moment gets the same ID at ANY sampling rate:")
    for fps in (1, 2, 5, 25):
        stamps = [i / fps for i in range(int(3 * fps))]
        target = min(stamps, key=lambda s: abs(s - 2.0))
        print(f"    {fps:>3} fps   frame nearest 2.00s is index "
              f"{stamps.index(target):>3}  ->  temporal ID "
              f"{seconds_to_position(target)}")

    # --- what the naive version costs -----------------------------------
    print()
    print("  Frame-INDEX positions drift from the audio clock:")
    for fps in (1, 2, 5, 25):
        for elapsed in (10.0, 60.0):
            audio_pos = seconds_to_position(elapsed)
            naive_pos = int(elapsed * fps)
            print(f"    {fps:>3} fps @ {elapsed:>5.1f}s   "
                  f"audio ID {audio_pos:<6} naive video ID {naive_pos:<6}"
                  f"  drift {abs(audio_pos - naive_pos):>6}")

    # --- the interleave --------------------------------------------------
    print()
    print(bar)
    print("  2-second interleave — a 6 s clip sampled at 2 fps")
    print(bar)
    seq = interleave_video_audio(
        video_timestamps=[i / 2.0 for i in range(12)],
        audio_duration=6.0,
        grid_h=2, grid_w=2,
    )
    print(describe(seq))

    print()
    print("  Without chunking, the same content laid out end to end:")
    flat = InterleavedSequence()
    flat.tokens = (video_positions([i / 2.0 for i in range(12)], 2, 2)
                   + audio_positions(150, 0.0))
    flat.chunk_boundaries = [0]
    print(f"    worst-case video<->audio gap: "
          f"{flat.max_cross_modal_distance()} tokens "
          f"(vs {seq.max_cross_modal_distance()} chunked)")
