"""
Full-duplex conversation — listening while speaking.

WHAT "FULL DUPLEX" ACTUALLY MEANS
---------------------------------
Almost every voice system you have used is **half duplex**: it listens, then it
thinks, then it speaks, and while it is speaking it is deaf. That is a
walkie-talkie. Real conversation is not like that, and the gap is not a polish
issue -- it is structural. In natural dialogue people:

    - interrupt, and expect the other party to STOP
    - say "mm-hm" while the other is still talking, without taking the floor
    - begin replying before the speaker has finished
    - notice mid-sentence that they have been misunderstood, and correct

None of that is expressible if the model cannot hear itself being interrupted.

This is the last hard problem in the video-speech-to-speech stack, and it is
the one that most obviously separates a demo from a system. `02_thinker_talker/`
solved *understanding* two synchronized input streams and emitting speech.
Full duplex asks a harder question:

    Can it keep listening -- and keep WATCHING -- while it talks?

The video part matters and is easy to forget. A user who starts shaking their
head is interrupting just as surely as one who starts talking, and a system
that only listens for barge-in misses it entirely.

THE MECHANISM: TIME-SLICED AUTOREGRESSION
-----------------------------------------
DuplexOmni's answer is to stop thinking in turns and start thinking in **fixed
480 ms slices**. Every slice, regardless of who is talking, the model:

    1. consumes the last 480 ms of user audio AND the video frames from that
       same window
    2. consumes its own dialogue state
    3. emits 480 ms of its own speech (or silence)

Because that loop never stops, input never stops. Interruption is not a special
case handled by an interrupt handler -- it is just what the next slice happens
to contain.

WHY 480 ms
----------
It is a compromise between two failures. Shorter slices lower the floor on
response latency and give the model too little acoustic context to decide
anything; a 100 ms window cannot distinguish a breath from the start of a word.
Longer slices give better decisions and make the system feel sluggish, because
480 ms is already close to the ~500 ms that people perceive as a natural
conversational gap.

THE CONSTRAINT THAT GOVERNS EVERYTHING: RTF < 1
-----------------------------------------------
The real-time factor is compute time divided by audio duration produced:

    RTF = time_to_generate / duration_generated

**RTF < 1 is not a performance target. It is a correctness condition.** At
RTF > 1 the model produces 480 ms of speech in more than 480 ms, so it falls
progressively further behind, the backlog grows without bound, and the
conversation degrades until it collapses. There is no batch size that fixes it
and no amount of waiting that catches up.

This is the same shape of constraint as `08_vtt/03_streaming_memory/`: not
"make it fast" but "make per-unit cost bounded, or the system does not work."

THE CONTROL TOKENS
------------------
Barge-in is handled with three tokens rather than a separate classifier, so the
model learns turn-taking from data instead of having it imposed:

    ^        a second speaker began HERE, during assistant speech
    [CUT]    the assistant's audio actually stops HERE
    [WAIT]   suspend background reasoning; the user's intent has changed

The gap between `^` and `[CUT]` is the interesting part. They are deliberately
not the same instant, because people do not stop the microsecond someone else
starts -- they finish the word. And the text the assistant *would* have said
after `[CUT]` is retained as **ghost text**: never spoken, but kept in context,
so the model knows what it was in the middle of saying and can resume or refer
back to it coherently.

WHAT IS HERE
------------
A `DuplexSession` implementing the slice loop, the barge-in policy, ghost text,
and honest latency/RTF accounting. Pure Python, no model, no GPU -- because
turn-taking is a *policy* problem, and policies are testable. Covered by
`tests/test_duplex.py`.

Reference: "DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for
Full-Duplex Interaction." https://arxiv.org/abs/2606.09186
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

SLICE_SECONDS = 0.48          # 480 ms — the fundamental tick of the loop
DEFAULT_BARGE_IN_SLICES = 2   # ~960 ms of user speech before we yield the floor


class State(Enum):
    """
    Who holds the floor.

    Deliberately small. Every additional state is another transition to get
    wrong, and turn-taking bugs present as "the model talks over me sometimes,"
    which is close to impossible to reproduce.
    """

    LISTENING = "listening"    # user has the floor; we are silent
    SPEAKING = "speaking"      # we have the floor
    YIELDING = "yielding"      # barge-in confirmed; finishing the current word


@dataclass
class Slice:
    """One 480 ms window of the world."""

    index: int
    user_speaking: bool = False
    user_gesture: bool = False        # e.g. a head-shake: visual barge-in
    video_frames: int = 0             # frames sampled in this window

    @property
    def start_time(self) -> float:
        return self.index * SLICE_SECONDS

    @property
    def user_active(self) -> bool:
        """
        Barge-in can arrive through either channel.

        Forgetting the visual half is the classic omni-model bug: the system
        listens for interruption but does not WATCH for it, so a user who
        starts shaking their head is talked over. If the model can see, it has
        no excuse.
        """
        return self.user_speaking or self.user_gesture


@dataclass
class SliceResult:
    """What the model did during one slice, and what it cost."""

    index: int
    state: State
    spoke: bool
    text: str = ""
    ghost_text: str = ""              # intended but never spoken
    control: List[str] = field(default_factory=list)
    compute_seconds: float = 0.0

    @property
    def rtf(self) -> float:
        """
        Real-time factor for this slice. Must stay below 1.0.

        A slice we stayed silent for still consumed compute (we processed the
        user's audio and video), so it still has an RTF. Reporting it only for
        speaking slices would hide the case where merely *listening* is too
        slow -- which is the failure that looks like the model ignoring you.
        """
        return self.compute_seconds / SLICE_SECONDS


class DuplexSession:
    """
    A full-duplex conversation as a slice loop.

    Usage:
        session = DuplexSession()
        for sl in incoming_slices:
            result = session.step(sl, planned_text="...", compute_seconds=0.1)

    The invariant worth holding onto: `step` is called for **every** slice,
    including ones where the model is mid-sentence. There is no code path where
    the model stops consuming input. That is what full duplex means, and it is
    the thing a half-duplex system cannot retrofit.
    """

    def __init__(
        self,
        barge_in_slices: int = DEFAULT_BARGE_IN_SLICES,
        yield_slices: int = 1,
    ):
        """
        Args:
            barge_in_slices: Consecutive user-active slices before we yield.
                1 (480 ms) makes the model twitchy -- a cough or a "mm-hm"
                takes the floor. 3+ (1.44 s) makes it feel rude, because the
                user has to talk over it for a second and a half. 2 is the
                default for that reason.
            yield_slices: Slices spent between `^` and `[CUT]`, i.e. how long
                we take to actually stop. Zero would be instant and sounds
                wrong -- people finish the word they are on.
        """
        if barge_in_slices < 1:
            raise ValueError(f"barge_in_slices must be >= 1, got {barge_in_slices}")
        if yield_slices < 0:
            raise ValueError(f"yield_slices must be >= 0, got {yield_slices}")

        self.barge_in_slices = barge_in_slices
        self.yield_slices = yield_slices

        self.state = State.LISTENING
        self.results: List[SliceResult] = []

        self._user_active_run = 0
        self._yield_countdown = 0
        self._pending_text: List[str] = []
        self.ghost_text: List[str] = []
        self._user_turn_started: Optional[int] = None
        self.response_latencies: List[float] = []

    # -- the loop ---------------------------------------------------------

    def step(
        self,
        sl: Slice,
        planned_text: str = "",
        compute_seconds: float = 0.0,
        want_floor: bool = False,
    ) -> SliceResult:
        """
        Advance one 480 ms slice.

        Args:
            sl: What arrived from the user in this window (audio and video).
            planned_text: What the model would like to say this slice. Supplied
                by the caller because *what* to say is the Talker's job; this
                class owns only *whether* and *when*.
            compute_seconds: Wall-clock spent. Used for RTF accounting.
            want_floor: The model wishes to start speaking.

        Returns:
            A `SliceResult` recording what happened and what it cost.
        """
        control: List[str] = []
        spoke = False
        text = ""
        ghost = ""

        # Track how long the user has been active, for barge-in detection.
        if sl.user_active:
            self._user_active_run += 1
            if self._user_turn_started is None:
                self._user_turn_started = sl.index
        else:
            self._user_active_run = 0

        # ---- SPEAKING: we hold the floor, but we are still listening ----
        if self.state == State.SPEAKING:
            if self._user_active_run >= self.barge_in_slices:
                # Barge-in confirmed. Mark WHERE it began, not where we noticed
                # -- the run started barge_in_slices ago.
                control.append("^")
                self.state = State.YIELDING
                self._yield_countdown = self.yield_slices
                if self._yield_countdown == 0:
                    control.append("[CUT]")
                    ghost = self._flush_ghost(planned_text)
                    self.state = State.LISTENING
                else:
                    # Still finishing the current word.
                    spoke, text = True, planned_text
            else:
                spoke, text = True, planned_text
                if planned_text:
                    self._pending_text.append(planned_text)

        # ---- YIELDING: finishing the word, then stopping ----------------
        elif self.state == State.YIELDING:
            self._yield_countdown -= 1
            if self._yield_countdown <= 0:
                control.append("[CUT]")
                ghost = self._flush_ghost(planned_text)
                self.state = State.LISTENING
            else:
                spoke, text = True, planned_text

        # ---- LISTENING: silent, deciding whether to take the floor ------
        else:
            if want_floor and not sl.user_active:
                self.state = State.SPEAKING
                spoke, text = True, planned_text
                if planned_text:
                    self._pending_text.append(planned_text)
                if self._user_turn_started is not None:
                    # Latency from when the user STOPPED to when we started.
                    self.response_latencies.append(
                        (sl.index - self._user_turn_started) * SLICE_SECONDS
                    )
                    self._user_turn_started = None

        # A user whose intent changed mid-stream invalidates background work.
        if sl.user_active and self.state == State.LISTENING and self._pending_text:
            control.append("[WAIT]")

        result = SliceResult(
            index=sl.index,
            state=self.state,
            spoke=spoke,
            text=text,
            ghost_text=ghost,
            control=control,
            compute_seconds=compute_seconds,
        )
        self.results.append(result)
        return result

    def _flush_ghost(self, planned_text: str) -> str:
        """
        Retain what we were about to say but never said.

        Ghost text is not a log. It goes back into context, so the model knows
        it was cut off mid-thought and can say "as I was saying" or drop the
        point entirely -- a choice it cannot make if the unsaid words simply
        vanished.
        """
        if planned_text:
            self.ghost_text.append(planned_text)
        return planned_text

    # -- accounting -------------------------------------------------------

    @property
    def slices_processed(self) -> int:
        return len(self.results)

    @property
    def mean_rtf(self) -> float:
        """Mean real-time factor. Must be < 1.0 or the system falls behind."""
        if not self.results:
            return 0.0
        return sum(r.rtf for r in self.results) / len(self.results)

    @property
    def worst_rtf(self) -> float:
        """
        Worst single-slice RTF.

        More important than the mean. A system averaging 0.6 with occasional
        spikes to 1.4 stutters audibly, and the mean hides it completely.
        """
        return max((r.rtf for r in self.results), default=0.0)

    @property
    def mean_response_latency(self) -> float:
        """Mean seconds from the user finishing to us starting."""
        if not self.response_latencies:
            return 0.0
        return sum(self.response_latencies) / len(self.response_latencies)

    def is_realtime(self) -> bool:
        """Whether every slice met the deadline. The correctness condition."""
        return self.worst_rtf < 1.0

    def report(self) -> str:
        """The four numbers that decide whether this is shippable."""
        barge_ins = sum(1 for r in self.results if "^" in r.control)
        cuts = sum(1 for r in self.results if "[CUT]" in r.control)
        spoken = sum(1 for r in self.results if r.spoke)

        lines = [
            f"  slices          {self.slices_processed}  "
            f"({self.slices_processed * SLICE_SECONDS:.1f} s of conversation)",
            f"  spoke in        {spoken} slices "
            f"({spoken * SLICE_SECONDS:.1f} s of speech)",
            f"  barge-ins       {barge_ins} detected, {cuts} completed",
            f"  ghost text      {len(self.ghost_text)} fragments retained",
            f"  mean RTF        {self.mean_rtf:.3f}",
            f"  worst RTF       {self.worst_rtf:.3f}  "
            f"{'OK' if self.worst_rtf < 1.0 else 'TOO SLOW — falls behind'}",
            f"  mean latency    {self.mean_response_latency:.3f} s",
        ]
        if not self.is_realtime():
            lines.append(
                "  WARNING: RTF >= 1 means the backlog grows without bound. "
                "This is not slowness, it is failure."
            )
        return "\n".join(lines)


if __name__ == "__main__":
    bar = "=" * 74
    print(bar)
    print("  Full-duplex conversation — listening and watching while speaking")
    print(bar)
    print(f"  slice           {SLICE_SECONDS * 1000:.0f} ms")
    print(f"  barge-in after  {DEFAULT_BARGE_IN_SLICES} active slices "
          f"({DEFAULT_BARGE_IN_SLICES * SLICE_SECONDS * 1000:.0f} ms)")
    print(bar)

    # A scripted conversation: the assistant starts answering, the user
    # interrupts by TALKING, and later interrupts again by GESTURING only.
    script = [
        # (user_speaking, user_gesture)
        (True,  False), (True,  False), (False, False),   # user asks
        (False, False), (False, False), (False, False),   # we answer
        (False, False), (True,  False), (True,  False),   # user cuts in
        (True,  False), (False, False), (False, False),   # we stopped
        (False, False), (False, True),  (False, True),    # head-shake only
        (False, False), (False, False),
    ]

    session = DuplexSession()
    words = ["The", "capital", "of", "France", "is", "Paris", "which",
             "has", "been", "the", "seat", "of", "government", "since",
             "the", "tenth", "century"]

    print()
    print("  slice  user      state       assistant")
    print("  " + "-" * 62)

    for i, (speaking, gesture) in enumerate(script):
        sl = Slice(index=i, user_speaking=speaking, user_gesture=gesture,
                   video_frames=12)
        # We want the floor once the user has been quiet for a slice.
        quiet = not speaking and not gesture
        want = quiet and session.state == State.LISTENING and i > 2
        res = session.step(sl, planned_text=words[i % len(words)],
                           compute_seconds=0.11, want_floor=want)

        user_col = ("speaking" if speaking else
                    "gesture" if gesture else "-")
        marks = " ".join(res.control)
        said = res.text if res.spoke else ""
        print(f"  {i:>5}  {user_col:<9} {res.state.value:<11} "
              f"{said:<12} {marks}")

    print()
    print(bar)
    print(session.report())
    print(bar)
    print()
    print("  Ghost text (intended, never spoken):")
    print(f"    {session.ghost_text or '(none)'}")
    print()
    print("  The model never stopped consuming input — not for a single slice.")
    print("  That is the whole difference between duplex and a walkie-talkie.")
