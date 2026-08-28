# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Regression test: full-duplex turn-taking must never stop listening.

Run:
    uv run tests/test_duplex.py

Why this suite exists
---------------------
Turn-taking bugs are the worst kind to debug from a bug report. They present as
*"it talks over me sometimes"* — intermittent, dependent on exactly when the
user started speaking relative to a slice boundary, and essentially impossible
to reproduce by hand.

Turn-taking is a *policy*, though, and policies are deterministic given a
script. So the scripts live here and the timing is asserted exactly.

Four properties, in descending order of how badly they break things:

  1. **Input is never dropped.** If there is any code path where the model
     stops consuming while speaking, it is half duplex wearing a costume.
  2. **Barge-in fires, and only when it should.** Too eager and a cough takes
     the floor; too slow and the user shouts over it for two seconds.
  3. **Gesture-only barge-in works.** The omni-specific one. A model that can
     SEE has no excuse for talking over someone shaking their head, and this
     is the property a port from an audio-only system silently loses.
  4. **RTF accounting is honest.** RTF >= 1 is not slowness, it is unbounded
     backlog, and it must be reported as failure rather than a slow number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "09_vss" / "03_duplex_streaming"))

from duplex import (  # noqa: E402
    DEFAULT_BARGE_IN_SLICES,
    SLICE_SECONDS,
    DuplexSession,
    Slice,
    State,
)


def run(session: DuplexSession, script, compute=0.1, speak_from=0):
    """
    Drive a session through a script of (user_speaking, user_gesture) pairs.

    The model asks for the floor whenever it is silent and the user is not
    active — a simple, deterministic policy so the test measures the TURN-TAKING
    rather than a speaking heuristic.
    """
    out = []
    for i, (speaking, gesture) in enumerate(script):
        sl = Slice(index=i, user_speaking=speaking, user_gesture=gesture,
                   video_frames=12)
        want = (i >= speak_from
                and session.state == State.LISTENING
                and not sl.user_active)
        out.append(session.step(sl, planned_text=f"w{i}",
                                compute_seconds=compute, want_floor=want))
    return out


def test_never_stops_listening(r: Results) -> None:
    """The defining property. One result per slice, no exceptions."""
    script = [(False, False)] * 4 + [(True, False)] * 6 + [(False, False)] * 4
    session = DuplexSession()
    results = run(session, script, speak_from=0)

    r.check(len(results) == len(script),
            "every slice produced a result — input is never dropped",
            f"{len(results)} results for {len(script)} slices")
    r.check(session.slices_processed == len(script),
            "the session counted every slice")

    # Crucially: slices arriving WHILE the model was speaking must still have
    # been processed. A half-duplex system would have no record of them.
    while_speaking = [res for res in results if res.state == State.SPEAKING]
    r.check(bool(while_speaking),
            "the model did speak at some point (the test is meaningful)")
    r.check(all(res.index is not None for res in while_speaking),
            "slices arriving during assistant speech were still consumed")


def test_barge_in_timing(r: Results) -> None:
    """Barge-in must fire after exactly `barge_in_slices` active slices."""
    # Model speaks from slice 1; user cuts in at slice 4.
    script = [(False, False)] * 4 + [(True, False)] * 6
    session = DuplexSession(barge_in_slices=2, yield_slices=1)
    results = run(session, script, speak_from=1)

    marks = [i for i, res in enumerate(results) if "^" in res.control]
    cuts = [i for i, res in enumerate(results) if "[CUT]" in res.control]

    r.check(len(marks) == 1, "exactly one barge-in marker",
            f"markers at {marks}")
    r.check(len(cuts) == 1, "exactly one cut", f"cuts at {cuts}")

    # User starts at slice 4; with barge_in_slices=2 the mark lands at slice 5.
    r.check(marks and marks[0] == 5,
            "the ^ marker fires after 2 consecutive active slices",
            f"expected slice 5, got {marks}")
    r.check(cuts and marks and cuts[0] > marks[0],
            "[CUT] comes AFTER ^ — we finish the word before stopping",
            f"^ at {marks}, [CUT] at {cuts}")
    r.check(cuts and marks and cuts[0] - marks[0] == 1,
            "the yield gap is exactly yield_slices",
            f"gap was {cuts[0] - marks[0] if cuts and marks else None}")

    # After the cut, we must be silent.
    r.check(all(not res.spoke for res in results[cuts[0]:]),
            "the model is silent after [CUT]")

    # A brief blip must NOT take the floor.
    blip = [(False, False)] * 3 + [(True, False)] + [(False, False)] * 4
    quiet = DuplexSession(barge_in_slices=2)
    blip_results = run(quiet, blip, speak_from=1)
    r.check(not any("^" in res.control for res in blip_results),
            "a single-slice blip does NOT trigger barge-in",
            "otherwise a cough or an 'mm-hm' steals the floor")


def test_gesture_barge_in(r: Results) -> None:
    """
    The omni-specific property: a head-shake interrupts, with no audio at all.

    This is what a port from an audio-only duplex system silently loses. The
    model can see; a user shaking their head is interrupting.
    """
    script = [(False, False)] * 4 + [(False, True)] * 6
    session = DuplexSession(barge_in_slices=2, yield_slices=1)
    results = run(session, script, speak_from=1)

    r.check(any("^" in res.control for res in results),
            "a gesture with NO speech triggers barge-in",
            "a model that can see has no excuse for talking over a head-shake")
    r.check(any("[CUT]" in res.control for res in results),
            "the gesture barge-in completes with a cut")

    # Mixed evidence must work too — speech and gesture are one signal.
    mixed = [(False, False)] * 4 + [(True, False), (False, True)] * 3
    mixed_session = DuplexSession(barge_in_slices=2, yield_slices=1)
    mixed_results = run(mixed_session, mixed, speak_from=1)
    r.check(any("^" in res.control for res in mixed_results),
            "alternating speech and gesture still counts as one active run",
            "treating them as separate counters would reset the run and "
            "never reach the threshold")

    r.check(Slice(0, user_speaking=False, user_gesture=True).user_active,
            "a gesture alone marks the slice active")
    r.check(not Slice(0).user_active, "an empty slice is inactive")


def test_ghost_text(r: Results) -> None:
    """What we were about to say must be retained, not discarded."""
    script = [(False, False)] * 4 + [(True, False)] * 6
    session = DuplexSession(barge_in_slices=2, yield_slices=1)
    results = run(session, script, speak_from=1)

    r.check(bool(session.ghost_text),
            "text intended but never spoken is retained",
            "without it the model cannot know it was cut off mid-thought")

    cut = next((res for res in results if "[CUT]" in res.control), None)
    r.check(cut is not None and bool(cut.ghost_text),
            "the cutting slice carries its ghost text")

    # Ghost text must be text that was NOT spoken.
    spoken = {res.text for res in results if res.spoke}
    r.check(all(g not in spoken for g in session.ghost_text),
            "ghost text was never actually spoken",
            f"ghost={session.ghost_text}, spoken={sorted(spoken)}")


def test_rtf_accounting(r: Results) -> None:
    """RTF >= 1 must be reported as failure, not as a slow number."""
    script = [(False, False)] * 10

    fast = DuplexSession()
    run(fast, script, compute=0.1)
    r.check(abs(fast.mean_rtf - 0.1 / SLICE_SECONDS) < 1e-9,
            "mean RTF = compute / slice duration",
            f"got {fast.mean_rtf}")
    r.check(fast.is_realtime(), "RTF 0.21 is real-time")

    slow = DuplexSession()
    run(slow, script, compute=0.6)
    r.check(not slow.is_realtime(),
            "600 ms of compute for a 480 ms slice is NOT real-time",
            f"RTF {slow.worst_rtf:.2f} — the backlog grows without bound")
    r.check("WARNING" in slow.report(),
            "the report warns explicitly when RTF >= 1")
    r.check("falls behind" in slow.report(),
            "the report names the consequence, not just the number")

    # Worst-case must be surfaced separately — a good mean hides stutter.
    spiky = DuplexSession()
    for i in range(10):
        spiky.step(Slice(index=i), planned_text="x",
                   compute_seconds=0.6 if i == 5 else 0.05)
    r.check(spiky.mean_rtf < 1.0 and spiky.worst_rtf > 1.0,
            "a single slow slice is caught by worst_rtf though the mean is fine",
            f"mean {spiky.mean_rtf:.2f}, worst {spiky.worst_rtf:.2f} — "
            "reporting only the mean would hide an audible stutter")
    r.check(not spiky.is_realtime(),
            "is_realtime() uses the worst case, not the mean")

    # A silent slice still costs compute and must still be measured.
    r.check(spiky.results[0].rtf > 0,
            "silent slices still report an RTF (listening costs compute too)")


def test_state_machine(r: Results) -> None:
    """States must be reachable, and transitions legal."""
    script = [(False, False)] * 4 + [(True, False)] * 4 + [(False, False)] * 4
    session = DuplexSession(barge_in_slices=2, yield_slices=1)
    results = run(session, script, speak_from=1)

    seen = {res.state for res in results}
    r.check(State.LISTENING in seen, "LISTENING is reached")
    r.check(State.SPEAKING in seen, "SPEAKING is reached")
    r.check(State.YIELDING in seen, "YIELDING is reached")

    # We must never speak while in LISTENING.
    r.check(all(not res.spoke for res in results if res.state == State.LISTENING),
            "the model is silent whenever it is in LISTENING")

    # Legal transitions only.
    legal = {
        (State.LISTENING, State.LISTENING), (State.LISTENING, State.SPEAKING),
        (State.SPEAKING, State.SPEAKING), (State.SPEAKING, State.YIELDING),
        (State.SPEAKING, State.LISTENING),
        (State.YIELDING, State.YIELDING), (State.YIELDING, State.LISTENING),
    }
    bad = [(a.state, b.state) for a, b in zip(results, results[1:])
           if (a.state, b.state) not in legal]
    r.check(not bad, "every state transition is legal", f"illegal: {set(bad)}")

    for bad_kwargs, label in [({"barge_in_slices": 0}, "barge_in_slices=0"),
                              ({"yield_slices": -1}, "yield_slices=-1")]:
        try:
            DuplexSession(**bad_kwargs)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects {label}")

    r.check(abs(SLICE_SECONDS - 0.48) < 1e-9, "the slice is 480 ms")
    r.check(DEFAULT_BARGE_IN_SLICES == 2,
            "the default barge-in threshold is 2 slices (~960 ms)")


def main() -> int:
    r = Results("Full-duplex turn-taking — never stops listening")
    test_never_stops_listening(r)
    test_barge_in_timing(r)
    test_gesture_barge_in(r)
    test_ghost_text(r)
    test_rtf_accounting(r)
    test_state_machine(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
