# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Logic test for 10_sweep_check.

Run:
    uv run tests/test_sweep_check.py

TODO(contributor): replace this with real assertions.

What belongs here
-----------------
The point of tests/ is to verify examples that CANNOT be run locally — no GPU,
no multi-GB download. So test the LOGIC of the change, not the training run.

Prefer asserting mathematical or structural PROPERTIES over shapes. The bugs
this repository has actually shipped all had the same character: the code ran
fine, the loss decreased, and the result was quietly wrong. A shape assertion
would have passed on every one of them.

`tests/_srcload.py` extracts a single function from a training script via `ast`,
so you can test the ACTUAL shipped source without importing torch or deepspeed:

    from _srcload import Results, load_function, source_contains

    fn = load_function("10_sweep_check/train_sweep_check.py", "my_function")
    r.check(fn(...) == expected, "describes what is guaranteed")

Add this file to tests/run_all.sh and .github/workflows/tests.yml when it does
something real.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results, source_contains  # noqa: E402

FOLDER = "10_sweep_check"


def main() -> int:
    r = Results("Sweep — logic checks")

    # A starter check that is genuinely worth keeping: the preflight must be
    # present, or a CPU-only reader gets an unreadable CUDA error instead of
    # an explanation.
    r.check(
        source_contains(f"{FOLDER}/train_sweep_check.py", "require_gpu"),
        "the entry point has a require_gpu() preflight",
        "Every example must fail gracefully without a GPU. See CONTRIBUTING.md.",
    )

    # TODO(contributor): add checks that would have caught a real bug.

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
