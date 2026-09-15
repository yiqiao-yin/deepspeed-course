# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.19", "pillow"]
# ///
"""
Regression test: the CIFAR-10 mirror really is CIFAR-10.

Run:
    uv run tests/test_cifar10_source.py

Why this suite exists
---------------------
`01_basics/03_convnet_cifar10` used to fetch CIFAR-10 through
`torchvision.datasets.CIFAR10(download=True)`, which pulls from
`cs.toronto.edu`. Measured raw fetch, from two unrelated networks -- a rented
cloud box and a home connection:

    cs.toronto.edu         73 - 82 kB/s
    huggingface.co     30,000 - 40,000 kB/s

That is roughly **400x**, or about 40 MINUTES against about 6 SECONDS for the
170 MB archive, and it made the lab unusable on a cold box: the download
outlived the orchestrator's 900 s window, so the job reported success having
never reached a single training step -- no loss, no accuracy, just progress
bars. Swapping the source is the whole fix.

But a mirror is a trust decision, and the failure mode is nasty. Data that is
*nearly* CIFAR-10 -- a different split boundary, a relabelling, a subset, images
in BGR -- would train fine and quietly produce numbers that cannot be compared
to any published result. Nothing would crash. So the mirror's properties are
asserted here rather than assumed, against values that are documented facts
about CIFAR-10 and not merely whatever the mirror happens to contain:

  * 50,000 train and 10,000 test images
  * ten classes, EXACTLY balanced at 5,000 and 1,000 per class
  * the canonical label order torchvision uses, because the class *names* are
    what a reader maps an integer prediction back to. A mirror with the same
    images under permuted indices would score identically and caption every
    picture wrong.
  * RGB, 32x32

This test needs the network but no GPU and no torch. It is the one suite here
that touches the Hub, which is deliberate: the claim being checked is about a
remote artifact, so mocking it would assert nothing.
"""

import sys

PASS = FAIL = 0

# torchvision's CIFAR10.classes, in order. The integer a model predicts is an
# index into THIS list.
CANONICAL_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

EXPECTED_ROWS = {"train": 50000, "test": 10000}
EXPECTED_PER_CLASS = {"train": 5000, "test": 1000}


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")


def main() -> None:
    import collections

    bar = "=" * 74
    print(bar)
    print("  test_cifar10_source.py")
    print(bar)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  SKIP: `datasets` is not installed in this environment.")
        print("        uv run tests/test_cifar10_source.py provisions it.")
        sys.exit(0)

    try:
        ds = load_dataset("uoft-cs/cifar10")
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  SKIP: could not reach the Hub ({type(exc).__name__}: {exc}).")
        print("        This suite needs the network by design -- it checks a")
        print("        remote artifact, and mocking it would assert nothing.")
        sys.exit(0)

    print("\n  -- the mirror is shaped like CIFAR-10 --")
    check(f"splits are exactly {sorted(EXPECTED_ROWS)}",
          sorted(ds.keys()) == sorted(EXPECTED_ROWS),
          f"got {sorted(ds.keys())}")

    for split, want in EXPECTED_ROWS.items():
        if split not in ds:
            continue
        check(f"{split}: {len(ds[split]):,} rows (CIFAR-10 has {want:,})",
              len(ds[split]) == want,
              "a mirror with a different split boundary produces numbers that "
              "cannot be compared with any published CIFAR-10 result")

    print("\n  -- labels --")
    names = ds["train"].features["label"].names
    check("class names match torchvision's canonical ORDER",
          list(names) == CANONICAL_CLASSES,
          f"got {list(names)}\nexpected {CANONICAL_CLASSES}\n"
          "Same images under permuted indices would train to an identical "
          "accuracy and caption every prediction wrong — the quietest possible "
          "way for a mirror to be wrong.")

    for split, per in EXPECTED_PER_CLASS.items():
        if split not in ds:
            continue
        counts = collections.Counter(ds[split]["label"])
        check(f"{split}: 10 classes, exactly {per:,} images each",
              len(counts) == 10 and set(counts.values()) == {per},
              f"got {len(counts)} classes, counts "
              f"{sorted(set(counts.values()))} -- CIFAR-10 is exactly balanced, "
              "so any imbalance means this is a subset or a resample")

    print("\n  -- images --")
    img = ds["train"][0]["img"]
    check(f"32x32 (got {getattr(img, 'size', None)})",
          getattr(img, "size", None) == (32, 32))
    check(f"RGB (got {getattr(img, 'mode', None)})",
          getattr(img, "mode", None) == "RGB",
          "the lab normalises with per-channel CIFAR-10 means; a different "
          "mode or channel order would silently mis-normalise every image")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
