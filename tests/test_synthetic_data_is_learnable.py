# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Regression test: synthetic training data actually contains a learnable signal.

Run:
    uv run tests/test_synthetic_data_is_learnable.py

Why this suite exists
---------------------
`01_basics/02_convnet` generated its dataset like this:

    x_data = torch.randn(num_samples, 1, 28, 28)     # noise images
    y_data = torch.randint(0, 10, (num_samples,))    # labels INDEPENDENT of x

There is zero mutual information between inputs and labels, so ~10% is not a
poor result on 10 classes — it is the information-theoretic **ceiling**. No
architecture, learning rate or epoch count can beat it.

The script exited 0, printed "Finished Successfully", and then advised:

    "Poor. Consider training longer or adjusting hyperparameters"

which sends a learner to tune an unreachable target. Two runs on a 3090
returned 10.29% and 9.49%, identical from first epoch to last — the signature
of a classifier collapsing to one class, the correct degenerate answer when
there is no signal.

Nothing in CI could catch it. The script runs, exits 0, and reports a number.
`compileall` sees valid Python; the manifest checker sees a valid lab.

What this asserts
-----------------
For each generator, that a small model trained on it beats chance on a HELD-OUT
split drawn with a different seed. That is the property the fix is about:

  * held-out, not training accuracy — memorising random labels is possible on
    the training set and proves nothing (Zhang et al., "Understanding deep
    learning requires rethinking generalization"). It is exactly the thing the
    old script mistook for evidence of learning.
  * a different seed for the eval split, so train and eval describe the same
    task. A generator that draws a fresh hidden task per call makes them
    unrelated problems, and training then makes the metric worse — a bug this
    course shipped once already, in a ranking generator.

And the counterexample, without which the check proves nothing: the ORIGINAL
random-label generator is reconstructed here and asserted to FAIL. A test that
only ever sees good data would pass if it always returned True.
"""

import ast
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def load_generator(rel_path: str, func: str):
    """
    Pull one generator out of a training script without importing it.

    The scripts import deepspeed at module scope in places, and deepspeed on a
    CPU box without a launcher tries MPI discovery and dies. Parsing out the
    single function keeps this suite CPU-only and dependency-light.
    """
    src = (REPO / rel_path).read_text()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == func), None)
    if fn is None:
        return None
    ns = {"torch": torch, "DataLoader": DataLoader,
          "TensorDataset": TensorDataset}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), rel_path, "exec"), ns)
    return ns[func]


def beats_chance(loader_fn, n_classes: int, epochs: int = 2) -> float:
    """Held-out accuracy of a small MLP, as a percentage."""
    # Pass only the kwargs this generator actually accepts. A generator with
    # no `seed` parameter must still be MEASURED -- the first version of this
    # helper passed seed= unconditionally and died with
    #     TypeError: get_data_loader() got an unexpected keyword argument 'seed'
    # against the very generator it was written to catch. CI went red, but for
    # the wrong reason and with a message that named a signature mismatch
    # instead of the finding.
    import inspect
    accepts = set(inspect.signature(loader_fn).parameters)

    def build(n, seed):
        kw = {"batch_size": 256, "num_samples": n}
        if "seed" in accepts:
            kw["seed"] = seed
        else:
            torch.manual_seed(seed)   # the only handle such a generator offers
        return loader_fn(**kw)

    torch.manual_seed(0)
    train = build(6000, 1)
    test = build(1500, 2)

    x0, _ = next(iter(train))
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(x0[0].numel(), 64), nn.ReLU(),
                          nn.Linear(64, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in train:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    correct = total = 0
    with torch.no_grad():
        for x, y in test:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def main() -> None:
    bar = "=" * 74
    print(bar)
    print("  test_synthetic_data_is_learnable.py")
    print(bar)

    print("\n  -- 01_basics/02_convnet --")
    gen = load_generator("01_basics/02_convnet/train_ds.py", "get_data_loader")
    check("get_data_loader was found in the shipped source", gen is not None)
    if gen is None:
        print(f"\n  {PASS} passed, {FAIL} failed")
        sys.exit(1)

    acc = beats_chance(gen, n_classes=10)
    chance = 10.0
    check(f"held-out accuracy {acc:.1f}% clears the {chance:.0f}% chance floor",
          acc > chance * 2.5,
          f"got {acc:.1f}%. If the labels are independent of the images there "
          "is no signal to find, and chance is the CEILING rather than a poor "
          "result -- the script would then advise a reader to train longer "
          "toward a target that cannot be reached.")

    # Learnable but not trivial: if one epoch already saturates, the accuracy
    # number stops discriminating and the run length teaches nothing.
    quick = beats_chance(gen, n_classes=10, epochs=1)
    check(f"one epoch already shows learning ({quick:.1f}%), so a smoke test "
          "reads as success", quick > chance * 2.5,
          "Clawdeck runs this lab with --epochs 1; if that lands at chance a "
          "beginner concludes they broke something")

    # ---- the counterexample -------------------------------------------------
    # Without this the check proves nothing: a function that always returned
    # True would pass everything above.
    print("\n  -- the counterexample: the generator this replaced --")

    def random_labels(batch_size: int, num_samples: int = 10000, seed: int = 42,
                      **kw):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(num_samples, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (num_samples,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                          shuffle=True)

    bad = beats_chance(random_labels, n_classes=10)
    check(f"random labels stay at chance ({bad:.1f}%) and are REJECTED",
          bad <= chance * 2.5,
          f"got {bad:.1f}% on data with zero mutual information -- if this "
          "passes, the check is not measuring learnability at all")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
