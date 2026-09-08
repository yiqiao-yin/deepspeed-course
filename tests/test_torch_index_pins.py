# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Regression test: torch-linked packages resolve from the SAME index as torch.

Run:
    uv run tests/test_torch_index_pins.py

Why this suite exists
---------------------
`01_basics/03_convnet_cifar10` was green in every existing check and still could
not run. On a CUDA 12.8 box both ranks died during import:

    RuntimeError: operator torchvision::nms does not exist

raised from inside `torch/_library/fake_impl.py`, which reads like a torch bug
and is not one. The cause was resolution:

    [tool.uv.sources]
    torch = { index = "pytorch-cu128" }     # torchvision was NOT listed

With `explicit = true`, only packages named in `[tool.uv.sources]` come from
that index. So torch resolved to `2.11.0+cu128` from download.pytorch.org while
torchvision resolved to a plain PyPI wheel built against a *different* torch.
torchvision's compiled `_C.so` then fails to register its ops, and the first
thing to touch one raises.

Nothing syntactic is wrong, both packages install cleanly, and the failure
appears only when the environment is built and imported — which is why this
needed its own check.

What this checks, and why it reads the LOCK rather than the pyproject
--------------------------------------------------------------------
The obvious lint is "if pyproject pins torch, it must pin torchvision too".
That is a check on the *declaration*. This one checks the *resolution*: for
every lab whose `uv.lock` draws torch from a custom index, every torch-linked
package in that same lock must come from the same host.

That is strictly stronger. It also catches a lock that has drifted from a
correct pyproject — regenerate a lock against a different index and the
declaration still looks right while the environment is broken.

It is offline and instant: lock files are text, and nothing is installed.

Scope note
----------
A package belongs here only if it ships a compiled extension linked against
torch. `torchvision` and `torchaudio` are the ones this repo uses; the set is
listed explicitly rather than inferred, because a wrong guess in either
direction is worse than a short list.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Packages with a compiled extension linked against a specific torch build.
# Being drawn from a different index than torch is what breaks them.
TORCH_LINKED = ("torchvision", "torchaudio", "torchtext", "torchcodec")

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def resolved(lock_text: str, pkg: str):
    """(version, host) for a package in a uv.lock, or None if absent."""
    m = re.search(
        rf'\nname = "{pkg}"\nversion = "([^"]+)"\nsource = \{{ \w+ = "([^"]+)" \}}',
        lock_text)
    if not m:
        return None
    version, url = m.groups()
    host = url.split("/")[2] if "//" in url else url
    return version, host


def main() -> None:
    bar = "=" * 74
    print(bar)
    print("  test_torch_index_pins.py")
    print(bar)

    locks = sorted(REPO.glob("*/*/uv.lock"))
    check(f"found lock files to inspect ({len(locks)})", len(locks) > 10)

    print("\n  -- torch-linked packages share torch's index --")
    checked = 0
    for lock in locks:
        lab = lock.parent.relative_to(REPO)
        text = lock.read_text()
        t = resolved(text, "torch")
        if not t:
            continue
        t_ver, t_host = t
        for pkg in TORCH_LINKED:
            got = resolved(text, pkg)
            if not got:
                continue
            p_ver, p_host = got
            checked += 1
            check(f"{lab}: {pkg} {p_ver} from {p_host} matches torch's "
                  f"{t_host}",
                  p_host == t_host,
                  f"torch is {t_ver} from {t_host} but {pkg} is {p_ver} from "
                  f"{p_host}. {pkg} has a compiled extension linked against a "
                  "specific torch build; drawn from a different index its ops "
                  "never register and the lab dies on import with e.g. "
                  f"'operator {pkg}::nms does not exist'. Add "
                  f"`{pkg} = {{ index = ... }}` to [tool.uv.sources] and "
                  "re-run `uv lock`.")

    check(f"at least one torch-linked pairing was actually inspected "
          f"({checked})", checked > 0,
          "if this is 0 the suite is passing vacuously")

    # The declaration should agree with the resolution, so a future `uv lock`
    # cannot quietly undo the fix.
    print("\n  -- pyproject declares what the lock resolved --")
    for lock in locks:
        lab = lock.parent
        pyproject = lab / "pyproject.toml"
        if not pyproject.is_file():
            continue
        decl = pyproject.read_text()
        if "torch = { index" not in decl:
            continue
        for pkg in TORCH_LINKED:
            # Only require the pin if the lab actually depends on the package.
            if not re.search(rf'"{pkg}[><=~\[\]"]', decl):
                continue
            check(f"{lab.relative_to(REPO)}: pyproject pins {pkg} to an index",
                  f"{pkg} = {{ index" in decl,
                  f"torch is pinned to a custom index but {pkg} is not, so "
                  "`uv lock` will draw it from PyPI and the mismatch returns "
                  "the next time the lock is regenerated")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
