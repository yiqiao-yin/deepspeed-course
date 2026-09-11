# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""
Regression test: multi-GPU labs must not download a dataset from every rank.

Run:
    uv run tests/test_multigpu_download_guard.py

Why this suite exists
---------------------
`01_basics/03_convnet_cifar10` shipped a `download_cifar10()` whose docstring
said "This prevents multiple processes from downloading simultaneously" and
whose body did nothing of the kind. Under the lab's own manifest command,
`deepspeed --num_gpus=2`, both ranks wrote the same 170 MB tarball into the
same `./data` and extracted over each other. torchvision's integrity check
then failed for BOTH with:

    RuntimeError: Dataset not found or corrupted. You can use download=True ...

which is a spectacularly misleading message for a file that downloaded fine,
twice. The tell is two interleaved progress bars both reaching 170M.

Three properties make this worth a static check rather than a runtime one:

  * **It passes on one GPU.** A single rank cannot race itself, so every
    local smoke test and every 1-GPU CI job is green.
  * **It is syntactically valid**, so `compileall` cannot see it.
  * **It was masked for months by an accident.** The data used to be
    re-hydrated onto the box at boot, so `./data` was already populated and
    neither rank ever downloaded. When that was cleaned up, the race surfaced
    on the first genuinely cold 2-GPU run. Nothing in the repo changed.

Scope, deliberately narrow
--------------------------
This checks `torchvision`-style downloads — a `download=` keyword — because
those do **no locking**. HuggingFace `from_pretrained` / `snapshot_download` /
`load_dataset` go through `huggingface_hub`, which takes `.lock` files and
survives concurrency on a normal filesystem. Flagging those too would produce
six findings that are not bugs, and a checker that cries wolf gets muted.

Why this is AST-based and not a grep
------------------------------------
The obvious implementation is

    assert re.search(r"get_rank\(\) == 0|is_main_process", src)

and it is the same mistake this repo has now made three times: keying on the
**presence** of a guard rather than on whether the guard actually governs the
dangerous call. Several scripts here carry rank guards around *printing* and
*checkpoint saving* while downloading unguarded — a file-wide grep passes all
of them. So this walks the tree and asks whether each individual download call
is lexically inside a rank- or world-size-gated branch, or has a rank-derived
`download=` argument.

The counterexamples at the bottom are permanent. A checker that has not been
watched rejecting bad input is not a checker.
"""

import ast
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent

# Names that indicate a condition is about which rank we are, or how many
# processes exist. A download gated on any of these is not a free-for-all.
RANK_NAMES = {
    "rank", "local_rank", "global_rank", "world_size", "is_main", "is_master",
    "is_main_process", "is_rank_zero", "is_local_main_process",
    "RANK", "LOCAL_RANK", "WORLD_SIZE",
}
RANK_CALLS = {"get_rank", "get_local_rank", "get_world_size", "is_initialized"}

PASS = FAIL = 0


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


def mentions_rank(node: ast.AST) -> bool:
    """True if an expression consults the rank or the world size."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in RANK_NAMES:
            return True
        if isinstance(sub, ast.Constant) and sub.value in RANK_NAMES:
            return True   # os.environ.get("RANK", ...)
        if isinstance(sub, ast.Attribute) and sub.attr in RANK_CALLS:
            return True
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in RANK_CALLS:
                return True
    return False


def unguarded_downloads(src: str) -> list[tuple[int, str]]:
    """
    Return (lineno, reason) for every download call NOT governed by a rank or
    world-size condition.

    A call is considered governed when either

      (a) its ``download=`` argument is itself rank-derived
          (``download=is_main`` — the pattern in train_modern_cifar10.py), or
      (b) it sits lexically inside an ``if``/``else`` whose test consults the
          rank or the world size.

    Walking is done with an explicit guard stack rather than ast.walk(), so a
    call nested several blocks deep inside a rank check is still seen as
    guarded. (An earlier checker in this repo recursed into a node's children
    and thereby never tested statements that were themselves block members —
    it silently passed everything.)
    """
    findings: list[tuple[int, str]] = []

    def visit(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "download":
                    literal_true = (isinstance(kw.value, ast.Constant)
                                    and kw.value.value is True)
                    if literal_true and not guarded:
                        findings.append(
                            (node.lineno,
                             "download=True with no enclosing rank/world_size guard"))
            for child in ast.iter_child_nodes(node):
                visit(child, guarded)
            return

        if isinstance(node, ast.If):
            inner = guarded or mentions_rank(node.test)
            for stmt in node.body:
                visit(stmt, inner)
            for stmt in node.orelse:
                visit(stmt, inner)
            visit(node.test, guarded)
            return

        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(ast.parse(src), False)
    return findings


def unbound_barriers(src: str) -> list[tuple[int, str]]:
    """
    Return (lineno, reason) for every ``barrier()`` that may land on cuda:0 for
    all ranks at once.

    Found the hard way, on the very fix that closed the download race. The
    guard worked -- rank 1 waited, the download happened once -- and then the
    job died 12 minutes later in the barrier itself:

        WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1)
          ran for 721595 milliseconds before timing out

    NCCL implements barrier as an all-reduce of a one-element tensor, so it
    must choose a device. torch picks, in order: (1) ``barrier(device_ids=)``,
    (2) the device bound at ``init_process_group``, (3) CPU, and failing those
    (4) *the current device* -- which, with nothing set, is cuda:0 on EVERY
    rank. torch's own source says this "may use default device 0, causing
    issues like hang or all processes creating context on device 0."

    Normally ``deepspeed.initialize()`` binds the device for you. A download
    guard that runs *before* initialize() -- which is the whole point of a
    download guard -- is therefore exactly the window where this bites.

    So: a barrier is safe if it passes ``device_ids``, or if a ``set_device``
    call appears earlier in the file. Line order is a sound proxy here because
    both live in the same straight-line preamble.
    """
    tree = ast.parse(src)
    set_device_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "set_device"
    ]
    findings = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        is_barrier = (isinstance(f, ast.Attribute) and f.attr == "barrier") or \
                     (isinstance(f, ast.Name) and f.id == "barrier")
        if not is_barrier:
            continue
        if any(kw.arg == "device_ids" for kw in n.keywords):
            continue
        if any(ln < n.lineno for ln in set_device_lines):
            continue
        findings.append(
            (n.lineno,
             "barrier() with no device_ids and no earlier set_device — "
             "every rank may post the all-reduce to cuda:0 and hang"))
    return findings


def has_barrier(src: str) -> bool:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "barrier":
                return True
            if isinstance(f, ast.Name) and f.id == "barrier":
                return True
    return False


def multi_gpu_labs() -> list[str]:
    manifest = yaml.safe_load((REPO / "clawdeck.yaml").read_text())
    labs = manifest.get("labs", manifest) if isinstance(manifest, dict) else manifest
    out = []
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        gpu = lab.get("gpu") or {}
        if int(gpu.get("count", 1) or 1) > 1:
            out.append(lab["id"])
    return out


def main() -> None:
    bar = "=" * 74
    print(bar)
    print("  test_multigpu_download_guard.py")
    print(bar)

    labs = multi_gpu_labs()
    print(f"\n  -- {len(labs)} labs declared multi-GPU in clawdeck.yaml --")
    for lab in labs:
        print(f"     {lab}")

    print("\n  -- no unguarded download in a multi-GPU lab --")
    checked = 0
    for lab in labs:
        for py in sorted((REPO / lab).glob("*.py")):
            src = py.read_text(errors="ignore")
            if "download" not in src:
                continue
            checked += 1
            bad = unguarded_downloads(src)
            rel = py.relative_to(REPO)
            detail = "\n".join(
                f"{rel}:{ln}  {why}" for ln, why in bad) + (
                "\n\nOnly rank 0 may download; the others must wait on a "
                "barrier. torchvision does no locking, so two ranks writing "
                "one directory corrupt it and BOTH then fail the integrity "
                "check. See download_cifar10() in "
                "01_basics/03_convnet_cifar10/cifar10_deepspeed.py."
            )
            check(f"{rel}", not bad, detail if bad else "")

            if not bad and any(
                    kw.arg == "download"
                    for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.Call) for kw in n.keywords):
                check(f"{rel} also barriers, so the waiters actually wait",
                      has_barrier(src),
                      "A rank guard WITHOUT a barrier is worse than no guard: "
                      "rank 1 skips the download and races ahead to read a "
                      "directory rank 0 is still writing. That fails only "
                      "sometimes, which is far harder to debug.")
    print(f"\n     ({checked} files mentioning a download were parsed)")

    print("\n  -- every barrier names its device, or binds one first --")
    for lab in labs:
        for py in sorted((REPO / lab).glob("*.py")):
            src = py.read_text(errors="ignore")
            if "barrier" not in src:
                continue
            bad = unbound_barriers(src)
            rel = py.relative_to(REPO)
            detail = "\n".join(f"{rel}:{ln}  {why}" for ln, why in bad) + (
                "\n\nPass device_ids=[local_rank], or call "
                "torch.cuda.set_device(local_rank) BEFORE the collective. "
                "deepspeed.initialize() would do it for you, but a download "
                "guard runs before initialize() by design — which is exactly "
                "when this hangs for 12 minutes and then SIGABRTs."
            )
            check(f"{rel}", not bad, detail if bad else "")

    # ---- the counterexamples ------------------------------------------------
    # Without these, a checker that returned [] unconditionally would pass
    # everything above and look perfect.
    print("\n  -- the checker can actually reject bad input --")

    exact_shipped_bug = '''
import torchvision
def download_cifar10():
    """This prevents multiple processes from downloading simultaneously."""
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
'''
    check("flags the bug as it actually shipped",
          len(unguarded_downloads(exact_shipped_bug)) == 1,
          "this is the verbatim shape of the code that failed on a 2-GPU box")

    guard_elsewhere = '''
import torchvision, torch.distributed as dist
def main():
    if dist.get_rank() == 0:
        print("only rank 0 prints")
    torchvision.datasets.CIFAR10(root='./data', download=True)
'''
    check("flags a download whose file HAS a rank guard, just not around it",
          len(unguarded_downloads(guard_elsewhere)) == 1,
          "this is precisely the case a grep-based check passes; if this ever "
          "returns 0 the checker has degraded to presence-matching")

    nested_deep = '''
import torchvision
def main():
    if world_size > 1:
        if is_main:
            for split in (True, False):
                torchvision.datasets.CIFAR10(root='./d', train=split, download=True)
'''
    check("does NOT flag a download nested several blocks inside a guard",
          unguarded_downloads(nested_deep) == [],
          "over-flagging a correct lab trains people to ignore the checker")

    kwarg_derived = '''
import torchvision
ds = torchvision.datasets.CIFAR10(root='./d', download=is_main)
'''
    check("does NOT flag download=is_main (train_modern_cifar10.py's pattern)",
          unguarded_downloads(kwarg_derived) == [],
          "the rank test can live in the argument rather than in an if")

    # The second bug, caught on real 2-GPU hardware by the fix for the first.
    naked_barrier = '''
import torch
if world_size > 1:
    deepspeed.init_distributed()
torch.distributed.barrier()
'''
    check("flags a barrier with no device_ids and no set_device",
          len(unbound_barriers(naked_barrier)) == 1,
          "this shape hung for 721 s on 2x3090 and then SIGABRTed")

    set_device_too_late = '''
import torch
torch.distributed.barrier()
torch.cuda.set_device(local_rank)
'''
    check("flags set_device that comes AFTER the barrier",
          len(unbound_barriers(set_device_too_late)) == 1,
          "train_modern_cifar10.py had set_device six lines BELOW its barrier; "
          "an order-insensitive check would have called that file correct")

    check("does NOT flag a barrier that passes device_ids",
          unbound_barriers(
              "import torch\ntorch.distributed.barrier(device_ids=[0])\n") == [])
    check("does NOT flag a barrier preceded by set_device",
          unbound_barriers("import torch\ntorch.cuda.set_device(0)\n"
                           "torch.distributed.barrier()\n") == [])

    check("barrier detection rejects a file with no barrier",
          not has_barrier(exact_shipped_bug))
    check("barrier detection accepts torch.distributed.barrier()",
          has_barrier("import torch\ntorch.distributed.barrier()\n"))

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
