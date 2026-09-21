# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==5.16.1",
#   "trl==1.12.0",
#   "peft==0.20.0",
#   "torch==2.11.0",
#   "tomli; python_version < '3.11'",
# ]
# ///
"""
Regression test: every config kwarg is accepted by the INSTALLED library.

Run:
    uv run tests/test_config_kwargs.py

Why this suite exists
---------------------
A learner on Clawdeck ran `03_llms/02_trl_sft` on two rented GPUs. Both
ranks launched, the model loaded (596M params), the dataset loaded, and then:

    TypeError: TrainingArguments.__init__() got an unexpected keyword argument
    'logging_dir'

`logging_dir` was removed in transformers 5.x, which is what that lab's
uv.lock pins. The compile check in CI could never catch it: `logging_dir=` is
SYNTACTICALLY VALID. It fails only when the object is constructed — so library
API drift shipped silently and was discovered by someone paying for GPU time.

That is the gap this closes. For every call to a known config constructor, the
keyword arguments are parsed with `ast` and checked against the installed
class's signature. It needs no GPU and no model download, because the
constructors import fine on CPU — which is the only reason this is catchable at
all.

Running it against the tree at the time it was written found **20** rejected
kwargs across 13 files, not the one that was reported:

    logging_dir           removed          4 sites
    warmup_ratio          removed          6 sites  (only warmup_steps survives)
    overwrite_output_dir  removed          5 sites
    save_safetensors      removed          2 sites
    max_prompt_length     removed          1 site   (GRPOConfig)

Four of those sat in commands Clawdeck offers by name, so three more labs would
have failed the same way the reported one did.

Two properties that make this worth having rather than worse than nothing
-------------------------------------------------------------------------
1. **A class whose signature takes `**kwargs` is SKIPPED.** It accepts
   everything, so a pass would prove nothing and would give false confidence.
2. **The pinned versions here must match what the labs actually install.**
   This file imports one transformers; the labs each have their own uv.lock. If
   those drift apart, this suite happily validates a version no learner runs —
   passing while the labs are broken, which is the exact failure it exists to
   prevent. So the lock files are checked against the pins above, and a
   mismatch fails.
"""

import ast
import inspect
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories that are not this repository's source.
SKIP_DIRS = {".venv", "node_modules", "build", ".git", "__pycache__",
             "docusaurus-docs", ".docusaurus"}

# The versions this suite validates against. They must equal what every lab's
# uv.lock resolves, or the check is testing something no learner runs.
PINNED = {"transformers": "5.16.1", "trl": "1.12.0", "peft": "0.20.0",
          "torch": "2.11.0"}

# Free functions whose keyword arguments are worth checking, not just
# constructors. torch.distributed.barrier() earned its place: a fix in this
# repo passed it timeout=, which torch 2.13 accepts and torch 2.11 -- what
# every lab here locks -- does not. It raised TypeError on rented GPUs after
# both ranks had launched and rank 0 was 4 MB into a 170 MB download.
#
# The mistake underneath was reading the signature out of whichever torch a
# `find` happened to return first. The cache held 2.9, 2.10, 2.11, 2.13 and
# 2.14; timeout= exists in the last two only. That is precisely the failure
# this suite's pin-vs-lock check exists to prevent, so the fix belongs here.
TORCH_FUNCS = ("barrier", "init_process_group", "all_reduce", "broadcast",
               "all_gather", "reduce_scatter", "new_group")

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def load_constructors() -> dict:
    """
    Map constructor name -> (accepted kwargs, takes **kwargs).

    Imported from the installed libraries rather than hardcoded, so the check
    tracks the pin instead of a snapshot someone has to remember to update.
    """
    out = {}

    def add(name, cls):
        sig = inspect.signature(cls.__init__)
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
        out[name] = (set(sig.parameters) - {"self"}, var_kw)

    import transformers
    # The CONFIG classes...
    for n in ("TrainingArguments", "Seq2SeqTrainingArguments"):
        if hasattr(transformers, n):
            add(n, getattr(transformers, n))
    # ...and the TRAINERS themselves, which this file did not check for far
    # too long. `Trainer(tokenizer=...)` was deprecated in 4.x and REMOVED in
    # 5.x for `processing_class=`; 03_ocr called it and died at the Trainer
    # construction, after the model and dataset had loaded, on rented GPUs.
    # The kwarg scan below would have caught it the day it was written -- it
    # simply never knew to look at Trainer. None of these take **kwargs, so
    # every one of them is checkable.
    for n in ("Trainer", "Seq2SeqTrainer"):
        if hasattr(transformers, n):
            add(n, getattr(transformers, n))
    import trl
    for n in ("SFTConfig", "DPOConfig", "GRPOConfig", "RewardConfig",
              "OnlineDPOConfig", "CPOConfig", "KTOConfig", "ORPOConfig",
              "SFTTrainer", "DPOTrainer", "GRPOTrainer", "RewardTrainer",
              "OnlineDPOTrainer", "KTOTrainer"):
        if hasattr(trl, n):
            add(n, getattr(trl, n))
    import peft
    for n in ("LoraConfig",):
        if hasattr(peft, n):
            add(n, getattr(peft, n))

    # Free functions. Same contract: read the signature off the installed
    # library so the check tracks the pin rather than a remembered snapshot.
    import torch.distributed as dist
    for n in TORCH_FUNCS:
        fn = getattr(dist, n, None)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
        out[n] = (set(sig.parameters), var_kw)
    return out


def locked_version(lock: Path, pkg: str):
    m = re.search(rf'\nname = "{pkg}"\nversion = "([^"]+)"', lock.read_text())
    return m.group(1) if m else None



def fallback_imports(tree) -> set:
    """
    Line numbers of imports that have a GENUINE alternative elsewhere in the
    file -- i.e. the same symbol imported from a different module.

    That is what a real version fallback looks like:

        try:    from trl.experimental.cpo import CPOConfig
        except: from trl import CPOConfig

    Being inside a try/except is NOT sufficient, and assuming it was is how the
    first version of this check missed a live bug. 03_ocr wrapped its whole
    import block in `try/except ImportError` that printed "Missing required
    package" and exited 1. That is not a fallback -- it is a crash with better
    formatting, and the lab died on it. Only the presence of an ALTERNATIVE
    source for the same name makes an import safe to skip.
    """
    from collections import defaultdict
    sources = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                sources[alias.name].add(node.module)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if all(len(sources[a.name]) > 1 for a in node.names):
                out.add(node.lineno)
    return out


def main() -> None:
    bar = "=" * 74
    print(bar)
    print("  test_config_kwargs.py")
    print(bar)

    # ---- the pins must match what the labs install ------------------------
    print("\n  -- this suite validates the version the labs actually use --")
    import transformers, trl, peft, torch
    installed = {"transformers": transformers.__version__,
                 "trl": trl.__version__, "peft": peft.__version__,
                 # 2.11.0+cu128 and 2.11.0+cu130 are the same API on different
                 # CUDA builds; the local build need not match the labs' index.
                 "torch": torch.__version__.split("+")[0]}
    for pkg, want in PINNED.items():
        check(f"{pkg} {installed[pkg]} is the pinned {want}",
              installed[pkg] == want,
              "the PEP 723 header above and the running interpreter disagree")

    drift = []
    for lock in sorted(REPO.glob("*/*/uv.lock")):
        for pkg, want in PINNED.items():
            got = locked_version(lock, pkg)
            # torch locks as 2.11.0+cu128; the API is the version, not the
            # CUDA build, and this suite installs the PyPI wheel.
            if got is not None:
                got = got.split("+")[0]
            if got is not None and got != want:
                drift.append(f"{lock.parent}: {pkg} {got} != {want}")
    check(f"every lab's uv.lock agrees with these pins "
          f"({len(list(REPO.glob('*/*/uv.lock')))} locks)",
          not drift,
          "; ".join(drift[:4]) + " -- this suite would validate a version no "
          "learner runs, passing while the labs are broken")

    # ---- the constructors -------------------------------------------------
    print("\n  -- constructors under test --")
    ctors = load_constructors()
    check(f"loaded {len(ctors)} config constructors and functions", len(ctors) >= 8,
          f"got {sorted(ctors)}")
    skipped = []
    for name in sorted(ctors):
        accepted, var_kw = ctors[name]
        if var_kw:
            # Not a failure -- but say so out loud. A class that accepts
            # everything cannot be validated, and silently counting it as a
            # pass is how a checker starts lying.
            skipped.append(name)
            print(f"  SKIP  {name}: takes **kwargs, so a pass would prove nothing")
        else:
            check(f"{name}: {len(accepted)} accepted kwargs, checkable",
                  len(accepted) > 0)
    if skipped:
        print(f"        {len(skipped)} constructor(s) skipped: {skipped}")

    # ---- every call site --------------------------------------------------
    print("\n  -- every config constructor call in the repo --")
    findings = []
    scanned = 0
    n_archive = 0
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if any(p in SKIP_DIRS for p in rel.parts):
            continue
        # archive/ holds superseded scripts that nothing runs -- not the
        # manifest, not runpod_ctl, not CI. They are NOT maintained against the
        # current pins and three of them still call Trainer(tokenizer=...).
        # Excluded here for the same reason the symbol scan excludes them, and
        # REPORTED for the same reason: an exclusion nobody can see is how a
        # checker starts lying about its own coverage.
        if "archive" in rel.parts:
            n_archive += 1
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        scanned += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name not in ctors:
                continue
            accepted, var_kw = ctors[name]
            if var_kw:
                continue
            for kw in node.keywords:
                if kw.arg is None:
                    continue          # **spread; cannot be checked statically
                if kw.arg not in accepted:
                    findings.append((rel, kw.lineno, name, kw.arg))

    check(f"scanned {scanned} python files", scanned > 50)
    print(f"        (not scanned: {n_archive} file(s) under archive/, "
          f"superseded code run by nothing — they DO still call "
          f"Trainer(tokenizer=...))")
    check(f"no rejected kwargs ({len(findings)} found)", not findings,
          "; ".join(f"{p}:{ln} {c}(... {k}=...)"
                    for p, ln, c, k in findings[:6]))
    for rel, ln, cls, kw in findings:
        print(f"        {rel}:{ln}  {cls}(... {kw}=...) is not accepted by the "
              f"installed library")

    # ---- attributes the libraries have REMOVED ----------------------------
    # Kwargs are only half of API drift. `trainer.tokenizer` is not a keyword
    # argument -- it is an attribute read at the end of training, so nothing
    # above sees it. It crashed a real 2-GPU GRPO run at the SAVE step, after
    # the model had trained and the adapter was already on disk: the most
    # expensive possible place to fail.
    #
    # The table is validated against the installed library rather than merely
    # asserted, so it cannot rot into a snapshot. If transformers ever brings
    # `tokenizer` back, the first check below fails and says so, instead of this
    # suite quietly policing a rule that no longer exists.
    print("\n  -- attributes removed by the installed libraries --")
    from transformers import Trainer as _Trainer

    REMOVED_ATTRS = {"tokenizer": "processing_class"}
    live_removals = {}
    for gone, replacement in REMOVED_ATTRS.items():
        init_params = set(inspect.signature(_Trainer.__init__).parameters)
        really_gone = not hasattr(_Trainer, gone) and gone not in init_params
        check(f"Trainer.{gone} really is absent in transformers "
              f"{transformers.__version__}",
              really_gone,
              f"it exists again -- drop {gone!r} from REMOVED_ATTRS rather than "
              "leaving a check that polices a rule the library no longer has")
        # The replacement is set on the INSTANCE (self.processing_class = ...),
        # so it is absent from dir(cls) and must be looked for in __init__.
        check(f"the replacement Trainer.{replacement} exists",
              replacement in init_params or hasattr(_Trainer, replacement),
              f"{replacement!r} is not accepted either; the advice this check "
              "prints would send someone to a second dead attribute")
        if really_gone:
            live_removals[gone] = replacement

    def _is_trainer_ish(node: ast.AST) -> bool:
        """`trainer`, `self.trainer`, `grpo_trainer` -- but not `self.tokenizer`."""
        if isinstance(node, ast.Name):
            return "trainer" in node.id.lower()
        if isinstance(node, ast.Attribute):
            return "trainer" in node.attr.lower()
        return False

    attr_findings = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if any(p in SKIP_DIRS for p in rel.parts) or rel.parts[0] == "tests":
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in live_removals
                    and _is_trainer_ish(node.value)):
                attr_findings.append((rel, node.lineno, node.attr,
                                      live_removals[node.attr]))

    check(f"no removed Trainer attributes in use ({len(attr_findings)} found)",
          not attr_findings,
          "; ".join(f"{p}:{ln} .{a} -> use .{r}"
                    for p, ln, a, r in attr_findings[:6]))
    for rel, ln, attr, repl in attr_findings:
        print(f"        {rel}:{ln}  trainer.{attr} was removed — "
              f"use getattr(trainer, {repl!r}, None), or keep your own "
              f"reference to the tokenizer you passed in")

    # ---- every imported symbol must EXIST in the pinned library ----------
    # Kwargs and attributes were only two thirds of API drift. The third is a
    # symbol that simply vanishes: AutoModelForVision2Seq was renamed to
    # AutoModelForImageTextToText in transformers 5.x, and 03_ocr imported the
    # old name -- and never used it. A dead import took the whole lab down with
    #     cannot import name 'AutoModelForVision2Seq' from 'transformers'
    # on a lock pinning the very version this suite validates. compileall
    # cannot see it; only resolving the name against the real module can.
    #
    # Imports inside `try/except ImportError` are SKIPPED. Those are deliberate
    # version fallbacks and flagging them would punish the defensive pattern.
    print("\n  -- every imported symbol exists in the pinned libraries --")
    import importlib
    libs = {"transformers": transformers, "trl": trl, "peft": peft}
    missing, files, skipped_guarded = [], 0, 0
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if any(p in SKIP_DIRS for p in rel.parts):
            continue
        # archive/ is explicitly superseded code, run by nothing. Skipping it
        # is a real gap, so it is reported rather than silently dropped.
        if "archive" in rel.parts:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        files += 1
        guarded = fallback_imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.lineno in guarded:
                skipped_guarded += 1
                continue
            root = node.module.split(".")[0]
            if root not in libs:
                continue
            try:
                mod = (libs[root] if node.module == root
                       else importlib.import_module(node.module))
            except Exception:                              # noqa: BLE001
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    missing.append((rel, node.lineno, node.module, alias.name))

    check(f"scanned {files} files ({skipped_guarded} genuine fallbacks skipped)",
          files > 50)
    check(f"no removed symbols imported ({len(missing)} found)", not missing,
          "; ".join(f"{r}:{ln} from {m} import {n}"
                    for r, ln, m, n in missing[:5]))
    for rel, ln, mod, name in missing:
        print(f"        {rel}:{ln}  `from {mod} import {name}` — that name does "
              f"not exist in the pinned version")
    n_archive = sum(1 for p in REPO.rglob("*.py") if "archive" in p.parts)
    print(f"        (not scanned: {n_archive} file(s) under archive/, "
          f"superseded code run by nothing)")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
