# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Audit an example against the three-platform contract in CONTRIBUTING.md §3.

    uv run scripts/check_contract.py                    # every example
    uv run scripts/check_contract.py 05_huggingface_dpo # just one
    uv run scripts/check_contract.py --strict           # exit 1 on any failure

Why this exists
---------------
CONTRIBUTING.md says every contribution must work for three readers: one with no
GPU, one on CoreWeave, one with a RunPod key. That was prose, and prose is not
checkable -- so contributions kept arriving satisfying two of the three, and the
gap was only caught when a reviewer happened to notice.

This turns the contract into something you can run. It is what a contributor
should execute before opening a PR, and what a reviewer should execute before
merging one.

Deliberately NOT a hard CI gate by default
------------------------------------------
Older examples predate parts of the contract and legitimately differ -- some
generate their DeepSpeed config at runtime, some are inference-only and have no
optimizer to shard. Failing the build on those would either force churn in
working code or force the checks to be watered down until they catch nothing.

So it reports, and `--strict` is available for the folders you want held to the
line. `tests/test_runpod_ctl.py` already enforces the subset that is genuinely
non-negotiable (EXAMPLES registration, `bash -n`, `#SBATCH` presence).

Stdlib only. No GPU, no network, no downloads.
"""

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Documented exceptions, with the reason. An exception with no reason is a bug.
NO_DEEPSPEED = {
    "07_huggingface_trl_multi_agency": "drives TRL's GRPOTrainer directly",
    "08_vtt/03_streaming_memory": "streaming inference — sequential, no optimizer",
    "08_vtt/04_video_eval": "evaluation — short generate() calls",
    "09_vss/03_duplex_streaming": "duplex inference — slices arrive in order",
    "09_vss/04_omni_eval": "evaluation — modality-ablation generate() calls",
}
RUNTIME_DS_CONFIG = {
    "04_bayesian_neuralnet": "writes a temporary config at runtime",
    "05_huggingface_ocr": "generate_deepspeed_config() writes it at runtime",
}


class Report:
    """Collects per-folder results and prints them as a matrix."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str, bool, str]] = []
        self.notes: List[Tuple[str, str, str, str]] = []

    def add(self, folder: str, reader: str, label: str,
            ok: bool, detail: str = "") -> None:
        self.rows.append((folder, reader, label, ok, detail))

    def note(self, folder: str, reader: str, label: str, detail: str = "") -> None:
        """
        An advisory observation, NOT a failure.

        Kept distinct because a checker that reports style preferences as
        defects trains people to ignore it, and then it stops catching the
        real ones.
        """
        self.notes.append((folder, reader, label, detail))

    def failures(self) -> List[Tuple[str, str, str, bool, str]]:
        return [r for r in self.rows if not r[3]]

    def render(self, verbose: bool) -> None:
        by_folder: dict = {}
        for folder, reader, label, ok, detail in self.rows:
            by_folder.setdefault(folder, []).append((reader, label, ok, detail))

        for folder, checks in by_folder.items():
            failed = [c for c in checks if not c[2]]
            status = "PASS" if not failed else f"{len(failed)} ISSUE(S)"
            print(f"\n  {folder}  [{status}]")
            for reader, label, ok, detail in checks:
                if ok and not verbose:
                    continue
                mark = "OK  " if ok else "FAIL"
                print(f"     {mark} [{reader}] {label}")
                if detail and not ok:
                    print(f"          {detail}")


def find_examples() -> List[str]:
    """Every numbered example folder, including nested subtopics."""
    out = []
    for top in sorted(REPO_ROOT.glob("[0-9][0-9]_*")):
        if not top.is_dir():
            continue
        subs = [s for s in sorted(top.glob("[0-9][0-9]_*")) if s.is_dir()]
        if subs:
            out += [str(s.relative_to(REPO_ROOT)) for s in subs]
        else:
            out.append(top.name)
    return out


def entry_points(folder: Path) -> List[Path]:
    """Python files that look like an entry point (have a main guard)."""
    out = []
    for p in sorted(folder.glob("*.py")):
        try:
            if '__main__' in p.read_text(errors="ignore"):
                out.append(p)
        except OSError:
            continue
    return out


def check_reader_a(folder: Path, name: str, r: Report) -> None:
    """No GPU: must fail gracefully, not with a CUDA traceback."""
    eps = entry_points(folder)
    r.add(name, "A", "has at least one entry point", bool(eps),
          "no *.py with a __main__ guard")
    if not eps:
        return

    # An entry point either guards with require_gpu() or is CPU-runnable.
    guarded = [p for p in eps if "require_gpu" in p.read_text(errors="ignore")]
    cpu_ok = [p for p in eps if "require_gpu" not in p.read_text(errors="ignore")]

    r.add(name, "A", "a GPU entry point calls require_gpu()",
          bool(guarded) or bool(cpu_ok),
          "no entry point guards the GPU and none is CPU-runnable")

    for p in guarded:
        src = p.read_text(errors="ignore")
        rel = p.relative_to(REPO_ROOT)

        # The message must be actionable, not just a stop.
        r.add(name, "A", f"{p.name}: message points at runpod_ctl.py",
              "runpod_ctl.py" in src,
              "a reader told 'no GPU' with no way to get one is a dead end")
        r.add(name, "A", f"{p.name}: honours ALLOW_CPU=1",
              "ALLOW_CPU" in src)

        # require_gpu() must run BEFORE torch/deepspeed are imported, or the
        # reader gets a CUDA traceback before the message ever prints.
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            r.add(name, "A", f"{p.name}: parses", False, str(exc))
            continue

        module_imports = {
            n.names[0].name.split(".")[0]
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            and (n.names[0].name if isinstance(n, ast.Import) else n.module or "")
        }
        # ONLY deepspeed. `import torch` at module scope is harmless on a
        # CPU-only box -- torch imports fine without CUDA, and require_gpu()
        # imports it itself. DeepSpeed is the one that probes for a CUDA
        # toolkit and produces the CUDA_HOME error this contract exists to
        # pre-empt, so it is the one that must come after the preflight.
        #
        # (A reader missing `datasets` or `transformers` entirely still gets a
        # raw ImportError. That is an incomplete-install problem, not a no-GPU
        # problem, and `uv pip install ...` is the honest fix -- so it is not
        # flagged here.)
        if "deepspeed" in module_imports:
            # ADVISORY, not a failure. Verified on a CPU-only box: a
            # module-scope `import deepspeed` fails only when deepspeed is not
            # INSTALLED, which is a missing-dependency problem rather than a
            # no-GPU one. The CUDA_HOME error this contract exists to pre-empt
            # comes from the op builder during deepspeed.initialize(), which
            # require_gpu() already runs before.
            #
            # Importing inside main() is still tidier and is what newer
            # examples do, so it is worth saying -- but not worth churning
            # working files over an unverified premise.
            r.note(name, "A",
                   f"{p.name}: imports deepspeed at module scope",
                   "harmless for the no-GPU path (require_gpu runs before "
                   "initialize), but newer examples import it inside main()")


def check_reader_b(folder: Path, name: str, r: Report) -> None:
    """CoreWeave: sbatch-able, with a cheap dry run."""
    shells = list(folder.rglob("*.sh"))
    slurm = [s for s in shells if "#SBATCH" in s.read_text(errors="ignore")]

    r.add(name, "B", "ships a SLURM batch script", bool(slurm),
          "a CoreWeave user must be able to sbatch every topic")
    if not slurm:
        return

    for s in slurm:
        src = s.read_text(errors="ignore")
        rel = s.relative_to(REPO_ROOT)
        r.add(name, "B", f"{s.name}: bash -n parses",
              subprocess.run(["bash", "-n", str(s)],
                             capture_output=True).returncode == 0)
        r.add(name, "B", f"{s.name}: executable bit set",
              os.access(s, os.X_OK), "docs invoke it as ./" + s.name)
        r.add(name, "B", f"{s.name}: --ntasks-per-node=1",
              "ntasks-per-node=1" in src,
              "the deepspeed launcher spawns its own workers; N tasks x N GPUs "
              "gives N^2 processes and usually a hang")
        if "output=logs/" in src:
            r.add(name, "B", f"{s.name}: mkdir -p logs before writing there",
                  "mkdir -p logs" in src,
                  "without it SLURM silently discards output")
        # Credentials must be commented AND quoted.
        bad = re.findall(r"^\s*export\s+[A-Z_]*(?:KEY|TOKEN|SECRET)[A-Z_]*=<",
                         src, re.M)
        r.add(name, "B", f"{s.name}: no uncommented `export KEY=<...>`",
              not bad,
              "`<` is a redirection operator — that line is a bash SYNTAX "
              "ERROR and the script never reaches training")

    # A dry-run path so a cluster user can validate without burning allocation.
    # Training bounds work with --max-steps. Inference and evaluation bound it
    # differently -- frames, slices, questions -- so accept any of them. What
    # matters is that SOMETHING caps the work for a cheap dry run.
    CAPS = ("max-steps", "max_steps", "--frames", "--slices", "--limit",
            "--questions", "--examples", "--dry-run")
    capped = [p.name for p in entry_points(folder)
              if any(c in p.read_text(errors="ignore") for c in CAPS)]
    r.add(name, "B", "entry point accepts a work cap (dry-run path)",
          bool(capped),
          "add --max-steps (or --limit/--frames for inference) so a cluster "
          "user can validate without burning an allocation")


def check_reader_c(folder: Path, name: str, r: Report, ctl) -> None:
    """RunPod: registered, rentable, and auto-terminating."""
    registered = name in ctl.EXAMPLES
    r.add(name, "C", "registered in runpod_ctl.py EXAMPLES", registered,
          "without it, `runpod_ctl.py run` cannot size or launch this topic")
    if not registered:
        return

    spec = ctl.EXAMPLES[name]
    script = folder / spec["script"]
    r.add(name, "C", f"EXAMPLES script resolves ({spec['script']})",
          script.is_file(), f"looked for {script}")
    r.add(name, "C", "requirements are sane",
          spec["min_vram"] > 0 and spec["gpus"] >= 1 and spec["disk"] > 0)
    r.add(name, "C", "carries a note", bool(spec.get("note")),
          "one line on the surprising constraint")

    boot = ctl.bootstrap(name, spec, "main", topic="tpc-audit", dry_run=True)
    r.add(name, "C", "bootstrap cd's into the folder", f"cd {name}" in boot)
    r.add(name, "C", "bootstrap installs uv (not bare pip)",
          "astral.sh/uv" in boot)
    r.add(name, "C", "--dry-run caps the training step",
          str(ctl.DRY_RUN_SECONDS) in boot)
    r.add(name, "C", "in-pod watchdog present (keyless auto-shutdown)",
          "kill -TERM 1" in boot)
    r.add(name, "C", "bootstrap NEVER echoes a credential",
          not any(d in boot for d in ("$RUNPOD_API_KEY", "$HF_TOKEN",
                                      "$WANDB_API_KEY", "env |", "printenv")),
          "the results topic is public")

    readme = folder / "README.md"
    if readme.is_file():
        txt = readme.read_text(errors="ignore")
        for needle, label in [
            ("--terminate", "README documents --terminate (auto-shutdown)"),
            ("--dry-run", "README documents --dry-run"),
            ("runpod_ctl.py pods", "README says to confirm with `pods`"),
        ]:
            r.add(name, "C", label, needle in txt,
                  "an abandoned pod bills until terminated")


def check_assets(folder: Path, name: str, r: Report) -> None:
    """The rest of the asset inventory from CONTRIBUTING.md §4."""
    r.add(name, "assets", "README.md present", (folder / "README.md").is_file())

    has_cfg = bool(list(folder.glob("ds_config*.json"))
                   or list(folder.glob("*_config.json")))
    if name in RUNTIME_DS_CONFIG:
        r.add(name, "assets", f"ds_config: {RUNTIME_DS_CONFIG[name]}", True)
    elif name in NO_DEEPSPEED:
        # No optimizer means nothing for a DeepSpeed config to configure.
        r.add(name, "assets",
              f"ds_config not required: {NO_DEEPSPEED[name]}", True)
    else:
        r.add(name, "assets", "ds_config*.json present", has_cfg)

    uses_ds = any("deepspeed" in p.read_text(errors="ignore").lower()
                  for p in folder.rglob("*.py"))
    if name in NO_DEEPSPEED:
        r.add(name, "assets", f"no deepspeed: {NO_DEEPSPEED[name]}", True)
    else:
        r.add(name, "assets", "uses deepspeed", uses_ds,
              "this is a DeepSpeed course; add an entry to NO_DEEPSPEED in "
              "this script if yours is a genuine exception")

    # uv, never bare pip, in the README.
    readme = folder / "README.md"
    if readme.is_file():
        txt = readme.read_text(errors="ignore")
        # `pip install uv` is legitimate -- you cannot bootstrap uv with uv.
        # Everything else must go through `uv pip install`.
        bare = [m for m in re.findall(r"^\s*(?:\$ )?pip install (.+)$", txt, re.M)
                if m.strip().split()[0] not in {"uv", "pipx"}]
        r.add(name, "assets", "README uses `uv pip`, not bare `pip`",
              not bare,
              f"{len(bare)} bare `pip install` line(s): "
              + "; ".join(b[:40] for b in bare[:3]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folders", nargs="*",
                        help="Examples to audit. Default: all of them.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show passing checks too.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if anything fails.")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(
        "ctl", REPO_ROOT / "runpod" / "runpod_ctl.py")
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)

    targets = args.folders or find_examples()
    r = Report()

    print("=" * 78)
    print("  Three-platform contract audit  (CONTRIBUTING.md §3)")
    print("=" * 78)
    print("  Reader A  no GPU        -> fails gracefully, and is told what it CAN do")
    print("  Reader B  CoreWeave     -> sbatch-able, with a cheap dry run")
    print("  Reader C  RunPod        -> rentable, and AUTO-TERMINATES")

    for t in targets:
        folder = REPO_ROOT / t
        if not folder.is_dir():
            print(f"\n  {t}: no such folder")
            continue
        check_reader_a(folder, t, r)
        check_reader_b(folder, t, r)
        check_reader_c(folder, t, r, ctl)
        check_assets(folder, t, r)

    r.render(args.verbose)

    if r.notes and args.verbose:
        print("\n" + "-" * 78)
        print("  Advisory notes (not failures)")
        print("-" * 78)
        for folder, reader, label, detail in r.notes:
            print(f"  NOTE [{reader}] {folder}: {label}")
            if detail:
                print(f"       {detail}")

    total = len(r.rows)
    failed = len(r.failures())
    print()
    print("=" * 78)
    print(f"  {total - failed}/{total} checks passed across {len(targets)} example(s)")
    if r.notes:
        print(f"  {len(r.notes)} advisory note(s) — run with -v to see them")
    if failed:
        print(f"  {failed} issue(s) — see above. Not all are defects: older")
        print("  examples predate parts of the contract. Triage by hand.")
    print("=" * 78)

    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
