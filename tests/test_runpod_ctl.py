# /// script
# requires-python = ">=3.9"
# ///
"""
Offline tests for runpod/runpod_ctl.py — no API key, no network.

    uv run tests/test_runpod_ctl.py

Covers the logic that can be wrong without RunPod being involved: the example
requirements table agreeing with what is actually in the repository, GPU
selection, bootstrap-command generation, and the guards that stop money being
spent by accident.

The live API paths (gpus / recommend / create / pods / terminate) are exercised
manually — see runpod/README.md.
"""

import importlib.util
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import REPO_ROOT, Results, source_contains  # noqa: E402

CTL = REPO_ROOT / "runpod" / "runpod_ctl.py"


def load_ctl():
    """Import runpod_ctl without executing main()."""
    spec = importlib.util.spec_from_file_location("runpod_ctl", CTL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    r = Results("runpod_ctl — offline logic")

    r.check(CTL.exists(), "runpod/runpod_ctl.py exists")
    ctl = load_ctl()

    # ---- 1. The example table must match the repository ----------------
    for name, spec in ctl.EXAMPLES.items():
        folder = REPO_ROOT / name
        r.check(folder.is_dir(), f"{name}: folder exists")
        script = folder / spec["script"]
        r.check(script.is_file(), f"{name}: script {spec['script']} exists",
                f"looked for {script}")
        r.check(spec["min_vram"] > 0 and spec["gpus"] >= 1 and spec["disk"] > 0,
                f"{name}: requirements are sane")

    # Every example folder in the repo should be represented.
    #
    # The taxonomy is two levels: NN_section/NN_topic. The top-level numbered
    # directories are SECTIONS (01_basics, 03_huggingface, ...) and are pure
    # containers -- they hold no training script and are not registered. The
    # examples are their numbered children. Checking the top level instead, as
    # this did before the reorganisation, would demand that every section be
    # registered as an example.
    on_disk = {
        f"{section.name}/{topic.name}"
        for section in REPO_ROOT.iterdir()
        if section.is_dir() and section.name[:2].isdigit()
        for topic in section.iterdir()
        if topic.is_dir() and topic.name[:2].isdigit()
    }
    missing = on_disk - set(ctl.EXAMPLES)
    r.check(not missing, "every numbered example is in the requirements table",
            f"missing: {sorted(missing)}")

    # ---- 2. Image must be a devel tag -----------------------------------
    r.check("devel" in ctl.DEFAULT_IMAGE,
            "default image is a 'devel' tag (ships nvcc)",
            "A 'runtime' image has no nvcc, so DeepSpeed cannot build its ops "
            "and every example fails with CUDA_HOME errors.")

    # ---- 3. Bootstrap command --------------------------------------------
    spec = ctl.EXAMPLES["03_huggingface/06_grpo"]
    boot = ctl.bootstrap("03_huggingface/06_grpo", spec, "main")
    for needle, label in [
        ("git clone", "clones the repository"),
        ("astral.sh/uv/install.sh", "installs uv"),
        ("uv pip install", "installs deps with uv, not pip"),
        ("deepspeed --num_gpus=1", "launches with the right GPU count"),
        ("grpo_gsm8k_train.py", "runs the right script"),
        ("/workspace/run.log", "writes its log to the persistent volume"),
        ("HF_HOME=/workspace", "points the HF cache at the volume"),
    ]:
        r.check(needle in boot, f"bootstrap {label}")

    r.check("pip install " not in boot.replace("uv pip install ", ""),
            "bootstrap never calls bare pip")
    r.check(ctl.bootstrap("01_basics/01_neuralnet", ctl.EXAMPLES["01_basics/01_neuralnet"],
                          "feature-x").count("-b feature-x") == 1,
            "bootstrap honours --branch")

    # ---- 4. Cost guard ----------------------------------------------------
    r.check(source_contains("runpod/runpod_ctl.py", "Refusing to create without --yes"),
            "create refuses without --yes")
    r.check(source_contains("runpod/runpod_ctl.py", "no instances currently available"),
            "capacity exhaustion is handled as a friendly message")
    r.check(source_contains("runpod/runpod_ctl.py", "User-Agent"),
            "sets a User-Agent (Cloudflare 403s urllib's default)")

    # ---- 5. No secrets baked in ------------------------------------------
    text = CTL.read_text()
    r.check("RUNPOD_API_KEY" in text and not any(
        tok in text for tok in ("rpa_", "sk-", "Bearer rp")),
        "reads the key from the environment; none hard-coded")

    # ---- 6. 05_video_speech carries its host-RAM warning ---------------------------
    r.check(source_contains("runpod/runpod_ctl.py", "3 TB of HOST RAM")
            or source_contains("runpod/runpod_ctl.py", "HOST RAM"),
            "05_video_speech warns that host RAM, not VRAM, is the constraint")

    # ---- 7. Every example is sbatch-able (the CoreWeave promise) ---------
    for name in sorted(on_disk):
        folder = REPO_ROOT / name
        has_slurm = any("#SBATCH" in p.read_text(errors="ignore")
                        for p in folder.rglob("*.sh"))
        r.check(has_slurm, f"{name}: has a SLURM batch script",
                "A CoreWeave user must be able to sbatch every topic.")

    # ---- 7b. Result collection (the no-SSH path) --------------------------
    boot_c = ctl.bootstrap("01_basics/01_neuralnet", ctl.EXAMPLES["01_basics/01_neuralnet"],
                           "main", topic="tpc123", dry_run=True)
    r.check("tpc123" in boot_c, "bootstrap embeds the results topic")
    r.check("report()" in boot_c, "bootstrap defines a progress reporter")
    r.check("DONE" in boot_c, "bootstrap emits a DONE marker for fetch --wait")
    r.check("Filename:" in boot_c, "bootstrap attaches the log file")
    r.check(f"timeout {ctl.DRY_RUN_SECONDS}" in boot_c,
            f"--dry-run caps the run at {ctl.DRY_RUN_SECONDS}s")

    boot_plain = ctl.bootstrap("01_basics/01_neuralnet", ctl.EXAMPLES["01_basics/01_neuralnet"],
                               "main")
    r.check("timeout" not in boot_plain, "without --dry-run there is no timeout cap")
    r.check("ntfy" not in boot_plain.lower() or "report(){ :; }" in boot_plain,
            "without --collect nothing is pushed anywhere")

    # ---- The three bugs that made every pod run report success ----------
    # All three shipped in 9c696ed and survived 78 commits. Each one made a
    # FAILED run look like a successful one, which is the worst failure mode
    # for a verification harness: it does not merely lose information, it
    # manufactures false confidence.

    # (1) rc=$? was evaluated AFTER the log-upload curl, so the DONE marker
    #     reported the curl's exit code -- essentially always 0. A real GLM-5.3
    #     pod run reported "DONE rc=0" while its log showed return code 2.
    i_rc = boot_c.find("; rc=$?")
    i_curl = boot_c.find("curl -s -m 60 -T")
    r.check(i_rc != -1 and i_curl != -1 and i_rc < i_curl,
            "rc is captured right after the launcher, before the log upload",
            "If the curl runs first, $? is the curl's status and every run "
            "reports success no matter what training did.")
    r.check('rc=$rc' in boot_c and 'DONE rc=$?' not in boot_c,
            "the DONE marker reports the captured $rc, not a fresh $?")

    # (2) '[2/6] repo cloned' was printed unconditionally and the following
    #     `cd` failed silently, so a failed clone ran the launcher from
    #     /workspace and the failure looked like a broken example.
    r.check("if [ ! -d /workspace/deepspeed-course ]" in boot_c,
            "the clone result is checked, not assumed")
    r.check("no such example dir" in boot_c,
            "a failed `cd <example>` aborts instead of running from the wrong cwd")

    # (3) --dry-run appended `|| true`, forcing rc=0 whatever happened -- and
    #     --dry-run is the command every README documents as the default, so
    #     the documented path could not surface a failure at all. Only
    #     timeout's own 124 ("I stopped it") may count as success.
    r.check("[ $rc -eq 124 ] && rc=0" in boot_c,
            "--dry-run treats ONLY timeout's 124 as success")
    after_launch = boot_c[boot_c.find(f"timeout {ctl.DRY_RUN_SECONDS}"):][:120]
    r.check("|| true" not in after_launch,
            "--dry-run does not blanket-suppress the launcher's exit code",
            "`timeout N cmd || true` reports success for a crash at second 3.")

    # GitHub rate-limits anonymous clones from cloud IP ranges and answers with
    # an auth challenge, which hangs git on a prompt nothing can answer.
    r.check("GIT_TERMINAL_PROMPT=0" in boot_c,
            "git cannot block on a credential prompt")
    r.check("codeload.github.com" in boot_c,
            "there is a tarball fallback when the anonymous clone is refused")
    for cred in ("GITHUB_TOKEN", "GH_TOKEN", "git config --global credential"):
        r.check(cred not in boot_c,
                f"the clone fallback does not put {cred} on the pod")

    # The transport is public, so the bootstrap must never echo credentials.
    for danger in ("$RUNPOD_API_KEY", "$HF_TOKEN", "$WANDB_API_KEY", "env |", "printenv"):
        r.check(danger not in boot_c,
                f"bootstrap does not leak {danger} to a public topic")

    r.check(hasattr(ctl, "cmd_fetch"), "fetch command exists")
    r.check(source_contains("runpod/runpod_ctl.py", "DSC_NTFY_SERVER"),
            "transport server is overridable for self-hosting")

    # ---- 7c. No committed secrets anywhere in the repo --------------------
    # The repository is public. A leaked key is the one mistake that cannot be
    # undone by a follow-up commit.
    import re as _re
    patterns = {
        "RunPod key": _re.compile(r"\brpa_[A-Za-z0-9]{20,}"),
        "OpenAI key": _re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        "HF token": _re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
        "AWS key id": _re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "GitHub token": _re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
        "private key": _re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    }
    skip_dirs = {".git", "node_modules", "__pycache__", "build", ".docusaurus", ".venv"}
    scanned = leaks = 0
    for f in REPO_ROOT.rglob("*"):
        if not f.is_file() or set(f.parts) & skip_dirs:
            continue
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".wav", ".ico"}:
            continue
        try:
            body = f.read_text(errors="ignore")
        except Exception:
            continue
        scanned += 1
        for label, pat in patterns.items():
            if pat.search(body):
                leaks += 1
                r.check(False, f"no {label} in {f.relative_to(REPO_ROOT)}",
                        "A committed credential must be rotated immediately.")
    r.check(leaks == 0, f"no credentials found in {scanned} scanned files")

    # Credentials must come from the environment, never be hard-coded.
    r.check(source_contains("runpod/runpod_ctl.py", 'os.environ.get("RUNPOD_API_KEY")'),
            "runpod_ctl reads its key from the environment")

    # ---- 7d. Scripts invoked as ./x.sh must be executable ----------------
    # A doc that says `./submit_job.sh` against a non-executable file fails
    # with "Permission denied" — a real malfunction, not a style nit.
    import os as _os
    docs_text = "\n".join(
        f.read_text(errors="ignore")
        for f in list(REPO_ROOT.rglob("*.md"))
        if "node_modules" not in f.parts and ".git" not in f.parts
    )
    for sh in sorted(REPO_ROOT.rglob("*.sh")):
        if set(sh.parts) & {".git", "node_modules"}:
            continue
        if f"./{sh.name}" in docs_text:
            r.check(_os.access(sh, _os.X_OK),
                    f"{sh.relative_to(REPO_ROOT)}: executable (docs invoke ./{sh.name})",
                    "chmod +x it, or the documented command fails.")

    # ---- 8. Every shell script must PARSE ---------------------------------
    # `export VAR=<PLACEHOLDER>` is a bash syntax error ('<' is a redirection
    # operator), so a script containing it dies on that line and never reaches
    # training. Seven SLURM scripts shipped that way.
    import subprocess
    for sh in sorted(REPO_ROOT.rglob("*.sh")):
        if ".git" in sh.parts or "node_modules" in sh.parts:
            continue
        rel = sh.relative_to(REPO_ROOT)
        proc = subprocess.run(["bash", "-n", str(sh)],
                              capture_output=True, text=True)
        r.check(proc.returncode == 0, f"{rel}: valid bash syntax",
                proc.stderr.strip()[:200])

    # ---- 8a. Every example must be a uv PROJECT ---------------------------
    # CONTRIBUTING.md section 4 requires that
    #     cd <example> && uv sync
    # works from a fresh clone. That needs two committed files, and the lock is
    # the one that matters: without it `uv pip install torch deepspeed` resolves
    # to whatever is newest that day, so a tutorial that worked in March breaks
    # in September with nobody having touched it.
    #
    # Checked structurally rather than by running uv, so this stays a fast,
    # network-free CPU test.
    import tomllib
    for name, spec in ctl.EXAMPLES.items():
        folder = REPO_ROOT / name
        if not folder.is_dir():
            continue
        # The project lives beside the registered training script, which is not
        # always the topic root: `04_video_text` and `05_video_speech` are legacy aggregate
        # entries whose script sits inside a subtopic that carries its own
        # project. Walk up from the script to the topic root and take the first
        # pyproject found.
        cand = (folder / spec["script"]).parent
        proj_dir = None
        while True:
            if (cand / "pyproject.toml").is_file():
                proj_dir = cand
                break
            if cand == folder or folder not in cand.parents:
                break
            cand = cand.parent

        r.check(proj_dir is not None, f"{name}: has pyproject.toml",
                f"expected beside {spec['script']} or at the topic root; "
                "cd there and run `uv add <deps>` -- CONTRIBUTING.md section 4")
        if proj_dir is None:
            continue
        pyproject = proj_dir / "pyproject.toml"
        r.check((proj_dir / "uv.lock").is_file(),
                f"{name}: has a committed uv.lock",
                f"run `uv lock` in {proj_dir.relative_to(REPO_ROOT)} and commit "
                "it -- an unlocked example is a tutorial with an unwritten "
                "expiry date")
        try:
            cfg = tomllib.loads(pyproject.read_text())
        except Exception as exc:
            r.check(False, f"{name}: pyproject.toml parses", str(exc)[:120])
            continue
        proj = cfg.get("project", {})
        r.check(bool(proj.get("dependencies")),
                f"{name}: declares dependencies")
        r.check(cfg.get("tool", {}).get("uv", {}).get("package") is False,
                f"{name}: sets [tool.uv] package = false",
                "these are runnable examples, not distributable libraries; "
                "without it uv tries to BUILD the folder and uv sync fails")
        # W&B is optional everywhere -- the scripts guard the import.
        r.check("wandb" not in " ".join(proj.get("dependencies", [])),
                f"{name}: wandb is not a hard dependency",
                "every script wraps `import wandb` in try/except; it belongs "
                "in [project.optional-dependencies]")

    # ---- 8b. Every launcher-run script must TOLERATE --local_rank ----------
    # The deepspeed launcher injects --local_rank=N into each worker's argv. A
    # script using strict parse_args() exits 2 with "unrecognized arguments"
    # before training starts, so `deepspeed --num_gpus=N <script>` -- the exact
    # command these examples document -- fails every time, on every GPU count.
    #
    # It is invisible locally: the script runs fine under plain `python`, and
    # the CPU suite never invokes a launcher. Six examples shipped this way and
    # it took renting a 2-GPU pod to notice.
    #
    # parse_known_args() is the fix; declaring --local_rank explicitly also
    # works, so either counts.
    import re as _re
    for name, spec in ctl.EXAMPLES.items():
        if spec.get("launcher") == "python":
            continue                      # not started by the deepspeed launcher
        script = REPO_ROOT / name / spec["script"]
        if not script.is_file():
            continue
        src = script.read_text(errors="ignore")
        if "argparse" not in src:
            continue                      # no parser, nothing to reject
        # Strip comments first. The obvious check -- "parse_known_args" in
        # src -- is satisfied by a COMMENT mentioning it, which is exactly
        # what the fix for this bug adds. Verified: without this the guard
        # passes on a file whose only mention is the comment above the call.
        code = "\n".join(_re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        # Match --local_rank ANYWHERE in the add_argument call, not just as
        # the first option string: 03_huggingface/03_ocr declares it as
        # add_argument("--local-rank", "--local_rank", ...), which an anchored
        # pattern reads as missing. That was a checker bug, not a code bug.
        tolerant = (_re.search(r"\.parse_known_args\s*\(", code) is not None
                    or _re.search(r"add_argument\([^)]*--local[-_]rank", code)
                    is not None)
        r.check(tolerant,
                f"{name}/{spec['script']}: tolerates --local_rank",
                "uses strict parse_args(); the deepspeed launcher injects "
                "--local_rank=N and argparse will exit 2 before training")

    # ---- 9. The POD START COMMAND must parse too ---------------------------
    # bootstrap() returns one string that the pod runs as `bash -lc "<string>"`.
    # A syntax error there is far worse than in a checked-in script, because
    # nothing local ever executes it: the pod is created, it BILLS, and it
    # silently runs nothing -- no clone, no training, no report. There is no
    # error message anywhere, only an idle pod and an empty results directory.
    #
    # This is not hypothetical. The auto-terminate watchdog ended its line with
    # `&`, and the steps are joined with "; ", producing `) &; report ...` --
    # unparseable. Every --collect pod created after that commit did nothing at
    # all, and the only symptom was silence.
    import tempfile
    for ex in sorted(ctl.EXAMPLES):
        cmd = ctl.bootstrap(ex, ctl.EXAMPLES[ex], "main",
                            topic="t", dry_run=True)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(cmd)
            tmp = fh.name
        proc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        pathlib.Path(tmp).unlink(missing_ok=True)
        r.check(proc.returncode == 0,
                f"{ex}: pod start command is valid bash",
                proc.stderr.strip()[:200]
                + "  <- the pod would bill while running NOTHING")

    # The watchdog must survive being joined with "; " -- assert the shape
    # directly, so the reason is documented at the point of failure.
    one = ctl.bootstrap("01_basics/01_neuralnet", ctl.EXAMPLES["01_basics/01_neuralnet"],
                        "main", topic="t")
    r.check(") &;" not in one,
            "the watchdog does not leave a bare `&` before a `;`",
            "`cmd &; next` is a bash syntax error and kills the whole command")

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
