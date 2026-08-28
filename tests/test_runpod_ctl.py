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
    on_disk = {p.name for p in REPO_ROOT.iterdir()
               if p.is_dir() and p.name[:2].isdigit()}
    missing = on_disk - set(ctl.EXAMPLES)
    r.check(not missing, "every numbered example is in the requirements table",
            f"missing: {sorted(missing)}")

    # ---- 2. Image must be a devel tag -----------------------------------
    r.check("devel" in ctl.DEFAULT_IMAGE,
            "default image is a 'devel' tag (ships nvcc)",
            "A 'runtime' image has no nvcc, so DeepSpeed cannot build its ops "
            "and every example fails with CUDA_HOME errors.")

    # ---- 3. Bootstrap command --------------------------------------------
    spec = ctl.EXAMPLES["06_huggingface_grpo"]
    boot = ctl.bootstrap("06_huggingface_grpo", spec, "main")
    for needle, label in [
        ("git clone", "clones the repository"),
        ("astral.sh/uv/install.sh", "installs uv"),
        ("uv pip install", "installs deps with uv, not pip"),
        ("deepspeed --num_gpus=1", "launches with the right GPU count"),
        ("grpo_gsm8k_train.py", "runs the right script"),
        ("tee /workspace/train.log", "tees a log to the persistent volume"),
        ("HF_HOME=/workspace", "points the HF cache at the volume"),
    ]:
        r.check(needle in boot, f"bootstrap {label}")

    r.check("pip install " not in boot.replace("uv pip install ", ""),
            "bootstrap never calls bare pip")
    r.check(ctl.bootstrap("01_basic_neuralnet", ctl.EXAMPLES["01_basic_neuralnet"],
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

    # ---- 6. 09_vss carries its host-RAM warning ---------------------------
    r.check(source_contains("runpod/runpod_ctl.py", "3 TB of HOST RAM")
            or source_contains("runpod/runpod_ctl.py", "HOST RAM"),
            "09_vss warns that host RAM, not VRAM, is the constraint")

    # ---- 7. Every example is sbatch-able (the CoreWeave promise) ---------
    for name in sorted(on_disk):
        folder = REPO_ROOT / name
        has_slurm = any("#SBATCH" in p.read_text(errors="ignore")
                        for p in folder.rglob("*.sh"))
        r.check(has_slurm, f"{name}: has a SLURM batch script",
                "A CoreWeave user must be able to sbatch every topic.")

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

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
