"""Offline checks for the ZeRO-3 launcher. GPU commands are replaced with stubs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


LAB = Path(__file__).resolve().parent


@pytest.fixture
def launcher(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    for name in ("run_deepspeed.sh", "train_zero3.py", "ds_config_bf16.json", "ds_config_nf4.json"):
        shutil.copy2(LAB / name, lab)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    trace = tmp_path / "commands.jsonl"
    for name in ("nvidia-smi", "uv"):
        stub = bin_dir / name
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['LAUNCH_TRACE'], 'a', encoding='utf-8') as file:\n"
            "    print(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]), file=file)\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LAUNCH_TRACE": str(trace)}
    env.pop("SLURM_SUBMIT_DIR", None)
    return lab, env, trace


@pytest.mark.parametrize("args, arm", [(("--plan",), "bf16"), (("bf16", "--plan"), "bf16"),
                                      (("--plan", "nf4"), "nf4")])
def test_plan_runs_before_gpu_checks(launcher, args, arm):
    lab, env, trace = launcher
    result = subprocess.run([lab / "run_deepspeed.sh", *args], env=env,
                            cwd=lab.parent, capture_output=True, text=True, check=True)
    assert f"Arm: {arm}" in result.stdout
    assert "GPU measurements: NOT PERFORMED" in result.stdout
    assert not trace.exists()
    assert not (lab / "results").exists()


@pytest.mark.parametrize("args, message", [
    (("nf4", "--max-steps", "1"), "Unsupported argument: --max-steps"),
    (("fp16",), "Unsupported argument: fp16"),
    (("bf16", "nf4"), "Choose one arm"),
    (("--plan", "--plan"), "Specify --plan only once"),
])
def test_rejects_ambiguous_or_unsupported_arguments(launcher, args, message):
    lab, env, trace = launcher
    result = subprocess.run(["bash", lab / "run_deepspeed.sh", *args], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr
    assert not trace.exists()


@pytest.mark.parametrize("args, arm", [((), "bf16"), (("bf16",), "bf16"), (("nf4",), "nf4")])
def test_gpu_launch_arguments(launcher, args, arm):
    lab, env, trace = launcher
    subprocess.run(["bash", lab / "run_deepspeed.sh", *args], env=env, cwd=lab.parent,
                   capture_output=True, text=True, check=True)
    assert [json.loads(line) for line in trace.read_text().splitlines()] == [
        ["nvidia-smi"],
        ["uv", "run", "--locked", "deepspeed", "--num_gpus=2", "train_zero3.py",
         "--arm", arm, "--output-dir", f"results/{arm}"],
    ]


def test_spooled_slurm_script_uses_submission_directory(launcher):
    lab, env, trace = launcher
    spool_script = shutil.copy2(lab / "run_deepspeed.sh", lab.parent / "slurm_script")
    subprocess.run(
        ["bash", spool_script, "nf4"], env={**env, "SLURM_SUBMIT_DIR": str(lab)},
        cwd=lab.parent, capture_output=True, text=True, check=True,
    )
    assert json.loads(trace.read_text().splitlines()[-1])[-4:] == [
        "--arm", "nf4", "--output-dir", "results/nf4",
    ]
