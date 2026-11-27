"""Offline tests for ZeRO-3 RunPod upload and download."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_URL = "https://example.invalid/fixture"
SPEC = importlib.util.spec_from_file_location("runpod_ctl", ROOT / "runpod/runpod_ctl.py")
ctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctl)

sys.path.insert(0, str(ROOT / ctl.ZERO3_EXAMPLE))
from train_zero3 import create_run_directory, mark_measurements_complete, save_rank_results


def completed_run(root, arm="bf16"):
    run = create_run_directory(root, arm)
    for rank in range(2):
        save_rank_results({
            "arm": arm, "rank": rank, "world_size": 2, "run_id": run.name,
        }, run)
    mark_measurements_complete(run, arm, 2)
    return run


def test_upload_selects_fresh_run_and_sends_manifest_last(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results"
    old = completed_run(root / "runpod-old")
    (root / "bf16_rank0.json").write_text("{}", encoding="utf-8")
    run = completed_run(root / "runpod-current", "nf4")
    assert old != run
    upload = Mock()
    monkeypatch.setattr(ctl.subprocess, "run", upload)

    assert ctl.upload_zero3(run.parent.relative_to(tmp_path), UPLOAD_URL) == 0
    expected = [run.relative_to(tmp_path) / name for name in
                ("nf4_rank0.json", "nf4_rank1.json", "measurements_complete.json")]
    assert [call.args[0][5] for call in upload.call_args_list] == [str(path) for path in expected]
    assert [call.args[0][7] for call in upload.call_args_list] == [
        f"Filename: {ctl.ZERO3_EXAMPLE}/{path}" for path in expected
    ]


@pytest.mark.parametrize("problem", ["missing", "mismatch", "incomplete", "ambiguous"])
def test_invalid_run_uploads_nothing(tmp_path, monkeypatch, problem):
    run = completed_run(tmp_path)
    rank = run / "bf16_rank1.json"
    if problem == "missing":
        rank.unlink()
    elif problem == "mismatch":
        row = json.loads(rank.read_text())
        rank.write_text(json.dumps({**row, "rank": 0, "run_id": "bf16-old"}))
    elif problem == "incomplete":
        (run / "measurements_complete.json").unlink()
    else:
        completed_run(tmp_path)
    upload = Mock()
    monkeypatch.setattr(ctl.subprocess, "run", upload)

    assert ctl.upload_zero3(tmp_path, UPLOAD_URL) == 1
    upload.assert_not_called()


def test_upload_failure_stops_before_publishing_manifest(tmp_path, monkeypatch):
    completed_run(tmp_path)
    upload = Mock(side_effect=[None, subprocess.CalledProcessError(22, "curl")])
    monkeypatch.setattr(ctl.subprocess, "run", upload)

    assert ctl.upload_zero3(tmp_path, UPLOAD_URL) == 1
    assert upload.call_count == 2
    assert all("measurements_complete" not in call.args[0][5] for call in upload.call_args_list)


@pytest.mark.parametrize("problem", [None, "download failure", "missing attachment", "failed launch"])
def test_receiver_requires_success_and_current_rank_downloads(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(ctl, "__file__", str(tmp_path / "runpod_ctl.py"))
    run = completed_run(tmp_path / "source")
    prefix = Path(ctl.ZERO3_EXAMPLE) / "results" / "runpod-fixture" / run.name
    messages = []
    responses = {}
    for path in run.iterdir():
        url = f"https://example.invalid/{path.name}"
        if problem == "missing attachment" and path.name == "bf16_rank1.json":
            # A previous fetch must not fill a hole in the current fetch.
            stale = tmp_path / "results" / "fixture" / prefix / path.name
            stale.parent.mkdir(parents=True)
            stale.write_bytes(path.read_bytes())
            continue
        messages.append({
            "event": "message", "id": path.name,
            "attachment": {"url": url, "name": str(prefix / path.name)},
        })
        if problem != "download failure" or path.name != "bf16_rank1.json":
            responses[url] = path.read_bytes()
    status = 124 if problem == "failed launch" else 0
    messages.append({"event": "message", "id": "done", "message": f"[6/6] DONE ZeRO-3 rc={status}"})
    responses[f"{ctl.NTFY}/fixture/json?poll=1"] = "\n".join(map(json.dumps, messages)).encode()

    def open_response(request, timeout):
        if request.full_url not in responses:
            raise OSError("synthetic download failure")
        return io.BytesIO(responses[request.full_url])

    monkeypatch.setattr(ctl.urllib.request, "urlopen", open_response)
    args = SimpleNamespace(topic="fixture", wait=True, wait_seconds=0, interval=0, terminate=False)
    assert ctl.cmd_fetch(args) == (0 if problem is None else 1)
    manifest = tmp_path / "results" / "fixture" / prefix / "measurements_complete.json"
    assert manifest.exists() is (problem is None)


def test_bootstrap_isolates_artifacts_and_preserves_failed_launch_status():
    command = ctl.bootstrap(ctl.ZERO3_EXAMPLE, ctl.EXAMPLES[ctl.ZERO3_EXAMPLE],
                            "main", topic="fixture", dry_run=True)
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "mktemp -d results/runpod-XXXXXXXX" in command
    assert ' --arm bf16 --output-dir "$output_root"' in command
    assert 'if [ "$rc" -eq 0 ]; then python3 -c' in command
    assert "|| rc=1" in command
    assert "DONE ZeRO-3 rc=$rc" in command
    assert "[ $rc -eq 124 ] && rc=0" not in command
