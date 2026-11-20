"""CPU-only tests for run isolation and measurement stage reporting."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import train_zero3 as experiment


def test_rank_reporting(tmp_path):
    # Both legacy flat files and completed, isolated runs must be ignored.
    (tmp_path / "bf16_rank1.json").write_text("{}", encoding="utf-8")
    old_run = experiment.create_run_directory(tmp_path, "bf16")
    new_run = experiment.create_run_directory(tmp_path, "bf16")
    assert new_run != old_run

    for rank in (0, 1):
        experiment.save_rank_results({"arm": "bf16", "rank": rank}, old_run)
    experiment.mark_measurements_complete(old_run, "bf16", 2)

    rank_zero = {"arm": "bf16", "rank": 0, "run_id": new_run.name}
    report = experiment.save_rank_results(rank_zero, new_run)
    assert json.loads(report.read_text()) == rank_zero
    with pytest.raises(RuntimeError, match="bf16_rank1.json"):
        experiment.mark_measurements_complete(new_run, "bf16", 2)
    assert not (new_run / "measurements_complete.json").exists()

    experiment.save_rank_results({**rank_zero, "rank": 1}, new_run)
    experiment.mark_measurements_complete(new_run, "bf16", 2)
    assert json.loads((new_run / "measurements_complete.json").read_text()) == {
        "arm": "bf16", "run_id": new_run.name, "world_size": 2,
        "rank_files": ["bf16_rank0.json", "bf16_rank1.json"],
    }


def test_failed_serialization_does_not_publish_rank_report(tmp_path):
    run_dir = experiment.create_run_directory(tmp_path, "nf4")
    with pytest.raises(TypeError):
        experiment.save_rank_results({"arm": "nf4", "rank": 0, "invalid": object()}, run_dir)
    assert not (run_dir / "nf4_rank0.json").exists()
    assert not (run_dir / "measurements_complete.json").exists()


@pytest.mark.parametrize("failure", [None, "engine", "barrier"])
def test_measurement_stages(monkeypatch, tmp_path, failure):
    config = experiment.load_config("nf4")
    loaded = torch.nn.Linear(1, 2, bias=False)
    with_lora = torch.nn.Linear(1, 3, bias=False)
    engine_module = torch.nn.Linear(1, 4, bias=False)
    distributed = Mock()
    cuda = Mock()
    cuda.memory_allocated.return_value = 20
    cuda.max_memory_allocated.return_value = 30
    cuda.get_device_name.return_value = "synthetic"
    initialize = Mock(return_value=(SimpleNamespace(module=engine_module), None, None, None))
    if failure == "engine":
        initialize.side_effect = RuntimeError("engine failed")
    if failure == "barrier":
        distributed.barrier.side_effect = RuntimeError("barrier failed")

    # Supply rank 1's report only once rank 0 reaches the collective.
    if not failure:
        def peer_finishes():
            run_dir, = tmp_path.iterdir()
            experiment.save_rank_results({"arm": "nf4", "rank": 1}, run_dir)
        distributed.barrier.side_effect = peer_finishes

    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(initialize=initialize))
    monkeypatch.setattr(torch, "distributed", distributed)
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "synthetic")
    monkeypatch.setattr(experiment, "initialize_distributed", lambda: {
        "local_rank": 0, "rank": 0, "world_size": 2, "rank_assignments": [],
    })
    monkeypatch.setattr(experiment, "load_model", Mock(return_value=(loaded, object())))
    monkeypatch.setattr(experiment, "attach_lora", Mock(return_value=with_lora))

    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            experiment.measure_model_loading("nf4", config, tmp_path)
        run_dir, = tmp_path.iterdir()
        assert not (run_dir / "measurements_complete.json").exists()
        if failure == "engine":
            assert not (run_dir / "nf4_rank0.json").exists()
        return

    result = experiment.measure_model_loading("nf4", config, tmp_path)
    initialize.assert_called_once_with(
        model=with_lora, model_parameters=list(with_lora.parameters()), config=config,
    )
    expected = {
        "before_lora": ("after_model_load_before_lora", 2),
        "before_engine": ("after_lora_before_engine", 3),
        "after_engine": ("after_deepspeed_engine_initialization", 4),
    }
    for stage, (label, count) in expected.items():
        assert result[stage]["measurement_stage"] == label
        assert result[stage]["logical_parameters"] == count
    assert "measurement_stage" not in result
    assert "logical_parameters" not in result
    assert "verified_post_engine" not in result
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "measurements_complete.json").is_file()
