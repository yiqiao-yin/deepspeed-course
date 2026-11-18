"""CPU-only tests for ZeRO-3 parameter measurements."""

import json
from types import SimpleNamespace

import torch

from train_zero3 import parameter_stats


class FakeModel:
    """Minimal model exposing named parameters."""

    def __init__(self, weight):
        self.weight = weight

    def named_parameters(self):
        yield "base.weight", self.weight


def test_unpartitioned():
    weight = torch.nn.Parameter(torch.zeros(100))
    stats = parameter_stats(FakeModel(weight))
    row, = stats["parameter_details"]

    assert stats["logical_parameters"] == 100
    assert stats["local_elements"] == 100
    assert stats["shard_elements"] == 0
    assert stats["partitioned_tensors"] == 0
    assert row["partition_state"] == "unmanaged_parameter"


def test_empty_view():
    _check_view_and_shard(0, "NOT_AVAILABLE")


def test_gathered_view():
    _check_view_and_shard(99, "AVAILABLE")


def _check_view_and_shard(view_size, status):
    weight = torch.nn.Parameter(torch.zeros(view_size))
    weight.ds_id = 1
    weight.ds_status = status
    weight.ds_numel = 99
    weight.ds_numel_aligned = 100
    weight.ds_tensor = torch.zeros(25, dtype=torch.bfloat16)
    weight.ds_tensor.ds_numel = 25
    weight.ds_tensor.status = "AVAILABLE"

    stats = parameter_stats(FakeModel(weight))
    row, = stats["parameter_details"]

    assert stats["logical_parameters"] == 99
    assert stats["local_elements"] == view_size
    assert stats["local_parameter_bytes"] == view_size * 4
    assert stats["shard_elements"] == 25
    assert stats["shard_bytes"] == 50
    assert stats["partitioned_tensors"] == 1
    assert row["name"] == "base.weight"
    assert row["ds_status"] == status
    assert row["ds_padding_elements"] == 1
    assert row["shard_status"] == "AVAILABLE"
    assert row["shard_device"] == "cpu"
    assert row["shard_dtype"] == "torch.bfloat16"
    assert row["partition_state"] == "locally_partitioned"


def test_missing_shard():
    weight = torch.nn.Parameter(torch.zeros(0))
    weight.ds_id = 1
    weight.ds_numel = 100
    weight.ds_tensor = None

    stats = parameter_stats(FakeModel(weight))
    row, = stats["parameter_details"]

    assert stats["deepspeed_managed_tensors"] == 1
    assert stats["missing_shard_tensors"] == 1
    assert stats["partitioned_tensors"] == 0
    assert row["shard_elements"] is None
    assert row["shard_bytes"] is None
    assert row["partition_state"] == "missing_deepspeed_shard"


def test_unmanaged_packed_nf4():
    _check_packed_nf4(None, "quantized_without_shard_metadata")


def test_partitioned_packed_nf4():
    _check_packed_nf4(25, "locally_partitioned")


def test_full_size_packed_nf4():
    _check_packed_nf4(50, "full_size_deepspeed_managed")


def _check_packed_nf4(partition_elements, expected_state):
    weight = torch.nn.Parameter(torch.zeros(50, dtype=torch.uint8), requires_grad=False)
    weight.quant_state = SimpleNamespace(
        shape=torch.Size([10, 10]), quant_type="nf4", blocksize=64, nested=True,
    )
    weight.quant_storage = torch.uint8
    if partition_elements is not None:
        weight.ds_id = 1
        weight.ds_numel = 50
        weight.ds_tensor = torch.zeros(partition_elements, dtype=torch.uint8)
        weight.ds_tensor.ds_numel = partition_elements

    stats = parameter_stats(FakeModel(weight))
    row, = stats["parameter_details"]

    assert stats["logical_parameters"] == 100
    assert stats["quantized_tensors"] == 1
    assert stats["local_parameter_bytes"] == 50
    assert row["quant_type"] == "nf4"
    assert row["quant_shape"] == [10, 10]
    assert row["quant_blocksize"] == 64
    assert row["quant_nested"] is True
    assert row["quant_storage_dtype"] == "torch.uint8"
    assert row["shard_bytes"] == partition_elements
    assert row["partition_state"] == expected_state
    assert json.loads(json.dumps(row)) == row


def test_packed_deepspeed_shard_uses_partition_capacity():
    weight = torch.nn.Parameter(torch.zeros(0))
    weight.ds_id = 1
    weight.ds_numel = 100
    weight.ds_tensor = torch.zeros(50, dtype=torch.uint8)
    weight.ds_tensor.ds_numel = 100
    weight.ds_tensor.ds_quant_scale = torch.ones(1)

    stats = parameter_stats(FakeModel(weight))
    row, = stats["parameter_details"]

    assert row["shard_bytes"] == 50
    assert row["shard_has_zero_quant_scale"] is True
    assert row["shard_partition_elements"] == 100
    assert row["partition_state"] == "full_size_deepspeed_managed"
    assert stats["partitioned_tensors"] == 0


if __name__ == "__main__":
    test_unpartitioned()
    test_empty_view()
    test_gathered_view()
    test_missing_shard()
    test_unmanaged_packed_nf4()
    test_partitioned_packed_nf4()
    test_full_size_packed_nf4()
    test_packed_deepspeed_shard_uses_partition_capacity()
    print("PASS: All synthetic CPU parameter tests passed.")
