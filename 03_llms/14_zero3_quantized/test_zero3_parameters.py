"""CPU-only tests for ZeRO-3 parameter measurements."""

import json
from types import SimpleNamespace

import pytest
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
        weight.ds_zero_partition_world_size = 50 // partition_elements
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
    weight.ds_zero_partition_world_size = 1
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


@pytest.mark.parametrize(
    "size, aligned, capacity, group_size, expected",
    [
        (1, 2, 1, 2, "locally_partitioned"),
        (1, None, 1, 8, "locally_partitioned"),
        (1, 8, 1, None, "locally_partitioned"),
        (1, None, 1, None, "incomplete_deepspeed_shard_metadata"),
        (1, 1, 1, 1, "full_size_deepspeed_managed"),
        (5, 8, 2, 4, "locally_partitioned"),
        (5, 8, 4, 2, "unexpected_deepspeed_size"),
        (5, 8, 3, None, "unexpected_deepspeed_size"),
        (5, 4, 2, None, "unexpected_deepspeed_size"),
        (0, 0, 0, 4, "empty_deepspeed_parameter"),
        (0, 1, 0, None, "unexpected_deepspeed_size"),
    ],
)
def test_zero_shard_capacity(size, aligned, capacity, group_size, expected):
    weight = torch.nn.Parameter(torch.zeros(size))
    weight.ds_id = 1
    weight.ds_numel = size
    weight.ds_numel_aligned = aligned
    weight.ds_zero_partition_world_size = group_size
    weight.ds_tensor = torch.zeros(capacity)
    weight.ds_tensor.ds_numel = capacity

    row, = parameter_stats(FakeModel(weight))["parameter_details"]
    assert row["partition_state"] == expected


@pytest.mark.parametrize("managed", [False, True])
def test_nf4_missing_logical_shape(managed):
    class Params4bit(torch.nn.Parameter):
        pass

    class CustomWeight(Params4bit):
        pass

    weight = CustomWeight(torch.zeros(13, dtype=torch.uint8), requires_grad=False)
    weight.quant_type = "nf4"
    weight.bnb_quantized = True
    weight.quant_storage = torch.uint8
    weight.blocksize = 64
    weight.compress_statistics = True
    if managed:
        weight.ds_id = 1
        weight.ds_numel = 13
        weight.ds_zero_partition_world_size = 2
        weight.ds_tensor = torch.zeros(7, dtype=torch.uint8)
        weight.ds_tensor.ds_numel = 7

    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.register_parameter("adapter", torch.nn.Parameter(torch.zeros(5)))
    stats = parameter_stats(model)
    row = stats["parameter_details"][0]

    assert row["logical_elements"] is None
    assert stats["logical_parameters"] is None
    assert stats["frozen_parameters"] is None
    assert stats["trainable_parameters"] == stats["known_logical_parameters"] == 5
    assert stats["unknown_logical_parameter_tensors"] == 1
    assert row["quant_type"] == "nf4"
    assert row["quant_shape"] is None
    assert row["quant_blocksize"] == 64
    assert row["quant_storage_dtype"] == "torch.uint8"
    assert row["bnb_quantized"] is True
    assert row["quant_compress_statistics_configured"] is True
    assert row["quant_nested"] is None
    assert row["local_bytes"] == 13
    assert row["partition_state"] == (
        "locally_partitioned" if managed else "quantized_without_shard_metadata"
    )


def test_nf4_recovers_shape_from_shard_with_partial_parameter_metadata():
    weight = torch.nn.Parameter(torch.zeros(0), requires_grad=False)
    weight.ds_id = 1
    weight.ds_numel = 13
    weight.quant_state = SimpleNamespace(quant_type="nf4", nested=False)
    weight.ds_tensor = torch.zeros(7, dtype=torch.uint8)
    weight.ds_tensor.ds_numel = 7
    weight.ds_tensor.quant_state = SimpleNamespace(shape=(5, 5), blocksize=64)

    row, = parameter_stats(FakeModel(weight))["parameter_details"]
    assert row["logical_elements"] == 25
    assert row["quant_nested"] is False
    assert row["quant_blocksize"] == 64
    assert row["shard_bytes"] == 7
