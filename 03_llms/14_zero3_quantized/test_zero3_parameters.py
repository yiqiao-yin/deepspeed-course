"""CPU-only tests for ZeRO-3 parameter measurements."""

import torch

from train_zero3 import ( 
    parameter_stats,
    parameter_partition_details,
    classify_partition_state,
)

class FakeModel:
    """Minimal model exposing named parameters."""

    def __init__(self, parameters):
        self.parameters_dict = parameters

    def named_parameters(self):
        return iter(self.parameters_dict.items())


def test_unpartitioned():
    """An ordinary parameter must not be classified as sharded."""

    weight = torch.nn.Parameter(torch.zeros(100))

    model = FakeModel({"base.weight": weight})
    stats = parameter_stats(model)

    assert stats["logical_parameters"] == 100
    assert stats["local_elements"] == 100
    assert stats["partitioned_tensors"] == 0

    print("PASS: Unpartitioned parameter correctly identified.")


def test_partitioned():
    """A ZeRO-managed local shard must be identified."""

    weight = torch.nn.Parameter(torch.zeros(25))
    weight.ds_id = 1
    weight.ds_numel = 100

    model = FakeModel({"base.weight": weight})
    stats = parameter_stats(model)

    assert stats["logical_parameters"] == 100
    assert stats["local_elements"] == 25
    assert stats["deepspeed_managed_tensors"] == 1
    assert stats["partitioned_tensors"] == 1

    print("PASS: Simulated ZeRO-3 shard correctly identified.")


def test_parameter_details():
    """Parameter diagnostics must preserve identifying information."""

    weight = torch.nn.Parameter(torch.zeros(25))
    weight.ds_id = 1
    weight.ds_numel = 100

    model = FakeModel({"base.weight": weight})

    details = parameter_partition_details(model)

    assert len(details) == 1
    assert details[0]["name"] == "base.weight"
    assert details[0]["logical_elements"] == 100
    assert details[0]["local_elements"] == 25
    assert details[0]["deepspeed_managed"] is True

    print("PASS: Parameter-level diagnostics correctly recorded.")

def test_partition_classification():
    """Ensure NF4 compression is not mistaken for ZeRO-3 sharding."""

    cases = [
        (
            {
                "has_shard_metadata": True,
                "local_elements": 25,
                "ds_numel": 100,
                "is_quantized": False,
            },
            "locally_partitioned",
        ),
        (
            {
                "has_shard_metadata": True,
                "local_elements": 100,
                "ds_numel": 100,
                "is_quantized": False,
            },
            "full_size_deepspeed_managed",
        ),
        (
            {
                "has_shard_metadata": False,
                "local_elements": 50,
                "ds_numel": None,
                "is_quantized": True,
            },
            "quantized_without_shard_metadata",
        ),
        (
            {
                "has_shard_metadata": False,
                "local_elements": 100,
                "ds_numel": None,
                "is_quantized": False,
            },
            "unmanaged_parameter",
        ),
    ]

    for row, expected in cases:
        actual = classify_partition_state(row)

        assert actual == expected, (
            f"Expected {expected}, got {actual}"
        )

    print("PASS: Partition-state classification.")
    print("PASS: NF4 compression is not mistaken for ZeRO-3 sharding.")

if __name__ == "__main__":
    test_unpartitioned()
    test_partitioned()
    test_parameter_details()
    test_partition_classification()

    print("\nALL PARAMETER TESTS PASSED")
    print("NOTE: These are synthetic CPU tests, not GPU measurements.")