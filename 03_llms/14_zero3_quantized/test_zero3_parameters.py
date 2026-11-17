"""CPU-only tests for ZeRO-3 parameter measurements."""

import torch

from train_zero3 import parameter_stats, parameter_partition_details


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


if __name__ == "__main__":
    test_unpartitioned()
    test_partitioned()
    test_parameter_details()

    print("\nALL PARAMETER TESTS PASSED")
    print("NOTE: These are synthetic CPU tests, not GPU measurements.")