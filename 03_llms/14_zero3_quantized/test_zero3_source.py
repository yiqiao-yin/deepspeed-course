"""Verify the Transformers initialization behavior relevant to Issue #124.

This is a source-code regression check, NOT a GPU partitioning test.
"""

import inspect

from transformers import PreTrainedModel


def test_zero3_quantization_guard():
    """Verify that quantization affects the ZeRO-3 initialization path."""

    source = inspect.getsource(PreTrainedModel.get_init_context)

    assert "is_deepspeed_zero3_enabled()" in source

    assert "not is_quantized and not _is_ds_init_called" in source

    assert "deepspeed.zero.Init" in source

    assert "elif is_quantized:" in source

    assert 'torch.device("meta")' in source

    print("PASS: ZeRO-3 quantization initialization guards found.")
    print("NOTE: No GPU partitioning measurements have been performed.")


if __name__ == "__main__":
    test_zero3_quantization_guard()