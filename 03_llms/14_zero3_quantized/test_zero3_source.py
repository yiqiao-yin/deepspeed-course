"""Verify the Transformers initialization behavior relevant to Issue #124.

This is a source-code regression check, NOT a GPU partitioning test.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from transformers import PreTrainedModel

from train_zero3 import MODEL_ID, load_config, load_model


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


def test_nf4_loader_uses_rank_device_selection():
    config = load_config("nf4")
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    calls = Mock()
    calls.from_pretrained.return_value = model

    with (
        patch("transformers.integrations.HfDeepSpeedConfig", calls.ds_config),
        patch("transformers.BitsAndBytesConfig", calls.quant_config),
        patch("transformers.AutoModelForCausalLM.from_pretrained", calls.from_pretrained),
    ):
        loaded, ds_config = load_model("nf4", config)

    assert calls.mock_calls[0] == call.ds_config(config)
    calls.quant_config.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    calls.from_pretrained.assert_called_once_with(
        MODEL_ID, dtype=torch.bfloat16, quantization_config=calls.quant_config.return_value,
    )
    assert loaded is model
    assert ds_config is calls.ds_config.return_value
    assert loaded.config.use_cache is False


if __name__ == "__main__":
    test_zero3_quantization_guard()
    test_nf4_loader_uses_rank_device_selection()
