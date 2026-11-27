"""Verify the Transformers initialization behavior relevant to Issue #124.

This is a source-code regression check, NOT a GPU partitioning test.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from transformers import PreTrainedModel

from train_zero3 import CONTROL_MODEL_ID, MODEL_ID, load_config, load_model


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


@pytest.mark.parametrize("arm", ["bf16", "nf4"])
@pytest.mark.parametrize("model_id", [None, CONTROL_MODEL_ID])
def test_model_loader_uses_selected_checkpoint_and_rank_device(arm, model_id):
    config = load_config(arm)
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    calls = Mock()
    calls.from_pretrained.return_value = model

    with (
        patch("transformers.integrations.HfDeepSpeedConfig", calls.ds_config),
        patch("transformers.BitsAndBytesConfig", calls.quant_config),
        patch("transformers.AutoModelForCausalLM.from_pretrained", calls.from_pretrained),
    ):
        model_args = (model_id,) if model_id else ()
        loaded, ds_config = load_model(arm, config, *model_args)

    assert calls.mock_calls[0] == call.ds_config(config)
    kwargs = {"dtype": torch.bfloat16}
    if arm == "nf4":
        calls.quant_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["quantization_config"] = calls.quant_config.return_value
    else:
        calls.quant_config.assert_not_called()
    calls.from_pretrained.assert_called_once_with(model_id or MODEL_ID, **kwargs)
    assert loaded is model
    assert ds_config is calls.ds_config.return_value
    assert loaded.config.use_cache is False


if __name__ == "__main__":
    test_zero3_quantization_guard()
    test_model_loader_uses_selected_checkpoint_and_rank_device("nf4", None)
