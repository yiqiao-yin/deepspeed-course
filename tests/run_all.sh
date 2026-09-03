#!/bin/bash
# Run every regression test. Requires only `uv` — each test declares its own
# dependencies inline (PEP 723), so uv provisions them automatically.
#
#     ./tests/run_all.sh
#
# No GPU required: these tests cover the pure logic of the fixes, not the
# distributed training itself.

set -u

cd "$(dirname "$0")/.." || exit 1

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

FAILED=0
TESTS=(
    tests/test_ds_configs.py
    tests/test_stock_leakage.py
    tests/test_attention_layers.py
    tests/test_ts_forecasting.py
    tests/test_grpo_rewards.py
    tests/test_preference_losses.py
    tests/test_reward_model.py
    tests/test_video_frames.py
    tests/test_runpod_ctl.py
    tests/test_docs_style.py
    tests/test_token_compression.py
    tests/test_star_memory.py
    tests/test_video_eval.py
    tests/test_tmrope.py
    tests/test_duplex.py
    tests/test_omni_eval.py
    tests/test_modern_cifar.py
    tests/test_ocr_metrics.py
    tests/test_ranking_losses.py
    tests/test_groupwise_ranking.py
    tests/test_glm53_arch.py
    tests/test_qwen38_arch.py
    tests/test_clawdeck_manifest.py
)

for test in "${TESTS[@]}"; do
    if ! uv run "$test"; then
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "========================================================================"
if [ "$FAILED" -eq 0 ]; then
    echo "ALL SUITES PASSED"
else
    echo "$FAILED SUITE(S) FAILED"
fi
echo "========================================================================"

exit "$FAILED"
