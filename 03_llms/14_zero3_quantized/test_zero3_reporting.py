"""CPU-only tests for ZeRO-3 JSON reporting.

Uses synthetic data. Does not perform GPU measurements.
"""

import json
import tempfile
from pathlib import Path

from train_zero3 import save_rank_results


def test_rank_reporting():
    """Verify that ranks receive separate, valid JSON files."""

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)

        for rank in (0, 1):
            stats = {
                "arm": "bf16",
                "rank": rank,
                "world_size": 2,
                "zero_stage": 3,
                "logical_parameters": 1000,
                "local_elements": 500,
                "measurement_stage": "synthetic_test",
            }

            save_rank_results(stats, output_dir=output)

        for rank in (0, 1):
            filename = output / f"bf16_rank{rank}.json"

            assert filename.exists(), (
                f"Missing JSON file for rank {rank}"
            )

            with open(filename, encoding="utf-8") as file:
                result = json.load(file)

            assert result["rank"] == rank
            assert result["world_size"] == 2
            assert result["zero_stage"] == 3
            assert result["measurement_stage"] == "synthetic_test"

        print("PASS: Both GPU ranks produce separate valid JSON files.")
        print("NOTE: Synthetic test only. No GPU measurements performed.")


if __name__ == "__main__":
    test_rank_reporting()