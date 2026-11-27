#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Match the lab lock's Torch release and Transformers source while using CPU
# wheels. The GPU experiment still runs from the committed lab lock.
exec uv run --no-project --python 3.12 \
    --index https://download.pytorch.org/whl/cpu \
    --with 'torch==2.11.0+cpu' --with 'transformers==5.16.1' --with pytest \
    python -m pytest -q \
    test_zero3_parameters.py test_zero3_reporting.py test_zero3_launcher.py test_zero3_source.py
