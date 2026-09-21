#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REWARD_MODEL="${REWARD_MODEL:-}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/outputs/test_time_alignment/results.json}"
PYTHON="${PYTHON:-python}"

"$PYTHON" - "$REWARD_MODEL" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from config import CONFIG
from experiment import run_experiment

reward_model_override, output = sys.argv[1:]
reward_model = reward_model_override or CONFIG["reward_model_hf_repo"]
run_experiment(
    CONFIG,
    reward_model,
    CONFIG["base_model_names"],
    CONFIG["max_test_samples"],
    Path(output),
    quantized=True,
)
PY
