#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REWARD_MODEL="${REWARD_MODEL:-$SCRIPT_DIR/outputs/reward_model_hh_rlhf}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/outputs/test_time_alignment/results.json}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "$REWARD_MODEL/config.json" ]]; then
    print -u2 "Reward model checkpoint not found: $REWARD_MODEL"
    print -u2 "Copy or download the checkpoint into outputs/reward_model_hh_rlhf before running."
    exit 1
fi

"$PYTHON" - "$REWARD_MODEL" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from config import CONFIG
from experiment import run_experiment

reward_model, output = sys.argv[1:]
run_experiment(
    CONFIG,
    reward_model,
    CONFIG["base_model_names"],
    CONFIG["max_test_samples"],
    Path(output),
    quantized=True,
)
PY
