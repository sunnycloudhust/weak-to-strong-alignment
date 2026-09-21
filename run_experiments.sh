#!/bin/zsh
set -euo pipefail

REWARD_MODEL="outputs/reward_model_hh_rlhf"
MAX_PROMPTS=50
OUTPUT="outputs/test_time_alignment/results.json"

conda run -n research python - "$REWARD_MODEL" "$MAX_PROMPTS" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from config import CONFIG
from experiment import run_experiment

reward_model, max_prompts, output = sys.argv[1:]
run_experiment(
    CONFIG,
    reward_model,
    CONFIG["base_model_names"],
    int(max_prompts),
    Path(output),
    quantized=False,
)
PY
