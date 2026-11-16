#!/usr/bin/env bash
set -euo pipefail
# Activate a Python 3.11+ environment and install this package with pip install -e '.[test]'.
# Run from the repository root. Writes to runs/reproduction, preserving committed results.
ROOT="${1:-runs/reproduction}"
python -m pytest -q
python -m grid_ast.cli train --steps 32768 --seed 7 --out "$ROOT/training_seed7"
MODEL="$ROOT/training_seed7/ppo.zip"
python -m grid_ast.cli evaluate --model "$MODEL" --config configs/morning_stress.json --out "$ROOT/evaluation_morning" --episodes 200 --seed 20000
python -m grid_ast.cli evaluate --model "$MODEL" --config configs/daily.json --out "$ROOT/evaluation_daily" --episodes 100 --seed 40000
python -m grid_ast.cli benchmark --model "$MODEL" --config configs/morning_stress.json --out "$ROOT/search" --budget 6000 --seeds 1000,1001,1002,1003,1004 --persistent-rollout-probability 0
python -m grid_ast.cli benchmark --model "$MODEL" --config configs/morning_stress.json --out "$ROOT/search_persistent" --budget 6000 --seeds 6000,6001,6002,6003,6004 --persistent-rollout-probability .25
python -m grid_ast.cli sensitivity --model "$MODEL" --config configs/morning_stress.json --out "$ROOT/sensitivity" --episodes 100 --seed 30000 --load-scales .12,.30,.45
python -m grid_ast.cli benchmark --model "$MODEL" --config "$ROOT/sensitivity/load_sigma_0.45/config.json" --out "$ROOT/search_high_noise" --budget 6000 --seeds 1000,1001,1002,1003,1004 --persistent-rollout-probability 0
python scripts/summarize_results.py "$ROOT"
python -m grid_ast.cli plot --results "$ROOT"
python scripts/audit_results.py "$ROOT"
