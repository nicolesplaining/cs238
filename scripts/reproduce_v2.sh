#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root in the installed Python environment.
# Downloads ~107 MiB to the ignored run directory; no API key is needed.
ROOT="${1:-runs/reproduction_v2}"
mkdir -p "$ROOT/data"
python -m pytest -q
python -m grid_ast.calibrate --source "$ROOT/data/opsd_15min.csv" --cache "$ROOT/data/residuals.npz" --config "$ROOT/empirical_config.json" --report "$ROOT/calibration.json"
python -m grid_ast.study protocol --root "$ROOT"
for REGIME in synthetic empirical; do
  python -m grid_ast.study train --root "$ROOT" --config "$ROOT/empirical_config.json" --regime "$REGIME"
  python -m grid_ast.study evaluate --root "$ROOT" --config "$ROOT/empirical_config.json" --cache "$ROOT/data/residuals.npz" --regime "$REGIME"
  python -m grid_ast.study search --root "$ROOT" --config "$ROOT/empirical_config.json" --regime "$REGIME"
done
python -m grid_ast.study baselines --root "$ROOT" --config "$ROOT/empirical_config.json" --cache "$ROOT/data/residuals.npz"
python -m grid_ast.study summarize --root "$ROOT"
python scripts/plot_v2.py --root "$ROOT" --config "$ROOT/empirical_config.json" --cache "$ROOT/data/residuals.npz"
python scripts/audit_v2.py "$ROOT"
