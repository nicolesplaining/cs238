"""Validate saved counts, probabilities and exact failure replays.

Usage: python scripts/audit_results.py results
"""

import csv
import json
from pathlib import Path
import sys
import numpy as np

from grid_ast.model import Config, Disturbances
from grid_ast.policies import PPOPolicy
from grid_ast.search import replay


def audit(root):
    root = Path(root)
    policy = PPOPolicy(root / "training_seed7" / "ppo.zip")
    counts = {"search_runs": 0, "failure_replays": 0, "evaluation_tables": 0}
    for path in sorted(root.glob("search*/*seed*.json")):
        run = json.loads(path.read_text())
        config = Config.load(path.parent / "config.json")
        assert run["transitions"] == run["budget"], path
        assert run["solver_failures"] == 0, path
        assert run["total_power_flows"] == run["transitions"]+run["reset_power_flows"]+run["shield_predictions"], path
        counts["search_runs"] += 1
        if run["best"] is None:
            assert run["failed_rollouts"] == 0, path
            continue
        best = run["best"]
        assert np.isclose(best["log_probability"], Disturbances(config).log_probability(best["sequence"])), path
        assert np.isclose(np.exp(best["log_probability"]), best["prefix_probability"], rtol=1e-10, atol=0), path
        rows = replay(config, policy, best["sequence"], run["shielded"])
        assert rows[-1]["voltage_failure"], path
        assert not any(r["voltage_failure"] for r in rows[:-1]), path
        assert len(rows) == len(best["sequence"]), path
        np.testing.assert_allclose([r["voltages"] for r in rows], [r["voltages"] for r in best["trajectory"]], atol=1e-10)
        assert run["history"][-1]["failures"] == run["failed_rollouts"], path
        counts["failure_replays"] += 1
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text())
        for name, result in summary.items():
            table = path.parent / f"{name}_episodes.csv"
            if not table.exists():
                continue
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == result["episodes"], table
            assert sum(int(r["voltage_failure"]) for r in rows) == result["failed_episodes"], table
            assert sum(int(r["solver_failure"]) for r in rows) == 0, table
            assert np.isclose(np.mean([float(r["cost"]) for r in rows]), result["mean_cost"]), table
            counts["evaluation_tables"] += 1
    print(json.dumps(counts, indent=2))
    return counts


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv)>1 else "results")
