"""Build numerical summary and paired replay from saved experiment results."""

import json
from pathlib import Path
import sys
import numpy as np
from grid_ast.model import Config
from grid_ast.policies import PPOPolicy
from grid_ast.search import replay
from grid_ast.cli import dump


root = Path(sys.argv[1] if len(sys.argv)>1 else "results")
folder = root / "search_persistent"
runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*seed*.json"))]
summary = {"comparison": {}, "nominal_evaluation": json.loads((root/"evaluation_morning"/"summary.json").read_text())}
for shielded in (False, True):
    for method in ("mc", "mcts"):
        group = [r for r in runs if r["method"] == method and r["shielded"] == shielded]
        if not group:
            continue
        found = [r["best"]["log_probability"] for r in group if r["best"]]
        key = method + ("_shield" if shielded else "_raw")
        summary["comparison"][key] = {
            "runs": len(group), "runs_with_failure": len(found),
            "transitions_per_run": group[0]["budget"],
            "failed_rollouts_range": [min(r["failed_rollouts"] for r in group), max(r["failed_rollouts"] for r in group)],
            "unique_failure_prefixes_range": [min(r["unique_failure_prefixes"] for r in group), max(r["unique_failure_prefixes"] for r in group)],
            "best_log_probability": max(found) if found else None,
            "median_best_log_probability": float(np.median(found)) if found else None,
            "total_power_flows": sum(r["total_power_flows"] for r in group),
            "total_seconds": sum(r["elapsed_seconds"] for r in group)}
raw = [r for r in runs if r["method"] == "mcts" and not r["shielded"] and r["best"]]
if raw:
    run = max(raw, key=lambda r:r["best"]["log_probability"])
    config = Config.load(folder/"config.json")
    policy = PPOPolicy(root/"training_seed7"/"ppo.zip")
    best = run["best"]
    trace = {"source": f"search_persistent/mcts_raw_seed{run['seed']}.json",
             "sequence": best["sequence"], "log_probability": best["log_probability"],
             "raw": replay(config, policy, best["sequence"], False),
             "shield": replay(config, policy, best["sequence"], True)}
    dump(root/"failure_replay.json", trace)
    summary["representative_failure"] = {"seed": run["seed"], "length": len(best["sequence"]),
                                          "log_probability": best["log_probability"],
                                          "minimum_voltage": min(trace["raw"][-1]["voltages"]),
                                          "soc": trace["raw"][-1]["soc"],
                                          "shield_also_fails": trace["shield"][-1]["voltage_failure"]}
dump(root/"key_metrics.json", summary)
print(json.dumps(summary["comparison"], indent=2))
