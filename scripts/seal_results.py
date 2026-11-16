"""Record source, model and result checksums after completing an experiment run."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from grid_ast.train import versions


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(sys.argv[1] if len(sys.argv)>1 else "results")
repo = Path(__file__).resolve().parents[1]
model = root / "training_seed7" / "ppo.zip"
# Keep machine-generated records compact and normalize CSV line endings.
# Parsing round-trips preserve all numerical results and replay information.
for path in root.rglob("*.csv"):
    path.write_text(path.read_text())
for path in root.glob("search*/*seed*.json"):
    path.write_text(json.dumps(json.loads(path.read_text()), separators=(",", ":"), allow_nan=False)+"\n")
manifest = {
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    "model_sha256": sha256(model),
    "versions": versions(),
    "source_sha256": {str(p.relative_to(repo)):sha256(p) for folder in ("grid_ast", "tests", "scripts") for p in sorted((repo/folder).rglob("*.py"))},
    "result_sha256": {str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"},
    "scope": "Synthetic six-bus benchmark; one PPO training seed. No field-data calibration or safety guarantee.",
    "search_protocol": {
        "search": "Exploratory nominal-rollout MCTS and MC, seeds 1000:1004, persistence=0.",
        "search_high_noise": "Exploratory load_sigma=0.45, nominal-rollout MCTS and MC, seeds 1000:1004, persistence=0.",
        "search_persistent": "Final held-out search seeds 6000:6004, MCTS rollout persistence=0.25, MC unchanged.",
        "tuning": "Persistence variant selected after nominal-rollout failures to discover rare paths; pilot seed 5100 used 2000 transitions. Pilot is excluded from final comparison.",
        "budget": "6000 actual environment transitions per run; reset solves and shield forecast solves reported separately. Prefix replay is charged."
    },
    "evaluation_protocol": "Morning: 200 paired episodes, seeds 20000:20199. Daily: 100 paired episodes, seeds 40000:40099. Sensitivity: 100 paired episodes per scale, seeds 30000:30099.",
    "likelihood_caution": "Exact latent disturbance-prefix masses under the synthetic model, conditional on the fixed initial state. Best found is not globally optimal and is not an estimate of total failure probability."
}
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"result_files": len(manifest["result_sha256"]), "source_commit": manifest["source_commit"]}))
