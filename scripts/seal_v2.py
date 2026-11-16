"""Normalize result serialization and record v2 source/result provenance."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from grid_ast.train import versions
from grid_ast.calibrate import SOURCE_URL, SOURCE_SHA256


root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/v2')
repo=Path(__file__).resolve().parents[1]
for path in root.rglob('*.csv'):
    if 'data' not in path.relative_to(root).parts: path.write_text(path.read_text())
for path in (root/'search').glob('*/*/*_*.json'):
    value=json.loads(path.read_text())
    value['preserve_training_timebase']=True
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
# Early worker processes retained an unused IID Wilson field; historical
# uncertainty is instead computed with seed and calendar-block resampling.
for path in list((root/'evaluation').glob('*/*/historical*.json'))+list((root/'baselines').glob('*historical*.json')):
    value=json.loads(path.read_text());value.pop('failure_rate_wilson95',None)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_paths=[p for folder in ('grid_ast','tests','scripts','configs') for p in (repo/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')]
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
          'source_url':SOURCE_URL,'source_data_sha256':SOURCE_SHA256,'versions':versions(),
          'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sorted(source_paths)},
          'result_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json' and 'data' not in p.relative_to(root).parts},
          'scope':'Five seeds per training regime, fixed hyperparameters, no held-out tuning. Historical national errors are transferred onto toy-feeder forecasts.',
          'evaluation_clock':'Preserved training-time elapsed/96 feature. Supersedes v1 shorter-window clock normalization.',
          'uncertainty':'Training-seed variation and seven-day blocks of paired historical days. All-zero bootstrap failure intervals are not zero-risk evidence.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'result_files':len(manifest['result_sha256']),'source_commit':manifest['source_commit']}))
