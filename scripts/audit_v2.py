"""Audit all expected policies, evaluation counts, likelihoods and failure replays."""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from grid_ast.model import Config, make_disturbances
from grid_ast.policies import PPOPolicy
from grid_ast.search import replay
from grid_ast.study import SEEDS


def audit(root):
    root=Path(root);counts={'trained_policies':0,'evaluation_tables':0,'search_runs':0,'failure_replays':0}
    expected_days=json.loads((root/'historical_days.json').read_text())['included']
    for regime in ('synthetic','empirical'):
        for seed in SEEDS:
            folder=root/'training'/regime/f'seed{seed}'
            metadata=json.loads((folder/'training.json').read_text())
            assert metadata['seed']==seed and metadata['actual_steps']==65536,folder
            config=Config.load(folder/'config.json')
            assert (config.disturbance_model is not None)==(regime=='empirical'),folder
            if config.disturbance_model:
                assert config.disturbance_model['provenance']['fit_years']==[2017,2018]
            counts['trained_policies']+=1
            for name in ['generated_synthetic','generated_empirical','historical_soc0.50','historical_soc0.15']+(['historical_shield_soc0.50'] if regime=='empirical' else []):
                path=root/'evaluation'/regime/f'seed{seed}'/f'{name}.json'
                report=json.loads(path.read_text());table=pd.read_csv(path.with_suffix('.csv'))
                assert len(table)==report['episodes'],path
                assert table.voltage_failure.sum()==report['failed_episodes'],path
                assert np.isclose(table.cost.mean(),report['mean_cost']),path
                assert table.solver_failure.sum()==0,path
                assert (table.steps==32).all(),path
                if name.startswith('historical'):
                    assert table.date.tolist()==expected_days,path
                else:
                    assert table.seed.tolist()==list(range(70000,70200)),path
                counts['evaluation_tables']+=1
            policy=PPOPolicy(folder/'ppo.zip',preserve_training_timebase=True)
            digest=hashlib.sha256((folder/'ppo.zip').read_bytes()).hexdigest()
            search_folder=root/'search'/regime/f'seed{seed}'
            config=Config.load(search_folder/'config.json')
            noise=make_disturbances(config)
            for name in ['mc_raw','mcts_raw']+(['mcts_shield'] if regime=='empirical' else []):
                path=search_folder/f'{name}.json';r=json.loads(path.read_text())
                assert r['training_seed']==seed and r['model_sha256']==digest,path
                assert r['transitions']==r['budget']==6000,path
                assert r['solver_failures']==0,path
                assert r['total_power_flows']==r['transitions']+r['reset_power_flows']+r['shield_predictions'],path
                counts['search_runs']+=1
                if r['best'] is not None:
                    best=r['best']
                    assert np.isclose(noise.log_probability(best['sequence'],config.start_step),best['log_probability']),path
                    rows=replay(config,policy,best['sequence'],r['shielded'])
                    assert rows[-1]['voltage_failure'],path
                    assert not any(x['voltage_failure'] for x in rows[:-1]),path
                    np.testing.assert_allclose([x['voltages'] for x in rows],[x['voltages'] for x in best['trajectory']],atol=1e-10)
                    counts['failure_replays']+=1
    print(json.dumps(counts,indent=2))
    return counts


if __name__=='__main__': audit(sys.argv[1] if len(sys.argv)>1 else 'results/v2')
