"""Matched multi-seed experiment with generated and held-out historical errors."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .model import Config
from .policies import PPOPolicy, IdlePolicy, RulePolicy
from .evaluate import evaluate, write_csv
from .historical import historical_days, evaluate_history, seed_day_summary
from .search import search
from .cli import dump


SEEDS = [7,17,27,37,47]


def protocol(root):
    value={"training_seeds":SEEDS,"steps_per_policy":65536,"training_regimes":["synthetic","empirical"],
           "policy_selection":"Every listed training seed is reported. Final checkpoint only; no best-seed selection.",
           "evaluation_clock":"Preserve the trained elapsed/96 observation feature. The 32-step evaluation horizon is only a reporting cutoff.",
           "generated_evaluation":{"episodes":200,"seed":70000,"start_step":24,"horizon":32,"initial_soc":.5,
                                   "regimes":["synthetic","empirical"]},
           "historical_evaluation":{"year":2019,"start_step":24,"horizon":32,"initial_soc":[.5,.15],
                                    "initial_conditions":"Observed initial errors; 15% SoC is below the training initialization range of 25%-75%.",
                                    "days":"Every complete local-calendar day; gaps and DST days excluded and listed.",
                                    "shield":"Empirical-trained policies, initial_soc=0.5 only."},
           "search":{"budget":6000,"seed":8100,"methods":["mc","mcts"],"persistent_rollout_probability":.25,
                     "regime":"Each policy evaluated under its training disturbance regime.",
                     "shield":"Additional matched MCTS runs for empirical-trained policies."},
           "uncertainty":"Policy seeds and paired seven-day calendar blocks resampled separately; never pool seed-day pairs as independent episodes.",
           "historical_scope":"Unquantized national residual traces transferred to toy ANM6 forecasts; not historical operation of a real feeder."}
    dump(Path(root)/"protocol.json",value)
    return value


def save_evaluation(folder,name,result):
    folder=Path(folder)
    write_csv(folder/f"{name}.csv",result["rows"])
    dump(folder/f"{name}.json",{k:v for k,v in result.items() if k!='rows'})


def run_evaluations(root, empirical_config, cache, regime, seeds):
    root=Path(root)
    days,excluded=historical_days(cache)
    dump(root/"historical_days.json",{"year":2019,"included":[d['date'] for d in days],"excluded":excluded})
    c=replace(empirical_config,start_step=24,horizon=32,initial_soc=.5,random_start=False)
    for seed in seeds:
        folder=root/"evaluation"/regime/f"seed{seed}"
        policy=PPOPolicy(root/"training"/regime/f"seed{seed}"/"ppo.zip",preserve_training_timebase=True)
        for name,config in (("synthetic",replace(c,disturbance_model=None)),("empirical",c)):
            path=folder/f"generated_{name}.json"
            if not path.exists():
                result=evaluate(config,policy,episodes=200,seed=70000)
                save_evaluation(folder,f"generated_{name}",result)
                print(regime,seed,'generated',name,result['failed_episodes'],result['mean_cost'],flush=True)
        for soc in (.5,.15):
            name=f"historical_soc{soc:.2f}"
            if not (folder/f"{name}.json").exists():
                result=evaluate_history(replace(c,initial_soc=soc),policy,days)
                save_evaluation(folder,name,result)
                print(regime,seed,name,result['failed_episodes'],result['mean_cost'],flush=True)
        if regime=='empirical' and not (folder/'historical_shield_soc0.50.json').exists():
            result=evaluate_history(c,policy,days,True)
            save_evaluation(folder,'historical_shield_soc0.50',result)
            print(regime,seed,'historical_shield',result['failed_episodes'],result['mean_cost'],flush=True)


def run_baselines(root,empirical_config,cache):
    folder=Path(root)/"baselines"
    c=replace(empirical_config,start_step=24,horizon=32,initial_soc=.5,random_start=False)
    days,_=historical_days(cache)
    for name,policy in (("idle",IdlePolicy()),("rule",RulePolicy())):
        for regime,config in (("synthetic",replace(c,disturbance_model=None)),("empirical",c)):
            key=f"{name}_generated_{regime}"
            if not (folder/f"{key}.json").exists(): save_evaluation(folder,key,evaluate(config,policy,200,70000))
        for soc in (.5,.15):
            key=f"{name}_historical_soc{soc:.2f}"
            if not (folder/f"{key}.json").exists(): save_evaluation(folder,key,evaluate_history(replace(c,initial_soc=soc),policy,days))


def run_searches(root,empirical_config,regime,seeds):
    root=Path(root)
    c=replace(empirical_config,start_step=24,horizon=32,initial_soc=.5,random_start=False)
    if regime=='synthetic': c=replace(c,disturbance_model=None)
    for seed in seeds:
        policy=PPOPolicy(root/'training'/regime/f'seed{seed}'/'ppo.zip',preserve_training_timebase=True)
        folder=root/'search'/regime/f'seed{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        c.save(folder/'config.json')
        modes=[('mc',False),('mcts',False)]+([('mcts',True)] if regime=='empirical' else [])
        for method,shield in modes:
            target=folder/f'{method}_{"shield" if shield else "raw"}.json'
            if target.exists(): continue
            result=search(c,policy,method,6000,8100,shield,persistent_rollout_probability=.25)
            result['training_seed']=seed
            result['training_regime']=regime
            result['preserve_training_timebase']=True
            result['model_sha256']=hashlib.sha256((root/'training'/regime/f'seed{seed}'/'ppo.zip').read_bytes()).hexdigest()
            target.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
            print(regime,seed,method,shield,result['failed_rollouts'],None if result['best'] is None else result['best']['log_probability'],flush=True)


def summarize(root):
    root=Path(root)
    records=[]
    for path in sorted((root/'evaluation').glob('*/*/*.json')):
        regime,seed=path.parts[-3],int(path.parts[-2].removeprefix('seed'))
        x=json.loads(path.read_text())
        records.append({'regime':regime,'training_seed':seed,'evaluation':path.stem,
                        'episodes':x['episodes'],'failed_episodes':x['failed_episodes'],
                        'failure_rate':x['failure_rate'],'mean_cost':x['mean_cost'],'solver_failures':x['solver_failures']})
    write_csv(root/'per_seed_metrics.csv',records)
    frame=pd.DataFrame(records)
    summaries={}
    for (regime,evaluation),group in frame.groupby(['regime','evaluation']):
        group=group.sort_values('training_seed')
        if len(group)!=len(SEEDS): raise ValueError(f'Incomplete seed group: {regime}/{evaluation}')
        key=f'{regime}/{evaluation}'
        if evaluation.startswith('historical'):
            tables=[pd.read_csv(root/'evaluation'/regime/f'seed{s}'/f'{evaluation}.csv') for s in group.training_seed]
            dates=tables[0].date.tolist()
            if not all(t.date.tolist()==dates for t in tables): raise ValueError('Historical days must match across seeds.')
            summaries[key]=seed_day_summary([t.cost.to_numpy() for t in tables],[t.voltage_failure.to_numpy() for t in tables],dates)
            summaries[key]['mean_end_soc']=float(np.mean([t.end_soc.mean() for t in tables]))
            summaries[key]['per_seed_mean_end_soc']=[float(t.end_soc.mean()) for t in tables]
            summaries[key]['shield_interventions']=int(sum(t.shield_interventions.sum() for t in tables))
        else:
            summaries[key]={'policy_seeds':len(group),'mean_cost':float(group.mean_cost.mean()),
                            'cost_seed_std':float(group.mean_cost.std(ddof=1)),
                            'per_seed_mean_cost':group.mean_cost.tolist(),'per_seed_failure_rate':group.failure_rate.tolist(),
                            'failure_rate_seed_range':[float(group.failure_rate.min()),float(group.failure_rate.max())]}
    # Paired training-regime cost difference, using identical dates and seed labels.
    for soc in (.5,.15):
        name=f'historical_soc{soc:.2f}'
        a=[pd.read_csv(root/'evaluation'/'synthetic'/f'seed{s}'/f'{name}.csv') for s in SEEDS]
        b=[pd.read_csv(root/'evaluation'/'empirical'/f'seed{s}'/f'{name}.csv') for s in SEEDS]
        summaries[f'paired_empirical_minus_synthetic/{name}']=seed_day_summary(
            [y.cost.to_numpy()-x.cost.to_numpy() for x,y in zip(a,b)],
            [y.voltage_failure.to_numpy()-x.voltage_failure.to_numpy() for x,y in zip(a,b)],a[0].date.tolist())
    dump(root/'summary.json',summaries)
    search_rows=[]
    for path in sorted((root/'search').glob('*/*/*_*.json')):
        r=json.loads(path.read_text());best=r['best']
        search_rows.append({'training_regime':r['training_regime'],'training_seed':r['training_seed'],'method':r['method'],
                            'shielded':r['shielded'],'transitions':r['transitions'],'failed_rollouts':r['failed_rollouts'],
                            'unique_failure_prefixes':r['unique_failure_prefixes'],'best_log_probability':None if best is None else best['log_probability'],
                            'total_power_flows':r['total_power_flows'],'solver_failures':r['solver_failures']})
    write_csv(root/'search_summary.csv',search_rows)
    print(json.dumps(summaries,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['protocol','train','evaluate','baselines','search','summarize'])
    parser.add_argument('--root',default='results/v2')
    parser.add_argument('--config',default='configs/empirical_opsd.json')
    parser.add_argument('--cache',default='../data/opsd_residuals.npz')
    parser.add_argument('--regime',choices=['synthetic','empirical'],default='empirical')
    parser.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    args=parser.parse_args();seeds=[int(s) for s in args.seeds.split(',')]
    if args.command=='protocol': protocol(args.root);return
    if args.command=='summarize': summarize(args.root);return
    config=Config.load(args.config)
    if args.command=='train':
        from .train import train
        if args.regime=='synthetic': config=replace(config,disturbance_model=None)
        for seed in seeds: train(config,Path(args.root)/'training'/args.regime/f'seed{seed}',65536,seed)
    elif args.command=='evaluate': run_evaluations(args.root,config,args.cache,args.regime,seeds)
    elif args.command=='search': run_searches(args.root,config,args.regime,seeds)
    elif args.command=='baselines': run_baselines(args.root,config,args.cache)


if __name__=='__main__': main()
