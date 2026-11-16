"""Plot model diagnostics and seed-level outcomes from the second experiment."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid_ast.model import Config
from grid_ast.empirical import MarkovDisturbances
from grid_ast.calibrate import encode
from grid_ast.study import SEEDS


def plots(root,config_path,cache):
    root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    colors={'synthetic':'#8a3d42','empirical':'#315d8c'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    summary=json.loads((root/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for j,soc in enumerate((.5,.15)):
        for i,regime in enumerate(('synthetic','empirical')):
            x=summary[f'{regime}/historical_soc{soc:.2f}']
            costs=x['per_seed_mean_cost'];offsets=np.linspace(-.12,.12,len(costs))
            axes[j].scatter(i+offsets,costs,color=colors[regime],s=45)
            axes[j].hlines(np.mean(costs),i-.2,i+.2,color=colors[regime],linewidth=2)
            for k,cost in enumerate(costs): axes[j].annotate(str(SEEDS[k]),(i+offsets[k],cost),xytext=(3,4),textcoords='offset points',fontsize=8)
        axes[j].set(xticks=[0,1],xticklabels=['Synthetic training','Data-calibrated training'],
                    ylabel='Mean 8-hour operating cost ($)',title=f'2019 traces, initial SoC {soc:.0%}')
        baseline=json.loads((root/'baselines'/f'rule_historical_soc{soc:.2f}.json').read_text())
        axes[j].axhline(baseline['mean_cost'],color='#555555',linestyle='--',label='Rule baseline')
        axes[j].legend(frameon=False,fontsize=8)
    fig.savefig(out/'seed_costs.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained',sharey=True)
    for ax,regime in zip(axes,('synthetic','empirical')):
        for seed in SEEDS:
            frame=pd.read_csv(root/'training'/regime/f'seed{seed}'/'progress.csv')
            ax.plot(frame['time/total_timesteps'],frame['rollout/ep_rew_mean'],alpha=.65,label=f'seed {seed}')
        ax.set(xlabel='Training transitions',title=regime.capitalize()+' disturbance model')
        ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Mean recent episode return')
    fig.savefig(out/'training_seeds.png');plt.close(fig)

    noise=MarkovDisturbances(Config.load(config_path).disturbance_model)
    with np.load(cache,allow_pickle=False) as data:
        values,valid,years,steps=[data[k] for k in ('values','valid','years','steps')]
    test=(years==2019)&valid;observed=values[test]
    rng=np.random.default_rng(20261116);state=np.zeros(2);simulated=[]
    for step in steps[test]:
        action=noise.sample(rng,state,int(step));state=noise.evolve(state,action,int(step));simulated.append(state.copy())
    simulated=np.array(simulated)
    pairs=np.flatnonzero(test[1:]&test[:-1])+1
    ids=encode(np.nan_to_num(values),noise.edges)
    counts=np.array(json.loads((root/'calibration.json').read_text())['transition_counts'])
    marginal=counts.sum(axis=1)+.5;marginal/=marginal.sum(axis=1,keepdims=True)
    period=steps[pairs]//noise.period_steps
    markov_logp=np.log(noise.transitions[period,ids[pairs-1],ids[pairs]]).mean()
    iid_logp=np.log(marginal[period,ids[pairs]]).mean()
    diagnostics={'simulation_seed':20261116,'simulated_steps':len(simulated),
                 'heldout_markov_mean_log_probability':float(markov_logp),'heldout_time_of_day_iid_mean_log_probability':float(iid_logp),
                 'heldout_log_score_gain_nats_per_transition':float(markov_logp-iid_logp),
                 'heldout_mean':observed.mean(axis=0).tolist(),'simulated_mean':simulated.mean(axis=0).tolist(),
                 'heldout_std':observed.std(axis=0).tolist(),'simulated_std':simulated.std(axis=0).tolist(),
                 'simulated_lag1_correlation':[float(np.corrcoef(simulated[:-1,j],simulated[1:,j])[0,1]) for j in range(2)],
                 'caution':'The fitted model compresses tails and does not model the 2019 load-bias shift. Historical replay uses the full unquantized errors.'}
    (root/'calibration_validation.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for j,(ax,label) in enumerate(zip(axes,('Load error (% of forecast)','Solar error (% of capacity)'))):
        ax.plot(np.sort(observed[:,j])*100,np.arange(1,len(observed)+1)/len(observed),color='#8793a6',label='2019 measured errors')
        ax.step(np.sort(simulated[:,j])*100,np.arange(1,len(simulated)+1)/len(simulated),where='post',color='#315d8c',label='Fitted model samples')
        ax.set(xlabel=label,ylabel='Cumulative probability',title='Unquantized holdout vs discrete model')
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/'disturbance_validation.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/v2');p.add_argument('--config',default='configs/empirical_opsd.json')
    p.add_argument('--cache',default='../data/opsd_residuals.npz')
    a=p.parse_args();plots(a.root,a.config,a.cache)
