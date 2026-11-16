from dataclasses import replace
import copy
import numpy as np
import pytest

from grid_ast.model import Config, make_disturbances
from grid_ast.empirical import MarkovDisturbances
from grid_ast.env import BatteryEnv
from grid_ast.search import search
from grid_ast.calibrate import fit_model
from grid_ast.policies import policy_observation
from grid_ast.historical import seed_day_summary, historical_days


def example_model():
    matrix=np.array([[.8,.1,.1],[.2,.6,.2],[.1,.1,.8]])
    transitions=np.stack([matrix,matrix[:,::-1],matrix,matrix])
    return {"kind":"joint_markov_v1","centers":[[-.1,-.2],[0,0],[.8,.2]],
            "period_steps":24,"transitions":transitions.tolist(),"solar_error_unit":"capacity_fraction"}


def test_conditional_likelihood_uses_current_state_and_next_period():
    noise=MarkovDisturbances(example_model())
    # Start at zero state 1, cross midnight-period boundary at step 24.
    assert noise.log_probability([0,2],start_step=22) == pytest.approx(np.log(.2)+np.log(.8))
    assert not np.array_equal(noise.probabilities_at([0,0],23),noise.probabilities_at([.8,.2],23))
    assert noise.log_probability([0,0],start_step=22) != noise.log_probability([0,0],start_step=0)


def test_empirical_forecast_is_conditional_mean():
    n=MarkovDisturbances(example_model())
    np.testing.assert_allclose(n.forecast([0,0],0),np.array([.2,.6,.2])@n.values)


def test_empirical_environment_likelihood_and_capacity_scaling():
    env=BatteryEnv(Config(start_step=32,disturbance_model=example_model()))
    obs,_=env.reset()
    assert obs.shape==(19,)
    load,gen=env.injections(32,np.array([0.,.2]))
    assert gen[2]==pytest.approx(10.)  # 4 MW forecast + .2 * 30 MW capacity.
    assert env.injections(0,np.array([0.,.2]))[1][2]==0  # no nighttime PV.
    _,_,_,_,info=env.step_disturbance([0],0)
    assert info['log_probability']==pytest.approx(np.log(.2))
    env.reset()
    _,_,_,_,info=env.step_residual([0],[.01,-.1])
    assert info['log_probability'] is None


def test_variable_action_count_search_matches_exhaustive_oracle():
    config=Config(start_step=32,horizon=1,disturbance_model=example_model())
    class Charge:
        def act(self,env,observation): return 1.
    env=BatteryEnv(config)
    likelihoods=[]
    for action in range(3):
        env.reset()
        *_,info=env.step_disturbance([1],action)
        if info['voltage_failure']: likelihoods.append(info['log_probability'])
    result=search(config,Charge(),budget=3,seed=1)
    assert result['best']['log_probability']==pytest.approx(max(likelihoods))
    assert result['transitions']==3


def test_empirical_reference_backend_agrees():
    c=Config(start_step=24,horizon=12,disturbance_model=example_model())
    fast,ref=BatteryEnv(c),BatteryEnv(replace(c,backend='reference'))
    fast.reset();ref.reset()
    for i in range(12):
        a=[.7*np.sin(i)]
        x=fast.step_disturbance(a,i%3)
        y=ref.step_disturbance(a,i%3)
        np.testing.assert_allclose(x[0],y[0],atol=3e-5)
        assert x[-1]['voltage_failure']==y[-1]['voltage_failure']


def test_calibration_cannot_read_heldout_values():
    rng=np.random.default_rng(123)
    x=rng.normal(size=(3000,2))*.03
    steps=np.arange(len(x))%96
    valid=np.ones(len(x),bool)
    fit=np.arange(len(x))<2000
    daylight=np.ones(len(x),bool)
    first,counts,_=fit_model(x,steps,valid,fit,daylight)
    modified=x.copy();modified[~fit]=1000
    second,_,_=fit_model(modified,steps,valid,fit,daylight)
    assert first==second
    assert counts.sum()==1999
    n=MarkovDisturbances(first)
    assert n.n_actions==25
    assert np.all(n.transitions>0)
    np.testing.assert_allclose(n.transitions.sum(axis=-1),1)
    for i,center in enumerate(n.values): assert n.state_index(center)==i


def test_missing_rows_do_not_create_artificial_transition_pairs():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(3000,2))*.02
    valid=np.ones(3000,bool);valid[100:200]=False
    _,counts,_=fit_model(x,np.arange(3000)%96,valid,np.ones(3000,bool),np.ones(3000,bool))
    assert counts.sum()==2898


def test_model_validation_and_factory():
    model=example_model();model['transitions'][0][0][0]=0
    with pytest.raises(ValueError): MarkovDisturbances(model)
    model=example_model();model['centers'][0][0]=float('nan')
    with pytest.raises(ValueError): MarkovDisturbances(model)
    assert make_disturbances(Config(disturbance_model=example_model())).n_actions==3


def test_policy_feature_units_convert_between_training_regimes():
    env=BatteryEnv(Config(start_step=32,disturbance_model=example_model()))
    obs,_=env.reset();obs[12]=.2
    converted=policy_observation(env,obs,False)
    assert converted[12]==pytest.approx(1.5)  # .2*30/4.
    assert obs[12]==pytest.approx(.2)
    env2=BatteryEnv(Config(start_step=32));env2.reset()
    restored=policy_observation(env2,converted,True)
    np.testing.assert_allclose(restored,obs)
    assert policy_observation(env,obs,True) is obs


def test_evaluation_cutoff_does_not_change_policy_timebase():
    env=BatteryEnv(Config(horizon=32));env.reset()
    for _ in range(8): obs,*_=env.step_disturbance([0],4)
    converted=policy_observation(env,obs,False,training_horizon=96)
    assert obs[16]==pytest.approx(8/32)
    assert converted[16]==pytest.approx(8/96)
    np.testing.assert_array_equal(converted[:16],obs[:16])


def test_seed_uncertainty_is_not_flattened_into_independent_episodes():
    dates=np.arange('2019-01-01','2019-01-31',dtype='datetime64[D]')
    costs=np.array([[10.]*30,[30.]*30])
    summary=seed_day_summary(costs,np.zeros_like(costs),dates,draws=500)
    assert summary['mean_cost']==20
    assert summary['cost_seed_std']==pytest.approx(np.sqrt(200))
    assert summary['cost_seed_day_block_bootstrap95']==[10,30]
    assert summary['failure_seed_day_block_bootstrap95']==[0,0]
    assert 'not a zero-risk guarantee' in summary['zero_failures_caution']


def test_historical_loader_rejects_gaps_and_incomplete_days(tmp_path):
    n=96*3
    dates=np.repeat(['2019-01-01','2019-01-02','2019-01-03'],96)
    valid=np.ones(n,bool);valid[150]=False
    steps=np.tile(np.arange(96),3);steps[250]=0
    path=tmp_path/'cache.npz'
    np.savez(path,values=np.zeros((n,2)),valid=valid,steps=steps,years=np.full(n,2019),dates=dates,
             timestamps=np.arange(n,dtype=np.int64)*15*60*1_000_000_000)
    days,excluded=historical_days(path)
    assert [d['date'] for d in days]==['2019-01-01']
    assert excluded==['2019-01-02','2019-01-03']
