"""Replay held-out, unquantized measured errors through the toy feeder.

Historical error traces are transferred to ANM6's deterministic base forecasts.
These are not reconstructed measurements of the German grid or a real feeder.
"""

import numpy as np
from .env import BatteryEnv
from .policies import ForecastShield
from .search import action_for


def historical_days(cache, year=2019):
    with np.load(cache, allow_pickle=False) as arrays:
        values,valid,steps,years,dates,timestamps = [arrays[k] for k in ("values","valid","steps","years","dates","timestamps")]
    days,excluded = [],[]
    for date in sorted(set(dates[years==year])):
        indices = np.flatnonzero(dates==date)
        if (len(indices)!=96 or not np.all(valid[indices]) or
            not np.array_equal(steps[indices],np.arange(96)) or
            not np.all(np.diff(timestamps[indices])==15*60*1_000_000_000)):
            excluded.append(str(date))
            continue
        days.append({"date":str(date),"residuals":values[indices].copy()})
    if not days:
        raise ValueError("No complete held-out days found.")
    return days,excluded


def evaluate_history(config, policy, days, shielded=False):
    if config.disturbance_model is None or config.random_start:
        raise ValueError("Historical replay requires the empirical units and fixed initial conditions.")
    if config.start_step+config.horizon >= 96:
        raise ValueError("Replay window must fit within one measured day, including the initial state.")
    env=BatteryEnv(config)
    shield=ForecastShield() if shielded else None
    rows=[]
    for episode,day in enumerate(days):
        obs,initial=env.reset(seed=0,options={"initial_residual":day["residuals"][config.start_step],"allow_initial_failure":True})
        failed,invalid=initial["voltage_failure"],False
        cost=0.
        interventions=violation_steps=unsafe_forecasts=0
        vmin=min(initial["voltages"])
        for i in range(config.horizon):
            action,decision=action_for(env,policy,obs,shield)
            residual=day["residuals"][config.start_step+i+1]
            obs,_,terminated,truncated,info=env.step_residual([action],residual)
            failed |= info["voltage_failure"]
            invalid |= info["solver_failure"]
            cost += info["cost"]
            vmin=min(vmin,min(info["voltages"]))
            violation_steps += int(info["voltage_failure"])
            if decision:
                interventions += int(decision.intervened)
                unsafe_forecasts += int(not decision.forecast_safe)
            if terminated or truncated: break
        rows.append({"episode":episode,"date":day["date"],"steps":i+1,"initial_failure":int(initial["voltage_failure"]),
                     "voltage_failure":int(failed),"solver_failure":int(invalid),"cost":cost,"min_voltage":vmin,
                     "end_soc":env.soc,"violation_steps":violation_steps,"shield_interventions":interventions,
                     "unsafe_forecasts":unsafe_forecasts,"power_flow_calls":env.power_flow_calls})
    n=len(rows); failures=sum(r["voltage_failure"] for r in rows)
    return {"episodes":n,"failed_episodes":failures,"failure_rate":failures/n,
            "mean_cost":float(np.mean([r["cost"] for r in rows])),
            "solver_failures":sum(r["solver_failure"] for r in rows),"rows":rows,
            "interpretation":"Unquantized 2019 national forecast-error traces transferred to ANM6 forecasts. Days are consecutive and not statistically independent."}


def seed_day_summary(costs, failures, dates, bootstrap_seed=20261116, draws=3000, block_days=7):
    """Resample policies and week-long blocks of days, not flattened episodes.

    Policies are training-seed replicates. A shared day-block draw is used for
    every policy, preserving paired weather and within-week temporal dependence.
    Blocks stop at missing-calendar-date gaps. Intervals are exploratory with
    only five policy seeds and cannot account for the feeder-transfer error.
    """
    costs,failures=np.asarray(costs,float),np.asarray(failures,float)
    if costs.ndim!=2 or costs.shape!=failures.shape or costs.shape[1]!=len(dates):
        raise ValueError("Expected seed-by-day cost and failure matrices with matching dates.")
    nseeds,ndays=costs.shape
    dates=np.array(dates,dtype="datetime64[D]")
    blocks=[]
    for start in range(ndays):
        block=[start]
        for j in range(start+1,min(start+block_days,ndays)):
            if dates[j]-dates[j-1]!=np.timedelta64(1,'D'): break
            block.append(j)
        blocks.append(np.array(block))
    rng=np.random.default_rng(bootstrap_seed)
    samples=np.zeros((draws,2))
    for b in range(draws):
        policy_ids=rng.integers(nseeds,size=nseeds)
        selected=[]
        while len(selected)<ndays:
            selected.extend(blocks[int(rng.integers(ndays))])
        indices=np.array(selected[:ndays])
        samples[b]=[costs[np.ix_(policy_ids,indices)].mean(),failures[np.ix_(policy_ids,indices)].mean()]
    per_seed_cost=costs.mean(axis=1);per_seed_failure=failures.mean(axis=1)
    return {"policy_seeds":nseeds,"days":ndays,"mean_cost":float(costs.mean()),
            "cost_seed_std":float(per_seed_cost.std(ddof=1)) if nseeds>1 else None,
            "failure_rate":float(failures.mean()),"failure_rate_seed_range":[float(per_seed_failure.min()),float(per_seed_failure.max())],
            "per_seed_mean_cost":per_seed_cost.tolist(),"per_seed_failure_rate":per_seed_failure.tolist(),
            "cost_seed_day_block_bootstrap95":np.quantile(samples[:,0],[.025,.975]).tolist(),
            "failure_seed_day_block_bootstrap95":np.quantile(samples[:,1],[.025,.975]).tolist(),
            "bootstrap_seed":bootstrap_seed,"bootstrap_draws":draws,"block_days":block_days,
            "zero_failures_caution":"All-zero failures yield a degenerate bootstrap interval, not a zero-risk guarantee."}
