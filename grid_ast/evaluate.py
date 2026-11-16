"""Independent nominal Monte Carlo, paired across policies by innovation seed."""

import csv
import math
from pathlib import Path
import numpy as np
from .env import BatteryEnv
from .policies import ForecastShield
from .search import action_for


def wilson(successes, n, z=1.959963984540054):
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0.")
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def evaluate(config, policy, episodes=200, seed=20000, shielded=False):
    if episodes < 1 or config.random_start:
        raise ValueError("Use positive episodes and a fixed initial state for paired evaluation.")
    env = BatteryEnv(config)
    shield = ForecastShield() if shielded else None
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed+episode)
        cost = reward = 0.0
        failed = invalid = False
        voltage_steps = interventions = unsafe_forecasts = 0
        vmin, vmax = 1., 1.
        for step in range(config.horizon):
            action, decision = action_for(env, policy, obs, shield)
            obs, r, terminated, truncated, info = env.step([action])
            cost += info["cost"]
            reward += r
            failed |= info["voltage_failure"]
            invalid |= info["solver_failure"]
            voltage_steps += int(info["voltage_failure"])
            if decision:
                interventions += int(decision.intervened)
                unsafe_forecasts += int(not decision.forecast_safe)
            vmin, vmax = min(vmin, min(info["voltages"])), max(vmax, max(info["voltages"]))
            if terminated or truncated:
                break
        rows.append({"episode": episode, "seed": seed+episode, "steps": step+1,
                     "voltage_failure": int(failed), "solver_failure": int(invalid),
                     "violation_steps": voltage_steps, "cost": cost, "return": reward,
                     "min_voltage": vmin, "max_voltage": vmax, "end_soc": env.soc,
                     "shield_interventions": interventions, "unsafe_forecasts": unsafe_forecasts,
                     "power_flow_calls": env.power_flow_calls})
    nfailed = sum(r["voltage_failure"] for r in rows)
    costs = np.array([r["cost"] for r in rows])
    returns = np.array([r["return"] for r in rows])
    return {"episodes": episodes, "seed": seed, "shielded": shielded,
            "failed_episodes": nfailed, "failure_rate": nfailed/episodes,
            "failure_rate_wilson95": wilson(nfailed, episodes),
            "solver_failures": sum(r["solver_failure"] for r in rows),
            "mean_cost": float(costs.mean()), "cost_standard_error": float(costs.std(ddof=1)/math.sqrt(episodes)) if episodes>1 else None,
            "mean_return": float(returns.mean()),
            "mean_end_soc": float(np.mean([r["end_soc"] for r in rows])),
            "violation_step_rate": sum(r["violation_steps"] for r in rows)/sum(r["steps"] for r in rows),
            "shield_intervention_rate": sum(r["shield_interventions"] for r in rows)/sum(r["steps"] for r in rows),
            "rows": rows}


def paired_difference(unshielded, shielded, seed=2026):
    a, b = unshielded["rows"], shielded["rows"]
    if [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("Paired comparison requires identical episode seeds.")
    delta = np.array([y["voltage_failure"]-x["voltage_failure"] for x,y in zip(a,b)])
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"shield_minus_unshielded_failure_rate": float(delta.mean()),
            "paired_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "prevented_in_sample": int(sum(delta == -1)), "introduced_in_sample": int(sum(delta == 1)),
            "mean_cost_difference": float(np.mean([y["cost"]-x["cost"] for x,y in zip(a,b)])),
            "bootstrap_seed": seed,
            "caution": "Interval covers sampled disturbances for one frozen policy, not policy-training variation; degenerates if no discordant pairs."}


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
