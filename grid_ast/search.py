"""Finite-action Monte Carlo tree search for adaptive stress testing.

AST reference: Lee et al., JAIR 69, 2020, doi:10.1613/jair.1.12190.
All transitions, including the transition causing failure, contribute log p(x).
No discount is applied. A terminal safe-path penalty dominates any possible
failure path's negative log probability. Search does not estimate failure risk.
"""

from dataclasses import dataclass, field
import math
import time
import numpy as np

from .env import BatteryEnv
from .policies import ForecastShield


@dataclass
class Node:
    visits: int = 0
    total: float = 0.0
    children: dict = field(default_factory=dict)

    def select(self, rng, exploration, n_actions=9):
        missing = [a for a in range(n_actions) if a not in self.children]
        if missing:
            a = int(rng.choice(missing))
            self.children[a] = Node()
            return a, self.children[a], True
        scores = {a: child.total/child.visits + exploration*math.sqrt(math.log(max(self.visits, 1))/child.visits)
                  for a, child in self.children.items()}
        best = max(scores.values())
        a = int(rng.choice([a for a, s in scores.items() if abs(s-best)<1e-12]))
        return a, self.children[a], False


def terminal_score(log_probability, failure, distance, horizon, min_log_probability):
    alpha = -horizon*min_log_probability + 1
    return log_probability if failure else log_probability-alpha-1000*distance


def action_for(env, policy, observation, shield):
    proposed = policy.act(env, observation)
    decision = shield.choose(env, proposed) if shield is not None else None
    return (decision.action if decision else proposed), decision


def search(config, policy, method="mcts", budget=10000, seed=1000, shielded=False, exploration=.3,
           persistent_rollout_probability=.25):
    if method not in ("mcts", "mc") or budget < 1:
        raise ValueError("Expected mc or mcts and a positive transition budget.")
    if config.random_start:
        raise ValueError("Search requires a fixed initial state.")
    if not 0 <= persistent_rollout_probability <= 1:
        raise ValueError("Persistent rollout probability must be in [0, 1].")
    env, rng = BatteryEnv(config), np.random.default_rng(seed)
    shield = ForecastShield() if shielded else None
    root = Node()
    used = failures = completed = censored = solver_failures = 0
    best = None
    unique, modes, history = set(), set(), []
    flow_calls = resets = shield_predictions = 0
    started = time.perf_counter()
    alpha = -config.horizon*env.noise.min_log_probability+1
    while used < budget:
        observation, info = env.reset(seed=seed)
        resets += 1
        node, visited, expanding = root, [root], False
        # A search proposal only: some rollout suffixes sustain one innovation.
        # Their score still uses the original per-step probabilities, never this
        # proposal distribution. Set probability=0 for the nominal-rollout ablation.
        persistent_action = None
        if method == "mcts" and persistent_rollout_probability > 0:
            if rng.random() < persistent_rollout_probability:
                persistent_action = int(rng.integers(env.noise.n_actions))
        sequence, trajectory, logp = [], [], 0.0
        failed = invalid = False
        for t in range(config.horizon):
            if used >= budget:
                break
            if method == "mcts" and not expanding:
                a, node, expanding = node.select(rng, exploration, env.noise.n_actions)
                visited.append(node)
            else:
                a = persistent_action if persistent_action is not None else env.noise.sample(rng, env.residual, (env.time_step+1)%96)
            # Controller action is selected before the chosen disturbance is applied.
            action, decision = action_for(env, policy, observation, shield)
            if decision:
                shield_predictions += decision.predictions
            observation, _, terminated, truncated, info = env.step_disturbance([action], a)
            if decision:
                info.update({"proposed_action": decision.proposed, "shield_intervened": decision.intervened,
                             "forecast_safe": decision.forecast_safe})
            used += 1
            sequence.append(a)
            trajectory.append(info)
            logp += info["log_probability"]
            failed = info["voltage_failure"]
            invalid = info["solver_failure"]
            if failed or terminated or truncated:
                break
        flow_calls += env.power_flow_calls
        is_complete = failed or invalid or len(sequence) == config.horizon
        if not is_complete:
            censored += 1
            # The final budget-truncated rollout cannot update a terminal score.
            break
        completed += 1
        if failed:
            failures += 1
            unique.add(tuple(sequence))
            modes.add(info["failure_mode"])
            if best is None or logp > best["log_probability"]:
                best = {"log_probability": logp, "prefix_probability": math.exp(logp),
                        "sequence": sequence, "trajectory": trajectory, "found_at_transition": used}
        if invalid:
            solver_failures += 1
        score = terminal_score(logp, failed, info["distance"] if not invalid else 1.0,
                               config.horizon, env.noise.min_log_probability)
        if method == "mcts":
            for n in visited:
                n.visits += 1
                n.total += score/alpha
        history.append({"transitions": used, "completed": completed, "failures": failures,
                        "unique_failures": len(unique), "best_log_probability": None if best is None else best["log_probability"]})
    return {"method": method, "seed": seed, "shielded": shielded, "budget": budget,
            "exploration": exploration if method == "mcts" else None,
            "persistent_rollout_probability": persistent_rollout_probability if method == "mcts" else None,
            "transitions": used, "rollouts": completed, "censored_rollouts": censored,
            "failed_rollouts": failures, "unique_failure_prefixes": len(unique),
            "failure_modes": sorted(modes), "solver_failures": solver_failures,
            "reset_power_flows": resets, "shield_predictions": shield_predictions,
            "total_power_flows": flow_calls, "elapsed_seconds": time.perf_counter()-started,
            "best": best, "history": history,
            "interpretation": "Best-found prefix mass conditional on fixed initial state; not total failure probability."}


def replay(config, policy, sequence, shielded=False):
    env = BatteryEnv(config)
    observation, _ = env.reset(seed=0)
    shield = ForecastShield() if shielded else None
    rows = []
    for a in sequence:
        action, decision = action_for(env, policy, observation, shield)
        observation, _, terminated, truncated, info = env.step_disturbance([action], int(a))
        if decision:
            info["shield_intervened"] = decision.intervened
        rows.append(info)
        if info["voltage_failure"] or terminated or truncated:
            break
    return rows
