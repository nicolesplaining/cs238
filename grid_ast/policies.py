"""Frozen controllers and a forecast-only action-scaling shield."""

from dataclasses import dataclass
import numpy as np
from .model import tariff


class IdlePolicy:
    def act(self, env, observation):
        return 0.0


class RulePolicy:
    def act(self, env, observation):
        if tariff(env.time_step+1) > 40:
            return -1.0 if env.soc > .15 else 0.0
        return 0.5 if env.soc < .75 else 0.0


class PPOPolicy:
    def __init__(self, path):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        self.model = PPO.load(path, device="cpu")

    def act(self, env, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])


@dataclass
class ShieldDecision:
    action: float
    proposed: float
    intervened: bool
    forecast_safe: bool
    predictions: int


class ForecastShield:
    """Scale toward zero; return the least-violating candidate if none is safe.

    This is a heuristic, not a formal shielding guarantee. It cannot command
    extra discharge when the base controller chose zero, or protect against
    an unobserved future innovation. It never accesses an adversarial action.
    """

    scales = (1.0, .75, .5, .25, 0.0)

    def choose(self, env, proposed):
        proposed = float(np.clip(proposed, -1, 1))
        c = env.config
        best, best_gap, calls = proposed, np.inf, 0
        candidates = list(dict.fromkeys(proposed*s for s in self.scales))
        for action in candidates:
            v, converged = env.forecast_voltage(action)
            calls += 1
            gap = float(max(0, np.max(c.v_min+c.shield_margin-v), np.max(v-c.v_max+c.shield_margin))) if converged else np.inf
            if gap < best_gap:
                best, best_gap = action, gap
            if gap == 0:
                return ShieldDecision(action, proposed, abs(action-proposed)>1e-9, True, calls)
        return ShieldDecision(best, proposed, abs(best-proposed)>1e-9, False, calls)


def load_policy(name, model=None):
    if name == "idle":
        return IdlePolicy()
    if name == "rule":
        return RulePolicy()
    if name == "ppo" and model:
        return PPOPolicy(model)
    raise ValueError("Use idle, rule, or ppo with a model path.")
