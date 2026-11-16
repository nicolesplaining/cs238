"""Frozen controllers and a forecast-only action-scaling shield."""

from dataclasses import dataclass
import json
from pathlib import Path
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
    """Load a frozen policy; v2 uses preserve_training_timebase=True.

    The default retains the v1 results' behavior for exact historical replay.
    New shorter-window evaluations should keep the training observation clock.
    """
    def __init__(self, path, preserve_training_timebase=False):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        self.model = PPO.load(path, device="cpu")
        config_path = Path(path).parent/"config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.empirical_observation = config.get("disturbance_model") is not None
        self.training_horizon = int(config.get("horizon", 96)) if preserve_training_timebase else None

    def act(self, env, observation):
        observation = policy_observation(env, observation, self.empirical_observation, self.training_horizon)
        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])


def policy_observation(env, observation, empirical_training, training_horizon=None):
    """Keep the solar residual feature's units consistent across noise regimes.

    Legacy policies were trained with solar error / forecast; empirical policies
    use error / installed capacity. The measured physical features are unchanged.
    Conversion sets this feature to zero when the PV forecast is zero.
    """
    empirical_environment = env.config.disturbance_model is not None
    if empirical_training == empirical_environment and training_horizon is None:
        return observation
    converted = np.array(observation, copy=True)
    if empirical_training != empirical_environment:
        forecast = float(env.generators[0, env.time_step])
        if forecast <= 0:
            converted[12] = 0.
        elif empirical_training:
            converted[12] *= forecast/30
        else:
            converted[12] *= 30/forecast
    if training_horizon is not None:
        if training_horizon <= 0: raise ValueError("Training horizon must be positive.")
        converted[16] = env.elapsed/training_horizon
    return converted


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


def load_policy(name, model=None, preserve_training_timebase=False):
    if name == "idle":
        return IdlePolicy()
    if name == "rule":
        return RulePolicy()
    if name == "ppo" and model:
        return PPOPolicy(model, preserve_training_timebase=preserve_training_timebase)
    raise ValueError("Use idle, rule, or ppo with a model path.")
