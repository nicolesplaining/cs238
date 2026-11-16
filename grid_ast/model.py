"""Explicit finite disturbance model and experiment settings.

The disturbance probabilities are synthetic assumptions, not grid measurements.
Discrete innovations give probability masses, avoiding comparisons of continuous
densities with different units. The AR residuals are part of the observed state.
"""

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    horizon: int = 96
    start_step: int = 0
    initial_soc: float = 0.5
    battery_mw: float = 20.0
    load_rho: float = 0.8
    solar_rho: float = 0.7
    load_sigma: float = 0.18
    solar_sigma: float = 0.25
    innovation_tail_probability: float = 0.1
    v_min: float = 0.95
    v_max: float = 1.05
    shield_margin: float = 0.005
    voltage_penalty: float = 3000.0
    throughput_price: float = 2.0
    random_start: bool = False
    backend: str = "restricted"
    disturbance_model: dict | None = None

    def __post_init__(self):
        if self.horizon < 1 or not 0 <= self.start_step < 96:
            raise ValueError("Require a positive horizon and start_step in [0, 96).")
        if not 0 <= self.initial_soc <= 1 or not 0 < self.battery_mw <= 30:
            raise ValueError("Invalid SoC or battery active-power limit.")
        if not 0 < self.innovation_tail_probability < 0.5:
            raise ValueError("Innovation probabilities must be strictly positive.")
        if not 0 <= self.load_rho < 1 or not 0 <= self.solar_rho < 1:
            raise ValueError("AR coefficients must be in [0, 1).")
        if self.load_sigma < 0 or self.solar_sigma < 0:
            raise ValueError("Noise scales must be nonnegative.")
        if not 0 < self.v_min < self.v_max or not 0 <= self.shield_margin < (self.v_max-self.v_min)/2:
            raise ValueError("Invalid voltage limits or shield margin.")
        if self.backend not in ("restricted", "reference"):
            raise ValueError("Unknown backend.")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class Disturbances:
    """Independent ternary innovations for common load error and PV error.

    All three loads share the same multiplier. Wind stays at its forecast.
    Each innovation is -1, 0 or +1, with probabilities q, 1-2q, q.
    Residual updates are e' = rho*e + sigma*z. Physical device limits clip
    realized injections; likelihoods always refer to the latent innovations.
    """

    def __init__(self, config: Config):
        self.config = config
        self.values = np.array(list(product((-1, 0, 1), repeat=2)), dtype=float)
        q = config.innovation_tail_probability
        p = {-1: q, 0: 1 - 2*q, 1: q}
        self.probabilities = np.array([p[int(x)]*p[int(y)] for x, y in self.values])
        self.log_probabilities = np.log(self.probabilities)
        self.nominal = 4
        self.n_actions = len(self.values)
        self.min_log_probability = float(self.log_probabilities.min())

    def probabilities_at(self, residual, next_step):
        return self.probabilities

    def sample(self, rng, residual=None, next_step=None):
        return int(rng.choice(len(self.values), p=self.probabilities))

    def evolve(self, residual, action, next_step=None):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.values):
            raise ValueError("Disturbance action must be an integer in [0, 9).")
        c = self.config
        return np.array([c.load_rho, c.solar_rho])*residual + np.array([c.load_sigma, c.solar_sigma])*self.values[action]

    def forecast(self, residual, next_step):
        return self.evolve(residual, self.nominal)

    def log_probability(self, sequence, start_step=0, residual=None):
        return float(sum(self.log_probabilities[int(a)] for a in sequence))


def make_disturbances(config):
    if config.disturbance_model is not None:
        from .empirical import MarkovDisturbances
        return MarkovDisturbances(config.disturbance_model)
    return Disturbances(config)


def tariff(step):
    """Synthetic time-of-use import prices in dollars/MWh; exports earn zero."""
    hour = (step % 96) / 4
    return 120.0 if 8 <= hour < 11 or 18 <= hour < 21 else 40.0
