"""Finite joint Markov model of measured load and solar forecast errors.

An AST action selects the NEXT residual state. Its probability is conditional
on the current residual state and the next time-of-day period. This differs from
the legacy model, where actions select innovations in an AR process.
"""

import numpy as np


class MarkovDisturbances:
    def __init__(self, model):
        if model.get("kind") != "joint_markov_v1":
            raise ValueError("Unknown empirical disturbance model.")
        self.values = np.array(model["centers"], dtype=float)
        self.transitions = np.array(model["transitions"], dtype=float)
        self.period_steps = int(model["period_steps"])
        self.n_actions = len(self.values)
        if self.period_steps < 1 or 96 % self.period_steps:
            raise ValueError("Periods must partition a 96-step day.")
        if self.values.shape != (self.n_actions, 2) or self.n_actions < 2:
            raise ValueError("Expected at least two two-dimensional residual states.")
        if self.transitions.shape != (96//self.period_steps, self.n_actions, self.n_actions):
            raise ValueError("Transition matrix shape does not match periods and states.")
        if not np.isfinite(self.values).all() or not np.isfinite(self.transitions).all():
            raise ValueError("Model parameters must be finite.")
        if (self.transitions <= 0).any() or not np.allclose(self.transitions.sum(axis=-1), 1, atol=1e-10):
            raise ValueError("Transition rows must be strictly positive and sum to one.")
        self.scale = np.maximum(self.values.std(axis=0), 1e-8)
        self.edges = [np.asarray(e, dtype=float) for e in model["edges"]] if "edges" in model else None
        if self.edges is not None:
            if len(self.edges) != 2 or any(not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in self.edges):
                raise ValueError("Quantization edges must be finite and increasing.")
            if (len(self.edges[0])+1)*(len(self.edges[1])+1) != self.n_actions:
                raise ValueError("Quantization bins do not match the state count.")
        self.nominal = self.state_index(np.zeros(2))
        self.log_probabilities = np.log(self.transitions)
        self.min_log_probability = float(self.log_probabilities.min())

    def state_index(self, residual):
        if self.edges is not None:
            i, j = [int(np.searchsorted(edge, value, side="right")) for edge, value in zip(self.edges, residual)]
            return i*(len(self.edges[1])+1)+j
        distance = ((self.values-np.asarray(residual))/self.scale)**2
        return int(np.argmin(distance.sum(axis=1)))

    def probabilities_at(self, residual, next_step):
        return self.transitions[(int(next_step)%96)//self.period_steps, self.state_index(residual)]

    def sample(self, rng, residual=None, next_step=None):
        residual = np.zeros(2) if residual is None else residual
        return int(rng.choice(self.n_actions, p=self.probabilities_at(residual, next_step or 0)))

    def evolve(self, residual, action, next_step=None):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError("Invalid empirical residual-state action.")
        return self.values[action].copy()

    def forecast(self, residual, next_step):
        return self.probabilities_at(residual, next_step) @ self.values

    def log_probability(self, sequence, start_step=0, residual=None):
        state = np.zeros(2) if residual is None else np.array(residual, dtype=float)
        total = 0.0
        for i, action in enumerate(sequence):
            next_step = (start_step+i+1)%96
            total += float(np.log(self.probabilities_at(state, next_step)[int(action)]))
            state = self.evolve(state, int(action), next_step)
        return total
