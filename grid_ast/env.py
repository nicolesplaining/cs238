"""Gymnasium environment exposing every exogenous draw to the stress tester."""

from copy import deepcopy
import gymnasium as gym
import numpy as np
from gym_anm.envs.anm6_env.network import network
from gym_anm.envs.anm6_env.anm6_easy import _get_load_time_series, _get_gen_time_series
from gym_anm.simulator.simulator import Simulator

from .model import Config, make_disturbances, tariff
from .physics import RestrictedSimulator


class BatteryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or Config()
        self.noise = make_disturbances(self.config)
        sim_type = RestrictedSimulator if self.config.backend == "restricted" else Simulator
        self.sim = sim_type(deepcopy(network), delta_t=0.25, lamb=100)
        self.loads = _get_load_time_series()
        self.generators = _get_gen_time_series()
        # Positive normalized action charges; Gym-ANM injection has opposite sign.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
        self.done = True
        self.power_flow_calls = 0

    @property
    def soc(self):
        return float(self.sim.state["des_soc"]["MWh"][6] / 100.0)

    @property
    def voltages(self):
        return np.array([self.sim.state["bus_v_magn"]["pu"][i] for i in range(6)])

    def injections(self, time_step, residual):
        load = self.loads[:, time_step] * max(0.0, 1 + residual[0])
        gen = self.generators[:, time_step].copy()
        if self.config.disturbance_model is None:
            gen[0] *= max(0.0, 1 + residual[1])
        else:
            # Empirical solar errors are fractions of installed capacity, not
            # fractions of the forecast. Keep PV zero when its base forecast is zero.
            gen[0] = max(0.0, gen[0] + 30*residual[1]) if gen[0] > 0 else 0.0
        return dict(zip((1, 3, 5), load)), dict(zip((2, 4), gen))

    def _transition(self, sim, time_step, residual, charge_mw):
        load, gen = self.injections(time_step, residual)
        self.power_flow_calls += 1
        return sim.transition(load, gen, {2: 30, 4: 50, 6: -charge_mw}, {2: 0, 4: 0, 6: 0})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        c = self.config
        self.time_step = int(options.get("start_step", self.np_random.integers(96) if c.random_start else c.start_step))
        initial_soc = float(options.get("initial_soc", self.np_random.uniform(.25, .75) if c.random_start else c.initial_soc))
        if not 0 <= self.time_step < 96 or not 0 <= initial_soc <= 1:
            raise ValueError("Invalid initial time or SoC.")
        self.elapsed = 0
        self.residual = np.array(options.get("initial_residual", [0., 0.]), dtype=float)
        if self.residual.shape != (2,) or not np.isfinite(self.residual).all():
            raise ValueError("Expected two finite initial residuals.")
        self.done = False
        self.power_flow_calls = 0
        self.sim.devices[6].soc = initial_soc  # base is 100 MVA; capacity is 100 MWh.
        _, _, _, _, converged = self._transition(self.sim, self.time_step, self.residual, 0)
        if not converged:
            raise RuntimeError("Initial AC solve did not converge.")
        info = self._info()
        if info["voltage_failure"] and not options.get("allow_initial_failure", False):
            raise ValueError("Initial state is already a voltage failure.")
        return self._observation(), info

    def _observation(self):
        s = self.sim.state
        current = [-s["dev_p"]["MW"][i]/30 for i in (1, 3, 5)]
        current += [s["dev_p"]["MW"][2]/30, s["dev_p"]["MW"][4]/50]
        next_step = (self.time_step + 1) % 96
        angle = 2*np.pi*self.time_step/96
        return np.array(current + [self.soc] + list((self.voltages[1:]-1)*20)
                        + list(self.residual) + [np.sin(angle), np.cos(angle), tariff(next_step)/120,
                        self.elapsed/self.config.horizon, -sum(self.loads[:, next_step])/70,
                        self.generators[0, next_step]/30], dtype=np.float32)

    def _info(self):
        v = self.voltages
        c = self.config
        converged = bool(self.sim.pfe_converged and np.all(np.isfinite(v)))
        lower = c.v_min - v[1:]
        upper = v[1:] - c.v_max
        severity = float(max(np.max(lower), np.max(upper), 0)) if converged else None
        failure = bool(converged and severity > 0)
        mode = None
        if failure:
            mode = f"bus_{int(np.argmax(lower))+1}_low" if np.max(lower) >= np.max(upper) else f"bus_{int(np.argmax(upper))+1}_high"
        return {"time_step": self.time_step, "elapsed": self.elapsed, "soc": self.soc,
                "voltages": v.tolist(), "voltage_failure": failure, "failure_mode": mode,
                "solver_failure": not converged, "severity": severity,
                "distance": float(max(0, min(np.min(v[1:]-c.v_min), np.min(c.v_max-v[1:])))) if converged else None}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset before stepping a completed episode.")
        return self.step_disturbance(action, self.noise.sample(self.np_random, self.residual, (self.time_step+1)%96))

    def step_disturbance(self, action, disturbance):
        if self.done:
            raise RuntimeError("Reset before stepping a completed episode.")
        a = np.asarray(action).reshape(-1)
        if a.size != 1 or not np.isfinite(a[0]):
            raise ValueError("Expected one finite normalized battery action.")
        next_step = (self.time_step+1)%96
        residual = self.noise.evolve(self.residual, disturbance, next_step)
        logp = float(np.log(self.noise.probabilities_at(self.residual, next_step)[disturbance]))
        return self._step_residual(a, residual, disturbance, logp)

    def step_residual(self, action, residual):
        """Replay a measured residual vector; it has no discrete model likelihood."""
        if self.done:
            raise RuntimeError("Reset before stepping a completed episode.")
        a = np.asarray(action).reshape(-1)
        residual = np.asarray(residual, dtype=float)
        if a.size != 1 or not np.isfinite(a[0]) or residual.shape != (2,) or not np.isfinite(residual).all():
            raise ValueError("Expected one finite action and two finite residuals.")
        return self._step_residual(a, residual, None, None)

    def _step_residual(self, action, residual, disturbance, logp):
        charge = float(np.clip(action[0], -1, 1)*self.config.battery_mw)
        self.residual = np.asarray(residual, dtype=float).copy()
        self.time_step = (self.time_step + 1) % 96
        self.elapsed += 1
        self._transition(self.sim, self.time_step, self.residual, charge)
        info = self._info()
        actual_charge = -float(self.sim.state["dev_p"]["MW"][6])
        imported = max(0.0, float(self.sim.state["dev_p"]["MW"][0]))
        cost = .25*(tariff(self.time_step)*imported + self.config.throughput_price*abs(actual_charge))
        # Cost / 1000 puts the economic and voltage penalties on readable scales.
        reward = -cost/1000 - self.config.voltage_penalty*(info["severity"] or 0)
        if info["solver_failure"]:
            reward = -100.0
        terminated = info["solver_failure"]
        truncated = self.elapsed >= self.config.horizon
        self.done = terminated or truncated
        info.update({"requested_charge_mw": charge, "actual_charge_mw": actual_charge,
                     "cost": float(cost), "disturbance": disturbance,
                     "residual": self.residual.tolist(), "log_probability": logp})
        return self._observation(), float(reward), terminated, truncated, info

    def forecast_voltage(self, normalized_action):
        """One-step forecast from current residual only; never sees next innovation.

        Separate simulator copies prevent shield lookahead from consuming SoC.
        We use the conditional mean residual, which is a zero innovation for
        the original AR model and a probability-weighted state for the Markov model.
        Clipping means forecast injection need not equal conditional mean injection.
        """
        sim = deepcopy(self.sim)
        residual = self.noise.forecast(self.residual, (self.time_step+1)%96)
        _, _, _, _, ok = self._transition(sim, (self.time_step+1)%96, residual,
                                           float(normalized_action)*self.config.battery_mw)
        v = np.array([sim.state["bus_v_magn"]["pu"][i] for i in range(1, 6)])
        return v, bool(ok and np.all(np.isfinite(v)))
