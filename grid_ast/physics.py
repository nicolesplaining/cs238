"""Gym-ANM AC power flow with one battery and fixed zero reactive setpoints.

Source network and Newton-Raphson solver: Henry and Ernst, Gym-ANM, 2021,
https://github.com/robinhenry/gym-anm (pinned release 2.0.1).

The restricted backend skips general-purpose QP projection because Q=0 and
the allowed P interval has an exact scalar projection. The original Simulator
remains available as a reference backend and is checked in regression tests.
No linear or DC approximation is used for the power flow.
"""

import numpy as np
from gym_anm.simulator.simulator import Simulator
from gym_anm.simulator import solve_load_flow


class RestrictedSimulator(Simulator):
    def transition(self, P_load, P_potential, P_set_points, Q_set_points):
        if any(abs(q) > 1e-12 for q in Q_set_points.values()):
            raise ValueError("Restricted backend requires Q=0 for controllable devices.")
        for i in (1, 3, 5):
            self.devices[i].map_pq(P_load[i] / self.baseMVA)
        for i in (2, 4):
            d = self.devices[i]
            d.p_pot = float(np.clip(P_potential[i]/self.baseMVA, d.p_min, d.p_max))
            d.p = float(np.clip(P_set_points[i]/self.baseMVA, d.p_min, d.p_pot))
            d.q = 0.0
        d = self.devices[6]
        lo = max(d.p_min, (d.soc-d.soc_max)/(self.delta_t*d.eff))
        hi = min(d.p_max, d.eff*(d.soc-d.soc_min)/self.delta_t)
        d.p = float(np.clip(P_set_points[6]/self.baseMVA, lo, hi))
        d.q = 0.0
        d.update_soc(self.delta_t)
        self.devices[0].p = self.devices[0].q = 0.0
        self._get_bus_total_injections()
        _, self.pfe_converged = solve_load_flow.solve_pfe_newton_raphson(self, xtol=1e-5)
        self.state = self._gather_state()
        reward, loss, penalty = self._compute_reward()
        return self.state, reward, loss, penalty, self.pfe_converged
