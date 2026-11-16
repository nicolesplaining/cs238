from dataclasses import replace
from copy import deepcopy
import math
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from grid_ast.env import BatteryEnv
from grid_ast.model import Config, Disturbances
from grid_ast.policies import ForecastShield, IdlePolicy
from grid_ast.search import search, replay, terminal_score


def test_gym_contract():
    check_env(BatteryEnv(), warn=True)


@pytest.mark.parametrize("soc", [0., .03, .5, .98, 1.])
def test_restricted_backend_matches_original_ac_simulator(soc):
    config = Config(initial_soc=soc, start_step=24, horizon=24)
    fast, ref = BatteryEnv(config), BatteryEnv(replace(config, backend="reference"))
    fast.reset(seed=2)
    ref.reset(seed=2)
    rng = np.random.default_rng(42)
    for _ in range(24):
        action, disturbance = [rng.uniform(-1, 1)], int(rng.integers(9))
        a = fast.step_disturbance(action, disturbance)
        b = ref.step_disturbance(action, disturbance)
        np.testing.assert_allclose(a[0], b[0], atol=3e-5, rtol=1e-5)
        assert a[1] == pytest.approx(b[1], abs=.001)
        assert a[4]["voltage_failure"] == b[4]["voltage_failure"]


def test_battery_sign_energy_balance_and_saturation():
    env = BatteryEnv(Config(horizon=100))
    env.reset()
    env.step_disturbance([1], 4)
    assert env.soc == pytest.approx(.5 + .25*.9*20/100)
    env.step_disturbance([-1], 4)
    assert env.soc == pytest.approx(.545 - .25*20/.9/100)
    for _ in range(30):
        env.step_disturbance([1], 4)
    assert env.soc == pytest.approx(1)
    for _ in range(40):
        env.step_disturbance([-1], 4)
    assert env.soc == pytest.approx(0)


def test_noise_normalization_and_path_probability():
    d = Disturbances(Config())
    assert d.probabilities.sum() == pytest.approx(1)
    np.testing.assert_allclose(d.probabilities @ d.values, [0, 0], atol=1e-16)
    assert math.exp(d.log_probability([4, 8])) == pytest.approx(.64*.01)
    with pytest.raises(ValueError):
        d.evolve(np.zeros(2), 9)


def test_disturbance_replay_is_independent_of_rng():
    env1, env2 = BatteryEnv(), BatteryEnv()
    env1.reset(seed=1)
    env2.reset(seed=999)
    for a in [4, 8, 7, 6, 0, 1, 4]:
        np.testing.assert_array_equal(env1.step_disturbance([.2], a)[0], env2.step_disturbance([.2], a)[0])


def test_forecast_does_not_mutate_state_or_consume_randomness():
    env = BatteryEnv(Config(start_step=32))
    obs, _ = env.reset(seed=3)
    state = deepcopy(env.np_random.bit_generator.state)
    before = (env.soc, env.time_step, env.residual.copy(), env.voltages.copy())
    decision = ForecastShield().choose(env, 1.0)
    assert decision.intervened
    assert decision.forecast_safe
    assert 0 <= decision.action < 1
    assert env.soc == before[0] and env.time_step == before[1]
    np.testing.assert_array_equal(env.residual, before[2])
    np.testing.assert_array_equal(env.voltages, before[3])
    assert env.np_random.bit_generator.state == state
    np.testing.assert_array_equal(env._observation(), obs)


def test_failure_step_likelihood_and_strict_search_budget():
    config = Config(horizon=8, start_step=32, load_sigma=.8)
    class Charge:
        def act(self, env, observation):
            return 1.
    result = search(config, Charge(), method="mcts", budget=83, seed=3)
    assert result["transitions"] == 83
    assert result["total_power_flows"] == 83 + result["reset_power_flows"]
    assert result["best"] is not None
    best = result["best"]
    assert best["log_probability"] == pytest.approx(Disturbances(config).log_probability(best["sequence"]))
    rows = replay(config, Charge(), best["sequence"])
    assert rows[-1]["voltage_failure"]
    np.testing.assert_allclose(rows[-1]["voltages"], best["trajectory"][-1]["voltages"])


def test_safe_terminal_penalty_cannot_beat_a_failure():
    horizon, min_logp = 32, math.log(.01)
    worst_failure = terminal_score(horizon*min_logp, True, 0, horizon, min_logp)
    best_safe = terminal_score(0, False, 0, horizon, min_logp)
    assert worst_failure > best_safe


def test_one_step_mcts_matches_exhaustive_failure_oracle():
    config = Config(horizon=1, start_step=32, load_sigma=.8)
    class Charge:
        def act(self, env, observation):
            return 1.
    env, values = BatteryEnv(config), []
    for a in range(9):
        env.reset()
        _, _, _, _, info = env.step_disturbance([1], a)
        if info["voltage_failure"]:
            values.append(info["log_probability"])
    result = search(config, Charge(), method="mcts", budget=9, seed=4)
    assert result["best"]["log_probability"] == pytest.approx(max(values))
    assert result["rollouts"] == 9


def test_no_failure_is_reported_as_null_not_zero_probability():
    result = search(Config(horizon=2, load_sigma=0, solar_sigma=0), IdlePolicy(), budget=7)
    assert result["best"] is None
    assert result["censored_rollouts"] == 1
    assert result["transitions"] == 7


def test_seeded_mc_and_mcts_are_reproducible():
    for method in ("mc", "mcts"):
        config = Config(horizon=4, start_step=32)
        a = search(config, IdlePolicy(), method=method, budget=40, seed=17)
        b = search(config, IdlePolicy(), method=method, budget=40, seed=17)
        assert a["history"] == b["history"]


def test_terminal_and_bad_action_guards():
    env = BatteryEnv(Config(horizon=1))
    with pytest.raises(RuntimeError):
        env.step([0])
    env.reset()
    with pytest.raises(ValueError):
        env.step([np.nan])
    _, _, _, truncated, _ = env.step([0])
    assert truncated
    with pytest.raises(RuntimeError):
        env.step([0])
