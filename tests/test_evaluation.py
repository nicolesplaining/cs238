import pytest
from grid_ast.evaluate import wilson, evaluate, paired_difference
from grid_ast.model import Config
from grid_ast.policies import IdlePolicy


def test_wilson_nonzero_upper_bound_for_no_failures():
    lo, hi = wilson(0, 200)
    assert lo == pytest.approx(0, abs=1e-15)
    assert hi == pytest.approx(.018845326, abs=1e-8)
    assert wilson(200, 200)[1] == pytest.approx(1)


def test_identical_policies_give_zero_paired_difference():
    config = Config(horizon=4)
    a = evaluate(config, IdlePolicy(), episodes=3, seed=200)
    b = evaluate(config, IdlePolicy(), episodes=3, seed=200, shielded=True)
    result = paired_difference(a, b)
    assert result["shield_minus_unshielded_failure_rate"] == 0
    assert result["mean_cost_difference"] == 0
    assert [r["steps"] for r in a["rows"]] == [4, 4, 4]
    b["rows"][0]["seed"] += 1
    with pytest.raises(ValueError):
        paired_difference(a, b)
