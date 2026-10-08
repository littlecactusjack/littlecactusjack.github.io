import numpy as np

from strategyres import topstep_ev as t


def test_reproduces_the_closed_form_before_reporting_anything():
    assert t.validate(np.random.default_rng(0))["ok"]


def test_payout_never_exceeds_the_live_cap():
    f = t.simulate_funded(np.random.default_rng(1), 2000, 1200, 2.0, "eod_trailing", 5_000.0)
    assert f["expected_payout"] <= t.SPLIT * t.TOTAL_CAP + 1e-9


def test_edge_raises_pass_rate():
    lo = t.simulate_eval(np.random.default_rng(2), 4000, 400, 0.0, "eod_trailing")["p_pass"]
    hi = t.simulate_eval(np.random.default_rng(2), 4000, 400, 2.0, "eod_trailing")["p_pass"]
    assert hi > lo


def test_one_month_costs_one_fee_and_renewal_costs_more():
    one = t.simulate_eval(np.random.default_rng(3), 2000, 150, 0.0, "eod_trailing", months=1)
    six = t.simulate_eval(np.random.default_rng(3), 2000, 150, 0.0, "eod_trailing", months=6)
    assert one["fees"] == t.FEE and six["fees"] > t.FEE and six["p_pass"] >= one["p_pass"]
