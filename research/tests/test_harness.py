import numpy as np
import pytest

from futuresres.stats.trials import TrialLog
from strategyres import harness as h


@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "hypotheses.yaml"
    p.write_text("- id: C99\n  status: registered\n- id: C98\n  status: draft\n")
    return p


@pytest.fixture
def log(tmp_path):
    return TrialLog(tmp_path / "trials.jsonl")


def run(log, reg, hid="C99", backtest=lambda: [1.0, -0.5, 2.0, 0.3]):
    return h.run_logged(hid, params={"x": 1}, symbol="MNQ", date_range=("2023-01-01", "2026-01-01"),
                        cost_bps=None, backtest=backtest, log=log, registry=reg)


def test_unregistered_and_draft_are_refused_and_spend_nothing(log, reg):
    for hid in ("C00", "C98"):
        with pytest.raises(h.Unregistered):
            run(log, reg, hid)
    assert len(log) == 0


def test_completed_run_is_logged_net_of_cost(log, reg):
    run(log, reg)
    (t,) = log.read_all()
    assert t.trial_id == "lc00001" and t.status == "completed" and t.trade_count == 4
    net = np.array([1.0, -0.5, 2.0, 0.3]) - 0.48
    assert t.sharpe == pytest.approx(net.mean() / net.std(ddof=1))


def test_a_crash_still_counts_toward_n(log, reg):
    def boom():
        raise RuntimeError("bad bar")
    with pytest.raises(RuntimeError):
        run(log, reg, backtest=boom)
    run(log, reg)
    trials = log.read_all()
    assert [t.status for t in trials] == ["error", "completed"]
    assert "bad bar" in trials[0].note
    assert log.verify_chain().ok


def test_unknown_cost_must_be_supplied():
    with pytest.raises(ValueError):
        h.cost_for("MES", None)
    assert h.cost_for("MES", 0.3) == 0.3


def test_benjamini_hochberg_matches_hand_worked_case():
    # sorted p: .001 .008 .039 .041 .042 .06 ; bars at q=.05, m=6: .0083 .0167 .025 .033 .0417 .05
    # largest k with p_(k) <= bar is k=2, so exactly the first two are discoveries
    p = [0.039, 0.001, 0.06, 0.041, 0.008, 0.042]
    assert h.benjamini_hochberg(p).tolist() == [False, True, False, False, True, False]
    assert h.benjamini_hochberg([0.5, 0.9]).tolist() == [False, False]


def test_too_good_to_be_true_is_flagged(log):
    rng = np.random.default_rng(0)
    e = h.evaluate(rng.normal(5, 1, 500), symbol="MNQ", trades_per_year=250, log=log, n_upstream=0)
    joined = " ".join(e.flags)
    assert "annual Sharpe" in joined and "win rate" in joined and "DSR not computable" in joined


def test_cost_stress_and_upstream_deflation(log, reg):
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = list(rng.normal(0, 10, 200))
        run(log, reg, backtest=lambda r=r: r)
    gross = rng.normal(0.7, 10, 2000)  # clears 1x cost on average, not 2x
    e = h.evaluate(gross, symbol="MNQ", trades_per_year=500, log=log, n_upstream=761)
    assert e.net_mean_bps == pytest.approx(gross.mean() - 0.48)
    assert e.net_mean_2x_cost_bps == pytest.approx(gross.mean() - 0.96)
    assert e.dsr_with_upstream.n_trials == e.dsr_own.n_trials + 761
    assert e.dsr_with_upstream.dsr < e.dsr_own.dsr
