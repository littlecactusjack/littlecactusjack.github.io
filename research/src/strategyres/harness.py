"""The gate every backtest goes through. Built before any backtest exists, on purpose.

Three rules, all borrowed from futures-research:

1. **Nothing runs unregistered.** `run_logged` refuses a hypothesis id that is not in
   `hypotheses.yaml` with status `registered`, so parameters and predicted size are fixed
   before the data is seen.
2. **Every run is a trial.** The trial is appended to `trials.jsonl` in a `finally` block,
   so a run that crashes or is abandoned still counts toward N. Nothing is ever deleted.
3. **Deflate against everything tried, including upstream.** CISD, POI zones and FVGs are
   the same toolkit futures-research tested as L02, L03 and L07 on the same contracts, so its
   trials are prior looks at this question. DSR is reported at our N alone and at our N
   plus upstream N; the second is the one that decides.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from futuresres.stats.dsr import DSRResult, deflated_sharpe_ratio, sharpe_ratio
from futuresres.stats.trials import Trial, TrialLog

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / "trials.jsonl"
REGISTRY = ROOT / "hypotheses.yaml"
UPSTREAM_TRIALS = ROOT.parent / "futures-research" / "trials.jsonl"

#: Round-trip cost floor, bps of notional: commission plus a one-tick spread. Only figures
#: futures-research measured or derived are filled in. Anything else must be passed
#: explicitly, because a guessed cost is the commonest way a backtest lies.
COST_FLOOR_BPS: dict[str, float] = {
    "MNQ": 0.48,  # programme_conclusion.md §1
    "NQ": 0.22,   # programme_conclusion.md §10.3
}

#: "Too good to be true" lines. Crossing one means look for a bug before anything else.
SUSPECT_SHARPE_ANNUAL = 2.5
SUSPECT_PROFIT_FACTOR = 1.8
SUSPECT_WIN_RATE = 0.65
SUSPECT_TOP5_SHARE = 0.30


class Unregistered(RuntimeError):
    pass


@dataclass(frozen=True)
class DataSpan:
    """How much data a result rests on. Required on every logged run, so no result is quoted
    without it. There is no minimum: three weeks runs exactly like sixteen years, and the line
    recorded next to the result is what tells them apart."""

    source: str      # e.g. "databento GLBX.MDP3 ohlcv-1m MNQ.c.0", plus the manifest sha256 prefix
    start: str       # first session used, ISO date
    end: str         # last session used, ISO date
    sessions: int
    bars: int

    def __post_init__(self) -> None:
        if self.sessions < 1 or self.bars < 1 or self.start > self.end:
            raise ValueError(f"implausible data span: {self}")

    @property
    def years(self) -> float:
        return self.sessions / 252

    def describe(self, trades: int | None = None) -> str:
        t = f", {trades:,} trades" if trades is not None else ""
        return (f"data: {self.source}, {self.start}..{self.end}, {self.sessions:,} sessions "
                f"(~{self.years:.1f} yr), {self.bars:,} bars{t}")


def registered_ids(path: Path = REGISTRY) -> set[str]:
    entries = yaml.safe_load(path.read_text()) or []
    return {e["id"] for e in entries if e.get("status") == "registered"}


def upstream_n(path: Path = UPSTREAM_TRIALS) -> int:
    return TrialLog(path).n_for_deflation() if path.exists() else 0


def cost_for(symbol: str, cost_bps: float | None) -> float:
    if cost_bps is not None:
        return cost_bps
    if symbol not in COST_FLOOR_BPS:
        raise ValueError(f"no measured cost floor for {symbol!r}; pass cost_bps explicitly")
    return COST_FLOOR_BPS[symbol]


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries at false-discovery rate q (step-up procedure)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


@dataclass(frozen=True)
class Evaluation:
    n_trades: int
    gross_mean_bps: float
    net_mean_bps: float
    net_mean_2x_cost_bps: float
    sharpe_per_trade: float
    sharpe_annual: float
    profit_factor: float | None
    win_rate: float
    top5_share: float | None
    dsr_own: DSRResult | None
    dsr_with_upstream: DSRResult | None
    flags: list[str] = field(default_factory=list)

    @property
    def survives_2x_cost(self) -> bool:
        return self.net_mean_2x_cost_bps > 0


def evaluate(gross_bps: Sequence[float], *, symbol: str, trades_per_year: float,
             cost_bps: float | None = None, log: TrialLog | None = None,
             n_upstream: int | None = None) -> Evaluation:
    """Per-trade gross returns in bps of notional -> net metrics, DSR and suspicion flags."""
    g = np.asarray(gross_bps, dtype=float)
    if g.size < 2:
        raise ValueError("need at least 2 trades")
    c = cost_for(symbol, cost_bps)
    net = g - c
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    pf = float(wins / losses) if losses > 0 else None
    total = net.sum()
    top5 = float(np.sort(net)[-5:].sum() / total) if total > 0 else None
    sr = sharpe_ratio(net)
    sr_annual = sr * math.sqrt(trades_per_year)

    log = TrialLog(TRIALS) if log is None else log  # an empty TrialLog is falsy (__len__)
    sharpes = [t.sharpe for t in log.read_all() if t.sharpe is not None]
    n_own = log.n_for_deflation()
    up = upstream_n() if n_upstream is None else n_upstream
    dsr_own = dsr_up = None
    if len(sharpes) >= 2:
        dsr_own = deflated_sharpe_ratio(net, sharpes, n_trials=max(n_own, len(sharpes)))
        dsr_up = deflated_sharpe_ratio(net, sharpes, n_trials=max(n_own, len(sharpes)) + up)

    flags = []
    if sr_annual > SUSPECT_SHARPE_ANNUAL:
        flags.append(f"annual Sharpe {sr_annual:.2f} > {SUSPECT_SHARPE_ANNUAL}")
    if pf is None or pf > SUSPECT_PROFIT_FACTOR:
        flags.append("no losing trades" if pf is None else f"profit factor {pf:.2f} > {SUSPECT_PROFIT_FACTOR}")
    if (net > 0).mean() > SUSPECT_WIN_RATE:
        flags.append(f"win rate {(net > 0).mean():.0%} > {SUSPECT_WIN_RATE:.0%}")
    if top5 is not None and top5 > SUSPECT_TOP5_SHARE:
        flags.append(f"top 5 trades are {top5:.0%} of PnL")
    if len(sharpes) < 2:
        flags.append("fewer than 2 logged trial Sharpes: DSR not computable yet")

    return Evaluation(
        n_trades=int(g.size), gross_mean_bps=float(g.mean()), net_mean_bps=float(net.mean()),
        net_mean_2x_cost_bps=float((g - 2 * c).mean()), sharpe_per_trade=sr,
        sharpe_annual=sr_annual, profit_factor=pf, win_rate=float((net > 0).mean()),
        top5_share=top5, dsr_own=dsr_own, dsr_with_upstream=dsr_up, flags=flags,
    )


def run_logged(hypothesis_id: str, *, params: dict[str, Any], symbol: str,
               data: DataSpan, cost_bps: float | None,
               backtest: Callable[[], Sequence[float]], note: str = "",
               log: TrialLog | None = None, registry: Path = REGISTRY) -> list[float]:
    """Run `backtest` (returning per-trade gross bps) and log it as a trial, whatever happens."""
    if hypothesis_id not in registered_ids(registry):
        raise Unregistered(
            f"{hypothesis_id!r} is not registered in {registry.name}. Register it, with its "
            f"parameters and predicted effect size, before touching the data."
        )
    log = TrialLog(TRIALS) if log is None else log  # an empty TrialLog is falsy (__len__)
    status, sharpe, n, pf, err = "error", None, None, None, ""
    try:
        gross = list(backtest())
        net = np.asarray(gross, dtype=float) - cost_for(symbol, cost_bps)
        n = int(net.size)
        if n >= 2:
            sharpe = sharpe_ratio(net)
            sharpe = sharpe if math.isfinite(sharpe) else None
            losses = -net[net < 0].sum()
            pf = float(net[net > 0].sum() / losses) if losses > 0 else None
        status = "completed"
        return gross
    except KeyboardInterrupt:
        status, err = "abandoned", "interrupted"
        raise
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        # "lc" prefix keeps our trial ids distinct from upstream's "t" ids
        log.append(Trial(
            trial_id=log.next_id(prefix="lc"), hypothesis_id=hypothesis_id, params=params, symbol=symbol,
            date_range=(data.start, data.end), status=status, sharpe=sharpe, trade_count=n,
            profit_factor=pf, note="; ".join(x for x in (data.describe(n), note, err) if x),
        ))
