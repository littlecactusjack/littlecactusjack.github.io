"""Expected value of one Topstep 50K evaluation, by edge (annual Sharpe) and size (daily $ sigma).

A computation, not a trial: it searches nothing about the market. Adapted from futures-research's
`reporting/w_prop_ev.py` (decisions.md 73), whose return model it reuses unchanged: 13 marks per
session, Student-t (nu = 5) innovations, drift set by the annual Sharpe. Sharpe here is NET of cost.

    python -m strategyres.topstep_ev            # writes reports/topstep_ev.md and .json

THE ACCOUNT, as given by the user 2026-10-08; topstep.com was unreachable from the sandbox, so the
terms are the user's, cross-read against third-party summaries. Ambiguities are run both ways.

  evaluation   +$3,000 target. $90 per month. No daily loss limit. Minimum 2 trading days.
               Consistency: best day <= 50% of total profit at the time of passing.
               Time limit 1 month (21 sessions); `renew` re-buys another month at $90 instead.
  MLL          $2,000. Two readings:
                 eod_trailing   Topstep's published rule: floor = highest END-OF-DAY balance - $2,000,
                                frozen at the start balance once it gets there; breached intraday.
                 day_anchored   the user's wording read literally: floor = that day's opening balance
                                - $2,000, recomputed every day (in effect a $2,000 per-day limit).
  funded (XFA) starts at +$0 with the same MLL. Standard payout path, not the consistency path:
               after every 5 winning days (net >= $150), withdraw everything above a $2,000 buffer,
               up to `per_request_cap`; the trader keeps 90%; the MLL then sits at $0 for good.
               At $5,000 withdrawn in total the account moves to Live: counting stops there, so EV
               excludes anything a Live account later pays (conservative). Followed for at most 2 years.
  trader       stops for the day once the day is up $1,500 (so no day breaks the 50% rule) or the
               target is reached. A gamer would; it costs nothing in expectation at zero edge.

NOT MODELLED: Live account value, scaling plans, per-request 50%-of-balance caps beyond the buffer,
inactivity rules, slippage beyond the cost already inside Sharpe, the firm refusing a payout.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from futuresres.reporting.w_prop_ev import MARKS, _steps

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "reports" / "topstep_ev.md"
OUT_JSON = ROOT / "reports" / "topstep_ev.json"

TARGET, MLL, FEE = 3_000.0, 2_000.0, 90.0
MONTH_DAYS, MIN_DAYS, CONSISTENCY = 21, 2, 0.50
DAY_STOP = CONSISTENCY * TARGET
WIN_DAY, WIN_DAYS_PER_PAYOUT, BUFFER = 150.0, 5, 2_000.0
SPLIT, TOTAL_CAP, FUNDED_DAYS = 0.90, 5_000.0, 504

SHARPES = (0.0, 0.4, 0.7, 1.0, 1.5, 2.0)
SIGMAS = (150, 250, 400, 600, 900, 1200)


def _floor(mode: str, eod_peak, day_open):
    if mode == "eod_trailing":
        return np.minimum(eod_peak - MLL, 0.0)
    if mode == "day_anchored":
        return day_open - MLL
    raise ValueError(mode)


def simulate_eval(rng, n, sigma, sharpe, mode, months=1, rules=True, intraday_peak=False):
    """Returns P(pass) and expected fees paid. P&L is measured from the start balance."""
    pnl = np.zeros(n); eod_peak = np.zeros(n); alive = np.ones(n, bool); passed = np.zeros(n, bool)
    best = np.zeros(n); days = np.zeros(n); months_paid = np.ones(n)
    for d in range(months * MONTH_DAYS):
        if d and d % MONTH_DAYS == 0:
            months_paid += alive & ~passed
        steps = _steps(rng, n, 1, sigma, sharpe)[:, 0, :]
        day_open = pnl.copy(); stopped = ~alive | passed; peak = eod_peak.copy()
        for m in range(MARKS):
            act = ~stopped
            pnl = np.where(act, pnl + steps[:, m], pnl)
            if intraday_peak:
                peak = np.maximum(peak, pnl)
            broke = act & (pnl <= _floor(mode, peak, day_open))
            alive &= ~broke
            stopped |= broke
            if rules:
                stopped |= act & ((pnl - day_open >= DAY_STOP) | (pnl >= TARGET))
            elif intraday_peak:
                hit = act & ~broke & (pnl >= TARGET)
                passed |= hit; stopped |= hit
        traded = alive & ~passed
        days += traded
        best = np.where(traded, np.maximum(best, pnl - day_open), best)
        # in validation mode the floor trails the intraday high, which must carry across days
        eod_peak = np.where(alive, np.maximum(peak if intraday_peak else eod_peak, pnl), eod_peak)
        if rules:
            passed |= traded & (pnl >= TARGET) & (best <= CONSISTENCY * pnl) & (days >= MIN_DAYS)
        if not (alive & ~passed).any():
            break
    return {"p_pass": float(passed.mean()), "fees": float(FEE * months_paid.mean())}


def simulate_funded(rng, n, sigma, sharpe, mode, per_request_cap):
    pnl = np.zeros(n); eod_peak = np.zeros(n); alive = np.ones(n, bool); paid = np.zeros(n)
    wins = np.zeros(n); floor_zero = np.zeros(n, bool)
    for _ in range(FUNDED_DAYS):
        active = alive & (paid < TOTAL_CAP)
        if not active.any():
            break
        steps = _steps(rng, n, 1, sigma, sharpe)[:, 0, :]
        day_open = pnl.copy(); stopped = ~active
        for m in range(MARKS):
            act = ~stopped
            pnl = np.where(act, pnl + steps[:, m], pnl)
            floor = np.where(floor_zero, 0.0, _floor(mode, eod_peak, day_open))
            broke = act & (pnl <= floor)
            alive &= ~broke
            stopped |= broke
        ok = active & alive
        eod_peak = np.where(ok, np.maximum(eod_peak, pnl), eod_peak)
        wins += ok & (pnl - day_open >= WIN_DAY)
        due = ok & (wins >= WIN_DAYS_PER_PAYOUT) & (pnl > BUFFER)
        w = np.where(due, np.minimum.reduce([pnl - BUFFER, np.full(n, per_request_cap), TOTAL_CAP - paid]), 0.0)
        paid += w; pnl -= w
        wins = np.where(due, 0, wins)
        floor_zero |= due
    return {"expected_payout": float(SPLIT * paid.mean()), "p_reach_live": float((paid >= TOTAL_CAP).mean())}


def validate(rng) -> dict:
    """Intraday-trailing floor, no rules, long horizon: P(pass) -> exp(-1) * 2/3 = 0.245 (w_prop_ev's check)."""
    n = 40_000
    r = simulate_eval(rng, n, 400.0, 0.0, "eod_trailing", months=12, rules=False, intraday_peak=True)
    target = math.exp(-1) * 2 / 3
    tol = 2 * math.sqrt(target * (1 - target) / n) + 0.02
    return {"p_pass": r["p_pass"], "closed_form": target, "tolerance": tol, "ok": abs(r["p_pass"] - target) <= tol}


def run(n: int = 6_000, seed: int = 20261008) -> dict:
    rng = np.random.default_rng(seed)
    val = validate(rng)
    if not val["ok"]:
        raise SystemExit(f"simulator fails validation: {val}")
    grid = []
    for mode in ("eod_trailing", "day_anchored"):
        for months in (1, 6):
            for cap in (5_000.0, 2_000.0):
                if months == 6 and cap == 2_000.0:
                    continue
                for s in SHARPES:
                    for sig in SIGMAS:
                        e = simulate_eval(rng, n, sig, s, mode, months=months)
                        f = simulate_funded(rng, n, sig, s, mode, cap)
                        ev = e["p_pass"] * f["expected_payout"] - e["fees"]
                        grid.append({"mll": mode, "months": months, "per_request_cap": cap, "sharpe": s,
                                     "daily_sigma": sig, **e, **f, "ev": ev})
                        print(f"{mode} m={months} cap={cap:.0f} S={s} sig={sig} pass {e['p_pass']:.3f} "
                              f"payout {f['expected_payout']:6.0f} EV {ev:+6.0f}", flush=True)
    return {"validation": val, "n_paths": n, "grid": grid}


def render(r: dict) -> str:
    v = r["validation"]
    w = ["# Topstep 50K — EV per evaluation", "",
         "Generated by `python -m strategyres.topstep_ev`. A computation, no trial. Terms and readings in "
         "the module docstring. Sharpe is annual and net of cost; σ is the strategy's daily $ volatility.", "",
         f"**Validated first:** P(pass) {v['p_pass']:.3f} against the closed form {v['closed_form']:.3f} "
         f"(tolerance {v['tolerance']:.3f}).", ""]
    sections = [("eod_trailing", 1, 5000.0, "MLL end-of-day trailing (Topstep's rule) · 1 month · $5k per request"),
                ("eod_trailing", 1, 2000.0, "MLL end-of-day trailing · 1 month · $2k per request"),
                ("eod_trailing", 6, 5000.0, "MLL end-of-day trailing · renew monthly, up to 6 months · $5k per request"),
                ("day_anchored", 1, 5000.0, "MLL anchored to each day's open (literal reading) · 1 month · $5k per request")]
    for mode, months, cap, title in sections:
        w += [f"## {title}", "", "| Sharpe | " + " | ".join(f"σ ${s}" for s in SIGMAS) + " | best |",
              "|---|" + "---|" * (len(SIGMAS) + 1)]
        for s in SHARPES:
            row = [g for g in r["grid"] if (g["mll"], g["months"], g["per_request_cap"], g["sharpe"]) == (mode, months, cap, s)]
            best = max(row, key=lambda g: g["ev"])
            w.append(f"| {s} | " + " | ".join(f"{g['ev']:+,.0f} ({g['p_pass']:.0%})" for g in row)
                     + f" | **{best['ev']:+,.0f}** at ${best['daily_sigma']} |")
        w += ["", "Cells: EV in USD (P(pass)). EV = P(pass) × expected payout (90%, to the $5k Live cap) − fees.", ""]
    return "\n".join(w)


def main() -> int:
    r = run()
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(r, indent=1) + "\n")
    OUT_MD.write_text(render(r) + "\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
