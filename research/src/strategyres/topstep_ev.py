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
  funded (XFA) its OWN rules, not the evaluation's: no profit target, no consistency rule (standard
               payout path), no time limit. Balance starts at $0 with the MLL at -$2,000, trailing
               end of day and stopping at $0; after the first payout the MLL sits at $0 for good.
               After every 5 winning days (net >= $150) one withdrawal, by `payout_rule`:
                 half_balance   Topstep's published rule: up to 50% of the balance
                 above_buffer   the user's description: everything above a $2,000 buffer
               capped per request at `per_request_cap` ($2,000 is reported for a 50K with no daily
               loss limit; $5,000 is the headline figure), the trader keeping 90%. At $5,000 withdrawn
               in total the account moves to Live: counting stops there, so EV excludes anything a
               Live account later pays (conservative). Followed for at most 2 years.
               Scaling plan (2 / 3 / 5 lots by balance) is not modelled: it is in full-size contracts,
               and 2 NQ is ~$17,000/day of sigma, far above every size in the grid, so it never binds.
  sizing       the trader picks a size per phase. The evaluation needs size to finish in a month; the
               funded account does not. Because the funded account starts fresh, the two optimise
               separately: best EV = max over eval sizes of P(pass) x max over funded sizes of payout,
               minus fees. The same-size-in-both-phases figure is reported alongside.
  trader       in the evaluation only, stops for the day once up $1,500 (so no day breaks the 50%
               rule) or once the target is reached. A gamer would; it costs nothing at zero edge.

NOT MODELLED: Live account value, inactivity rules, slippage beyond the cost already inside Sharpe,
the firm refusing a payout.
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


def simulate_funded(rng, n, sigma, sharpe, mode, per_request_cap, payout_rule="half_balance"):
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
        if payout_rule == "half_balance":
            due, avail = ok & (wins >= WIN_DAYS_PER_PAYOUT) & (pnl > 0), 0.5 * pnl
        elif payout_rule == "above_buffer":
            due, avail = ok & (wins >= WIN_DAYS_PER_PAYOUT) & (pnl > BUFFER), pnl - BUFFER
        else:
            raise ValueError(payout_rule)
        w = np.where(due, np.minimum.reduce([avail, np.full(n, per_request_cap), TOTAL_CAP - paid]), 0.0)
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


FUNDED_SCEN = [("half_balance", 2_000.0), ("half_balance", 5_000.0), ("above_buffer", 5_000.0)]


def run(n: int = 6_000, seed: int = 20261008) -> dict:
    rng = np.random.default_rng(seed)
    val = validate(rng)
    if not val["ok"]:
        raise SystemExit(f"simulator fails validation: {val}")
    evals, funded = [], []
    for mode in ("eod_trailing", "day_anchored"):
        for s in SHARPES:
            for sig in SIGMAS:
                for months in (1, 6):
                    evals.append({"mll": mode, "months": months, "sharpe": s, "daily_sigma": sig,
                                  **simulate_eval(rng, n, sig, s, mode, months=months)})
                for rule, cap in FUNDED_SCEN:
                    funded.append({"mll": mode, "payout_rule": rule, "per_request_cap": cap, "sharpe": s,
                                   "daily_sigma": sig, **simulate_funded(rng, n, sig, s, mode, cap, rule)})
                print(f"{mode} S={s} sig={sig} done", flush=True)
    grid = []
    for mode in ("eod_trailing", "day_anchored"):
        for months in (1, 6):
            for rule, cap in FUNDED_SCEN:
                for s in SHARPES:
                    E = [e for e in evals if (e["mll"], e["months"], e["sharpe"]) == (mode, months, s)]
                    F = [f for f in funded if (f["mll"], f["payout_rule"], f["per_request_cap"], f["sharpe"]) == (mode, rule, cap, s)]
                    bf = max(F, key=lambda f: f["expected_payout"])
                    for e in E:
                        same = next(f for f in F if f["daily_sigma"] == e["daily_sigma"])
                        grid.append({"mll": mode, "months": months, "payout_rule": rule, "per_request_cap": cap,
                                     "sharpe": s, "daily_sigma": e["daily_sigma"], "p_pass": e["p_pass"], "fees": e["fees"],
                                     "ev_same_size": e["p_pass"] * same["expected_payout"] - e["fees"],
                                     "funded_sigma": bf["daily_sigma"], "expected_payout": bf["expected_payout"],
                                     "p_reach_live": bf["p_reach_live"],
                                     "ev": e["p_pass"] * bf["expected_payout"] - e["fees"]})
    return {"validation": val, "n_paths": n, "evals": evals, "funded": funded, "grid": grid}


SCENARIOS = [
    ("eod_trailing", 1, "half_balance", 2_000.0, "Topstep rules as published · 1 month · 50% of balance, $2k per request"),
    ("eod_trailing", 1, "half_balance", 5_000.0, "Same, with a $5k per-request cap"),
    ("eod_trailing", 1, "above_buffer", 5_000.0, "Payout = everything above a $2k buffer (your description)"),
    ("eod_trailing", 6, "half_balance", 2_000.0, "Evaluation renewed monthly, up to 6 months"),
    ("day_anchored", 1, "half_balance", 2_000.0, "Loss floor reset to each day's open (literal reading)"),
]


def render(r: dict) -> str:
    v = r["validation"]
    w = ["# Topstep 50K — EV per evaluation", "",
         "Generated by `python -m strategyres.topstep_ev`. A computation, no trial. Terms and readings in "
         "the module docstring. Sharpe is annual and net of cost; σ is the strategy's daily $ volatility "
         "in the evaluation. The funded account is sized separately, at whatever σ maximises its payout.", "",
         f"**Validated first:** P(pass) {v['p_pass']:.3f} against the closed form {v['closed_form']:.3f} "
         f"(tolerance {v['tolerance']:.3f}).", ""]
    for mode, months, rule, cap, title in SCENARIOS:
        w += [f"## {title}", "", "| Sharpe | " + " | ".join(f"eval σ ${s}" for s in SIGMAS) + " | best | funded σ | same size both phases |",
              "|---|" + "---|" * (len(SIGMAS) + 3)]
        for s in SHARPES:
            row = [g for g in r["grid"] if (g["mll"], g["months"], g["payout_rule"], g["per_request_cap"], g["sharpe"]) == (mode, months, rule, cap, s)]
            best = max(row, key=lambda g: g["ev"]); same = max(row, key=lambda g: g["ev_same_size"])
            w.append(f"| {s} | " + " | ".join(f"{g['ev']:+,.0f} ({g['p_pass']:.0%})" for g in row)
                     + f" | **{best['ev']:+,.0f}** at ${best['daily_sigma']} | ${best['funded_sigma']} "
                     f"| {same['ev_same_size']:+,.0f} at ${same['daily_sigma']} |")
        w += ["", "Cells: EV in USD (P(pass)). EV = P(pass) × expected funded payout (90%, to the $5k Live cap) − fees.", ""]
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
