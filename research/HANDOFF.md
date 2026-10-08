# Handoff: CISD/POI prop-firm strategy

Integrated 2026-10-05 from a Cowork session. A Databento API key was pasted into that chat;
it is deliberately **not** recorded here (this repo is public) and must be rotated before use.

## The system
- **Signal:** Pine Script v5. CISD (Change In State of Delivery) entries, filtered to
  higher-timeframe POI (point-of-interest) zones with a manual direction input; position size
  from account equity; JSON alert payloads. Source goes in `pine/` (not yet added).
- **Execution:** TradingView Pro webhooks → Flask receiver (ngrok) → Tradovate REST API.
  Demo account first, then live. Not in this repo yet.
- **Instruments:** NQ, ES, MNQ, MES on CME Globex (`GLBX.MDP3`).

## Data plan
Databento, `ohlcv-1m`, continuous front month (`NQ.c.0`, `ES.c.0`, `MNQ.c.0`, `MES.c.0`),
the full history Databento holds (2010 on for NQ/ES; MNQ/MES from May 2019). `python -m strategyres.fetch` prices the pull and downloads only with `--yes`.
Fallbacks: FirstRate Data (~$100/yr, up to 18 years of NQ), or a free Kaggle NQ 1-min 2022–2025 set.

Sandboxes with restricted egress (the Cowork one, and this Claude Code cloud environment
unless `hist.databento.com` is allowed) return 403 for Databento. A local machine does not.

## Methodology, from futures-research
- Every run, abandoned or crashed included, is appended to `trials.jsonl` (hash-chained,
  never edited). Enforced by `strategyres.harness.run_logged`.
- DSR (Bailey & López de Prado) is the headline metric, not raw Sharpe. Reported at our N and
  at our N plus futures-research's N, because M01 overlaps its L02/L03/L07.
- PBO via CSCV over every configuration tried; reject above 0.5.
- Benjamini–Hochberg at 5% FDR across any multi-cell scan.
- Detection floor: a null below the floor is absence of evidence, and is reported as such.
- Cost floor: MNQ 0.48 bps, NQ 0.22 bps round trip. A strategy that dies at 2× cost is not one.
- Suspect a bug before celebrating: annual Sharpe > 2.5, profit factor > 1.8, win rate > 65%,
  or top 5 trades > 30% of PnL.
- Re-read `futures-research/reports/decisions.md` §38 (L07) before trusting a clean separation.

## Prop-firm angle (unresolved)
Two separate projects with different honesty requirements: (a) a strategy with real edge;
(b) evaluations as a variance game. Passing an evaluation is not evidence of edge.
futures-research measured that with zero edge the chance of reaching the +4% lock before a 4%
trailing drawdown is about 36.8% at any size (programme_conclusion.md §4).

Regulatory note as of Oct 2026, to re-check: the CFTC's My Forex Funds case collapsed in 2025;
some firms (Topstep, Tradeify) are moving inside CFTC/NFA registration voluntarily. Treat
counterparty risk (withdrawal freezes, payout denials) as real.

## Next steps
1. Add the Pine Script to `pine/`.
2. Rotate the Databento key; pull data (`strategyres.fetch`).
3. Harness: built, tested (`pytest`), before any backtest.
4. Write M01 in `hypotheses.yaml` to registration standard, then port the Pine logic to Python
   and verify the port bar-for-bar against TradingView's trade list.
5. Only after a result survives: sim/demo routing through the existing Flask/Tradovate stack.
