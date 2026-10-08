# research/ — working agreements

Our own project, run under futures-research's rules. `../futures-research/` is a read-only
mirror synced daily from upstream: never edit it (edits conflict with the sync).

- **The repo is public.** Never stage `.env`, API keys, or anything under `data/`. The
  Databento key is read from `DATABENTO_API_KEY`. Never write a key into a tracked file.
- **Register before running.** No backtest code touches data for a hypothesis that is not
  `registered` in `hypotheses.yaml` with every required field filled.
- **Every run goes through `strategyres.harness.run_logged`.** `trials.jsonl` is append-only;
  never edit, reorder or delete a line. Errors and abandoned runs count toward N.
- **Report DSR at our N plus upstream N** for anything overlapping futures-research's tested
  mechanisms. Report net of cost and at 2× cost. Any suspicion flag means hunt for a bug first.
- **Every result states its data**: source, date range, sessions (and years), bars. `run_logged`
  requires a `DataSpan`; reports and the site must carry the same line next to each number.
- Spending money (Databento downloads) needs the user's explicit go-ahead each time.
- Setup: `pip install -e ../futures-research -e ".[dev,data]"`, then `pytest -q`.
