# littlecactusjack.github.io

## research/

Our own CISD/POI strategy research, run under futures-research's rules: pre-registration in
`hypotheses.yaml`, every run logged to a hash-chained `trials.jsonl`, DSR deflated by our trials
plus futures-research's. Start with [`research/HANDOFF.md`](research/HANDOFF.md).

## futures-research/

Imported with full commit history from
[coltontr419-droid/futures-research](https://github.com/coltontr419-droid/futures-research)
(git subtree, upstream `master` @ `80be417`). That repo has no license file; credit to its author.

MNQ / MGC intraday futures research, run to completion: 6 series, 66 candidates, 35 registered,
770 pre-registered trials, **0 promoted**. Start with
[`reports/programme_conclusion.md`](futures-research/reports/programme_conclusion.md).

Key carry-forward facts:
- Round-trip cost floor on MNQ (0.48 bps) exceeds the largest effect measured anywhere (R01, +0.452 bps),
  so trading more often or bigger cannot help.
- Reopening needs new **information** (order-flow / book data), **permission** (long/short or overnight
  holds), **cost**, or **time** — not a seventh series on the same 1-minute bars.
- `trials.jsonl` (N = 760, SR* 0.1368) is the multiple-testing budget; any new work appends to it.

Raw Databento bars (`data/`, ~548 MB) and `.env` credentials were never in the upstream repo and are
not here. To run: `cd futures-research && pip install -e ".[dev]" && pytest -q`
(450 pass, 12 skip without the raw data).

Pull later upstream changes: `git subtree pull --prefix=futures-research https://github.com/coltontr419-droid/futures-research master`
