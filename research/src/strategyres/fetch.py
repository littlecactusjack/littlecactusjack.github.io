"""Download 1-minute bars from Databento. Quotes the cost first and spends nothing without --yes.

    export DATABENTO_API_KEY=...        # or put it in research/.env; never commit it
    python -m strategyres.fetch                    # price the default pull, download nothing
    python -m strategyres.fetch --yes              # download it

Default pull: GLBX.MDP3, continuous front month (NQ, ES, MNQ, MES), ohlcv-1m, from the earliest
date Databento holds for the dataset (each symbol simply has no bars before it listed; MNQ and MES
start May 2019). Pass --start to take less. Each symbol lands in research/data/raw/<symbol>.dbn.zst with a manifest entry
carrying its sha256, so a later run can prove which bytes a result was computed from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOLS = ["NQ.c.0", "ES.c.0", "MNQ.c.0", "MES.c.0"]


def api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("DATABENTO_API_KEY is not set (environment or research/.env)")
    return key


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--start", default=None, help="ISO date; default is the dataset's earliest")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--yes", action="store_true", help="actually download (costs money)")
    a = ap.parse_args(argv)

    import databento as db

    client = db.Historical(api_key())
    if a.start is None:
        a.start = str(client.metadata.get_dataset_range(dataset=DATASET)["start"])[:10]
    common = dict(dataset=DATASET, schema=a.schema, stype_in="continuous", start=a.start, end=a.end)
    quotes = {s: client.metadata.get_cost(symbols=[s], **common) for s in a.symbols}
    for s, usd in quotes.items():
        print(f"{s:10} {a.schema} {a.start}..{a.end}  ${usd:,.2f}")
    print(f"{'total':10} ${sum(quotes.values()):,.2f}")
    if not a.yes:
        print("Quote only. Re-run with --yes to download.")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for s in a.symbols:
        out = RAW / f"{s}.{a.schema}.{a.start}_{a.end}.dbn.zst"
        client.timeseries.get_range(symbols=[s], path=out, **common)
        manifest[out.name] = {"symbol": s, **{k: v for k, v in common.items()},
                              "bytes": out.stat().st_size, "sha256": sha256(out),
                              "quoted_usd": quotes[s]}
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
