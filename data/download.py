import logging
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"
HOLDOUT_RAW = Path(__file__).resolve().parent / "raw_holdout"

NODES = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]
MARKET = "DAY_AHEAD_HOURLY"
START = "2025-01-01"
END = "2026-07-01"
HOLDOUT = ("2026-07-01", "2026-10-01")


def month_starts(start, end):
    return pd.date_range(start, end, freq="MS")


def fetch_month(iso, t0, t1, path, attempts=4):
    for k in range(attempts):
        try:
            df = iso.get_lmp(
                start=t0.strftime("%Y-%m-%d"),
                end=t1.strftime("%Y-%m-%d"),
                market=MARKET,
                locations=NODES,
                sleep=3,
            )
            if df.empty:
                raise RuntimeError("empty frame returned")
            df.to_csv(path, index=False)
            return df
        except Exception as exc:
            print(f"  attempt {k + 1} failed: {exc}")
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"could not fetch {t0.date()}..{t1.date()}")


def main(start=START, end=END, raw=RAW):
    logging.getLogger("gridstatus").setLevel(logging.WARNING)
    import gridstatus

    raw.mkdir(parents=True, exist_ok=True)
    iso = gridstatus.CAISO()

    edges = month_starts(start, end)
    for t0, t1 in zip(edges[:-1], edges[1:]):
        path = raw / f"caiso_dam_lmp_{t0:%Y%m}.csv"
        if path.exists() and len(pd.read_csv(path)) > 0:
            print(f"{t0:%Y-%m} cached")
            continue
        print(f"{t0:%Y-%m} fetching")
        df = fetch_month(iso, t0, t1, path)
        print(f"{t0:%Y-%m} -> {len(df)} rows")

    files = sorted(raw.glob("caiso_dam_lmp_*.csv"))
    print(f"\n{len(files)} monthly files, "
          f"{sum(len(pd.read_csv(f)) for f in files)} rows total")


if __name__ == "__main__":
    sys.exit(main(*HOLDOUT, HOLDOUT_RAW) if "--holdout" in sys.argv else main())
