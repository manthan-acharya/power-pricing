import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_COLORS, NODE_LABELS, apply

RAW = ROOT / "data" / "raw"
HOLDOUT_RAW = ROOT / "data" / "raw_holdout"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "writeup" / "figures"

# Chronological split boundaries (Pacific time); never shuffled, so the SDE and
# control policy only ever see the past at train time.
SPLITS = {
    "train": ("2025-01-01", "2026-02-01"),
    "val": ("2026-02-01", "2026-04-01"),
    "test": ("2026-04-01", "2026-07-01"),
}


def load_raw(raw=RAW):
    frames = [pd.read_csv(f) for f in sorted(raw.glob("caiso_dam_lmp_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["Interval Start"] = pd.to_datetime(df["Interval Start"], utc=True)
    df = df.drop_duplicates(subset=["Interval Start", "Location"])
    return df


def to_wide(df, col):
    wide = df.pivot(index="Interval Start", columns="Location", values=col)
    wide = wide.sort_index()
    full = pd.date_range(wide.index[0], wide.index[-1], freq="1h", tz="UTC")
    gaps = len(full) - len(wide)
    wide = wide.reindex(full)
    filled = int(wide.isna().sum().sum())
    wide = wide.interpolate(limit=3)
    if wide.isna().any().any():
        raise RuntimeError("gap longer than 3h found, inspect raw data")
    return wide, gaps, filled


def summary(prices):
    rows = {}
    for node in prices.columns:
        x = prices[node]
        d = x.diff().dropna()
        rows[NODE_LABELS[node]] = {
            "mean": x.mean(),
            "std": x.std(),
            "skew": stats.skew(x),
            "excess_kurtosis": stats.kurtosis(x),
            "pct_negative": 100 * (x < 0).mean(),
            "min": x.min(),
            "max": x.max(),
            "diff_excess_kurtosis": stats.kurtosis(d),
            "max_abs_hourly_move": d.abs().max(),
        }
    return pd.DataFrame(rows).T.round(3)


def divergence_window(prices, days=10):
    spread = (prices.iloc[:, 0] - prices.iloc[:, 1]).abs()
    roll = spread.rolling(24 * days).mean()
    end = roll.idxmax()
    return end - pd.Timedelta(days=days), end


def plot_prices(prices):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    local = prices.tz_convert("US/Pacific")
    for node in local.columns:
        axes[0].plot(local.index, local[node], lw=0.4, alpha=0.8,
                     color=NODE_COLORS[node], label=NODE_LABELS[node])
    axes[0].set_ylabel("DAM LMP ($/MWh)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].set_title("CAISO day-ahead hourly LMPs, Jan 2025 - Jun 2026")

    t0, t1 = divergence_window(prices)
    zoom = local.loc[t0:t1]
    for node in zoom.columns:
        axes[1].plot(zoom.index, zoom[node], lw=1.0,
                     color=NODE_COLORS[node], label=NODE_LABELS[node])
    axes[1].set_ylabel("DAM LMP ($/MWh)")
    axes[1].set_title(f"Highest-divergence window ({t0.date()} to {t1.date()})")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "hub_prices.pdf")
    print(f"divergence window: {t0} .. {t1}")


def main():
    apply()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    lmp, gaps, filled = to_wide(df, "LMP")
    cong, _, _ = to_wide(df, "Congestion")
    energy, _, _ = to_wide(df, "Energy")
    print(f"{len(lmp)} hours, {gaps} missing hours reindexed, {filled} values interpolated")

    lmp.to_csv(OUT / "lmp_hourly.csv")
    cong.to_csv(OUT / "congestion_hourly.csv")
    energy.to_csv(OUT / "energy_hourly.csv")

    bounds = {k: [str(pd.Timestamp(a, tz="US/Pacific").tz_convert("UTC")),
                  str(pd.Timestamp(b, tz="US/Pacific").tz_convert("UTC"))]
              for k, (a, b) in SPLITS.items()}
    pd.Series({k: f"{v[0]} .. {v[1]}" for k, v in bounds.items()}).to_csv(
        OUT / "splits.csv", header=False)
    for k, (a, b) in bounds.items():
        # .loc slices are closed at both ends, so drop the boundary hour that
        # belongs to the next split
        n = len(lmp.loc[a:b]) - (1 if k != "test" else 0)
        print(f"{k}: {SPLITS[k][0]} .. {SPLITS[k][1]} ({n} hours)")

    stats_df = summary(lmp)
    stats_df.to_csv(OUT / "summary_stats.csv")
    print(stats_df.to_string())

    spread = lmp[lmp.columns[0]] - lmp[lmp.columns[1]]
    print(f"NP15-SP15 spread: mean {spread.mean():.2f}, std {spread.std():.2f}, "
          f"pct |spread|>5: {100 * (spread.abs() > 5).mean():.1f}%")

    plot_prices(lmp)


# Kept in its own file so every artifact built from lmp_hourly.csv is unchanged.
def holdout():
    lmp, gaps, filled = to_wide(load_raw(HOLDOUT_RAW), "LMP")
    last = pd.read_csv(OUT / "lmp_hourly.csv", index_col=0, parse_dates=True).index[-1]
    assert lmp.index[0] == last + pd.Timedelta(hours=1), "holdout does not continue the series"
    lmp.to_csv(OUT / "lmp_hourly_holdout.csv")
    print(f"holdout: {len(lmp)} hours, {gaps} missing hours reindexed, "
          f"{filled} values interpolated")


if __name__ == "__main__":
    holdout() if "--holdout" in sys.argv else main()
