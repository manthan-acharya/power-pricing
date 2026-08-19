import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.baselines import (hour_of_day_profile, perfect_foresight,
                                profile_dispatch, soc_step)
from backtest.engine import (Policy, daily_totals, dispatch_from_q,
                             run_policy_on_prices)
from control.bsde import CAP, C_DEG, ETA, HUB, P_MAX, day_starts
from control.bsde_pi import PiPolicy
from control.grid_dp import run_dp_policy, solve_days
from sde.model import Scaler, extended_series, load_frames

OUT = ROOT / "backtest"
CACHE = OUT / "cache"
C_DEG_USD = C_DEG * 1000.0
S0 = 50.0
PI_RANGE = (0.0, 60.0)
QUARTERS = {"development": "test", "holdout": "holdout"}
SIMPLE = ["fixed_schedule", "mpc_yesterday", "mpc_weekly", "day_ahead_yesterday",
          "day_ahead_weekly"]
MONTHS = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}


# eta_d * trailing-week median of SP15 at 00:00-05:59 local, minus degradation
def terminal_values(index, prices, starts):
    hour = index.tz_convert("US/Pacific").hour
    raw = np.array([ETA * np.median(prices[d - 168:d][hour[d - 168:d] < 6])
                    - C_DEG_USD for d in starts])
    return np.clip(raw, *PI_RANGE), int(((raw < PI_RANGE[0]) | (raw > PI_RANGE[1])).sum())


# LP over the remaining hours of a day, end-of-day SOC valued at pi ($/MWh)
def plan(prices, s, pi):
    n = len(prices)
    L = np.tril(np.ones((n, n)))
    A = np.block([[-L / ETA, ETA * L], [L / ETA, -ETA * L]])
    b = np.concatenate([np.full(n, CAP - s), np.full(n, s)])
    c = np.concatenate([-(prices - C_DEG_USD - pi / ETA),
                        prices + C_DEG_USD - pi * ETA])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0.0, P_MAX)] * (2 * n),
                  method="highs")
    assert res.status == 0, res.message
    return res.x[:n], res.x[n:]


# re-plans every hour: realized price now, forecast for the rest of the day
def run_mpc(prices, starts, pis, forecast, s0=S0):
    s, pnl = s0, []
    for d, pi in zip(starts, pis):
        fc = forecast(d)
        for k in range(24):
            seq = np.concatenate([[prices[d + k]], fc[k + 1:]])
            ud, uc = plan(seq, s, pi)
            s, u = soc_step(s, ud[0] - uc[0])
            pnl.append(prices[d + k] * u - C_DEG_USD * abs(u))
    return np.array(pnl)


def run_pi_policy(policy, prices, x, E, v, off, starts, pis, s0=S0):
    s, pnl = s0, []
    for d, pi in zip(starts, pis):
        off0, E0, v0 = (torch.tensor(a[d:d + 1]) for a in (off, E, v))
        for k in range(24):
            q = policy.act(k, s, x[d + k], off0, E0, v0, pi)
            u = dispatch_from_q(prices[d + k], s, q)
            s = s - u / ETA if u >= 0 else s - ETA * u
            pnl.append(prices[d + k] * u - C_DEG_USD * abs(u))
    return np.array(pnl)


# commits at midnight to the day's LP plan against the forecast
def run_day_ahead(prices, starts, pis, forecast, s0=S0):
    s, pnl = s0, []
    for d, pi in zip(starts, pis):
        ud, uc = plan(forecast(d), s, pi)
        for k in range(24):
            s, u = soc_step(s, ud[k] - uc[k])
            pnl.append(prices[d + k] * u - C_DEG_USD * abs(u))
    return np.array(pnl)


def grid_dp_pnl(name, prices, starts, pis):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"grid_dp_{name}.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["starts"], starts) and np.allclose(z["pis"], pis):
            return run_dp_policy(prices, starts, list(z["conts"])), float(z["secs"].mean())
    conts, secs = solve_days(starts, pis / 1000.0, extended=True)
    np.savez(path, starts=starts, pis=pis, conts=np.array(conts), secs=np.array(secs))
    return run_dp_policy(prices, starts, conts), float(np.mean(secs))


def edge_ci(a, b, pf, block=5, n_boot=20000, seed=1):
    n = len(a)
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(n_boot, n // block))
    idx = ((st[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)
    draws = 100 * (a[idx].sum(1) - b[idx].sum(1)) / pf[idx].sum(1)
    return (100 * (a.sum() - b.sum()) / pf.sum(),
            np.percentile(draws, 2.5), np.percentile(draws, 97.5))


def evaluate(name, index, prices, x, E, v, off, cheap, dear):
    starts = day_starts(index, QUARTERS[name])
    n = len(starts)
    pis, n_clip = terminal_values(index, prices, starts)
    w0, w1 = starts[0], starts[-1] + 24
    local = index[w0:w1].tz_convert("US/Pacific")
    print(f"\n[{name}] {local[0].date()} .. {local[-1].date()} ({n} days); "
          f"pi_d ${pis.min():.1f}-{pis.max():.1f}/MWh, {n_clip} clipped", flush=True)

    hourly = {}
    hourly["deep_bsde"] = run_pi_policy(PiPolicy(ROOT), prices, x, E, v, off,
                                        starts, pis)
    hourly["grid_dp"], sec = grid_dp_pnl(name, prices, starts, pis)
    print(f"  grid DP: {sec:.0f} s per day instance", flush=True)
    hourly["fixed_schedule"], _ = profile_dispatch(prices[w0:w1], local.hour,
                                                   cheap, dear, s0=S0)
    yesterday = lambda d: prices[d - 24:d]
    weekly = lambda d: np.mean([prices[d - 24 * j:d - 24 * j + 24]
                                for j in range(1, 8)], axis=0)
    hourly["mpc_yesterday"] = run_mpc(prices, starts, pis, yesterday)
    hourly["mpc_weekly"] = run_mpc(prices, starts, pis, weekly)
    hourly["day_ahead_yesterday"] = run_day_ahead(prices, starts, pis, yesterday)
    hourly["day_ahead_weekly"] = run_day_ahead(prices, starts, pis, weekly)
    hourly["bsde_fixed_salvage"], _, _ = run_policy_on_prices(
        Policy(ROOT), prices, x, E, v, off, starts)
    hourly["pf_daily"] = run_day_ahead(prices, starts, pis, lambda d: prices[d:d + 24])
    hourly["pf"], _, _ = perfect_foresight(prices[w0:w1], s0=S0, pi_term=0.0)

    days = local[::24].date
    daily = pd.DataFrame({k: daily_totals(p, n) for k, p in hourly.items()},
                         index=pd.Index(days, name="day"))
    pf = daily["pf"].values
    res = pd.DataFrame({"quarter": name, "strategy": daily.columns,
                        "revenue": daily.sum().values,
                        "pct_of_pf": 100 * daily.sum().values / pf.sum()})
    best = max(SIMPLE, key=lambda k: daily[k].sum())

    month = np.array([MONTHS[d.month] for d in days])
    rng_day = np.array([prices[d:d + 24].max() - prices[d:d + 24].min() for d in starts])
    tercile = np.array(["low", "mid", "high"])[np.searchsorted(
        np.quantile(rng_day, [1 / 3, 2 / 3]), rng_day, side="right")]
    neg = np.where([prices[d:d + 24].min() < 0 for d in starts],
                   "negative-price day", "no negative price")

    edges, regimes = [], []
    for strat in ("deep_bsde", "grid_dp"):
        for comp in SIMPLE + (["deep_bsde"] if strat == "grid_dp" else []):
            e, lo, hi = edge_ci(daily[strat].values, daily[comp].values, pf)
            by_month = {m: 100 * (daily[strat][month == m].sum() - daily[comp][month == m].sum())
                        / pf[month == m].sum() for m in dict.fromkeys(month)}
            edges.append({"quarter": name, "strategy": strat, "comparator": comp,
                          "comparator_is_best_simple": comp == best,
                          "edge_pp": e, "ci_lo": lo, "ci_hi": hi,
                          "months_positive": sum(v > 0 for v in by_month.values()),
                          **{f"edge_{m}": v for m, v in by_month.items()}})
        for kind, labels in (("month", month), ("range_tercile", tercile),
                             ("negative_prices", neg)):
            for lab in dict.fromkeys(labels):
                sel = labels == lab
                regimes.append({"quarter": name, "strategy": strat, "slice": kind,
                                "value": lab, "n_days": int(sel.sum()),
                                "edge_vs_best_simple_pp": 100 * (
                                    daily[strat][sel].sum() - daily[best][sel].sum())
                                / pf[sel].sum()})
    for r in res.itertuples():
        print(f"  {r.strategy:20s} ${r.revenue:10,.0f}  {r.pct_of_pf:5.1f}% of PF")
    print(f"  best simple baseline: {best}")
    return daily, res, pd.DataFrame(edges), pd.DataFrame(regimes)


def verdict(edges):
    rows = []
    for strat in ("deep_bsde", "grid_dp"):
        get = lambda q: edges[(edges.quarter == q) & (edges.strategy == strat)
                              & edges.comparator_is_best_simple].iloc[0]
        h = get("holdout")
        crit = {"holdout_ci_excludes_zero": h.ci_lo > 0,
                "positive_in_2_of_3_months": h.months_positive >= 2,
                "same_sign_on_development": np.sign(get("development").edge_pp)
                == np.sign(h.edge_pp)}
        rows.append({"strategy": strat, **crit, "works": all(crit.values())})
        print(f"{strat}: holdout edge {h.edge_pp:+.2f} pp "
              f"[{h.ci_lo:+.2f}, {h.ci_hi:+.2f}] over {h.comparator}; "
              + ", ".join(f"{k} {v}" for k, v in crit.items())
              + f" -> {'WORKS' if all(crit.values()) else 'does not meet the rule'}")
    return pd.DataFrame(rows)


def main(quarters):
    torch.set_num_threads(1)
    ck = torch.load(ROOT / "sde/checkpoint.pt", weights_only=False)
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    index, x, E, v, off = extended_series(ROOT, scaler)
    lmp, _, splits = load_frames(ROOT)
    hold = ROOT / "data/processed/lmp_hourly_holdout.csv"
    if hold.exists():
        lmp = pd.concat([lmp, pd.read_csv(hold, index_col=0, parse_dates=True)])
    assert lmp.index.equals(index)
    prices = lmp.values[:, HUB].astype(np.float64)
    cheap, dear = hour_of_day_profile(splits["train"][0].iloc[:, HUB], 5, 4)

    out = [evaluate(q, index, prices, x, E, v, off, cheap, dear) for q in quarters]
    tag = "" if len(quarters) == 2 else f"_{quarters[0]}"
    pd.concat([d.assign(quarter=q) for (d, *_), q in zip(out, quarters)]).round(2).to_csv(
        OUT / f"holdout_daily{tag}.csv")
    edges = pd.concat([o[2] for o in out])
    pd.concat([o[1] for o in out]).round(2).to_csv(OUT / f"holdout_results{tag}.csv", index=False)
    edges.round(3).to_csv(OUT / f"holdout_edges{tag}.csv", index=False)
    pd.concat([o[3] for o in out]).round(3).to_csv(OUT / f"holdout_regimes{tag}.csv", index=False)
    if len(quarters) == 2:
        print()
        verdict(edges).to_csv(OUT / "holdout_verdict.csv", index=False)


if __name__ == "__main__":
    main(["development"] if "--development" in sys.argv else ["development", "holdout"])
