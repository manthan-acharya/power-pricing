import sys
from pathlib import Path

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.bsde import CAP, C_DEG, ETA, P_MAX

C_DEG_USD = C_DEG * 1000.0


def soc_step(s, u):
    if u >= 0:
        u = min(u, ETA * s)
        return s - u / ETA, u
    u = max(u, -(CAP - s) / ETA)
    return s - ETA * u, u


def naive_threshold(prices, lo, hi, s0=50.0):
    lo = np.broadcast_to(np.asarray(lo, float), prices.shape)
    hi = np.broadcast_to(np.asarray(hi, float), prices.shape)
    s, pnl, soc, us = s0, [], [], []
    for t, p in enumerate(prices):
        u = P_MAX if p > hi[t] else (-P_MAX if p < lo[t] else 0.0)
        s, u_eff = soc_step(s, u)
        pnl.append(p * u_eff - C_DEG_USD * abs(u_eff))
        soc.append(s)
        us.append(u_eff)
    return np.array(pnl), np.array(soc), np.array(us)


def rolling_percentiles(series, window=168, q_lo=25, q_hi=75):
    lo = series.rolling(window).quantile(q_lo / 100).shift(1)
    hi = series.rolling(window).quantile(q_hi / 100).shift(1)
    return lo.bfill().values, hi.bfill().values


def hour_of_day_profile(train_series, n_charge, n_discharge):
    local = train_series.tz_convert("US/Pacific")
    prof = local.groupby(local.index.hour).mean()
    cheap = set(prof.nsmallest(n_charge).index)
    dear = set(prof.nlargest(n_discharge).index)
    return cheap, dear


def profile_dispatch(prices, hours, cheap, dear, s0=50.0):
    s, pnl, us = s0, [], []
    for p, h in zip(prices, hours):
        u = P_MAX if h in dear else (-P_MAX if h in cheap else 0.0)
        s, u_eff = soc_step(s, u)
        pnl.append(p * u_eff - C_DEG_USD * abs(u_eff))
        us.append(u_eff)
    return np.array(pnl), np.array(us)


def perfect_foresight(prices, s0=50.0, pi_term=0.0):
    n = len(prices)
    ud = cp.Variable(n, nonneg=True)
    uc = cp.Variable(n, nonneg=True)
    s = cp.Variable(n + 1)
    cons = [s[0] == s0, s >= 0, s <= CAP, ud <= P_MAX, uc <= P_MAX,
            s[1:] == s[:-1] - ud / ETA + ETA * uc]
    rev = prices @ ud - prices @ uc - C_DEG_USD * cp.sum(ud + uc) \
        + pi_term * s[n]
    prob = cp.Problem(cp.Maximize(rev), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal", prob.status
    pnl = prices * ud.value - prices * uc.value \
        - C_DEG_USD * (ud.value + uc.value)
    return pnl, s.value[1:], ud.value - uc.value
