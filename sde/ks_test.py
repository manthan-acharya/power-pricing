# Run: python3 -m sde.ks_test

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_LABELS

N_BOOT = 2000
NGRID = 2048
B_GRID = (6, 12, 24, 48, 96)


def _flat_top(t):
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def _acov(x, kmax):
    xc = x - x.mean()
    n = len(xc)
    return np.array([float(xc[:n - k] @ xc[k:]) / n for k in range(kmax + 1)])


# Politis-White bandwidth M = 2m, with m the first lag after which the ACF stays
# inside +-2 sqrt(log10(n)/n) for K_N consecutive lags.
def pw_bandwidth(x):
    n = len(x)
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    kmax = int(min(n - 2, mmax + kn))
    rho = _acov(x, kmax)
    rho = rho / rho[0]
    thresh = 2.0 * np.sqrt(np.log10(n) / n)
    for m in range(1, kmax - kn + 1):
        if np.all(np.abs(rho[m:m + kn]) < thresh):
            m_hat = m - 1
            break
    else:
        over = np.nonzero(np.abs(rho[1:]) > thresh)[0]
        m_hat = int(over[-1]) + 1 if len(over) else 1
    return int(np.clip(2 * m_hat, 1, mmax))


# Removes the hour-of-day profile but keeps the series mean, so a level bias
# still shows up.
def deseasonalize(x, hours):
    x, hours = np.asarray(x, float), np.asarray(hours)
    prof = np.array([x[hours == h].mean() if (hours == h).any() else x.mean()
                     for h in range(24)])
    return x - prof[hours] + x.mean()


# Politis-White block length b = (2G^2/D)^(1/3) n^(1/3) with D = (4/3)g(0)^2;
# returns 0 when G <= 0, which a strong diurnal cycle can cause.
def pw_block_length(x):
    n = len(x)
    M = min(pw_bandwidth(x), n - 2)
    r = _acov(x, M)
    k = np.arange(-M, M + 1)
    Rk = r[np.abs(k)]
    lam = _flat_top(k / M)
    g = float((lam * Rk).sum())
    G = float((lam * np.abs(k) * Rk).sum())
    if g <= 0 or G <= 0:
        return 0
    b = (2.0 * G ** 2 / ((4.0 / 3.0) * g ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    bmax = max(1.0, min(3.0 * np.sqrt(n), n / 3.0))
    return int(np.clip(round(b), 1, bmax))


# Falls back to the deseasonalized series, then to the n^(1/3) rate, when
# Politis-White degenerates on the raw diurnal series.
def select_block_length(x, hours=None):
    b = pw_block_length(x)
    if b == 0 and hours is not None:
        b = pw_block_length(deseasonalize(x, hours))
    return b if b > 0 else max(1, int(round(len(x) ** (1.0 / 3.0))))


# Integrated autocorrelation time 1 + 2*sum rho(k), flat-top truncated.
def act(x):
    M = min(pw_bandwidth(x), len(x) - 2)
    r = _acov(x, M)
    k = np.arange(1, M + 1)
    tau = 1.0 + 2.0 * float((_flat_top(k / M) * (r[k] / r[0])).sum())
    return max(tau, 1.0)


# Rank-transform to a common ECDF: KS is invariant under this transform, so the
# rank-space null matches the statistic computed on raw prices.
def _bins(x, ngrid=NGRID):
    u = (stats.rankdata(x) - 0.5) / len(x)
    return np.minimum((u * ngrid).astype(np.int64), ngrid - 1)


def _ks_bins(a, b, ngrid=NGRID):
    ca = np.cumsum(np.bincount(a, minlength=ngrid)) / len(a)
    cb = np.cumsum(np.bincount(b, minlength=ngrid)) / len(b)
    return float(np.abs(ca - cb).max())


def _draw(bins, starts, b, n_out, rng):
    nb = int(np.ceil(n_out / b))
    s = starts[rng.integers(0, len(starts), size=nb)]
    return bins[(s[:, None] + np.arange(b)).ravel()[:n_out]]


def block_bootstrap_ks(real, sim, hours_real=None, hours_sim=None,
                       b_real=None, b_sim=None, n_boot=N_BOOT, seed=0):
    P, T = sim.shape
    flat = sim.ravel()
    N, M = len(real), len(flat)
    obs = stats.ks_2samp(real, flat)

    br = b_real or select_block_length(real, hours_real)
    bs = b_sim or int(np.median([
        select_block_length(sim[i], None if hours_sim is None else hours_sim[i])
        for i in range(0, P, 5)]))
    br, bs = max(1, min(br, N)), max(1, min(bs, T))

    rbins, sbins = _bins(real), _bins(flat)
    rstarts = np.arange(N - br + 1)
    sstarts = (np.arange(P)[:, None] * T + np.arange(T - bs + 1)).ravel()

    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for i in range(n_boot):
        null[i] = _ks_bins(_draw(rbins, rstarts, br, N, rng),
                           _draw(sbins, sstarts, bs, M, rng))

    d95 = float(np.percentile(null, 95))
    n_eff = 1.0 / max((d95 / 1.358) ** 2 - 1.0 / M, 1e-12)
    return {
        "D": float(obs.statistic),
        "p_block": float((1 + (null >= obs.statistic).sum()) / (1 + n_boot)),
        "p_nominal": float(obs.pvalue),
        "b_real": br, "b_sim": bs,
        "d95_null": d95,
        "n_eff_boot": min(n_eff, float(N)),
        "n_eff_acf": N / act(real),
        "null": null,
    }


def build_arrays(reals, sims, hod, j):
    real = np.concatenate([rw[:, j] for rw in reals])
    rh = np.concatenate([np.asarray(h) for h in hod])
    sim = np.concatenate([sw[:, :, j] for sw in sims], axis=0)
    sh = np.concatenate([np.tile(np.asarray(h), (sw.shape[0], 1))
                         for sw, h in zip(sims, hod)], axis=0)
    return real, rh, sim, sh


def run_ks_check(reals, sims, hod, nodes, out_csv=None, n_boot=N_BOOT):
    rows = []
    for j, node in enumerate(nodes):
        lab = NODE_LABELS[node]
        real, rh, sim, sh = build_arrays(reals, sims, hod, j)
        raw = block_bootstrap_ks(real, sim, rh, sh, seed=1000 + j)

        real_d = deseasonalize(real, rh)
        sim_d = np.stack([deseasonalize(sim[i], sh[i]) for i in range(sim.shape[0])])
        des = block_bootstrap_ks(real_d, sim_d, seed=2000 + j)

        sweep = {b: block_bootstrap_ks(real, sim, b_real=b, b_sim=b,
                                       n_boot=n_boot // 4, seed=3000 + j)["p_block"]
                 for b in B_GRID}

        rows.append({
            "node": lab,
            "n_real": len(real), "n_sim": sim.size,
            "ks_D": raw["D"], "p_nominal_iid": raw["p_nominal"],
            "p_block_bootstrap": raw["p_block"],
            "block_len_real_h": raw["b_real"], "block_len_sim_h": raw["b_sim"],
            "n_eff_acf": raw["n_eff_acf"], "n_eff_bootstrap": raw["n_eff_boot"],
            "null_D95": raw["d95_null"],
            "ks_D_deseason": des["D"], "p_block_deseason": des["p_block"],
            "block_len_real_deseason_h": des["b_real"],
            "n_eff_deseason": des["n_eff_boot"],
            "null_D95_deseason": des["d95_null"],
            **{f"p_block_b{b}": sweep[b] for b in B_GRID},
        })
        print(f"{lab}: D {raw['D']:.3f} p_block {raw['p_block']:.4f} "
              f"(nominal {raw['p_nominal']:.1e})  b {raw['b_real']}/{raw['b_sim']}h  "
              f"n_eff {raw['n_eff_acf']:.0f} (acf) {raw['n_eff_boot']:.0f} (boot)  "
              f"null D95 {raw['d95_null']:.3f}")
        print(f"{lab}: deseasonalized D {des['D']:.3f} p_block {des['p_block']:.4f}"
              f"  b {des['b_real']}/{des['b_sim']}h  null D95 {des['d95_null']:.3f}")
        print(f"{lab}: p_block over fixed block lengths "
              + " ".join(f"b={b}:{sweep[b]:.3f}" for b in B_GRID))

    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False, float_format="%.6g")
    ok = bool((df["p_block_bootstrap"] > 0.05).all()
              and (df["p_block_deseason"] > 0.05).all())
    print("marginal-distribution criterion, block-bootstrap KS: "
          + ("PASS" if ok else "FAIL"))
    return ok, df


def main():
    from sde.model import full_series, load_frames
    from sde.validate import (PATHS_PER_WINDOW, WINDOW_DAYS, load_model,
                              run_window, window_starts)

    model, scaler, stage = load_model(ROOT / "sde" / "checkpoint.pt")
    print(f"checkpoint stage {stage}")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    nodes = list(val.columns)
    span_h = WINDOW_DAYS * 24

    starts = window_starts(index, val.index[0], 4, span_h)
    sims, reals, hod = [], [], []
    for k, i0 in enumerate(starts):
        sim = run_window(model, scaler, x, E, v, off, i0, span_h,
                         PATHS_PER_WINDOW, seed=100 + k)
        sims.append(sim[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span_h])
        hod.append(index[i0 + 1:i0 + 1 + span_h].tz_convert("US/Pacific").hour)
    print(f"{len(starts)} windows x {PATHS_PER_WINDOW} paths x {span_h}h")

    ok, _ = run_ks_check(reals, sims, hod, nodes,
                        out_csv=ROOT / "sde" / "ks_test_results.csv")
    return ok


if __name__ == "__main__":
    passed = main()
    print("marginal-distribution test PASSED" if passed
          else "marginal-distribution test FAILED")
