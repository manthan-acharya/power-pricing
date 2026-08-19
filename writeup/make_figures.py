"""Regenerate every figure in writeup/figures from saved checkpoints and CSVs.

Nothing here trains a model or writes a results CSV. Simulations reuse the
seeds of the scripts that produced the reported numbers, so annotated values
match the tables in the paper.

    OMP_NUM_THREADS=1 python writeup/make_figures.py            # all figures
    OMP_NUM_THREADS=1 python writeup/make_figures.py backtest qq
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import (AQUA, BLUE, INK, INK2, MUTED, NODE_COLORS, NODE_LABELS,
                       ORANGE, RULE, apply, headline)

FIG = ROOT / "writeup" / "figures"
HUBS = list(NODE_COLORS)
HUB_C = [NODE_COLORS[h] for h in HUBS]
HUB_L = [NODE_LABELS[h] for h in HUBS]
LIGHT = "#c9c8c2"
TZ = "US/Pacific"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {name}.pdf")


def end_label(ax, x, y, text, color, dy=0.0, **kw):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                va="center", ha="left", fontsize=7.5, color=color, **kw)


def date_axis(ax, fmt="%b %Y", interval=None):
    loc = (mdates.MonthLocator(interval=interval) if interval
           else mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=TZ))


# ----------------------------------------------------------------- data


def frames():
    from sde.model import load_frames
    return load_frames(ROOT)


_VAL = {}


def validation_sims(seed_base=100):
    """Re-anchored 14-day ensembles over the validation window (sde/validate.py)."""
    if seed_base in _VAL:
        return _VAL[seed_base]
    import sde.validate as V
    from sde.model import full_series
    torch.set_num_threads(1)
    model, scaler, _ = V.load_model(ROOT / "sde" / "checkpoint.pt")
    lmp, _, splits = frames()
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    span = V.WINDOW_DAYS * 24
    starts = V.window_starts(index, val.index[0], 4, span)
    sims, reals, times = [], [], []
    for k, i0 in enumerate(starts):
        sims.append(V.run_window(model, scaler, x, E, v, off, i0, span,
                                 V.PATHS_PER_WINDOW, seed=seed_base + k)[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span])
        times.append(index[i0 + 1:i0 + 1 + span])
    _VAL[seed_base] = (reals, sims, times)
    return _VAL[seed_base]


_BT = {}


def backtest_series():
    """Hourly P&L, dispatch and SOC on the test quarter, as in backtest/engine.py."""
    if _BT:
        return _BT
    from backtest.baselines import (hour_of_day_profile, naive_threshold,
                                    perfect_foresight, profile_dispatch,
                                    rolling_percentiles)
    from backtest.engine import S0, Policy, run_policy_on_prices
    from control.bsde import CAP, ETA, HUB, P_MAX, day_starts
    from sde.model import Scaler, full_series
    ck = torch.load(ROOT / "sde" / "checkpoint.pt", weights_only=False)
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    index, x, E, v, off = full_series(ROOT, scaler)
    lmp, _, splits = frames()
    policy = Policy(ROOT)
    starts = day_starts(index, "test")
    # the paper's 90-day test quarter predates the <= fix in day_starts that
    # admits 30 June; keep the rule that produced backtest/results.csv
    starts = starts[starts + 24 < len(index)]
    prices = lmp.values[:, HUB]
    t0, t1 = starts[0], starts[-1] + 24
    tp, ti = prices[t0:t1], index[t0:t1]
    pol, soc, u = run_policy_on_prices(policy, prices, x, E, v, off, starts)
    train = splits["train"][0].values[:, HUB]
    nvs, _, _ = naive_threshold(tp, np.percentile(train, 25),
                                np.percentile(train, 75), s0=S0)
    lo_r, hi_r = rolling_percentiles(pd.Series(prices, index=index), 168, 25, 75)
    nvr, _, _ = naive_threshold(tp, lo_r[t0:t1], hi_r[t0:t1], s0=S0)
    cheap, dear = hour_of_day_profile(splits["train"][0].iloc[:, HUB],
                                      math.ceil(CAP / (ETA * P_MAX)),
                                      math.ceil(ETA * CAP / P_MAX))
    hours = ti.tz_convert(TZ).hour
    hod, u_hod = profile_dispatch(tp, hours, cheap, dear, s0=S0)
    soc_hod = S0 + np.cumsum(np.where(u_hod >= 0, -u_hod / ETA, -ETA * u_hod))
    pf, _, _ = perfect_foresight(tp, s0=S0, pi_term=policy.pi_term * 1000.0)
    _BT.update(index=ti, prices=tp, pol=pol, soc=soc, u=u, nvs=nvs, nvr=nvr,
               hod=hod, u_hod=u_hod, soc_hod=soc_hod, pf=pf,
               pi=policy.pi_term * 1000.0, eta=ETA, cap=CAP, pmax=P_MAX)
    return _BT


# ----------------------------------------------------------------- figures


def fig_hub_prices():
    lmp, _, splits = frames()
    loc = lmp.tz_convert(TZ)
    daily = loc.resample("D").mean()
    dmin, dmax = loc.resample("D").min(), loc.resample("D").max()
    spread = (loc.iloc[:, 0] - loc.iloc[:, 1]).resample("D").mean()

    fig = plt.figure(figsize=(8.6, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 2.0], hspace=0.55)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

    bounds = {"validation": ("2026-02-01", "2026-04-01"),
              "test": ("2026-04-01", "2026-07-01")}
    for ax in (a, b):
        for lab, (s, e) in bounds.items():
            ax.axvspan(pd.Timestamp(s, tz=TZ), pd.Timestamp(e, tz=TZ),
                       color="#f1f0ec", lw=0, zorder=0)
    for lab, (s, e) in bounds.items():
        mid = pd.Timestamp(s, tz=TZ) + (pd.Timestamp(e, tz=TZ) - pd.Timestamp(s, tz=TZ)) / 2
        a.text(mid, 1.0, lab, transform=a.get_xaxis_transform(), ha="center",
               va="bottom", fontsize=7, color=INK2)
    a.text(pd.Timestamp("2025-07-15", tz=TZ), 1.0, "training",
           transform=a.get_xaxis_transform(), ha="center", va="bottom",
           fontsize=7, color=INK2)

    sp = HUBS[1]
    a.fill_between(daily.index, dmin[sp], dmax[sp], color=ORANGE, alpha=0.18,
                   lw=0, label="SP15 daily min to max")
    for h in HUBS:
        a.plot(daily.index, daily[h], color=NODE_COLORS[h], lw=1.0,
               label=f"{NODE_LABELS[h]} daily mean")
    a.axhline(0, color=MUTED, lw=0.7)
    a.set_ylabel("$/MWh")
    a.set_title("(a) Daily mean by hub; shading is SP15's daily range",
                pad=14)
    a.legend(ncol=4, loc="upper right", bbox_to_anchor=(1, 0.99))
    date_axis(a, "%b %Y", 2)
    a.set_xlim(daily.index[0], daily.index[-1])
    b.set_xlim(daily.index[0], daily.index[-1])
    for ax in (a, b):
        ax.axvline(pd.Timestamp("2026-04-01", tz=TZ), color="white", lw=1.5)

    b.fill_between(spread.index, 0, spread.values, where=spread.values >= 0,
                   color=BLUE, lw=0, alpha=0.8, step="mid")
    b.fill_between(spread.index, 0, spread.values, where=spread.values < 0,
                   color=MUTED, lw=0, alpha=0.8, step="mid")
    b.axhline(0, color=INK2, lw=0.6)
    b.set_ylabel("$/MWh")
    b.set_title("(b) NP15 minus SP15, daily mean")
    date_axis(b, "%b %Y", 2)

    sp_h = (lmp.iloc[:, 0] - lmp.iloc[:, 1]).abs()
    end = sp_h.rolling(240).mean().idxmax()
    zoom = loc.loc[end.tz_convert(TZ) - pd.Timedelta(days=10):end.tz_convert(TZ)]
    c.fill_between(zoom.index, zoom.iloc[:, 1], zoom.iloc[:, 0],
                   where=zoom.iloc[:, 0] > zoom.iloc[:, 1], color=BLUE,
                   alpha=0.12, lw=0, interpolate=True)
    for j, h in enumerate(HUBS):
        c.plot(zoom.index, zoom[h], color=NODE_COLORS[h], lw=1.1)
        end_label(c, zoom.index[-1], zoom[h].iloc[-1], NODE_LABELS[h],
                  NODE_COLORS[h], dy=[6, 2, -8][j])
    c.axhline(0, color=MUTED, lw=0.7)
    c.set_ylabel("$/MWh")
    c.set_title(f"(c) Hourly prices, {zoom.index[0]:%d %b} to "
                f"{zoom.index[-1]:%d %b %Y}")
    date_axis(c, "%d %b")
    c.set_xlim(zoom.index[0], zoom.index[-1] + pd.Timedelta(hours=20))

    headline(fig, "CAISO day-ahead hub prices, January 2025 to June 2026", y=0.94)
    save(fig, "hub_prices")


def fig_congestion_experiment():
    from grid.dcopf import congestion_experiment
    base, pert, _ = congestion_experiment()
    l0 = base["lines"][0]
    mu, lam = pert["mu"][l0], pert["lam"]
    buses = np.arange(1, base["lmp"].size + 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.2),
                               gridspec_kw={"width_ratios": [1.35, 1]})
    a.plot(buses, base["lmp"], "o-", color=MUTED, ms=4, label="base load")
    a.plot(buses, pert["lmp"], "o-", color=ORANGE, ms=4,
           label="+40 MW at bus 3")
    a.text(14.3, lam, f"$\\lambda$ = \\${lam:.0f}", va="center", fontsize=7.5,
           color=INK2)
    a.annotate("bus 3\n(load added)", (3, pert["lmp"][2]), xytext=(5.2, 29.2),
               fontsize=7, color=INK2,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    a.set_xticks(buses)
    a.set_xlabel("bus")
    a.set_ylabel("LMP ($/MWh)")
    a.set_xlim(0.5, 15.6)
    a.set_title("(a) Nodal prices, base load and +40 MW at bus 3")
    a.legend(loc="center right")

    H = pert["H"][l0]
    dl = pert["lmp"] - lam
    pred = -mu * H
    err = np.abs(dl - pred).max()
    lim = max(np.abs(dl).max(), np.abs(pred).max()) * 1.12
    b.plot([-lim, lim], [-lim, lim], lw=0.8, color=MUTED, zorder=1)
    b.scatter(pred, dl, s=26, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    b.set_aspect("equal")
    b.set_xlabel(r"predicted $-\mu H_{l,k}$ (\$/MWh)")
    b.set_ylabel(r"realized LMP$_k - \lambda$ (\$/MWh)")
    b.set_title("(b) Congestion component vs PTDF")
    b.text(0.04, 0.96, f"shadow price $\\mu$ = \\${mu:.1f}/MWh\n"
           f"max error {err:.1e} $/MWh", transform=b.transAxes, va="top",
           fontsize=7.5, color=INK2)
    headline(fig, "DC-OPF congestion on IEEE case14")
    fig.tight_layout()
    save(fig, "congestion_experiment")


def fig_congestion_link():
    import grid.congestion_link as CL
    rows = []
    lmp, cong, energy, loss = CL.load()
    dP, dC = CL.spread_space(lmp, cong, energy, loss, rows)
    C_toy, _ = CL.ptdf_reference_field()
    d_real, d_toy, null = CL.geometry(cong, C_toy, rows)
    R, jump, _, hl_real, hl_comp, hl_sim = CL.sde_consistency(lmp, cong, energy, rows)
    res = {r["metric"]: r["value"] for r in rows}

    fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.6))
    a = ax[0, 0]
    for j in (1, 2):
        xx, yy = dP[:, 0] - dP[:, j], dC[:, 0] - dC[:, j]
        r2 = np.corrcoef(xx, yy)[0, 1] ** 2
        a.scatter(xx, yy, s=2, alpha=0.18, color=HUB_C[j], lw=0, rasterized=True,
                  label=f"NP15 - {HUB_L[j]}   $R^2$ = {r2:.3f}")
    lim = np.percentile(np.abs(dP[:, 0] - dP[:, 1]), 99.5)
    a.plot([-lim, lim], [-lim, lim], lw=0.8, color=INK2)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("hourly change in hub spread ($/MWh)")
    a.set_ylabel("hourly change in CAISO congestion\ndifferential ($/MWh)")
    a.set_title("(a) Spread changes vs congestion changes")
    a.legend(markerscale=5, loc="upper left")

    a = ax[0, 1]
    d = np.arange(1, 11)
    a.fill_between(d, np.percentile(null, 2.5, 0), np.percentile(null, 97.5, 0),
                   color=LIGHT, alpha=0.7, lw=0)
    a.plot(d, null.mean(0), color=MUTED, lw=1.0, ls="--")
    a.plot(d, d_real, "o-", color=BLUE, ms=4)
    a.plot(d, d_toy, "s-", color=ORANGE, ms=4)
    end_label(a, 10, d_real[-1], "CAISO", BLUE, dy=4)
    end_label(a, 10, null.mean(0)[-1], "Gaussian null", INK2, dy=-6)
    end_label(a, 10, d_toy[-1], "case14 DC-OPF", ORANGE)
    a.set_xlim(0.6, 13.4)
    a.set_xticks(d)
    a.set_xlabel(r"decile of congestion magnitude $\|C_t\|$")
    a.set_ylabel("angular dispersion (deg)")
    a.set_title("(b) Angular dispersion by congestion decile")

    a = ax[1, 0]
    w, V = np.linalg.eigh(np.cov(cong.T))
    load = V[:, -1] * np.sign(V[0, -1]) * -1
    bars = a.bar(np.arange(3), load, 0.55,
                 color=[BLUE if v >= 0 else ORANGE for v in load])
    for bb, v in zip(bars, load):
        a.text(bb.get_x() + bb.get_width() / 2, v + (0.04 if v >= 0 else -0.04),
               f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
               fontsize=7.5, color=INK2)
    a.axhline(0, color=INK2, lw=0.6)
    a.set_xticks(np.arange(3))
    a.set_xticklabels(HUB_L)
    a.set_ylim(-0.85, 0.85)
    a.set_ylabel("loading on leading component")
    a.set_title(f"(c) Leading component of congestion ({100 * w[-1] / w.sum():.1f}% "
                "of variance)")
    a.text(0.98, 0.04, f"participation ratio {res['participation_ratio_caiso']:.2f}"
           " of 3", transform=a.transAxes, ha="right", fontsize=7.5, color=INK2)

    a = ax[1, 1]
    x = np.arange(3)
    for i, (arr, lab, c) in enumerate(((hl_comp, "CAISO components", LIGHT),
                                       (hl_real, "observed prices", INK2),
                                       (hl_sim, "fitted SDE", ORANGE))):
        a.bar(x + (i - 1) * 0.26, arr, 0.24, color=c, label=lab)
    a.set_xticks(x)
    a.set_xticklabels(["system level", "NP15-SP15", "NP15-ZP26"])
    a.set_ylabel("deseasonalized half-life (h)")
    a.set_title("(d) Deseasonalized half-lives")
    a.legend(loc="upper right")
    headline(fig, "DC-OPF congestion predictions against CAISO data")
    fig.tight_layout()
    save(fig, "congestion_link")


def fig_qq():
    reals, sims, _ = validation_sims()
    real = np.concatenate(reals)
    sim = np.concatenate([s.reshape(-1, 3) for s in sims])
    ks = pd.read_csv(ROOT / "sde" / "ks_test_results.csv").set_index("node")
    q = np.linspace(0.005, 0.995, 199)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharex=True, sharey=True)
    for j, ax in enumerate(axes):
        rq, sq = np.quantile(real[:, j], q), np.quantile(sim[:, j], q)
        lo, hi = -30, 75
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="#f6f5f2", lw=0)
        ax.plot([lo, hi], [lo, hi], lw=0.8, color=MUTED)
        ax.plot(rq, sq, lw=1.6, color=HUB_C[j])
        for p in (0.1, 0.5, 0.9):
            k = np.argmin(np.abs(q - p))
            ax.plot(rq[k], sq[k], "o", ms=4, color=HUB_C[j], mec="white", mew=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        k = ks.loc[HUB_L[j]]
        off = sim[:, j].mean() - real[:, j].mean()
        ax.text(0.04, 0.96, f"mean {off:+.1f} $/MWh\nKS D = {k.ks_D:.2f}, "
                f"p = {k.p_block_bootstrap:.4f}", transform=ax.transAxes,
                va="top", fontsize=7.2, color=INK2)
        ax.text(0.96, 0.05, "above diagonal:\nsimulated too high", ha="right",
                transform=ax.transAxes, fontsize=6.5, color=MUTED)
        ax.set_title(HUB_L[j])
        ax.set_xlabel("real quantile ($/MWh)")
    axes[0].set_ylabel("simulated quantile ($/MWh)")
    headline(fig, "Simulated vs realized price quantiles, validation windows")
    fig.tight_layout()
    save(fig, "marginal_qq")


def fig_acf(nlags=72):
    from sde.validate import acf_curve
    reals, sims, _ = validation_sims()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    lags = np.arange(nlags + 1)
    for j, ax in enumerate(axes):
        ar = np.mean([acf_curve(rw[:, j], nlags) for rw in reals], axis=0)
        asim = np.stack([acf_curve(sw[i, :, j], nlags)
                         for sw in sims for i in range(0, sw.shape[0], 5)])
        lo, md, hi = np.percentile(asim, [5, 50, 95], axis=0)
        ax.fill_between(lags, lo, hi, color=HUB_C[j], alpha=0.22, lw=0,
                        label="simulated 5-95%")
        ax.plot(lags, md, color=HUB_C[j], lw=1.1, label="simulated median")
        ax.plot(lags, ar, color=INK, lw=1.2, label="real")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks([0, 24, 48, 72])
        ax.set_xlabel("lag (hours)")
        ax.set_title(HUB_L[j])
    axes[0].set_ylabel("autocorrelation")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.0))
    headline(fig, "Autocorrelation of hourly prices, validation windows")
    fig.tight_layout()
    save(fig, "autocorrelation")


def fig_paths(days=10):
    reals, sims, times = validation_sims()
    hrs = 24 * days
    t = times[0][:hrs].tz_convert(TZ)
    j = 1
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True, sharey=True)
    axes[0].plot(t, reals[0][:hrs, j], color=INK, lw=1.0)
    axes[0].set_title("(a) Realized SP15")
    for i, c in enumerate((BLUE, ORANGE, AQUA)):
        axes[1].plot(t, sims[0][i, :hrs, j], color=c, lw=0.9,
                     label=f"path {i + 1}")
    axes[1].plot(t, reals[0][:hrs, j], color=INK, lw=0.6, alpha=0.35,
                 label="realized")
    axes[1].set_title("(b) Three simulated paths, same initial state")
    axes[1].legend(ncol=4, loc="upper right", bbox_to_anchor=(1, 1.16))
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel("$/MWh")
    date_axis(axes[1], "%d %b")
    headline(fig, f"SP15, first {days} days of the validation window")
    fig.tight_layout()
    save(fig, "simulated_paths")


def fig_rank():
    from sde.calibration import SEED, ranks_of
    reals, sims, _ = validation_sims(seed_base=700)
    cal = pd.read_csv(ROOT / "sde" / "calibration.csv").set_index("node")
    rng = np.random.default_rng(SEED)
    M = sims[0].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    for j, ax in enumerate(axes):
        r = np.concatenate([ranks_of(reals[w][:, j], sims[w][:, :, j], rng)
                            for w in range(len(reals))])
        h = np.bincount(r, minlength=M + 1) / len(r) * (M + 1)
        ax.bar(np.arange(M + 1), h, width=0.82, color=HUB_C[j], lw=0)
        ax.axhline(1.0, color=INK, lw=0.9, ls="--")
        c = cal.loc[HUB_L[j]]
        ax.text(0.97, 0.95, f"mean PIT {c.mean_PIT:.2f}\n95% CI "
                f"[{c.PIT_ci_lo:.2f}, {c.PIT_ci_hi:.2f}]\ncalibrated = 0.50",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                color=INK2)
        ax.set_title(HUB_L[j])
        ax.set_xlabel("rank of realized price among 50 members")
        ax.set_xlim(-1, M + 1)
    axes[0].set_ylabel("frequency relative to uniform")
    axes[0].text(M, 1.06, "flat = calibrated", ha="right", va="bottom",
                 fontsize=6.8, color=INK2)
    headline(fig, "Rank histograms, validation windows")
    fig.tight_layout()
    save(fig, "rank_histograms")


def fig_seed_robustness():
    from sde.seed_robustness import CRITERIA, N_SEEDS
    draws = pd.read_csv(ROOT / "sde" / "seed_robustness.csv")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (crit, (name, (lo, hi))) in zip(axes, CRITERIA.items()):
        vals = draws[draws.criterion == crit]["value"]
        thr = min([t for t in (lo, hi) if t is not None],
                  key=lambda t: min(abs(vals.min() - t), abs(vals.max() - t)))
        left, right = min(vals.min(), thr), max(vals.max(), thr)
        pad = 0.18 * (right - left)
        left, right = left - pad, right + pad
        ax.axvspan(left if thr == lo else thr, thr if thr == lo else right,
                   color="#f3e3dc", lw=0, zorder=0)
        ax.axvline(thr, color=INK2, ls="--", lw=0.9)
        for i, lab in enumerate(HUB_L):
            y = 2 - i
            d = draws[(draws.node == lab) & (draws.criterion == crit)]
            ax.scatter(d.value, y + rng.uniform(-0.18, 0.18, len(d)), s=4,
                       color=HUB_C[i], alpha=0.35, lw=0, rasterized=True)
            ax.plot([d.value.mean()], [y], "o", color=HUB_C[i], ms=5,
                    mec="white", mew=0.8, zorder=3)
            frac = d.passed.mean()
            if frac < 1:
                ax.text(right, y + 0.3, f"{100 * frac:.1f}% pass",
                        ha="right", fontsize=7, color=INK2)
        ax.set_xlim(left, right)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(HUB_L)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("pass " + (f"$\\leq$ {hi}" if lo is None else f"{lo} to {hi}"))
        ax.grid(axis="y", visible=False)
    headline(fig, f"Validation checks across {N_SEEDS} simulation seeds")
    fig.tight_layout()
    save(fig, "seed_robustness")


def fig_variants():
    sde = ROOT / "sde"
    tf = pd.read_csv(sde / "twofactor_seeds.csv")
    mh = pd.read_csv(sde / "mh_comparison.csv")
    pin = pd.read_csv(sde / "pin_comparison.csv")
    obs = pd.read_csv(sde / "twofactor_comparison.csv").set_index("model")
    obs = obs.loc["observed prices"]
    inc = mh[mh.label == "incumbent"].iloc[0]
    groups = [
        ("incum-\nbent", [inc.halflife_level_h], [inc.spread_over_level], [inc.val_nll]),
        ("two-factor\ndrift", tf.halflife_level_h,
         (tf.halflife_spread1_h + tf.halflife_spread2_h) / 2 / tf.halflife_level_h,
         tf.val_nll),
        ("ACF in\nStage B", mh[mh.label != "incumbent"].halflife_level_h,
         mh[mh.label != "incumbent"].spread_over_level,
         mh[mh.label != "incumbent"].val_nll),
        ("pinned\nrates", pin[pin.label.str.startswith("pin_")].halflife_level_h,
         pin[pin.label.str.startswith("pin_")].spread_over_level,
         pin[pin.label.str.startswith("pin_")].val_nll),
        ("pinned +\nrescaled noise",
         pin[pin.label.str.startswith("pin2_")].halflife_level_h,
         pin[pin.label.str.startswith("pin2_")].spread_over_level,
         pin[pin.label.str.startswith("pin2_")].val_nll),
    ]
    targets = [(obs.halflife_level_h, "observed 13.0 h"),
               (obs.spread_over_level_halflife, "observed 0.38"), (None, None)]
    titles = ["(a) Level half-life (h)", "(b) Spread-to-level half-life ratio",
              "(c) Held-out one-step NLL"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    rng = np.random.default_rng(1)
    for k, ax in enumerate(axes):
        for i, g in enumerate(groups):
            vals = np.asarray(g[k + 1], dtype=float)
            jit = rng.uniform(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jit), vals, s=22,
                       color=MUTED if i == 0 else BLUE, edgecolor="white",
                       lw=0.7, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=INK, lw=1.2)
        tv, tl = targets[k]
        if tv is not None:
            ax.axhline(tv, color=ORANGE, lw=1.2, ls="--")
            ax.text(-0.5, tv, tl, color=ORANGE, fontsize=7.2,
                    ha="left", va="bottom")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(titles[k], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    headline(fig, "Price-model variants by training seed")
    fig.tight_layout()
    save(fig, "twofactor_variant")


def fig_dp_check():
    df = pd.read_csv(ROOT / "control" / "dp_crosscheck.csv")
    tol = 0.15
    bad = df.rel_err > tol
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    hi = max(df.V_dp.max(), df.V_bsde.max()) * 1.06
    a.plot([0, hi], [0, hi], color=MUTED, lw=0.8)
    a.scatter(df.V_dp[~bad], df.V_bsde[~bad], s=20, color=BLUE,
              edgecolor="white", lw=0.6, label=f"within 15% ({(~bad).sum()})")
    a.scatter(df.V_dp[bad], df.V_bsde[bad], s=30, color=ORANGE,
              edgecolor="white", lw=0.6, label=f"outside 15% ({bad.sum()})")
    a.set_xlim(0, hi)
    a.set_ylim(0, hi)
    a.set_aspect("equal")
    a.set_xlabel("grid DP value (k$)")
    a.set_ylabel("deep BSDE value (k$)")
    a.set_title("(a) Value at 60 state-time points")
    a.legend(loc="upper left")

    b.axhspan(tol * 100, 40, color="#f3e3dc", lw=0, zorder=0)
    b.axhline(tol * 100, color=INK2, ls="--", lw=0.9)
    b.text(df.V_dp.max(), tol * 100 + 0.8, "15% tolerance", ha="right",
           fontsize=7.2, color=INK2)
    b.scatter(df.V_dp[~bad], 100 * df.rel_err[~bad], s=20, color=BLUE,
              edgecolor="white", lw=0.6)
    b.scatter(df.V_dp[bad], 100 * df.rel_err[bad], s=30, color=ORANGE,
              edgecolor="white", lw=0.6)
    for r in df[bad].itertuples():
        b.annotate(f"t={r.t0}h, s={r.s:.0f} MWh", (r.V_dp, 100 * r.rel_err),
                   xytext=(6, 0), textcoords="offset points", fontsize=6.8,
                   va="center", color=INK2)
    b.set_ylim(0, 36)
    b.set_xlim(0, hi)
    b.set_xlabel("grid DP value (k$)")
    b.set_ylabel("relative error (%)")
    b.set_title("(b) Relative error vs DP value")
    headline(fig, "Deep BSDE vs grid DP, reduced one-hub problem")
    fig.tight_layout()
    save(fig, "dp_cross_check")


def fig_bsde_loss():
    log = pd.read_csv(ROOT / "control" / "bsde_train_log.csv")
    smooth = log.loss.rolling(200, center=True, min_periods=20).median()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.semilogy(log.iter, log.loss, lw=0.5, color=BLUE, alpha=0.35,
                label="per iteration")
    ax.semilogy(log.iter, smooth, lw=1.4, color=BLUE, label="rolling median (200)")
    final = smooth.dropna().iloc[-1]
    ax.annotate(f"{final:.3f}", (log.iter.iloc[-1], final), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=INK2,
                va="center")
    ax.set_xlabel("training iteration")
    ax.set_ylabel(r"terminal-matching loss (k\$$^2$)")
    ax.legend(loc="upper right")
    headline(fig, "Deep BSDE training loss, three-hub problem")
    fig.tight_layout()
    save(fig, "bsde_loss")


def fig_backtest():
    bt = backtest_series()
    n = len(bt["pol"]) // 24
    days = pd.DatetimeIndex(bt["index"][::24]).tz_convert(TZ)[:n]

    def cum(p):
        return np.cumsum(p[:n * 24].reshape(n, 24).sum(1)) / 1000

    gdp = pd.read_csv(ROOT / "control" / "grid_dp_backtest.csv")
    series = [("perfect foresight", cum(bt["pf"]), INK, "--"),
              ("deep BSDE", cum(bt["pol"]), BLUE, "-"),
              ("grid DP", np.cumsum(gdp.grid_dp.values) / 1000, AQUA, "-"),
              ("hour-of-day rule", cum(bt["hod"]), ORANGE, "-"),
              ("rolling percentile rule", cum(bt["nvr"]), MUTED, "-"),
              ("static percentile rule", cum(bt["nvs"]), LIGHT, "-")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1]})
    offs = {"deep BSDE": 7, "grid DP": 0, "hour-of-day rule": -7}
    for lab, y, c, ls in series:
        a.plot(days, y, color=c, ls=ls, lw=1.3)
        end_label(a, days[-1], y[-1], f"{lab}  ${y[-1]:.0f}k", c if c != LIGHT
                  else MUTED, dy=offs.get(lab, 0))
    a.set_ylabel("cumulative revenue (k$)")
    a.set_xlim(days[0], days[-1] + pd.Timedelta(days=21))
    b.set_xticks([pd.Timestamp(d, tz=TZ) for d in
                  ("2026-04-01", "2026-04-15", "2026-05-01", "2026-05-15",
                   "2026-06-01", "2026-06-15", "2026-06-30")])
    a.set_title("(a) Cumulative revenue")
    for m in ("2026-05-01", "2026-06-01"):
        for ax in (a, b):
            ax.axvline(pd.Timestamp(m, tz=TZ), color=RULE, lw=1.0, zorder=0)

    hod = cum(bt["hod"])
    ci = pd.read_csv(ROOT / "backtest" / "capture_ci.csv")
    g = ci[(ci.quantity == "policy_minus_hod_pp") & (ci.role == "primary")].iloc[0]
    b.axhline(0, color=ORANGE, lw=1.0)
    b.plot(days, cum(bt["pol"]) - hod, color=BLUE, lw=1.4)
    b.plot(days, np.cumsum(gdp.grid_dp.values) / 1000 - hod, color=AQUA, lw=1.4)
    end_label(b, days[-1], (cum(bt["pol"]) - hod)[-1], "deep BSDE", BLUE, dy=5)
    end_label(b, days[-1], (np.cumsum(gdp.grid_dp.values) / 1000 - hod)[-1],
              "grid DP", AQUA, dy=-5)
    end_label(b, days[-1], 0, "hour-of-day rule", ORANGE, dy=-8)
    b.set_ylabel("lead over hour-of-day\nrule (k$)")
    b.set_title("(b) Lead over the hour-of-day rule")
    b.text(0.01, 0.92, f"final gap {g.point:+.1f} points of the perfect-foresight "
           f"ceiling, 95% CI {g.ci_lo:+.1f} to {g.ci_hi:+.1f}",
           transform=b.transAxes, fontsize=7.5, color=INK2, va="top")
    b.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TZ))
    headline(fig, "Battery revenue on realized SP15 prices, April to June 2026")
    fig.tight_layout()
    save(fig, "backtest_pnl")


def fig_dispatch(days=6, start=24 * 33):
    bt = backtest_series()
    sel = slice(start, start + days * 24)
    t = pd.DatetimeIndex(bt["index"][sel]).tz_convert(TZ)
    thr = bt["pi"] / bt["eta"] + 2.0
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 5.8), sharex=True)

    ev = [(d + pd.Timedelta(hours=18), d + pd.Timedelta(hours=22))
          for d in pd.date_range(t[0].normalize(), t[-1], freq="D")]
    for ax in axes:
        for s, e in ev:
            ax.axvspan(s, e, color="#f6efe0", lw=0, zorder=0)
    a, b, c = axes
    a.step(t, bt["prices"][sel], where="post", color=INK, lw=1.0)
    a.axhline(thr, color=BLUE, ls="--", lw=1.0)
    a.text(t[0], thr + 1.5, f"policy's late-day discharge threshold "
           f"$\\pi/\\eta_d + c_{{deg}}$ = \\${thr:.0f}", fontsize=7.2, color=BLUE)
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_ylabel("SP15 ($/MWh)")
    a.set_title("(a) SP15 price; shading marks 18:00-22:00")

    b.fill_between(t, bt["u"][sel], 0, step="post", color=BLUE, alpha=0.75,
                   lw=0, label="deep BSDE")
    b.step(t, bt["u_hod"][sel], where="post", color=ORANGE, lw=1.0,
           label="hour-of-day rule")
    b.axhline(0, color=INK2, lw=0.6)
    b.set_ylim(-bt["pmax"] * 1.25, bt["pmax"] * 1.25)
    b.set_ylabel("MW (+ = sell)")
    b.set_title("(b) Dispatch")
    b.legend(ncol=2, loc="upper right", bbox_to_anchor=(1, 1.22))

    c.plot(t, bt["soc"][sel], color=BLUE, lw=1.3)
    c.plot(t, bt["soc_hod"][sel], color=ORANGE, lw=1.0)
    c.set_ylim(0, bt["cap"] * 1.05)
    c.set_ylabel("stored energy (MWh)")
    c.set_title("(c) State of charge")
    date_axis(c, "%a %d %b")
    c.set_xlim(t[0], t[-1] + pd.Timedelta(hours=1))
    headline(fig, f"Dispatch on SP15, {t[0]:%d} to {t[-1]:%d %B %Y}")
    fig.tight_layout()
    save(fig, "dispatch_detail")


FIGS = {
    "hub_prices": fig_hub_prices,
    "congestion_experiment": fig_congestion_experiment,
    "congestion_link": fig_congestion_link,
    "qq": fig_qq,
    "acf": fig_acf,
    "paths": fig_paths,
    "rank": fig_rank,
    "seed_robustness": fig_seed_robustness,
    "variants": fig_variants,
    "dp_check": fig_dp_check,
    "bsde_loss": fig_bsde_loss,
    "backtest": fig_backtest,
    "dispatch": fig_dispatch,
}

if __name__ == "__main__":
    apply()
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
