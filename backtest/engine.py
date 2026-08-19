import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.baselines import (hour_of_day_profile, naive_threshold,
                                perfect_foresight, profile_dispatch,
                                rolling_percentiles)
from control.bsde import (CAP, C_DEG, ETA, HUB, P_MAX, DeepBSDE,
                          day_conditioning, day_starts)
from plotstyle import apply
from sde.model import HubSDE, Scaler, full_series, load_frames, simulate

OUT = ROOT / "backtest"
FIG = ROOT / "writeup" / "figures"
C_DEG_USD = C_DEG * 1000.0
S0 = 50.0


class Policy:
    def __init__(self, root):
        ck = torch.load(root / "control/bsde_checkpoint.pt", weights_only=False)
        self.nets = DeepBSDE()
        self.nets.load_state_dict(ck["nets"])
        self.nets.eval()
        self.eps = ck["eps"]
        self.pi_term = ck["pi_term"]

    @torch.no_grad()
    def act(self, k, s, x_std, off0, E0, v0):
        cond = day_conditioning(off0, E0, v0)
        xdev = torch.as_tensor(x_std, dtype=torch.float32).reshape(1, -1) - E0
        feats = self.nets.feats(torch.tensor([s], dtype=torch.float32),
                                xdev, cond)
        Z = self.nets.z(k, feats)
        return Z[0, 0].item() / self.eps


def dispatch_from_q(p_usd, s, q_kusd):
    u_dis = min(P_MAX, ETA * max(s, 0.0))
    u_chg = min(P_MAX, (CAP - s) / ETA)
    p = p_usd / 1000.0
    dis_val = (p - C_DEG - q_kusd / ETA) * u_dis
    chg_val = (ETA * q_kusd - C_DEG - p) * u_chg
    if dis_val >= chg_val and dis_val > 0:
        return u_dis
    if chg_val > 0:
        return -u_chg
    return 0.0


def run_policy_on_prices(policy, prices_usd, x_std, E, v, off, starts, s0=S0):
    s = s0
    pnl, soc, us = [], [], []
    for d in starts:
        off0 = torch.tensor(off[d:d + 1])
        E0 = torch.tensor(E[d:d + 1])
        v0 = torch.tensor(v[d:d + 1])
        for k in range(24):
            t = d + k
            q = policy.act(k, s, torch.tensor(x_std[t]), off0, E0, v0)
            u = dispatch_from_q(prices_usd[t], s, q)
            if u >= 0:
                s = s - u / ETA
            else:
                s = s - ETA * u
            pnl.append(prices_usd[t] * u - C_DEG_USD * abs(u))
            soc.append(s)
            us.append(u)
    return np.array(pnl), np.array(soc), np.array(us)


def sharpe_like(path_revs):
    m, sd = np.mean(path_revs), np.std(path_revs)
    return m / sd if sd > 0 else np.nan


def daily_totals(pnl, n):
    return pnl[:n * 24].reshape(n, 24).sum(1)


def paired_daily_test(pnl_a, pnl_b, seed=0):
    n = len(pnl_a) // 24
    d = daily_totals(pnl_a, n) - daily_totals(pnl_b, n)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, n, replace=True).mean() for _ in range(20000)])
    return {"n_days": n, "mean_daily_edge": d.mean(), "sd_daily_edge": d.std(ddof=1),
            "paired_t": t, "ci_lo": np.percentile(bs, 2.5),
            "ci_hi": np.percentile(bs, 97.5), "days_won": int((d > 0).sum())}


def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def capture_bootstrap(pnls, n_boot=20000, block=5, seed=1):
    n = len(pnls["perfect_foresight"]) // 24
    daily = {k: daily_totals(v, n) for k, v in pnls.items()}
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(n_boot, n // block))
    idx = ((st[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)
    tot = {k: v[idx].sum(1) for k, v in daily.items()}
    pf, pf0 = tot["perfect_foresight"], daily["perfect_foresight"].sum()

    def row(name, draws, point):
        return {"quantity": name, "block_days": block, "point": point,
                "boot_mean": draws.mean(), "se": draws.std(ddof=1),
                "ci_lo": np.percentile(draws, 2.5),
                "ci_hi": np.percentile(draws, 97.5),
                "p_le_0": float((draws <= 0).mean())}

    # ratio of summed revenues within each resample, not a mean of daily ratios
    rows = [row(f"{k}_pct_of_pf", 100 * tot[k] / pf,
                100 * daily[k].sum() / pf0)
            for k in ("bsde_policy", "hour_of_day", "naive_rolling",
                      "naive_static")]
    rows.append(row("policy_minus_hod_pp",
                    100 * (tot["bsde_policy"] - tot["hour_of_day"]) / pf,
                    100 * (daily["bsde_policy"].sum()
                           - daily["hour_of_day"].sum()) / pf0))
    return pd.DataFrame(rows)


def simulated_evaluation(policy, model, scaler, index, x, E, v, off,
                         n_paths=200, block_days=28):
    starts = day_starts(index, "test")
    anchors = [starts[0], starts[28], starts[56]]
    revs, dds = [], []
    for a_i, anchor in enumerate(anchors):
        x0 = torch.tensor(x[anchor:anchor + 1]).repeat(n_paths // len(anchors), 1)
        E0 = torch.tensor(E[anchor:anchor + 1]).repeat(x0.shape[0], 1)
        v0 = torch.tensor(v[anchor:anchor + 1]).repeat(x0.shape[0], 1)
        off0 = torch.tensor(off[anchor:anchor + 1]).repeat(x0.shape[0], 1)
        # torchsde seeds its Brownian motion from numpy's global RNG
        np.random.seed(900 + a_i)
        sim = simulate(model, x0, off0, E0, v0, n_hours=block_days * 24,
                       seed=900 + a_i)
        sim_usd = scaler.inverse(sim.numpy().transpose(1, 0, 2))
        for i in range(sim_usd.shape[0]):
            xs = scaler.transform(sim_usd[i])
            Eb, vb = E[anchor].copy(), v[anchor].copy()
            s, pnl = S0, []
            for day in range(block_days):
                offd = torch.tensor(off[anchor + day * 24:anchor + day * 24 + 1]) \
                    if anchor + day * 24 < len(off) else torch.tensor(off[anchor:anchor + 1])
                E0d = torch.tensor(Eb[None, :], dtype=torch.float32)
                v0d = torch.tensor(vb[None, :], dtype=torch.float32)
                for k in range(24):
                    t = day * 24 + k
                    q = policy.act(k, s, torch.tensor(xs[t]), offd, E0d, v0d)
                    u = dispatch_from_q(sim_usd[i, t, HUB], s, q)
                    s = s - u / ETA if u >= 0 else s - ETA * u
                    pnl.append(sim_usd[i, t, HUB] * u - C_DEG_USD * abs(u))
                alpha = math.exp(-1.0 / 168.0)
                for t in range(day * 24, (day + 1) * 24):
                    Eb = alpha * Eb + (1 - alpha) * xs[t]
                    if t > 0:
                        vb = alpha * vb + (1 - alpha) * np.abs(xs[t] - xs[t - 1]).mean(keepdims=True)
            pnl = np.array(pnl)
            revs.append(pnl.sum())
            dds.append(max_drawdown(np.cumsum(pnl)))
    return np.array(revs), np.array(dds)


def main():
    apply()
    torch.set_num_threads(1)
    ck = torch.load(ROOT / "sde/checkpoint.pt", weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    index, x, E, v, off = full_series(ROOT, scaler)
    lmp, _, splits = load_frames(ROOT)
    policy = Policy(ROOT)
    pi_term = policy.pi_term

    test_starts = day_starts(index, "test")
    prices = lmp.values[:, HUB]
    t0, t1 = test_starts[0], test_starts[-1] + 24
    test_prices = prices[t0:t1]
    test_index = index[t0:t1]
    print(f"test window: {index[t0]} .. {index[t1 - 1]}  ({len(test_prices)}h)")

    pnl_pol, soc_pol, u_pol = run_policy_on_prices(
        policy, prices, x, E, v, off, test_starts)

    train_prices = splits["train"][0].values[:, HUB]
    lo_s, hi_s = np.percentile(train_prices, 25), np.percentile(train_prices, 75)
    pnl_nvs, _, _ = naive_threshold(test_prices, lo_s, hi_s, s0=S0)
    ser = pd.Series(prices, index=index)
    lo_r, hi_r = rolling_percentiles(ser, 168, 25, 75)
    pnl_nvr, _, _ = naive_threshold(test_prices, lo_r[t0:t1], hi_r[t0:t1], s0=S0)

    # full-power hours needed to fill from empty and to drain from full
    n_chg = math.ceil(CAP / (ETA * P_MAX))
    n_dis = math.ceil(ETA * CAP / P_MAX)
    cheap, dear = hour_of_day_profile(splits["train"][0].iloc[:, HUB], n_chg, n_dis)
    test_hours = test_index.tz_convert("US/Pacific").hour
    pnl_hod, u_hod = profile_dispatch(test_prices, test_hours, cheap, dear, s0=S0)
    print(f"hour-of-day heuristic: charge {sorted(cheap)}, discharge {sorted(dear)}")

    pnl_pf, _, _ = perfect_foresight(test_prices, s0=S0,
                                     pi_term=pi_term * 1000.0)

    months = test_index.tz_convert("US/Pacific").month
    rows = []
    for label, sel in [("Apr", months == 4), ("May", months == 5),
                       ("Jun", months == 6), ("total", slice(None))]:
        row = {"window": label,
               "bsde_policy": pnl_pol[sel].sum(),
               "naive_static": pnl_nvs[sel].sum(),
               "naive_rolling": pnl_nvr[sel].sum(),
               "hour_of_day": pnl_hod[sel].sum(),
               "perfect_foresight": pnl_pf[sel].sum()}
        row["pct_of_pf"] = 100 * row["bsde_policy"] / row["perfect_foresight"]
        rows.append(row)
        print(f"{label:6s} policy {row['bsde_policy']:9.0f}  naive_s "
              f"{row['naive_static']:9.0f}  naive_r {row['naive_rolling']:9.0f}"
              f"  hod {row['hour_of_day']:9.0f}"
              f"  PF {row['perfect_foresight']:9.0f}  %PF {row['pct_of_pf']:5.1f}")
    df = pd.DataFrame(rows)
    df.round(1).to_csv(OUT / "results.csv", index=False)

    sig = paired_daily_test(pnl_pol, pnl_hod)
    print(f"policy vs hour-of-day: daily edge ${sig['mean_daily_edge']:.0f} "
          f"(sd ${sig['sd_daily_edge']:.0f}), paired t {sig['paired_t']:.2f}, "
          f"95% CI [${sig['ci_lo']:.0f}, ${sig['ci_hi']:.0f}], "
          f"won {sig['days_won']}/{sig['n_days']} days")
    pd.Series(sig).round(3).to_csv(OUT / "significance.csv", header=False)

    dis = np.clip(u_pol, 0.0, None)
    evening = (test_hours >= 18) & (test_hours < 22)
    disp = {"policy_throughput_mwh": np.abs(u_pol).sum(),
            "hour_of_day_throughput_mwh": np.abs(u_hod).sum(),
            "policy_discharge_share_00_07": dis[test_hours < 7].sum() / dis.sum(),
            "policy_discharge_share_18_22": dis[evening].sum() / dis.sum(),
            "policy_mean_end_of_day_soc": soc_pol[test_hours == 23].mean()}
    print(f"throughput: policy {disp['policy_throughput_mwh']:.0f} MWh, hour-of-day "
          f"{disp['hour_of_day_throughput_mwh']:.0f} MWh; policy discharge "
          f"{disp['policy_discharge_share_00_07']:.1%} in 00-07h, "
          f"{disp['policy_discharge_share_18_22']:.1%} in 18-22h; mean end-of-day SOC "
          f"{disp['policy_mean_end_of_day_soc']:.1f} MWh")
    pd.Series(disp).round(3).to_csv(OUT / "dispatch_stats.csv", header=False)

    pnls = {"bsde_policy": pnl_pol, "hour_of_day": pnl_hod,
            "naive_rolling": pnl_nvr, "naive_static": pnl_nvs,
            "perfect_foresight": pnl_pf}
    ci = capture_bootstrap(pnls)
    for r in ci.itertuples():
        print(f"{r.quantity:26s} {r.point:6.2f}  95% CI "
              f"[{r.ci_lo:6.2f}, {r.ci_hi:6.2f}]  se {r.se:4.2f}")
    sens = pd.concat([capture_bootstrap(pnls, block=b).query(
        "quantity == 'policy_minus_hod_pp'") for b in (1, 3, 7, 10)])
    ci["role"], sens["role"] = "primary", "block_sensitivity"
    print("  block-length sensitivity on the gap: " + "  ".join(
        f"L={r.block_days} [{r.ci_lo:.1f},{r.ci_hi:.1f}]"
        for r in sens.itertuples()))
    pd.concat([ci, sens]).round(4).to_csv(OUT / "capture_ci.csv", index=False)

    revs, dds = simulated_evaluation(policy, model, scaler, index, x, E, v, off)
    print(f"simulated 28d paths: mean rev {revs.mean():.0f}  std {revs.std():.0f}"
          f"  sharpe-like {sharpe_like(revs):.2f}  median maxDD {np.median(dds):.0f}")
    pd.DataFrame({"rev": revs, "max_dd": dds}).round(1).to_csv(
        OUT / "sim_paths.csv", index=False)

    plot_backtest(test_index, pnl_pol, pnl_nvs, pnl_nvr, pnl_hod, pnl_pf, ci)
    plot_dispatch(test_index, test_prices, soc_pol, u_pol)


def plot_backtest(idx, pol, nvs, nvr, hod, pf, ci):
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    loc = idx.tz_convert("US/Pacific")
    for pnl, lab, col in [(pf, "perfect foresight", "#999999"),
                          (pol, "deep BSDE policy", "#0072B2"),
                          (hod, "hour-of-day heuristic", "#009E73"),
                          (nvr, "naive rolling 25/75", "#D55E00"),
                          (nvs, "naive static 25/75", "#CC79A7")]:
        ax.plot(loc, np.cumsum(pnl) / 1000, lw=1.1, label=lab, color=col)
    ax.set_ylabel(r"cumulative P\&L (k\$)")
    ax.legend(loc="upper left")
    ax.set_title("Battery dispatch on realized SP15 test prices, Apr-Jun 2026")
    q = ci.set_index("quantity")
    ax.text(0.985, 0.15,
            "policy captures %.1f%% of PF, 95%% CI [%.1f, %.1f]\n"
            "gap over hour-of-day %.1f pp, 95%% CI [%.1f, %.1f]"
            % (q.point["bsde_policy_pct_of_pf"],
               q.ci_lo["bsde_policy_pct_of_pf"], q.ci_hi["bsde_policy_pct_of_pf"],
               q.point["policy_minus_hod_pp"],
               q.ci_lo["policy_minus_hod_pp"], q.ci_hi["policy_minus_hod_pp"]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#444444", linespacing=1.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "backtest_pnl.pdf")


def plot_dispatch(idx, prices, soc, u, days=6, start=24 * 33):
    sel = slice(start, start + days * 24)
    loc = idx.tz_convert("US/Pacific")[sel]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.6), sharex=True)
    axes[0].plot(loc, prices[sel], lw=1.0, color="#333333")
    axes[0].set_ylabel(r"SP15 LMP (\$/MWh)")
    axes[1].fill_between(loc, np.asarray(u[sel]), 0, alpha=0.6,
                         color="#0072B2", lw=0)
    axes[1].set_ylabel("dispatch (MW)")
    axes[1].set_ylim(-P_MAX * 1.1, P_MAX * 1.1)
    axes[2].plot(loc, soc[sel], lw=1.0, color="#009E73")
    axes[2].set_ylabel("SOC (MWh)")
    axes[2].set_ylim(0, CAP)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "dispatch_detail.pdf")


if __name__ == "__main__":
    main()
