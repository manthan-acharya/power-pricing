import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_COLORS, NODE_LABELS, apply
from sde.ks_test import run_ks_check
from sde.model import HubSDE, Scaler, full_series, load_frames, simulate

FIG = ROOT / "writeup" / "figures"
WINDOW_DAYS = 14
PATHS_PER_WINDOW = 50
SPIKE_USD = 25.0

CHECKS = {
    "mean_offset_max": 0.5,      # |sim mean - real mean| in units of real std
    "std_ratio": (0.7, 1.4),
    "neg_freq_ratio": (0.5, 2.0),
    "spike_ratio": (0.5, 2.0),
    "halflife_ratio": (0.5, 2.0),
}


def load_model(path):
    ck = torch.load(path, weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    return model, scaler, ck.get("stage", "?")


def window_starts(index, val_t0, n_windows, span_h):
    starts = []
    for k in range(n_windows):
        t = val_t0 + pd.Timedelta(hours=k * span_h)
        starts.append(index.get_loc(t))
    return starts


# seed pins the jumps; the diffusion's Brownian path comes from numpy's legacy
# global RNG via torchsde, so it must be seeded too, and np_seed lets that
# stream vary independently of the jump seed (used by sde/seed_robustness.py).
def run_window(model, scaler, x, E, v, off, i0, span_h, n_paths, seed,
               np_seed=None):
    np.random.seed(seed if np_seed is None else np_seed)
    x0 = torch.tensor(x[i0:i0 + 1]).repeat(n_paths, 1)
    E0 = torch.tensor(E[i0:i0 + 1]).repeat(n_paths, 1)
    v0 = torch.tensor(v[i0:i0 + 1]).repeat(n_paths, 1)
    off0 = torch.tensor(off[i0:i0 + 1]).repeat(n_paths, 1)
    xs = simulate(model, x0, off0, E0, v0, n_hours=span_h, seed=seed)
    return scaler.inverse(xs.numpy().transpose(1, 0, 2))


# Half-life criterion statistic: half-life implied by the lag-1 autocorrelation
# of the series with its hour-of-day profile removed.
def halflife_arr(r, hours):
    prof = pd.Series(r).groupby(hours).transform("mean").values
    resid = r - prof
    phi = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    if not (0 < phi < 1):
        return np.nan
    return np.log(2) / -np.log(phi)


def main():
    apply()
    model, scaler, stage = load_model(ROOT / "sde" / "checkpoint.pt")
    print(f"checkpoint stage {stage}")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    nodes = list(val.columns)
    span_h = WINDOW_DAYS * 24

    val_t0 = val.index[0]
    starts = window_starts(index, val_t0, 4, span_h)
    sims, reals, hod = [], [], []
    for k, i0 in enumerate(starts):
        sim = run_window(model, scaler, x, E, v, off, i0, span_h,
                         PATHS_PER_WINDOW, seed=100 + k)
        sims.append(sim[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span_h])
        hod.append(index[i0 + 1:i0 + 1 + span_h].tz_convert("US/Pacific").hour)
    sim_pool = np.concatenate([s.reshape(-1, 3) for s in sims])
    real_pool = np.concatenate(reals)
    print(f"{len(starts)} windows x {PATHS_PER_WINDOW} paths x {span_h}h")

    results, rows = {}, []
    for j, node in enumerate(nodes):
        lab = NODE_LABELS[node]
        r, s = real_pool[:, j], sim_pool[:, j]
        mean_off = abs(s.mean() - r.mean()) / r.std()
        std_ratio = s.std() / r.std()
        neg_ratio = ((s < 0).mean() + 1e-9) / ((r < 0).mean() + 1e-9)

        rspk = np.mean([(np.abs(np.diff(rw[:, j])) > SPIKE_USD).mean()
                        for rw in reals])
        sspk = np.mean([(np.abs(np.diff(sw[:, :, j], axis=1)) > SPIKE_USD).mean()
                        for sw in sims])
        spike_ratio = (sspk + 1e-9) / (rspk + 1e-9)

        hl_real = np.nanmedian([halflife_arr(rw[:, j], h)
                                for rw, h in zip(reals, hod)])
        hl_sims = []
        for sw, h in zip(sims, hod):
            for i in range(0, sw.shape[0], 10):
                hl_sims.append(halflife_arr(sw[i, :, j], h))
        hl_sim = np.nanmedian(hl_sims)
        hl_ratio = hl_sim / hl_real

        ks_ds = [stats.ks_2samp(rw[:, j], sw[i, :, j]).statistic
                 for rw, sw in zip(reals, sims) for i in range(0, sw.shape[0], 25)]

        rows.append({
            "node": lab, "real_mean": r.mean(), "sim_mean": s.mean(),
            "real_std": r.std(), "sim_std": s.std(),
            "real_pct_neg": 100 * (r < 0).mean(), "sim_pct_neg": 100 * (s < 0).mean(),
            "real_spikes_per_1000h": 1000 * rspk, "sim_spikes_per_1000h": 1000 * sspk,
            "real_halflife_h": hl_real, "sim_halflife_h": hl_sim,
            "ks_D_median": np.median(ks_ds),
        })
        results[lab] = {
            "a_mean": mean_off <= CHECKS["mean_offset_max"],
            "a_std": CHECKS["std_ratio"][0] <= std_ratio <= CHECKS["std_ratio"][1],
            "a_neg": CHECKS["neg_freq_ratio"][0] <= neg_ratio <= CHECKS["neg_freq_ratio"][1],
            "b_spike": CHECKS["spike_ratio"][0] <= spike_ratio <= CHECKS["spike_ratio"][1],
            "c_halflife": CHECKS["halflife_ratio"][0] <= hl_ratio <= CHECKS["halflife_ratio"][1],
        }
        print(f"{lab}: mean {r.mean():.1f}/{s.mean():.1f}  std {r.std():.1f}/{s.std():.1f}"
              f"  neg% {100 * (r < 0).mean():.1f}/{100 * (s < 0).mean():.1f}"
              f"  spikes/1000h {1000 * rspk:.1f}/{1000 * sspk:.1f}"
              f"  halflife {hl_real:.1f}/{hl_sim:.1f}h  ksD {np.median(ks_ds):.3f}")

    df = pd.DataFrame(rows).round(3)
    df.to_csv(ROOT / "sde" / "validation_metrics.csv", index=False)

    flat = {f"{lab}.{k}": bool(vv) for lab, res in results.items() for k, vv in res.items()}
    n_fail = sum(not vv for vv in flat.values())
    for k, vv in flat.items():
        if not vv:
            print(f"FAIL {k}")
    print(f"moment/spike/half-life checks: {len(flat) - n_fail}/{len(flat)} passed")

    ks_ok, _ = run_ks_check(reals, sims, hod, nodes,
                           out_csv=ROOT / "sde" / "ks_test_results.csv")

    free_run_diagnostic(model, scaler, x, E, v, off, starts[0], val, nodes)
    plot_qq(real_pool, sim_pool, nodes)
    plot_acf(reals, sims, nodes)
    plot_paths(val, sims[0], nodes)
    return n_fail == 0 and ks_ok


def free_run_diagnostic(model, scaler, x, E, v, off, i0, val, nodes):
    n_hours = len(val) - 1
    sim = run_window(model, scaler, x, E, v, off, i0, n_hours, 50, seed=555)
    out = {}
    for j, node in enumerate(nodes):
        r, s = val.values[1:, j], sim[:, 1:, j].ravel()
        out[NODE_LABELS[node]] = {"real_mean": r.mean(), "sim_mean": s.mean(),
                                  "real_std": r.std(), "sim_std": s.std()}
        print(f"free-run 59d {NODE_LABELS[node]}: mean {r.mean():.1f}/{s.mean():.1f}"
              f"  std {r.std():.1f}/{s.std():.1f}")
    pd.DataFrame(out).T.round(2).to_csv(ROOT / "sde" / "freerun_diagnostic.csv")


def plot_qq(real, sim, nodes):
    q = np.linspace(0.005, 0.995, 199)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for j, (ax, node) in enumerate(zip(axes, nodes)):
        rq = np.quantile(real[:, j], q)
        sq = np.quantile(sim[:, j], q)
        lo, hi = min(rq.min(), sq.min()), max(rq.max(), sq.max())
        ax.plot([lo, hi], [lo, hi], lw=0.8, color="#999999")
        ax.plot(rq, sq, lw=1.4, color=NODE_COLORS[node])
        ax.set_title(NODE_LABELS[node])
        ax.set_xlabel(r"real quantile (\$/MWh)")
        if j == 0:
            ax.set_ylabel(r"simulated quantile (\$/MWh)")
    fig.suptitle("Marginal distribution over validation windows", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "marginal_qq.pdf")


def acf_curve(x, nlags):
    xc = x - x.mean()
    v = (xc * xc).mean()
    return np.array([1.0] + [(xc[:-L] * xc[L:]).mean() / v
                             for L in range(1, nlags + 1)])


def plot_acf(reals, sims, nodes, nlags=72):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for j, (ax, node) in enumerate(zip(axes, nodes)):
        ar = np.mean([acf_curve(rw[:, j], nlags) for rw in reals], axis=0)
        asim = np.stack([acf_curve(sw[i, :, j], nlags)
                         for sw in sims for i in range(0, sw.shape[0], 5)])
        lo, md, hi = np.percentile(asim, [5, 50, 95], axis=0)
        lags = np.arange(nlags + 1)
        ax.fill_between(lags, lo, hi, alpha=0.25, color=NODE_COLORS[node], lw=0)
        ax.plot(lags, md, lw=1.0, color=NODE_COLORS[node], label="simulated")
        ax.plot(lags, ar, lw=1.2, color="#333333", label="real")
        ax.set_title(NODE_LABELS[node])
        ax.set_xlabel("lag (h)")
        if j == 0:
            ax.set_ylabel("autocorrelation")
            ax.legend()
    fig.suptitle("ACF over validation windows, real vs simulated (5-95% band)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "autocorrelation.pdf")


def plot_paths(val, sim0, nodes, days=10):
    hrs = 24 * days
    t = val.index[1:hrs + 1]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True)
    j = 1  # SP15
    axes[0].plot(t, val.values[1:hrs + 1, j], lw=0.9, color="#333333")
    axes[0].set_title(f"Real {NODE_LABELS[nodes[j]]}, first {days} days of validation")
    axes[0].set_ylabel(r"\$/MWh")
    for i in range(3):
        axes[1].plot(t, sim0[i, :hrs, j], lw=0.9, alpha=0.9)
    axes[1].set_title("Three simulated paths, same window and initial state")
    axes[1].set_ylabel(r"\$/MWh")
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(lo, hi)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "simulated_paths.pdf")


if __name__ == "__main__":
    ok = main()
    print("validation criteria PASSED" if ok else "validation criteria FAILED")
