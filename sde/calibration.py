import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_COLORS, NODE_LABELS, apply
from sde.model import full_series, load_frames
from sde.validate import PATHS_PER_WINDOW, WINDOW_DAYS, load_model, run_window, window_starts

FIG = ROOT / "writeup" / "figures"
N_BOOT = 4000
SEED = 4


# Rank of each observation among its own ensemble, ties broken at random so the
# null is exactly uniform on {0..M}.
def ranks_of(real, sim, rng):
    below = (sim < real[None, :]).sum(0)
    ties = (sim == real[None, :]).sum(0)
    return below + (rng.random(real.shape) * (ties + 1)).astype(int)


# Moving-block bootstrap CI for a statistic of an autocorrelated series,
# resampling the observed series (rather than simulating a null) to keep the
# dependence structure that is actually present.
def block_ci(series, block, rng, stat=np.mean, n_boot=N_BOOT, alpha=0.05):
    n = len(series)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    draws = stat(series[idx], axis=1)
    return (np.percentile(draws, 100 * alpha / 2),
            np.percentile(draws, 100 * (1 - alpha / 2)), draws)


def autocorr_block(r, max_lag=72):
    x = r - r.mean()
    v = (x * x).mean()
    if v <= 0:
        return 1
    ac = [(x[:-L] * x[L:]).mean() / v for L in range(1, max_lag + 1)]
    tau = 1 + 2 * sum(a for a in ac if a > 0)
    return max(1, int(round(tau)))


# Jolliffe-Primo decomposition: orthogonal linear and squared contrasts isolate
# bias and dispersion errors within the chi-square.
def contrasts(counts):
    m = len(counts)
    n = counts.sum()
    p = counts / n
    x = (np.arange(m) - (m - 1) / 2) / ((m - 1) / 2)
    lin = np.sum(x * p) * np.sqrt(3 * n * m)
    q = x ** 2 - np.mean(x ** 2)
    sq = np.sum(q * p) * n / np.sqrt(n * np.sum(q ** 2) / m)
    return lin, sq


def main():
    apply()
    rng = np.random.default_rng(SEED)
    model, scaler, _ = load_model(ROOT / "sde" / "checkpoint.pt")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    nodes = list(val.columns)
    span = WINDOW_DAYS * 24
    starts = window_starts(index, val.index[0], 4, span)

    sims, reals = [], []
    for k, i0 in enumerate(starts):
        np.random.seed(700 + k)
        sims.append(run_window(model, scaler, x, E, v, off, i0, span,
                               PATHS_PER_WINDOW, seed=700 + k)[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span])
    sim = np.stack(sims)
    real = np.stack(reals)
    n_bins = PATHS_PER_WINDOW + 1

    rows, hists = [], {}
    for j, node in enumerate(nodes):
        r = np.concatenate([ranks_of(real[w, :, j], sim[w, :, :, j], rng)
                            for w in range(len(starts))])
        counts = np.bincount(r, minlength=n_bins)
        n = len(r)
        exp = n / n_bins
        chi2 = ((counts - exp) ** 2 / exp).sum()
        block = autocorr_block(r.astype(float))
        lin, sq = contrasts(counts)

        # targeted tests: an omnibus chi-square over 51 bins has little power
        # against a smooth slope, which is the alternative the histogram shows
        pit = r / (n_bins - 1)
        lo, hi, _ = block_ci(pit, block, rng)
        blo, bhi, _ = block_ci((r == 0).astype(float), block, rng)
        alo, ahi, _ = block_ci((r == n_bins - 1).astype(float), block, rng)
        bias_rejects = not (lo <= 0.5 <= hi)
        tail_rejects = not (blo <= 1 / n_bins <= bhi)

        rows.append({"node": NODE_LABELS[node], "n": n, "members": PATHS_PER_WINDOW,
                     "block_len": block, "chi2": chi2, "chi2_eff": chi2 * block / n * n_bins,
                     "linear_contrast": lin, "squared_contrast": sq,
                     "mean_PIT": pit.mean(), "PIT_ci_lo": lo, "PIT_ci_hi": hi,
                     "bias_rejects_at_05": bias_rejects,
                     "frac_below_ensemble": (r == 0).mean(),
                     "below_ci_lo": blo, "below_ci_hi": bhi,
                     "frac_above_ensemble": (r == n_bins - 1).mean(),
                     "above_ci_lo": alo, "above_ci_hi": ahi,
                     "expected_tail_frac": 1 / n_bins,
                     "tail_rejects_at_05": tail_rejects})
        hists[NODE_LABELS[node]] = counts / n * n_bins
        print(f"{NODE_LABELS[node]}: block {block}h  mean PIT {pit.mean():.3f} "
              f"95% CI [{lo:.3f}, {hi:.3f}] vs 0.500 -> "
              f"{'REJECT' if bias_rejects else 'no reject'}")
        print(f"        below ensemble {(r==0).mean():.3f} CI [{blo:.3f}, {bhi:.3f}]"
              f"  above {(r==n_bins-1).mean():.3f} CI [{alo:.3f}, {ahi:.3f}]"
              f"  expected {1/n_bins:.3f} -> {'REJECT' if tail_rejects else 'no reject'}")

    df = pd.DataFrame(rows).round(4)
    df.to_csv(ROOT / "sde" / "calibration.csv", index=False)

    any_rej = any(r["bias_rejects_at_05"] or r["tail_rejects_at_05"] for r in rows)
    print(f"\ncalibration: {'REJECTED' if any_rej else 'not rejected'}")
    for r in rows:
        d = ("biased HIGH (obs rank low)" if r["mean_PIT"] < 0.5 else "biased LOW")
        print(f"  {r['node']}: {d}; mean PIT {r['mean_PIT']:.3f}; "
              f"obs below whole ensemble {r['frac_below_ensemble']/r['expected_tail_frac']:.1f}x expected, "
              f"above {r['frac_above_ensemble']/r['expected_tail_frac']:.1f}x")

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    for ax, node in zip(axes, nodes):
        lab = NODE_LABELS[node]
        ax.bar(np.arange(n_bins), hists[lab], width=1.0,
               color=NODE_COLORS[node], alpha=0.85, lw=0)
        ax.axhline(1.0, color="#333333", lw=0.9, ls="--")
        ax.set_title(lab)
        ax.set_xlabel("rank of observation in ensemble")
        if lab == "NP15":
            ax.set_ylabel("relative frequency")
    fig.suptitle("Rank histograms: flat = calibrated, sloped = biased", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "rank_histograms.pdf")
    print(f"saved {FIG / 'rank_histograms.pdf'}")
    return not any_rej


if __name__ == "__main__":
    main()
