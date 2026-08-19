# Run: OMP_NUM_THREADS=1 python3 -m sde.seed_robustness

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_COLORS, NODE_LABELS, apply
from sde import validate as V
from sde.model import full_series, load_frames

N_SEEDS = 200
N_DIAG_SEEDS = 25
MARGINAL_FRAC = 0.15
DRAWS_CSV = ROOT / "sde" / "seed_robustness.csv"
SUMMARY_CSV = ROOT / "sde" / "seed_robustness_summary.csv"
FIG = ROOT / "writeup" / "figures"

CRITERIA = {
    "a_mean": ("mean offset", (None, V.CHECKS["mean_offset_max"])),
    "a_std": ("std ratio", V.CHECKS["std_ratio"]),
    "a_neg": ("neg-price freq ratio", V.CHECKS["neg_freq_ratio"]),
    "b_spike": ("spike rate ratio", V.CHECKS["spike_ratio"]),
    "c_halflife": ("half-life ratio", V.CHECKS["halflife_ratio"]),
}


# The five per-hub validation statistics, computed exactly as in validate.main.
def check_stats(reals, sims, hod, nodes):
    sim_pool = np.concatenate([s.reshape(-1, 3) for s in sims])
    real_pool = np.concatenate(reals)
    out = {}
    for j, node in enumerate(nodes):
        r, s = real_pool[:, j], sim_pool[:, j]
        mean_off = abs(s.mean() - r.mean()) / r.std()
        std_ratio = s.std() / r.std()
        neg_ratio = ((s < 0).mean() + 1e-9) / ((r < 0).mean() + 1e-9)

        rspk = np.mean([(np.abs(np.diff(rw[:, j])) > V.SPIKE_USD).mean()
                        for rw in reals])
        sspk = np.mean([(np.abs(np.diff(sw[:, :, j], axis=1)) > V.SPIKE_USD).mean()
                        for sw in sims])
        spike_ratio = (sspk + 1e-9) / (rspk + 1e-9)

        hl_real = np.nanmedian([V.halflife_arr(rw[:, j], h)
                                for rw, h in zip(reals, hod)])
        hl_sims = []
        for sw, h in zip(sims, hod):
            for i in range(0, sw.shape[0], 10):
                hl_sims.append(V.halflife_arr(sw[i, :, j], h))
        hl_ratio = np.nanmedian(hl_sims) / hl_real

        out[NODE_LABELS[node]] = {
            "a_mean": mean_off, "a_std": std_ratio, "a_neg": neg_ratio,
            "b_spike": spike_ratio, "c_halflife": hl_ratio,
        }
    return out


def passes(crit, value):
    lo, hi = CRITERIA[crit][1]
    if lo is None:
        return value <= hi
    return lo <= value <= hi


# Relative distance to the nearest binding threshold; negative if failing.
def margin(crit, value):
    lo, hi = CRITERIA[crit][1]
    d = (hi - value) / hi if lo is None else min((value - lo) / lo, (hi - value) / hi)
    return d


# Well-separated (numpy, torch) seed pairs so replicates are independent rather
# than consecutive integers off one stream.
def replicate_seeds(rep, n_windows):
    s = np.random.SeedSequence([20260803, rep]).generate_state(2 * n_windows)
    return [(int(s[2 * k]), int(s[2 * k + 1])) for k in range(n_windows)]


def simulate_replicate(rep, model, scaler, x, E, v, off, lmp, index, starts, span_h):
    sims, reals, hod = [], [], []
    for (np_seed, torch_seed), i0 in zip(replicate_seeds(rep, len(starts)), starts):
        sim = V.run_window(model, scaler, x, E, v, off, i0, span_h,
                           V.PATHS_PER_WINDOW, seed=torch_seed, np_seed=np_seed)
        sims.append(sim[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span_h])
        hod.append(index[i0 + 1:i0 + 1 + span_h].tz_convert("US/Pacific").hour)
    return reals, sims, hod


# Recomputes the half-life ratio from all 200 simulated paths, rather than the
# 20 validate.main strides over, to check whether the check's seed sensitivity
# is a striding artifact.
def halflife_path_diagnostic(model, scaler, x, E, v, off, lmp, index, starts,
                             nodes, span_h, n_seeds=N_DIAG_SEEDS):
    rows = []
    for rep in range(n_seeds):
        reals, sims, hod = simulate_replicate(1000 + rep, model, scaler, x, E, v,
                                              off, lmp, index, starts, span_h)
        for j, node in enumerate(nodes):
            hl_real = np.nanmedian([V.halflife_arr(rw[:, j], h)
                                    for rw, h in zip(reals, hod)])
            row = {"seed": rep, "node": NODE_LABELS[node]}
            for stride in (10, 1):
                hl = [V.halflife_arr(sw[i, :, j], h) for sw, h in zip(sims, hod)
                      for i in range(0, sw.shape[0], stride)]
                row[f"ratio_stride{stride}"] = np.nanmedian(hl) / hl_real
            rows.append(row)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(ROOT / "sde" / "seed_robustness_halflife_paths.csv", index=False)
    print(f"\nhalf-life ratio, {n_seeds} seeds, 20-path median vs all 200 paths:")
    for lab, g in df.groupby("node", sort=False):
        print(f"  {lab}: stride10 mean {g.ratio_stride10.mean():.3f} sd "
              f"{g.ratio_stride10.std(ddof=1):.3f} | all-path mean "
              f"{g.ratio_stride1.mean():.3f} sd {g.ratio_stride1.std(ddof=1):.3f}")
    return df


def summarize(draws):
    rows = []
    for (lab, crit), g in draws.groupby(["node", "criterion"], sort=False):
        val = g["value"].values
        lo, hi = CRITERIA[crit][1]
        rows.append({
            "node": lab, "criterion": crit, "statistic": CRITERIA[crit][0],
            "threshold_lo": lo, "threshold_hi": hi,
            "mean": val.mean(), "std": val.std(ddof=1),
            "min": val.min(), "max": val.max(),
            "pass_frac": g["passed"].mean(),
            "min_margin": min(margin(crit, val.min()), margin(crit, val.max())),
        })
    order = {c: i for i, c in enumerate(CRITERIA)}
    df = pd.DataFrame(rows)
    return df.sort_values(["criterion", "node"], key=lambda s: s.map(order).fillna(0)
                          if s.name == "criterion" else s).reset_index(drop=True)


# Each panel is zoomed on the threshold the draws actually approach; the far
# side of a two-sided band is never within reach and is only labelled.
def plot_spread(draws, labels):
    apply()
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.2))
    rng = np.random.default_rng(0)
    for ax, (crit, (name, (lo, hi))) in zip(axes, CRITERIA.items()):
        col_vals = draws[draws.criterion == crit]["value"]
        thr = min([t for t in (lo, hi) if t is not None],
                  key=lambda t: min(abs(col_vals.min() - t), abs(col_vals.max() - t)))
        left = min(col_vals.min(), thr)
        right = max(col_vals.max(), thr)
        pad = 0.18 * (right - left)
        left, right = left - pad, right + pad

        ax.axvspan(thr if thr == lo else left, right if thr == lo else thr,
                   color="#dddddd", alpha=0.5, lw=0, zorder=0)
        ax.axvline(thr, color="#444444", ls="--", lw=1.0)
        for i, lab in enumerate(labels):
            y = len(labels) - 1 - i
            node = [n for n, l in NODE_LABELS.items() if l == lab][0]
            col = NODE_COLORS[node]
            val = draws[(draws.node == lab) & (draws.criterion == crit)]["value"].values
            ax.scatter(val, y + rng.uniform(-0.17, 0.17, val.size), s=4,
                       color=col, alpha=0.3, lw=0)
            ax.plot([val.min(), val.max()], [y, y], color=col, lw=1.2)
            ax.plot([val.mean()], [y], "o", color=col, ms=5,
                    mec="white", mew=0.7, zorder=3)
            frac = draws[(draws.node == lab) & (draws.criterion == crit)]["passed"].mean()
            if frac < 1:
                ax.text(right - 0.02 * (right - left), y + 0.26,
                        f"{100 * frac:.0f}% pass", ha="right", va="bottom",
                        fontsize=7, color=col)
        ax.set_xlim(left, right)
        ax.set_ylim(-0.6, len(labels) - 0.35)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_title(name)
        ax.set_xlabel("pass " + (f"$\\leq$ {hi}" if lo is None else f"{lo}-{hi}"))
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Validation criteria across {N_SEEDS} simulation seeds "
                 "(shaded = passing side, dot = mean, bar = min-max)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "seed_robustness.pdf")


def main():
    torch.set_num_threads(1)
    model, scaler, stage = V.load_model(ROOT / "sde" / "checkpoint.pt")
    print(f"checkpoint stage {stage}")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    nodes = list(val.columns)
    labels = [NODE_LABELS[n] for n in nodes]
    span_h = V.WINDOW_DAYS * 24
    starts = V.window_starts(index, val.index[0], 4, span_h)
    print(f"{N_SEEDS} seeds x {len(starts)} windows x {V.PATHS_PER_WINDOW} paths x {span_h}h")

    rows = []
    for rep in range(N_SEEDS):
        reals, sims, hod = simulate_replicate(rep, model, scaler, x, E, v, off,
                                              lmp, index, starts, span_h)
        stats = check_stats(reals, sims, hod, nodes)
        n_pass = 0
        for lab in labels:
            for crit in CRITERIA:
                value = stats[lab][crit]
                ok = passes(crit, value)
                n_pass += ok
                rows.append({"seed": rep, "node": lab, "criterion": crit,
                             "value": value, "passed": ok})
        pd.DataFrame(rows).to_csv(DRAWS_CSV, index=False)
        print(f"seed {rep:3d}  {n_pass}/15 passed"
              + "".join(f"  {lab} hl {stats[lab]['c_halflife']:.3f}" for lab in labels))

    draws = pd.DataFrame(rows)
    summary = summarize(draws)
    summary.round(4).to_csv(SUMMARY_CSV, index=False)

    print(summary.round(3).to_string(index=False))
    totals = draws.groupby("seed")["passed"].sum().value_counts().sort_index(ascending=False)
    print("\npassing checks per seed:")
    for k, n in totals.items():
        print(f"  {k}/15 in {n}/{N_SEEDS} seeds")

    print("\nmarginal checks (worst draw within "
          f"{100 * MARGINAL_FRAC:.0f}% of a threshold):")
    for _, r in summary.iterrows():
        if r["min_margin"] < MARGINAL_FRAC:
            print(f"  {r['node']} {r['criterion']}: range "
                  f"{r['min']:.3f}-{r['max']:.3f}, pass fraction {r['pass_frac']:.2f},"
                  f" worst margin {100 * r['min_margin']:.1f}%")

    halflife_path_diagnostic(model, scaler, x, E, v, off, lmp, index, starts,
                             nodes, span_h)
    plot_spread(draws, labels)


if __name__ == "__main__":
    main()
