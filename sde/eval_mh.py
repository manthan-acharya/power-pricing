# Run: OMP_NUM_THREADS=1 python3 -u -m sde.eval_mh <checkpoint.pt> <label>

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from plotstyle import NODE_LABELS
from sde.calibration import autocorr_block, block_ci, contrasts, ranks_of
from sde.model import full_series, load_frames
from sde.train import split_slice
from sde.train_twofactor import factor_halflives
from sde.validate import (CHECKS, PATHS_PER_WINDOW, SPIKE_USD, WINDOW_DAYS,
                          halflife_arr, load_model, run_window, window_starts)

OUT = ROOT / "sde"
N_WINDOWS = 4


# np.random.seed before every run_window: torchsde seeds its Brownian motion
# from numpy's legacy global RNG, so without this the diffusion is
# irreproducible even though the jumps are not.
def simulate_windows(model, scaler, x, E, v, off, starts, span, base_seed):
    sims = []
    for k, i0 in enumerate(starts):
        np.random.seed(base_seed + k)
        sims.append(run_window(model, scaler, x, E, v, off, i0, span,
                               PATHS_PER_WINDOW, seed=base_seed + k)[:, 1:])
    return sims


def moment_checks(reals, sims, hod, nodes):
    sim_pool = np.concatenate([s.reshape(-1, 3) for s in sims])
    real_pool = np.concatenate(reals)
    rows, flat = [], {}
    for j, node in enumerate(nodes):
        lab = NODE_LABELS[node]
        r, s = real_pool[:, j], sim_pool[:, j]
        mean_off = abs(s.mean() - r.mean()) / r.std()
        std_ratio = s.std() / r.std()
        neg_ratio = ((s < 0).mean() + 1e-9) / ((r < 0).mean() + 1e-9)
        rspk = np.mean([(np.abs(np.diff(rw[:, j])) > SPIKE_USD).mean() for rw in reals])
        sspk = np.mean([(np.abs(np.diff(sw[:, :, j], axis=1)) > SPIKE_USD).mean()
                        for sw in sims])
        spike_ratio = (sspk + 1e-9) / (rspk + 1e-9)
        hl_real = np.nanmedian([halflife_arr(rw[:, j], h) for rw, h in zip(reals, hod)])
        hl_sims = [halflife_arr(sw[i, :, j], h)
                   for sw, h in zip(sims, hod) for i in range(0, sw.shape[0], 10)]
        hl_ratio = np.nanmedian(hl_sims) / hl_real
        ks_ds = [stats.ks_2samp(rw[:, j], sw[i, :, j]).statistic
                 for rw, sw in zip(reals, sims) for i in range(0, sw.shape[0], 25)]
        res = {
            "a_mean": mean_off <= CHECKS["mean_offset_max"],
            "a_std": CHECKS["std_ratio"][0] <= std_ratio <= CHECKS["std_ratio"][1],
            "a_neg": CHECKS["neg_freq_ratio"][0] <= neg_ratio <= CHECKS["neg_freq_ratio"][1],
            "b_spike": CHECKS["spike_ratio"][0] <= spike_ratio <= CHECKS["spike_ratio"][1],
            "c_halflife": CHECKS["halflife_ratio"][0] <= hl_ratio <= CHECKS["halflife_ratio"][1],
        }
        flat.update({f"{lab}.{k}": bool(vv) for k, vv in res.items()})
        rows.append({"node": lab, "real_mean": r.mean(), "sim_mean": s.mean(),
                     "real_std": r.std(), "sim_std": s.std(),
                     "mean_offset_sd": mean_off, "std_ratio": std_ratio,
                     "neg_ratio": neg_ratio, "spike_ratio": spike_ratio,
                     "real_halflife_h": hl_real,
                     "sim_halflife_h": np.nanmedian(hl_sims),
                     "halflife_ratio": hl_ratio, "ks_D_median": np.median(ks_ds),
                     **res})
        print(f"  {lab}: mean off {mean_off:.3f}  std ratio {std_ratio:.3f}"
              f"  neg {neg_ratio:.3f}  spike {spike_ratio:.3f}"
              f"  hl {hl_ratio:.3f}  ksD {np.median(ks_ds):.3f}", flush=True)
    return rows, flat


def calibration_checks(reals, sims, nodes, seed=4):
    rng = np.random.default_rng(seed)
    sim = np.stack(sims)
    real = np.stack(reals)
    n_bins = PATHS_PER_WINDOW + 1
    rows = []
    for j, node in enumerate(nodes):
        r = np.concatenate([ranks_of(real[w, :, j], sim[w, :, :, j], rng)
                            for w in range(real.shape[0])])
        counts = np.bincount(r, minlength=n_bins)
        block = autocorr_block(r.astype(float))
        lin, sq = contrasts(counts)
        pit = r / (n_bins - 1)
        lo, hi, _ = block_ci(pit, block, rng)
        blo, bhi, _ = block_ci((r == 0).astype(float), block, rng)
        alo, ahi, _ = block_ci((r == n_bins - 1).astype(float), block, rng)
        rows.append({"node": NODE_LABELS[node], "block_len": block,
                     "mean_PIT": pit.mean(), "PIT_ci_lo": lo, "PIT_ci_hi": hi,
                     "bias_rejects_at_05": not (lo <= 0.5 <= hi),
                     "linear_contrast": lin, "squared_contrast": sq,
                     "frac_below_ensemble": (r == 0).mean(),
                     "below_ci_lo": blo, "below_ci_hi": bhi,
                     "frac_above_ensemble": (r == n_bins - 1).mean(),
                     "above_ci_lo": alo, "above_ci_hi": ahi,
                     "expected_tail_frac": 1 / n_bins,
                     "tail_rejects_at_05": not (blo <= 1 / n_bins <= bhi)})
        print(f"  {NODE_LABELS[node]}: mean PIT {pit.mean():.3f} "
              f"CI [{lo:.3f}, {hi:.3f}] -> "
              f"{'REJECT' if not (lo <= 0.5 <= hi) else 'no reject'}"
              f"   below {(r == 0).mean() / (1 / n_bins):.1f}x"
              f"  above {(r == n_bins - 1).mean() / (1 / n_bins):.1f}x", flush=True)
    return rows


def evaluate(ckpt, label):
    model, scaler, stage = load_model(ckpt)
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    val = splits["val"][0]
    nodes = list(val.columns)
    span = WINDOW_DAYS * 24
    starts = window_starts(index, val.index[0], N_WINDOWS, span)

    with torch.no_grad():
        T = {n: tuple(torch.tensor(a[split_slice(index, n)])
                      for a in (x, off, E, v)) for n in ["val"]}
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                  Eva[:-1], vva[:-1]).item()
    print(f"\n=== {label} (stage {stage}) ===\nheld-out one-step NLL {va:.4f}", flush=True)

    reals = [lmp.values[i0 + 1:i0 + 1 + span] for i0 in starts]
    hod = [index[i0 + 1:i0 + 1 + span].tz_convert("US/Pacific").hour for i0 in starts]

    # checks and calibration use separate seeds (100+k and 700+k) so the two
    # verdicts never share a Monte Carlo draw
    print("checks (sde.validate thresholds):", flush=True)
    check_rows, flat = moment_checks(reals, simulate_windows(
        model, scaler, x, E, v, off, starts, span, 100), hod, nodes)
    n_pass = sum(flat.values())
    print(f"  {n_pass}/{len(flat)} checks passed"
          + ("" if n_pass == len(flat) else
             "   FAILING: " + ",".join(k for k, vv in flat.items() if not vv)), flush=True)

    print("calibration (sde.calibration rank histogram):", flush=True)
    cal_rows = calibration_checks(reals, simulate_windows(
        model, scaler, x, E, v, off, starts, span, 700), nodes)
    cal_rejects = any(r["bias_rejects_at_05"] or r["tail_rejects_at_05"] for r in cal_rows)
    pit_ok = all(not r["bias_rejects_at_05"] for r in cal_rows)
    print(f"  calibration: {'rejected' if cal_rejects else 'not rejected'}"
          f"   (mean-PIT CIs include 0.5 on all hubs: {pit_ok})", flush=True)

    np.random.seed(5)
    hl = factor_halflives(model, scaler, x, E, v, off)
    level, s1, s2 = float(hl[0]), float(hl[1]), float(hl[2])
    ratio = (s1 + s2) / 2 / level
    print(f"timescales: level {level:.2f}h  spreads {s1:.2f}/{s2:.2f}h"
          f"  ratio {ratio:.3f}   (observed 13.04h / 0.383)", flush=True)

    summary = {
        "label": label, "checkpoint": Path(ckpt).name, "stage": stage,
        "val_nll": va, "checks_passed": n_pass, "checks_total": len(flat),
        "all_checks_pass": n_pass == len(flat),
        "pit_all_include_half": pit_ok, "calibration_rejected": cal_rejects,
        "halflife_level_h": level, "halflife_spread1_h": s1,
        "halflife_spread2_h": s2, "spread_over_level": ratio,
        "PRIMARY_pass": pit_ok,
        "SECONDARY_pass": level >= 8.0 and ratio <= 0.6,
        "MUSTNOTREGRESS_pass": (n_pass == len(flat)) and va <= -2.80,
    }
    for r in cal_rows:
        summary[f"meanPIT_{r['node']}"] = r["mean_PIT"]
        summary[f"PITlo_{r['node']}"] = r["PIT_ci_lo"]
        summary[f"PIThi_{r['node']}"] = r["PIT_ci_hi"]
    for r in check_rows:
        summary[f"halflife_ratio_{r['node']}"] = r["halflife_ratio"]
        summary[f"mean_offset_{r['node']}"] = r["mean_offset_sd"]
        summary[f"ksD_{r['node']}"] = r["ks_D_median"]

    print(f"PRIMARY {'PASS' if summary['PRIMARY_pass'] else 'FAIL'}   "
          f"SECONDARY {'PASS' if summary['SECONDARY_pass'] else 'FAIL'}   "
          f"MUST-NOT-REGRESS {'PASS' if summary['MUSTNOTREGRESS_pass'] else 'FAIL'}",
          flush=True)
    return summary, check_rows, cal_rows


def main():
    ckpt = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else ckpt.stem
    summary, check_rows, cal_rows = evaluate(ckpt, label)
    tag = label.replace(" ", "_").replace("/", "_")
    pd.DataFrame(check_rows).round(4).to_csv(OUT / f"eval_mh_checks_{tag}.csv", index=False)
    pd.DataFrame(cal_rows).round(4).to_csv(OUT / f"eval_mh_cal_{tag}.csv", index=False)
    row = pd.DataFrame([summary]).round(4)
    path = OUT / "mh_comparison.csv"
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev["label"] != label]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(path, index=False)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
