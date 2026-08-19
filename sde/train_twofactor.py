import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchsde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from plotstyle import ACCENT, NODE_COLORS, NODE_LABELS, apply
from sde.ks_test import run_ks_check
from sde.model import (DIM, HubSDE, Scaler, WindowSDE, ema_update, full_series,
                       load_frames)
from sde.model_twofactor import TwoFactorSDE
from sde.train import (CENTER_WEIGHT, NLL_BUDGET, SEED, sample_windows,
                       split_slice, stage_a, window_moments)
from sde.validate import (CHECKS, PATHS_PER_WINDOW, SPIKE_USD, WINDOW_DAYS,
                          halflife_arr, run_window, window_starts)

OUT = ROOT / "sde"
FIG = ROOT / "writeup" / "figures"
N_WINDOWS = 4
HL_PATHS, HL_HOURS = 64, 24 * 30


# sde.train.rollout with the jump sizes drawn per factor and rotated
def rollout(model, x0, ow, E0, v0, span, jump_occ, comp, eps, adjoint_params):
    sde = WindowSDE(model, ow)
    x, E, v = x0, E0, v0
    path = [x0]
    for h in range(span):
        ts = torch.tensor([float(h), float(h + 1)])
        x_prev = x
        y = torchsde.sdeint_adjoint(sde, sde.pack(x, E, v), ts, method="euler",
                                    dt=0.5, adjoint_params=adjoint_params)[-1]
        x = sde.unpack(y)[0] + jump_occ[h].unsqueeze(1) * model.jump_sizes(comp[h], eps[h])
        E, v = ema_update(E, v, x_prev, x)
        path.append(x)
    return torch.stack(path, dim=1)


# sde.train.stage_b unchanged apart from the rotated jump draw; same moment
# vector, weights, NLL budget and centering penalty
def stage_b(model, T, nll_anchor, spike_thr, neg_thr, iters=300, batch=96,
            span=72, lr=1e-4):
    xtr, otr, Etr, vtr = T["train"]
    xva, ova, Eva, vva = T["val"]
    gen = torch.Generator().manual_seed(SEED + 2)
    with torch.no_grad():
        tgt = torch.stack([
            window_moments(sample_windows(xtr, otr, Etr, vtr, span, 256, gen)[0],
                           spike_thr, neg_thr)
            for _ in range(40)]).mean(dim=0)
    scale = tgt.abs().clamp_min(0.05)
    weights = torch.ones_like(tgt)
    weights[15:18] = 3.0

    for p in [model.lam_raw, model.jump_w_raw]:
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    log, best, best_any = [], (np.inf, None), (np.inf, None, None)
    for it in range(iters):
        xw, ow, E0, v0, _ = sample_windows(xtr, otr, Etr, vtr, span, batch, gen)
        with torch.no_grad():
            jump_occ = torch.stack([torch.rand(batch, generator=gen) <
                                    model.lam(torch.tensor(float(h + 1)), ow)
                                    for h in range(span)])
            comp = torch.where(torch.rand(span, batch, generator=gen) <
                               torch.sigmoid(model.jump_w_raw), 0, 1)
            eps = torch.randn(span, batch, DIM, generator=gen)
        sim = rollout(model, xw[:, 0], ow, E0, v0, span, jump_occ, comp, eps, params)
        smm = (weights * ((window_moments(sim, spike_thr, neg_thr) - tgt) / scale) ** 2).mean()
        idx = torch.randint(0, xtr.shape[0] - 1, (1024,), generator=gen)
        nll = model.transition_nll(xtr[idx], xtr[idx + 1], otr[idx], Etr[idx], vtr[idx])
        t0 = torch.zeros(1)
        mu = model.drift(xtr[idx], t0, otr[idx], Etr[idx], vtr[idx]) \
            + model.jump_compensator(t0, otr[idx])
        loss = smm + 1.0 * nll + CENTER_WEIGHT * (mu.mean(dim=0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        log.append((it, smm.item(), nll.item()))
        if it % 20 == 0 or it == iters - 1:
            with torch.no_grad():
                va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                          Eva[:-1], vva[:-1]).item()
            print(f"iter {it:3d}  smm {smm.item():.4f}  nll {nll.item():.4f}  val nll {va:.4f}")
            if va <= nll_anchor + NLL_BUDGET and smm.item() < best[0]:
                best = (smm.item(), {k: p.clone() for k, p in model.state_dict().items()})
            if smm.item() < best_any[0]:
                best_any = (smm.item(), va,
                            {k: p.clone() for k, p in model.state_dict().items()})
    for p in model.parameters():
        p.requires_grad_(True)
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"stage B kept checkpoint with smm {best[0]:.4f} within NLL budget")
    else:
        print("stage B never met the NLL budget; keeping stage A weights")
    print(f"stage B best smm overall {best_any[0]:.4f} at val nll {best_any[1]:.4f}")
    return (best[1] is not None, best_any,
            pd.DataFrame(log, columns=["iter", "smm", "nll"]))


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _, _, splits = load_frames(ROOT)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, E, v, off = full_series(ROOT, scaler)

    T = {}
    for name in ["train", "val"]:
        m = split_slice(index, name)
        T[name] = tuple(torch.tensor(a[m]) for a in (x, off, E, v))
    model = TwoFactorSDE()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"train {T['train'][0].shape[0]}h  val {T['val'][0].shape[0]}h  "
          f"parameters {n_par}")

    nll_a, log_a = stage_a(model, T)
    spike_thr = torch.tensor(25.0 / scaler.std)
    neg_thr = torch.tensor(-scaler.mean / scaler.std)
    used_b, _, log_b = stage_b(model, T, nll_a, spike_thr, neg_thr)
    with torch.no_grad():
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1], Eva[:-1], vva[:-1]).item()
    print(f"final val nll {va:.4f} (stage {'B' if used_b else 'A'})")
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "stage": "B" if used_b else "A"}, OUT / "checkpoint_twofactor.pt")
    log_a.to_csv(OUT / "train_log_twofactor_a.csv", index=False)
    log_b.to_csv(OUT / "train_log_twofactor_b.csv", index=False)
    report_rates(model, scaler)


def report_rates(model, scaler):
    index, x, E, v, off = full_series(ROOT, scaler)
    m = split_slice(index, "train")
    args = [torch.tensor(a[m]) for a in (x, off, E, v)]
    with torch.no_grad():
        _, _, w, _ = model.inputs(args[0], torch.zeros(1), args[1], args[2], args[3])
        K = model.rates(w).numpy()
    hl = np.log(2) / K
    for i, lab in enumerate(("level", "spread C1", "spread C2")):
        print(f"  fitted K_{lab}: median {np.median(K[:, i]):.4f}/h "
              f"(implied half-life {np.median(hl[:, i]):.2f}h)")


def stage_a_diagnostic():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _, _, splits = load_frames(ROOT)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, E, v, off = full_series(ROOT, scaler)
    T = {}
    for name in ["train", "val"]:
        m = split_slice(index, name)
        T[name] = tuple(torch.tensor(a[m]) for a in (x, off, E, v))

    model = TwoFactorSDE()
    nll_a, _ = stage_a(model, T)
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "stage": "A"}, OUT / "checkpoint_twofactor_stage_a.pt")
    print(f"two-factor stage A: val nll {nll_a:.4f}")
    report_rates(model, scaler)
    hl = factor_halflives(model, scaler, x, E, v, off)
    print(f"two-factor stage A half-lives {np.round(hl, 2)}  "
          f"spread/level ratio {np.mean(hl[1:]) / hl[0]:.2f}  "
          f"diffusion level share {diffusion_split(model, x, E, v, off):.4f}")

    inc, sc_i, _ = load_incumbent(OUT / "checkpoint_stage_a.pt")
    index, xi, Ei, vi, offi = full_series(ROOT, sc_i)
    hl_i = factor_halflives(inc, sc_i, xi, Ei, vi, offi)
    print(f"incumbent stage A half-lives {np.round(hl_i, 2)}  "
          f"spread/level ratio {np.mean(hl_i[1:]) / hl_i[0]:.2f}  "
          f"diffusion level share {diffusion_split(inc, xi, Ei, vi, offi):.4f}")


def load_two(path):
    ck = torch.load(path, weights_only=False)
    model = TwoFactorSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Scaler(ck["scaler"]["mean"], ck["scaler"]["std"]), ck.get("stage", "?")


def load_incumbent(path):
    ck = torch.load(path, weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Scaler(ck["scaler"]["mean"], ck["scaler"]["std"]), ck.get("stage", "?")


# the price-model validation protocol from sde/validate.py, unmodified: four
# re-anchored 14-day windows, 50 paths each, same statistics and thresholds
def check_metrics(model, scaler, x, E, v, off, index, lmp, starts, span_h, nodes):
    sims, reals, hod = [], [], []
    for k, i0 in enumerate(starts):
        sim = run_window(model, scaler, x, E, v, off, i0, span_h,
                         PATHS_PER_WINDOW, seed=100 + k)
        sims.append(sim[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span_h])
        hod.append(index[i0 + 1:i0 + 1 + span_h].tz_convert("US/Pacific").hour)
    sim_pool = np.concatenate([s.reshape(-1, DIM) for s in sims])
    real_pool = np.concatenate(reals)
    _, ks_df = run_ks_check(reals, sims, hod, nodes)

    rows = []
    for j, node in enumerate(nodes):
        r, s = real_pool[:, j], sim_pool[:, j]
        mean_off = abs(s.mean() - r.mean()) / r.std()
        std_ratio = s.std() / r.std()
        neg_ratio = ((s < 0).mean() + 1e-9) / ((r < 0).mean() + 1e-9)

        rspk = np.mean([(np.abs(np.diff(rw[:, j])) > SPIKE_USD).mean() for rw in reals])
        sspk = np.mean([(np.abs(np.diff(sw[:, :, j], axis=1)) > SPIKE_USD).mean()
                        for sw in sims])
        spike_ratio = (sspk + 1e-9) / (rspk + 1e-9)

        hl_real = np.nanmedian([halflife_arr(rw[:, j], h) for rw, h in zip(reals, hod)])
        hl_sim = np.nanmedian([halflife_arr(sw[i, :, j], h)
                               for sw, h in zip(sims, hod)
                               for i in range(0, sw.shape[0], 10)])
        rows.append({
            "node": NODE_LABELS[node],
            "real_mean": r.mean(), "sim_mean": s.mean(),
            "real_std": r.std(), "sim_std": s.std(),
            "real_pct_neg": 100 * (r < 0).mean(), "sim_pct_neg": 100 * (s < 0).mean(),
            "real_spikes_per_1000h": 1000 * rspk, "sim_spikes_per_1000h": 1000 * sspk,
            "real_halflife_h": hl_real, "sim_halflife_h": hl_sim,
            "mean_offset_sd": mean_off, "std_ratio": std_ratio, "neg_ratio": neg_ratio,
            "spike_ratio": spike_ratio, "halflife_ratio": hl_sim / hl_real,
            "ks_D": ks_df["ks_D"][j], "ks_p_block": ks_df["p_block_bootstrap"][j],
            "ks_D_deseason": ks_df["ks_D_deseason"][j],
            "ks_p_block_deseason": ks_df["p_block_deseason"][j],
            "ks_null_D95": ks_df["null_D95"][j],
            "a_mean": mean_off <= CHECKS["mean_offset_max"],
            "a_std": CHECKS["std_ratio"][0] <= std_ratio <= CHECKS["std_ratio"][1],
            "a_neg": CHECKS["neg_freq_ratio"][0] <= neg_ratio <= CHECKS["neg_freq_ratio"][1],
            "b_spike": CHECKS["spike_ratio"][0] <= spike_ratio <= CHECKS["spike_ratio"][1],
            "c_halflife": (CHECKS["halflife_ratio"][0] <= hl_sim / hl_real
                           <= CHECKS["halflife_ratio"][1]),
        })
    return rows, reals, sims, hod


# congestion_link.py's protocol: 64 paths over the last 30 days, half-lives
# of the deseasonalized level and the two hub spreads
def factor_halflives(model, scaler, x, E, v, off):
    i0 = len(x) - HL_HOURS
    sim = run_window(model, scaler, x, E, v, off, i0, HL_HOURS, HL_PATHS, seed=5)
    hs = (off[i0, 0] + np.arange(sim.shape[1])) % 24
    return np.nanmedian([[halflife_arr(s.mean(1), hs),
                          halflife_arr(s[:, 0] - s[:, 1], hs),
                          halflife_arr(s[:, 0] - s[:, 2], hs)] for s in sim], axis=0)


def observed_halflives(lmp, cong, energy):
    hod = lmp.index.tz_convert("US/Pacific").hour.values
    P = lmp.values
    real = [halflife_arr(P.mean(1), hod), halflife_arr(P[:, 0] - P[:, 1], hod),
            halflife_arr(P[:, 0] - P[:, 2], hod)]
    comp = [halflife_arr(energy[:, 0], hod),
            halflife_arr(cong[:, 0] - cong[:, 1], hod),
            halflife_arr(cong[:, 0] - cong[:, 2], hod)]
    return np.array(real), np.array(comp)


def free_run(model, scaler, x, E, v, off, i0, n_hours):
    sim = run_window(model, scaler, x, E, v, off, i0, n_hours, 50, seed=555)
    return sim[:, 1:].mean(axis=(0, 1)), sim[:, 1:].std(axis=(0, 1))


# read off the state-averaged covariance S S', not the correlation factor,
# since the two models parameterize correlation in different bases
def diffusion_split(model, x, E, v, off, stride=7):
    idx = np.arange(0, len(x) - 1, stride)
    args = [torch.tensor(a[idx]) for a in (x, off, E, v)]
    with torch.no_grad():
        S = model.sigma_tril(args[0], torch.zeros(1), args[1], args[2], args[3])
        Sig = (S @ S.transpose(1, 2)).mean(0).numpy()
    q = np.ones(DIM) / np.sqrt(DIM)
    return float(q @ Sig @ q / np.trace(Sig))


def evaluate():
    apply()
    lmp, cong, splits = load_frames(ROOT)
    energy = pd.read_csv(ROOT / "data/processed/energy_hourly.csv",
                         index_col=0, parse_dates=True).values
    val = splits["val"][0]
    nodes = list(lmp.columns)
    span_h = WINDOW_DAYS * 24
    hl_real, hl_comp = observed_halflives(lmp, cong.values, energy)

    models = {
        "incumbent (per-hub)": load_incumbent(OUT / "checkpoint.pt"),
        "two-factor": load_two(OUT / "checkpoint_twofactor.pt"),
    }
    out, packs, hls = [], {}, {}
    for name, (model, scaler, stage) in models.items():
        index, x, E, v, off = full_series(ROOT, scaler)
        starts = window_starts(index, val.index[0], N_WINDOWS, span_h)
        vm = (index >= val.index[0]) & (index <= val.index[-1])
        xv, ov, Ev, vv = (torch.tensor(a[vm]) for a in (x, off, E, v))
        with torch.no_grad():
            nll = model.transition_nll(xv[:-1], xv[1:], ov[:-1], Ev[:-1], vv[:-1]).item()
        print(f"\n=== {name} (stage {stage})  held-out val NLL {nll:.4f} ===")

        rows, reals, sims, hod = check_metrics(model, scaler, x, E, v, off, index,
                                              lmp, starts, span_h, nodes)
        fr_mean, fr_std = free_run(model, scaler, x, E, v, off, starts[0], len(val) - 1)
        hl_sim = factor_halflives(model, scaler, x, E, v, off)
        hls[name] = hl_sim
        packs[name] = (reals, sims, hod)

        n_fail = 0
        for j, row in enumerate(rows):
            row["model"] = name
            row["val_nll"] = nll
            row["n_params"] = sum(p.numel() for p in model.parameters())
            row["freerun_mean"] = fr_mean[j]
            row["freerun_std"] = fr_std[j]
            row["freerun_real_mean"] = val.values[1:, j].mean()
            row["freerun_real_std"] = val.values[1:, j].std()
            row["halflife_level_h"] = hl_sim[0]
            row["halflife_spread1_h"] = hl_sim[1]
            row["halflife_spread2_h"] = hl_sim[2]
            row["spread_over_level_halflife"] = np.mean(hl_sim[1:]) / hl_sim[0]
            row["diffusion_level_share"] = diffusion_split(model, x, E, v, off)
            n_fail += sum(not row[k] for k in
                          ["a_mean", "a_std", "a_neg", "b_spike", "c_halflife"])
            print(f"  {row['node']}: mean {row['real_mean']:.1f}/{row['sim_mean']:.1f}"
                  f"  std {row['real_std']:.1f}/{row['sim_std']:.1f}"
                  f"  neg% {row['real_pct_neg']:.1f}/{row['sim_pct_neg']:.1f}"
                  f"  spikes/1000h {row['real_spikes_per_1000h']:.1f}/"
                  f"{row['sim_spikes_per_1000h']:.1f}"
                  f"  halflife {row['real_halflife_h']:.1f}/{row['sim_halflife_h']:.1f}h")
        print(f"  checks {15 - n_fail}/15   KS D {[round(r['ks_D'], 3) for r in rows]}"
              f"  p {[round(r['ks_p_block'], 4) for r in rows]}")
        print(f"  deseasonalized half-lives (h) [level, NP-SP, NP-ZP] {np.round(hl_sim, 2)}"
              f"   spread/level ratio {np.mean(hl_sim[1:]) / hl_sim[0]:.2f}")
        print(f"  free-run 59d means {np.round(fr_mean, 1)} vs real "
              f"{np.round(val.values[1:].mean(axis=0), 1)}")
        out += rows

    sc = models["incumbent (per-hub)"][1]
    dxs = np.diff(sc.transform(lmp.values.astype(np.float32)), axis=0)
    C = np.cov(dxs.T)
    q = np.ones(DIM) / np.sqrt(DIM)
    emp_share = float(q @ C @ q / np.trace(C))
    for lab, arr in (("observed prices", hl_real), ("CAISO components", hl_comp)):
        out.append({"model": lab, "node": "-",
                    "halflife_level_h": arr[0], "halflife_spread1_h": arr[1],
                    "halflife_spread2_h": arr[2],
                    "spread_over_level_halflife": np.mean(arr[1:]) / arr[0],
                    "diffusion_level_share": emp_share if lab == "observed prices" else None})
    print(f"realized hourly increments carry {emp_share:.4f} of their variance "
          f"in the level direction")
    print(f"\nreference half-lives: observed {np.round(hl_real, 2)} "
          f"ratio {np.mean(hl_real[1:]) / hl_real[0]:.2f}; "
          f"CAISO components {np.round(hl_comp, 2)} "
          f"ratio {np.mean(hl_comp[1:]) / hl_comp[0]:.2f}")

    cols = ["model", "node", "n_params", "val_nll",
            "ks_D", "ks_p_block", "ks_D_deseason", "ks_p_block_deseason", "ks_null_D95",
            "real_mean", "sim_mean", "real_std", "sim_std",
            "real_pct_neg", "sim_pct_neg",
            "real_spikes_per_1000h", "sim_spikes_per_1000h",
            "real_halflife_h", "sim_halflife_h",
            "mean_offset_sd", "std_ratio", "neg_ratio", "spike_ratio", "halflife_ratio",
            "a_mean", "a_std", "a_neg", "b_spike", "c_halflife",
            "halflife_level_h", "halflife_spread1_h", "halflife_spread2_h",
            "spread_over_level_halflife", "diffusion_level_share",
            "freerun_mean", "freerun_std", "freerun_real_mean", "freerun_real_std"]
    df = pd.DataFrame(out).reindex(columns=cols)
    df.to_csv(OUT / "twofactor_comparison.csv", index=False, float_format="%.6g")
    print(f"wrote {OUT / 'twofactor_comparison.csv'}")
    figure(df, packs, hls, hl_real, hl_comp, nodes)


def figure(df, packs, hls, hl_real, hl_comp, nodes):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    labels = list(packs)

    ax = axes[0, 0]
    series = [("CAISO components", hl_comp, ACCENT),
              ("observed prices", hl_real, "#333333"),
              (labels[0], hls[labels[0]], NODE_COLORS[nodes[0]]),
              (labels[1], hls[labels[1]], NODE_COLORS[nodes[1]])]
    w = 0.2
    for i, (lab, arr, c) in enumerate(series):
        ax.bar(np.arange(3) + (i - 1.5) * w, arr, w, label=lab, color=c)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["level\n(energy)", "NP15-SP15\n(congestion)", "NP15-ZP26"])
    ax.set_ylabel("deseasonalized half-life (h)")
    ax.set_title("Deseasonalized Mean-Reversion Half-Lives")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    q = np.linspace(0.005, 0.995, 199)
    for lab, style in zip(labels, ("--", "-")):
        reals, sims, _ = packs[lab]
        rp = np.concatenate(reals)
        sp = np.concatenate([s.reshape(-1, DIM) for s in sims])
        for j, node in enumerate(nodes):
            ax.plot(np.quantile(rp[:, j], q), np.quantile(sp[:, j], q), style,
                    color=NODE_COLORS[node], lw=1.2,
                    label=f"{NODE_LABELS[node]}, {lab}")
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="#999999", zorder=0)
    ax.set_xlabel(r"real quantile (\$/MWh)")
    ax.set_ylabel(r"simulated quantile (\$/MWh)")
    ax.set_title("Marginal QQ over the four validation windows")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 0]
    sub = df[df["node"] != "-"]
    for i, lab in enumerate(labels):
        d = sub[sub["model"] == lab]
        ax.bar(np.arange(3) + (i - 0.5) * 0.35, d["ks_D"].values, 0.35, label=lab,
               color=[NODE_COLORS[nodes[0]], NODE_COLORS[nodes[1]]][i])
    d95 = sub.groupby("model")["ks_null_D95"].mean().max()
    ax.axhline(d95, color=ACCENT, ls=":", lw=1.0,
               label="block-bootstrap 95% null")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([NODE_LABELS[n] for n in nodes])
    ax.set_ylabel("KS statistic $D$")
    verdict = {lab: "pass" if (sub[sub["model"] == lab]["ks_p_block"] > 0.05).all()
               else "reject" for lab in labels}
    ax.set_title("Marginal distribution: "
                 + ", ".join(f"{k} {v}" for k, v in verdict.items()))
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for lab, style in zip(labels, ("--", "-")):
        d = sub[sub["model"] == lab]
        ax.plot(np.arange(3), d["halflife_ratio"].values, style, marker="o", ms=4,
                color=[NODE_COLORS[nodes[0]], NODE_COLORS[nodes[1]]][labels.index(lab)],
                label=lab)
    ax.axhspan(0.5, 2.0, color="#dddddd", zorder=0, label="acceptance tolerance")
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([NODE_LABELS[n] for n in nodes])
    ax.set_ylabel("simulated / real half-life")
    ax.set_title("Half-life ratio, per hub")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "twofactor_variant.pdf")
    print(f"wrote {FIG / 'twofactor_variant.pdf'}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"):
        train()
    if what in ("eval", "all"):
        evaluate()
    if what == "stagea":
        stage_a_diagnostic()
