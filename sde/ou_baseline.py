import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import linalg, optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotstyle import NODE_LABELS
from sde.ks_test import run_ks_check
from sde.model import DIM, LAMBDA_CAP, clock_features, full_series, load_frames
from sde.validate import (CHECKS, PATHS_PER_WINDOW, SPIKE_USD, WINDOW_DAYS,
                          halflife_arr, load_model, run_window, window_starts)

N_WINDOWS = 4
EM_ITERS = 250


def as_off(off):
    return off if torch.is_tensor(off) else torch.as_tensor(np.asarray(off))


def harmonics(off, t=0.0):
    c = clock_features(t, as_off(off))
    return torch.cat([torch.ones(c.shape[0], 1), c], dim=1)


def lam_design(off, t=0.0):
    tau = (as_off(off)[:, 0] + t) % 24.0
    two_pi = 2 * np.pi
    return torch.stack([torch.ones_like(tau),
                        torch.sin(two_pi * tau / 24.0), torch.cos(two_pi * tau / 24.0),
                        torch.sin(2 * two_pi * tau / 24.0), torch.cos(2 * two_pi * tau / 24.0)],
                       dim=1)


# Exact Gaussian MLE for Y_t = C z_t + eps: identical regressors in every
# equation, so the seemingly-unrelated system collapses to per-row OLS.
def ols(Y, Z):
    C = np.linalg.solve(Z.T @ Z, Z.T @ Y).T
    R = Y - Z @ C.T
    return C, R.T @ R / len(Y)


# Maps the fitted one-step law (A, V) back to the continuous generator: K =
# -log A, and Sigma follows from the discrete Lyapunov solution V = P - A P A'
# for stationary P solving KP + PK' = Sigma.
def continuous_params(A, V):
    Klog = linalg.logm(A)
    if np.iscomplexobj(Klog):
        if np.abs(Klog.imag).max() > 1e-8:
            raise ValueError("matrix log of A is not real; A has complex eigenvalues")
        Klog = Klog.real
    K = -Klog
    P = linalg.solve_discrete_lyapunov(A, V)
    Sig = K @ P + P @ K.T
    Sig = 0.5 * (Sig + Sig.T)
    ev = np.linalg.eigvalsh(Sig)
    if ev.min() <= 0:
        print(f"  recovered Sigma not PD (min eig {ev.min():.2e}); clipping")
        w, Q = np.linalg.eigh(Sig)
        Sig = Q @ np.diag(w.clip(1e-8)) @ Q.T
    return K, Sig


# Numeric check of V = int_0^1 e^{-Ks} Sigma e^{-K's} ds.
def transition_cov(K, Sig, n=200):
    s = (np.arange(n) + 0.5) / n
    tot = np.zeros_like(Sig)
    for si in s:
        Es = linalg.expm(-K * si)
        tot += Es @ Sig @ Es.T
    return tot / n


def mvn_logpdf(e, mean, cov):
    d = e - mean
    L = np.linalg.cholesky(cov)
    z = linalg.solve_triangular(L, d.T, lower=True)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * DIM * np.log(2 * np.pi)


# dX = K(theta(t) - X) dt + Sigma^{1/2} dW + dJ, with theta(t) = Theta h(t) a
# deterministic Fourier profile (or E_t + Theta h(t) for the level-anchored
# variant), exposing the same drift/sigma_tril/sample_jumps interface as HubSDE
# so sde.model.simulate integrates it unchanged.
class SeasonalOU(nn.Module):

    def __init__(self, A, B, V, jump=None, anchored=False, cvec=None):
        super().__init__()
        K, Sig = continuous_params(A, V)
        self.anchored = anchored
        self.has_jumps = jump is not None
        self.gap = np.abs(transition_cov(K, Sig) - V).max()
        Theta = np.linalg.solve(np.eye(DIM) - A, B)
        if cvec is None:
            cvec = np.ones(24)
        for name, arr in [("A", A), ("B", B), ("V", V), ("K", K),
                          ("Sig", Sig), ("Theta", Theta), ("cvec", cvec),
                          ("Lc", np.linalg.cholesky(Sig)),
                          ("Lv", np.linalg.cholesky(V))]:
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))
        if jump is not None:
            for name in ["lam_coef", "jump_w", "jump_mean", "jump_sig"]:
                self.register_buffer(name, torch.tensor(jump[name], dtype=torch.float32))

    def theta(self, t, off0, E):
        prof = harmonics(off0, t) @ self.Theta.T
        return E + prof if self.anchored else prof

    def drift(self, x, t, off0, E, v):
        return (self.theta(t, off0, E) - x) @ self.K.T

    def hod_scale(self, t, off0):
        b = ((as_off(off0)[:, 0] + t) % 24.0).long()
        return self.cvec[b].sqrt()

    def sigma_tril(self, x, t, off0, E, v):
        s = self.hod_scale(t, off0).reshape(-1, 1, 1)
        return s * self.Lc.unsqueeze(0)

    def lam(self, t, off0):
        if not self.has_jumps:
            return torch.zeros(np.asarray(off0).shape[0])
        return LAMBDA_CAP * torch.sigmoid(lam_design(off0, t) @ self.lam_coef)

    def sample_jumps(self, t, off0, gen=None):
        B = off0.shape[0]
        if not self.has_jumps:
            return torch.zeros(B, DIM)
        lam = self.lam(float(t), off0)
        occurs = torch.rand(B, generator=gen) < lam
        comp = torch.where(torch.rand(B, generator=gen) < self.jump_w, 0, 1)
        eps = torch.randn(B, DIM, generator=gen)
        return occurs.unsqueeze(1) * (self.jump_mean[comp] + self.jump_sig[comp] * eps)

    # Exact one-step law of the fitted process, truncated at one jump per hour
    # to match how the neural model's likelihood is computed.
    def transition_nll(self, x0, x1, off0, E0, v0, dt=1.0):
        mean0 = E0 + (x0 - E0) @ self.A.T if self.anchored else x0 @ self.A.T
        mean0 = mean0 + harmonics(off0, 0.0) @ self.B.T
        c = self.hod_scale(0.0, off0) ** 2
        Vt = c.reshape(-1, 1, 1) * self.V
        base = torch.distributions.MultivariateNormal(
            mean0, scale_tril=torch.linalg.cholesky(Vt))
        if not self.has_jumps:
            return -base.log_prob(x1).mean()
        lam = self.lam(0.0, off0)
        logps = [torch.log1p(-lam) + base.log_prob(x1)]
        for k, wk in [(0, self.jump_w), (1, 1 - self.jump_w)]:
            covk = Vt + torch.diag(self.jump_sig[k] ** 2)
            comp = torch.distributions.MultivariateNormal(
                mean0 + self.jump_mean[k], scale_tril=torch.linalg.cholesky(covk))
            logps.append(torch.log(lam * wk + 1e-12) + comp.log_prob(x1))
        return -torch.logsumexp(torch.stack(logps, dim=0), dim=0).mean()


def cond_mean(mod, x0, off0, E0, v0):
    if isinstance(mod, SeasonalOU):
        m = E0 + (x0 - E0) @ mod.A.T if mod.anchored else x0 @ mod.A.T
        m = m + harmonics(off0, 0.0) @ mod.B.T
        if not mod.has_jumps:
            return m
        size = mod.jump_w * mod.jump_mean[0] + (1 - mod.jump_w) * mod.jump_mean[1]
        return m + mod.lam(0.0, off0).unsqueeze(1) * size
    t0 = torch.zeros(1)
    return x0 + mod.drift(x0, t0, off0, E0, v0) + mod.jump_compensator(t0, off0)


# M-step for lambda(tau) = LAMBDA_CAP * sigmoid(H a); no closed form, so a
# five-parameter Bernoulli likelihood is maximized directly.
def fit_lam(r, H):
    def obj(a):
        lam = LAMBDA_CAP / (1 + np.exp(-H @ a))
        lam = np.clip(lam, 1e-9, 1 - 1e-9)
        return -(r * np.log(lam) + (1 - r) * np.log(1 - lam)).sum()
    return optimize.minimize(obj, np.array([-2.5, 0.0, 0.0, 0.0, 0.0]),
                             method="L-BFGS-B").x


# EM for the compound-Poisson term with jump size treated as latent: every
# component shares the diffusion covariance given the size, so the regression
# M-step stays (weighted) least squares on jump-corrected targets, and with hod
# supplied the diffusion covariance becomes c(tau) V, an hour-of-day variance
# profile fitted in the same loop.
def fit_jump_em(Y, Z, H, C0, V0, hod=None, iters=EM_ITERS):
    n = len(Y)
    C = C0.copy()
    e = Y - Z @ C.T
    hot = np.abs(e).max(1) > 2.5 * np.sqrt(np.diag(V0)).max()
    up = hot & (e.sum(1) > 0)
    dn = hot & (e.sum(1) <= 0)
    m = np.stack([e[up].mean(0), e[dn].mean(0)])
    s = np.stack([e[up].std(0), e[dn].std(0)]).clip(0.05)
    w = float(np.clip(up.sum() / max(hot.sum(), 1), 0.05, 0.95))
    a = np.array([-2.5, 0.0, 0.0, 0.0, 0.0])
    V = np.cov(e[~hot].T)
    bins = np.zeros(n, dtype=int) if hod is None else np.asarray(hod)
    nb = 1 if hod is None else 24
    sel = [bins == b for b in range(nb)]
    cnt = np.array([sl.sum() for sl in sel], dtype=float)
    c = np.ones(nb)
    print(f"  init: {hot.sum()} outlier transitions of {n}, w0 {w:.2f}")

    ll = -np.inf
    for it in range(iters):
        res = Y - Z @ C.T
        lam = np.clip(LAMBDA_CAP / (1 + np.exp(-H @ a)), 1e-9, 0.999)
        logs = np.empty((3, n))
        post_mu = [np.empty((n, DIM)), np.empty((n, DIM))]
        post_cov = np.zeros((nb, 2, DIM, DIM))
        for b in range(nb):
            sl = sel[b]
            Vb = c[b] * V
            Vbi = np.linalg.inv(Vb)
            logs[0, sl] = np.log1p(-lam[sl]) + mvn_logpdf(res[sl], 0.0, Vb)
            for k, wk in [(0, w), (1, 1 - w)]:
                Sk = np.diag(s[k] ** 2)
                logs[k + 1, sl] = (np.log(lam[sl] * wk + 1e-12)
                                   + mvn_logpdf(res[sl], m[k], Vb + Sk))
                Pc = np.linalg.inv(np.linalg.inv(Sk) + Vbi)
                post_cov[b, k] = Pc
                post_mu[k][sl] = (np.linalg.solve(Sk, m[k])[None, :]
                                  + res[sl] @ Vbi.T) @ Pc.T
        mx = logs.max(0)
        lse = mx + np.log(np.exp(logs - mx).sum(0))
        new_ll = lse.sum() / n
        r = np.exp(logs - lse)

        u = 1.0 / c[bins]
        Zbar = r[1][:, None] * post_mu[0] + r[2][:, None] * post_mu[1]
        Zw = Z * u[:, None]
        C = np.linalg.solve(Zw.T @ Z, Zw.T @ (Y - Zbar)).T
        res = Y - Z @ C.T

        Sb = np.zeros((nb, DIM, DIM))
        for b in range(nb):
            sl = sel[b]
            Sb[b] = (r[0][sl][:, None] * res[sl]).T @ res[sl]
            for k in range(2):
                d = res[sl] - post_mu[k][sl]
                Sb[b] += ((r[k + 1][sl][:, None] * d).T @ d
                          + r[k + 1][sl].sum() * post_cov[b, k])
        Vn = sum(Sb[b] / c[b] for b in range(nb)) / n
        V = 0.5 * (Vn + Vn.T)
        if nb > 1:
            Vi = np.linalg.inv(V)
            c = np.array([np.trace(Vi @ Sb[b]) / (DIM * max(cnt[b], 1))
                          for b in range(nb)]).clip(1e-3)
            kappa = float((c * cnt).sum() / cnt.sum())
            c, V = c / kappa, V * kappa

        pcs = post_cov[bins]
        for k in range(2):
            wk = r[k + 1]
            m[k] = (wk[:, None] * post_mu[k]).sum(0) / wk.sum()
            var = ((wk[:, None] * (post_mu[k] - m[k]) ** 2).sum(0)
                   + (wk[:, None] * np.diagonal(pcs[:, k], axis1=1, axis2=2)).sum(0)) / wk.sum()
            s[k] = np.sqrt(var).clip(0.05)
        w = np.clip(r[1].sum() / (r[1].sum() + r[2].sum()), 0.02, 0.98)
        a = fit_lam(np.clip(r[1] + r[2], 1e-9, 1 - 1e-9), H)

        if it % 20 == 0 or it == iters - 1:
            print(f"  EM {it:3d}  loglik/obs {new_ll:+.4f}  lam_bar {lam.mean():.4f}"
                  + (f"  c range {c.min():.2f}-{c.max():.2f}" if nb > 1 else ""))
        if new_ll < ll - 1e-6:
            print(f"  EM log-likelihood decreased at iter {it} ({ll:.5f} -> {new_ll:.5f})")
        ll = new_ll

    jump = {"lam_coef": a, "jump_w": float(w), "jump_mean": m, "jump_sig": s}
    return C, V, jump, (c if nb > 1 else None)


def design(x, E, off, mask, anchored):
    idx = np.where(mask)[0]
    idx = idx[idx + 1 < len(x)]
    idx = idx[mask[idx + 1]]
    x0, E0 = x[idx].astype(np.float64), E[idx].astype(np.float64)
    x1 = x[idx + 1].astype(np.float64)
    h = harmonics(off[idx], 0.0).numpy().astype(np.float64)
    H = lam_design(off[idx]).numpy().astype(np.float64)
    tau = off[idx][:, 0].astype(int) % 24
    if anchored:
        return x1 - E0, np.concatenate([x0 - E0, h], axis=1), H, tau
    return x1, np.concatenate([x0, h], axis=1), H, tau


# Shifts the profile's constant column by -K^{-1} dbar so the sample-mean total
# drift is zero, matching the centering penalty in train.py.
def center_drift(mod, x, E, off, mask):
    idx = np.where(mask)[0]
    x0, E0, o0 = (torch.tensor(a[idx]) for a in (x, E, off))
    with torch.no_grad():
        mu = mod.drift(x0, 0.0, o0, E0, None)
        if mod.has_jumps:
            size = mod.jump_w * mod.jump_mean[0] + (1 - mod.jump_w) * mod.jump_mean[1]
            mu = mu + mod.lam(0.0, o0).unsqueeze(1) * size
        dbar = mu.mean(0)
        delta = torch.linalg.solve(mod.K, dbar)
        mod.Theta[:, 0] -= delta
        mod.B[:, 0] -= (torch.eye(DIM) - mod.A) @ delta
    return dbar.numpy()


def fit_variant(x, E, off, mask, anchored, jumps, hod_sigma=False):
    Y, Z, H, tau = design(x, E, off, mask, anchored)
    C, V = ols(Y, Z)
    cvec = None
    if jumps:
        C, V, jump, cvec = fit_jump_em(Y, Z, H, C, V, hod=tau if hod_sigma else None)
    else:
        jump = None
    mod = SeasonalOU(C[:, :DIM], C[:, DIM:], V, jump=jump, anchored=anchored,
                     cvec=cvec)
    before = center_drift(mod, x, E, off, mask)
    after = center_drift(mod, x, E, off, mask)
    print(f"  mean total drift {np.round(before, 5)} -> {np.round(after, 5)} per hour")
    return mod


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
        hl_sims = [halflife_arr(sw[i, :, j], h)
                   for sw, h in zip(sims, hod) for i in range(0, sw.shape[0], 10)]
        hl_sim = np.nanmedian(hl_sims)
        hl_ratio = hl_sim / hl_real

        ks_ds = [stats.ks_2samp(rw[:, j], sw[i, :, j]).statistic
                 for rw, sw in zip(reals, sims) for i in range(0, sw.shape[0], 25)]

        rows.append({
            "node": NODE_LABELS[node],
            "real_mean": r.mean(), "sim_mean": s.mean(),
            "real_std": r.std(), "sim_std": s.std(),
            "real_pct_neg": 100 * (r < 0).mean(), "sim_pct_neg": 100 * (s < 0).mean(),
            "real_spikes_per_1000h": 1000 * rspk, "sim_spikes_per_1000h": 1000 * sspk,
            "real_halflife_h": hl_real, "sim_halflife_h": hl_sim,
            "ks_D_median": np.median(ks_ds),
            "mean_offset_sd": mean_off, "std_ratio": std_ratio,
            "neg_ratio": neg_ratio, "spike_ratio": spike_ratio, "halflife_ratio": hl_ratio,
            "ks_p_block": ks_df["p_block_bootstrap"][j],
            "ks_p_block_deseason": ks_df["p_block_deseason"][j],
            "a_mean": mean_off <= CHECKS["mean_offset_max"],
            "a_std": CHECKS["std_ratio"][0] <= std_ratio <= CHECKS["std_ratio"][1],
            "a_neg": CHECKS["neg_freq_ratio"][0] <= neg_ratio <= CHECKS["neg_freq_ratio"][1],
            "b_spike": CHECKS["spike_ratio"][0] <= spike_ratio <= CHECKS["spike_ratio"][1],
            "c_halflife": CHECKS["halflife_ratio"][0] <= hl_ratio <= CHECKS["halflife_ratio"][1],
        })
    return rows


def free_run(model, scaler, x, E, v, off, i0, n_hours):
    sim = run_window(model, scaler, x, E, v, off, i0, n_hours, 50, seed=555)
    return sim[:, 1:].mean(axis=(0, 1)), sim[:, 1:].std(axis=(0, 1))


def main():
    torch.manual_seed(0)
    neural, scaler, stage = load_model(ROOT / "sde" / "checkpoint.pt")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    nodes = list(lmp.columns)
    val = splits["val"][0]
    span_h = WINDOW_DAYS * 24
    starts = window_starts(index, val.index[0], N_WINDOWS, span_h)

    tr0 = pd.Timestamp("2025-01-01", tz="US/Pacific")
    tr1 = pd.Timestamp("2026-02-01", tz="US/Pacific")
    train_mask = (index >= tr0) & (index < tr1)
    va0, va1 = val.index[0], val.index[-1]
    val_mask = (index >= va0) & (index <= va1)
    print(f"train transitions {train_mask.sum()}  val {val_mask.sum()}  "
          f"neural checkpoint stage {stage}")

    variants = {}
    print("fitting seasonal OU (Gaussian MLE on transitions)")
    variants["OU"] = fit_variant(x, E, off, train_mask, anchored=False, jumps=False)
    print("fitting seasonal OU + compound-Poisson jumps (EM)")
    variants["OU+jumps"] = fit_variant(x, E, off, train_mask, anchored=False, jumps=True)
    print("fitting level-anchored seasonal OU + jumps (EM)")
    variants["OU+jumps+EMA"] = fit_variant(x, E, off, train_mask, anchored=True, jumps=True)
    print("fitting seasonal OU + jumps + hour-of-day volatility (EM)")
    variants["OU+jumps+seasonal vol"] = fit_variant(x, E, off, train_mask,
                                                    anchored=False, jumps=True,
                                                    hod_sigma=True)
    print("fitting level-anchored seasonal OU + jumps + hour-of-day volatility (EM)")
    variants["OU+jumps+EMA+seasonal vol"] = fit_variant(x, E, off, train_mask,
                                                        anchored=True, jumps=True,
                                                        hod_sigma=True)
    variants["neural SDE"] = neural

    xv = torch.tensor(x[val_mask])
    ov = torch.tensor(off[val_mask])
    Ev = torch.tensor(E[val_mask])
    vv = torch.tensor(v[val_mask])

    out = []
    for name, mod in variants.items():
        with torch.no_grad():
            nll = mod.transition_nll(xv[:-1], xv[1:], ov[:-1], Ev[:-1], vv[:-1]).item()
            err = cond_mean(mod, xv[:-1], ov[:-1], Ev[:-1], vv[:-1]) - xv[1:]
            rmse = (err ** 2).mean(dim=0).sqrt().numpy() * scaler.std
        if isinstance(mod, SeasonalOU):
            hl = np.log(2) / np.linalg.eigvals(mod.K.numpy()).real
            cv = mod.cvec.numpy()
            print(f"{name}: val NLL {nll:.4f}  1-step RMSE {rmse.round(2)}"
                  f"  K eigen half-lives {np.sort(hl).round(2)}h"
                  f"  hod var scale {cv.min():.2f}-{cv.max():.2f}"
                  f"  Lyapunov residual {mod.gap:.2e}")
        else:
            print(f"{name}: val NLL {nll:.4f}  1-step RMSE {rmse.round(2)}")
        rows = check_metrics(mod, scaler, x, E, v, off, index, lmp, starts, span_h, nodes)
        fr_mean, fr_std = free_run(mod, scaler, x, E, v, off, starts[0], len(val) - 1)
        n_fail = 0
        for row in rows:
            row["model"] = name
            row["val_nll"] = nll
            j = nodes.index([k for k, lab in NODE_LABELS.items()
                             if lab == row["node"]][0])
            row["freerun_mean"] = fr_mean[j]
            row["freerun_std"] = fr_std[j]
            row["onestep_rmse_usd"] = rmse[j]
            n_fail += sum(not row[k] for k in
                          ["a_mean", "a_std", "a_neg", "b_spike", "c_halflife"])
            print(f"  {row['node']}: mean {row['real_mean']:.1f}/{row['sim_mean']:.1f}"
                  f"  std {row['real_std']:.1f}/{row['sim_std']:.1f}"
                  f"  neg% {row['real_pct_neg']:.1f}/{row['sim_pct_neg']:.1f}"
                  f"  spikes/1000h {row['real_spikes_per_1000h']:.1f}/"
                  f"{row['sim_spikes_per_1000h']:.1f}"
                  f"  halflife {row['real_halflife_h']:.1f}/{row['sim_halflife_h']:.1f}h"
                  f"  ksD {row['ks_D_median']:.3f}")
        print(f"  {15 - n_fail}/15 checks passed   free-run 59d means "
              f"{np.round(fr_mean, 1)} vs real "
              f"{np.round(val.values[1:].mean(axis=0), 1)}")
        out += rows

    cols = ["model", "node", "val_nll", "onestep_rmse_usd",
            "real_mean", "sim_mean", "real_std", "sim_std",
            "real_pct_neg", "sim_pct_neg", "real_spikes_per_1000h",
            "sim_spikes_per_1000h", "real_halflife_h", "sim_halflife_h", "ks_D_median",
            "mean_offset_sd", "std_ratio", "neg_ratio", "spike_ratio", "halflife_ratio",
            "ks_p_block", "ks_p_block_deseason", "freerun_mean", "freerun_std",
            "a_mean", "a_std", "a_neg", "b_spike", "c_halflife"]
    df = pd.DataFrame(out)[cols]
    df.to_csv(ROOT / "sde" / "ou_comparison.csv", index=False)
    print(f"wrote {ROOT / 'sde' / 'ou_comparison.csv'}")


if __name__ == "__main__":
    main()
