import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.bsde import (CAP, C_DEG, DIM, DT, EPS_FINAL, EPS_SOC, ETA, HUB,
                          P_MAX, DeepBSDE, PriceGenerator,
                          battery_terminal_price, day_conditioning, day_starts,
                          train_bsde)
from sde.model import HubSDE, Scaler, full_series

OUT = ROOT / "control"
FIG = ROOT / "writeup" / "figures"
TOL = 0.15
T_CHECKS = [0, 6, 12, 18]
S_CHECKS = [10.0, 50.0, 90.0]
XDEV_CHECKS = [-1.5, -0.75, 0.0, 0.75, 1.5]


def median_day(index, E):
    starts = day_starts(index, "train")
    e2 = E[starts, HUB]
    pick = starts[np.argsort(e2)[len(e2) // 2]]
    print(f"reduced-problem day instance: {index[pick]} (median regime level)")
    return pick


class ReducedCoeffs:
    def __init__(self, model, scaler, off0, E0, v0):
        self.model = model
        self.scaler = scaler
        self.off0 = off0
        self.E0 = E0
        self.v0 = v0
        gen = PriceGenerator(model, scaler)
        self.M2_22 = gen.M2[HUB, HUB].item()
        self.mbar_2 = gen.mbar[HUB].item()
        w, jmean, jsig = model.jump_dists()
        self.jump = {"w": w.item(), "m": jmean.detach()[:, HUB].numpy(),
                     "s": jsig.detach()[:, HUB].numpy()}

    @torch.no_grad()
    def parts(self, x2, t):
        n = x2.shape[0]
        E = self.E0.expand(n, DIM)
        v = self.v0.expand(n, 1)
        off = self.off0.expand(n, 3)
        xvec = E + (torch.as_tensor(x2, dtype=torch.float32).unsqueeze(1)
                    - E[:, HUB:HUB + 1])
        tt = torch.as_tensor(float(t))
        mu_c = self.model.drift(xvec, tt, off, E, v)[:, HUB]
        L = self.model.sigma_tril(xvec, tt, off, E, v)
        var_c = (L @ L.transpose(1, 2))[:, HUB, HUB]
        lam = self.model.lam(tt, off)[0].item()
        return mu_c.numpy(), var_c.numpy(), lam

    def mm(self, x2, t):
        mu_c, var_c, lam = self.parts(x2, t)
        return mu_c + lam * self.mbar_2, var_c + lam * self.M2_22

    def price_usd(self, x2):
        return self.scaler.mean[HUB] + self.scaler.std[HUB] * x2


def transition_matrix(coeffs, xgrid, t, kind):
    edges = np.concatenate([[-np.inf], (xgrid[:-1] + xgrid[1:]) / 2, [np.inf]])
    if kind == "mm":
        mu, var = coeffs.mm(xgrid, t)
        mean = xgrid + mu * DT
        sd = np.sqrt(var * DT)
        cdf = norm.cdf((edges[None, :] - mean[:, None]) / sd[:, None])
        T = np.diff(cdf, axis=1)
    else:
        mu_c, var_c, lam = coeffs.parts(xgrid, t)
        base_mean = xgrid + mu_c * DT
        sd_c = np.sqrt(var_c * DT)
        j = coeffs.jump
        comps = [(1 - lam * DT, base_mean, sd_c)]
        for k, wk in [(0, j["w"]), (1, 1 - j["w"])]:
            comps.append((lam * DT * wk, base_mean + j["m"][k],
                          np.sqrt(var_c * DT + j["s"][k] ** 2)))
        T = np.zeros((len(xgrid), len(xgrid)))
        for wgt, mean, sd in comps:
            cdf = norm.cdf((edges[None, :] - mean[:, None]) / sd[:, None])
            T += wgt * np.diff(cdf, axis=1)
    return T / T.sum(axis=1, keepdims=True)


def solve_dp(coeffs, pi_term, n_s, n_x, kind="mm", n_steps=24):
    sgrid = np.linspace(0.0, CAP, n_s)
    xgrid = coeffs.E0[0, HUB].item() + np.linspace(-4.5, 4.5, n_x)
    controls = np.linspace(-P_MAX, P_MAX, 21)
    price = coeffs.price_usd(xgrid) / 1000.0

    V = pi_term * sgrid[:, None] * np.ones((1, n_x))
    Vs = [None] * n_steps + [V]
    for k in range(n_steps - 1, -1, -1):
        T = transition_matrix(coeffs, xgrid, k, kind)
        cont = Vs[k + 1] @ T.T
        best = np.full((n_s, n_x), -np.inf)
        for u in controls:
            u_eff = np.clip(u, -np.minimum(P_MAX, (CAP - sgrid) / (ETA * DT)),
                            np.minimum(P_MAX, ETA * sgrid / DT))
            s_next = sgrid + np.where(u_eff >= 0, -u_eff / ETA, -ETA * u_eff) * DT
            idx = np.clip(np.searchsorted(sgrid, s_next) - 1, 0, n_s - 2)
            whi = (s_next - sgrid[idx]) / (sgrid[idx + 1] - sgrid[idx])
            cont_i = (1 - whi)[:, None] * cont[idx] + whi[:, None] * cont[idx + 1]
            reward = (price[None, :] * u_eff[:, None]
                      - C_DEG * np.abs(u_eff)[:, None]) * DT
            best = np.maximum(best, reward + cont_i)
        Vs[k] = best
    return sgrid, xgrid, Vs


def dp_value_at(sgrid, xgrid, Vk, s, x):
    i = np.clip(np.searchsorted(sgrid, s) - 1, 0, len(sgrid) - 2)
    j = np.clip(np.searchsorted(xgrid, x) - 1, 0, len(xgrid) - 2)
    ws = (s - sgrid[i]) / (sgrid[i + 1] - sgrid[i])
    wx = (x - xgrid[j]) / (xgrid[j + 1] - xgrid[j])
    return ((1 - ws) * (1 - wx) * Vk[i, j] + ws * (1 - wx) * Vk[i + 1, j]
            + (1 - ws) * wx * Vk[i, j + 1] + ws * wx * Vk[i + 1, j + 1])


class ReducedProblem:
    def __init__(self, coeffs, t_offset=0):
        self.coeffs = coeffs
        self.t_offset = t_offset
        self.price_dim = 1

    def step(self, x, k, off0, E0, v0, xi):
        mu, var = self.coeffs.mm(x[:, 0].numpy(), k + self.t_offset)
        mu = torch.tensor(mu, dtype=torch.float32)
        sd = torch.tensor(np.sqrt(var), dtype=torch.float32)
        return x + (mu * DT + sd * math.sqrt(DT) * xi[:, 0]).unsqueeze(1)

    def price(self, x):
        return torch.as_tensor(self.coeffs.price_usd(x[:, 0].numpy())) / 1000.0

    def xdev(self, x, E0):
        return x - E0[:, HUB:HUB + 1]


def make_reduced_sampler(coeffs, t0):
    off_shift = torch.tensor([float(t0), t0 / 24.0, t0 / 24.0])

    def sampler(batch, gen_rng):
        s0 = torch.rand(batch, generator=gen_rng) * CAP
        e2 = coeffs.E0[0, HUB]
        x0 = (e2 + (torch.rand(batch, generator=gen_rng) * 5.0 - 2.5)).unsqueeze(1)
        off0 = (coeffs.off0 + off_shift).expand(batch, 3)
        return s0, x0, off0, coeffs.E0.expand(batch, DIM), coeffs.v0.expand(batch, 1)

    return sampler


def train_reduced_bsde(coeffs, pi_term, t0):
    nets = DeepBSDE(n_steps=24 - t0, price_dim=1)
    problem = ReducedProblem(coeffs, t_offset=t0)
    sampler = make_reduced_sampler(coeffs, t0)
    # rectification bias grows with eps*sqrt(T), so shorter horizons scale eps
    # by sqrt(T/24), floored to keep the q = Z0/eps readout stable
    scale = math.sqrt((24 - t0) / 24.0)
    train_bsde(problem, nets, sampler, pi_term, iters=8000, batch=512,
               lr=1e-3, log_every=1000, tag=f"[t0={t0:2d}] ",
               eps=max(4.0, EPS_SOC * scale),
               eps_final=max(1.75, EPS_FINAL * scale))
    return nets, problem, sampler


@torch.no_grad()
def bsde_value_at(nets, coeffs, t0, s, xdev):
    off_shift = torch.tensor([float(t0), t0 / 24.0, t0 / 24.0])
    off0 = (coeffs.off0 + off_shift).expand(1, 3)
    cond = day_conditioning(off0, coeffs.E0, coeffs.v0)
    feats = nets.feats(torch.tensor([float(s)], dtype=torch.float32),
                       torch.tensor([[float(xdev)]], dtype=torch.float32), cond)
    return nets.y0(feats).item()


def run_check(coeffs, pi_term, n_s, n_x, bsde_nets):
    rows = []
    sgrid, xgrid, Vs_mm = solve_dp(coeffs, pi_term, n_s, n_x, "mm")
    _, _, Vs_j = solve_dp(coeffs, pi_term, n_s, n_x, "jump")
    e2 = coeffs.E0[0, HUB].item()
    for t0 in T_CHECKS:
        for s in S_CHECKS:
            for xd in XDEV_CHECKS:
                v_dp = dp_value_at(sgrid, xgrid, Vs_mm[t0], s, e2 + xd)
                v_dpj = dp_value_at(sgrid, xgrid, Vs_j[t0], s, e2 + xd)
                v_b = bsde_value_at(bsde_nets[t0], coeffs, t0, s, xd)
                rows.append({"t0": t0, "s": s, "xdev": xd, "V_dp": v_dp,
                             "V_dp_jump": v_dpj, "V_bsde": v_b})
    df = pd.DataFrame(rows)
    rng = df["V_dp"].max() - df["V_dp"].min()
    df["rel_err"] = (df["V_bsde"] - df["V_dp"]).abs() / \
        np.maximum(df["V_dp"].abs(), 0.1 * rng)
    df["mm_vs_jump"] = (df["V_dp"] - df["V_dp_jump"]).abs() / \
        np.maximum(df["V_dp_jump"].abs(), 0.1 * rng)
    return df


def eps_sensitivity(coeffs, pi_term, df):
    d0 = df[df.t0 == 0].reset_index(drop=True)
    rows = []
    for eps in [2.0, 4.0]:
        nets = DeepBSDE(n_steps=24, price_dim=1)
        problem = ReducedProblem(coeffs, t_offset=0)
        sampler = make_reduced_sampler(coeffs, 0)
        train_bsde(problem, nets, sampler, pi_term, iters=6000, batch=512,
                   lr=1e-3, log_every=2000, tag=f"[eps={eps:.0f}] ", eps=eps)
        for _, r in d0.iterrows():
            v = bsde_value_at(nets, coeffs, 0, r.s, r.xdev)
            rows.append({"eps": eps, "s": r.s, "xdev": r.xdev, "V": v,
                         "V_ref": r.V_bsde, "V_dp": r.V_dp})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "eps_sensitivity.csv", index=False)
    for eps in [2.0, 4.0]:
        d = out[out.eps == eps]
        print(f"eps={eps:.0f}: mean |V - V_ref| {np.abs(d.V - d.V_ref).mean():.4f} k$, "
              f"mean |V - V_dp| {np.abs(d.V - d.V_dp).mean():.4f} k$")


def main():
    torch.set_num_threads(1)
    np.random.seed(3)
    ck = torch.load(ROOT / "sde/checkpoint.pt", weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    index, _, E, v, off = full_series(ROOT, scaler)
    pick = median_day(index, E)
    coeffs = ReducedCoeffs(model, scaler,
                           torch.tensor(off[pick:pick + 1]),
                           torch.tensor(E[pick:pick + 1]),
                           torch.tensor(v[pick:pick + 1]))
    pi_term = battery_terminal_price(ROOT)

    bsde_nets = {}
    for t0 in T_CHECKS:
        nets, _, _ = train_reduced_bsde(coeffs, pi_term, t0)
        bsde_nets[t0] = nets

    df = run_check(coeffs, pi_term, 21, 41, bsde_nets)
    worst = df["rel_err"].max()
    print(f"\nDP grid 21x41: worst |V_bsde - V_dp| rel err {worst:.3f} "
          f"(median {df['rel_err'].median():.3f})")
    print(f"moment-matching cost (DP mm vs DP jump): median "
          f"{df['mm_vs_jump'].median():.3f}, max {df['mm_vs_jump'].max():.3f}")

    if worst > TOL:
        print(f"tolerance {TOL} breached; refining DP grid to 41x81")
        df2 = run_check(coeffs, pi_term, 41, 81, bsde_nets)
        worst2 = df2["rel_err"].max()
        print(f"DP grid 41x81: worst rel err {worst2:.3f} "
              f"(median {df2['rel_err'].median():.3f})")
        gap_shrunk = worst2 < worst
        print(f"gap {'shrank' if gap_shrunk else 'did not shrink'} under refinement")
        df, worst = df2, worst2
    df.round(4).to_csv(OUT / "dp_crosscheck.csv", index=False)

    import matplotlib.pyplot as plt
    from plotstyle import apply
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {0: "#0072B2", 6: "#D55E00", 12: "#009E73", 18: "#CC79A7"}
    for t0 in T_CHECKS:
        d = df[df.t0 == t0]
        axes[0].scatter(d.V_dp, d.V_bsde, s=14, color=colors[t0], label=f"t={t0}h")
    lo = min(df.V_dp.min(), df.V_bsde.min())
    hi = max(df.V_dp.max(), df.V_bsde.max())
    pad = 0.05 * (hi - lo)
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=0.8, color="#999999")
    axes[0].set_xlabel(r"grid DP value (k\$)")
    axes[0].set_ylabel(r"deep BSDE value (k\$)")
    axes[0].set_title("Value Function Comparison (BSDE vs. DP)")
    axes[0].legend()
    axes[1].hist(100 * df.rel_err, bins=20, color="#0072B2", alpha=0.85)
    axes[1].axvline(100 * TOL, color="#D55E00", lw=1.0)
    axes[1].set_xlabel("relative error (%)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Relative Error Distribution (Deep BSDE vs. Grid DP)")
    fig.tight_layout()
    fig.savefig(FIG / "dp_cross_check.pdf")
    print(f"final worst rel err {worst:.3f} vs tolerance {TOL}")

    eps_sensitivity(coeffs, pi_term, df)
    return worst <= TOL


if __name__ == "__main__":
    ok = main()
    print("dp cross-check PASSED" if ok else "dp cross-check FAILED")
