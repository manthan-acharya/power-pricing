import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sde.model import DIM, HubSDE, Scaler, full_series, load_frames

OUT = ROOT / "control"
FIG = ROOT / "writeup" / "figures"
SEED = 21

P_MAX = 25.0
CAP = 100.0
ETA = math.sqrt(0.88)
C_DEG = 2.0 / 1000.0
EPS_SOC = 8.0
EPS_FINAL = 2.5
DT = 1.0
HUB = 1  # SP15


def battery_terminal_price(root):
    _, _, splits = load_frames(root)
    train = splits["train"][0]
    local = train.tz_convert("US/Pacific")
    overnight = local[local.index.hour < 6].iloc[:, HUB]
    return (ETA * overnight.median() - C_DEG * 1000.0) / 1000.0


class PriceGenerator:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        w, jmean, jsig = model.jump_dists()
        self.w = w.detach()
        self.M2 = (self.w * (jmean[0].outer(jmean[0]) + torch.diag(jsig[0] ** 2))
                   + (1 - self.w) * (jmean[1].outer(jmean[1]) + torch.diag(jsig[1] ** 2))).detach()
        self.mbar = (self.w * jmean[0] + (1 - self.w) * jmean[1]).detach()

    @torch.no_grad()
    def coeffs(self, x, t, off0, E, v):
        lam = self.model.lam(torch.as_tensor(float(t)), off0)
        mu = self.model.drift(x, torch.as_tensor(float(t)), off0, E, v) \
            + lam.unsqueeze(1) * self.mbar.unsqueeze(0)
        L = self.model.sigma_tril(x, torch.as_tensor(float(t)), off0, E, v)
        cov = L @ L.transpose(1, 2) + lam.view(-1, 1, 1) * self.M2
        chol = torch.linalg.cholesky(cov + 1e-7 * torch.eye(DIM))
        return mu, chol

    def price_usd(self, x):
        return self.scaler.mean[HUB] + self.scaler.std[HUB] * x[:, HUB]


# max_u [P u - c|u| + phi(u) q] over the SOC-clamped control set, bang-bang by
# piecewise linearity
def hamiltonian(p_kusd, s, q):
    u_dis = torch.clamp(ETA * s.clamp_min(0.0) / DT, max=P_MAX)
    u_chg = torch.clamp((CAP - s).clamp_min(0.0) / (ETA * DT), max=P_MAX)
    dis_val = (p_kusd - C_DEG - q / ETA) * u_dis
    chg_val = (ETA * q - C_DEG - p_kusd) * u_chg
    H = torch.maximum(torch.zeros_like(dis_val),
                      torch.maximum(dis_val, chg_val))
    u = torch.where((dis_val >= chg_val) & (dis_val > 0), u_dis,
                    torch.where(chg_val > 0, -u_chg, torch.zeros_like(u_dis)))
    return H, u


class StepNets(nn.Module):
    def __init__(self, n_steps, n_in, n_out, width=64):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(n_in, width), nn.Tanh(),
                          nn.Linear(width, width), nn.Tanh(),
                          nn.Linear(width, n_out))
            for _ in range(n_steps)])

    def forward(self, k, feats):
        return self.nets[k](feats)


class DeepBSDE(nn.Module):
    def __init__(self, n_steps=24, price_dim=DIM, width=64, n_extra=0):
        super().__init__()
        self.n_steps = n_steps
        n_cond = DIM + 1 + 4 + n_extra
        n_in = 1 + price_dim + n_cond
        self.y0 = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(),
                                nn.Linear(width, width), nn.Tanh(),
                                nn.Linear(width, 1))
        self.z = StepNets(n_steps, n_in, 1 + price_dim, width)
        with torch.no_grad():
            for net in self.z.nets:
                net[-1].weight.mul_(0.05)
                net[-1].bias.zero_()

    def feats(self, s, xdev, cond):
        return torch.cat([s.unsqueeze(1) / CAP, xdev, cond], dim=1)


def day_conditioning(off0, E0, v0):
    two_pi = 2 * math.pi
    dow = off0[:, 1]
    doy = off0[:, 2]
    cal = torch.stack([torch.sin(two_pi * dow / 7.0), torch.cos(two_pi * dow / 7.0),
                       torch.sin(two_pi * doy / 365.25), torch.cos(two_pi * doy / 365.25)], dim=1)
    return torch.cat([E0, v0, cal], dim=1)


def day_starts(index, split):
    bounds = {"train": ("2025-01-01", "2026-02-01"),
              "val": ("2026-02-01", "2026-04-01"),
              "test": ("2026-04-01", "2026-07-01"),
              "holdout": ("2026-07-01", "2026-10-01")}
    a, b = bounds[split]
    t0 = pd.Timestamp(a, tz="US/Pacific")
    t1 = pd.Timestamp(b, tz="US/Pacific")
    local = index.tz_convert("US/Pacific")
    mask = (index >= t0) & (index < t1) & (local.hour == 0)
    idx = np.where(mask)[0]
    return idx[idx + 24 <= len(index)]


class FullProblem:
    def __init__(self, gen):
        self.gen = gen
        self.price_dim = DIM

    def step(self, x, k, off0, E0, v0, xi):
        mu, chol = self.gen.coeffs(x, k, off0, E0, v0)
        return x + mu * DT + (chol @ xi.unsqueeze(-1)).squeeze(-1) * math.sqrt(DT)

    def price(self, x):
        return torch.as_tensor(self.gen.price_usd(x)) / 1000.0

    def xdev(self, x, E0):
        return x - E0


def rollout_bsde(problem, nets, s0, x0, off0, E0, v0, pi_term, gen_rng,
                 eps=EPS_SOC):
    B = s0.shape[0]
    cond = day_conditioning(off0, E0, v0)
    s, x = s0, x0
    feats = nets.feats(s, problem.xdev(x, E0), cond)
    Y = nets.y0(feats).squeeze(1)
    for k in range(nets.n_steps):
        Z = nets.z(k, feats)
        H, _ = hamiltonian(problem.price(x), s, Z[:, 0] / eps)
        dw0 = torch.randn(B, generator=gen_rng) * math.sqrt(DT)
        xi = torch.randn(B, problem.price_dim, generator=gen_rng)
        Y = Y - H * DT + Z[:, 0] * dw0 + (Z[:, 1:] * xi).sum(1) * math.sqrt(DT)
        s = s + eps * dw0
        x = problem.step(x, k, off0, E0, v0, xi)
        feats = nets.feats(s, problem.xdev(x, E0), cond)
    # linear terminal extension avoids spurious Jensen value from the
    # artificial SOC noise crossing the s=0 kink
    g = pi_term * s
    loss = ((Y - g) ** 2).mean()
    return loss


# homotopy in the viscosity parameter: hold at eps_hi, anneal geometrically to
# eps_lo where the regularized equation approximates the true one
def eps_at(it, iters, eps_hi, eps_lo):
    a, b = 0.3, 0.8
    f = it / iters
    if f < a or eps_lo is None:
        return eps_hi
    if f > b:
        return eps_lo
    return eps_hi * (eps_lo / eps_hi) ** ((f - a) / (b - a))


def train_bsde(problem, nets, sampler, pi_term, iters=6000, batch=512,
               lr=1e-3, log_every=200, tag="", eps=EPS_SOC, eps_final=None):
    torch.manual_seed(SEED)
    gen_rng = torch.Generator().manual_seed(SEED + 1)
    opt = torch.optim.Adam(nets.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr / 20)
    log = []
    for it in range(iters):
        s0, x0, off0, E0, v0 = sampler(batch, gen_rng)
        loss = rollout_bsde(problem, nets, s0, x0, off0, E0, v0, pi_term,
                            gen_rng, eps=eps_at(it, iters, eps, eps_final))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5.0)
        opt.step()
        sched.step()
        log.append((it, loss.item()))
        if it % log_every == 0 or it == iters - 1:
            print(f"{tag}iter {it:5d}  terminal loss {loss.item():.5f}")
    return pd.DataFrame(log, columns=["iter", "loss"])


def make_train_sampler(index, x, E, v, off):
    starts = day_starts(index, "train")
    xt = torch.tensor(x)
    Et = torch.tensor(E)
    vt = torch.tensor(v)
    ot = torch.tensor(off)

    starts_t = torch.tensor(starts)

    def sampler(batch, gen_rng):
        pick = starts_t[torch.randint(0, len(starts_t), (batch,), generator=gen_rng)]
        s0 = torch.rand(batch, generator=gen_rng) * CAP
        return s0, xt[pick], ot[pick], Et[pick], vt[pick]

    return sampler


@torch.no_grad()
def policy_stats(problem, nets, sampler, n=2048):
    gen_rng = torch.Generator().manual_seed(SEED + 7)
    s0, x0, off0, E0, v0 = sampler(n, gen_rng)
    cond = day_conditioning(off0, E0, v0)
    s, x = s0, x0
    counts = {"charge": 0, "discharge": 0, "idle": 0}
    soc_min, soc_max = s.clone(), s.clone()
    for k in range(nets.n_steps):
        feats = nets.feats(s, problem.xdev(x, E0), cond)
        Z = nets.z(k, feats)
        _, u = hamiltonian(problem.price(x), s, Z[:, 0] / EPS_SOC)
        counts["charge"] += (u < -1e-6).sum().item()
        counts["discharge"] += (u > 1e-6).sum().item()
        counts["idle"] += (u.abs() <= 1e-6).sum().item()
        phi = torch.where(u >= 0, -u / ETA, -ETA * u)
        s = (s + phi * DT).clamp(0.0, CAP)
        soc_min = torch.minimum(soc_min, s)
        soc_max = torch.maximum(soc_max, s)
        xi = torch.randn(n, problem.price_dim, generator=gen_rng)
        x = problem.step(x, k, off0, E0, v0, xi)
    tot = sum(counts.values())
    fr = {k: c / tot for k, c in counts.items()}
    print(f"policy fractions: charge {fr['charge']:.3f}  discharge "
          f"{fr['discharge']:.3f}  idle {fr['idle']:.3f}")
    print(f"mean SOC range used: {(soc_max - soc_min).mean().item():.1f} MWh "
          f"of {CAP:.0f}")
    return fr


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    ck = torch.load(ROOT / "sde/checkpoint.pt", weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    index, x, E, v, off = full_series(ROOT, scaler)

    pi_term = battery_terminal_price(ROOT)
    print(f"terminal salvage {pi_term * 1000:.1f} $/MWh")

    gen = PriceGenerator(model, scaler)
    problem = FullProblem(gen)
    sampler = make_train_sampler(index, x, E, v, off)
    nets = DeepBSDE()
    log = train_bsde(problem, nets, sampler, pi_term, iters=8000,
                     eps_final=EPS_FINAL)
    log.to_csv(OUT / "bsde_train_log.csv", index=False)
    torch.save({"nets": nets.state_dict(), "pi_term": pi_term,
                "eps": EPS_FINAL}, OUT / "bsde_checkpoint.pt")

    fr = policy_stats(problem, nets, sampler)
    assert fr["idle"] < 0.98 and fr["charge"] > 0.02 and fr["discharge"] > 0.02, \
        "degenerate policy"

    import matplotlib.pyplot as plt
    from plotstyle import apply
    apply()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.semilogy(log["iter"], log["loss"], lw=0.7, color="#0072B2")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"terminal matching loss (k\$$^2$)")
    ax.set_title("Deep BSDE training, full 3-hub problem")
    fig.tight_layout()
    fig.savefig(FIG / "bsde_loss.pdf")


if __name__ == "__main__":
    main()
