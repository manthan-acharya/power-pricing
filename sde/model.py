import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DIM = 3
LAMBDA_CAP = 0.15
SIGMA_FLOOR = 0.03
EMA_HOURS = 168.0
ALPHA = math.exp(-1.0 / EMA_HOURS)
K_MIN, K_MAX = 0.02, 2.0
RESID_CAP = 0.5
ANCHOR_CAP = 2.0


def clock_offsets(index):
    local = index.tz_convert("US/Pacific")
    tau = local.hour + local.minute / 60.0
    dow = local.dayofweek + tau / 24.0
    doy = local.dayofyear - 1 + tau / 24.0
    return np.stack([tau, dow, doy], axis=1).astype(np.float32)


# Three harmonics of hour of day (a duck curve is far from sinusoidal), one of
# day of week and day of year.
def clock_features(t, off0):
    tau = (off0[:, 0] + t) % 24.0
    dow = (off0[:, 1] + t / 24.0) % 7.0
    doy = (off0[:, 2] + t / 24.0) % 365.25
    two_pi = 2 * math.pi
    feats = []
    for k in (1, 2, 3):
        feats += [torch.sin(k * two_pi * tau / 24.0),
                  torch.cos(k * two_pi * tau / 24.0)]
    feats += [torch.sin(two_pi * dow / 7.0), torch.cos(two_pi * dow / 7.0),
              torch.sin(two_pi * doy / 365.25), torch.cos(two_pi * doy / 365.25)]
    return torch.stack(feats, dim=1)


# Causal trailing EMAs of the per-hub level E and the system-wide mean absolute
# hourly move v, over a standardized [N, 3] array.
def ema_states(x):
    n = x.shape[0]
    E = np.zeros_like(x)
    v = np.zeros(n, dtype=np.float32)
    E[0] = x[0]
    v[0] = 0.5
    for t in range(1, n):
        E[t] = ALPHA * E[t - 1] + (1 - ALPHA) * x[t]
        move = np.abs(x[t] - x[t - 1]).mean()
        v[t] = ALPHA * v[t - 1] + (1 - ALPHA) * move
    return E, v[:, None]


def mlp(n_in, n_out, width=48):
    return nn.Sequential(
        nn.Linear(n_in, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, n_out),
    )


# dX = K(z)(A - X)dt + r(z)dt + sigma(z)dW + jumps, anchor A = E + a(clock),
# with level-free inputs z so the dynamics are translation invariant in level.
class HubSDE(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        n_full = DIM + 2 + 1 + 10
        n_anchor = 10
        self.k_net = mlp(n_full, DIM, width)
        self.a_net = mlp(n_anchor, DIM, width)
        self.r_net = mlp(n_full, DIM, width)
        self.sig_net = mlp(n_full, DIM, width)
        self.corr_raw = nn.Parameter(torch.eye(DIM) + 0.1 * torch.randn(DIM, DIM))
        self.lam_raw = nn.Parameter(torch.tensor([-2.5, 0.0, 0.0, 0.0, 0.0]))
        self.jump_w_raw = nn.Parameter(torch.tensor(0.0))
        self.jump_mean = nn.Parameter(torch.tensor([[1.5] * DIM, [-1.5] * DIM]))
        self.jump_lsig = nn.Parameter(torch.full((2, DIM), math.log(0.7)))

    def inputs(self, x, t, off0, E, v):
        spreads = torch.stack([x[:, 0] - x[:, 1], x[:, 0] - x[:, 2]], dim=1)
        c = clock_features(t, off0)
        z = torch.cat([x - E, spreads, v, c], dim=1)
        return z, c

    # Mean jump drift lambda * E[Z] per hour; needed so the centering penalty
    # sees the total drift, not just the continuous part.
    def jump_compensator(self, t, off0):
        lam = self.lam(t, off0)
        w, jmean, _ = self.jump_dists()
        mean_size = w * jmean[0] + (1 - w) * jmean[1]
        return lam.unsqueeze(1) * mean_size.unsqueeze(0)

    def drift(self, x, t, off0, E, v):
        z, zc = self.inputs(x, t, off0, E, v)
        K = K_MIN + (K_MAX - K_MIN) * torch.sigmoid(self.k_net(z))
        anchor = E + ANCHOR_CAP * torch.tanh(self.a_net(zc))
        resid = RESID_CAP * torch.tanh(self.r_net(z))
        return K * (anchor - x) + resid

    def corr_factor(self):
        L = torch.tril(self.corr_raw)
        return L / L.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def sigma_tril(self, x, t, off0, E, v):
        z, _ = self.inputs(x, t, off0, E, v)
        d = nn.functional.softplus(self.sig_net(z)) + SIGMA_FLOOR
        return d.unsqueeze(-1) * self.corr_factor().unsqueeze(0)

    def lam(self, t, off0):
        tau = (off0[:, 0] + t) % 24.0
        two_pi = 2 * math.pi
        h = torch.stack([torch.ones_like(tau),
                         torch.sin(two_pi * tau / 24.0), torch.cos(two_pi * tau / 24.0),
                         torch.sin(2 * two_pi * tau / 24.0), torch.cos(2 * two_pi * tau / 24.0)], dim=1)
        return LAMBDA_CAP * torch.sigmoid(h @ self.lam_raw)

    def jump_dists(self):
        w = torch.sigmoid(self.jump_w_raw)
        sig = self.jump_lsig.exp()
        return w, self.jump_mean, sig

    # Exact NLL of the Euler-discretized jump-diffusion over one step,
    # truncated at one jump per step (lambda*dt <= 0.15).
    def transition_nll(self, x0, x1, off0, E0, v0, dt=1.0):
        t = torch.zeros(1)
        mean0 = x0 + self.drift(x0, t, off0, E0, v0) * dt
        L = self.sigma_tril(x0, t, off0, E0, v0) * math.sqrt(dt)
        cov = L @ L.transpose(1, 2) + 1e-6 * torch.eye(DIM)
        lam = self.lam(t, off0) * dt
        w, jmean, jsig = self.jump_dists()

        logps = []
        base = torch.distributions.MultivariateNormal(
            mean0, scale_tril=torch.linalg.cholesky(cov))
        logps.append(torch.log1p(-lam) + base.log_prob(x1))
        for k, wk in [(0, w), (1, 1 - w)]:
            covk = cov + torch.diag(jsig[k] ** 2)
            comp = torch.distributions.MultivariateNormal(
                mean0 + jmean[k], scale_tril=torch.linalg.cholesky(covk))
            logps.append(torch.log(lam * wk + 1e-12) + comp.log_prob(x1))
        return -torch.logsumexp(torch.stack(logps, dim=0), dim=0).mean()

    def sample_jumps(self, t, off0, gen=None):
        B = off0.shape[0]
        lam = self.lam(torch.as_tensor(float(t)), off0)
        w, jmean, jsig = self.jump_dists()
        occurs = torch.rand(B, generator=gen) < lam
        comp = torch.where(torch.rand(B, generator=gen) < w, 0, 1)
        eps = torch.randn(B, DIM, generator=gen)
        sizes = jmean[comp] + jsig[comp] * eps
        return occurs.unsqueeze(1) * sizes


# torchsde wrapper for the continuous part over one hour; EMA states ride along
# as zero-drift, zero-noise solver components so the adjoint pass propagates
# gradients through them, updating on the hourly grid outside the solver.
class WindowSDE(nn.Module):
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, model, off0):
        super().__init__()
        self.model = model
        self.off0 = off0

    @staticmethod
    def pack(x, E, v):
        return torch.cat([x, E, v], dim=1)

    @staticmethod
    def unpack(y):
        return y[:, :DIM], y[:, DIM:2 * DIM], y[:, 2 * DIM:]

    def f(self, t, y):
        x, E, v = self.unpack(y)
        mu = self.model.drift(x, t, self.off0, E, v)
        return torch.cat([mu, torch.zeros_like(E), torch.zeros_like(v)], dim=1)

    def g(self, t, y):
        x, E, v = self.unpack(y)
        top = self.model.sigma_tril(x, t, self.off0, E, v)
        zeros = torch.zeros(y.shape[0], DIM + 1, DIM)
        return torch.cat([top, zeros], dim=1)


def ema_update(E, v, x_prev, x_new):
    move = (x_new - x_prev).abs().mean(dim=1, keepdim=True)
    return ALPHA * E + (1 - ALPHA) * x_new, ALPHA * v + (1 - ALPHA) * move


# seed pins only the jumps: without bm= torchsde draws its Brownian motion from
# numpy's legacy global RNG, so callers also need np.random.seed.
@torch.no_grad()
def simulate(model, x0, off0, E0, v0, n_hours, dt=0.25, seed=None):
    import torchsde
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
        torch.manual_seed(seed + 1)
    sde = WindowSDE(model, off0)
    x, E, v = x0, E0.clone(), v0.clone()
    out = [x0]
    for h in range(n_hours):
        ts = torch.tensor([float(h), float(h + 1)])
        x_prev = x
        y = torchsde.sdeint(sde, sde.pack(x, E, v), ts, method="euler", dt=dt)[-1]
        x = sde.unpack(y)[0] + model.sample_jumps(h + 1, off0, gen=gen)
        E, v = ema_update(E, v, x_prev, x)
        out.append(x)
    return torch.stack(out, dim=0)


class Scaler:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse(self, arr):
        return arr * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}


def load_frames(root):
    lmp = pd.read_csv(root / "data/processed/lmp_hourly.csv",
                      index_col=0, parse_dates=True)
    cong = pd.read_csv(root / "data/processed/congestion_hourly.csv",
                       index_col=0, parse_dates=True)
    splits = {}
    for name, (a, b) in {"train": ("2025-01-01", "2026-02-01"),
                         "val": ("2026-02-01", "2026-04-01"),
                         "test": ("2026-04-01", "2026-07-01")}.items():
        t0 = pd.Timestamp(a, tz="US/Pacific")
        t1 = pd.Timestamp(b, tz="US/Pacific") - pd.Timedelta(hours=1)
        splits[name] = (lmp.loc[t0:t1], cong.loc[t0:t1])
    return lmp, cong, splits


# Standardized full-history arrays with causal EMA states, so any window start
# can be initialized with its true trailing regime.
def full_series(root, scaler):
    lmp, _, _ = load_frames(root)
    x = scaler.transform(lmp.values.astype(np.float32))
    E, v = ema_states(x)
    off = clock_offsets(lmp.index)
    return lmp.index, x, E, v, off


# full_series extended with the July-September 2026 holdout once it exists
def extended_series(root, scaler):
    lmp, _, _ = load_frames(root)
    path = root / "data/processed/lmp_hourly_holdout.csv"
    if path.exists():
        lmp = pd.concat([lmp, pd.read_csv(path, index_col=0, parse_dates=True)])
    x = scaler.transform(lmp.values.astype(np.float32))
    E, v = ema_states(x)
    return lmp.index, x, E, v, clock_offsets(lmp.index)
