# Run: OMP_NUM_THREADS=1 python3 -u -m sde.model_pin

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from sde.model import ALPHA, DIM, Scaler, full_series, load_frames, mlp
from sde.model_twofactor import (ANCHOR_CAP_Z, BASIS, RESID_CAP_Z, ROT,
                                 TwoFactorSDE)
from sde.train import split_slice

# dropping the reversion network costs 3,315 parameters; widening the three
# survivors from 48 to 57 restores the incumbent's 12,999 to within 0.3%
WIDTH = 57


# removes the hour-of-day x calendar-month mean, since a pooled hour-of-day
# profile alone leaves the seasonal change in shape in the residual, where it
# reads as slow reversion
def deseasonalize(index, f):
    local = index.tz_convert("US/Pacific")
    s = pd.Series(np.asarray(f, dtype=np.float64))
    return (s - s.groupby([local.month.values, local.hour.values])
            .transform("mean")).values


# closed-form reversion rate kappa = -log(rho_l)/l of each rotated factor; for
# an OU factor these agree at every lag, so the spread across lags serves as a
# specification check on the AR(1) reading
def reversion_rates(index, x, lags=(1,)):
    z = np.asarray(x, dtype=np.float64) @ BASIS
    out = []
    for j in range(z.shape[1]):
        r = deseasonalize(index, z[:, j])
        out.append([-np.log(np.corrcoef(r[:-L], r[L:])[0, 1]) / L for L in lags])
    return np.array(out)


# kappa is the factor's total relaxation rate; the pinned level rate subtracts
# the EMA anchor's share (1-alpha) since the level's linearized decaying mode
# is K + (1-alpha), while the congestion factors have no anchor feedback and
# take kappa unchanged
def pinned_rates(index, x):
    kappa = reversion_rates(index, x)[:, 0]
    return kappa - np.array([1.0 - ALPHA, 0.0, 0.0])


# dz = K (A(w) - z) dt + r(w) dt + diag(d(w)) L_rho dW + dJ with K a fixed
# per-factor constant rather than a network output; anchor, residual,
# diffusion, jump and likelihood structure are inherited from the two-factor
# model.
class PinnedSDE(TwoFactorSDE):

    def __init__(self, kappa, width=WIDTH):
        # the parent's __init__ builds a reversion network this model does not
        # have, so the module is initialized directly
        nn.Module.__init__(self)
        n_full = DIM + 2 + 1 + 10
        self.a_net = mlp(10, DIM, width)
        self.r_net = mlp(n_full, DIM, width)
        self.sig_net = mlp(n_full, DIM, width)
        self.corr_raw = nn.Parameter(torch.eye(DIM) + 0.1 * torch.randn(DIM, DIM))
        self.lam_raw = nn.Parameter(torch.tensor([-2.5, 0.0, 0.0, 0.0, 0.0]))
        self.jump_w_raw = nn.Parameter(torch.tensor(0.0))
        self.jump_mean = nn.Parameter(torch.tensor([[1.5 * ROT, 0.0, 0.0],
                                                    [-1.5 * ROT, 0.0, 0.0]]))
        self.jump_lsig = nn.Parameter(torch.full((2, DIM), math.log(0.7)))
        self.register_buffer("Q", torch.tensor(BASIS, dtype=torch.float32))
        self.register_buffer("kappa", torch.as_tensor(np.asarray(kappa),
                                                      dtype=torch.float32))

    def rates(self, w):
        return self.kappa.expand(w.shape[0], DIM)

    def drift_parts(self, x, t, off0, E, v):
        z, zE, w, c = self.inputs(x, t, off0, E, v)
        a = ANCHOR_CAP_Z * torch.tanh(self.a_net(c))
        anchor = torch.cat([zE[:, :1] + a[:, :1], a[:, 1:]], dim=1)
        return anchor - z, RESID_CAP_Z * torch.tanh(self.r_net(w))

    def drift_z(self, x, t, off0, E, v):
        dev, resid = self.drift_parts(x, t, off0, E, v)
        return self.kappa * dev + resid

    # OLS slope of each factor's residual drift on its own anchor deviation;
    # pinning K is vacuous unless this stays near zero, since the effective
    # reversion rate is K plus this slope and the residual could otherwise buy
    # back the reversion the pin removed.
    def residual_slope(self, x, t, off0, E, v):
        dev, resid = self.drift_parts(x, t, off0, E, v)
        d = dev - dev.mean(dim=0, keepdim=True)
        r = resid - resid.mean(dim=0, keepdim=True)
        return (d * r).sum(dim=0) / (d * d).sum(dim=0).clamp_min(1e-8)


def train_rates(root=ROOT):
    _, _, splits = load_frames(root)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, _, _, _ = full_series(root, scaler)
    m = split_slice(index, "train")
    return index[m], x[m], scaler


def main():
    index, x, _ = train_rates()
    lags = (1, 2, 3, 4, 6, 8, 12)
    kap = reversion_rates(index, x, lags)
    names = ["level (1,1,1)", "spread (1,-1,0)", "spread (1,1,-2)"]
    print(f"train split {len(index)}h  {index[0].date()} to {index[-1].date()}")
    print("kappa by lag (1/h), and implied half-life at lag 1:")
    for j, n in enumerate(names):
        row = "  ".join(f"L{L}:{k:.4f}" for L, k in zip(lags, kap[j]))
        print(f"  {n:18s} {row}   hl {np.log(2) / kap[j, 0]:.2f}h")

    alt = np.asarray(x, dtype=np.float64) @ np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    r = deseasonalize(index, alt)
    k_alt = -np.log(np.corrcoef(r[:-1], r[1:])[0, 1])
    print(f"  {'spread (1,0,-1)':18s} L1:{k_alt:.4f}   hl {np.log(2) / k_alt:.2f}h")

    kp = pinned_rates(index, x)
    print(f"\nEMA anchor share (1-alpha) = {1 - ALPHA:.5f}/h")
    print("pinned K:", np.round(kp, 5),
          " implied model half-lives:",
          np.round(np.log(2) / (kp + np.array([1 - ALPHA, 0, 0])), 2), "h")
    n_par = sum(p.numel() for p in PinnedSDE(kp).parameters())
    print(f"parameters {n_par} (incumbent 12,999)")


if __name__ == "__main__":
    main()
