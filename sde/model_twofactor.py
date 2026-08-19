import math

import numpy as np
import torch
import torch.nn as nn

from sde.model import (ANCHOR_CAP, DIM, K_MAX, K_MIN, LAMBDA_CAP, RESID_CAP,
                       SIGMA_FLOOR, clock_features, mlp)

# columns are q_L = (1,1,1)/sqrt(3), the NP15-SP15 spread direction, and the
# remaining orthogonal complement; z = Q' x
BASIS = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]).T
BASIS = BASIS / np.linalg.norm(BASIS, axis=0)

# the incumbent's caps are stated per hub; rotating a vector with |r_j| <= c
# into the level direction admits |r_L| <= c*sqrt(3), so the z-space caps are
# set to that bound to keep the two models equally permissive
ROT = math.sqrt(3.0)
ANCHOR_CAP_Z = ANCHOR_CAP * ROT
RESID_CAP_Z = RESID_CAP * ROT


# dz = K(w) (A(w) - z) dt + r(w) dt + diag(d(w)) L_rho dW + dJ in rotated
# coordinates z = Q'x, with K_L, K_C1, K_C2 separate network outputs and
# network inputs kept level-free: (z_L - E_L, z_C, z_C - E_C, v, clock).
class TwoFactorSDE(nn.Module):

    def __init__(self, width=48):
        super().__init__()
        n_full = DIM + 2 + 1 + 10
        self.k_net = mlp(n_full, DIM, width)
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

    def rotate(self, x):
        return x @ self.Q

    def unrotate(self, z):
        return z @ self.Q.T

    def inputs(self, x, t, off0, E, v):
        z, zE = self.rotate(x), self.rotate(E)
        c = clock_features(t, off0)
        w = torch.cat([z[:, :1] - zE[:, :1], z[:, 1:], z[:, 1:] - zE[:, 1:], v, c], dim=1)
        return z, zE, w, c

    def rates(self, w):
        return K_MIN + (K_MAX - K_MIN) * torch.sigmoid(self.k_net(w))

    def drift_z(self, x, t, off0, E, v):
        z, zE, w, c = self.inputs(x, t, off0, E, v)
        a = ANCHOR_CAP_Z * torch.tanh(self.a_net(c))
        anchor = torch.cat([zE[:, :1] + a[:, :1], a[:, 1:]], dim=1)
        resid = RESID_CAP_Z * torch.tanh(self.r_net(w))
        return self.rates(w) * (anchor - z) + resid

    def drift(self, x, t, off0, E, v):
        return self.unrotate(self.drift_z(x, t, off0, E, v))

    def corr_factor(self):
        L = torch.tril(self.corr_raw)
        return L / L.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def sigma_z(self, x, t, off0, E, v):
        _, _, w, _ = self.inputs(x, t, off0, E, v)
        d = nn.functional.softplus(self.sig_net(w)) + SIGMA_FLOOR
        return d.unsqueeze(-1) * self.corr_factor().unsqueeze(0)

    # Diffusion in hub coordinates, Q S_z; not triangular, which the
    # general-noise solver wrapper and the Cholesky of S S' both allow.
    def sigma_tril(self, x, t, off0, E, v):
        return self.Q.unsqueeze(0) @ self.sigma_z(x, t, off0, E, v)

    def lam(self, t, off0):
        tau = (off0[:, 0] + t) % 24.0
        two_pi = 2 * math.pi
        h = torch.stack([torch.ones_like(tau),
                         torch.sin(two_pi * tau / 24.0), torch.cos(two_pi * tau / 24.0),
                         torch.sin(2 * two_pi * tau / 24.0), torch.cos(2 * two_pi * tau / 24.0)], dim=1)
        return LAMBDA_CAP * torch.sigmoid(h @ self.lam_raw)

    def jump_dists(self):
        w = torch.sigmoid(self.jump_w_raw)
        return w, self.jump_mean, self.jump_lsig.exp()

    def jump_compensator(self, t, off0):
        lam = self.lam(t, off0)
        w, jmean, _ = self.jump_dists()
        mean_size = self.unrotate((w * jmean[0] + (1 - w) * jmean[1]).unsqueeze(0))
        return lam.unsqueeze(1) * mean_size

    def jump_sizes(self, comp, eps):
        _, jmean, jsig = self.jump_dists()
        return self.unrotate(jmean[comp] + jsig[comp] * eps)

    # Same three-component Euler mixture as the incumbent, in hub coordinates:
    # Q is orthogonal so the density needs no Jacobian and the held-out NLL is
    # directly comparable.
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
            jc = self.Q @ torch.diag(jsig[k] ** 2) @ self.Q.T
            comp = torch.distributions.MultivariateNormal(
                mean0 + self.unrotate(jmean[k].unsqueeze(0)),
                scale_tril=torch.linalg.cholesky(cov + jc))
            logps.append(torch.log(lam * wk + 1e-12) + comp.log_prob(x1))
        return -torch.logsumexp(torch.stack(logps, dim=0), dim=0).mean()

    def sample_jumps(self, t, off0, gen=None):
        B = off0.shape[0]
        lam = self.lam(torch.as_tensor(float(t)), off0)
        w, _, _ = self.jump_dists()
        occurs = torch.rand(B, generator=gen) < lam
        comp = torch.where(torch.rand(B, generator=gen) < w, 0, 1)
        eps = torch.randn(B, DIM, generator=gen)
        return occurs.unsqueeze(1) * self.jump_sizes(comp, eps)
