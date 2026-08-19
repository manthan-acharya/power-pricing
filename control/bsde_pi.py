import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.bsde import (DT, EPS_FINAL, EPS_SOC, SEED, DeepBSDE,
                          FullProblem, PriceGenerator, day_conditioning,
                          eps_at, hamiltonian, make_train_sampler)
from sde.model import HubSDE, Scaler, full_series

OUT = ROOT / "control"
PI_MAX = 60.0 / 1000.0


# day conditioning plus the terminal value, scaled to [0, 1]
def conditioning(off0, E0, v0, pi_kusd):
    return torch.cat([day_conditioning(off0, E0, v0),
                      (pi_kusd / PI_MAX).reshape(-1, 1)], dim=1)


# control.bsde.rollout_bsde with a per-sample terminal value pi_kusd
def rollout(problem, nets, s0, x0, off0, E0, v0, pi_kusd, gen_rng, eps):
    B = s0.shape[0]
    cond = conditioning(off0, E0, v0, pi_kusd)
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
    return ((Y - pi_kusd * s) ** 2).mean()


def train(problem, nets, sampler, iters=8000, batch=512, lr=1e-3,
          log_every=200):
    torch.manual_seed(SEED)
    gen_rng = torch.Generator().manual_seed(SEED + 1)
    opt = torch.optim.Adam(nets.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr / 20)
    log = []
    for it in range(iters):
        s0, x0, off0, E0, v0 = sampler(batch, gen_rng)
        pi = torch.rand(batch, generator=gen_rng) * PI_MAX
        loss = rollout(problem, nets, s0, x0, off0, E0, v0, pi, gen_rng,
                       eps_at(it, iters, EPS_SOC, EPS_FINAL))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5.0)
        opt.step()
        sched.step()
        log.append((it, loss.item()))
        if it % log_every == 0 or it == iters - 1:
            print(f"iter {it:5d}  terminal loss {loss.item():.5f}", flush=True)
    return pd.DataFrame(log, columns=["iter", "loss"])


class PiPolicy:
    def __init__(self, root):
        ck = torch.load(root / "control/bsde_pi_checkpoint.pt", weights_only=False)
        self.nets = DeepBSDE(n_extra=1)
        self.nets.load_state_dict(ck["nets"])
        self.nets.eval()
        self.eps = ck["eps"]

    @torch.no_grad()
    def act(self, k, s, x_std, off0, E0, v0, pi_usd):
        cond = conditioning(off0, E0, v0,
                            torch.tensor([pi_usd / 1000.0], dtype=torch.float32))
        xdev = torch.as_tensor(x_std, dtype=torch.float32).reshape(1, -1) - E0
        feats = self.nets.feats(torch.tensor([s], dtype=torch.float32), xdev, cond)
        return self.nets.z(k, feats)[0, 0].item() / self.eps


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

    problem = FullProblem(PriceGenerator(model, scaler))
    nets = DeepBSDE(n_extra=1)
    log = train(problem, nets, make_train_sampler(index, x, E, v, off))
    log.to_csv(OUT / "bsde_pi_train_log.csv", index=False)
    torch.save({"nets": nets.state_dict(), "eps": EPS_FINAL, "pi_max": PI_MAX},
               OUT / "bsde_pi_checkpoint.pt")


if __name__ == "__main__":
    main()
