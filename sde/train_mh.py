# Run: OMP_NUM_THREADS=1 python3 -u -m sde.train_mh [seed]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchsde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from sde.model import (DIM, HubSDE, Scaler, WindowSDE, ema_update, full_series,
                       load_frames)
from sde.train import (CENTER_WEIGHT, NLL_BUDGET, acf, sample_windows,
                       split_slice, stage_a)

OUT = ROOT / "sde"
SPAN = 336
BATCH = 64
ITERS = 400
LR = 3e-4
LAGS_RAW = [1, 6, 12, 24]
LAGS_DES = [1, 2, 3, 4, 6, 8, 12]
NLL_HARD = -2.80          # held-out one-step NLL floor


# each window's hour-of-day profile is its own mean over days at fixed hour;
# the trailing partial day is dropped
def deseason(x):
    B, T, D = x.shape
    r = x[:, :T - T % 24].reshape(B, -1, 24, D)
    return (r - r.mean(dim=1, keepdim=True)).reshape(B, -1, D)


# level direction (1,1,1)/sqrt(3) up to scale, plus the two hub spreads that
# span its orthogonal complement
def factors(r):
    return [r.mean(dim=2), r[..., 0] - r[..., 1], r[..., 0] - r[..., 2]]


# per-lag reversion rate -log(rho_L)/L: for an OU these coincide at every lag
# and, unlike rho itself, are linear in the quantity of interest
def implied_rates(f, lags):
    rho = acf(f, lags).clamp(0.05, 0.999)
    return -torch.log(rho) / torch.tensor([float(L) for L in lags])


# adds the deseasonalized level/congestion block, since raw autocorrelation is
# dominated by the diurnal cycle
def window_moments_mh(x, spike_thr, neg_thr):
    m = [x.mean(dim=(0, 1)), x.std(dim=(0, 1))]
    dx = x[:, 1:] - x[:, :-1]
    m.append(dx.std(dim=(0, 1)))
    m.append((dx ** 4).mean(dim=(0, 1)) / (dx ** 2).mean(dim=(0, 1)) ** 2)
    m.append(torch.sigmoid((dx.abs() - spike_thr) / 0.05).mean(dim=(0, 1)) * 10)
    m.append(torch.sigmoid((neg_thr - x) / 0.05).mean(dim=(0, 1)) * 10)
    m.append((x[..., 0] - x[..., 1]).std(dim=1).mean().reshape(1))
    m.append((x[..., 0] - x[..., 2]).std(dim=1).mean().reshape(1))
    for d in range(DIM):
        m.append(acf(x[..., d], LAGS_RAW))
    for f in factors(deseason(x)):
        m.append(implied_rates(f, LAGS_DES))
        m.append(f.std(dim=1).mean().reshape(1))
    return torch.cat(m)


def rollout(model, x0, ow, E0, v0, span, jump_occ, comp, eps, params=None):
    sde = WindowSDE(model, ow)
    x, E, v = x0, E0, v0
    path = [x0]
    for h in range(span):
        ts = torch.tensor([float(h), float(h + 1)])
        x_prev = x
        if params is None:
            y = torchsde.sdeint(sde, sde.pack(x, E, v), ts, method="euler", dt=0.5)[-1]
        else:
            y = torchsde.sdeint_adjoint(sde, sde.pack(x, E, v), ts, method="euler",
                                        dt=0.5, adjoint_params=params)[-1]
        jsize = model.jump_mean[comp[h]] + model.jump_lsig.exp()[comp[h]] * eps[h]
        x = sde.unpack(y)[0] + jump_occ[h].unsqueeze(1) * jsize
        E, v = ema_update(E, v, x_prev, x)
        path.append(x)
    return torch.stack(path, dim=1)


# penalized because a drift bias too small to see instantaneously accumulates
# over a 14-day rollout
def rollout_bias(sim, xw):
    return (sim[:, 1:] - xw[:, 1:]).mean(dim=(0, 1))


def draw_batch(T, model, span, batch, gen):
    xw, ow, E0, v0, _ = sample_windows(*T["train"], span, batch, gen)
    with torch.no_grad():
        jump_occ = torch.stack([torch.rand(batch, generator=gen) <
                                model.lam(torch.tensor(float(h + 1)), ow)
                                for h in range(span)])
        comp = torch.where(torch.rand(span, batch, generator=gen) <
                           torch.sigmoid(model.jump_w_raw), 0, 1)
        eps = torch.randn(span, batch, DIM, generator=gen)
    return xw, ow, E0, v0, jump_occ, comp, eps


# frozen batch (same windows, jump draws, Brownian entropy every call) so
# checkpoint selection isn't affected by Monte Carlo noise
@torch.no_grad()
def score(model, pack, tgt, scale, weights, spike_thr, neg_thr):
    xw, ow, E0, v0, jo, cp, ep = pack
    np.random.seed(90210)
    sim = rollout(model, xw[:, 0], ow, E0, v0, xw.shape[1] - 1, jo, cp, ep)
    smm = (weights * ((window_moments_mh(sim, spike_thr, neg_thr)
                       - tgt) / scale) ** 2).mean().item()
    return smm, (rollout_bias(sim, xw) ** 2).sum().item()


def stage_b(model, T, nll_anchor, spike_thr, neg_thr, seed,
            iters=ITERS, batch=BATCH, span=SPAN, lr=LR):
    xtr, otr, Etr, vtr = T["train"]
    xva, ova, Eva, vva = T["val"]
    gen = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        tgt = torch.stack([
            window_moments_mh(sample_windows(xtr, otr, Etr, vtr, span, 256, gen)[0],
                              spike_thr, neg_thr)
            for _ in range(40)]).mean(dim=0)
    scale = tgt.abs().clamp_min(0.05)
    weights = torch.ones_like(tgt)
    weights[15:18] = 3.0          # negative-price frequency block

    for p in [model.lam_raw, model.jump_w_raw]:
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    ev = draw_batch(T, model, span, 128, torch.Generator().manual_seed(seed + 99))
    opt = torch.optim.Adam(params, lr=lr)
    log, best = [], (np.inf, None)
    for it in range(iters):
        xw, ow, E0, v0, jump_occ, comp, eps = draw_batch(T, model, span, batch, gen)
        # torchsde draws Brownian entropy from numpy's legacy global RNG, which
        # score() resets to a fixed seed; reseed here or training noise repeats
        # the eval draws
        np.random.seed((seed * 1000003 + it) % (2 ** 31 - 1))
        sim = rollout(model, xw[:, 0], ow, E0, v0, span,
                      jump_occ, comp, eps, params)
        smm = (weights * ((window_moments_mh(sim, spike_thr, neg_thr)
                           - tgt) / scale) ** 2).mean()
        center = (rollout_bias(sim, xw) ** 2).sum()
        idx = torch.randint(0, xtr.shape[0] - 1, (1024,), generator=gen)
        nll = model.transition_nll(xtr[idx], xtr[idx + 1], otr[idx],
                                   Etr[idx], vtr[idx])
        t0 = torch.zeros(1)
        mu = model.drift(xtr[idx], t0, otr[idx], Etr[idx], vtr[idx]) \
            + model.jump_compensator(t0, otr[idx])
        loss = smm + 1.0 * nll + CENTER_WEIGHT * (center + (mu.mean(dim=0) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        log.append((it, smm.item(), center.item(), nll.item()))
        if it % 20 == 0 or it == iters - 1:
            with torch.no_grad():
                va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                          Eva[:-1], vva[:-1]).item()
            e_smm, e_center = score(model, ev, tgt, scale, weights, spike_thr, neg_thr)
            obj = e_smm + CENTER_WEIGHT * e_center
            print(f"iter {it:3d}  eval smm {e_smm:.4f}  eval center {e_center:.2e}"
                  f"  obj {obj:.4f}  val nll {va:.4f}", flush=True)
            # accept only within the NLL budget of this seed's stage A optimum
            # and under the absolute floor NLL_HARD
            if va <= min(nll_anchor + NLL_BUDGET, NLL_HARD) and obj < best[0]:
                best = (obj, {k: p.clone() for k, p in model.state_dict().items()})
    for p in model.parameters():
        p.requires_grad_(True)
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"stage B kept checkpoint with objective {best[0]:.4f} within NLL budget")
    else:
        print("stage B never met the NLL budget; keeping stage A weights")
    return (best[1] is not None,
            pd.DataFrame(log, columns=["iter", "smm", "center", "nll"]))


def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, splits = load_frames(ROOT)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, E, v, off = full_series(ROOT, scaler)

    T = {}
    for name in ["train", "val"]:
        m = split_slice(index, name)
        T[name] = tuple(torch.tensor(a[m]) for a in (x, off, E, v))
    print(f"seed {seed}  train {T['train'][0].shape[0]}h  val {T['val'][0].shape[0]}h")

    model = HubSDE()
    nll_a, log_a = stage_a(model, T)
    spike_thr = torch.tensor(25.0 / scaler.std)
    neg_thr = torch.tensor(-scaler.mean / scaler.std)
    used_b, log_b = stage_b(model, T, nll_a, spike_thr, neg_thr, seed)
    with torch.no_grad():
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1], Eva[:-1], vva[:-1]).item()
    print(f"final val nll {va:.4f} (stage {'B' if used_b else 'A'})")
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "stage": "B" if used_b else "A", "seed": seed},
               OUT / f"checkpoint_mh_{seed}.pt")
    log_a.to_csv(OUT / f"train_log_mh_{seed}_a.csv", index=False)
    log_b.to_csv(OUT / f"train_log_mh_{seed}_b.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
