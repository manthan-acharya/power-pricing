import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchsde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sde.model import (DIM, HubSDE, Scaler, WindowSDE, ema_update,
                       full_series, load_frames)

OUT = ROOT / "sde"
SEED = 7
NLL_BUDGET = 0.35
CENTER_WEIGHT = 50.0


def split_slice(index, name):
    bounds = {"train": ("2025-01-01", "2026-02-01"),
              "val": ("2026-02-01", "2026-04-01")}
    a, b = bounds[name]
    t0 = pd.Timestamp(a, tz="US/Pacific")
    t1 = pd.Timestamp(b, tz="US/Pacific")
    return (index >= t0) & (index < t1)


def stage_a(model, T, epochs=80, batch=1024, lr=3e-3):
    xtr, otr, Etr, vtr = T["train"]
    xva, ova, Eva, vva = T["val"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = xtr.shape[0] - 1
    log, best = [], (np.inf, None)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            nll = model.transition_nll(xtr[idx], xtr[idx + 1], otr[idx],
                                       Etr[idx], vtr[idx])
            # pin batch-mean total drift (drift + jump compensator) at zero:
            # invisible to one-step NLL but compounds over rollouts
            t0 = torch.zeros(1)
            mu = model.drift(xtr[idx], t0, otr[idx], Etr[idx], vtr[idx]) \
                + model.jump_compensator(t0, otr[idx])
            loss = nll + CENTER_WEIGHT * (mu.mean(dim=0) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += nll.item() * len(idx)
        sched.step()
        with torch.no_grad():
            va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                      Eva[:-1], vva[:-1]).item()
        log.append((ep, tot / n, va))
        if va < best[0]:
            best = (va, {k: p.clone() for k, p in model.state_dict().items()})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  train nll {tot / n:.4f}  val nll {va:.4f}")
    model.load_state_dict(best[1])
    print(f"stage A best val nll {best[0]:.4f}")
    return best[0], pd.DataFrame(log, columns=["epoch", "train_nll", "val_nll"])


def acf(x, lags):
    xc = x - x.mean(dim=1, keepdim=True)
    var = (xc * xc).mean(dim=1).mean(dim=0) + 1e-8
    out = []
    for L in lags:
        cov = (xc[:, :-L] * xc[:, L:]).mean(dim=1).mean(dim=0)
        out.append(cov / var)
    return torch.stack(out)


def window_moments(x, spike_thr, neg_thr):
    lags = [1, 6, 12, 24]
    m = [x.mean(dim=(0, 1)), x.std(dim=(0, 1))]
    dx = x[:, 1:] - x[:, :-1]
    m.append(dx.std(dim=(0, 1)))
    m.append((dx ** 4).mean(dim=(0, 1)) / (dx ** 2).mean(dim=(0, 1)) ** 2)
    m.append(torch.sigmoid((dx.abs() - spike_thr) / 0.05).mean(dim=(0, 1)) * 10)
    m.append(torch.sigmoid((neg_thr - x) / 0.05).mean(dim=(0, 1)) * 10)
    m.append((x[..., 0] - x[..., 1]).std(dim=1).mean().reshape(1))
    m.append((x[..., 0] - x[..., 2]).std(dim=1).mean().reshape(1))
    for d in range(DIM):
        m.append(acf(x[..., d], lags))
    return torch.cat(m)


def sample_windows(xtr, otr, Etr, vtr, span, batch, gen):
    starts = torch.randint(0, xtr.shape[0] - span - 1, (batch,), generator=gen)
    xw = torch.stack([xtr[s:s + span + 1] for s in starts])
    return xw, otr[starts], Etr[starts], vtr[starts], starts


def rollout(model, x0, ow, E0, v0, span, jump_occ, comp, eps, adjoint_params):
    sde = WindowSDE(model, ow)
    x, E, v = x0, E0, v0
    path = [x0]
    for h in range(span):
        ts = torch.tensor([float(h), float(h + 1)])
        x_prev = x
        y = torchsde.sdeint_adjoint(sde, sde.pack(x, E, v), ts, method="euler",
                                    dt=0.5, adjoint_params=adjoint_params)[-1]
        jsize = model.jump_mean[comp[h]] + model.jump_lsig.exp()[comp[h]] * eps[h]
        x = sde.unpack(y)[0] + jump_occ[h].unsqueeze(1) * jsize
        E, v = ema_update(E, v, x_prev, x)
        path.append(x)
    return torch.stack(path, dim=1)


def stage_b(model, T, nll_anchor, spike_thr, neg_thr, iters=300, batch=96, span=72, lr=1e-4):
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
    weights[15:18] = 3.0  # negative-price frequency block

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
        nll = model.transition_nll(xtr[idx], xtr[idx + 1], otr[idx],
                                   Etr[idx], vtr[idx])
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lmp, cong, splits = load_frames(ROOT)
    train = splits["train"][0]
    scaler = Scaler(train.mean().values, train.std().values)
    index, x, E, v, off = full_series(ROOT, scaler)

    T = {}
    for name in ["train", "val"]:
        m = split_slice(index, name)
        T[name] = tuple(torch.tensor(a[m]) for a in (x, off, E, v))
    print(f"train {T['train'][0].shape[0]}h  val {T['val'][0].shape[0]}h")

    spread = lmp.iloc[:, 0] - lmp.iloc[:, 1]
    cdiff = cong.iloc[:, 0] - cong.iloc[:, 1]
    print(f"NP15-SP15 spread vs congestion diff corr: {spread.corr(cdiff):.4f}")

    model = HubSDE()
    nll_a, log_a = stage_a(model, T)
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "stage": "A"}, OUT / "checkpoint_stage_a.pt")

    spike_thr = torch.tensor(25.0 / scaler.std)
    neg_thr = torch.tensor(-scaler.mean / scaler.std)
    used_b, best_any, log_b = stage_b(model, T, nll_a, spike_thr, neg_thr)
    with torch.no_grad():
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                  Eva[:-1], vva[:-1]).item()
    print(f"final val nll {va:.4f} (stage {'B' if used_b else 'A'})")
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "stage": "B" if used_b else "A"}, OUT / "checkpoint.pt")
    if best_any[2] is not None:
        torch.save({"model": best_any[2], "scaler": scaler.state_dict(),
                    "stage": "B-smm"}, OUT / "checkpoint_smm_best.pt")
    log_a.to_csv(OUT / "train_log_a.csv", index=False)
    log_b.to_csv(OUT / "train_log_b.csv", index=False)


if __name__ == "__main__":
    main()
