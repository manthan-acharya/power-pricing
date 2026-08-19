# Run: OMP_NUM_THREADS=1 python3 -u -m sde.train_pin [probe|train] [seed]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchsde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from sde.model import Scaler, WindowSDE, ema_update, full_series, load_frames
from sde.model_pin import PinnedSDE, pinned_rates
from sde.train import CENTER_WEIGHT, NLL_BUDGET, sample_windows, split_slice
from sde.train_mh import (BATCH, ITERS, LR, NLL_HARD, SPAN, draw_batch,
                          rollout_bias, window_moments_mh)
from sde.train_twofactor import factor_halflives

OUT = ROOT / "sde"
SLOPE_WEIGHT = CENTER_WEIGHT


# residual feedback as a fraction of the pinned rate it would corrupt, so the
# constraint is scale-free across factors whose rates differ by 4x
def slope_penalty(model, x, off, E, v):
    b = model.residual_slope(x, torch.zeros(1), off, E, v)
    return ((b / model.kappa) ** 2).sum()


# sde.train_mh.rollout with the jump sizes drawn per factor and rotated, as in
# sde.train_twofactor.rollout
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
        x = sde.unpack(y)[0] + jump_occ[h].unsqueeze(1) * model.jump_sizes(comp[h], eps[h])
        E, v = ema_update(E, v, x_prev, x)
        path.append(x)
    return torch.stack(path, dim=1)


# sde.train.stage_a plus the residual-slope penalty: left free, the one-step
# likelihood buys the fast level reversion back through the residual and the
# pinned rate means nothing
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
            t0 = torch.zeros(1)
            mu = model.drift(xtr[idx], t0, otr[idx], Etr[idx], vtr[idx]) \
                + model.jump_compensator(t0, otr[idx])
            slope = slope_penalty(model, xtr[idx], otr[idx], Etr[idx], vtr[idx])
            loss = nll + CENTER_WEIGHT * (mu.mean(dim=0) ** 2).sum() \
                + SLOPE_WEIGHT * slope
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += nll.item() * len(idx)
        sched.step()
        with torch.no_grad():
            va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                      Eva[:-1], vva[:-1]).item()
            sl = slope_penalty(model, xtr, otr, Etr, vtr).item()
        log.append((ep, tot / n, va, sl))
        if va < best[0]:
            best = (va, {k: p.clone() for k, p in model.state_dict().items()})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  train nll {tot / n:.4f}  val nll {va:.4f}"
                  f"  slope pen {sl:.4f}", flush=True)
    model.load_state_dict(best[1])
    print(f"stage A best val nll {best[0]:.4f}", flush=True)
    return best[0], pd.DataFrame(log, columns=["epoch", "train_nll", "val_nll", "slope"])


# restated rather than imported from sde.train_mh, whose score() resolves
# rollout in its own namespace and would draw jumps in hub coordinates
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
        # score() resets to a fixed seed
        np.random.seed((seed * 1000003 + it) % (2 ** 31 - 1))
        sim = rollout(model, xw[:, 0], ow, E0, v0, span, jump_occ, comp, eps, params)
        smm = (weights * ((window_moments_mh(sim, spike_thr, neg_thr)
                           - tgt) / scale) ** 2).mean()
        center = (rollout_bias(sim, xw) ** 2).sum()
        idx = torch.randint(0, xtr.shape[0] - 1, (1024,), generator=gen)
        nll = model.transition_nll(xtr[idx], xtr[idx + 1], otr[idx],
                                   Etr[idx], vtr[idx])
        t0 = torch.zeros(1)
        mu = model.drift(xtr[idx], t0, otr[idx], Etr[idx], vtr[idx]) \
            + model.jump_compensator(t0, otr[idx])
        slope = slope_penalty(model, xtr[idx], otr[idx], Etr[idx], vtr[idx])
        loss = smm + 1.0 * nll + CENTER_WEIGHT * (center + (mu.mean(dim=0) ** 2).sum()) \
            + SLOPE_WEIGHT * slope
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        log.append((it, smm.item(), center.item(), nll.item(), slope.item()))
        if it % 20 == 0 or it == iters - 1:
            with torch.no_grad():
                va = model.transition_nll(xva[:-1], xva[1:], ova[:-1],
                                          Eva[:-1], vva[:-1]).item()
                b = model.residual_slope(xtr, torch.zeros(1), otr, Etr, vtr)
            e_smm, e_center = score(model, ev, tgt, scale, weights, spike_thr, neg_thr)
            obj = e_smm + CENTER_WEIGHT * e_center
            keff = (model.kappa + b).numpy()
            print(f"iter {it:3d}  eval smm {e_smm:.4f}  eval center {e_center:.2e}"
                  f"  obj {obj:.4f}  val nll {va:.4f}"
                  f"  K_eff {np.round(keff, 4)}", flush=True)
            # accept only within the NLL budget of this seed's stage A optimum
            # and under the absolute floor NLL_HARD
            if va <= min(nll_anchor + NLL_BUDGET, NLL_HARD) and obj < best[0]:
                best = (obj, {k: p.clone() for k, p in model.state_dict().items()})
    for p in model.parameters():
        p.requires_grad_(True)
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"stage B kept checkpoint with objective {best[0]:.4f} within NLL budget",
              flush=True)
    else:
        print("stage B never met the NLL budget; keeping stage A weights", flush=True)
    return (best[1] is not None,
            pd.DataFrame(log, columns=["iter", "smm", "center", "nll", "slope"]))


def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, splits = load_frames(ROOT)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, E, v, off = full_series(ROOT, scaler)
    T = {}
    for name in ["train", "val"]:
        m = split_slice(index, name)
        T[name] = tuple(torch.tensor(a[m]) for a in (x, off, E, v))
    kappa = pinned_rates(index[split_slice(index, "train")],
                         x[split_slice(index, "train")])
    return scaler, x, E, v, off, T, kappa


def report(model, scaler, x, E, v, off, T):
    xtr, otr, Etr, vtr = T["train"]
    with torch.no_grad():
        b = model.residual_slope(xtr, torch.zeros(1), otr, Etr, vtr).numpy()
    np.random.seed(5)
    hl = factor_halflives(model, scaler, x, E, v, off)
    print(f"  pinned K {np.round(model.kappa.numpy(), 4)}"
          f"  residual slope {np.round(b, 4)}"
          f"  effective K {np.round(model.kappa.numpy() + b, 4)}", flush=True)
    print(f"  rollout half-lives level {hl[0]:.2f}h  spreads {hl[1]:.2f}/{hl[2]:.2f}h"
          f"  ratio {np.mean(hl[1:]) / hl[0]:.3f}   (observed 13.04h / 0.383)", flush=True)
    return hl


def main(mode, seed):
    scaler, x, E, v, off, T, kappa = setup(seed)
    print(f"seed {seed}  train {T['train'][0].shape[0]}h  val {T['val'][0].shape[0]}h")
    print(f"pinned rates {np.round(kappa, 5)} /h", flush=True)

    model = PinnedSDE(kappa)
    print(f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)
    nll_a, log_a = stage_a(model, T)
    print("after stage A:", flush=True)
    report(model, scaler, x, E, v, off, T)
    if mode == "probe":
        return

    spike_thr = torch.tensor(25.0 / scaler.std)
    neg_thr = torch.tensor(-scaler.mean / scaler.std)
    used_b, log_b = stage_b(model, T, nll_a, spike_thr, neg_thr, seed)
    with torch.no_grad():
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1], Eva[:-1], vva[:-1]).item()
    print(f"final val nll {va:.4f} (stage {'B' if used_b else 'A'})", flush=True)
    print("after stage B:", flush=True)
    report(model, scaler, x, E, v, off, T)

    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "kappa": kappa, "stage": "B" if used_b else "A", "seed": seed},
               OUT / f"checkpoint_pin_{seed}.pt")
    log_a.to_csv(OUT / f"train_log_pin_{seed}_a.csv", index=False)
    log_b.to_csv(OUT / f"train_log_pin_{seed}_b.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args and args[0] in ("probe", "train") else "train"
    seed = int(args[-1]) if args and args[-1].isdigit() else 7
    main(mode, seed)
