import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.bsde import (CAP, C_DEG, DIM, DT, ETA, HUB, P_MAX,
                          PriceGenerator, day_starts)
from sde.model import HubSDE, Scaler, extended_series, full_series, load_frames

OUT = ROOT / "control"
N_X = 29
HALF_WIDTH = 3.5
RADIUS = 5
S_GRID = np.linspace(0.0, CAP, 21)
U_GRID = np.linspace(-P_MAX, P_MAX, 21)
WORKERS = max(1, (os.cpu_count() or 2) - 2)

_ctx = {}


def load_context(extended=False):
    ck = torch.load(ROOT / "sde/checkpoint.pt", weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])
    series = extended_series if extended else full_series
    index, x, E, v, off = series(ROOT, scaler)
    bck = torch.load(ROOT / "control/bsde_checkpoint.pt", weights_only=False)
    return {"gen": PriceGenerator(model, scaler), "index": index, "x": x,
            "E": E, "v": v, "off": off, "pi_term": bck["pi_term"]}


def _init_worker(extended):
    torch.set_num_threads(1)
    _ctx.update(load_context(extended))


# Moment-matched 1-h Euler transition of the full 3-hub price state, the same
# generator the deep BSDE integrates, as a Gaussian kernel truncated to
# +-RADIUS cells per hub and renormalized (edge mass piles onto boundary nodes).
def transition(gen, k, grid, off0, E0, v0):
    h = grid[0][1] - grid[0][0]
    lo = np.array([g[0] for g in grid])
    nodes = np.stack(np.meshgrid(*grid, indexing="ij"), -1).reshape(-1, DIM)
    n = len(nodes)
    X = torch.tensor(nodes, dtype=torch.float32)
    mu, chol = gen.coeffs(X, k, off0.expand(n, 3), E0.expand(n, DIM),
                          v0.expand(n, 1))
    mean = (X + mu * DT).double().numpy()
    Linv = np.linalg.inv(chol.double().numpy() * np.sqrt(DT))
    centre = np.clip(np.rint((mean - lo) / h).astype(int), 0, N_X - 1)
    offs = np.arange(-RADIUS, RADIUS + 1)
    box = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, DIM)
    rows, cols, vals = [], [], []
    for a in range(0, n, 2000):
        b = min(n, a + 2000)
        cell = np.clip(centre[a:b, None] + box, 0, N_X - 1)
        z = np.einsum("nij,nmj->nmi", Linv[a:b], lo + cell * h - mean[a:b, None])
        w = np.exp(-0.5 * (z ** 2).sum(-1))
        w /= w.sum(1, keepdims=True)
        rows.append(np.repeat(np.arange(a, b, dtype=np.int32), len(box)))
        cols.append(((cell[..., 0] * N_X + cell[..., 1]) * N_X
                     + cell[..., 2]).ravel().astype(np.int32))
        vals.append(w.ravel())
    T = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return T, nodes


# max over the SOC-clamped control grid of reward + continuation, with the
# continuation linearly interpolated in post-decision SOC
def best_action(price_kusd, s, cont):
    best = np.full((len(s), cont.shape[1]), -np.inf)
    arg = np.zeros_like(best)
    for u in U_GRID:
        u_eff = np.clip(u, -np.minimum(P_MAX, (CAP - s) / (ETA * DT)),
                        np.minimum(P_MAX, ETA * s / DT))
        s_next = s + np.where(u_eff >= 0, -u_eff / ETA, -ETA * u_eff) * DT
        i = np.clip(np.searchsorted(S_GRID, s_next) - 1, 0, len(S_GRID) - 2)
        w = ((s_next - S_GRID[i]) / (S_GRID[i + 1] - S_GRID[i]))[:, None]
        val = ((price_kusd[None] * u_eff[:, None] - C_DEG * np.abs(u_eff)[:, None]) * DT
               + (1 - w) * cont[i] + w * cont[i + 1])
        better = val > best
        best = np.where(better, val, best)
        arg = np.where(better, u_eff[:, None], arg)
    return best, arg


def solve_day(ctx, d, pi_term):
    off0, E0, v0 = (torch.tensor(ctx[a][d:d + 1]) for a in ("off", "E", "v"))
    grid = [E0[0, j].item() + np.linspace(-HALF_WIDTH, HALF_WIDTH, N_X)
            for j in range(DIM)]
    V = pi_term * np.repeat(S_GRID[:, None], N_X ** DIM, 1)
    conts = [None] * 24
    for k in range(23, -1, -1):
        T, nodes = transition(ctx["gen"], k, grid, off0, E0, v0)
        conts[k] = (T @ V.T).T
        V, _ = best_action(ctx["gen"].price_usd(nodes) / 1000.0, S_GRID, conts[k])
    return grid, conts


# Continuation E[V_{k+1} | x_k] at the realized price state of each hour, one
# value per post-decision SOC node; the day's policy needs nothing else.
def realized_continuations(job):
    i, d, pi_term = job
    t = time.time()
    grid, conts = solve_day(_ctx, d, pi_term)
    lo = [g[0] for g in grid]
    hi = [g[-1] for g in grid]
    C = np.empty((24, len(S_GRID)))
    for k in range(24):
        f = RegularGridInterpolator(tuple(grid), np.moveaxis(
            conts[k].reshape(len(S_GRID), N_X, N_X, N_X), 0, -1))
        C[k] = f(np.clip(_ctx["x"][d + k], lo, hi))[0]
    return i, C, time.time() - t


# pi_terms in k$/MWh, one per day start
def solve_days(starts, pi_terms, extended=False, workers=WORKERS):
    conts, secs = [None] * len(starts), []
    jobs = [(i, d, p) for i, (d, p) in enumerate(zip(starts, pi_terms))]
    with Pool(workers, initializer=_init_worker, initargs=(extended,)) as pool:
        for i, C, sec in pool.imap_unordered(realized_continuations, jobs):
            conts[i] = C
            secs.append(sec)
    return conts, secs


def run_dp_policy(prices_usd, starts, conts, s0=50.0):
    s, pnl = s0, []
    for i, d in enumerate(starts):
        for k in range(24):
            p = prices_usd[d + k]
            _, u = best_action(np.array([p / 1000.0]), np.array([s]),
                               conts[i][k][:, None])
            u = float(u[0, 0])
            s = s - u / ETA if u >= 0 else s - ETA * u
            pnl.append(p * u - C_DEG * 1000.0 * abs(u))
    return np.array(pnl)


def main():
    from backtest.engine import (Policy, daily_totals, paired_daily_test,
                                 run_policy_on_prices)
    torch.set_num_threads(1)
    ctx = load_context()
    index = ctx["index"]
    lmp, _, _ = load_frames(ROOT)
    prices = lmp.values[:, HUB]
    starts = day_starts(index, "test")
    print(f"grid: {len(S_GRID)} SOC x {N_X}^{DIM} price = "
          f"{len(S_GRID) * N_X ** DIM:,} nodes per hour, {len(U_GRID)} controls")

    conts, secs = solve_days(starts, [ctx["pi_term"]] * len(starts))
    print(f"{len(starts)} day instances solved, {np.mean(secs):.0f} s each "
          f"on one core (range {min(secs):.0f}-{max(secs):.0f} s)")

    pnl_dp = run_dp_policy(prices, starts, conts)
    pnl_bsde, _, _ = run_policy_on_prices(Policy(ROOT), prices, ctx["x"],
                                          ctx["E"], ctx["v"], ctx["off"], starts)
    sig = paired_daily_test(pnl_dp, pnl_bsde)
    n = len(starts)
    print(f"realized test quarter: grid DP ${pnl_dp.sum():,.0f}  deep BSDE "
          f"${pnl_bsde.sum():,.0f}  daily edge ${sig['mean_daily_edge']:.0f}, "
          f"paired t {sig['paired_t']:.2f}, DP ahead on {sig['days_won']}/{n} days")

    days = index[starts].tz_convert("US/Pacific").date
    pd.DataFrame({"day": days, "grid_dp": daily_totals(pnl_dp, n),
                  "deep_bsde": daily_totals(pnl_bsde, n)}).round(1).to_csv(
        OUT / "grid_dp_backtest.csv", index=False)
    pd.Series({"soc_nodes": len(S_GRID), "price_nodes_per_hub": N_X,
               "nodes_per_hour": len(S_GRID) * N_X ** DIM,
               "solve_seconds_mean": np.mean(secs),
               "solve_seconds_max": max(secs),
               "grid_dp_total": pnl_dp.sum(), "deep_bsde_total": pnl_bsde.sum(),
               "mean_daily_edge": sig["mean_daily_edge"],
               "paired_t": sig["paired_t"], "ci_lo": sig["ci_lo"],
               "ci_hi": sig["ci_hi"], "days_dp_ahead": sig["days_won"],
               "n_days": n}).round(3).to_csv(OUT / "grid_dp_summary.csv", header=False)


if __name__ == "__main__":
    main()
