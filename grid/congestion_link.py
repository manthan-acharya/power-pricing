import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grid.dcopf import solve
from grid.topology import build

LEVEL = np.ones(3) / np.sqrt(3)
# orthonormal basis of the plane orthogonal to the level direction
SPREAD = np.linalg.qr(np.column_stack([LEVEL, np.eye(3)[:, :2]]))[0][:, 1:]
SPLITS = {"train": ("2025-01-01", "2026-02-01"), "val": ("2026-02-01", "2026-04-01"),
          "test": ("2026-04-01", "2026-07-01")}


def load():
    f = {n: pd.read_csv(ROOT / f"data/processed/{n}_hourly.csv", index_col=0,
                        parse_dates=True) for n in ("lmp", "congestion", "energy")}
    loss = f["lmp"].values - f["energy"].values - f["congestion"].values
    assert np.allclose(f["energy"].values[:, 0], f["energy"].values[:, 1:].T), \
        "energy component must be common across hubs"
    return f["lmp"], f["congestion"].values, f["energy"].values, loss


def total_var(X):
    return np.trace(np.cov(X.T))


def rec(rows, section, metric, value):
    rows.append({"section": section, "metric": metric, "value": value})


# LMP = energy*1 + congestion + loss with energy common, so the level direction
# carries the energy component and the orthogonal spread plane carries only
# congestion and losses.
def spread_space(lmp, cong, energy, loss, rows):
    dP, dC, dL = (np.diff(a, axis=0) for a in (lmp.values, cong, loss))
    de = np.diff(energy[:, 0])

    tot = total_var(dP)
    print(f"hourly increment variance (trace) {tot:.1f} ($/MWh)^2 over {len(dP)} steps")
    print(f"  energy*1 {total_var(de[:, None] * np.ones(3)) / tot:.3f}"
          f"  congestion {total_var(dC) / tot:.3f}  loss {total_var(dL) / tot:.3f}")

    for nm, val in (("energy", total_var(de[:, None] * np.ones(3)) / tot),
                    ("congestion", total_var(dC) / tot), ("loss", total_var(dL) / tot)):
        rec(rows, "decomposition", f"increment_var_share_{nm}", val)

    lev = np.var(dP @ LEVEL) / tot
    R = np.corrcoef(dP.T)
    w, V = np.linalg.eigh(R)
    print(f"  variance along level direction {lev:.3f}; PC1 of the increment "
          f"correlation explains {w[-1] / 3:.3f} at cosine "
          f"{abs(V[:, -1] @ LEVEL):.4f} to (1,1,1)/sqrt(3)")

    rec(rows, "decomposition", "increment_var_share_level_direction", lev)
    rec(rows, "decomposition", "increment_corr_pc1_share", w[-1] / 3)
    rec(rows, "decomposition", "increment_corr_pc1_cos_to_ones", abs(V[:, -1] @ LEVEL))

    p, pc, pl = dP @ SPREAD, dC @ SPREAD, dL @ SPREAD
    r2 = 1 - total_var(p - pc) / total_var(p)
    rec(rows, "decomposition", "spread_plane_r2_congestion_all", r2)
    rec(rows, "decomposition", "spread_plane_share_loss", total_var(pl) / total_var(p))
    print(f"  in the spread plane: congestion explains {r2:.4f} of innovation "
          f"variance, losses contribute {total_var(pl) / total_var(p):.4f}")
    for name, (a, b) in SPLITS.items():
        m = window(lmp.index, a, b)[1:]
        q, qc = dP[m] @ SPREAD, dC[m] @ SPREAD
        r2s = 1 - total_var(q - qc) / total_var(q)
        rec(rows, "decomposition", f"spread_plane_r2_congestion_{name}", r2s)
        print(f"    {name:5s} n={m.sum():5d}  congestion R^2 {r2s:.4f}")
    return dP, dC


def window(index, a, b):
    t0 = pd.Timestamp(a, tz="US/Pacific")
    t1 = pd.Timestamp(b, tz="US/Pacific") - pd.Timedelta(hours=1)
    return (index >= t0) & (index <= t1)


# Direction vs magnitude by decile: flat under DC-OPF (direction is a PTDF row,
# magnitude a shadow price), falling for a correlated Gaussian.
def dispersion(C, n_dec=10):
    C = C[np.linalg.norm(C, axis=1) > 1e-9]
    v1 = np.linalg.eigh(np.cov(C.T))[1][:, -1]
    mag = np.linalg.norm(C, axis=1)
    edge = np.quantile(mag, np.linspace(0, 1, n_dec + 1))
    out = []
    for i in range(n_dec):
        m = (mag >= edge[i]) & (mag <= edge[i + 1])
        if m.sum() < 20:
            out.append(np.nan)
            continue
        U = C[m] / mag[m, None]
        U *= np.sign(U @ v1)[:, None]
        out.append(np.degrees(np.arccos(np.clip(np.linalg.norm(U.mean(0)), 0, 1))))
    return np.array(out)


def participation_ratio(C):
    w = np.linalg.eigvalsh(np.cov(C.T))
    return w.sum() ** 2 / (w ** 2).sum()


def gaussian_null(C, n=200, seed=0):
    rng = np.random.default_rng(seed)
    A = np.linalg.cholesky(np.cov(C.T) + 1e-12 * np.eye(3))
    return np.array([dispersion(rng.standard_normal(C.shape) @ A.T + C.mean(0))
                     for _ in range(n)])


# Congestion vectors at three arbitrarily chosen case14 buses under randomized
# load, giving the DC-OPF prediction for the shape (not the CAISO
# correspondence) of a congestion field.
def ptdf_reference_field(n_draw=2000, hubs=(2, 8, 13), n_lines=3, seed=7):
    case = build()
    rng = np.random.default_rng(seed)
    base = case["load"].copy()

    def draw(k):
        s = np.exp(rng.normal(0, 0.18, size=(k, 1)))
        return np.clip(base * s * (1 + 0.35 * rng.normal(size=(k, len(base)))), 0, None)

    case["limits"] = {}
    flows = []
    for d in draw(300):
        case["load"] = d
        flows.append(solve(case)["flows"])
    flows = np.array(flows)
    lines = np.argsort(-flows.std(0))[:n_lines]
    # limits at the 75th percentile of unconstrained flow so each can bind
    case["limits"] = dict(zip(lines.tolist(),
                              np.quantile(np.abs(flows[:, lines]), 0.75, axis=0)))

    C, nbind, infeasible = [], [], 0
    for d in draw(n_draw):
        case["load"] = d
        try:
            r = solve(case)
        except AssertionError:
            infeasible += 1  # load unservable within the limits (shed event)
            continue
        C.append(r["congestion"][list(hubs)])
        nbind.append(sum(abs(v) > 1e-7 for v in r["mu"].values()))
    C, nbind = np.array(C), np.array(nbind)
    print(f"case14 reference field: lines {lines.tolist()} limited, "
          f"{np.mean(nbind):.2f} constraints binding on average, "
          f"{np.mean(nbind == 0):.2f} of draws uncongested, "
          f"{infeasible / n_draw:.3f} dropped as infeasible")
    return C, nbind


def geometry(C_real, C_toy, rows):
    d_real, d_toy = dispersion(C_real), dispersion(C_toy)
    null = gaussian_null(C_real)
    print(f"congestion covariance participation ratio: CAISO "
          f"{participation_ratio(C_real):.3f}, case14 {participation_ratio(C_toy):.3f} "
          f"(3.0 = isotropic)")
    rec(rows, "geometry", "participation_ratio_caiso", participation_ratio(C_real))
    rec(rows, "geometry", "participation_ratio_case14", participation_ratio(C_toy))
    # Eigen-decomposition of the congestion covariance: PC1 share and loading
    # give the field's common direction.
    w, V = np.linalg.eigh(np.cov(C_real.T))
    rec(rows, "geometry", "congestion_pc1_share_caiso", w[-1] / w.sum())
    for i, lab in enumerate(("NP15", "SP15", "ZP26")):
        rec(rows, "geometry", f"congestion_pc1_loading_{lab}", V[i, -1])
    print(f"  CAISO congestion PC1 {w[-1] / w.sum():.3f} of variance, "
          f"loading {V[:, -1].round(3)}")
    print("  angular dispersion, low to high |C| decile (deg)")
    print(f"    CAISO          {np.round(d_real, 1)}")
    print(f"    gaussian null  {np.round(null.mean(0), 1)} +/- {np.round(null.std(0), 1)}")
    print(f"    case14 DC-OPF  {np.round(d_toy, 1)}   median {np.nanmedian(d_toy):.1f}")
    rng_null = null[:, 0] - null[:, -1]
    for i in range(10):
        rec(rows, "geometry", f"dispersion_deg_caiso_d{i + 1}", d_real[i])
        rec(rows, "geometry", f"dispersion_deg_null_d{i + 1}", null.mean(0)[i])
        rec(rows, "geometry", f"dispersion_deg_case14_d{i + 1}", d_toy[i])
    rec(rows, "geometry", "magnitude_dependence_caiso_deg", d_real[0] - d_real[-1])
    rec(rows, "geometry", "magnitude_dependence_null_deg", rng_null.mean())
    rec(rows, "geometry", "magnitude_dependence_null_sd_deg", rng_null.std())
    print(f"  magnitude dependence (first minus last decile): CAISO "
          f"{d_real[0] - d_real[-1]:.1f} deg vs null {rng_null.mean():.1f} +/- "
          f"{rng_null.std():.1f}, outside all {len(null)} null draws "
          f"{d_real[0] - d_real[-1] < rng_null.min()}")
    for thr in (1.0, 5.0):
        s = C_real[np.linalg.norm(C_real, axis=1) > thr]
        d, n = dispersion(s), gaussian_null(s, n=50, seed=2)
        rec(rows, "geometry", f"magnitude_dependence_caiso_absC_gt{thr:.0f}_deg", d[0] - d[-1])
        print(f"    restricted to |C|>{thr:.0f} (n={len(s)}): {d[0] - d[-1]:.1f} deg vs "
              f"null {np.mean(n[:, 0] - n[:, -1]):.1f}")
    return d_real, d_toy, null


def halflife(r, hod):
    prof = pd.Series(r).groupby(hod).transform("mean").values
    e = r - prof
    phi = np.corrcoef(e[:-1], e[1:])[0, 1]
    return np.log(2) / -np.log(phi) if 0 < phi < 1 else np.nan


# Checks the fitted SDE's leading-eigenvector alignment, jump direction, and
# level/spread half-life split against the decomposition, the parts stage-B
# moment matching does not already constrain.
def sde_consistency(lmp, cong, energy, rows):
    import torch
    from sde.model import (HubSDE, Scaler, clock_offsets, ema_states, simulate)

    ck = torch.load(ROOT / "sde" / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = HubSDE()
    model.load_state_dict(ck["model"])
    model.eval()
    scaler = Scaler(ck["scaler"]["mean"], ck["scaler"]["std"])

    R = model.corr_factor().detach().numpy()
    R = R @ R.T
    w, V = np.linalg.eigh(R)
    print(f"fitted diffusion correlation: level-direction share {LEVEL @ R @ LEVEL / 3:.4f}, "
          f"PC1 {w[-1] / 3:.4f} at cosine {abs(V[:, -1] @ LEVEL):.4f} to (1,1,1)/sqrt(3)")
    rec(rows, "sde", "diffusion_corr_level_share", LEVEL @ R @ LEVEL / 3)
    rec(rows, "sde", "diffusion_corr_pc1_cos_to_ones", abs(V[:, -1] @ LEVEL))
    _, jm, _ = model.jump_dists()
    for k, name in ((0, "up"), (1, "down")):
        u = jm[k].detach().numpy()
        u = u / np.linalg.norm(u)
        rec(rows, "sde", f"jump_{name}_cos_to_ones", abs(u @ LEVEL))
        print(f"  fitted {name}-jump direction {u.round(3)}, cosine "
              f"{abs(u @ LEVEL):.4f} to the level direction")

    x = scaler.transform(lmp.values.astype(np.float32))
    E, v = ema_states(x)
    off = clock_offsets(lmp.index)
    idx = np.arange(0, len(x) - 1, 7)
    args = [torch.tensor(a[idx]) for a in (x, off, E, v)]
    t = torch.zeros(1)
    resp = {}
    for name, d in (("NP15-SP15", [0.5, -0.5, 0.0]), ("NP15-ZP26", [0.5, 0.0, -0.5]),
                    ("level", [1 / 3, 1 / 3, 1 / 3])):
        d = np.array(d)
        dv = torch.tensor(d, dtype=torch.float32)
        eps = 1e-3
        with torch.no_grad():
            J = (model.drift(args[0] + eps * dv, t, *args[1:])
                 - model.drift(args[0] - eps * dv, t, *args[1:])) / (2 * eps)
        resp[name] = J.numpy() @ d / (d @ d)
        r = resp[name]
        rec(rows, "sde", f"drift_response_{name}_per_h", r.mean())
        rec(rows, "sde", f"drift_response_{name}_frac_restoring", (r < 0).mean())
        print(f"  d(drift)/d({name}) = {r.mean():+.4f}/h on average, restoring at "
              f"{(r < 0).mean():.3f} of sampled states")

    B, hours = 64, 24 * 30
    i0 = len(x) - hours
    def rep(a):
        return torch.tensor(np.repeat(a[i0][None], B, 0))

    # simulate()'s seed pins only the jumps; torchsde draws Brownian motion
    # from numpy's legacy global RNG, so this is needed too.
    np.random.seed(5)
    sim = scaler.inverse(simulate(model, rep(x), rep(off), rep(E), rep(v),
                                  n_hours=hours, seed=5).numpy().transpose(1, 0, 2))
    hs = (off[i0, 0] + np.arange(sim.shape[1])) % 24
    hl_sim = np.nanmedian([[halflife(s.mean(1), hs), halflife(s[:, 0] - s[:, 1], hs),
                            halflife(s[:, 0] - s[:, 2], hs)] for s in sim], axis=0)

    hod = lmp.index.tz_convert("US/Pacific").hour.values
    P = lmp.values
    hl_real = [halflife(P.mean(1), hod), halflife(P[:, 0] - P[:, 1], hod),
               halflife(P[:, 0] - P[:, 2], hod)]
    hl_comp = [halflife(energy[:, 0], hod), halflife(cong[:, 0] - cong[:, 1], hod),
               halflife(cong[:, 0] - cong[:, 2], hod)]
    for i, lab in enumerate(("level", "NP15-SP15", "NP15-ZP26")):
        rec(rows, "sde", f"halflife_h_observed_{lab}", hl_real[i])
        rec(rows, "sde", f"halflife_h_caiso_component_{lab}", hl_comp[i])
        rec(rows, "sde", f"halflife_h_fitted_sde_{lab}", hl_sim[i])
    print("  deseasonalized half-lives (h)  [level, NP-SP, NP-ZP]")
    print(f"    observed prices    {np.round(hl_real, 2)}")
    print(f"    CAISO components   {np.round(hl_comp, 2)}   (energy, congestion diffs)")
    print(f"    fitted SDE         {np.round(hl_sim, 2)}")
    print(f"    spread/level ratio: observed {np.mean(hl_real[1:]) / hl_real[0]:.2f}, "
          f"components {np.mean(hl_comp[1:]) / hl_comp[0]:.2f}, "
          f"SDE {np.mean(hl_sim[1:]) / hl_sim[0]:.2f}")
    return R, jm.detach().numpy(), resp, np.array(hl_real), np.array(hl_comp), hl_sim


# Runs the marginal validation criterion on level and spread coordinates
# instead of per hub: a level-driven KS failure should survive projection onto
# the level and vanish on the spreads.
def marginal_localization(rows):
    from sde.ks_test import block_bootstrap_ks
    from sde.model import full_series, load_frames
    from sde.validate import (PATHS_PER_WINDOW, WINDOW_DAYS, load_model,
                              run_window, window_starts)

    model, scaler, _ = load_model(ROOT / "sde" / "checkpoint.pt")
    lmp, _, splits = load_frames(ROOT)
    index, x, E, v, off = full_series(ROOT, scaler)
    span = WINDOW_DAYS * 24
    starts = window_starts(index, splits["val"][0].index[0], 4, span)

    reals, sims, hod = [], [], []
    for k, i0 in enumerate(starts):
        sim = run_window(model, scaler, x, E, v, off, i0, span, PATHS_PER_WINDOW,
                         seed=100 + k)
        sims.append(sim[:, 1:])
        reals.append(lmp.values[i0 + 1:i0 + 1 + span])
        hod.append(index[i0 + 1:i0 + 1 + span].tz_convert("US/Pacific").hour)

    # the SDE's own coordinates: system level, then the two spreads
    proj = {"level": np.ones(3) / 3, "NP15-SP15": np.array([1.0, -1.0, 0.0]),
            "NP15-ZP26": np.array([1.0, 0.0, -1.0])}
    out = {}
    for i, (name, w) in enumerate(proj.items()):
        real = np.concatenate([rw @ w for rw in reals])
        rh = np.concatenate([np.asarray(h) for h in hod])
        sim = np.concatenate([sw @ w for sw in sims], axis=0)
        sh = np.concatenate([np.tile(np.asarray(h), (sw.shape[0], 1))
                             for sw, h in zip(sims, hod)], axis=0)
        r = block_bootstrap_ks(real, sim, rh, sh, seed=2000 + i)
        out[name] = r
        flat = sim.ravel()
        grid = np.linspace(min(real.min(), flat.min()), max(real.max(), flat.max()), 2001)
        gap = np.searchsorted(np.sort(flat), grid) / len(flat) - \
            np.searchsorted(np.sort(real), grid) / len(real)
        rec(rows, "marginal", f"std_ratio_sim_over_real_{name}", flat.std() / real.std())
        rec(rows, "marginal", f"max_gap_at_usd_{name}", grid[np.argmax(np.abs(gap))])
        print(f"  {name:10s} sim/real std {flat.std() / real.std():.3f}, "
              f"largest CDF gap at {grid[np.argmax(np.abs(gap))]:+.1f} $/MWh "
              f"(sim {'above' if gap[np.argmax(np.abs(gap))] > 0 else 'below'} real)")
        rec(rows, "marginal", f"ks_D_{name}", r["D"])
        rec(rows, "marginal", f"ks_p_block_{name}", r["p_block"])
        rec(rows, "marginal", f"ks_null_D95_{name}", r["d95_null"])
        print(f"  {name:10s} KS D {r['D']:.4f}  p_block {r['p_block']:.4f}  "
              f"blocks {r['b_real']}/{r['b_sim']} h  null D95 {r['d95_null']:.4f}")
    return out


def figure(dP, dC, disp_real, disp_toy, null, R, jump, hl_real, hl_comp, hl_sim):
    import matplotlib.pyplot as plt
    from plotstyle import ACCENT, NODE_COLORS, NODE_LABELS, apply
    apply()
    cols = list(NODE_COLORS)
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.4))

    a = ax[0, 0]
    for j, c in ((1, cols[1]), (2, cols[2])):
        x, y = dP[:, 0] - dP[:, j], dC[:, 0] - dC[:, j]
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        a.scatter(x, y, s=2, alpha=0.25, color=NODE_COLORS[c],
                  label=f"NP15-{NODE_LABELS[c]}  $R^2$={r2:.3f}")
    lim = np.percentile(np.abs(dP[:, 0] - dP[:, 1]), 99.5)
    a.plot([-lim, lim], [-lim, lim], lw=0.8, color=ACCENT)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("hourly change in hub spread (\\$/MWh)")
    a.set_ylabel("change in congestion differential")
    a.set_title("Hub Spread Innovations vs. Congestion Differential Innovations")
    a.legend(markerscale=4)

    a = ax[0, 1]
    d = np.arange(1, 11)
    a.fill_between(d, null.mean(0) - 2 * null.std(0), null.mean(0) + 2 * null.std(0),
                   color="#bbbbbb", alpha=0.5, label="covariance-matched gaussian")
    a.plot(d, disp_real, "o-", color=NODE_COLORS[cols[0]], ms=4, label="CAISO congestion")
    a.plot(d, disp_toy, "s-", color=NODE_COLORS[cols[2]], ms=4, label="case14 DC-OPF")
    a.set_xlabel("decile of $\\|C_t\\|$")
    a.set_ylabel("angular dispersion (deg)")
    a.set_title("Angular Dispersion vs. Congestion Magnitude")
    a.legend()

    a = ax[1, 0]
    V = np.linalg.eigh(R)[1]
    emp = np.linalg.eigh(np.corrcoef(dP.T))[1][:, -1]
    jd = np.abs(jump[0]) / np.linalg.norm(jump[0])
    for arr, lab, c, mk in ((np.abs(LEVEL), "energy factor (theory)", ACCENT, "-"),
                            (np.abs(emp), "observed increments", NODE_COLORS[cols[0]], "o-"),
                            (np.abs(V[:, -1]), "fitted SDE diffusion", NODE_COLORS[cols[1]], "s--"),
                            (jd, "fitted SDE jump size", NODE_COLORS[cols[2]], "^:")):
        a.plot(np.arange(3), arr, mk, color=c, ms=5, label=lab)
    a.set_xticks(np.arange(3))
    a.set_xticklabels([NODE_LABELS[c] for c in cols])
    a.set_ylim(0.45, 0.72)
    a.set_ylabel("leading eigenvector loading")
    a.set_title("Leading Eigenvector Loadings Across Hubs")
    a.legend(loc="upper left", fontsize=8)

    a = ax[1, 1]
    x = np.arange(3)
    for i, (arr, lab, c) in enumerate(((hl_comp, "CAISO components", ACCENT),
                                       (hl_real, "observed prices", NODE_COLORS[cols[0]]),
                                       (hl_sim, "fitted SDE", NODE_COLORS[cols[1]]))):
        a.bar(x + (i - 1) * 0.27, arr, 0.25, color=c, label=lab)
    a.set_xticks(x)
    a.set_xticklabels(["level\n(energy)", "NP15-SP15\n(congestion)", "NP15-ZP26"])
    a.set_ylabel("deseasonalized half-life (h)")
    a.set_title("Deseasonalized Mean-Reversion Half-Lives")
    a.legend()

    fig.tight_layout()
    out = ROOT / "writeup" / "figures" / "congestion_link.pdf"
    fig.savefig(out)
    print(f"saved {out}")


def main():
    rows = []
    lmp, cong, energy, loss = load()
    dP, dC = spread_space(lmp, cong, energy, loss, rows)
    print()
    C_toy, _ = ptdf_reference_field()
    disp_real, disp_toy, null = geometry(cong, C_toy, rows)
    print()
    R, jump, _, hl_real, hl_comp, hl_sim = sde_consistency(lmp, cong, energy, rows)
    print()
    print("Marginal validation criterion, projected onto level and spread coordinates:")
    marginal_localization(rows)
    print()
    figure(dP, dC, disp_real, disp_toy, null, R, jump, hl_real, hl_comp, hl_sim)
    out = ROOT / "grid" / "congestion_link_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
