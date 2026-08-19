import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grid.topology import build


# Injection shift factors from the reduced susceptance matrix (slack row/column
# dropped to fix the angle reference).
def ptdf(n, fbus, tbus, x, slack):
    m = len(fbus)
    A = np.zeros((m, n))
    A[np.arange(m), fbus] = 1.0
    A[np.arange(m), tbus] = -1.0
    Bd = np.diag(1.0 / x)
    B = A.T @ Bd @ A
    keep = [i for i in range(n) if i != slack]
    Binv = np.zeros((n, n))
    Binv[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
    return Bd @ A @ Binv


def solve(case):
    n, g = case["n"], len(case["gen_bus"])
    H = ptdf(n, case["from"], case["to"], case["x"], case["slack"])
    Mg = np.zeros((n, g))
    Mg[case["gen_bus"], np.arange(g)] = 1.0
    d = case["load"]

    lines = sorted(case["limits"])
    Hl = H[lines]
    fbar = np.array([case["limits"][l] for l in lines])
    # flow = H (Mg p - d); enforce -fbar <= flow <= fbar
    A_ub = np.vstack([Hl @ Mg, -Hl @ Mg])
    b_ub = np.concatenate([fbar + Hl @ d, fbar - Hl @ d])
    A_eq = np.ones((1, g))
    b_eq = [d.sum()]

    res = linprog(case["cost"], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, pm) for pm in case["pmax"]], method="highs")
    assert res.status == 0, res.message

    lam = res.eqlin.marginals[0]
    mu = -res.ineqlin.marginals
    mu_pos, mu_neg = mu[:len(lines)], mu[len(lines):]
    # LMP_k = lambda - sum_l (mu+_l - mu-_l) H_{l,k}
    cong = -(mu_pos - mu_neg) @ Hl
    inj = Mg @ res.x - d
    return {
        "p": res.x, "cost": res.fun, "lam": lam,
        "mu": dict(zip(lines, mu_pos - mu_neg)),
        "lmp": lam + cong, "congestion": cong,
        "flows": H @ inj, "H": H, "lines": lines,
    }


def check_ptdf_3bus():
    # Wood & Wollenberg 3-bus example (equal reactances, slack at bus 3):
    # injection at bus 1 splits 2/3 on the direct line 1-3, 1/3 via bus 2.
    H = ptdf(3, np.array([0, 1, 0]), np.array([1, 2, 2]),
             np.array([1.0, 1.0, 1.0]), slack=2)
    expect = np.array([1 / 3, 1 / 3, 2 / 3])
    err = np.abs(H[:, 0] - expect).max()
    print(f"3-bus textbook PTDF: got {H[:, 0].round(6)}, max err {err:.2e}")
    return err < 1e-12


def check_ptdf_case14(case):
    import pandapower.networks as pn
    from pandapower.converter.pypower import to_ppc
    from pandapower.pypower.makePTDF import makePTDF

    ppc = to_ppc(pn.case14(), init="flat")
    H_ref = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"],
                     slack=case["slack"])
    H = ptdf(case["n"], case["from"], case["to"], case["x"], case["slack"])
    err = np.abs(H - H_ref).max()
    print(f"case14 PTDF vs pypower makePTDF: max abs diff {err:.2e}")
    return err < 1e-8


def congestion_experiment():
    base = solve(build())
    pert = solve(build(extra_load={2: 40.0}))
    l0 = base["lines"][0]

    print(f"Base: lambda={base['lam']:.2f}, line {l0} flow={base['flows'][l0]:.1f} MW, mu={base['mu'][l0]:.3f}")
    print(f"Perturbed: lambda={pert['lam']:.2f}, line {l0} flow={pert['flows'][l0]:.1f} MW, mu={pert['mu'][l0]:.3f}")

    # with one binding line, LMP_k - lam must equal -mu * H_{l,k} exactly
    pred = -pert["mu"][l0] * pert["H"][l0]
    err = np.abs((pert["lmp"] - pert["lam"]) - pred).max()
    print(f"Congestion component vs PTDF prediction max abs err: {err:.2e}")

    corr_sign = np.sign(pert["H"][l0]) == np.sign(-(pert["lmp"] - pert["lam"]))
    nz = np.abs(pert["H"][l0]) > 1e-9
    print(f"Sign match: {corr_sign[nz].sum()}/{nz.sum()} buses")
    return base, pert, err < 1e-8


def figure(base, pert):
    import matplotlib.pyplot as plt
    from plotstyle import apply
    apply()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    buses = np.arange(base["lmp"].size)
    axes[0].plot(buses, base["lmp"], "o-", color="#0072B2", ms=4,
                 label="base load")
    axes[0].plot(buses, pert["lmp"], "s-", color="#D55E00", ms=4,
                 label="+40 MW at bus 3")
    axes[0].set_xlabel("bus")
    axes[0].set_ylabel("LMP ($/MWh)")
    axes[0].set_title("Nodal prices, line 1-2 limited to 190 MW")
    axes[0].legend()

    l0 = base["lines"][0]
    H = pert["H"][l0]
    dlmp = pert["lmp"] - pert["lam"]
    axes[1].scatter(-pert["mu"][l0] * H, dlmp, s=18, color="#009E73")
    lim = np.abs(dlmp).max() * 1.15
    axes[1].plot([-lim, lim], [-lim, lim], lw=0.8, color="#999999")
    axes[1].set_xlabel(r"$-\mu \cdot H_{l,k}$ predicted (\$/MWh)")
    axes[1].set_ylabel(r"LMP$_k - \lambda$ realized (\$/MWh)")
    axes[1].set_title("Congestion component vs PTDF prediction")
    fig.tight_layout()
    out = ROOT / "writeup" / "figures" / "congestion_experiment.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    ok3 = check_ptdf_3bus()
    ok14 = check_ptdf_case14(build())
    base, pert, okc = congestion_experiment()
    figure(base, pert)
    print("All checks passed." if (ok3 and ok14 and okc) else "CHECK FAILED.")

