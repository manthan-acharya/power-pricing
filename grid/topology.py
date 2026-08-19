import warnings

import numpy as np

GEN_COST = {0: 15.0, 1: 30.0, 2: 40.0, 5: 45.0, 7: 50.0}

# Branch 0 (bus 1 -> bus 2): unconstrained flow is 181.4 MW at base load and
# 211.2 MW under the +40 MW perturbation, so a 190 MW limit is slack in the
# base case and binds under the perturbation.
LINE_LIMITS = {0: 190.0}


def build(extra_load=None):
    warnings.filterwarnings("ignore")
    import pandapower.networks as pn
    from pandapower.converter.pypower import to_ppc

    ppc = to_ppc(pn.case14(), init="flat")
    bus = np.real(ppc["bus"])
    branch = np.real(ppc["branch"])
    gen = np.real(ppc["gen"])

    n = bus.shape[0]
    d = bus[:, 2].copy()
    if extra_load:
        for b, mw in extra_load.items():
            d[b] += mw

    # off-nominal taps scale the reactance: f = (th_f - th_t) / (x * tap)
    tap = branch[:, 8].copy()
    tap[tap == 0] = 1.0

    case = {
        "n": n,
        "baseMVA": ppc["baseMVA"],
        "from": branch[:, 0].astype(int),
        "to": branch[:, 1].astype(int),
        "x": branch[:, 3] * tap,
        "slack": 0,
        "gen_bus": gen[:, 0].astype(int),
        "pmax": gen[:, 8],
        "cost": np.array([GEN_COST[int(b)] for b in gen[:, 0]]),
        "load": d,
        "limits": dict(LINE_LIMITS),
    }
    return case
