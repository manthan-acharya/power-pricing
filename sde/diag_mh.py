# Run: OMP_NUM_THREADS=1 python3 -u -m sde.diag_mh <ckpt> [<ckpt> ...]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from sde.model import full_series, load_frames, simulate
from sde.train import sample_windows, split_slice
from sde.train_mh import LAGS_DES, deseason, factors, implied_rates
from sde.validate import load_model

SPAN = 336


# Simulates SPAN hours from each window start in the standardized coordinates
# the moment vector is computed in.
def sim_windows(model, x, E, v, off, starts, n_paths=8):
    out = []
    for k, i0 in enumerate(starts):
        rep = lambda a: torch.tensor(a[i0:i0 + 1]).repeat(n_paths, 1)
        np.random.seed(31337 + k)
        xs = simulate(model, rep(x), rep(off), rep(E), rep(v),
                      n_hours=SPAN, seed=31337 + k)
        out.append(xs.numpy().transpose(1, 0, 2))
    return torch.tensor(np.concatenate(out))


def rate_block(xw):
    f = factors(deseason(xw))
    return torch.stack([implied_rates(fi, LAGS_DES) for fi in f])


def main():
    lmp, _, splits = load_frames(ROOT)
    _, scaler, _ = load_model(ROOT / "sde" / "checkpoint.pt")
    index, x, E, v, off = full_series(ROOT, scaler)
    tr = split_slice(index, "train")
    xtr, otr, Etr, vtr = (torch.tensor(a[tr]) for a in (x, off, E, v))

    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        real = torch.stack([rate_block(sample_windows(
            xtr, otr, Etr, vtr, SPAN, 256, gen)[0]) for _ in range(8)]).mean(0)

    lab = ["level", "spread1", "spread2"]
    print(f"lags {LAGS_DES}")
    print("REAL train windows (implied rate /h, then half-life h):")
    for i in range(3):
        print(f"  {lab[i]:8s} {np.round(real[i].numpy(), 4)}"
              f"   hl {np.round(np.log(2) / real[i].numpy(), 2)}")

    starts = list(np.arange(0, xtr.shape[0] - SPAN - 1, 900))
    rows = []
    for ck in sys.argv[1:]:
        model, _, _ = load_model(ck)
        with torch.no_grad():
            xw = sim_windows(model, x[tr], E[tr], v[tr], off[tr], starts)
            sim = rate_block(xw)
        name = Path(ck).stem
        print(f"\n{name} SIMULATED:")
        for i in range(3):
            hl = np.log(2) / sim[i].numpy()
            print(f"  {lab[i]:8s} {np.round(sim[i].numpy(), 4)}"
                  f"   hl {np.round(hl, 2)}"
                  f"   rate ratio sim/real {np.round((sim[i] / real[i]).numpy(), 2)}")
            rows.append({"checkpoint": name, "factor": lab[i],
                         **{f"rate_lag{L}": float(sim[i][j])
                            for j, L in enumerate(LAGS_DES)},
                         **{f"hl_lag{L}": float(hl[j])
                            for j, L in enumerate(LAGS_DES)}})
        r = sim[0] / real[0]
        print(f"  level rate ratio mean {r.mean():.2f} (1.0 = matched, "
              f">1 = still too fast)")
    for i in range(3):
        rows.append({"checkpoint": "REAL", "factor": lab[i],
                     **{f"rate_lag{L}": float(real[i][j])
                        for j, L in enumerate(LAGS_DES)},
                     **{f"hl_lag{L}": float(np.log(2) / real[i][j])
                        for j, L in enumerate(LAGS_DES)}})
    pd.DataFrame(rows).round(4).to_csv(ROOT / "sde" / "mh_rate_diagnostic.csv",
                                       index=False)
    print(f"\nwrote {ROOT / 'sde' / 'mh_rate_diagnostic.csv'}")


if __name__ == "__main__":
    main()
