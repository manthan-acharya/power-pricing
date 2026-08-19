# Run: OMP_NUM_THREADS=1 python3 -u -m sde.diag_bias <ckpt> [<ckpt> ...]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from plotstyle import NODE_LABELS
from sde.model import full_series, load_frames
from sde.validate import (PATHS_PER_WINDOW, WINDOW_DAYS, load_model,
                          run_window, window_starts)


def main():
    rows = []
    for ck in sys.argv[1:]:
        model, scaler, _ = load_model(ck)
        lmp, _, splits = load_frames(ROOT)
        index, x, E, v, off = full_series(ROOT, scaler)
        val = splits["val"][0]
        nodes = list(val.columns)
        span = WINDOW_DAYS * 24
        starts = window_starts(index, val.index[0], 4, span)

        sims, reals = [], []
        for k, i0 in enumerate(starts):
            np.random.seed(700 + k)
            sims.append(run_window(model, scaler, x, E, v, off, i0, span,
                                   PATHS_PER_WINDOW, seed=700 + k)[:, 1:])
            reals.append(lmp.values[i0 + 1:i0 + 1 + span])
        sim = np.stack(sims)                    # [W, paths, T, 3]
        real = np.stack(reals)                  # [W, T, 3]
        bias = sim.mean(1) - real               # [W, T, 3] ensemble mean error
        name = Path(ck).stem
        print(f"\n=== {name} ===")
        for j, node in enumerate(nodes):
            b = bias[:, :, j]
            day = b.reshape(len(starts), WINDOW_DAYS, 24).mean(axis=(0, 2))
            d0, d13 = float(day[0]), float(day[-1])
            print(f"  {NODE_LABELS[node]}: day0 {d0:+.2f}  day13 {d13:+.2f}"
                  f"  drift {d13 - d0:+.2f}  window mean {b.mean():+.2f} $/MWh")
            rows.append({"checkpoint": name, "node": NODE_LABELS[node],
                         "bias_day0_usd": d0, "bias_day13_usd": d13,
                         "drift_usd": d13 - d0, "bias_mean_usd": float(b.mean())})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(ROOT / "sde" / "mh_bias_profile.csv", index=False)
    print(f"\nwrote {ROOT / 'sde' / 'mh_bias_profile.csv'}")


if __name__ == "__main__":
    main()
