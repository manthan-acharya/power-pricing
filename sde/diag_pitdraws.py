# Run: OMP_NUM_THREADS=1 python3 -u -m sde.diag_pitdraws <ckpt> [<ckpt> ...]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from plotstyle import NODE_LABELS
from sde.calibration import autocorr_block, block_ci, ranks_of
from sde.eval_mh import simulate_windows
from sde.model import full_series, load_frames
from sde.validate import PATHS_PER_WINDOW, WINDOW_DAYS, load_model, window_starts

DRAWS = [700, 1300, 2100, 2900, 3700]


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
        reals = np.stack([lmp.values[i0 + 1:i0 + 1 + span] for i0 in starts])
        name = Path(ck).stem
        print(f"\n=== {name} ===", flush=True)
        for d in DRAWS:
            sims = np.stack(simulate_windows(model, scaler, x, E, v, off,
                                             starts, span, d))
            rng = np.random.default_rng(4)
            n_bins = PATHS_PER_WINDOW + 1
            msg, allok = [], True
            for j, node in enumerate(nodes):
                r = np.concatenate([ranks_of(reals[w, :, j], sims[w, :, :, j], rng)
                                    for w in range(len(starts))])
                pit = r / (n_bins - 1)
                lo, hi = block_ci(pit, autocorr_block(r.astype(float)), rng)[:2]
                ok = lo <= 0.5 <= hi
                allok &= ok
                msg.append(f"{NODE_LABELS[node]} {pit.mean():.3f}"
                           f"[{lo:.3f},{hi:.3f}]{'ok' if ok else 'REJ'}")
                rows.append({"checkpoint": name, "draw": d,
                             "node": NODE_LABELS[node], "mean_PIT": pit.mean(),
                             "ci_lo": lo, "ci_hi": hi, "covers_half": ok})
            print(f"  draw {d}: " + "  ".join(msg)
                  + f"   -> PRIMARY {'PASS' if allok else 'FAIL'}", flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(ROOT / "sde" / "mh_pit_draws.csv", index=False)
    print("\nfraction of draws where all three hubs cover 0.5:")
    for name, g in df.groupby("checkpoint", sort=False):
        per = g.groupby("draw")["covers_half"].all()
        print(f"  {name}: {per.sum()}/{len(per)}")
    print(f"wrote {ROOT / 'sde' / 'mh_pit_draws.csv'}")


if __name__ == "__main__":
    main()
