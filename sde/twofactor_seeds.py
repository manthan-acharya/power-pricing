# Run: OMP_NUM_THREADS=1 python3 -m sde.twofactor_seeds

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sde.train as base
import sde.train_twofactor as tf
from sde.model import Scaler, full_series, load_frames
from sde.train import split_slice

SEEDS = [7, 11, 23, 41, 57]
OUT = ROOT / "sde" / "twofactor_seeds.csv"


def train_one(seed):
    tf.SEED = seed
    base.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, splits = load_frames(ROOT)
    scaler = Scaler(splits["train"][0].mean().values, splits["train"][0].std().values)
    index, x, E, v, off = full_series(ROOT, scaler)
    T = {n: tuple(torch.tensor(a[split_slice(index, n)])
                  for a in (x, off, E, v)) for n in ["train", "val"]}
    model = tf.TwoFactorSDE()
    nll_a, _ = tf.stage_a(model, T)
    spike_thr = torch.tensor(25.0 / scaler.std)
    neg_thr = torch.tensor(-scaler.mean / scaler.std)
    tf.stage_b(model, T, nll_a, spike_thr, neg_thr)
    with torch.no_grad():
        xva, ova, Eva, vva = T["val"]
        va = model.transition_nll(xva[:-1], xva[1:], ova[:-1], Eva[:-1], vva[:-1]).item()
    return model, scaler, va, x, E, v, off


def main():
    torch.set_num_threads(1)
    rows = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model, scaler, va, x, E, v, off = train_one(seed)
        hl = tf.factor_halflives(model, scaler, x, E, v, off)
        level, s1, s2 = hl[0], hl[1], hl[2]
        ratio = (s1 + s2) / 2 / level
        row = {"seed": seed, "val_nll": va, "halflife_level_h": level,
               "halflife_spread1_h": s1, "halflife_spread2_h": s2,
               "spread_over_level": ratio}
        rows.append(row)
        print(f"seed {seed}: val nll {va:.4f}  level {level:.2f}h "
              f"spreads {s1:.2f}/{s2:.2f}h  ratio {ratio:.3f}", flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n=== across seeds ===")
    print(f"val nll         mean {df.val_nll.mean():.4f}  sd {df.val_nll.std():.4f}")
    print(f"level half-life mean {df.halflife_level_h.mean():.2f}h  sd "
          f"{df.halflife_level_h.std():.2f}  min {df.halflife_level_h.min():.2f}  "
          f"max {df.halflife_level_h.max():.2f}   (observed 13.04h)")
    print(f"spread/level    mean {df.spread_over_level.mean():.3f}  sd "
          f"{df.spread_over_level.std():.3f}  min {df.spread_over_level.min():.3f}  "
          f"max {df.spread_over_level.max():.3f}  (observed 0.38, incumbent 0.80)")
    reaches = (df.halflife_level_h > 8).sum()
    print(f"\nseeds reaching a level half-life above 8h: {reaches}/{len(df)}")
    print("single-run negative:" +
          (" not robust, some seeds behave differently" if reaches else
           " reproduced across seeds"))


if __name__ == "__main__":
    main()
