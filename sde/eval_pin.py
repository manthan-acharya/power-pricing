# Run: OMP_NUM_THREADS=1 python3 -u -m sde.eval_pin <checkpoint.pt> <label>

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

import sde.eval_mh as evm
from sde.model import Scaler, full_series
from sde.model_pin import PinnedSDE
from sde.train import split_slice

OUT = ROOT / "sde"


def load_pin(path):
    ck = torch.load(path, weights_only=False)
    model = PinnedSDE(ck["kappa"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Scaler(ck["scaler"]["mean"], ck["scaler"]["std"]), ck.get("stage", "?")


# Pinned K plus the residual's regression slope on the anchor deviation,
# measured over the train split: the rate the drift actually applies.
def effective_rates(model, scaler):
    index, x, E, v, off = full_series(ROOT, scaler)
    m = split_slice(index, "train")
    args = [torch.tensor(a[m]) for a in (x, off, E, v)]
    with torch.no_grad():
        b = model.residual_slope(args[0], torch.zeros(1), args[1], args[2], args[3])
    return model.kappa.numpy(), b.numpy()


def main():
    ckpt = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else ckpt.stem
    evm.load_model = load_pin
    summary, check_rows, cal_rows = evm.evaluate(ckpt, label)

    model, scaler, _ = load_pin(ckpt)
    kap, b = effective_rates(model, scaler)
    for i, n in enumerate(["level", "spread1", "spread2"]):
        summary[f"K_pin_{n}"] = kap[i]
        summary[f"K_eff_{n}"] = kap[i] + b[i]
    print(f"pinned K {np.round(kap, 4)}  effective K {np.round(kap + b, 4)}", flush=True)

    tag = label.replace(" ", "_").replace("/", "_")
    pd.DataFrame(check_rows).round(4).to_csv(OUT / f"eval_pin_checks_{tag}.csv", index=False)
    pd.DataFrame(cal_rows).round(4).to_csv(OUT / f"eval_pin_cal_{tag}.csv", index=False)
    pd.DataFrame([summary]).round(4).to_csv(OUT / f"eval_pin_row_{tag}.csv", index=False)
    # summary table is rebuilt from per-seed row files rather than
    # read-modify-written, since seeds are scored concurrently
    rows = [pd.read_csv(p) for p in sorted(OUT.glob("eval_pin_row_*.csv"))]
    path = OUT / "pin_comparison.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
