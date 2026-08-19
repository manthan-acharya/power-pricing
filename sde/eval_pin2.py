# Run: OMP_NUM_THREADS=1 python3 -u -m sde.eval_pin2 <checkpoint.pt> <label>

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sde.eval_pin as ep
from sde.model import Scaler
from sde.train_pin2 import RescaledSDE


def load_rescaled(path):
    ck = torch.load(path, weights_only=False)
    model = RescaledSDE(np.asarray(ck["kappa"], dtype=np.float32))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Scaler(ck["scaler"]["mean"], ck["scaler"]["std"]), ck.get("stage", "?")


if __name__ == "__main__":
    ep.load_pin = load_rescaled
    ep.main()
