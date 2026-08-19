# Run: OMP_NUM_THREADS=1 python3 -m sde.train_pin2 <seed>

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sde.train_pin as tp
from sde.model_pin import PinnedSDE

# per-hub rate the incumbent settled on, which set its noise budget
KAPPA_REF = 0.16


# diffusion and jump sizes scaled by sqrt(kappa/kappa_ref) so stationary
# variance is preserved when rates are pinned slower
class RescaledSDE(PinnedSDE):

    def __init__(self, kappa, **kw):
        super().__init__(kappa, **kw)
        s = np.sqrt(np.asarray(kappa, dtype=np.float32) / KAPPA_REF)
        self.register_buffer("noise_scale", torch.as_tensor(s, dtype=torch.float32))
        with torch.no_grad():
            self.jump_lsig.add_(float(np.log(s.mean())))
            self.jump_mean.mul_(float(s.mean()))

    def sigma_z(self, x, t, off0, E, v):
        return self.noise_scale * super().sigma_z(x, t, off0, E, v)


def main(seed):
    torch.set_num_threads(1)
    orig = tp.PinnedSDE
    tp.PinnedSDE = RescaledSDE
    try:
        tp.main("train", seed)
    finally:
        tp.PinnedSDE = orig


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 101)
