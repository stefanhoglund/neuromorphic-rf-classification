import numpy as np
import torch


class IQGridEncoder:
    def __init__(self, grid_size=16, subsample=4):
        self.grid_size = grid_size
        self.subsample = subsample

    def encode(self, x_iq):
        # x_iq: (1024, 2)
        x_iq = x_iq[::self.subsample]
        T = x_iq.shape[0]
        G = self.grid_size

        I = x_iq[:, 0]
        Q = x_iq[:, 1]

        I_norm = (I + 1.0) * 0.5 * (G - 1)
        Q_norm = (Q + 1.0) * 0.5 * (G - 1)

        i_idx = np.clip(I_norm.astype(int), 0, G - 1)
        j_idx = np.clip(Q_norm.astype(int), 0, G - 1)

        spikes = np.zeros((T, 1, G, G), dtype=np.float32)
        for t in range(T):
            spikes[t, 0, i_idx[t], j_idx[t]] = 1.0

        return torch.from_numpy(spikes)  # [T, C=1, H, W]
