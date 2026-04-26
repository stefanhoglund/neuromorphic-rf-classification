# rf_experiments/encoders/sigma_delta.py

import numpy as np
import torch


class SigmaDeltaEncoder:
    """
    Simple sigma-delta-like encoder for complex IQ samples.

    - Input:  x_iq: np.ndarray of shape (T, 2) with I/Q in approx [-1, 1].
    - Output: torch.Tensor of shape (T, 1, G, G), where
        T = time steps (after subsampling)
        G = grid_size

      We map I and Q to a GxG grid as in IQGridEncoder, but only emit
      a spike at time t if the change in I or Q vs t-1 exceeds a threshold.

    Parameters
    ----------
    grid_size : int
        Size of the IQ grid (G). Same semantics as IQGridEncoder.
    subsample : int
        Temporal decimation factor (keep every subsample-th sample).
    diff_threshold : float
        Threshold on |ΔI| or |ΔQ| to emit a spike. Smaller → more spikes.
    """

    def __init__(self, grid_size: int = 16, subsample: int = 1, diff_threshold: float = 0.02):
        self.grid_size = int(grid_size)
        self.subsample = int(subsample)
        self.diff_threshold = float(diff_threshold)

    def _iq_to_grid_indices(self, I: np.ndarray, Q: np.ndarray):
        """
        Normalize I/Q ∈ [-1, 1] to integer indices in [0, G-1].
        """
        G = self.grid_size

        I_norm = (I + 1.0) * 0.5 * (G - 1)
        Q_norm = (Q + 1.0) * 0.5 * (G - 1)

        i_idx = np.clip(I_norm.astype(int), 0, G - 1)
        j_idx = np.clip(Q_norm.astype(int), 0, G - 1)

        return i_idx, j_idx

    def encode(self, x_iq: np.ndarray) -> torch.Tensor:
        """
        Encode a single IQ frame to spikes.

        x_iq: (T, 2) array-like of floats (I, Q).

        Returns:
          spikes: torch.Tensor of shape (T, 1, G, G)
        """
        # subsample along time if requested
        if self.subsample > 1:
            x_iq = x_iq[::self.subsample]

        T = x_iq.shape[0]
        G = self.grid_size

        I = x_iq[:, 0].astype(np.float32)
        Q = x_iq[:, 1].astype(np.float32)

        # grid indices for each time step
        i_idx, j_idx = self._iq_to_grid_indices(I, Q)

        # allocate spikes
        spikes = np.zeros((T, 1, G, G), dtype=np.float32)

        # always emit a spike at t=0 to mark the start state
        spikes[0, 0, i_idx[0], j_idx[0]] = 1.0

        # sigma-delta-like: fire only when change exceeds threshold
        thr = self.diff_threshold
        for t in range(1, T):
            dI = I[t] - I[t - 1]
            dQ = Q[t] - Q[t - 1]
            if abs(dI) > thr or abs(dQ) > thr:
                spikes[t, 0, i_idx[t], j_idx[t]] = 1.0

        return torch.from_numpy(spikes)
