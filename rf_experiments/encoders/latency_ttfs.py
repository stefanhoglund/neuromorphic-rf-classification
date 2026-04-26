# rf_experiments/encoders/latency_ttfs.py

import numpy as np
import torch


class LatencyTTFSEncoder:
    """
    Latency-based time-to-first-spike (TTFS) encoder for complex I/Q RF samples.

    Input:
      x_iq: np.ndarray, shape (T, 2), values in [-1, 1] for I and Q.

    Logic:
      - Optionally subsample the time axis.
      - Compute a single amplitude summary per (I+, I-, Q+, Q-) or (I, Q) channel.
      - Map amplitude a ∈ [0, 1] to a spike time index:
            t_spike = round((1 - a) * (T' - 1))
        so a=1 → t=0 (earliest), a=0 → t=T'-1 (latest).
      - Emit exactly one spike per channel.

    Output:
      - If grid_size is None:
          * use_sign=True:  (T', 4)  with channels [I+, I-, Q+, Q-]
          * use_sign=False: (T', 2)  with channels [I, Q]
      - If grid_size is not None:
          * (T', 1, H, W) where H=W=grid_size
          * channels are encoded spatially in quadrants / halves.
    """

    def __init__(
        self,
        subsample: int = 1,
        use_sign: bool = True,
        pool: str = "max",      # "max" or "rms"
        grid_size: int | None = None,
    ):
        self.subsample = max(1, int(subsample))
        self.use_sign = bool(use_sign)
        if pool not in ("max", "rms"):
            raise ValueError(f"Unknown pool mode: {pool}")
        self.pool = pool

        if grid_size is not None and grid_size <= 0:
            raise ValueError("grid_size must be positive or None.")
        self.grid_size = grid_size

    def _summary(self, sig: np.ndarray) -> float:
        """Compute amplitude summary in [0, 1]."""
        sig = np.asarray(sig, dtype=np.float32)
        sig = np.clip(sig, -1.0, 1.0)
        if self.pool == "max":
            a = float(np.max(np.abs(sig)))
        else:  # "rms"
            a = float(np.sqrt(np.mean(sig ** 2)))
        # ensure within [0, 1]
        return float(np.clip(a, 0.0, 1.0))

    def _to_grid(self, spikes_tc: np.ndarray) -> torch.Tensor:
        """
        Map (T, C) spikes to (T, 1, H, W) by assigning channels to spatial regions.

        For C=4 and grid_size=H=W:
          ch 0: top-left quadrant
          ch 1: top-right quadrant
          ch 2: bottom-left quadrant
          ch 3: bottom-right quadrant

        For C=2:
          ch 0: top half
          ch 1: bottom half
        """
        T, C = spikes_tc.shape
        H = W = self.grid_size
        if H is None:
            raise ValueError("grid_size must not be None to call _to_grid.")
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("grid_size must be even to split into quadrants/halves.")

        maps = np.zeros((T, 1, H, W), dtype=np.float32)
        h2 = H // 2
        w2 = W // 2

        for t in range(T):
            s = spikes_tc[t]

            if C == 4:
                # I+, I-, Q+, Q-
                if s[0] == 1.0:
                    maps[t, 0, 0:h2, 0:w2] = 1.0           # top-left
                if s[1] == 1.0:
                    maps[t, 0, 0:h2, w2:W] = 1.0           # top-right
                if s[2] == 1.0:
                    maps[t, 0, h2:H, 0:w2] = 1.0           # bottom-left
                if s[3] == 1.0:
                    maps[t, 0, h2:H, w2:W] = 1.0           # bottom-right

            elif C == 2:
                # I, Q
                if s[0] == 1.0:
                    maps[t, 0, 0:h2, :] = 1.0              # top half
                if s[1] == 1.0:
                    maps[t, 0, h2:H, :] = 1.0              # bottom half

            else:
                # fallback: if any spike, fill whole map
                if s.any():
                    maps[t, 0, :, :] = 1.0

        return torch.from_numpy(maps).float()

    def encode(self, x_iq: np.ndarray) -> torch.Tensor:
        """
        x_iq: np.ndarray, shape (T, 2)
        Returns:
          if grid_size is None: (T', C)
          else: (T', 1, H, W)
        """
        if x_iq.ndim != 2 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq shape (T, 2), got {x_iq.shape}")

        # Subsample time
        x = x_iq[::self.subsample]
        T_prime = x.shape[0]

        if T_prime <= 0:
            raise ValueError("Empty signal after subsampling")

        I = x[:, 0]
        Q = x[:, 1]

        if self.use_sign:
            # Positive and negative parts separately
            I_pos = np.maximum(I, 0.0)
            I_neg = -np.minimum(I, 0.0)
            Q_pos = np.maximum(Q, 0.0)
            Q_neg = -np.minimum(Q, 0.0)

            amps = [
                self._summary(I_pos),  # I+
                self._summary(I_neg),  # I-
                self._summary(Q_pos),  # Q+
                self._summary(Q_neg),  # Q-
            ]
            C = 4
        else:
            amps = [
                self._summary(I),
                self._summary(Q),
            ]
            C = 2

        # TTFS spike matrix (T, C)
        spikes = np.zeros((T_prime, C), dtype=np.float32)

        for ch, a in enumerate(amps):
            a_clamped = float(np.clip(a, 0.0, 1.0))
            # a=1 => t=0; a=0 => t=T'-1
            t_spike = int(round((1.0 - a_clamped) * (T_prime - 1)))
            spikes[t_spike, ch] = 1.0

        # If no grid requested, return (T, C)
        if self.grid_size is None:
            return torch.from_numpy(spikes).float()

        # Otherwise, convert to (T, 1, H, W)
        return self._to_grid(spikes)
