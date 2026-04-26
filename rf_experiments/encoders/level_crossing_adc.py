import numpy as np
import torch


class LevelCrossingADCEncoder:
    """
    Level-crossing ADC encoder for complex I/Q RF samples.

    Input:
      x_iq: np.ndarray, shape (T, 2), values in [-1, 1] for I and Q.

    Idea:
      - After optional subsampling, track the *absolute* amplitude of each component.
      - Quantize amplitude into n_levels bins.
      - Emit a spike whenever the quantized bin index changes (i.e., crosses any level).
      - If use_sign=True, we split by sign into [I+, I-, Q+, Q-].
        Otherwise we ignore sign and use [I, Q].

    Output:
      - If grid_size is None:
          use_sign=True:  shape (T, 4)  -> channels [I+, I-, Q+, Q-]
          use_sign=False: shape (T, 2)  -> channels [I, Q]
      - If grid_size is not None:
          shape (T, 1, H, W) with H=W=grid_size, channels mapped to regions:
            C=4:  I+ top-left, I- top-right, Q+ bottom-left, Q- bottom-right
            C=2:  I top half, Q bottom half
    """

    def __init__(
        self,
        n_levels: int = 8,
        subsample: int = 1,
        use_sign: bool = True,
        grid_size: int | None = None,
    ):
        if n_levels <= 0:
            raise ValueError("n_levels must be > 0.")
        self.n_levels = int(n_levels)
        self.subsample = max(1, int(subsample))
        self.use_sign = bool(use_sign)

        if grid_size is not None and grid_size <= 0:
            raise ValueError("grid_size must be positive or None.")
        self.grid_size = grid_size

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _amp_bin(self, x: float) -> int:
        """Quantize |x| in [-1, 1] into an integer bin [0, n_levels-1]."""
        a = abs(float(x))
        a = max(0.0, min(1.0, a))
        # scale to [0, n_levels)
        b = int(a * self.n_levels)
        if b == self.n_levels:
            b = self.n_levels - 1
        return b

    def _to_grid(self, spikes_tc: np.ndarray) -> torch.Tensor:
        """
        Map (T, C) spikes to (T, 1, H, W) by assigning channels to spatial regions.

        For C=4:
          ch 0 (I+): top-left
          ch 1 (I-): top-right
          ch 2 (Q+): bottom-left
          ch 3 (Q-): bottom-right

        For C=2:
          ch 0 (I): top half
          ch 1 (Q): bottom half
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
                # Fallback: if any spike, fill whole map
                if s.any():
                    maps[t, 0, :, :] = 1.0

        return torch.from_numpy(maps).float()

    # ------------------------------------------------------------------ #
    # Main encode
    # ------------------------------------------------------------------ #
    def encode(self, x_iq: np.ndarray) -> torch.Tensor:
        """
        x_iq: np.ndarray, shape (T, 2)
        Returns:
          if grid_size is None: (T', C)
          else: (T', 1, H, W)
        """
        if x_iq.ndim != 2 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq shape (T, 2), got {x_iq.shape}")

        # Time subsampling
        x = x_iq[::self.subsample]
        T_prime = x.shape[0]
        if T_prime <= 0:
            raise ValueError("Empty signal after subsampling")

        I = x[:, 0]
        Q = x[:, 1]

        if self.use_sign:
            C = 4  # I+, I-, Q+, Q-
        else:
            C = 2  # I, Q

        spikes = np.zeros((T_prime, C), dtype=np.float32)

        prev_I = 0.0
        prev_Q = 0.0
        prev_bin_I = self._amp_bin(prev_I)
        prev_bin_Q = self._amp_bin(prev_Q)
        prev_sign_I = 0
        prev_sign_Q = 0

        for t in range(T_prime):
            It = float(I[t])
            Qt = float(Q[t])

            bin_I = self._amp_bin(It)
            bin_Q = self._amp_bin(Qt)

            sign_I = 1 if It >= 0.0 else -1
            sign_Q = 1 if Qt >= 0.0 else -1

            # Crossing event if quantized amplitude changed or sign changed
            crossing_I = (bin_I != prev_bin_I) or (sign_I != prev_sign_I)
            crossing_Q = (bin_Q != prev_bin_Q) or (sign_Q != prev_sign_Q)

            if self.use_sign:
                # I channels
                if crossing_I:
                    if It >= 0.0:
                        spikes[t, 0] = 1.0  # I+
                    else:
                        spikes[t, 1] = 1.0  # I-
                # Q channels
                if crossing_Q:
                    if Qt >= 0.0:
                        spikes[t, 2] = 1.0  # Q+
                    else:
                        spikes[t, 3] = 1.0  # Q-
            else:
                if crossing_I:
                    spikes[t, 0] = 1.0   # I
                if crossing_Q:
                    spikes[t, 1] = 1.0   # Q

            prev_I = It
            prev_Q = Qt
            prev_bin_I = bin_I
            prev_bin_Q = bin_Q
            prev_sign_I = sign_I
            prev_sign_Q = sign_Q

        # No grid requested: return (T, C)
        if self.grid_size is None:
            return torch.from_numpy(spikes).float()

        # Grid requested: (T, 1, H, W)
        return self._to_grid(spikes)
