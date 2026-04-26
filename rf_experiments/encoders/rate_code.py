import numpy as np
import torch


class RateCodeEncoder:
    """
    Rate-coded encoder for complex I/Q RF samples.

    Modes via `channels`:
      - channels = 1: amplitude-coded |I + jQ| into a single channel
      - channels = 2: [I, Q] amplitude-based channels
      - channels = 4: [I+, I-, Q+, Q-] sign-separated channels

    Output:
      spikes : torch.FloatTensor, shape (T', C, H, W)
    """

    def __init__(
        self,
        max_rate: float = 0.5,
        stochastic: bool = True,
        threshold: float | None = None,
        subsample: int = 1,
        use_sign: bool = True,
        grid_size: int = 16,
        channels: int = 1,
        rng_seed: int | None = None,
    ):
        self.max_rate = float(max_rate)
        self.stochastic = bool(stochastic)
        self.threshold = threshold
        self.subsample = max(1, int(subsample))
        self.use_sign = bool(use_sign)
        self.grid_size = int(grid_size)
        self.channels = int(channels)

        if rng_seed is not None:
            self._rng = np.random.RandomState(rng_seed)
        else:
            self._rng = None

        # Basic sanity
        if self.channels not in (1, 2, 4):
            raise ValueError(f"Unsupported channels={self.channels}, use 1, 2, or 4")

    def _rand(self):
        if self._rng is not None:
            return self._rng.rand()
        else:
            return np.random.rand()

    def encode(self, x_iq: np.ndarray) -> torch.Tensor:
        """
        x_iq : np.ndarray, shape (T, 2), values in [-1, 1] for I and Q

        Returns:
          spikes : torch.FloatTensor, shape (T', C, H, W)
        """
        # time subsampling
        x_iq = x_iq[::self.subsample]
        T = x_iq.shape[0]

        C = self.channels
        H = W = self.grid_size

        spikes = np.zeros((T, C, H, W), dtype=np.float32)
        thr = self.threshold if self.threshold is not None else 0.5

        for t in range(T):
            I = float(x_iq[t, 0])
            Q = float(x_iq[t, 1])

            ch = np.zeros((C,), dtype=np.float32)

            if C == 1:
                # ----- Single amplitude-coded channel -----
                # |I + jQ| in [0, sqrt(2)] → normalize to [0, 1]
                amp = np.sqrt(I * I + Q * Q)
                amp = min(amp / np.sqrt(2.0), 1.0)

                if self.stochastic:
                    p = self.max_rate * amp
                    if self._rand() < p:
                        ch[0] = 1.0
                else:
                    if amp >= thr:
                        ch[0] = 1.0

            elif C == 2:
                # ----- Two channels: [I, Q] amplitude-based -----
                amp_I = abs(I)
                amp_Q = abs(Q)

                if self.stochastic:
                    p_I = self.max_rate * amp_I
                    p_Q = self.max_rate * amp_Q
                    if self._rand() < p_I:
                        ch[0] = 1.0
                    if self._rand() < p_Q:
                        ch[1] = 1.0
                else:
                    if amp_I >= thr:
                        ch[0] = 1.0
                    if amp_Q >= thr:
                        ch[1] = 1.0

            elif C == 4:
                # ----- Four sign-separated channels: [I+, I-, Q+, Q-] -----
                amp_I = abs(I)
                amp_Q = abs(Q)

                # I
                if self.stochastic:
                    p_I = self.max_rate * amp_I
                    if self._rand() < p_I:
                        if I >= 0:
                            ch[0] = 1.0  # I+
                        else:
                            ch[1] = 1.0  # I-
                else:
                    if amp_I >= thr:
                        if I >= 0:
                            ch[0] = 1.0
                        else:
                            ch[1] = 1.0

                # Q
                if self.stochastic:
                    p_Q = self.max_rate * amp_Q
                    if self._rand() < p_Q:
                        if Q >= 0:
                            ch[2] = 1.0  # Q+
                        else:
                            ch[3] = 1.0  # Q-
                else:
                    if amp_Q >= thr:
                        if Q >= 0:
                            ch[2] = 1.0
                        else:
                            ch[3] = 1.0

            # Broadcast channel vector → spatial map
            spikes[t] = ch[:, None, None]

        return torch.from_numpy(spikes).float()
