# rf_experiments/encoders/tf_events.py

import numpy as np
import torch


class TFEventsEncoder:
    """
    Time-frequency event-based encoder for IQ signals.

    Pipeline (per frame):
      1. Optional temporal subsampling on IQ samples.
      2. Build complex baseband z = I + jQ.
      3. Compute short-time Fourier transform (STFT) with sliding window.
      4. Take magnitude |STFT| for positive frequencies.
      5. Downsample frequency axis to `grid_size` bins.
      6. Log-normalize magnitudes to [0, 1].
      7. For each time step and freq bin, place a spike at
         (freq_bin, amplitude_bin) in a GxG grid.

    Input:
      x_iq: np.ndarray with shape (T, 2) and values approx in [-1, 1]

    Output:
      spikes: torch.Tensor with shape (T_frames, 1, G, G)
        T_frames ~ number of STFT frames
        G = grid_size
    """

    def __init__(
        self,
        grid_size: int = 16,
        subsample: int = 1,
        n_fft: int = 64,
        hop_length: int = 32,
        use_log: bool = True,
    ):
        """
        Parameters
        ----------
        grid_size : int
            Number of bins for both frequency and magnitude axes.
        subsample : int
            Temporal decimation factor on raw IQ samples (keep every k-th).
        n_fft : int
            FFT window length for STFT.
        hop_length : int
            Hop size between STFT frames.
        use_log : bool
            If True, apply log1p before normalization.
        """
        self.grid_size = int(grid_size)
        self.subsample = int(subsample)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.use_log = bool(use_log)

    # ------------------------------------------------------------------

    def _compute_stft_mag(self, z: np.ndarray) -> np.ndarray:
        """
        Compute magnitude STFT for complex baseband signal z.

        z: (T,) complex64
        Returns: np.ndarray of shape (T_frames, F_bins),
                 where F_bins = n_fft//2 (positive freqs)
        """
        T = z.shape[0]
        n_fft = self.n_fft
        hop = self.hop_length

        if T < n_fft:
            # pad to at least one frame
            pad = n_fft - T
            z = np.pad(z, (0, pad), mode="constant")
            T = z.shape[0]

        frames = []
        for start in range(0, T - n_fft + 1, hop):
            seg = z[start : start + n_fft]
            # Hann window
            window = np.hanning(n_fft).astype(np.float32)
            seg_win = seg * window
            spec = np.fft.fft(seg_win, n_fft)
            mag = np.abs(spec[: n_fft // 2])  # positive freqs
            frames.append(mag.astype(np.float32))

        if not frames:
            # fallback: single zero frame
            return np.zeros((1, n_fft // 2), dtype=np.float32)

        return np.stack(frames, axis=0)  # (T_frames, F_bins)

    def _downsample_freq(self, mag: np.ndarray) -> np.ndarray:
        """
        Downsample frequency axis to grid_size using index sampling.

        mag: (T_frames, F_bins)
        Returns: (T_frames, grid_size)
        """
        T_frames, F_bins = mag.shape
        G = self.grid_size

        if F_bins == 0:
            return np.zeros((T_frames, G), dtype=np.float32)

        # pick G frequency indices across [0, F_bins-1]
        idx = np.linspace(0, F_bins - 1, G).astype(int)
        return mag[:, idx]  # (T_frames, G)

    # ------------------------------------------------------------------

    def encode(self, x_iq: np.ndarray) -> torch.Tensor:
        """
        Encode a single IQ frame to spikes.

        x_iq: np.ndarray, shape (T, 2), columns = [I, Q].
        Returns: torch.Tensor, shape (T_frames, 1, G, G).
        """
        # subsample raw IQ
        if self.subsample > 1:
            x_iq = x_iq[::self.subsample]

        if x_iq.ndim != 2 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected x_iq shape (T, 2), got {x_iq.shape}")

        I = x_iq[:, 0].astype(np.float32)
        Q = x_iq[:, 1].astype(np.float32)
        z = I + 1j * Q  # complex baseband

        # 1) STFT magnitude
        mag = self._compute_stft_mag(z)  # (T_frames, F_bins)

        # 2) downsample frequency to grid_size
        mag = self._downsample_freq(mag)  # (T_frames, G)

        # 3) log + normalize to [0, 1]
        if self.use_log:
            mag = np.log1p(mag)
        max_val = mag.max()
        if max_val > 0:
            mag = mag / max_val
        else:
            mag = np.zeros_like(mag, dtype=np.float32)

        T_frames, G = mag.shape
        assert G == self.grid_size

        # 4) map to spikes on (freq_bin, amplitude_bin) in GxG grid
        spikes = np.zeros((T_frames, 1, G, G), dtype=np.float32)

        for t in range(T_frames):
            for f in range(G):
                amp = mag[t, f]
                if amp <= 0.0:
                    continue
                j = int(amp * (G - 1))  # amplitude bin index
                spikes[t, 0, f, j] = 1.0

        return torch.from_numpy(spikes)
