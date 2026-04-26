import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RadioMLHDF5Dataset(Dataset):
    """
    Generic HDF5-backed dataset for RadioML-style data.

    Assumes file contains:
      - X: RF samples, either shape (N, T, 2) or (N, 2, T)
        where last or second dim is I/Q.
      - Y: one-hot labels, shape (N, num_classes)
      - Z: SNR values, shape (N, 1) or (N,)

    Parameters
    ----------
    h5_path : str
        Path to .h5 / .hdf5 file.
    encoder : object
        Must expose encode(x_iq: np.ndarray) -> torch.Tensor.
        x_iq will always have shape (T, 2), float32.
    snr_min : float or None
        If given, only keep examples with SNR >= snr_min.
    snr_max : float or None
        If given, only keep examples with SNR <= snr_max.
    limit_samples : int or None
        If given, truncate filtered index to this many samples.
    x_key, y_key, z_key : str
        Dataset keys inside the HDF5 file. Defaults to 'X', 'Y', 'Z'.
    """

    def __init__(
        self,
        h5_path: str,
        encoder,
        snr_min: float | None = None,
        snr_max: float | None = None,
        limit_samples: int | None = None,
        x_key: str = "X",
        y_key: str = "Y",
        z_key: str = "Z",
    ):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")

        if x_key not in self.h5 or y_key not in self.h5 or z_key not in self.h5:
            raise KeyError(
                f"Expected datasets '{x_key}', '{y_key}', '{z_key}' "
                f"in {h5_path}. Found: {list(self.h5.keys())}"
            )

        self.X = self.h5[x_key]  # HDF5 dataset (N, T, 2) or (N, 2, T)
        self.Y = self.h5[y_key]  # HDF5 dataset (N, num_classes)
        self.Z = self.h5[z_key][:]  # load SNR as np array

        self.encoder = encoder

        # Build filtered index based on SNR
        N = self.X.shape[0]
        idx = np.arange(N)

        # Normalize Z to shape (N,)
        z = self.Z
        if z.ndim > 1:
            z = z.reshape(-1)
        if z.shape[0] != N:
            raise ValueError(f"SNR array Z length {z.shape[0]} != X length {N}")

        if snr_min is not None:
            idx = idx[z[idx] >= snr_min]
        if snr_max is not None:
            idx = idx[z[idx] <= snr_max]

        if limit_samples is not None:
            idx = idx[: int(limit_samples)]

        self.idx = idx
        self.snr_values = z[self.idx].copy().astype(float)

        # Cache shape info for sanity
        self._x_shape = self.X.shape
        self._num_classes = self.Y.shape[1]

    def __len__(self) -> int:
        return len(self.idx)

    def _get_iq(self, i: int) -> np.ndarray:
        """
        Returns x_iq as np.ndarray, shape (T, 2), float32.
        Handles both (T, 2) and (2, T) layouts in the HDF5.
        """
        x = self.X[i]  # HDF5 returns np array view
        x = np.array(x)  # ensure real np.ndarray

        if x.ndim != 2:
            raise ValueError(f"Expected X[i] to be 2D, got shape {x.shape}")

        # Case 1: (T, 2)
        if x.shape[-1] == 2:
            x_iq = x.astype(np.float32)
        # Case 2: (2, T) -> transpose
        elif x.shape[0] == 2:
            x_iq = x.T.astype(np.float32)
        else:
            raise ValueError(
                f"Cannot infer IQ layout from shape {x.shape}; expected (T,2) or (2,T)"
            )

        return x_iq

    def __getitem__(self, k: int):
        i = int(self.idx[k])

        # x_iq: (T, 2)
        x_iq = self._get_iq(i)

        # y_onehot: (num_classes,)
        y_onehot = np.array(self.Y[i])
        if y_onehot.ndim != 1:
            y_onehot = y_onehot.reshape(-1)
        label = int(np.argmax(y_onehot))

        snr = float(self.snr_values[k])

        # encoder encodes IQ into spikes tensor
        spikes = self.encoder.encode(x_iq)  # torch.Tensor, typically [T, C,...]

        return spikes, label, snr

    def inspect_distribution(self):
        """
        Return basic stats about labels and SNRs *after* filtering.

        Returns:
            dict with:
              - classes: np.ndarray of class indices present
              - class_counts: np.ndarray of counts per class
              - snrs: np.ndarray of unique SNR values present
              - snr_counts: np.ndarray of counts per SNR
        """
        # Y is one-hot: shape (N, num_classes)
        y_sel = self.Y[self.idx]  # shape (len(idx), num_classes)
        if y_sel.ndim == 2:
            labels = np.argmax(y_sel, axis=1)
        else:
            labels = y_sel.reshape(len(self.idx), -1).argmax(axis=1)

        classes, class_counts = np.unique(labels, return_counts=True)
        snrs = self.snr_values
        snr_vals, snr_counts = np.unique(snrs, return_counts=True)

        return {
            "classes": classes,
            "class_counts": class_counts,
            "snrs": snr_vals,
            "snr_counts": snr_counts,
        }