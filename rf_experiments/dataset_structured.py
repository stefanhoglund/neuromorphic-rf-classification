# rf_experiments/dataset_structured.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SNR_VALUES = np.arange(-20, 32, 2, dtype=int)  # [-20, -18, ..., 30]


class RadioML2018StructuredHDF5(Dataset):
    """
    HDF5-backed subset of RadioML 2018 using known (mod, SNR, frame) layout.

    X is expected to be shape (N, 1024, 2) with N = 24*26*4096.
    """

    def __init__(
        self,
        h5_path: str,
        encoder,
        frames_per_combo: int | None = 256,
        mods_to_use=None,
        snrs_to_use=None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.encoder = encoder

        self.num_mods = 24
        self.num_snrs = 26
        self.frames_per_original_combo = 4096

        # which mods?
        if mods_to_use is None:
            mods = np.arange(self.num_mods, dtype=int)
        else:
            mods = np.array(mods_to_use, dtype=int)

        self.mods = mods
        # map original mod index -> contiguous [0..K-1] class index
        self.class_map = {int(m): i for i, m in enumerate(self.mods)}
        self.num_classes = len(self.mods)

        # which SNRs? (convert dB values to index 0..25)
        if snrs_to_use is None:
            snr_idxs = np.arange(self.num_snrs, dtype=int)
        else:
            snr_idxs = np.array(
                [np.where(SNR_VALUES == s)[0][0] for s in snrs_to_use],
                dtype=int,
            )

        if frames_per_combo is None or frames_per_combo > self.frames_per_original_combo:
            fpc = self.frames_per_original_combo
        else:
            fpc = int(frames_per_combo)

        # Precompute mapping: dataset index -> global frame index
        indices = []
        mod_list = []
        snr_list = []

        for m in mods:
            for si in snr_idxs:
                base = (m * self.num_snrs + si) * self.frames_per_original_combo
                frames = base + np.arange(fpc, dtype=int)  # contiguous slice
                indices.append(frames)
                mod_list.append(np.full_like(frames, m, dtype=int))
                snr_list.append(np.full_like(frames, si, dtype=int))

        self.global_indices = np.concatenate(indices)   # [N_sel]
        self.mod_indices = np.concatenate(mod_list)     # [N_sel] original mod ids
        self.snr_indices = np.concatenate(snr_list)     # [N_sel] snr index 0..25
        self.snr_values = SNR_VALUES[self.snr_indices]  # [N_sel] actual dB values
        self.N = self.global_indices.shape[0]

        print(
            f"[dataset] HDF5={h5_path}, frames_per_combo={fpc}, "
            f"mods={mods}, snrs={SNR_VALUES[snr_idxs].tolist()}, total_samples={self.N}"
        )

    def __len__(self):
        return self.N

    def _get_iq(self, global_i: int) -> np.ndarray:
        # single read from X; we don't keep file open to avoid pickling issues
        with h5py.File(self.h5_path, "r") as f:
            x = np.array(f["X"][global_i])  # (1024, 2) or (2, 1024)
        if x.shape == (2, 1024):
            x = x.T
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        gi = int(self.global_indices[idx])
        mod_idx = int(self.mod_indices[idx])
        snr_db = float(self.snr_values[idx])

        x_iq = self._get_iq(gi)  # (T, 2)

        # remap original modulation index -> [0..num_classes-1]
        label = self.class_map[mod_idx]

        spikes = self.encoder.encode(x_iq)  # torch.Tensor

        return spikes, label, snr_db

    # ------------------------------------------------------------------
    # New: simple distribution inspection for logging / sanity checks
    # ------------------------------------------------------------------
    def inspect_distribution(self):
        """
        Return class and SNR distributions for the *logical* dataset
        after mods_to_use, snrs_to_use, and frames_per_combo filtering.

        Returns dict with:
          - classes: np.ndarray of class indices [0..num_classes-1]
          - class_counts: np.ndarray of counts, same order as `classes`
          - snrs: np.ndarray of SNR values (dB)
          - snr_counts: np.ndarray of counts, same order as `snrs`
        """

        # class_map is bijective on self.mods, so we can aggregate via original mod ids
        orig_mods, class_counts = np.unique(self.mod_indices, return_counts=True)
        classes = np.array(
            [self.class_map[int(m)] for m in orig_mods],
            dtype=int,
        )

        snrs, snr_counts = np.unique(self.snr_values, return_counts=True)

        return {
            "classes": classes,
            "class_counts": class_counts,
            "snrs": snrs,
            "snr_counts": snr_counts,
        }
