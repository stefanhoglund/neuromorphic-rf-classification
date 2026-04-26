# check_labels.py
import h5py
import numpy as np

h5_path = "data/GOLD_XYZ_OSC.0001_1024.hdf5"

with h5py.File(h5_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    X = f["X"]
    Y = f["Y"]
    Z = f["Z"]

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Z shape:", Z.shape)

    Y_arr = Y[:]  # (N, 24)
    # sanity check: is it one-hot?
    row_sums = Y_arr.sum(axis=1)
    print("Row-sum min/max in Y:", row_sums.min(), row_sums.max())

    # modulation index per sample (0..23)
    mod_idx = np.argmax(Y_arr, axis=1)
    uniq_mod, counts_mod = np.unique(mod_idx, return_counts=True)
    print("Unique modulation ids:", uniq_mod)
    print("Counts per modulation id:", counts_mod)

    # SNR values
    Z_arr = Z[:]  # shape should be (N,) or similar
    print("Z dtype:", Z_arr.dtype, "Z shape:", Z_arr.shape)
    uniq_snr = np.unique(Z_arr)
    print("Unique SNR values:", uniq_snr)
