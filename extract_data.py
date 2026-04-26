#!/usr/bin/env python3
import h5py
import numpy as np
import os

SRC = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
DST = "data/GOLD_XYZ_OSC_subset_mods_1_2_3_4_SNR0_10_20_30.hdf5"

MODS_TO_KEEP = [1, 2, 3, 4]
SNRS_TO_KEEP = [0,10,20,30]  # or None for all SNRs
MAX_FRAMES = None  # e.g. 20000

# We used this file to extract smaller data subsets to reduce size/disk space requirements
def main():
    if not os.path.isfile(SRC):
        raise FileNotFoundError(SRC)

    with h5py.File(SRC, "r") as src:
        X = src["X"]      
        Y = src["Y"]   
        Z = src["Z"]      

        print("Loading labels (Y) and SNRs (Z)...")
        Y_arr = Y[...]
        Z_arr = Z[...]

        Z_arr = np.asarray(Z_arr).reshape(-1)

        print("Building modulation/SNR mask...")
        mod_idx = np.argmax(Y_arr, axis=1)  # (N,)

        mask = np.isin(mod_idx, MODS_TO_KEEP)
        if SNRS_TO_KEEP is not None:
            mask &= np.isin(Z_arr, SNRS_TO_KEEP)

        keep_idx = np.nonzero(mask)[0]
        print(f"Total matched frames before subsampling: {len(keep_idx)}")

        if MAX_FRAMES is not None and len(keep_idx) > MAX_FRAMES:
            print(f"Subsampling to {MAX_FRAMES} frames...")
            keep_idx = np.random.choice(keep_idx, size=MAX_FRAMES, replace=False)
            keep_idx = np.sort(keep_idx)

        print(f"Final number of frames to keep: {len(keep_idx)}")

        if len(keep_idx) == 0:
            raise RuntimeError("Mask is empty; nothing to write.")


        print(f"Writing subset to {DST} ...")
        with h5py.File(DST, "w") as dst:
          
            dst.create_dataset(
                "X",
                data=X[keep_idx, ...],
                compression="gzip",
                compression_opts=4,
            )
            dst.create_dataset(
                "Y",
                data=Y_arr[keep_idx, ...],
                compression="gzip",
                compression_opts=4,
            )
            dst.create_dataset(
                "Z",
                data=Z_arr[keep_idx, ...],
                compression="gzip",
                compression_opts=4,
            )

    print("Done.")


if __name__ == "__main__":
    main()
