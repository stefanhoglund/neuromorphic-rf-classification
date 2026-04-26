from rf_experiments.analysis import collect_runs, runs_to_dataframe
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Load runs into DataFrame
# -------------------------------------------------------------------
runs = collect_runs("results")
df = runs_to_dataframe(runs)
print(df.head())
print("Columns:", df.columns.tolist())


# -------------------------------------------------------------------
# Plot 1: test accuracy by encoding/model/trainer
# -------------------------------------------------------------------
def plot_test_accuracy(
    df: pd.DataFrame,
    group_cols=("encoding_type", "model_type", "trainer"),
    sort_by: str = "test_acc",
):
    if df.empty:
        print("DataFrame is empty; nothing to plot.")
        return

    # one row per run; group by cols and take best test_acc if duplicates
    agg = (
        df.groupby(list(group_cols), as_index=False)
          .agg({"test_acc": "max"})
          .sort_values(by=sort_by, ascending=False)
    )

    labels = [
        f"{e}|{m}|{t}"
        for e, m, t in zip(
            agg["encoding_type"], agg["model_type"], agg["trainer"]
        )
    ]
    accs = agg["test_acc"].values

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(accs)), accs)
    plt.xticks(range(len(accs)), labels, rotation=45, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy by encoding/model/trainer")
    plt.tight_layout()   
    plt.show()              


# -------------------------------------------------------------------
# Plot 2: accuracy vs SNR
# -------------------------------------------------------------------
def plot_acc_vs_snr(
    df: pd.DataFrame,
    filter_trainer: Optional[str] = None,
    filter_model: Optional[str] = None,
):
    snr_cols = [c for c in df.columns if c.startswith("per_snr_")]
    if not snr_cols:
        print("No per_snr_* columns found; make sure metrics['per_snr_acc'] "
              "is flattened into the DataFrame.")
        return

    # sort SNR columns numerically
    snr_values = sorted(int(c.split("_")[-1]) for c in snr_cols)
    snr_cols_sorted = [f"per_snr_{s}" for s in snr_values]

    # filter runs if needed
    df_sel = df.copy()
    if filter_trainer is not None:
        df_sel = df_sel[df_sel["trainer"] == filter_trainer]
    if filter_model is not None:
        df_sel = df_sel[df_sel["model_type"] == filter_model]

    if df_sel.empty:
        print("No rows left after filtering; adjust filter_trainer/filter_model.")
        return

    # group by encoding_type and take mean across runs with same encoding
    group = df_sel.groupby("encoding_type")[snr_cols_sorted].mean()

    plt.figure(figsize=(8, 5))
    for enc, row in group.iterrows():
        plt.plot(snr_values, row.values, marker="o", label=enc)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR by encoding")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Plot 3: param vs test accuracy
# -------------------------------------------------------------------
def plot_param_vs_acc(
    df: pd.DataFrame,
    param_col: str,
    subset_mask=None,
    xscale: str = "linear",
):
    if param_col not in df.columns:
        print(f"Column '{param_col}' not found in DataFrame.")
        return

    if subset_mask is not None:
        df = df[subset_mask]

    # drop NaN params
    df = df[~df[param_col].isna()]

    if df.empty:
        print(f"No rows with non-NaN '{param_col}' after filtering.")
        return

    # group by param value; mean test_acc
    agg = (
        df.groupby(param_col, as_index=False)
          .agg({"test_acc": ["mean", "std", "count"]})
    )
    agg.columns = [param_col, "acc_mean", "acc_std", "count"]

    plt.figure(figsize=(6, 4))
    plt.errorbar(agg[param_col], agg["acc_mean"],
                 yerr=agg["acc_std"], fmt="-o")
    plt.xlabel(param_col)
    plt.ylabel("Test accuracy (mean ± std)")
    plt.xscale(xscale)
    plt.title(f"{param_col} vs test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if not df.empty:
    # 1) Overall test accuracy ranking
    plot_test_accuracy(df)

    # 2) Accuracy vs SNR
    plot_acc_vs_snr(df, filter_trainer="bptt", filter_model="conv2d_snn")

    # 3) Example: sigma-delta step vs acc
    if "sigma_delta_step" in df.columns:
        mask = (
            (df["encoding_type"] == "sigma_delta")
            & (df["model_type"] == "conv1d_snn")
        )
        plot_param_vs_acc(df, "sigma_delta_step", subset_mask=mask)
