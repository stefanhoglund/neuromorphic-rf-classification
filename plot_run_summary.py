#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_snr_accuracy_from_arrays(y_true, y_pred, snr):
    """Return dict: snr_value -> accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    snr = np.asarray(snr)

    acc_by_snr = {}
    for s in np.unique(snr):
        mask = snr == s
        if mask.sum() == 0:
            continue
        acc = (y_true[mask] == y_pred[mask]).mean()
        acc_by_snr[float(s)] = acc
    return acc_by_snr


def load_snr_accuracy_from_npz(path):
    """Load y_true, y_pred, snr from NPZ and compute per-SNR accuracy."""
    d = np.load(path)
    if not {"y_true", "y_pred", "snr"}.issubset(d.files):
        raise ValueError(f"{path} must contain y_true, y_pred and snr arrays.")
    return compute_snr_accuracy_from_arrays(d["y_true"], d["y_pred"], d["snr"])


def plot_grouped_snr_accuracy(npz_paths, labels, out_path):
    """
    Grouped bar chart:
      x-axis: SNR
      bars: one per condition (e.g. normal / dropout / time-shuffle).
    Fonts are reduced so the plot is less cluttered.
    """
    if len(npz_paths) != len(labels):
        raise ValueError("npz_paths and labels lengths must match.")

    acc_dicts = [load_snr_accuracy_from_npz(p) for p in npz_paths]

    # union of all SNRs across experiments
    all_snrs = sorted({s for d in acc_dicts for s in d.keys()})
    x = np.arange(len(all_snrs))

    n = len(acc_dicts)
    width = 0.8 / n  # total width ~0.8 of the group

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, (label, acc_d) in enumerate(zip(labels, acc_dicts)):
        heights = [acc_d.get(s, np.nan) for s in all_snrs]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=label)

    # X axis: SNR values
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(round(s))) for s in all_snrs], fontsize=7)
    ax.set_xlabel("SNR (dB)", fontsize=7)
    ax.set_ylabel("Accuracy", fontsize=7)
    ax.set_ylim(0.0, 1.0)

    ax.set_title("Test accuracy per SNR – comparison", fontsize=8)

    # smaller ticks / legend
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)




def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    out_path,
    normalize=True,
    title=None,
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # avoid division by zero
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized count" if normalize else "Count")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if title is None:
        title = "Confusion matrix (normalized)" if normalize else "Confusion matrix"
    ax.set_title(title)

    # Annotate cells with small font so it’s not a dense blob
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices_per_snr(
    snr,
    y_true,
    y_pred,
    class_names,
    out_dir,
    normalize=True,
):
    """
    For each unique SNR value, plot a confusion matrix using only samples
    with that SNR. Saves one PNG per SNR into out_dir.
    """
    if snr is None:
        print("[WARN] No 'snr' in npz; skipping per-SNR confusion matrices.")
        return

    snr = np.asarray(snr)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_snrs = np.unique(snr)

    for s in unique_snrs:
        mask = snr == s
        n = int(mask.sum())
        if n == 0:
            continue

        s_int = int(round(float(s))) 
        out_path = os.path.join(out_dir, f"confusion_matrix_snr_{s_int}.png")

        title = f"Confusion matrix (normalized) – SNR {s_int} dB" if normalize \
                else f"Confusion matrix – SNR {s_int} dB"

        plot_confusion_matrix(
            y_true[mask],
            y_pred[mask],
            class_names,
            out_path,
            normalize=normalize,
            title=title,
        )
        print(f"[INFO] Saved confusion matrix for SNR {s_int} dB to {out_path}")

def plot_per_snr_curves(run_dir: str):
    snr_csv = os.path.join(run_dir, "metrics_per_snr_epoch.csv")
    if not os.path.exists(snr_csv):
        print(f"No metrics_per_snr_epoch.csv in {run_dir}, skipping per-SNR plots.")
        return

    df = pd.read_csv(snr_csv)

    df_val = df[df["split"] == "val"].copy()

    if df_val.empty:
        print("No validation rows in metrics_per_snr_epoch.csv, skipping per-SNR plots.")
        return

    # ---- accuracy per SNR vs epoch ----
    plt.figure(figsize=(8, 6))
    for snr in sorted(df_val["snr"].unique()):
        sub = df_val[df_val["snr"] == snr]
        plt.plot(sub["epoch"], sub["acc"], marker="o", label=f"SNR {snr} dB")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy per SNR over epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_accuracy_per_snr.png"), dpi=150)
    plt.close()

    # ---- loss per SNR vs epoch ----
    if "loss" in df_val.columns:
        plt.figure(figsize=(8, 6))
        for snr in sorted(df_val["snr"].unique()):
            sub = df_val[df_val["snr"] == snr]
            plt.plot(sub["epoch"], sub["loss"], marker="o", label=f"SNR {snr} dB")

        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("Validation loss per SNR over epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "val_loss_per_snr.png"), dpi=150)
        plt.close()
    else:
        print("No 'loss' column in metrics_per_snr_epoch.csv, skipping per-SNR loss plot.")


def plot_train_curves(run_dir, out_dir):
    metrics_path = os.path.join(run_dir, "metrics_per_epoch.csv")
    if not os.path.exists(metrics_path):
        print(f"[WARN] {metrics_path} not found, skipping train/val curves.")
        return

    df = pd.read_csv(metrics_path)

    # Expect columns: epoch, split, loss, acc
    if not {"epoch", "split", "loss", "acc"}.issubset(df.columns):
        print(f"[WARN] Unexpected columns in {metrics_path}, got {df.columns}.")
        return

    df["epoch"] = df["epoch"].astype(int)

    train = df[df["split"] == "train"].sort_values("epoch")
    val = df[df["split"] == "val"].sort_values("epoch")

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    if not train.empty:
        ax.plot(train["epoch"], train["loss"], label="train")
    if not val.empty:
        ax.plot(val["epoch"], val["loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Loss over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(7, 4))
    if not train.empty:
        ax.plot(train["epoch"], train["acc"], label="train")
    if not val.empty:
        ax.plot(val["epoch"], val["acc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200)
    plt.close(fig)


def plot_snr_accuracy_from_predictions(snr, y_true, y_pred, out_path):
    if snr is None:
        print("[WARN] No 'snr' in npz; skipping per-SNR plot.")
        return

    snr = np.asarray(snr)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_snrs = np.unique(snr)
    accs = []
    for s in unique_snrs:
        mask = snr == s
        if mask.sum() == 0:
            accs.append(0.0)
            continue
        correct = (y_true[mask] == y_pred[mask]).sum()
        accs.append(correct / mask.sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(unique_snrs.astype(str), accs)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test accuracy per SNR (from predictions)")
    ax.set_ylim(0.0, 1.0)          # <-- force y-scale to [0, 1]
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Summarize a training run: confusion matrix + curves."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Path to the run directory (with CSV + NPZ)."
    )
    parser.add_argument(
        "--pred-file",
        default="test_predictions_final.npz",
        help="Prediction npz filename inside run-dir "
             "(e.g. test_predictions_final.npz or test_predictions_epoch37.npz).",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional path to a text file with one class name per line.",
    )
    parser.add_argument(
        "--compare-pred-files",
        nargs="+",
        default=None,
        help=(
            "Optional list of prediction NPZ files (relative to run-dir or absolute) "
            "to compare in a grouped per-SNR plot (e.g. baseline/dropout/shuffle)."
        ),
    )
    parser.add_argument(
        "--compare-labels",
        nargs="+",
        default=None,
        help=(
            "Labels for the compared prediction files; must match length of "
            "--compare-pred-files. If omitted, generic labels are used."
        ),
    )



    args = parser.parse_args()

    run_dir = args.run_dir
    pred_path = args.pred_file
    if not os.path.isabs(pred_path):
        pred_path = os.path.join(run_dir, pred_path)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")


    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pred_base = os.path.splitext(os.path.basename(pred_path))[0]
    if pred_base.startswith("test_predictions_"):
        scenario_tag = pred_base[len("test_predictions_"):]
    else:
        scenario_tag = pred_base


    # 1) Load predictions
    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    snr = data["snr"] if "snr" in data.files else None

    latency_ms_per_sample = None
    energy_j_per_sample = None

    if "latency_ms_per_sample" in data.files:
        latency_ms_per_sample = float(data["latency_ms_per_sample"])

    if "energy_j_per_sample" in data.files:
        energy_j_per_sample = float(data["energy_j_per_sample"])

    # 2) Class names
    class_names = None
    if args.class_names is not None:
        with open(args.class_names, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

    # 3) Confusion matrix
    cm_path = os.path.join(plots_dir, f"confusion_matrix_{scenario_tag}.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, normalize=True)
    print(f"[INFO] Saved confusion matrix to {cm_path}")


    # 3b) Confusion matrix per SNR
    plot_confusion_matrices_per_snr(snr, y_true, y_pred, class_names, plots_dir)



    # 4) Train / val curves (loss, acc)
    plot_train_curves(run_dir, plots_dir)
    print(f"[INFO] Saved train/val curves to {plots_dir}")

    plot_per_snr_curves(args.run_dir)


    # 5) Per-SNR accuracy from predictions
    snr_acc_path = os.path.join(plots_dir, f"test_acc_per_snr_{scenario_tag}.png")
    plot_snr_accuracy_from_predictions(snr, y_true, y_pred, snr_acc_path)
    print(f"[INFO] Saved test per-SNR accuracy plot to {snr_acc_path}")


    # 6) Latency / energy summary
    if latency_ms_per_sample is not None:
        print(f"[INFO] avg latency per sample: {latency_ms_per_sample:.3f} ms")

    if energy_j_per_sample is not None:
        print(
            f"[INFO] approx energy per sample: {energy_j_per_sample*1e3:.3f} mJ "
            f"(based on assumed_power_w in config)"
        )


    # 7) Optional grouped comparison (baseline vs dropout vs time-shuffle)
    if args.compare_pred_files:
        # Resolve to absolute paths
        compare_paths = []
        for p in args.compare_pred_files:
            if not os.path.isabs(p):
                p_full = os.path.join(run_dir, p)
            else:
                p_full = p
            if not os.path.exists(p_full):
                print(f"[WARN] compare pred file not found, skipping: {p_full}")
                continue
            compare_paths.append(p_full)

        if len(compare_paths) >= 2:
            if args.compare_labels and len(args.compare_labels) == len(compare_paths):
                labels = args.compare_labels
            else:
                labels = [f"run{i+1}" for i in range(len(compare_paths))]

            grouped_path = os.path.join(plots_dir, "grouped_test_acc_per_snr.png")
            plot_grouped_snr_accuracy(compare_paths, labels, grouped_path)
            print(f"[INFO] Saved grouped per-SNR accuracy plot to {grouped_path}")
        else:
            print("[WARN] Need at least 2 valid --compare-pred-files to make grouped plot.")

if __name__ == "__main__":
    main()
