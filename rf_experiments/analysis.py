# rf_experiments/analysis.py

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import yaml

RUN_TS_RE = re.compile(r".*_(\d{8}_\d{6})$")

def _infer_run_datetime(run_name: str, run_dir: str) -> datetime | None:
    """
    Infer run datetime from the run_name suffix '_YYYYMMDD_HHMMSS'.
    If that fails, fall back to the directory's mtime.
    """
    m = RUN_TS_RE.match(run_name)
    if m:
        ts_str = m.group(1)
        try:
            return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    # fallback: filesystem modification time
    try:
        return datetime.fromtimestamp(os.path.getmtime(run_dir))
    except OSError:
        return None


@dataclass
class RunSummary:
    run_dir: str
    experiment_name: str
    encoding_type: str
    model_type: str
    trainer: str

    # encoding hyperparams
    grid_size: Optional[int]
    sigma_delta_step: Optional[float]
    sigma_delta_edge_only: Optional[bool]
    tf_events_threshold: Optional[float]

    # training meta
    seed: int
    training_device: Optional[str]
    train_time_seconds: Optional[float]
    total_time_seconds: Optional[float]

    # NEW: dataset meta
    num_classes: int
    snrs_used: List[int]          # list of SNRs used for this run

    # metrics
    val_loss: float
    val_acc: float
    test_loss: float
    test_acc: float

    # optional aggregated metrics
    per_snr_acc: Dict[int, float]


def _safe_get(d: Dict, path: List[str], default=None):
    x = d
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return default
        x = x[p]
    return x


def collect_runs(results_root: str) -> List[RunSummary]:
    runs: List[RunSummary] = []

    for entry in os.listdir(results_root):
        run_dir = os.path.join(results_root, entry)
        if not os.path.isdir(run_dir):
            continue

        cfg_path = os.path.join(run_dir, "config.yaml")
        metrics_path = os.path.join(run_dir, "metrics.json")

        if not (os.path.exists(cfg_path) and os.path.exists(metrics_path)):
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        with open(metrics_path, "r") as f:
            metrics_all = json.load(f)

        # by convention from ExperimentRunner
        best_val = metrics_all.get("best_val", {})
        test = metrics_all.get("test", metrics_all)

        encoding_type = _safe_get(cfg, ["encoding", "type"], "unknown")
        model_type = _safe_get(cfg, ["model", "type"], "unknown")
        trainer = _safe_get(cfg, ["training", "trainer"], "unknown")

        params = cfg.get("encoding", {}).get("params", {})

        dataset_cfg = cfg.get("dataset", {})
        mods_to_use = dataset_cfg.get("mods_to_use", None)
        model_params = cfg.get("model", {}).get("params", {})

        if mods_to_use is None:
            num_classes = int(model_params.get("num_classes", 24))
        else:
            num_classes = len(mods_to_use)

        # --- infer SNRs used --------------------------------------------
        snrs_to_use = dataset_cfg.get("snrs_to_use", None)
        if snrs_to_use is None:
            # full RadioML SNR range: -20..30 step 2 (26 values)
            snrs_used = list(range(-20, 32, 2))
        else:
            snrs_used = list(snrs_to_use)


        summary = RunSummary(
            run_dir=run_dir,
            experiment_name=_safe_get(cfg, ["experiment", "name"], entry),
            encoding_type=encoding_type,
            model_type=model_type,
            trainer=trainer,
            grid_size=params.get("grid_size"),
            sigma_delta_step=params.get("step"),
            sigma_delta_edge_only=params.get("edge_only"),
            tf_events_threshold=params.get("threshold"),
            seed=_safe_get(cfg, ["experiment", "seed"], 0),
            
            training_device=metrics_all.get("training_device"),
            train_time_seconds=metrics_all.get("train_time_seconds"),
            total_time_seconds=metrics_all.get("total_time_seconds"),
            
            num_classes=num_classes,
            snrs_used=snrs_used,


            val_loss=float(best_val.get("loss", float("nan"))),
            val_acc=float(best_val.get("acc", float("nan"))),
            test_loss=float(test.get("loss", float("nan"))),
            test_acc=float(test.get("acc", float("nan"))),
            per_snr_acc=test.get("per_snr_acc", {}),
        )
        runs.append(summary)

    return runs


def runs_to_dataframe(runs: List[RunSummary]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = asdict(r)

        run_dir = r.run_dir
        run_name = os.path.basename(run_dir)
        row["run_name"] = run_name

        dt = _infer_run_datetime(run_name, run_dir)
        if dt is not None:
            row["run_datetime"] = dt.isoformat(sep=" ")
            row["run_date"] = dt.date().isoformat()
            row["run_time"] = dt.time().isoformat(timespec="seconds")
        else:
            row["run_datetime"] = None
            row["run_date"] = None
            row["run_time"] = None

        # expand per_snr_acc dict into separate columns
        per_snr = row.pop("per_snr_acc", {}) or {}
        for snr, acc in per_snr.items():
            row[f"per_snr_{snr}"] = acc

        # NEW: flatten snrs_used list into meta columns
        snrs_used = row.pop("snrs_used", []) or []
        row["snr_list"] = ",".join(str(s) for s in snrs_used) if snrs_used else ""
        row["snr_min"] = min(snrs_used) if snrs_used else None
        row["snr_max"] = max(snrs_used) if snrs_used else None
        row["num_snrs"] = len(snrs_used)

        row["training_device"] = r.training_device
        row["train_time_seconds"] = r.train_time_seconds
        row["total_time_seconds"] = r.total_time_seconds

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_test_accuracy(df: pd.DataFrame,
                       group_cols=("encoding_type", "model_type", "trainer"),
                       sort_by="test_acc"):
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
    plt.tight_layout

def plot_acc_vs_snr(df: pd.DataFrame,
                    filter_trainer: Optional[str] = None,
                    filter_model: Optional[str] = None):
    snr_cols = [c for c in df.columns if c.startswith("per_snr_")]
    if not snr_cols:
        print("No per_snr_* columns found")
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


def plot_param_vs_acc(df: pd.DataFrame,
                      param_col: str,
                      subset_mask=None,
                      xscale="linear"):
    if subset_mask is not None:
        df = df[subset_mask]

    # drop NaN params
    df = df[~df[param_col].isna()]

    if df.empty:
        print(f"No rows with param {param_col}")
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

def plot_encoding_model_heatmap(
    df: pd.DataFrame,
    metric_col: str = "test_acc",
    out_path: str | None = None,
):
    """
    Heatmap of mean metric (default: test_acc) by (encoding_type, model_type).

    Each cell is the average test_acc across runs with same encoding+model.
    """
    if df.empty:
        raise ValueError("DataFrame is empty, nothing to plot.")

    required_cols = {"encoding_type", "model_type", metric_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for heatmap: {missing}")

    pivot = df.pivot_table(
        index="encoding_type",
        columns="model_type",
        values=metric_col,
        aggfunc="mean",
    )

    if pivot.empty:
        raise ValueError("Pivot table is empty, nothing to plot.")

    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * pivot.shape[1],
                                    1.5 + 0.4 * pivot.shape[0]))

    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("model_type")
    ax.set_ylabel("encoding_type")
    ax.set_title(f"{metric_col} (mean) by encoding/model")

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < pivot.values.mean() else "black",
                )

    fig.colorbar(im, ax=ax, label=metric_col)
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig, ax

def plot_metric_box_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "test_acc",
    out_path: str | None = None,
):
    """
    Boxplot of metric (default: test_acc) grouped by group_col
    (e.g. encoding_type, model_type, trainer).
    """
    if df.empty:
        raise ValueError("DataFrame is empty, nothing to plot.")

    for c in (group_col, metric_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

    df_plot = df[[group_col, metric_col]].dropna()
    if df_plot.empty:
        raise ValueError(f"No non-NaN values for '{metric_col}' to plot.")

    groups = sorted(df_plot[group_col].unique())
    data = [df_plot[df_plot[group_col] == g][metric_col].values for g in groups]

    fig, ax = plt.subplots(figsize=(1.5 + 0.7 * len(groups), 5))

    ax.boxplot(data, labels=groups)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} by {group_col}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig, ax

def plot_learning_curves_for_run(run_dir: str, out_prefix: str | None = None):
    """
    Plot train/val loss and accuracy vs epoch for a single run.

    Expects metrics_per_epoch.json in run_dir with keys like:
      epoch, train_loss, val_loss, train_acc, val_acc
    """
    mpe_path = os.path.join(run_dir, "metrics_per_epoch.json")
    if not os.path.isfile(mpe_path):
        raise FileNotFoundError(f"No metrics_per_epoch.json in {run_dir}")

    with open(mpe_path, "r") as f:
        mpe = json.load(f)

    if not mpe:
        raise ValueError(f"metrics_per_epoch.json in {run_dir} is empty")

    df = pd.DataFrame(mpe)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    if "train_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss vs epoch")
    axes[0].legend()

    # Accuracy
    if "train_acc" in df.columns:
        axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    if "val_acc" in df.columns:
        axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy vs epoch")
    axes[1].legend()

    fig.suptitle(os.path.basename(run_dir))
    fig.tight_layout()

    if out_prefix is not None:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        fig.savefig(out_prefix + "_learning_curves.png", dpi=150)

    return fig, axes

if __name__ == "__main__":
    root = "results"
    runs = collect_runs(root)
    df = runs_to_dataframe(runs)
    print(df.head())