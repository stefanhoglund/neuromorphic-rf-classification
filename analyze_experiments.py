# analyze_experiments.py

import argparse
from rf_experiments.analysis import collect_runs, runs_to_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="results",
        help="Root directory containing experiment run subdirectories.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="If set, save the full runs DataFrame to this CSV file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="*",
        dest="num_classes",
        help=(
            "Optional filter: only include runs whose num_classes is in this list. "
            "Example: --num-classes 2 24"
        ),
    )
    args = parser.parse_args()

    runs = collect_runs(args.root)
    df = runs_to_dataframe(runs)

    if df.empty:
        print(f"No runs with metrics found under {args.root}")
        return

    # Optional filter by num_classes
    if args.num_classes:
        df = df[df["num_classes"].isin(args.num_classes)]
        if df.empty:
            print(
                f"No runs found with num_classes in {args.num_classes} "
                f"under {args.root}"
            )
            return

    # Sort by test accuracy descending
    df = df.sort_values("test_acc", ascending=False, na_position="last")

    # ------------------------------------------------------------------ #
    # 1) Main run summary table
    # ------------------------------------------------------------------ #
    cols_to_show = [
        "run_date",
        "run_time",
        "experiment_name",
        "encoding_type",
        "model_type",
        "trainer",
        "training_device",
        "num_classes",
        "num_snrs",
        "snr_min",
        "snr_max",
        "test_acc",
        "train_time_seconds",
        "total_time_seconds",
    ]
    existing_cols = [c for c in cols_to_show if c in df.columns]

    print("\n=== Run summary (sorted by test_acc) ===")
    print(df[existing_cols].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2) Breakdown by num_classes (best / mean / std / count)
    # ------------------------------------------------------------------ #
    if "num_classes" in df.columns and "test_acc" in df.columns:
        cls_stats = (
            df.groupby("num_classes")["test_acc"]
              .agg(best="max", mean="mean", std="std", runs="count")
              .reset_index()
              .sort_values("num_classes")
        )

        print("\n=== Accuracy by num_classes ===")
        print(cls_stats.to_string(index=False))
    else:
        print("\n[WARN] num_classes or test_acc missing; cannot compute class breakdown.")

    # ------------------------------------------------------------------ #
    # 3) Breakdown by (num_classes, SNR range) (best / mean / std / count)
    # ------------------------------------------------------------------ #
    if {"num_classes", "snr_min", "snr_max", "test_acc"}.issubset(df.columns):
        cls_snr_stats = (
            df.groupby(["num_classes", "snr_min", "snr_max"])["test_acc"]
              .agg(best="max", mean="mean", std="std", runs="count")
              .reset_index()
              .sort_values(["num_classes", "snr_min", "snr_max"])
        )

        print("\n=== Accuracy by num_classes and SNR range ===")
        print(cls_snr_stats.to_string(index=False))
    else:
        print(
            "\n[WARN] num_classes/snr_min/snr_max/test_acc missing; "
            "cannot compute (classes,SNR) breakdown."
        )

    # ------------------------------------------------------------------ #
    # 4) Save CSV if requested
    # ------------------------------------------------------------------ #
    if args.out_csv is not None:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved full runs DataFrame to {args.out_csv}")


if __name__ == "__main__":
    main()
