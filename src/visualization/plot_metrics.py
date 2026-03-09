from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training or inference metrics from a CSV file.")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to the metrics CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to <metrics_csv_stem>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {args.metrics_csv}")

    metrics_df = pd.read_csv(args.metrics_csv)
    if metrics_df.empty:
        raise ValueError(f"Metrics CSV is empty: {args.metrics_csv}")

    numeric_df = metrics_df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError(f"Metrics CSV does not contain numeric columns: {args.metrics_csv}")

    output_path = args.output or args.metrics_csv.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "epoch" in metrics_df.columns and len(metrics_df) > 1:
        for column in numeric_df.columns:
            if column == "epoch":
                continue
            ax.plot(metrics_df["epoch"], metrics_df[column], marker="o", linewidth=1.5, label=column)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.set_title("Training Metrics")
    else:
        first_row = numeric_df.iloc[0].sort_values(ascending=False)
        ax.bar(first_row.index, first_row.values)
        ax.set_ylabel("value")
        ax.set_title("Metrics Summary")
        ax.tick_params(axis="x", rotation=45)

    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved metric plot to {output_path}")


if __name__ == "__main__":
    main()
