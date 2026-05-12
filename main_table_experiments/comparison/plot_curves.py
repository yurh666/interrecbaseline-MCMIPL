"""Plot metric curves/bars from collected main table results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main_table.csv")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["SR@5", "SR@10", "SR@15", "AvgT", "hDCG"]:
        if metric not in df.columns:
            continue
        plot_df = df.copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        means = plot_df.groupby(["dataset", "method"])[metric].mean().reset_index()
        pivot = means.pivot(index="dataset", columns="method", values=metric)
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        safe_metric = metric.replace("@", "at")
        plt.savefig(out_dir / f"{safe_metric}.png", dpi=150)
        plt.close()

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
