"""Create mean ± std main table from per-seed results."""

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["SR@5", "SR@10", "SR@15", "AvgT", "hDCG", "reward"]


def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main_table.csv")
    parser.add_argument("--output", default="results/main_table_mean_std.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for (method, dataset), group in df.groupby(["method", "dataset"], dropna=False):
        row = {"method": method, "dataset": dataset, "num_seeds": group["seed"].nunique()}
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = fmt_mean_std(values.mean(), values.std(ddof=1))
        rows.append(row)

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {len(out)} rows to {output}")


if __name__ == "__main__":
    main()
