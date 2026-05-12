"""Run paired significance tests for key metrics."""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


KEY_METRICS = ["SR@15", "AvgT", "hDCG"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main_table.csv")
    parser.add_argument("--baseline", default="MCMIPL")
    parser.add_argument("--method", default="InterRec")
    parser.add_argument("--output", default="results/significance_tests.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for dataset in sorted(df["dataset"].dropna().unique()):
        base = df[(df["dataset"] == dataset) & (df["method"] == args.baseline)]
        method = df[(df["dataset"] == dataset) & (df["method"] == args.method)]
        merged = base.merge(method, on=["dataset", "seed"], suffixes=("_baseline", "_method"))
        for metric in KEY_METRICS:
            a = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
            b = pd.to_numeric(merged[f"{metric}_method"], errors="coerce")
            valid = ~(a.isna() | b.isna())
            if valid.sum() >= 2:
                stat, pvalue = stats.ttest_rel(b[valid], a[valid])
            else:
                stat, pvalue = float("nan"), float("nan")
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "baseline": args.baseline,
                "method": args.method,
                "paired_n": int(valid.sum()),
                "t_stat": stat,
                "p_value": pvalue,
            })

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {len(out)} tests to {output}")


if __name__ == "__main__":
    main()
