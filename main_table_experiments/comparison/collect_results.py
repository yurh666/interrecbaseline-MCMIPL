"""
从 MCMIPL 训练日志中提取完整 eval 的整体均值（非 observe 批次行），写 CSV/JSON。
解析逻辑见 mcmipl_log_metrics.py。
"""

import json
import argparse
import csv
import numpy as np
from pathlib import Path

from mcmipl_log_metrics import (
    METRICS,
    extract_evals,
    best_checkpoint_by_sr15,
)

DATASETS = ["LAST_FM_STAR", "YELP_STAR", "BOOK", "MOVIE"]
SEEDS = [0, 1, 2]

PRETTY = {"SR5": "SR@5", "SR10": "SR@10", "SR15": "SR@15",
          "AvgT": "AvgT", "Rank": "hDCG", "reward": "reward"}


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--log_dir", default=str(base / "logs"))
    parser.add_argument("--out_dir", default=str(base / "comparison" / "results" / "mcmipl"))
    parser.add_argument("--summary", default=str(base / "comparison" / "results" / "mcmipl_main_table.csv"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for ds in DATASETS:
        seed_results = {}
        for s in SEEDS:
            log_file = log_dir / f"train_{ds}_s{s}.log"
            evals = extract_evals(log_file)
            best = best_checkpoint_by_sr15(evals)
            done = log_file.exists() and "=== DONE:" in log_file.read_text(errors="ignore")

            print(f"[{ds}] seed={s}: {len(evals)} eval(s) | done={done} | "
                  f"best SR@15={best['SR15'] if best else 'N/A'}")

            if best:
                row = {
                    "method": "MCMIPL",
                    "dataset": ds,
                    "seed": s,
                    "n_evals": len(evals),
                    "training_done": done,
                }
                for k in METRICS:
                    row[PRETTY[k]] = best[k]
                all_rows.append(row)
                seed_results[s] = best

                json_path = out_dir / f"mcmipl_{ds}_s{s}.json"
                json_path.write_text(json.dumps(row, indent=2))
            else:
                print("  -> 暂无结果（训练尚未完成第一次 eval）")

        if len(seed_results) >= 2:
            for metric_key, pretty_key in PRETTY.items():
                vals = [seed_results[s][metric_key] for s in seed_results]
                print(f"  {pretty_key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                      f"(seeds={list(seed_results.keys())})")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        cols = ["method", "dataset", "seed", "n_evals", "training_done"] + list(PRETTY.values())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n已保存 {len(all_rows)} 行到 {summary_path}")
    else:
        print("\n暂无结果可保存。")

    print("\n" + "=" * 60)
    print("MCMIPL 复现结果汇总（mean ± std over seeds）")
    print("=" * 60)
    print(f"{'数据集':<15} {'SR@5':>12} {'SR@10':>12} {'SR@15':>12} {'AvgT':>10} {'hDCG':>10}")
    print("-" * 60)
    for ds in DATASETS:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        if not ds_rows:
            print(f"{ds:<15} {'—':>12} {'—':>12} {'—':>12} {'—':>10} {'—':>10}")
            continue

        def ms(mk):
            vals = [r[PRETTY[mk]] for r in ds_rows]
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

        print(f"{ds:<15} {ms('SR5'):>12} {ms('SR10'):>12} {ms('SR15'):>12} "
              f"{ms('AvgT'):>10} {ms('Rank'):>10}")


if __name__ == "__main__":
    main()
