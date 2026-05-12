"""
在每个数据集训练完成后生成 Markdown 报告。
用法：
  python generate_report.py --dataset LAST_FM_STAR
  python generate_report.py --dataset ALL
"""
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from mcmipl_log_metrics import (
    extract_evals,
    best_checkpoint_by_sr15,
    METRICS,
    parse_max_training_steps,
    count_completed_training_steps,
    eval_training_step_labels,
)

DATASETS = ["LAST_FM_STAR", "YELP_STAR", "BOOK", "MOVIE"]
SEEDS = [0, 1, 2]
EVAL_NUM = 10

PRETTY = {"SR5": "SR@5", "SR10": "SR@10", "SR15": "SR@15",
          "AvgT": "AvgT", "Rank": "hDCG", "reward": "reward"}

# 与 run_all_datasets.sh 一致；LAST_FM_STAR 早期单独跑时使用 100。
DEFAULT_MAX_STEPS = {
    "LAST_FM_STAR": 100,
    "YELP_STAR": 50,
    "BOOK": 50,
    "MOVIE": 50,
}

DS_META = {
    "LAST_FM_STAR": {"name": "LastFM", "eval_users": 4000, "domain": "音乐推荐"},
    "YELP_STAR": {"name": "Yelp", "eval_users": 2500, "domain": "餐厅/商家推荐"},
    "BOOK": {"name": "Amazon Books", "eval_users": 2500, "domain": "图书推荐"},
    "MOVIE": {"name": "MovieLens", "eval_users": 2500, "domain": "电影推荐"},
}


def get_start_time(log_path: Path):
    if not log_path.exists():
        return "未知"
    text = log_path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    for line in text.split("\n"):
        if "MCMIPL |" in line and "seed=" in line:
            return line.strip()
    return "未知"


def scheduled_eval_steps(n_evals: int) -> list[int]:
    """当前复现脚本统一使用官方默认 eval_num=10（含训练前 step 0 的一次 eval）。"""
    return eval_training_step_labels(n_evals, eval_num=EVAL_NUM)


def generate_dataset_report(dataset: str, log_dir: Path, out_dir: Path):
    meta = DS_META[dataset]
    seed_data = {}

    for s in SEEDS:
        log_path = log_dir / f"train_{dataset}_s{s}.log"
        evals = extract_evals(log_path)
        done = log_path.exists() and "=== DONE:" in log_path.read_text(errors="ignore")
        steps_done = count_completed_training_steps(log_path)
        planned = parse_max_training_steps(log_path) or DEFAULT_MAX_STEPS.get(dataset)
        best = best_checkpoint_by_sr15(evals)
        seed_data[s] = {
            "evals": evals,
            "best": best,
            "done": done,
            "steps_done": steps_done,
            "planned_steps": planned,
            "start": get_start_time(log_path),
        }

    completed_seeds = [s for s in SEEDS if seed_data[s]["done"] and seed_data[s]["best"]]
    has_results = len(completed_seeds) > 0

    if not has_results:
        completed_seeds = [s for s in SEEDS if seed_data[s]["best"]]

    if not completed_seeds:
        print(f"[{dataset}] 暂无任何 eval 结果，跳过报告。")
        return

    max_planned = seed_data[completed_seeds[0]]["planned_steps"]

    best_vals = {k: [seed_data[s]["best"][k] for s in completed_seeds] for k in METRICS}
    mean_std = {k: (np.mean(best_vals[k]), np.std(best_vals[k])) for k in METRICS}

    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# MCMIPL 复现报告：{meta['name']} ({dataset})")
    lines.append(f"\n> 生成时间：{now}  |  数据集领域：{meta['domain']}")

    lines.append("\n## 一、指标解析说明（重要）\n")
    lines.append("""
官方 `RL_evaluate.py` 在评估中会打印两类行：
- **批次行**：行末带 `Total epoch_uesr:N`，仅为最近 `observe_num` 名用户的滑动统计，**数值波动大**，不能代表整次 eval。
- **整体均值行**：同样含 `reward:`（单数），但**行尾无** `Total epoch_uesr`，是一次完整评估（LAST_FM：`test_size=4000`；Yelp 等：`2500`）后的真实均值，与源码中追踪 `best!!!` 的逻辑一致。

本报告与 `collect_results.py` **只采纳整体均值行**，并在各次 eval 中取 **SR@15 最高**的 checkpoint（与官方 `RL_model.train` 中 `SR15_best` 更新方式一致）。
""")

    lines.append("## 二、实验目的\n")
    lines.append(f"""
本实验是对 MCMIPL（Multiple Choice Questions Based Multi-Interest Policy Learning for Conversational Recommendation, WWW 2022）的官方复现，
目的是在 {meta['name']} 数据集上获得其标准性能指标，作为 InterRec 方法的主表比较基准。

- **基准方法**：MCMIPL 官方实现（commit `01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5`）
- **数据集**：{meta['name']}（{meta['domain']}）
- **评估协议**：每隔 `eval_num={EVAL_NUM}` 个训练步做一次完整评估，取历次 eval 上 **SR@15 最优**结果为最终报表数值
- **随机种子**：{", ".join(str(s) for s in completed_seeds)}
""")

    lines.append("\n### 典型超参数（与当前流水线一致时请以此为准）\n")
    planned_display = max_planned if max_planned is not None else "见 run_all_datasets.sh"
    lines.append(f"""
| 参数 | 值 | 说明 |
|------|-----|------|
| max_steps | {planned_display} | RL 外层迭代步数 |
| sample_times | 100 | 每步采样的 episode 数 |
| eval_num | {EVAL_NUM} | 每若干步做一次完整评估 |
| max_turn | 15 | 单会话最大对话轮数 |
| choice_num | 4 | MCQ 选项个数 |
| embed | transe | TransE 初始化 |
""")
    lines.append("（若日志首行包含 `steps=…`，则以该值为 max_steps；否则上表默认值来自 `comparison/generate_report.py` 中的 `DEFAULT_MAX_STEPS`。）")

    lines.append("\n## 三、训练过程与各次 eval（整体均值）\n")
    for s in SEEDS:
        sd = seed_data[s]
        planned = sd["planned_steps"]
        if sd["done"]:
            progress = "✅ 已完成"
        elif planned is not None:
            progress = f"⏳ 进行中（训练步进度 {sd['steps_done']}/{planned}）"
        else:
            progress = (
                f"⏳ 进行中（已完成 {sd['steps_done']} 训练步,"
                f" {len(sd['evals'])} 次完整 eval）"
            )
        lines.append(f"**Seed {s}**：{progress}")
        lines.append(f"- 启动标记：`{sd['start']}`")
        if sd["evals"]:
            step_labels = scheduled_eval_steps(len(sd["evals"]))
            lines.append(f"- 各 eval 的整体均值 SR@15（checkpoint 对齐 training step，`eval_num={EVAL_NUM}`）：")
            for i, e in enumerate(sd["evals"]):
                st = (
                    step_labels[i]
                    if i < len(step_labels)
                    else i * EVAL_NUM
                )
                lines.append(
                    f"  - step {st}: SR@5={e['SR5']:.3f}, SR@10={e['SR10']:.3f}, "
                    f"SR@15={e['SR15']:.4f}, AvgT={e['AvgT']:.2f}, hDCG={e['Rank']:.4f}"
                )
            if sd["best"]:
                b = sd["best"]
                lines.append(
                    f"- **按 SR@15 选最优 checkpoint**：SR@15={b['SR15']:.4f}, "
                    f"AvgT={b['AvgT']:.2f}, hDCG={b['Rank']:.4f}"
                )
        else:
            lines.append("- 暂无 eval 整体均值输出")
        lines.append("")

    lines.append("\n## 四、汇总结果（各 seed 上取最优 checkpoint 后再跨 seed 聚合）\n")
    lines.append(f"\n基于 {len(completed_seeds)} 个 seed（{completed_seeds}）：\n")
    lines.append("| 指标 | " + " | ".join(f"Seed {s}" for s in completed_seeds) + " | **均值 ± 标准差** |")
    lines.append("|------|" + "-------|" * len(completed_seeds) + "---------|")
    for k in ["SR5", "SR10", "SR15", "AvgT", "Rank"]:
        pk = PRETTY[k]
        seed_cols = " | ".join(
            f"{seed_data[s]['best'][k]:.4f}"
            for s in completed_seeds
            if seed_data[s]["best"]
        )
        vals = [seed_data[s]["best"][k] for s in completed_seeds if seed_data[s]["best"]]
        m, std = np.mean(vals), np.std(vals)
        lines.append(f"| {pk} | {seed_cols} | **{m:.4f} ± {std:.4f}** |")

    lines.append("\n```")
    lines.append(f"数据集: {dataset}")
    for k in ["SR5", "SR10", "SR15", "AvgT", "Rank"]:
        pk = PRETTY[k]
        m, std = mean_std[k]
        lines.append(f"{pk:>8}: {m:.4f} ± {std:.4f}")
    lines.append("```")

    sr15_mean = mean_std["SR15"][0]
    avgt_mean = mean_std["AvgT"][0]
    sr5_mean = mean_std["SR5"][0]

    lines.append("\n## 五、简要解读\n")
    lines.append(f"- **SR@15={sr15_mean:.3f}**：主成功率指标。\n")
    lines.append(f"- **SR@5={sr5_mean:.3f}**；与 SR@15 的间隙反映后半程对话的贡献。\n")
    lines.append(f"- **AvgT={avgt_mean:.2f}**：成功会话的平均轮数。\n")

    lines.append("\n---\n数据来源：`comparison/mcmipl_log_metrics.py`（与 `collect_results.py` 共用）。")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{dataset}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存：{report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--dataset", default="ALL", choices=DATASETS + ["ALL"])
    parser.add_argument("--log_dir", default=str(base / "logs"))
    parser.add_argument("--out_dir", default=str(base / "comparison" / "results" / "reports"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    targets = DATASETS if args.dataset == "ALL" else [args.dataset]
    for ds in targets:
        generate_dataset_report(ds, log_dir, out_dir)


if __name__ == "__main__":
    main()
