# Main Table Experiments

本目录用于论文主表实验，和 `choice_diagnostic/` 选择题机制预实验严格区分。

## 目录结构

```text
main_table_experiments/
  baselines/
    mcmipl_official/
      MCMIPL/
      environment.yml
      Dockerfile
      scripts/
      configs/
      patches/
      results/
      docs/
  ours/
    interrec/
  comparison/
```

## MCMIPL 官方 baseline

官方 repo：

```text
https://github.com/ZYM6-6/MCMIPL
```

本地路径：

```text
baselines/mcmipl_official/MCMIPL
```

复现文档：

```text
baselines/mcmipl_official/docs/mcmipl_code_review.md
baselines/mcmipl_official/docs/mcmipl_reproduction_report.md
```

### 1. 环境

```bash
cd baselines/mcmipl_official
bash scripts/setup_env.sh
```

如果 conda 安装失败，使用 Dockerfile。

### 2. 数据准备

必须使用官方 released data，放在：

```text
MCMIPL/data/<data_name>
```

不要使用 `choice_diagnostic/data_preparation` 的小样本数据替代主表 baseline。

检查数据：

```bash
bash scripts/prepare_data.sh LAST_FM_STAR
```

### 3. Graph init

```bash
bash scripts/run_graph_init.sh LAST_FM_STAR
```

### 3b. CPU/GPU 分阶段（推荐用于跨机器）

| 阶段 | 脚本 | 说明 |
|------|------|------|
| A（CPU） | `run_pipeline_phase_cpu.sh` | 可选 `RUN_REBUILD_BOOK_MOVIE=1`；对每个数据集 `prepare_data` + `graph_init` |
| B（GPU RL） | `run_pipeline_phase_gpu.sh` | 需已有各集 `tmp/<slug>/embeds/transe.pkl`；可设 `REQUIRE_TRANSE=1` |
| B 队列（BOOK 后 1/2/4） | `scripts/run_phaseB_after_book_then_lastfm_yelp_movie.sh` | 等 `logs/train_BOOK_s{0,1,2}.log` 均出现 `DONE` 后，串行 Phase B：`LAST_FM_STAR`、`YELP_STAR`、`MOVIE`（跳过 BOOK）；`SKIP_WAIT=1` 可不调试用 |
| B（仅 CPU RL） | `run_pipeline_phase_rl_cpu.sh` | 设置 `MCMIPL_FORCE_CPU=1` 后调用 Phase B 调度 |
| 单机训练入口 | `run_mcmipl.sh` | `MCMIPL_DIR`/`LOG_DIR`/`MCMIPL_*_PYTHON` 可用环境变量覆盖 |

Phase A 完成后可运行 `bash scripts/record_phase_a_artifacts.sh` 生成 `artifacts/phase_a_manifest_*.txt`（sha256 清单），便于提交 git 与交给 GPU 机核对。

完整交接文案与「给 AI 的 GPU Prompt」见仓库根目录 [`docs/HANDOFF_RUNBOOK.md`](../docs/HANDOFF_RUNBOOK.md)。**若只在** `baselines/mcmipl_official/scripts/` **下找脚本**，同仓库也提供转发到上述入口的同名 `.sh`（避免路径混淆；以 `main_table_experiments/` 下文件为准）。

### 4. 训练

```bash
bash scripts/run_train.sh LAST_FM_STAR 0
bash scripts/run_train.sh LAST_FM_STAR 1
bash scripts/run_train.sh LAST_FM_STAR 2
```

### 5. 评估

```bash
bash scripts/run_eval.sh LAST_FM_STAR 100 0
bash scripts/run_eval.sh LAST_FM_STAR 100 1
bash scripts/run_eval.sh LAST_FM_STAR 100 2
```

评估 JSON 输出到：

```text
baselines/mcmipl_official/results/metrics/
```

## InterRec

我们的主方法代码放在：

```text
ours/interrec/
```

主表比较时应尽量复用 MCMIPL 的官方 data split、target / acceptable item 定义、max turns、success 判断和指标。

解析训练日志时注意：`collect_results.py` 与各报告只读取每次 eval **结束后**不含 `Total epoch_uesr` 后缀的整体均值行；带 `Total epoch_uesr:N` 的仅为 `observe_num` 一批用户的滑动统计，不能代替全量 `test_size` 结果。实现见 `comparison/mcmipl_log_metrics.py`。

## 主表汇总

```bash
cd comparison
python collect_results.py
python make_main_table.py
python statistical_test.py
python plot_curves.py
```

输出：

```text
comparison/results/main_table.csv
comparison/results/main_table_mean_std.csv
comparison/results/significance_tests.csv
comparison/results/figures/
```

## 选择题诊断实验 vs 主表实验

`choice_diagnostic/` 中的实验只用于分析选择题机制：intent-level choices 是否能覆盖真实偏好。它不能替代主表 baseline。

主表 MCMIPL baseline 必须走官方链路：

```text
official data -> graph_init.py -> TransE/OpenKE embeddings -> RL_model.py -> evaluate.py -> official metrics
```
