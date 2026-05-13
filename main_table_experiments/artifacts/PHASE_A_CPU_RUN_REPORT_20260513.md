# Phase A（CPU）执行情况报告 — 2026-05-13

## 结论

- **处理的数据集**：主表四条 **`LAST_FM_STAR`、`YELP_STAR`、`BOOK`、`MOVIE`** 均已在本机执行 **`run_pipeline_phase_cpu.sh`** 中的 **`prepare_data.sh` → `run_graph_init.sh`**（与仓库默认 Phase A 范围一致）。
- **无需补跑**：四条均在同一跑次内完成；日志末尾 **`EXIT=0`**。
- **相对上游提交的增量**：Git 仅需纳入 **`yelp_star` 的 `dataset.pkl` / `kg.pkl`** 二进制变更（其余三条图产物与已有提交一致）；另纳入本轮 **`prepare_data` / `graph_init` 文本日志**、**tmux 控制台日志**及 **`phase_a_manifest_*` 清单**。

## 证据路径（仓库内）

| 类型 | 路径 |
|------|------|
| Phase A 汇总控制台日志 | `main_table_experiments/logs/tmux_phase_a_20260513_181342.log` |
| `prepare_data` / `graph_init` 分数据集 tee 日志 | `main_table_experiments/baselines/mcmipl_official/results/raw_logs/prepare_data_*.log`、`graph_init_*.log` |
| 产物清单（含 sha256） | `main_table_experiments/artifacts/phase_a_manifest_latest.txt` → `phase_a_manifest_20260513_184825.txt` |

## 图初始化产物核对（MCMIPL/tmp）

每个 slug 下均存在 **`dataset.pkl`、`kg.pkl`、`embeds/transe.pkl`**（TransE 权重沿用仓库既有或与构图一致的副本；Phase A 脚本本身不训练 OpenKE）。

| Slug | dataset.pkl（约） | kg.pkl（约） | transe.pkl（约） |
|------|-------------------|--------------|------------------|
| last_fm_star | 52 KB | 1.8 MB | 4.3 MB |
| yelp_star | 9.7 MB | 17.9 MB | 24 MB |
| book | 55 KB | 940 KB | 7.2 MB |
| movie | 53 KB | 2.1 MB | 7.3 MB |

## 环境与命令备忘

- Conda：`mcmipl-reproduce`（示例：`/home/yurh/.conda/envs/mcmipl-reproduce/bin/python`）。
- Phase A 脚本内已 **`CUDA_VISIBLE_DEVICES=""`**。
- **未**设置 `RUN_REBUILD_BOOK_MOVIE=1`（未在本次强制重跑 InterRec BOOK/MOVIE 时序重建）。

## Git

- 清单生成时记录的 **`git HEAD`**：`e0bd5573174180c659d490de14f8a8f974be5992`（推送成功后以远端 **`git rev-parse HEAD`** 为准）。
