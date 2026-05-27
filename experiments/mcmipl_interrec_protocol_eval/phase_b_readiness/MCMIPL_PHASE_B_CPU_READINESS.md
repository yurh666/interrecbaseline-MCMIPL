# MCMIPL Phase B CPU Readiness Audit

**生成时间：** 2026-05-27  
**环境：** CPU 主机（`/home/yrh666`，venv: `/home/yrh666/venvs/mcmipl-cpu/bin/python`）  
**权威日志：** `main_table_experiments/logs/train_{DATASET}_s{seed}.log`（2026-05-16 起的 CPU master / resume 流水线）  
**不采用：** 2026-05-15 及之前的 Yelp 旧日志（如 `train_YELP_STAR_s1.log` 中的 DONE）

---

## 执行摘要

| 数据集 | Phase B 完成 seeds | 磁盘 RL checkpoint | 可直接 eval-only |
|--------|-------------------|-------------------|------------------|
| LAST_FM_STAR | s0, s1, s2 全部 DONE | 有 epoch-50（**仅 s2 写入**） | 否（缺 interrec eval 脚本；s0/s1 ckpt 已被覆盖） |
| BOOK | s0, s1, s2 全部 DONE | 有 epoch-50（**仅 s2 写入**） | 否 |
| MOVIE | s0, s2 DONE；**s1 未完成** | 有 epoch-50（**s2**） | 否 |
| YELP_STAR | **s0 运行中**；s1/s2 未开始 | 有 epoch-10/20/30（s0 进行中）；epoch-50 为**旧跑次残留** | 否 |

**当前 screen：** `mcmipl_phase_b_cpu` → `run_phase_b_resume_lastfm_yelp.sh` → 正在跑 **YELP_STAR seed=0**

---

## 逐数据集 / seed 明细

### LAST_FM_STAR (`last_fm_star`)

| seed | Phase B DONE | 权威日志 | best epoch | 磁盘 checkpoint | eval-only |
|------|-------------|---------|------------|----------------|-----------|
| 0 | 是 | train_LAST_FM_STAR_s0.log | 50 | 无（被 s2 覆盖） | 否 |
| 1 | 是 | train_LAST_FM_STAR_s1.log | 50 | 无（被 s2 覆盖） | 否 |
| 2 | 是 | train_LAST_FM_STAR_s2.log | 50 | `tmp/last_fm_star/RL-agent/*-epoch-50.pkl` | 否* |

\* Phase-A 工件齐全，但缺少 `scripts/eval_mcmipl_on_interrec_protocol.py`。

### BOOK (`book`)

| seed | Phase B DONE | 权威日志 | best epoch | 磁盘 checkpoint | eval-only |
|------|-------------|---------|------------|----------------|-----------|
| 0 | 是 | train_BOOK_s0.log | 50 | 无（被 s2 覆盖） | 否 |
| 1 | 是 | train_BOOK_s1.log | 50 | 无（被 s2 覆盖） | 否 |
| 2 | 是 | train_BOOK_s2.log | 50 | `tmp/book/RL-agent/*-epoch-50.pkl` | 否* |

### MOVIE (`movie`)

| seed | Phase B DONE | 权威日志 | best epoch | 磁盘 checkpoint | eval-only |
|------|-------------|---------|------------|----------------|-----------|
| 0 | 是 | train_MOVIE_s0.log（674 行，正常 DONE） | 50 | 无（被 s2 覆盖） | 否 |
| 1 | **否** | train_MOVIE_s1.log（5391 行） | — | **无**（日志内无 `RL policy model saved`） | 否 |
| 2 | 是 | train_MOVIE_s2.log | 50 | `tmp/movie/RL-agent/*-epoch-50.pkl` | 否* |

**MOVIE s1 说明：** 日志在 epoch-40 的 valid eval 中途（约 2100/2500 user，84%）即出现 `=== DONE` 行，属于 shell 在 Python 异常退出后仍打印的 DONE，**不是**完整 Phase B。当前 screen **不会**续跑 s1；需等 Yelp 流水线结束后**单独重跑 MOVIE seed=1**。

### YELP_STAR (`yelp_star`)

| seed | Phase B DONE | 权威日志 | 进度 | 磁盘 checkpoint | eval-only |
|------|-------------|---------|------|----------------|-----------|
| 0 | **运行中** | train_YELP_STAR_s0.log（自 2026-05-24） | 已完成 epoch-30 eval 并保存；正在向 epoch-40 训练 | epoch-10/20/30 | 否 |
| 1 | 未开始 | — | 排队在 s0 之后 | — | 否 |
| 2 | 未开始 | — | 排队在 s1 之后 | — | 否 |

**注意：** `tmp/yelp_star/RL-agent/*-epoch-50.pkl` 时间戳为 2026-05-16，来自旧跑次，**不能**作为当前权威结果。

---

## 关键风险：checkpoint 按 seed 覆盖

RL checkpoint 路径不含 seed：`tmp/<slug>/RL-agent/train-data-{DATA}-...-epoch-{N}.pkl`  
同一数据集跑多个 seed 时，**后跑 seed 会覆盖先跑 seed 的所有 epoch 文件**。

当前磁盘上保留的是每个数据集**最后一个完成 seed** 的 policy（BOOK/LAST_FM/MOVIE → seed 2；Yelp → 仅部分 s0 进度 + 旧 epoch-50）。

若主表需要 3 个 seed 的独立 RL policy，必须在每个 seed 完成后**归档** checkpoint，或对缺失 seed **重跑 Phase B**（不必重跑 Phase-A）。

---

## Phase-A 工件（CPU 上已存在，2026-05-16）

所有四个数据集在 CPU 上均有：

- `tmp/<slug>/dataset.pkl`
- `tmp/<slug>/kg.pkl`
- `tmp/<slug>/embeds/transe.pkl`
- `data/<slug>/UI_Interaction_data/review_dict_{train,valid,test}.json`

`evaluate.py` 存在于 MCMIPL 根目录；**不存在** `scripts/eval_mcmipl_on_interrec_protocol.py`，且 `valid_catalog_mode kg_item_all` 未实现。

---

## 相关文件

- CSV: `mcmipl_phase_b_cpu_readiness.csv`
- JSON: `mcmipl_phase_b_cpu_readiness.json`
- 设备决策: `MCMIPL_EVAL_ONLY_DEVICE_DECISION.md`
- 命令模板: `MCMIPL_READY_EVAL_ONLY_COMMANDS.md`
