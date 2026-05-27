# MCMIPL Checkpoint 排查与归档工作报告

**生成时间：** 2026-05-27  
**环境：** CPU 主机 `/home/yrh666`  
**项目根目录：** `/home/yrh666/interrecbaseline-MCMIPL`

---

## 1. 背景与目标

为与 **InterRec 协议** 公平对比，MCMIPL 需要：

1. 完成 Phase B（RL 训练）并保留 **按 seed 独立** 的 checkpoint（`.pkl`）
2. 保留 Phase-A 工件（`dataset.pkl` / `kg.pkl` / `transe.pkl`）
3. 后续基于 checkpoint 做 eval-only，导出 `predictions.jsonl` 再算 NDCG/MRR 等

本次工作：**不重训 Phase-A、不伪造结果**，只做 readiness 审计、发现 checkpoint 覆盖问题、建立归档与并行补跑机制。

---

## 2. 发现的核心问题

### 2.1 Checkpoint 原始路径不含 seed

官方保存逻辑（修改前）：

```
tmp/<slug>/RL-agent/train-data-{DATA}-RL-...-epoch-{N}.pkl
```

**文件名与目录均不含 seed。** 同一数据集连续跑 seed 0 → 1 → 2 时，后跑 seed **覆盖** 先跑 seed 的所有 epoch 文件。

**实际后果（2026-05-16 起权威日志）：**

| 数据集 | 日志上 DONE 的 seeds | 磁盘曾保留的 checkpoint |
|--------|---------------------|------------------------|
| BOOK | s0, s1, s2 | 仅 **s2**（最后写入） |
| LAST_FM | s0, s1, s2 | 仅 **s2** |
| MOVIE | s0, s2（s1 失败） | 仅 **s2** |
| YELP | s0 运行中 | s0 部分 epoch + 5/16 旧 epoch-50 残留 |

### 2.2 与 InterRec 的关系

- **CRS 口径**（SR10_CRS、AvgT）：可从训练 **log** 提取，**不强制** checkpoint
- **InterRec ID 协议**（NDCG@10、MRR@10、per-session Top-K）：**必须** checkpoint + 导出 eval，**无法**从现有 log 反算

### 2.3 其他缺口

- `scripts/eval_mcmipl_on_interrec_protocol.py` 尚未部署
- `--valid_catalog_mode kg_item_all` 未在 `evaluate.py` 实现
- CPU 模式用 `MCMIPL_FORCE_CPU=1`，**不是** `--gpu -1`

---

## 3. 已完成的工作

### 3.1 Phase B Readiness 审计（2026-05-27）

产出目录：`experiments/mcmipl_interrec_protocol_eval/phase_b_readiness/`

| 文件 | 说明 |
|------|------|
| `MCMIPL_PHASE_B_CPU_READINESS.md` | Phase B 完成度人类可读报告 |
| `mcmipl_phase_b_cpu_readiness.csv` / `.json` | 结构化审计数据 |
| `MCMIPL_EVAL_ONLY_CPU_REQUIREMENTS.md` | Phase-A + eval 脚本前置条件 |
| `mcmipl_eval_only_cpu_requirements.csv` | 同上 CSV |
| `MCMIPL_EVAL_ONLY_DEVICE_DECISION.md` | CPU/GPU eval 决策 |
| `MCMIPL_READY_EVAL_ONLY_COMMANDS.md` | eval 命令模板（未执行） |

### 3.2 一次性归档（无需重跑的信息）

**归档根目录：**

`/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/`

| 子目录 | 内容 |
|--------|------|
| `phase_a/<slug>/` | dataset.pkl, kg.pkl, transe.pkl, review_dict_*.json |
| `checkpoints/<slug>/seed_<N>/` | RL-agent 副本 + MANIFEST.txt |
| `logs_authoritative/` | 12 份权威 train 日志副本 |
| `crs_metrics_from_logs.json` | 从 log 提取的 SR10/AvgT（CRS，非 InterRec ID） |

**首次归档时已保存的 checkpoint：**

- `archives/checkpoints/book/seed_2/`
- `archives/checkpoints/last_fm_star/seed_2/`
- `archives/checkpoints/movie/seed_2/`
- `archives/checkpoints/yelp_star/seed_0/`（s0 进行中时的快照；s0 真正 DONE 后 watcher 会再归档一次）

### 3.3 代码与脚本改动（解决 seed 覆盖）

#### 3.3.1 Checkpoint 按 seed 分目录（2026-05-27）

**修改文件：**

- `main_table_experiments/baselines/mcmipl_official/MCMIPL/utils.py`  
  - 新增 `_rl_agent_dir()` / `_rl_agent_model_path()`  
  - 当环境变量 `MCMIPL_RL_SEED` 设置时，保存/加载路径为：  
    `tmp/<slug>/RL-agent/seed_<N>/*-epoch-*.pkl`  
  - 加载时若无 seed 子目录，回退旧 flat `RL-agent/`（兼容 Yelp s0 旧进程）

- `main_table_experiments/run_mcmipl.sh`  
  - `export MCMIPL_RL_SEED="$SEED"`  
  - 日志头打印 checkpoint 目标路径

#### 3.3.2 自动化脚本

目录：`main_table_experiments/scripts/`

| 脚本 | 作用 |
|------|------|
| `mcmipl_archive_seed.sh` | 复制指定 dataset/seed 的 checkpoint 到 archives |
| `mcmipl_archive_phase_a.sh` | 归档 Phase-A 工件 |
| `mcmipl_archive_all_existing.sh` | 一次性全量归档 |
| `mcmipl_extract_crs_from_logs.py` | 从 log 提取 CRS 指标 |
| `mcmipl_run_seed_with_archive.sh` | 跑一个 seed + 完成后自动 archive |
| `mcmipl_lane_sequential.sh` | 单数据集多 seed 串行（每 seed 自动 archive） |
| `mcmipl_launch_other_datasets_now.sh` | 立即启动 BOOK/LAST_FM/MOVIE 并行 lane |
| `mcmipl_watch_yelp_s0_handoff.sh` | 等 YELP s0 DONE → 归档 → 停旧 screen → 启 s1/s2 |
| `mcmipl_phase_b_parallel_recovery.sh` | 四数据集并行恢复（备用） |

操作说明：`archives/ARCHIVE_AND_PARALLEL_PLAN.md`

---

## 4. 当前运行状态（2026-05-27 ~11:35）

### 4.1 Screen 会话

| screen 名 | 任务 |
|-----------|------|
| `mcmipl_phase_b_cpu` | **YELP s0**（5/22 起，未打断） |
| `mcmipl_lane_BOOK` | BOOK **s0 → s1** |
| `mcmipl_lane_LAST_FM_STAR` | LAST_FM **s0 → s1** |
| `mcmipl_lane_MOVIE` | MOVIE **s0 → s1** |
| `mcmipl_watch_yelp_s0` | 监控 s0 DONE，随后归档并启动 YELP s1→s2 |

### 4.2 并行策略

- **4 个 RL 进程同时跑**（Yelp s0 + 另外 3 数据集 seed 0）
- 不同数据集写不同 `tmp/<slug>/`，互不覆盖
- 同一数据集内 seed **串行**，且新 checkpoint 写入 `RL-agent/seed_<N>/`
- 每个 seed 完成后 **自动 copy** 到 `archives/checkpoints/`

### 4.3 仍缺、待补跑的 checkpoint（InterRec 三 seed 主表）

| 数据集 | 已有归档 | 正在跑 | 待跑（watcher/ lane 自动） |
|--------|---------|--------|---------------------------|
| BOOK | seed 2 | seed 0 | seed 1 |
| LAST_FM | seed 2 | seed 0 | seed 1 |
| MOVIE | seed 2 | seed 0 | seed 1 |
| YELP | seed 0 快照 | seed 0 | s0 终归档 + seed 1, 2 |

**说明：** seed 2 对四数据集均已在首次归档中保存；BOOK/LAST_FM/MOVIE 的 seed 2 **不需要重跑**（除非你要统一用新 seed 子目录格式再训一遍，当前不必）。

---

## 5. Checkpoint 路径速查

### 5.1 训练时活跃路径（新格式，2026-05-27 起）

```
/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/book/RL-agent/seed_0/
/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/last_fm_star/RL-agent/seed_0/
/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/movie/RL-agent/seed_0/
/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/yelp_star/RL-agent/          ← s0 旧进程仍写 flat
```

（seed 1 完成后为 `seed_1/`，以此类推；首个 pkl 在 **epoch 10** 保存时出现。）

### 5.2 归档副本（长期保存，eval 前可拷回）

```
/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/
```

### 5.3 Phase-A 归档

```
/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/phase_a/
```

---

## 6. 当前四个数据集 — 可直接打开的 Log 文件

**日志目录：**  
`/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/`

### 6.1 正在写入的 log（点这些看实时进度）

| 数据集 | 当前跑的是 | Log 文件（绝对路径） |
|--------|-----------|---------------------|
| **YELP** | seed 0 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s0.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s0.log) |
| **BOOK** | seed 0（完成后同 lane 写 s1） | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_BOOK_s0.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_BOOK_s0.log) |
| **LAST_FM** | seed 0 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_LAST_FM_STAR_s0.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_LAST_FM_STAR_s0.log) |
| **MOVIE** | seed 0 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_MOVIE_s0.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_MOVIE_s0.log) |

> **注意：** 每个 seed 有独立 log。lane 跑到 seed 1 时会 **重写/追加** 对应 `train_*_s1.log`（`run_mcmipl.sh` 用 `tee` 写入该文件）。

### 6.2 后续会自动产生的 log

| 数据集 | 预计 log | 触发条件 |
|--------|---------|----------|
| YELP seed 1 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s1.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s1.log) | s0 DONE 后 watcher 启动 |
| YELP seed 2 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s2.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_YELP_STAR_s2.log) | s1 完成后 |
| BOOK seed 1 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_BOOK_s1.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_BOOK_s1.log) | s0 完成后同 lane |
| LAST_FM seed 1 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_LAST_FM_STAR_s1.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_LAST_FM_STAR_s1.log) | 同上 |
| MOVIE seed 1 | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_MOVIE_s1.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_MOVIE_s1.log) | 同上 |

### 6.3 辅助 / 监控 log

| 用途 | 文件 |
|------|------|
| Yelp s0 交接 watcher | [/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/watch_yelp_s0_handoff.log](/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/watch_yelp_s0_handoff.log) |
| Lane screen 汇总输出 | `/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/screen_mcmipl_lane_*.log` |
| 单 lane 流水 | `/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/lane_*.log` |

### 6.4 历史权威 log 副本（只读备份，不会随新跑更新）

目录：[/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/logs_authoritative/](/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/logs_authoritative/)

含 2026-05-27 补跑 **之前** 的 12 份 `train_*_s*.log` 快照。

---

## 7. 如何确认 checkpoint 已正确按 seed 保存

```bash
# 新格式（BOOK s0 跑到 epoch 10 后应出现）
ls /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/book/RL-agent/seed_0/

# 归档（每个 seed DONE 后自动写入）
ls /home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/book/seed_0/

# log 里应出现类似：
# RL policy model saved at ./tmp/book/RL-agent/seed_0/train-data-BOOK-RL-...-epoch-10.pkl
grep 'RL policy model saved' /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/logs/train_BOOK_s0.log
```

---

## 8. Eval 时使用归档 checkpoint（后续 InterRec）

```bash
SLUG=book
SEED=0
MCMIPL=/home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL
ARCH=/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/${SLUG}/seed_${SEED}

mkdir -p "${MCMIPL}/tmp/${SLUG}/RL-agent/seed_${SEED}"
cp -a "${ARCH}/RL-agent/." "${MCMIPL}/tmp/${SLUG}/RL-agent/seed_${SEED}/"
export MCMIPL_RL_SEED=${SEED}
# 再跑 evaluate.py 或未来的 eval_mcmipl_on_interrec_protocol.py
```

---

## 9. 下一步（checkpoint 齐了之后）

1. 等全部 lane + YELP s1/s2 完成，`archives/checkpoints/` 下应有各 slug 的 seed 0/1/2（BOOK/LAST_FM/MOVIE 的 seed 2 已有）
2. 实现/部署 InterRec eval 脚本，导出 `predictions.jsonl`
3. 用 TAIRA 侧 `ranking_metrics` 或等价脚本算 ID 协议主表
4. 若 GPU Phase-A 与 CPU 不一致，eval 前对齐 `phase_a/` 工件

---

## 10. 相关文档索引

| 文档 | 路径 |
|------|------|
| 本报告 | `experiments/mcmipl_interrec_protocol_eval/MCMIPL_CHECKPOINT_WORK_REPORT.md` |
| 归档与并行计划 | `experiments/mcmipl_interrec_protocol_eval/archives/ARCHIVE_AND_PARALLEL_PLAN.md` |
| Phase B readiness | `experiments/mcmipl_interrec_protocol_eval/phase_b_readiness/` |
| MCMIPL 主表报告（TAIRA 侧） | `interrecbaseline-TAIRA/experiments/baseline_final_comparison/MCMIPL/MCMIPL_FINAL_REPORT.md` |
