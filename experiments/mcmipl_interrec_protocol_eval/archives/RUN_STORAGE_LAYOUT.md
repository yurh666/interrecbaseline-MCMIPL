# Phase B 按 seed 存储约定（勿混用）

**更新：** 2026-05-27

## 全串行（单 RL 进程，恢复 ~2s/it 量级）

| 阶段 | Screen | 任务 |
|------|--------|------|
| 1 | `mcmipl_phase_b_cpu` | YELP s0（进行中，勿打断） |
| 2 | `mcmipl_watch_yelp_s0` | 等 s0 DONE → 归档 → 启动全串行 |
| 3 | `mcmipl_phase_b_serial` | BOOK 0,1 → LAST_FM 0,1 → MOVIE 0,1 → YELP 1,2 |

顺序与 `scripts/mcmipl_phase_b_full_serial.sh` 一致；已归档 seed 自动跳过。

## 目录结构

```
archives/
  checkpoints/<slug>/seed_<N>/     # 该 seed 最新 canonical 副本
    RL-agent/*.pkl
    RL-log-merge/                   # 该次训练结束时的 Evaluate 快照
    train_<DATASET>_s<N>.log
    MANIFEST.txt                    # dataset, seed, run_id, 时间戳
  runs/<slug>/seed_<N>/<run_id>/   # 单次归档不可变快照（防弄混）
    LATEST -> .../最新 run_id/
  run_registry.jsonl               # 每次归档追加一行审计
  phase_a/<slug>/                  # dataset.pkl / kg.pkl / transe.pkl
```

## run_id 命名

- 三数据集 lane：`book_s0_3ds_20260527_HHMMSS`
- Yelp handoff s0：`yelp_star_s0_handoff_...`
- Yelp s1/s2：`yelp_star_s1_yelp_...`

## 判定「已归档可跳过」

`bash main_table_experiments/scripts/mcmipl_seed_archive_status.sh <DATASET> <SEED>`

需同时存在：`MANIFEST.txt` + `RL-agent/*-epoch-50.pkl`（或 epoch-100）。

## 常用命令

```bash
# 启动三数据集 screen
bash main_table_experiments/scripts/mcmipl_start_screen_three_datasets_lane.sh

screen -ls
tail -f main_table_experiments/logs/lane_three_datasets_*.log
cat experiments/mcmipl_interrec_protocol_eval/archives/run_registry.jsonl
```
