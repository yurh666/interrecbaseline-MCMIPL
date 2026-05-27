# MCMIPL 归档与 Phase-B 分工（2026-05-27）

## 两台机器分工

| 机器 | 任务 | Screen |
|------|------|--------|
| **当前 CPU** | YELP_STAR **seed 0 → 1 → 2** 串行，每 seed 归档 | `mcmipl_phase_b_cpu`（s0）+ `mcmipl_watch_yelp_s0` → `mcmipl_lane_YELP_STAR`（s1,s2） |
| **新服务器（git 补跑）** | BOOK / LAST_FM / MOVIE **seed 0,1** 串行 | `mcmipl_rl_supplement_3ds`（见 `rl_supplement_three_datasets/`） |

**不再使用** `mcmipl_phase_b_full_serial`（Yelp 与三数据集混跑）。

## Yelp 流程

1. s0 在 `mcmipl_phase_b_cpu` 跑完 → log 出现 `=== DONE: YELP_STAR seed=0`
2. `mcmipl_watch_yelp_s0` 归档 s0 → 启动 `mcmipl_lane_YELP_STAR` 跑 **s1、s2**
3. 每 seed 写入 `archives/checkpoints/yelp_star/seed_<N>/`

## 三数据集补跑（git）

- 目录：`main_table_experiments/rl_supplement_three_datasets/`
- Prompt：`docs/PROMPT_NEW_SERVER_RL_SUPPLEMENT_S01.md`
- 已有 seed_2 仅作参考；补 seed_0、seed_1

## 存储约定

见 `RUN_STORAGE_LAYOUT.md`、`run_registry.jsonl`。
