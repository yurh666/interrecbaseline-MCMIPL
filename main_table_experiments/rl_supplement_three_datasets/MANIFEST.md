# RL supplement manifest — BOOK / LAST_FM / MOVIE seed 0,1

**Updated:** 2026-05-27

## 目标

| 数据集 | 待补 seed | 已归档 seed（参考） |
|--------|-----------|---------------------|
| BOOK | 0, 1 | 2 → `archives/checkpoints/book/seed_2/` |
| LAST_FM_STAR | 0, 1 | 2 → `archives/checkpoints/last_fm_star/seed_2/` |
| MOVIE | 0, 1 | 2 → `archives/checkpoints/movie/seed_2/` |

## Phase-A 工件（新服务器 `tmp/<slug>/` 必需）

从 git 内归档恢复（`restore_phase_a_to_tmp.sh`）：

| slug | 文件 |
|------|------|
| book | `dataset.pkl`, `kg.pkl`, `embeds/transe.pkl` |
| last_fm_star | 同上 |
| movie | 同上 |

路径：`experiments/mcmipl_interrec_protocol_eval/archives/phase_a/<slug>/`

## RL 输出（每 seed 跑完立即归档）

`archives/checkpoints/<slug>/seed_<N>/`：

- `RL-agent/seed_<N>/*-epoch-*.pkl`（训练时写入 `tmp/<slug>/RL-agent/seed_<N>/`）
- `RL-log-merge/` 快照
- `train_<DATASET>_s<N>.log`
- `MANIFEST.txt`, `TIMING.txt`
- `archives/run_registry.jsonl` 追加一行

## 不在此包内

- **YELP_STAR**：当前 CPU 机 `mcmipl_phase_b_cpu` + `mcmipl_lane_YELP_STAR`（s0→s1→s2）
- **tmp 下正在写入的 pkl**：不提交 git；仅提交 `archives/` 与脚本

## 校验命令

```bash
for ds in BOOK LAST_FM_STAR MOVIE; do
  for s in 0 1; do
    bash main_table_experiments/scripts/mcmipl_seed_archive_status.sh "$ds" "$s" && echo "OK $ds s$s" || echo "NEED $ds s$s"
  done
done
```
