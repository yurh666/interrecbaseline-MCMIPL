# MCMIPL Log Top-K Parse Audit

**扫描时间：** 2026-05-27  
**扫描日志数：** 141（`.log` / `.txt`，每文件前 12KB 抽样）

---

## 结论

| 项 | 结果 |
|----|------|
| **log_contains_per_session_topk** | **no**（141/141） |
| **parse_feasible** | **no**（0/141） |
| 主导模式 | **aggregate_only_not_enough_for_interrec_metrics** |

**无法**从现有 MCMIPL 训练/评估日志反算 InterRec-style HitRate/NDCG/MRR/Recall。

---

## 检查路径

- `interrecbaseline-MCMIPL/main_table_experiments/logs/`
- `interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/archives/`（含 `logs_authoritative/`、`checkpoints/*/RL-log-merge/`）
- `interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/*/RL-log-merge/`

---

## 日志内容特征

### 训练日志 `train_*.log`

典型行（聚合 CRS）：

```
SR5:0.43, SR10:0.57, SR15:0.66, AvgT:8.16, Rank:0.376..., rewards:... Total epoch_uesr:500
```

- 含 `SR5/SR10/SR15/AvgT/Rank`：**是**
- 含 `sorted_actions` 或 item id 数组：**否**（12KB 样本 + 全文件 grep 无匹配）

### `Evaluate-epoch-*.txt`

列为 epoch 级聚合 SR/Rank（无 session_id、无 item 列表）。

示例（`Evaluate-train-data-LAST_FM_STAR-...txt`）：

```
1	0.834	7.526	0.340	0.201
...
100	0.832	7.364	0.336	0.241
```

### 代码侧说明

`sorted_actions` 存在于 `agent.py` / `RL_evaluate.py` **运行时变量**，但 **未写入** 标准 train/Evaluate 日志格式。

---

## 按数据集

| 数据集 | 日志 | per-session Top-K 可解析 |
|--------|------|-------------------------|
| LastFM | `train_LAST_FM_STAR_s{0,1,2}.log` + Evaluate | **no** |
| Book | `train_BOOK_s*.log` + Evaluate | **no** |
| Movie | `train_MOVIE_s*.log` + Evaluate | **no** |
| Yelp | `train_YELP_STAR_s0.log`（进行中）+ Evaluate | **no** |

---

## CSV

`mcmipl_log_topk_parse_candidates.csv` — 每条日志的 `log_contains_per_session_topk` / `parse_feasible` / `reason`。

---

## 含义

仅有聚合 SR@10 / AvgT / `Evaluate-*.txt` **不够** 计算 InterRec 协议指标；必须依赖 **KGItemEval export 的 predictions.jsonl**（目前仅 LastFM 在 TAIRA 有）。
