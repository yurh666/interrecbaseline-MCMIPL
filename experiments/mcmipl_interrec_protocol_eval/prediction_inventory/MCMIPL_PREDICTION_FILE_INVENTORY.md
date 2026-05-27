# MCMIPL Prediction File Inventory

**扫描时间：** 2026-05-27  
**扫描范围：** `/home/yrh666/interrecbaseline-MCMIPL` + `/home/yrh666/interrecbaseline-TAIRA`  
**模式：** 只读关键词/文件名全盘搜索（59 个候选文件）

---

## 执行摘要

| 数据集 | per-session predictions.jsonl | 位置 |
|--------|------------------------------|------|
| **LastFM** | **有（500 行）** | 仅在 **TAIRA** 仓库，不在 MCMIPL 仓库 |
| **Book** | **无** | — |
| **Movie** | **无** | — |
| **Yelp** | **无** | — |

**MCMIPL 仓库内 jsonl predictions 文件数：0**

---

## LastFM 命中（usable = yes）

| 文件 | 行数 | dataset | protocol |
|------|------|---------|----------|
| `interrecbaseline-TAIRA/.../transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl` | 500 | last_fm_star | MCMIPL-KGItemEval |
| `interrecbaseline-TAIRA/FROM GPU/mcmipl_predictions_kgitem_full.jsonl` | 500 | 同上（MD5 相同） | 同上 |
| `interrecbaseline-TAIRA/.../kg_item_eval/mcmipl_predictions_kgitem_eval.jsonl` | 500 | 同上 | 同上 |

---

## Book / Movie / Yelp

- **未发现** `mcmipl_predictions_kgitem_book.jsonl`、`mcmipl_predictions_kgitem_movie.jsonl`、`mcmipl_predictions_kgitem_yelp*.jsonl`
- **未发现** 任何含 `session_id` + `kgitem_top10_ids` 的其它 jsonl

### 仅有聚合/训练产物（不可用算 InterRec ID 指标）

| 类型 | 示例路径 |
|------|----------|
| Evaluate 聚合 | `archives/checkpoints/*/seed_*/RL-log-merge/Evaluate-epoch-*.txt` |
| 训练日志 CRS | `main_table_experiments/logs/train_{BOOK,MOVIE,YELP,LAST_FM}_*.log` |
| CRS 汇总 JSON | `archives/crs_metrics_from_logs.json` |
| KGItem 聚合指标（仅 LastFM） | `interrecbaseline-TAIRA/.../kg_item_eval/mcmipl_kgitem_eval_metrics.*` |

---

## MCMIPL 仓库 checkpoint 现状（2026-05-27 扫描，非 predictions）

`main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/<slug>/`：

| slug | dataset.pkl | transe.pkl | RL-agent 最高 epoch |
|------|-------------|------------|---------------------|
| last_fm_star | 有 | 有 | **100** |
| book | 有 | 有 | **50** |
| movie | 有 | 有 | **50** |
| yelp_star | 有 | 有 | **50**（s0 训练进行中） |

> 有 checkpoint **≠** 有 predictions；导出需单独 eval-only KGItemEval 流水线。

---

## 结构化输出

- CSV：`mcmipl_prediction_file_inventory.csv`
- JSON：`mcmipl_prediction_file_inventory.json`

---

## 目录优先级说明

已按用户指定优先级搜索：`experiments/mcmipl_interrec_protocol_eval/`、`main_table_experiments/`、`FROM GPU/`、`transfer_to_cpu*/`、`kgitem_eval*/`、`RL-log-merge/`、`logs/`、`tmp/` 等。
