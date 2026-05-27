# MCMIPL Per-Session Top-K Usability Audit

**扫描时间：** 2026-05-27T11:49:24.784674

## 数据集结论

| 数据集 | lastfm/book/movie/yelp_has_usable_predictions | 说明 |
|--------|-----------------------------------------------|------|
| LastFM | **lastfm_has_usable_predictions = yes** | 仅 TAIRA 有 500 行 `mcmipl_predictions_kgitem_full.jsonl` |
| Book | **book_has_usable_predictions = no** | 仅有 CRS 聚合日志与 epoch-50 checkpoint |
| Movie | **movie_has_usable_predictions = no** | 同上 |
| Yelp | **yelp_has_usable_predictions = no** | 无 predictions；Phase B s0 仍在训练 |

## 可用文件标准

每行须含：`session_id` + `future_test_item_ids`（或等价 target）+ `kgitem_top5/10/20_ids`（或 ranked item list）。

## 已判定可用（MCMIPL baseline）

1. `/home/yrh666/interrecbaseline-TAIRA/experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl`
2. `/home/yrh666/interrecbaseline-TAIRA/FROM GPU/mcmipl_predictions_kgitem_full.jsonl`（与上相同）
3. `/home/yrh666/interrecbaseline-TAIRA/experiments/mcmipl_interrec_protocol_eval/kg_item_eval/mcmipl_predictions_kgitem_eval.jsonl`（与上相同）

## 不可用类型（本次扫描命中）

- `Evaluate-epoch-*.txt` / `Evaluate-train-*.txt`：**aggregate_only_not_enough_for_interrec_metrics**
- `train_*.log` 中 `SR5/SR10/AvgT` 行：聚合 CRS，**无** per-session item id 列表
- `mcmipl_kgitem_*_metrics.csv/json`：已算好的聚合指标，**非** predictions
- `crs_metrics_from_logs.json`：从 log 提取的 SR10/AvgT

## 非 MCMIPL 但命中关键词

- `interrecbaseline-TAIRA/experiments/baseline_interrec_protocol_eval/taira/taira_interrec_predictions.jsonl` — TAIRA 方法，非 MCMIPL
