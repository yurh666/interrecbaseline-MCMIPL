# MCMIPL LastFM Prediction File Audit

**扫描时间：** 2026-05-27T11:49:24.784674  
**扫描模式：** 只读（未修改任何 predictions）

---

## 1. 用户指定路径（MCMIPL 仓库）

| 项 | 值 |
|----|-----|
| 路径 | `/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl` |
| 是否存在 | **否** |

> **结论：** 在 `interrecbaseline-MCMIPL` 仓库下，**不存在** `transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl`。LastFM 可用文件位于 **`interrecbaseline-TAIRA`**（见下文）。

---

## 2. 实际可用文件（TAIRA / FROM GPU）

### 2.1 `transfer_to_cpu_kgitem_eval`（推荐主副本）

| 项 | 值 |
|----|-----|
| 绝对路径 | `/home/yrh666/interrecbaseline-TAIRA/experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl` |
| 存在 | True |
| 大小 | 1496442 bytes (~1.43 MB) |
| 行数 | **500** |
| mtime | 2026-05-26T22:51:46.113105 |
| ctime | 2026-05-26T22:53:53.504400 |
| MD5 | `54ae50d2ef34244c879e68db38f86ea1` |

### 2.2 `FROM GPU/`（GPU 拷贝原件）

| 项 | 值 |
|----|-----|
| 绝对路径 | `/home/yrh666/interrecbaseline-TAIRA/FROM GPU/mcmipl_predictions_kgitem_full.jsonl` |
| 与 transfer 副本 | **字节级相同**（MD5 一致） |
| mtime | 2026-05-26T22:51:46.113105 |

### 2.3 `kg_item_eval/mcmipl_predictions_kgitem_eval.jsonl`

与 `transfer_to_cpu` 副本 **MD5 相同**（同一 LastFM 500 行导出）。

---

## 3. 字段审计（前 2 行）

**第 1 行字段（节选）：** `session_id`, `future_test_item_ids`, `kgitem_top5_ids`, `kgitem_top10_ids`, `kgitem_top20_ids`, `mcmipl_raw_ranked_preview_ids`, `protocol`, `dataset`, `seed`, …

| 字段 | 存在 |
|------|------|
| session_id | True |
| future_test_item_ids | True |
| kgitem_top5_ids | True |
| kgitem_top10_ids | True |
| kgitem_top20_ids | True |
| mcmipl_raw_ranked_preview_ids | True |

- **protocol（前 500 行）：** 全部为 `MCMIPL-KGItemEval`
- **dataset：** `['last_fm_star']`（仅 LastFM）
- **seed：** `[1]`
- **示例 session_id：** `last_fm_star_505_1`

---

## 4. GPU 来源线索

| 证据文件 | 说明 |
|----------|------|
| `interrecbaseline-TAIRA/FROM GPU/README.md` | 标明 **eval-only export from RL checkpoint epoch 100**，无重训 |
| `interrecbaseline-TAIRA/FROM GPU/MCMIPL_KGITEM_FULL_EXPORT_REPORT.md` | GPU 源路径：`/root/interrecbaseline-MCMIPL/.../kgitem_eval_full/mcmipl_predictions_kgitem_full.jsonl` |
| 生成时间 | 2026-05-26T22:44:39（README / export report） |

**判断：** LastFM 文件是 **GPU 上专门执行 KGItemEval eval-only export** 后，拷贝到 `FROM GPU/`，再落到 TAIRA 的 `transfer_to_cpu_kgitem_eval/`。**不是** Phase B 训练日志自动产物。

---

## 5. InterRec 指标可用性

| 判定 | 说明 |
|------|------|
| **usable_for_interrec_metrics** | **yes**（500 sessions，逐 session Top-K + future_test） |
| CPU 计算脚本位置 | `interrecbaseline-TAIRA/scripts/evaluate_mcmipl_kgitem_protocol_metrics.py` |

---

## 6. 建议（只读结论）

1. 若要在 **MCMIPL 仓库** 下统一路径，可将 TAIRA 中已验证文件 **只读复制或软链** 到 `experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/`（本次审计未执行）。
2. Book/Movie/Yelp **无** 同类 `mcmipl_predictions_kgitem_*.jsonl`（见 `MCMIPL_PREDICTION_FILE_INVENTORY.md`）。
