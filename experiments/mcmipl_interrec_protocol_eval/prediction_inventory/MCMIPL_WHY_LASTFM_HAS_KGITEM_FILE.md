# 为什么 LastFM 有 mcmipl_predictions_kgitem_full.jsonl，而 Book/Movie/Yelp 不一定有？

**扫描时间：** 2026-05-27

---

## 直接回答

**LastFM 的 `mcmipl_predictions_kgitem_full.jsonl` 是后来在 GPU/MCMIPL 原环境上，针对 LastFM 单独执行了一次 KGItemEval eval-only export，再拷贝到 CPU（`FROM GPU/` → `transfer_to_cpu_kgitem_eval/`）才出现的。**

**Book / Movie / Yelp 若未执行同样的 export，就不会自动拥有该文件。** Phase B 训练、`RL-log-merge/Evaluate-*.txt`、以及从 log 提取的 CRS（SR@10、AvgT）**均不会**生成 InterRec 所需的 per-session Top-K predictions。

---

## 证据链

### 1. 文件仅覆盖 LastFM

- `dataset` 字段：**全部为 `last_fm_star`**
- 行数：**500**（与 InterRec test sessions 门控一致）
- `seed`：**1**（对应 GPU export 所用 checkpoint）

### 2. GPU transfer 文档

`interrecbaseline-TAIRA/FROM GPU/README.md`（2026-05-26）写明：

- Protocol: **MCMIPL-KGItemEval**
- **No retrain**, **No Phase-A**, **No API** — eval-only export from RL checkpoint **epoch 100**
- 产物：`mcmipl_predictions_kgitem_full.jsonl` + filtering audit + export report

`MCMIPL_KGITEM_FULL_EXPORT_REPORT.md` 记录 GPU 源：

```
/root/interrecbaseline-MCMIPL/.../kgitem_eval_full/mcmipl_predictions_kgitem_full.jsonl
```

### 3. 路径与时间

| 副本 | mtime（约） |
|------|------------|
| `FROM GPU/mcmipl_predictions_kgitem_full.jsonl` | 2026-05-26 22:51 |
| `transfer_to_cpu_kgitem_eval/...`（TAIRA） | 2026-05-26 22:51（MD5 相同） |

### 4. MCMIPL 仓库无该文件

用户常引用的路径：

`interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/`

**在 CPU 的 MCMIPL 仓库中不存在** — 说明拷贝落在 **TAIRA** 侧，或未同步回 MCMIPL 树。

### 5. 其它数据集计划文档已标明缺失

`interrecbaseline-TAIRA/.../MCMIPL_NEXT_DATASET_EVAL_PLAN.md`：

- Movie / Yelp / Book：**KGItemEval = missing**
- 推荐导出命名：`mcmipl_predictions_kgitem_book.jsonl` 等 — **尚未落盘**

---

## 与训练/checkpoint 的关系

| 现象 | 解释 |
|------|------|
| Book/Movie 有 CRS 日志、甚至有 `tmp/*/RL-agent/*.pkl` | 训练曾跑过，但 **export 是独立步骤** |
| LastFM CPU 有 epoch-100 pkl | 满足 export 条件；且 **已在 GPU 完成 export** |
| Book/Movie 最高 epoch-50（CPU tmp） | 即使本地有权重，也 **没有** 对应的 predictions jsonl |
| Yelp s0 仍在训练 | 更不可能已有 KGItemEval export |

---

## 结论表

| 问题 | 答案 |
|------|------|
| LastFM 文件是否来自 GPU 专门 export？ | **是** |
| 是否存在 `FROM GPU/`？ | **是**（在 TAIRA 仓库） |
| 其它数据集是否经过同样 export？ | **否**（无文件、无 report） |
| 其它数据集 predictions 是否可能在 GPU 未拷回？ | **可能** — 需在 GPU 侧检查（见 `MCMIPL_GPU_SIDE_PREDICTION_CHECK_PROMPT.md`） |
| Phase B 日志能否替代？ | **不能**（aggregate_only） |
