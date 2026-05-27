# GPU / MCMIPL 原环境 — Prediction 只读检查清单

**用途：** 在 CPU 未找到 Book/Movie/Yelp 的 per-session predictions 时，到 **GPU 或 `/root/interrecbaseline-MCMIPL` 原环境** 做只读排查。  
**禁止：** 重训、改 predictions、覆盖已有结果（除非明确要重新 export 且备份旧文件）。

---

## 1. 检查是否已有 KGItemEval 导出（未拷回 CPU）

在 GPU 项目根（示例 `/root/interrecbaseline-MCMIPL`）搜索：

```bash
find . -name 'mcmipl_predictions*.jsonl' -o -name '*kgitem*full*.jsonl' 2>/dev/null
```

重点目录：

- `experiments/mcmipl_interrec_protocol_eval/kgitem_eval_full/`
- `experiments/mcmipl_interrec_protocol_eval/kg_item_eval/`
- `experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/`
- `transfer_to_cpu_kgitem_eval_book/` / `*_movie/` / `*_yelp*`（若存在）
- 任意 `FROM GPU/` 或手工拷贝目录

期望文件名（参考 LastFM 命名惯例）：

- `mcmipl_predictions_kgitem_full.jsonl`（LastFM 已存在）
- `mcmipl_predictions_kgitem_book.jsonl`
- `mcmipl_predictions_kgitem_movie.jsonl`
- `mcmipl_predictions_kgitem_yelp_star.jsonl`

---

## 2. 对每个找到的 jsonl 记录

```bash
wc -l <file>
ls -la --time-style=full-iso <file>
head -n 1 <file> | python3 -m json.tool
```

确认：

- `protocol` == `MCMIPL-KGItemEval`
- `dataset` 为 `book` / `movie` / `yelp_star` / `last_fm_star`
- 含 `session_id`, `future_test_item_ids`, `kgitem_top10_ids`（及 top5/top20）
- 行数是否等于该数据集 InterRec test session 数（LastFM 为 500）

---

## 3. 检查 export 报告

```bash
ls -la experiments/mcmipl_interrec_protocol_eval/**/MCMIPL_KGITEM*REPORT*.md
ls -la **/FROM\ GPU/README.md
```

LastFM 参考：`MCMIPL_KGITEM_FULL_EXPORT_REPORT.md`（应记录 GPU 源路径、行数、protocol）。

---

## 4. 若无 predictions — 检查 checkpoint 是否可 eval-only export

```bash
for slug in book movie yelp_star last_fm_star; do
  echo "=== $slug ==="
  ls -la main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/$slug/RL-agent/*epoch-100*.pkl 2>/dev/null
  ls -la main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/$slug/{dataset,kg}.pkl 2>/dev/null
  ls -la main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/$slug/embeds/transe.pkl 2>/dev/null
done
```

| 数据集 | 若缺 epoch-100 pkl | 下一步 |
|--------|-------------------|--------|
| Book/Movie | 有 epoch-50 无 100 | 确认 export 是否必须用 100；或从备份恢复 |
| Yelp | 无完整 Phase B | **先完成训练**，再 export |

---

## 5. 检查 export 脚本/命令痕迹

```bash
grep -r "kgitem_eval\|KGItemEval\|export_predictions" \
  main_table_experiments/baselines/mcmipl_official/MCMIPL \
  experiments/mcmipl_interrec_protocol_eval/ 2>/dev/null | head -40
```

对照 LastFM 成功 export 的命令行（README 中：**eval-only, epoch 100, no API**）。

---

## 6. 拷回 CPU 约定（找到文件后）

目标目录（建议与 LastFM 一致）：

```
experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/
  mcmipl_predictions_kgitem_<dataset>.jsonl
  mcmipl_kgitem_filtering_audit_<dataset>.csv
```

拷回后 CPU 侧再跑 **只读** 行数/字段审计（勿覆盖 TAIRA 已有 LastFM 原件）。

---

## 7. 决策树（GPU 侧）

```
已有 jsonl？
├─ 是 → 记录 mtime/行数/dataset → scp 到 CPU → CPU 算 metrics
└─ 否 → 有 epoch-100（或约定 epoch）checkpoint + Phase-A 工件？
    ├─ 是 → eval-only KGItemEval export（不重训）
    └─ 否 → 恢复 checkpoint 或等 Phase B（Yelp）→ 再 export
```

---

## 8. 回报 CPU 侧最小信息

- 各数据集：predictions 路径 / 行数 / mtime / 是否存在
- checkpoint 最高 epoch / seed 目录结构
- 是否已有 `MCMIPL_KGITEM_FULL_EXPORT_REPORT.md` 同类报告
- 建议：仅 copy / 需 export / 需训练
