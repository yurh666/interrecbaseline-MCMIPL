# MCMIPL Prediction Next Step Decision

**生成时间：** 2026-05-27  
**前提：** 本次仅做只读扫描；**未**执行 eval-only / 重训 / API

---

## 按数据集状态

### LastFM

| 项 | 值 |
|----|-----|
| **状态** | `complete_for_kgitem_metrics`（predictions 已有） |
| predictions 位置 | `interrecbaseline-TAIRA/experiments/mcmipl_interrec_protocol_eval/transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_full.jsonl` |
| CPU 下一步 | 用 `evaluate_mcmipl_kgitem_protocol_metrics.py` 算 HitRate/NDCG/MRR/Recall（若尚未写入主表） |
| 可选整理 | 只读复制/软链到 MCMIPL 仓库统一路径（非必须） |

---

### Book

| 项 | 值 |
|----|-----|
| **状态** | `need_eval_only_export_from_checkpoint_or_gpu` |
| per-session predictions | **无** |
| CPU checkpoint | `tmp/book/RL-agent/*` 至 epoch-50；`dataset.pkl`/`kg.pkl`/`transe.pkl` **有** |
| 阻塞 | 缺 epoch-100 权重（若 export 协议要求与 LastFM 相同 epoch-100）；且 **无 export 脚本落盘** 于 MCMIPL 树 |
| 推荐 | 1) GPU 侧查是否已有 `kgitem_eval_book` 导出未拷回；2) 若无，在 **有 epoch-100 的环境** eval-only export；3) 拷回 `transfer_to_cpu_kgitem_eval/mcmipl_predictions_kgitem_book.jsonl` |

---

### Movie

| 项 | 值 |
|----|-----|
| **状态** | `need_eval_only_export_from_checkpoint_or_gpu` |
| per-session predictions | **无** |
| CPU checkpoint | epoch 至 50；Phase-A 工件 **有** |
| seed 注意 | seed1 训练日志可能不完整；export 前需对齐用哪个 seed/epoch |
| 若 GPU 无 checkpoint | 升级为 `need_phase_b_checkpoint`（补训或从 GPU 恢复 epoch-100） |

---

### Yelp

| 项 | 值 |
|----|-----|
| **状态** | `wait_for_phase_b_done` **+** 之后 `need_eval_only_export` |
| per-session predictions | **无** |
| 训练 | `train_YELP_STAR_s0.log` 仍在 sampling（2026-05-27 扫描） |
| CPU checkpoint | 仅至 epoch-50；Phase B 未完成 |
| 推荐顺序 | 等 s0/s1/s2 Phase B 完成 → 归档 seed 分目录 checkpoint → 再 KGItemEval export |

---

## 决策矩阵（动作类型）

| 动作 | LastFM | Book | Movie | Yelp |
|------|--------|------|-------|------|
| Need existing predictions search | 完成 | 完成（无） | 完成（无） | 完成（无） |
| Need eval-only export | 否 | **是** | **是** | **是**（训练后） |
| Need checkpoint | 否（GPU 已 export） | 可能（要 epoch-100） | 可能 | **是**（Phase B） |
| Need Phase B training | 否 | 否（CRS 已有） | 视 seed1 | **是** |
| Need GPU 侧检查 | 否 | **是** | **是** | **是** |

**不要**因缺 predictions 而直接建议重训 — 除非 GPU/CPU 均确认 **无** 可用 epoch-100（或目标 epoch）checkpoint。

---

## 推荐执行顺序

1. **LastFM** — CPU 直接算 metrics（文件已在 TAIRA）。
2. **GPU 只读检查** — Book/Movie（及 Yelp 若 GPU 有旧权重）是否已有未拷贝的 `mcmipl_predictions_kgitem_*.jsonl`。
3. **Book eval-only export** — CRS 已齐，优先于 Movie/Yelp。
4. **Movie eval-only export** — 对齐 seed/epoch 后导出。
5. **Yelp** — 等 Phase B → 再 export。

---

## 相关报告

- `MCMIPL_GPU_SIDE_PREDICTION_CHECK_PROMPT.md` — 给 GPU 环境的检查清单
- `MCMIPL_LASTFM_PREDICTION_FILE_AUDIT.md` — LastFM 文件详情
- `phase_b_readiness/MCMIPL_READY_EVAL_ONLY_COMMANDS.md` — eval 命令模板（未执行）
