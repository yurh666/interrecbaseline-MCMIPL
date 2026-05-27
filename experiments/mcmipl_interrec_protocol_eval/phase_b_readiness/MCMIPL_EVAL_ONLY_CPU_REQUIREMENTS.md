# MCMIPL Eval-Only CPU 前置条件检查

**生成时间：** 2026-05-27  
**MCMIPL 根目录：** `main_table_experiments/baselines/mcmipl_official/MCMIPL`

---

## 全局脚本 / 参数支持

| 检查项 | CPU 状态 | 说明 |
|--------|---------|------|
| `scripts/eval_mcmipl_on_interrec_protocol.py` | **缺失** | 本仓库与 TAIRA 均无此文件；TAIRA 仅有离线指标脚本 `evaluate_mcmipl_kgitem_protocol_metrics.py`（吃 jsonl，不生成预测） |
| `evaluate.py` | **存在** | 官方 MCMIPL 评测入口 |
| `--valid_catalog_mode kg_item_all` | **未实现** | `evaluate.py` / `RL_model.py` 中无此参数 |
| CPU 训练/推理 | **支持** | 通过 `export MCMIPL_FORCE_CPU=1`（`RL_model.py` 强制 `device=cpu`）；**不是** `--gpu -1` |

---

## 逐数据集 Phase-A + checkpoint 检查

| dataset | dataset.pkl | kg.pkl | transe.pkl | review_dict_* | RL ckpt (disk) | Phase B 全 seed 完成 | ready_eval_only_on_cpu |
|---------|-------------|--------|------------|---------------|----------------|---------------------|------------------------|
| last_fm_star | yes | yes | yes | yes | epoch-50 (s2 only) | yes (3/3 logs) | **no** — 缺 eval 脚本；s0/s1 ckpt 已覆盖 |
| book | yes | yes | yes | yes | epoch-50 (s2 only) | yes (3/3) | **no** |
| movie | yes | yes | yes | yes | epoch-50 (s2 only) | **no** (s1 失败) | **no** |
| yelp_star | yes | yes | yes | yes | epoch-10/20/30 + stale epoch-50 | **no** (s0 运行中) | **no** |

### 状态分类

- **ready_eval_only_on_cpu：** 0 个 dataset×seed
- **checkpoint_ready_but_missing_phase_a_artifacts：** 0（Phase-A 在 CPU 齐全）
- **checkpoint_ready_but_missing_on_disk_for_seed：** BOOK/LAST_FM/MOVIE 的 s0、s1（日志 DONE 但 pkl 被后序 seed 覆盖）
- **checkpoint_missing：** MOVIE s1；Yelp s1/s2
- **phase_b_incomplete：** YELP s0（运行中）、s1/s2（未开始）；MOVIE s1

---

## GPU Phase-A 是否还需要？

CPU 上已有 2026-05-16 生成的 `dataset.pkl` / `kg.pkl` / `transe.pkl`，且当前 Phase B 训练日志显示 `transe Embedding load successfully!`，说明 **CPU Phase-A 工件与当前 Phase B 兼容**。

若 GPU 环境的 Phase-A 是为 **InterRec 协议对齐**（不同 split / 不同 transe 训练），则 eval-only 前需确认 CPU 与 GPU 的 Phase-A 是否同一套；若不一致，应把 GPU 的 Phase-A 工件同步到 CPU **或** 把 CPU checkpoint 同步到 GPU，但**不要混用不同来源的 Phase-A 与 Phase-B**。

---

## CSV

详见同目录 `mcmipl_eval_only_cpu_requirements.csv`。
