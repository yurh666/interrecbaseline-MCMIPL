# MCMIPL Eval-Only 设备决策

**生成时间：** 2026-05-27

---

## 决策规则（本次 audit 应用结果）

1. CPU 有 checkpoint + Phase-A 工件 + interrec eval 脚本 → **CPU eval-only**
2. CPU 有 checkpoint 但缺 Phase-A → 从 GPU 同步 Phase-A 到 CPU，或反向同步 checkpoint
3. CPU 太慢 → 可选把 checkpoint 拷到 GPU 加速（非必须）
4. 无 checkpoint → **blocked**，等 Phase B

---

## 逐数据集决策

| dataset | recommended_eval_device | reason | required_copy_files |
|---------|------------------------|--------|---------------------|
| last_fm_star | **blocked** | 日志上 3 seed 均 DONE，但磁盘仅 s2 的 RL ckpt；缺 `eval_mcmipl_on_interrec_protocol.py` | 若要做 3-seed 主表：归档或重跑 s0/s1 Phase B → `RL-agent/*.pkl`；实现/同步 eval 脚本；若 GPU Phase-A 与 CPU 不一致则同步 `dataset.pkl, kg.pkl, embeds/transe.pkl` |
| book | **blocked** | 同 LastFM：仅 s2 ckpt 在盘；缺 interrec eval 脚本 | 同上 |
| movie | **blocked** | s1 Phase B 失败；仅 s2 ckpt；缺 eval 脚本 | 重跑 **MOVIE seed=1** Phase B；其余同 book |
| yelp_star | **blocked** | s0 在 screen 中运行（~epoch 40）；s1/s2 未开始 | 等 `run_phase_b_resume_lastfm_yelp.sh` 完成 3 seeds；删除/忽略 stale `epoch-50.pkl` |

---

## CPU vs GPU 建议

| 阶段 | 建议设备 | 说明 |
|------|---------|------|
| Phase B（当前） | **CPU** | 已在 screen 跑；勿打断 |
| Phase B 补跑 MOVIE s1 | **CPU** | 与现有流水线一致：`MCMIPL_FORCE_CPU=1 bash run_mcmipl.sh MOVIE 1` |
| Eval-only（InterRec 协议） | **CPU 优先**（工件已在 CPU） | Phase-A + 部分 RL ckpt 已在 CPU；实现 eval 脚本后可在 CPU 跑 |
| Eval-only 加速 | **GPU 可选** | 需同步 `tmp/<slug>/` 下 Phase-A + `RL-agent/` 到 GPU；设置 `MCMIPL_FORCE_CPU`  unset，用 GPU python |

**当前没有任何 dataset×seed 达到 ready_eval_only。**

---

## 与 GPU Phase-A 的关系

你提到之后从 GPU 拿 Phase-A 结果：

- 若 GPU Phase-A **与 CPU 现有 pkl 相同** → 无需再拷，直接等 Phase B + eval 脚本即可
- 若 GPU Phase-A **是 InterRec 对齐版本** → 在 eval-only 前替换 CPU 上 `tmp/<slug>/{dataset,kg,embeds/transe}.pkl`，**不要**在替换后沿用旧 RL checkpoint 而不验证兼容性（建议 Phase-A 替换后对受影响数据集重载 transe 并确认 RL eval 可 load）
