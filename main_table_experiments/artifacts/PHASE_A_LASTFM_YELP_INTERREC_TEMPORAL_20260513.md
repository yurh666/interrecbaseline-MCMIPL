# Phase A — LAST_FM_STAR / YELP_STAR InterRec 时序对齐后重跑（2026-05-13）

## 背景

上游 **`d31280d` / merged `01932fd`**：为 **`LAST_FM_STAR`、`YELP_STAR`** 增加与 BOOK/MOVIE 一致的 InterRec **时序切分协议**（`observed 40%` + future **`7:1:2`**），并新增脚本 `baselines/mcmipl_official/scripts/rebuild_lastfm_yelp_interrec_temporal.sh` 与对应 `interrec/configs/preprocess_mcmipl_{lastfm_star,yelp_star}.yaml`。

## 本机在 `01932fd` 上执行的内容

顺序执行（单次 tmux，`EXIT=0`）：

1. **全量** `rebuild_lastfm_yelp_interrec_temporal.sh`（`QUICK=0`，未切 smoke）  
   - MCMIPL → InterRec CSV → `preprocess_dataset.py` → 写回 `MCMIPL/data/{lastfm_star,yelp_star}/UI_*` → 末尾 **`graph_init`**（LAST_FM_STAR、YELP_STAR）。
2. **标准 Phase A 薄封装**：`bash run_pipeline_phase_cpu.sh LAST_FM_STAR YELP_STAR`  
   - 再次 `prepare_data` + **`run_graph_init.sh`**（与仓库 Phase A 日志路径一致，`raw_logs/` 可追溯）。

控制台与串联日志：`main_table_experiments/logs/tmux_phase_a_lastfm_yelp_20260513_194711.log`。

## 产物核对（交接 GPU 必读）

| 数据集 | Phase A / graph 产物变更（相对本次 pull 前） | `transe.pkl` |
|--------|-----------------------------------------------|--------------|
| **LAST_FM_STAR** | **`dataset.pkl` 体积未变，`kg.pkl` 已变更**（图结构刷新） | 仓库内二进制 **未随本次重写**（时间戳仍为旧）；与 **新 KG 可能不一致**。 |
| **YELP_STAR** | **`UI_*`、`dataset.pkl`、`kg.pkl` 与上一轮 Phase A（2026-05-13 午间）语义一致**，本次 rerun 后与当时提交相符 | 仍为既有 **`transe.pkl`**；若在你们流程里 **KG 必须与 TransE 同代训练**，请 **在两域上重做 OpenKE / TransE** 后再 RL。 |

**BOOK / MOVIE**：本轮 **未触碰**。

## manifest 与哈希

运行 `bash main_table_experiments/scripts/record_phase_a_artifacts.sh` 后清单：

- `main_table_experiments/artifacts/phase_a_manifest_latest.txt` → `phase_a_manifest_20260513_195102.txt`

## InterRec 中间文件（可复现实验）

已纳入提交的目录（体量约：**raw ~30MB + processed ~78MB**，均低于 GitHub 单文件告警阈值）：

- `interrec/data/raw_mcmipl_{lastfm_star,yelp_star}/`
- `interrec/data/processed_mcmipl_{lastfm_star,yelp_star}/`

## 备份（未提交）

单次运行生成的 UI 备份目录（仅存本机磁盘，不入库）示例：

`MCMIPL/data/_backup_official_split_20260513_194741/`

---

**交接 GPU 的同事**：请以本提交 **`git rev-parse HEAD`**、`phase_a_manifest_*.txt` 与 `HANDOFF_RUNBOOK.md` 第二节为准；对 LAST_FM_STAR 建议 **优先考虑重训 TransE** 后再进入 Phase B。
