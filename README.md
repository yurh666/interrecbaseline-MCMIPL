# interrecbaseline-MCMIPL

将 [MCMIPL](https://github.com/ZYM6-6/MCMIPL) 官方 baseline 与本仓库内 `interrec/`（时序预处理）pipeline 对齐后的复现资料：包含 **BOOK / MOVIE / LAST_FM_STAR / YELP_STAR** 的 `convert → preprocess → export → graph_init` 脚本，以及与本机实验目录同步的 **`main_table_experiments`** 与 **`interrec`** 源码（含 MCMIPL 官方 `data/` 布局）。

## 仓库布局

```text
interrecbaseline-MCMIPL/
  interrec/                          # InterRec：预处理、导出到 MCMIPL
  main_table_experiments/
    baselines/mcmipl_official/
      MCMIPL/                        # 官方代码 + data/book、data/movie 等
      scripts/
        rebuild_book_movie_interrec_temporal.sh      # BOOK+MOVIE 一键对齐
        rebuild_lastfm_yelp_interrec_temporal.sh       # LAST_FM_STAR+YELP_STAR 一键对齐
    comparison/                      # 主表收集脚本与报告（可选）
  docs/
    VIBE_CODING_BOOK_MOVIE.md        # 给 AI /「vibe coding」用的逐步 prompt
    HANDOFF_RUNBOOK.md              # CPU Phase A → git 推送 → GPU Phase B 交接说明 + 可复制 Prompt
```

说明：本仓库已包含 **`main_table_experiments/logs`**、**`MCMIPL/tmp`**（训练中间产物与评估日志），以及原嵌套仓库 **`MCMIPL/.git` 的 gzip 分片备份**（见 `baselines/mcmipl_official/archived_mcmipl_nested_git/`，单文件超过 100MB 故分片；还原步骤见该目录下 `README_RESTORE.md`）。

## CPU / GPU 分阶段与交接

- **Phase A（纯 CPU）**：`main_table_experiments/run_pipeline_phase_cpu.sh` — `prepare_data` + `graph_init`，可在无 GPU 服务器完成。
- **TransE / OpenKE**：插在 Phase A 与 RL 之间；产物放在 `MCMIPL/tmp/<slug>/embeds/transe.pkl`（与官方 [MCMIPL](https://github.com/ZYM6-6/MCMIPL) 一致）。
- **Phase B（RL，优先 GPU）**：`run_pipeline_phase_gpu.sh`；无 GPU 时用 `run_pipeline_phase_rl_cpu.sh`。

**从克隆、检查环境、跑 Phase A、记录产物、push 到 GitHub，再到把任务交给 AI 跑 GPU 的完整说明与一段可复制 Prompt**，见 **[`docs/HANDOFF_RUNBOOK.md`](docs/HANDOFF_RUNBOOK.md)**。

远程主仓库：<https://github.com/yurh666/interrecbaseline-MCMIPL>。


## 环境

- 按 `main_table_experiments/baselines/mcmipl_official` 内 `environment.yml` / `scripts/setup_env.sh` 创建 conda 环境。
- **InterRec** 与 **MCMIPL** 可使用同一 Python 环境，或分别为 InterRec 与 MCMIPL 安装依赖（以各自 `requirements` / 官方说明为准）。

## BOOK / MOVIE：与 InterRec 时序切分对齐

在仓库根目录执行（脚本已默认 `INTERREC=<repo>/interrec`、`BASELINE=<repo>/.../mcmipl_official`）：

```bash
# 烟测：截断少量用户，验证 graph_init
QUICK=1 QUICK_N=80 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh

# 全量：InterRec sessions → 写回 MCMIPL → graph_init（BOOK + MOVIE）
QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh
```

流程概要：官方合并字典 → InterRec CSV → `preprocess_mcmipl_book.yaml` / `preprocess_mcmipl_movie.yaml` →（可选 QUICK slice）→ 导出 `review_dict` + `UI_data` pkl → `graph_init.py --data_name BOOK|MOVIE`。备份目录在 `MCMIPL/data/_backup_official_split_*`。

**Movie 数据集**在 `interrec/configs/preprocess_mcmipl_movie.yaml` 中调整了 `min_user_interactions` / `min_observed_interactions` 等阈值，以适配序列极短的分布（详见该文件内注释）。

## LAST_FM_STAR / YELP_STAR：与 InterRec 时序切分对齐

配置分别为 `preprocess_mcmipl_lastfm_star.yaml`、`preprocess_mcmipl_yelp_star.yaml`（与 BOOK/MOVIE 相同的 observed / future 比例）。在仓库根目录执行：

```bash
QUICK=1 QUICK_N=80 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_lastfm_yelp_interrec_temporal.sh
QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_lastfm_yelp_interrec_temporal.sh
```

导出时会按 **KG 已有用户**过滤：`lastfm_star` 取 `user_friends.pkl` 与 `user_like.pkl` 的键交集，`yelp_star` 取 `user_dict.json` 的键，避免 `graph_init` 构图时 KeyError。脚本末尾会对 `LAST_FM_STAR` 与 `YELP_STAR` 各跑一次 `graph_init.py`。

## 后续：TransE 与 RL

与官方 MCMIPL 一致：完成 `graph_init` 后，按官方文档训练 TransE（OpenKE）与 `RL_model.py`。**交互切分或用户集合若相对旧实验有变，旧的 `tmp/.../embeds/transe.pkl` 会与当前图不一致，应重训 TransE。** 本仓库不强制绑定 GPU 与随机种子，与主表实验配置请以本地 `comparison/` 与脚本为准。

## 上游

- MCMIPL：<https://github.com/ZYM6-6/MCMIPL>

## Vibe coding

若要让 AI 按同一套逻辑扩展或复跑 BOOK/MOVIE，请直接使用 [`docs/VIBE_CODING_BOOK_MOVIE.md`](docs/VIBE_CODING_BOOK_MOVIE.md) 中的可复制 prompt。
