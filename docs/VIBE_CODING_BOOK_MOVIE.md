# Vibe coding：按现有逻辑跑通 MCMIPL BOOK 与 MOVIE（InterRec 时序对齐）

将下面整段复制给编码助手（Cursor / ChatGPT / 等），在已 clone 本仓库的前提下执行。目标：**沿用仓库里已有的 InterRec → MCMIPL 导出与 `graph_init` 顺序，不要发明新切分规则**。

---

## Prompt（复制自下一行开始）

你是资深机器学习工程助手。工作目录是已 clone 的仓库 **`interrecbaseline-MCMIPL`**，结构为：

- `interrec/`：时序预处理、会话导出脚本
- `main_table_experiments/baselines/mcmipl_official/MCMIPL/`：MCMIPL 官方代码与 `data/book`、`data/movie`

**任务**：在不改变现有协议的前提下，跑通 **BOOK** 与 **MOVIE** 两个数据集与 InterRec 对齐后的 pipeline，并说明如何接着跑官方 TransE + RL（与仓库内文档一致即可）。

**必须遵守的逻辑（与当前脚本一致）**

1. **唯一入口脚本**：`main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh`
   - 该脚本依次做：MCMIPL 官方字典 → InterRec CSV → `preprocess_dataset.py`（book / movie 各一份 yaml）→ 可选 QUICK 截断 → 写回 MCMIPL 的 `review_dict` + `UI_data` pkl → 在 `MCMIPL` 目录下执行 `graph_init.py --data_name BOOK` 与 `MOVIE`。
2. **不要**手工改 JSON/pkl 后跳过 `graph_init`；图侧必须与交互一致。
3. **Movie 预处理阈值**以 `interrec/configs/preprocess_mcmipl_movie.yaml` 为准（已针对短序列放宽 `min_user_interactions`、`min_observed_interactions` 等）；**Book** 以 `preprocess_mcmipl_book.yaml` 为准。若需调参，只改 yaml 并记录原因，且仍须跑完整条 pipeline。
4. 需要备份时，脚本会把当前 UI 数据拷到 `MCMIPL/data/_backup_official_split_*`；恢复方式在脚本结尾 echo 里有提示。

**请你执行并回报**

1. 先 **烟测**：`QUICK=1 QUICK_N=80` 运行上述 shell，确认无报错，`graph_init` 对 BOOK 和 MOVIE 均完成。
2. 再 **全量**：`QUICK=0` 运行同上脚本（或明确说明资源/时间问题只做烟测）。
3. 列出本次生成的关键产物路径：`sessions.json`（processed）、`review_dict_*.json`、`UI_data` 下 pkl、`Graph_generate_data` 等 graph_init 输出目录（以仓库实际为准）。
4. 用简短列表说明：**下一步**如何按 MCMIPL 官方流程跑 TransE（OpenKE）和 `RL_model.py`（只引用本仓库 `MCMIPL` 内已有入口脚本或 README，不要臆造参数）。

**约束**

- Python 路径：在 `interrec` 目录下跑 InterRec 脚本；在 `MCMIPL` 目录下跑 `graph_init.py`（与脚本内 `cd` 一致）。
- 若环境缺失依赖，根据 `mcmipl_official/environment.yml` / `setup_env.sh` 安装后再跑。
- 输出用简体中文，步骤带命令块，失败时贴出完整 stderr 片段。

---（Prompt 结束）

---

## 本地一行命令备忘录

```bash
cd /path/to/interrecbaseline-MCMIPL
QUICK=1 QUICK_N=80 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh
```

```bash
cd /path/to/interrecbaseline-MCMIPL
QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh
```
