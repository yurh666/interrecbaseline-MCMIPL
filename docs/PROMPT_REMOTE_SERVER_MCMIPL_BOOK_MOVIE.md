# 远程服务器：完整 MCMIPL + 继续跑 BOOK / MOVIE（可复制 Prompt）

将 **「分隔线以内」整段 **复制到新服务器的 Cursor / ChatGPT，在 **Ubuntu + NVIDIA GPU（建议 ≥16GB）+ CUDA 与 conda** 齐备的机器上执行。

---

## 【复制开始】

你是资深机器学习工程助手。请在**一台干净的 Linux GPU 服务器**上完成下列目标，并用**简体中文**汇报。

### A. 获取「完整」复现物料

1. **克隆仓库**（含 `main_table_experiments`、`interrec`、数据集与脚本快照）：  
   `git clone https://github.com/yurh666/interrecbaseline-MCMIPL.git`  
   （若私有需 PAT/SSH；若需嵌套仓库原始 `.git` 历史见 `baselines/mcmipl_official/archived_mcmipl_nested_git/README_RESTORE.md`。）

2. 确认目录存在：  
   - `interrecbaseline-MCMIPL/interrec/`  
   - `interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL/`（内含 `data/book`、`data/movie` 及官方代码）  

3. **环境与依赖**  
   - 在 `main_table_experiments/baselines/mcmipl_official` 按官方方式创建 conda 环境（参考 `environment.yml`、`scripts/setup_env.sh`）。  
   - InterRec：`interrec/requirements.txt` 或与 MCMIPL 共用一环境均可，但若冲突则分环境：**InterRec 只跑预处理与导出**，**MCMIPL 只跑 `graph_init`/OpenKE/`RL_model.py`**。

### B. 与既有协议一致：BOOK + MOVIE 的 InterRec 时序链路

不要用新的随机切分；**一律以仓库脚本为准**：

1. 仓库根：`QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh`  
   - 默认已从仓库根推导 `INTERREC` 与 `BASELINE`。  
   - Movie 预处理阈值仅以 `interrec/configs/preprocess_mcmipl_movie.yaml` 为准（短序列数据集已放宽阈值，勿擅自改回 LastFM 默认除非写明原因）。

2. 结束后必须已对 **BOOK** 与 **MOVIE** 各自执行：`python graph_init.py --data_name BOOK` / `MOVIE`（脚本内已调用）。

3. 产出核对：`*UI_Interaction_data/review_dict_*.json`、`*UI_data/*.pkl`、以及与图相关的 `Graph_generate_data`（依官方目录为准）。

### C. Baseline RL：在两个数据集上按官方流程继续

在主表实验中，**LAST_FM_STAR / YELP_STAR** 使用 `run_mcmipl.sh` 调 `RL_model.py`。BOOK/MOVIE 请你**类比同一入口**：

- 进入：`main_table_experiments/baselines/mcmipl_official/MCMIPL`  
- **先完成官方要求的 TransE / OpenKE embedding**（与 `embed transe`、`tmp/.../embeds/transe.pkl` 对齐；若仓库 snapshot 已有 `tmp` 可自检是否复用）。  
- 再运行 `RL_model.py`，`--data_name` 为 **`BOOK`** 与 **`MOVIE`**，seed 与用户约定一致（如三 seed：`0/1/2`），其余超参对齐主表：`--sample_times`、`--max_steps`、`--choice_num`、`--eval_num`、`--save_num`、`--gpu` 等与 `run_mcmipl.sh`/论文主表配置一致。**不要凭空改数据集名或路径。**

**GPU 约束**：`RL_model.py` 将 **GCN (`GraphEncoder`) + DQN** 置于 CUDA；长程训练时请保证单进程独占一张 GPU 或调好 `CUDA_VISIBLE_DEVICES`。

### D. 交付物

1. BOOK / MOVIE **各自**的完整训练日志路径（建议使用与主表一致的 `tee` 命名：`train_BOOK_s{seed}.log` 等）。  
2. 关键指标快照（能从日志解析 SR5/SR10/SR15 等则摘取最终轮）。  
3. 若在任一步失败：贴完整 **stderr**，并说明你修改了哪个文件。

### E. 不要做的事

- 不要跳过 `graph_init` 却只改交互 json/pkl。  
- 不要随意替换 `review_dict`/pkl 为其它切分而未记录。  
- 不要在未确认磁盘与显存的前提下并行开多个 `RL_model.py` 争用同一张卡。

**【复制结束】**

---

## 本机一页备忘（人工执行）

```bash
cd ~/interrecbaseline-MCMIPL
QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh
```

然后按官方文档完成 TransE，再对每个 seed 调用 `MCMIPL/RL_model.py`（BOOK、MOVIE）。
