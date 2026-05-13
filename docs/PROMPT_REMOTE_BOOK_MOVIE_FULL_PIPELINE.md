# 远程服务器操作 Prompt（BOOK / MOVIE 全流程）

下面整段可复制给编码助手。**前提**：已从 GitHub 克隆本仓库 [`interrecbaseline-MCMIPL`](https://github.com/yurh666/interrecbaseline-MCMIPL)。

---

你是资深 ML 工程助手，目标是在**新的 Linux GPU 服务器**上完成三件事：**拉代码 → 配环境 → 跑通 MCMIPL 上「后两个数据集」BOOK 与 MOVIE**，并得到可写进报告的指标与简短分析。

## 术语

- **后两个数据集**：相对主表里已跑的 **LAST_FM_STAR、YELP_STAR**，此处指 **BOOK、MOVIE**。
- **协议**：与仓库内 **`rebuild_book_movie_interrec_temporal.sh`** 完全一致（InterRec 时序切 → 导出 MCMIPL → `graph_init`），再上 **TransE/OpenKE**，最后 **`RL_model.py`**。

---

### 第一步：克隆与校验

```bash
git clone https://github.com/yurh666/interrecbaseline-MCMIPL.git
cd interrecbaseline-MCMIPL
```

确认目录存在：`interrec/`、`main_table_experiments/baselines/mcmipl_official/MCMIPL/`（内含 `data/book`、`data/movie` 等）。

若为私有仓库，改用 SSHclone 或 HTTPS + PAT。

---

### 第二步：Python / CUDA / Conda

1. 安装与你的 GPU 匹配的 **CUDA 驱动**（本机 `nvidia-smi` 正常）。
2. 创建 MCMIPL 环境：

```bash
cd main_table_experiments/baselines/mcmipl_official
bash scripts/setup_env.sh
# 或 conda env create -f environment.yml
```

3. InterRec 侧可在同一环境安装 `interrec/requirements.txt`，若冲突则：**InterRec 预处理一个 env，MCMIPL RL 另一个 env**，切换激活即可。

记下激活命令，后续步骤均在对应环境中执行。

---

### 第三步：BOOK + MOVIE 对齐（InterRec → MCMIPL → graph_init）

仓库根目录执行：

```bash
cd /path/to/interrecbaseline-MCMIPL

# 烟测（可选，几分钟级）
QUICK=1 QUICK_N=80 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh

# 正式全量
QUICK=0 bash main_table_experiments/baselines/mcmipl_official/scripts/rebuild_book_movie_interrec_temporal.sh
```

成功标志：无报错退出；`MCMIPL/data/_backup_official_split_*` 可能有备份；`python graph_init.py --data_name BOOK` 与 `MOVIE` 已在脚本末尾执行。**Movie** 预处理阈值仅以 `interrec/configs/preprocess_mcmipl_movie.yaml` 为准，勿私自改回 book 默认值而不记录。

---

### 第四步：TransE / OpenKE 嵌入（与 LAST_FM/Yelp 同源流程）

进入 `main_table_experiments/baselines/mcmipl_official/MCMIPL`，按官方 MCMIPL/README 对该仓库的流程，为 **BOOK**、**MOVIE** 各自生成 **`tmp/<dataset>/embeds/transe.pkl`**（及 kg 等中间文件）。  
若仓库 snapshot 已带可用的 `tmp/book`、`tmp/movie`，先校验文件时间与大小是否合理，缺失则重新训练嵌入。

---

### 第五步：RL 训练（多 seed）

对齐主表：**`main_table_experiments/run_mcmipl.sh`** 已写好与 `RL_model.py` 一致的参数（`sample_times=100`、`save_num=max_steps` 等）。远端先**编辑该文件**三处绝对路径为本机：

- `PYTHON=` → 你的 conda 里 `python` 可执行文件  
- `MCMIPL_DIR=` → `<仓库根>/main_table_experiments/baselines/mcmipl_official/MCMIPL`  
- `LOG_DIR=` → `<仓库根>/main_table_experiments/logs`  

然后在仓库内执行（参数：`DATASET`、`SEED`、`MAX_STEPS`、`SAMPLE_TIMES`、`EVAL_NUM`）：

```bash
cd /path/to/interrecbaseline-MCMIPL/main_table_experiments

# BOOK，三 seed（与 Last_FM 主表一致时常用 max_steps=100）
bash run_mcmipl.sh BOOK 0 100 100 10
bash run_mcmipl.sh BOOK 1 100 100 10
bash run_mcmipl.sh BOOK 2 100 100 10

# MOVIE 同理
bash run_mcmipl.sh MOVIE 0 100 100 10
bash run_mcmipl.sh MOVIE 1 100 100 10
bash run_mcmipl.sh MOVIE 2 100 100 10
```

若某条主表基线曾用 **50** 步（如部分 YELP 配置），把第三、四个数字改为 `50 50` 即可与那条线对齐；BOOK/MOVIE 与谁对齐由你统一约定，并**三 seed 都跑完**再汇总。

---

### 第六步：解析结果与写表

将日志放到 `main_table_experiments/logs/` 并命名为 `train_BOOK_s{0,1,2}.log`、`train_MOVIE_s{0,1,2}.log`（与现有 parser 一致）。

```bash
cd main_table_experiments
python3 comparison/collect_results.py
```

得到 `comparison/results/mcmipl/mcmipl_BOOK_s*.json`、`mcmipl_MOVIE_s*.json` 与 `mcmipl_main_table.csv`。

**分析输出**（给负责人看）：

1. 每个数据集三行的 **SR@5 / SR@10 / SR@15 / AvgT / hDCG**（择优规则同 `mcmipl_log_metrics`）。
2. **mean ± std（三 seed）**。
3. 与 **LAST_FM / YELP** 同表对比时，脚注 **BOOK/MOVIE 使用 InterRec 时序切分协议**（与 §3.2 一致），勿与官方随机划分 Yelp 行混读因果。

---

### 第七步：常见问题

| 现象 | 处理 |
|------|------|
| `graph_init` OOM / 慢 | 减小 batch 或换大显存卡；先 QUICK=1。 |
| Movie 预处理用户过少 | 勿改 yaml 阈值除非记录原因。 |
| RL 无 checkpoint | 检查 `save_num` 与 `max_steps`；单卡不要并行两个 `RL_model.py`。 |

---

### 约束

- 不许跳过 `graph_init` 只改 `review_dict`。
- 所有命令贴出 **完整 stderr** 若失败。
- 输出用 **简体中文**，步骤带可复制命令块。

---

（本文件路径：`docs/PROMPT_REMOTE_BOOK_MOVIE_FULL_PIPELINE.md`。）
