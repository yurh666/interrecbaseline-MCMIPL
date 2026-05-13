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

在 `MCMIPL/` 目录、已激活 conda 前提下，对齐主表超参示例（与 `run_mcmipl.sh` 同类）：

```bash
cd main_table_experiments/baselines/mcmipl_official/MCMIPL
conda activate <你的环境名>

# BOOK —— seed 0/1/2 示例（max_steps / sample_times / eval_num 等与主表 LAST_FM/Yelp 对齐）
python -u RL_model.py --data_name BOOK --embed transe --seed 0 --gpu 0 \
  --max_steps 100 --sample_times 100 --attr_num 20 --choice_num 4 --max_turn 15 \
  --eval_num 10 --save_num 100 ...
# 对每个 seed 重复，并 tee 保存日志到自建路径，如 ~/logs/train_BOOK_s0.log

# MOVIE 同理，--data_name MOVIE
```

**实际参数**请打开本仓库 **`main_table_experiments/run_mcmipl.sh`**，将 `DATASET` 改为 `BOOK`、`MOVIE`，`MAX_STEPS` 若与 LastFM 一致为 100，Yelp 曾为 50——**以你希望与主表哪条基线对齐为准**，三数据集之间步数可能不同属官方设定；务必 **三 seed 均跑完** 再汇总。

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
