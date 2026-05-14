# 给 CPU 同事的 Phase B（RL）交接 Prompt（可复制）

目标仓库：[interrecbaseline-MCMIPL](https://github.com/yurh666/interrecbaseline-MCMIPL)（或你们 fork 的同一套目录布局）。

## 你已具备的前置

1. **代码**：`git pull` 后与 GPU 侧 **同一 commit**（让对方发你 `git rev-parse HEAD`，或看本仓库最新提交说明）。
2. **Phase B 必需文件**（已在仓库的 `main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/<slug>/` 下，**四数据集各一份**）：
   - `embeds/transe.pkl`：与当前 `kg.pkl` **同代**的 TransE 嵌入（GPU 侧用 `train_transe_from_kg.py` 从 `kg.pkl` 训练，**非** OpenKE 二进制；格式与官方 MCMIPL RL 读取方式一致）。
   - `dataset.pkl`、`kg.pkl`：**InterRec 时序协议** Phase A（`graph_init`）产物，与主表 baseline 设定一致。
3. **自检（建议 pull 后立刻跑）**：在 `MCMIPL` 目录下、使用你跑 RL 的同一 Python：

   ```bash
   cd interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL
   python verify_transe_phase_b.py --data_name BOOK
   python verify_transe_phase_b.py --data_name MOVIE
   python verify_transe_phase_b.py --data_name LAST_FM_STAR
   python verify_transe_phase_b.py --data_name YELP_STAR
   ```

   四个均应打印 `OK ...`，表示 `transe.pkl` 行数与 `dataset.pkl` 的 `value_len` 对齐；否则不要开始长跑 RL。

4. **备选拷贝方式**：若你从对象存储/网盘拿包而非 git，结构仍须与上一致；GPU 侧历史脚本 `scripts/snapshot_offline_embeddings_four_datasets.sh` 可用于再打 tarball（大文件可不入库）。

## 环境与命令（纯 CPU RL）

```bash
cd interrecbaseline-MCMIPL/main_table_experiments
# 与仓库 README 一致；无 GPU 时强制走 CPU 路径
export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON=/path/to/your/python   # 可选

# 推荐 tmux / screen：四数据集 × 三 seed 耗时较长
# tmux new -s mcmipl_phase_b

# 四数据集 × seed 0,1,2（与 run_pipeline_phase_gpu.sh 默认 SEEDS 一致）
SEEDS="0 1 2" bash run_pipeline_phase_gpu.sh BOOK MOVIE LAST_FM_STAR YELP_STAR
```

或单个数据集单个 seed：

```bash
MCMIPL_FORCE_CPU=1 bash run_mcmipl.sh LAST_FM_STAR 0 50 100 10
```

参数与 `run_mcmipl.sh` 一致：`max_steps=50`、`sample_times=100`、`eval_num=10` 可按主表需要调整。

## GPU 侧训练记录（便于对账）

- 四数据集顺序训练与校验日志：`main_table_experiments/logs/transe_four_datasets_20260514_181602.log`（含各集 `OK <DATA>`）。
- GPU 采样：`logs/transe_gpu_monitor_20260514_181602.log`。
- 脚本：`scripts/run_transe_four_datasets_gpu_monitored.sh`；单集训练：`MCMIPL/train_transe_from_kg.py`（需在 `MCMIPL` 目录执行，且若 DGL/CUDA 缺库需按 README 配好 `LD_LIBRARY_PATH`）。

## 说明（方法 / 备注）

- 本文 **transe** 与 **kg** 对齐，适用于 **InterRec 时序切分后的图**；若你本地重做 `graph_init` 或改数据，须在本机重跑 `train_transe_from_kg.py`，不可沿用旧 `transe.pkl`。
- 官方 MCMIPL **默认支持 CPU**；RL 瓶颈多在环境与采样循环。剖析用环境变量见 `docs/RL_PHASE_TIMING_METHODOLOGY.md`。
- **不要**在正式论文实验中设置 `MCMIPL_RL_PROFILE_TEST_USERS`（仅剖析用）。

## 完成后回传

- 各 `logs/train_<DATASET>_s<seed>.log` 中含 `=== DONE: ... ===` 行。
- 可选：`bash scripts/record_phase_a_artifacts.sh` 或约定路径打包 `tmp/` 与日志供汇总主表。
