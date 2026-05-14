# 给 CPU 同事的 RL 阶段交接 Prompt（可复制）

目标仓库：[interrecbaseline-MCMIPL](https://github.com/yurh666/interrecbaseline-MCMIPL)（或你们 fork 的同一套目录布局）。

## 你已具备的前置

1. **代码**：`git clone https://github.com/yurh666/interrecbaseline-MCMIPL.git` 并 checkout 与 GPU 侧一致的 commit（见对方提供的 `git rev-parse HEAD`）。
2. **离线中间结果（四数据集）**：每个数据集需要与官方 MCMIPL 一致的目录（在 `main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/<slug>/`）：
   - `embeds/transe.pkl`（OpenKE/TransE，**每个数据集一份**，与 RL seed **无关**）
   - `dataset.pkl`、`kg.pkl`（`graph_init.py` 产物）
3. GPU 侧若已跑过快照脚本，可从 `main_table_experiments/artifacts/offline_mcmipl_graph_transe_*` 把子目录 **`book/`、`movie/`、`last_fm_star/`、`yelp_star/`** 拷回对应 `tmp/`（或直接用对方 `rsync` 整个 `MCMIPL/tmp`）。

## 环境与命令（纯 CPU RL）

```bash
cd interrecbaseline-MCMIPL/main_table_experiments
# Python：与仓库 README 一致；若无 GPU 可仅用 CPU 环境
export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON=/path/to/your/python   # 可选，覆盖默认

# 四数据集 × seed 0,1,2（与 run_pipeline_phase_gpu.sh 默认 SEEDS 一致）
SEEDS="0 1 2" bash run_pipeline_phase_gpu.sh BOOK MOVIE LAST_FM_STAR YELP_STAR
```

或单个数据集单个 seed：

```bash
MCMIPL_FORCE_CPU=1 bash run_mcmipl.sh LAST_FM_STAR 0 50 100 10
```

参数与 `run_mcmipl.sh` 一致：`max_steps=50`、`sample_times=100`、`eval_num=10` 可按主表需要调整。

## 说明（写给方法/备注）

- 官方 MCMIPL **默认支持 CPU**；RL 阶段瓶颈多为 **交互环境与采样循环**，GPU 收益因配置而异。短程剖析日志见 `main_table_experiments/logs/rl_gpu_cpu_profile_*.log` 与 `docs/RL_PHASE_TIMING_METHODOLOGY.md`。
- **不要**在正式论文实验中设置 `MCMIPL_RL_PROFILE_TEST_USERS`（仅剖析用）。

## 完成后回传

- 各 `logs/train_<DATASET>_s<seed>.log` 中含 `=== DONE: ... ===` 行。
- 可选：`bash scripts/record_phase_a_artifacts.sh` 或约定路径打包 `tmp/` 与日志供汇总主表。
