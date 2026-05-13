# CPU 阶段操作说明与 GPU 交接 Prompt

面向仓库：[interrecbaseline-MCMIPL](https://github.com/yurh666/interrecbaseline-MCMIPL)。

---

## 一、在本地/CPU 机上：从克隆到 Phase A 再到推送

### 1. 克隆与检查内容

```bash
git clone https://github.com/yurh666/interrecbaseline-MCMIPL.git
cd interrecbaseline-MCMIPL
git status
ls -la main_table_experiments/run_pipeline_phase_*.sh
ls -la main_table_experiments/baselines/mcmipl_official/MCMIPL/data/
```

应能看到：

- `main_table_experiments/run_pipeline_phase_cpu.sh`（Phase A，纯 CPU）
- `main_table_experiments/run_pipeline_phase_gpu.sh`（Phase B，优先 GPU）
- `main_table_experiments/run_pipeline_phase_rl_cpu.sh`（Phase B，强制 CPU）
- `main_table_experiments/run_mcmipl.sh`（路径已相对脚本目录，可设 `MCMIPL_GPU_PYTHON` / `MCMIPL_CPU_PYTHON`）
- `interrec/` 与 `MCMIPL/data/<dataset>/` 数据布局

### 2. 检查环境（Phase A）

Phase A 只要求 **`python` 能跑 `graph_init` 与 InterRec 脚本**（建议 `mcmipl-reproduce` 或官方 `environment.yml` 对应环境）。

```bash
cd main_table_experiments/baselines/mcmipl_official
# 若尚未创建环境，参见官方 scripts/setup_env.sh 或 environment.yml
python -c "import torch; print('torch', torch.__version__)"
python -c "import dgl; print('dgl ok')"
```

在 **纯 CPU** 机器上建议：

```bash
export CUDA_VISIBLE_DEVICES=""
```

### 3. 执行 Phase A（CPU）

在仓库根目录或 `main_table_experiments` 下：

```bash
cd main_table_experiments
# 仅对指定数据集做 prepare_data + graph_init
bash run_pipeline_phase_cpu.sh LAST_FM_STAR YELP_STAR BOOK MOVIE

# 若需先从 InterRec 重跑 BOOK+MOVIE 全链路再构图，加：
# RUN_REBUILD_BOOK_MOVIE=1 bash run_pipeline_phase_cpu.sh BOOK MOVIE
```

### 4. 记录中间产物（便于交接与审计）

```bash
bash scripts/record_phase_a_artifacts.sh
```

会在 `main_table_experiments/artifacts/phase_a_manifest_*.txt` 生成路径列表与关键 `transe.pkl` / `dataset.pkl` / `kg.pkl` 的 `sha256sum`。

### 5. 将更新提交并推送到 GitHub

```bash
cd /path/to/interrecbaseline-MCMIPL
git add -A
git status   # 确认无意外大文件或密钥
git commit -m "chore: Phase A artifacts + pipeline handoff docs"
git push origin main
```

若单个文件接近或超过 GitHub 限制（约 100MB），请改用 [Git LFS](https://git-lfs.github.com/) 或对象存储，**不要**硬推普通 blob。

---

## 二、复制给「跑 GPU 阶段」的 AI / 同事的 Prompt（全文粘贴）

把下面整段当作一条用户消息发给你的 coding agent（例如 Cursor），并附上 **你们刚 push 的分支名/commit** 与 **已完成的数据集列表**。

---

**（以下为可复制 Prompt 正文）**

你是负责在主表 MCMIPL baseline 上跑 **Phase B（GPU 侧）** 的编码助手。代码库为：

- https://github.com/yurh666/interrecbaseline-MCMIPL

请按顺序完成：

1. **拉代码**：`git clone` 或 `git pull` 到最新；记录 `git rev-parse HEAD`。
2. **核对 Phase A 产物**（应在 `main_table_experiments/baselines/mcmipl_official/MCMIPL/`）：
   - 每个要训的数据集：`tmp/<slug>/embeds/transe.pkl` 存在（slug：`last_fm_star` / `yelp_star` / `book` / `movie`）。
   - 若缺少 `transe.pkl`：按官方 MCMIPL README 用 OpenKE 训练 TransE，或向同事索取同名文件；`REQUIRE_TRANSE=1` 可在 `run_pipeline_phase_gpu.sh` 中强制检查。
3. **GPU 环境**：创建/激活带 CUDA 的 conda 环境（例如本仓库记录的 `mcmipl-baseline-gpu`），设置：
   - `MCMIPL_GPU_PYTHON=/path/to/env/bin/python`（若默认路径不存在）
   - 不要设置 `MCMIPL_FORCE_CPU`（除非明确要求 CPU RL）。
4. **跑 RL**（示例）：

   ```bash
   cd main_table_experiments
   REQUIRE_TRANSE=1 MAX_STEPS=50 SAMPLE_TIMES=100 EVAL_NUM=10 \
     bash run_pipeline_phase_gpu.sh BOOK MOVIE
   ```

   或按项目惯例单数据集、多 seed：`bash run_mcmipl.sh DATASET SEED MAX_STEPS SAMPLE_TIMES EVAL_NUM`。
5. **验证**：`grep 'DONE:' logs/train_*`；日志中带 `cuda`、评测指标行无异常 traceback。
6. **回传**：把各数据集各 seed 的最终日志路径、`nvidia-smi` 摘要、以及 `git rev-parse HEAD` 写进简短报告；如需把新日志/检查点入库，单独说明体积并建议是否 Git LFS。

**已知约束**：`RL_model.py` 在部分环境下 GPU 利用率可能长期较低（CPU/Python 循环为主），属预期现象；仍以 `cuda` 与训练日志为准判断是否在用 GPU Python。

**（Prompt 结束）**

---

## 三、分阶段含义速查

| 阶段 | 脚本 | 机器 | 依赖 |
|------|------|------|------|
| A | `run_pipeline_phase_cpu.sh` | 无 GPU 也可 | `graph_init`、数据 |
| TransE | OpenKE（官方流程） | GPU 推荐 | Phase A 后的图数据 |
| B | `run_pipeline_phase_gpu.sh` | 本机 GPU | `transe.pkl` + 同路径数据 |

更多设计说明见 `main_table_experiments/README.md` 中「CPU/GPU 分阶段」小节。
