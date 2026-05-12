# TAIRA Baseline 复现 Prompt

> **这份文档的定位**：可直接作为 Prompt 给 AI Agent 或人类工程师使用。
> 在此文档后面追加"请帮我复现 TAIRA baseline"，AI 即可获得完整背景，
> 直接进行 TAIRA 官方代码复现，并与 InterRec 主方法进行公平比较。
>
> **官方 TAIRA repo**：https://github.com/Alcein/TAIRA  
> **论文**：Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent，KDD 2026  
> **InterRec 项目路径**：`/home/yurh/interrec`

---

## 一、实验目标

主表中比较以下方法：

1. **官方 TAIRA**（本文档目标）；
2. **InterRec**（我们的方法，`/home/yurh/interrec`）；


TAIRA 必须完整复现：

- 官方多 Agent 框架（Manager Agent + Executor Agents）；
- 官方 Thought Pattern Distillation（TPD）机制；
- 官方 `main.py` 运行流程；
- 官方用户模拟器（`user_simulate/`）；
- 官方评估指标（SR@K / AvgT 等）。

**禁止**：

- 用自己写的 single-agent / template-based 方法替代 TAIRA；
- 修改 TAIRA 的 TPD 核心逻辑；
- 修改 TAIRA 的用户模拟器；
- 修改 TAIRA 的评估指标；
- 省略 Manager Agent 的 hierarchical planning；
- 不使用真实 LLM API（不允许 mock LLM）。

---

## 二、TAIRA 官方代码结构

官方 repo（https://github.com/Alcein/TAIRA）目录结构：

```
TAIRA/
  agents/                  # Executor Agents (Searcher, Item Retriever, etc.)
  core/                    # 核心逻辑
  data/                    # 数据处理
  storage/thought_patterns/ # TPD 存储的思维模式
  user_simulate/           # 用户模拟器
  utils/                   # 工具函数
  main.py                  # 主入口
  manager.py               # Manager Agent
  requirements.txt
  system_config.yaml       # 配置文件
```

**第一步**：必须 clone 官方 repo 并完整阅读：

```bash
git clone https://github.com/Alcein/TAIRA.git
```

必须阅读以下文件并在 code review 报告中总结：

- `README.md`
- `main.py`（入口流程）
- `manager.py`（Manager Agent 逻辑）
- `agents/`（各 Executor Agent 类型和接口）
- `core/`（核心推荐逻辑）
- `user_simulate/`（用户模拟器逻辑）
- `system_config.yaml`（所有配置项含义）
- `requirements.txt`（依赖版本）
- `storage/thought_patterns/`（TPD 存储格式）

---

## 三、项目目录结构

在 `/home/yurh/main_table_experiments/baselines/` 下创建：

```
taira_official/
  TAIRA/                          # 官方 repo 原样克隆
  scripts/
    setup_env.sh                  # 环境安装
    prepare_data.sh               # 数据下载与格式转换
    run_experiment.sh             # 运行主实验
    run_eval.sh                   # 单独评估（如支持）
    run_all.sh                    # 一键跑全部 seed
  configs/
    taira_reproduce.yaml          # 对应 system_config.yaml 的复现配置
  results/
    raw_logs/                     # stdout / stderr 完整输出
    metrics/                      # 解析后的指标 JSON
  docs/
    taira_code_review.md          # 代码 review 报告
    taira_reproduction_report.md  # 最终复现报告
```

---

## 四、Code Review 报告要求

输出：`taira_official/docs/taira_code_review.md`

必须总结：

1. 官方 Python 版本要求（README 中为 Python 3.12.7）；
2. 所有依赖包及版本（`requirements.txt`）；
3. 需要的外部 API（OpenAI API、Google Search API）；
4. 支持的数据集（`amazon_clothing` / `amazon_beauty` / `amazon_music`）；
5. 数据目录格式（`data/` 下每个数据集的文件结构）；
6. `main.py` 的完整运行流程；
7. `manager.py` 的 Manager Agent 如何分解任务、分配给 Executor Agents；
8. 各 Executor Agent（Searcher / Item Retriever / 等）的职责和接口；
9. Thought Pattern Distillation（TPD）的完整逻辑：
   - 从哪些轨迹提取；
   - 存储格式（`storage/thought_patterns/`）；
   - 如何在推理时使用；
10. 用户模拟器（`user_simulate/`）的逻辑：
    - 如何模拟用户 intent；
    - 如何判断推荐成功；
    - success 定义；
11. 评估指标在代码中如何计算（SR@5 / SR@10 / SR@15 / AvgT / hDCG 等）；
12. `system_config.yaml` 中每个配置项的含义；
13. `QUERY_NUMBER`、`TOPN_ITEMS`、`TOPK_ITEMS` 的默认值及对结果的影响；
14. 每个随机种子如何设置（是否支持 seed 参数）；
15. Google Search API 的使用场景（是否可以禁用或替换）；
16. 复现中可能遇到的依赖问题和 API 访问问题。

---

## 五、环境搭建

TAIRA 官方要求 Python 3.12.7+。

### 5.1 配置文件

提供：

```
taira_official/configs/taira_reproduce.yaml
```

内容对应 `system_config.yaml`，至少包含：

```yaml
QUERY_NUMBER: 500          # 处理的 query 数量
TOPN_ITEMS: 500            # 候选 item 池大小
TOPK_ITEMS: 10             # 最终推荐数量（对齐 InterRec top_k=10）
DOMAIN: "amazon_music"     # 优先用 amazon_music（与 LastFM 音乐域接近）
MODEL: "gpt-4o"            # LLM 模型
METHOD: "TAIRA"
OPENAI_BASE_URL: ""        # 填入实际 API endpoint
OPENAI_API_KEY: ""         # 填入实际 key（不要 hardcode，用环境变量）
GOOGLE_API_KEY: ""         # 填入实际 key
GOOGLE_CSE_ID: ""          # 填入实际 CSE ID
```

**重要**：API key 必须通过环境变量传入，禁止 hardcode 到配置文件中。

推荐方式：
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
export GOOGLE_CSE_ID="..."
```

### 5.2 setup_env.sh

```bash
#!/bin/bash
# taira_official/scripts/setup_env.sh
set -e

conda create -n taira python=3.12.7 -y
conda activate taira
pip install -r TAIRA/requirements.txt

# 验证
python -c "import openai; print('openai:', openai.__version__)"
python -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: not installed"

# 记录环境
python -m pip list --format=freeze > results/raw_logs/env_versions.txt
echo "Setup complete."
```

---

## 六、数据准备

### 6.1 TAIRA 官方数据集

TAIRA 使用 Amazon 系列数据集，不是 LastFM。

优先使用 `amazon_music`（与 InterRec 的 LastFM 音乐域最接近）。

数据需要放置到：

```
TAIRA/data/amazon_music/
```

具体文件格式需阅读官方 `data/` 目录和 README 后确认，典型格式包括：
- `user_history.json`（用户历史）
- `items.json`（item 信息）
- `interactions.json` 或类似名称

**注意**：
- 必须使用官方数据格式，不能自行构造假数据；
- 如需下载，按官方 README 指示的链接下载；
- 如官方 README 未提供下载链接，必须在 code review 报告中说明。

### 6.2 prepare_data.sh

```bash
#!/bin/bash
# taira_official/scripts/prepare_data.sh
DOMAIN=${1:-amazon_music}
echo "Preparing data for domain: $DOMAIN"

# 按官方 README 的数据准备步骤操作
# 如有预处理脚本则调用
cd TAIRA
python data/prepare.py --domain $DOMAIN 2>&1 | tee ../results/raw_logs/prepare_data_${DOMAIN}.log
echo "Data preparation done."
```

---

## 七、运行 TAIRA 实验

### 7.1 run_experiment.sh

```bash
#!/bin/bash
# taira_official/scripts/run_experiment.sh
DOMAIN=${1:-amazon_music}
SEED=${2:-0}

echo "Running TAIRA: domain=$DOMAIN seed=$SEED"

LOG_FILE="results/raw_logs/run_${DOMAIN}_seed${SEED}.log"
METRIC_FILE="results/metrics/run_${DOMAIN}_seed${SEED}.json"

mkdir -p results/raw_logs results/metrics

# 设置随机种子（若官方支持）
export PYTHONHASHSEED=$SEED

cd TAIRA
python main.py \
  --domain $DOMAIN \
  --seed $SEED \
  2>&1 | tee "../$LOG_FILE"

echo "Run complete. Log: $LOG_FILE"
```

**注意**：
1. 如官方 `main.py` 不支持 `--seed` 参数，必须在 code review 报告中说明；
2. 记录完整 stdout + stderr；
3. 记录运行时长；
4. 不修改 `main.py` 核心逻辑。

### 7.2 run_all.sh（多 seed）

```bash
#!/bin/bash
# taira_official/scripts/run_all.sh
DOMAIN=${1:-amazon_music}

for SEED in 0 1 2; do
  echo "=== Seed $SEED ==="
  bash scripts/run_experiment.sh $DOMAIN $SEED
done

echo "All seeds done."
```

---

## 八、评估指标解析

必须从 TAIRA 的输出日志中解析以下指标（与 InterRec / MCMIPL 对齐）：

| 指标 | 说明 | 对应关系 |
|------|------|----------|
| SR@5 | Success Rate at turn 5 | 主表必选 |
| SR@10 | Success Rate at turn 10 | 主表必选 |
| SR@15 | Success Rate at turn 15 | 主表必选 |
| AvgT | 平均对话轮数 | 主表必选 |
| hDCG | hit-weighted DCG | 主表必选（若 TAIRA 有输出）|

实现 `parse_taira_metrics.py`：

```python
# taira_official/scripts/parse_taira_metrics.py
"""Parse TAIRA raw log output into metrics JSON compatible with main table."""
import re, json, sys
from pathlib import Path

def parse_log(log_path: str) -> dict:
    text = Path(log_path).read_text()
    metrics = {}
    # 根据实际 TAIRA 输出格式填写 regex
    # 例如：SR@5: 0.XXX
    for k in ["SR@5", "SR@10", "SR@15", "AvgT", "hDCG"]:
        m = re.search(rf"{re.escape(k)}[:\s]+([0-9.]+)", text)
        if m:
            metrics[k] = float(m.group(1))
    return metrics

if __name__ == "__main__":
    log_path = sys.argv[1]
    out_path = sys.argv[2]
    metrics = parse_log(log_path)
    print(json.dumps(metrics, indent=2))
    Path(out_path).write_text(json.dumps(metrics, indent=2))
```

**注意**：parse 的正则必须根据 TAIRA 实际输出格式修改，不能假定格式。

---

## 九、与 InterRec 的设置对齐

TAIRA 和 InterRec 的数据域不同（Amazon vs LastFM），但需要尽量对齐以下设置：

| 设置项 | InterRec | TAIRA | 对齐方案 |
|--------|----------|-------|----------|
| 推荐 top-K | 10 | `TOPK_ITEMS=10` | 直接对齐 ✅ |
| 候选 item 池 | 全体 items | `TOPN_ITEMS=500` | 在报告中说明差异 |
| 最大对话轮数 | 5 | 视数据集而定 | 在报告中说明 |
| 成功判断 | target in top-K recs | 视官方逻辑 | 必须在报告说明 |
| 用户模拟器 | InterRec UserSimulator（softmax choice） | TAIRA user_simulate/ | 不同，报告说明 |
| 评估指标 | HitRate / NDCG / MRR + SR / AvgT / hDCG | SR / AvgT 等 | 主表用 SR / AvgT / hDCG |

**差异说明要求**：

在 `taira_reproduction_report.md` 中必须明确列出所有无法对齐的设置，
并说明这些差异对指标的潜在影响。不能在没有说明的情况下直接比较。

---

## 十、关于 Google Search API

TAIRA 的 Searcher Agent 依赖 Google Custom Search API。

如遇 API 访问限制：

1. **方案A（推荐）**：使用官方的 API Key，正常运行；
2. **方案B（替代）**：如 Google API 不可用，可以修改 Searcher Agent 使用
   本地 BM25 搜索替代（`/home/yurh/interrec/src/embedding/bm25_index.py` 已实现），
   但**必须在报告中明确标注**此替代对结果的影响；
3. **禁止**：完全移除 Searcher Agent 而不说明。

---

## 十一、禁止事项

严禁：

1. 自己写 single-pass LLM 方法然后声称是 TAIRA；
2. 省略 TPD（Thought Pattern Distillation）机制；
3. 省略 hierarchical planning（用 Manager Agent 的分解逻辑）；
4. 修改 TAIRA 的用户模拟器；
5. 用 mock LLM 替代真实 GPT-4o 调用（TAIRA 结果必须来自真实 LLM）；
6. 不说明就更换 LLM 模型（如用 GPT-3.5 替代 GPT-4o）；
7. 不说明就更换数据集；
8. 用 TAIRA 本身的训练集物品评估；
9. 不说明 Google Search API 替换情况。

---

## 十二、多随机种子

主表至少跑：

```
seed = 0, 1, 2
```

如 TAIRA 不支持 seed 参数，必须：
1. 在 code review 报告中说明；
2. 通过 `PYTHONHASHSEED` 环境变量尝试控制随机性；
3. 多次运行取均值，并在报告中说明方差来源。

保存：`mean ± std`。

---

## 十三、复现报告

输出：`taira_official/docs/taira_reproduction_report.md`

必须包含：

1. 官方 repo commit hash（`git rev-parse HEAD`）；
2. 使用的 LLM 模型版本（GPT-4o 具体版本）；
3. 是否使用 Google Search API（或替代方案）；
4. Python + 依赖版本；
5. 数据集及版本；
6. 使用的配置参数（`system_config.yaml` 完整内容）；
7. 每个 seed 的结果；
8. mean ± std；
9. 与论文报告结果是否一致；
10. 不一致的可能原因（LLM API 随机性、数据版本、Google API 替换等）；
11. 与 InterRec 设置的对齐/差异说明；
12. LLM API 调用总费用估算；
13. 是否做过任何 patch（必须附 patch 文件）。

---

## 十四、与 InterRec / MCMIPL 比较输出格式

最终结果需要写入 `/home/yurh/main_table_experiments/comparison/results/` 下，
格式与 MCMIPL / InterRec 对齐：

```csv
method,dataset,seed,SR@5,SR@10,SR@15,AvgT,hDCG
TAIRA,amazon_music,0,0.XXX,0.XXX,0.XXX,X.XX,X.XX
TAIRA,amazon_music,1,0.XXX,...
TAIRA,amazon_music,2,0.XXX,...
```

实现结果写入脚本：

```python
# taira_official/scripts/collect_taira_results.py
"""Read parsed metric JSONs for all seeds and write to comparison/results/taira_results.csv."""
import json, csv
from pathlib import Path

metric_dir = Path("results/metrics")
rows = []
for f in sorted(metric_dir.glob("run_*_seed*.json")):
    parts = f.stem.split("_")  # run_amazon_music_seed0
    seed = int(parts[-1].replace("seed", ""))
    domain = "_".join(parts[1:-1])
    m = json.loads(f.read_text())
    rows.append({"method": "TAIRA", "dataset": domain, "seed": seed, **m})

out = Path("../../../../comparison/results/taira_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"Written {len(rows)} rows to {out}")
```

---

## 十五、最终交付物清单

```
taira_official/
  TAIRA/                                  # 官方 repo（完整克隆，不修改核心）
  scripts/
    setup_env.sh                          # ✅ 环境安装
    prepare_data.sh                       # ✅ 数据准备
    run_experiment.sh                     # ✅ 单次运行（domain + seed）
    run_all.sh                            # ✅ 多 seed 批量运行
    parse_taira_metrics.py                # ✅ 日志解析
    collect_taira_results.py              # ✅ 结果汇总
  configs/
    taira_reproduce.yaml                  # ✅ 复现配置（无 API key）
  results/
    raw_logs/
      run_amazon_music_seed0.log          # 完整 stdout/stderr
      run_amazon_music_seed1.log
      run_amazon_music_seed2.log
      env_versions.txt                    # 环境版本记录
    metrics/
      run_amazon_music_seed0.json         # 解析后的指标
      run_amazon_music_seed1.json
      run_amazon_music_seed2.json
  docs/
    taira_code_review.md                  # ✅ 代码 review 报告
    taira_reproduction_report.md          # ✅ 最终复现报告
```

---

## 十六、对比实验逻辑链

```
TAIRA baseline:

官方 repo (https://github.com/Alcein/TAIRA)
  → 官方 amazon_music 数据
  → Manager Agent + Executor Agents (Searcher, Item Retriever, ...)
  → Thought Pattern Distillation (TPD)
  → hierarchical planning + dynamic replanning
  → GPT-4o LLM + Google Search API
  → 官方用户模拟器 (user_simulate/)
  → SR@K / AvgT / hDCG 评估

InterRec（我们的方法）:

同域数据（尽量对齐 amazon_music 或 LastFM）
  → TF-IDF/BGE item embedding
  → Bayesian belief N(μ, Σ)
  → Σ 特征分解 → LLM Agent 生成 intent hypothesis
  → IG / VOI 决策提问
  → Laplace 近似 belief 更新
  → 推荐
  → 相同评估指标

主表对比:

TAIRA (LLM multi-agent + TPD)
  vs
InterRec (Bayesian + LLM intent elicitation)
  vs
MCMIPL (RL + multiple choice questions)

输出: main_table_mean_std.csv
```
