# InterRec

**Bayesian-Guided Agent Elicitation for Interactive Recommendation**

## 项目目标

InterRec 使用高斯贝叶斯信念（Gaussian Belief）维护用户偏好的不确定性，由 LLM Agent（当前默认 mock）在线生成 intent-level 选择题，通过 VOI 框架决定何时提问，并用 Laplace 近似做贝叶斯后验更新，从而高效提升推荐质量。

---

## 数据切分说明

对每个用户的交互序列按时间排序后，严格按以下比例切分：

| 分区 | 比例 | 用途 |
|------|------|------|
| `observed_history` | 前 40% | 初始化 belief；构建系统上下文 |
| `future_train` | 后 60% 中的 70% ≈ 42% | 开发阶段调试；不用于最终测试 |
| `future_valid` | 后 60% 中的 10% ≈ 6% | 超参数选择；early stopping |
| `future_test` | 后 60% 中的 20% ≈ 12% | **最终评估**；不可提前暴露 |

`theta*`（模拟用户真实偏好）的来源严格对应评估阶段：
- 测试阶段 → `future_test`
- 验证阶段 → `future_valid`
- 训练阶段 → `future_train`

---

## 快速开始

```bash
cd /home/yurh/interrec
pip install -r requirements.txt
```

### 1. 预处理 LastFM 原始数据

如果是 hetrec2011-lastfm 格式的 TSV 文件，先转换格式：

```bash
python scripts/preprocess_lastfm.py \
    --raw data/raw/userid-timestamp-artid-artname-traid-traname.tsv \
    --out data/raw
```

然后运行通用预处理：

```bash
python scripts/preprocess_dataset.py --config configs/default.yaml
```

输出到 `data/processed/`：
- `interactions.csv` / `items.csv`
- `user_splits.json`（每用户的 observed/train/valid/test 分区）
- `sessions.json`

### 2. 构建 Item Embeddings

```bash
# 默认 TF-IDF + SVD（离线，无需 GPU）
python scripts/build_item_embeddings.py --config configs/default.yaml

# 如有 GPU，可切换 BGE
python scripts/build_item_embeddings.py --config configs/default.yaml --mode bge
```

输出：`data/processed/item_embeddings.npy` / `item_id_to_index.json` / `item_index.pkl`

### 3. 运行 InterRec 主实验

```bash
python scripts/run_main_experiment.py --config configs/default.yaml
```

可选参数：

```bash
--split test         # 使用 future_test 作为 theta* 来源（默认）
--max-users 50       # 只跑前 N 个用户（调试时用）
--seed 42
```

### 4. 运行 BM25 Baseline

```bash
python scripts/run_main_experiment.py --config configs/default.yaml --method bm25
```

---

## 如何查看日志和报告

每次运行自动生成目录：

```
experiments/runs/{run_id}/
  config.yaml          # 完整配置快照
  environment.txt      # Python 版本 + 包列表
  git_commit.txt       # Git commit hash
  run_summary.json     # 运行摘要 + 聚合指标
  full_log.jsonl       # 每轮详细日志（JSONL 格式）
  metrics.csv          # 聚合指标表（一行）
  metrics_by_turn.csv  # 每用户每轮指标
  report.md            # 可读报告
  artifacts/           # 大型张量（option vectors 等）
  figures/
```

手动生成报告：

```bash
python scripts/generate_report.py \
    --run-dir experiments/runs/<run_id> \
    --config configs/default.yaml
```

---

## 如何切换真实 LLM

修改 `configs/default.yaml`：

```yaml
llm:
  mode: openai          # 或 anthropic
  provider: openai
  log_prompts: true
  log_responses: true
```

确保已设置对应的 API key 环境变量（`OPENAI_API_KEY` 等）。

---

## 哪些模块使用了 Mock

以下模块在当前 milestone 默认使用 mock / rule-based 实现，**所有 mock 都会在 `run_summary.json` 和 `report.md` 中显式记录**：

| 模块 | 当前模式 | 说明 |
|------|----------|------|
| `LLMClient` | `mock_llm` | 返回预设 JSON，不调用 API |
| `DirectionTranslator` | `mock` | rule-based 文本拼接 |
| `HypothesisGenerator` | `mock` | 每方向生成固定格式假设 |
| `OptionWriter` | `template` | 直接使用 hypothesis text |
| Item Encoder | `tfidf_svd` | TF-IDF+SVD，非神经网络 |

---

## 目前已实现

- [x] 数据预处理 + 严格时序切分
- [x] TF-IDF+SVD item encoding
- [x] BM25 检索 index
- [x] Gaussian belief state（初始化 / entropy / eigendecompose）
- [x] Laplace 近似 belief update（softmax 似然 + MAP + Hessian）
- [x] Uncertainty direction 识别 + anchor 检索
- [x] Hypothesis 向量化
- [x] Information Gain（Monte Carlo 估计）
- [x] VOI 提问决策
- [x] 模拟用户（deterministic_argmax / stochastic_sample）
- [x] 完整 per-turn JSONL 日志
- [x] run_id / metrics.csv / metrics_by_turn.csv / run_summary.json
- [x] BM25 retrieval baseline

## 暂未实现（后续 milestone）

- [ ] BGE-M3 / BGE-Reranker baseline
- [ ] LLM Planning baselines（Zero-shot / CoT / ReAct / Reflexion 等）
- [ ] Interactive Agent baselines（TAIRA / InteRecAgent / MACRS）
- [ ] 消融实验
- [ ] 真实 LLM 集成（direction translation / hypothesis generation）
- [ ] 自动报告完整版（case study / 失败案例分析）

---

## 如何生成主表指标

```bash
# 汇总多个 run 的 metrics.csv
cat experiments/runs/*/metrics.csv | sort -u > outputs/metrics/main_table.csv
```

---

## 复现官方 MCMIPL Baseline

参见 `/home/yurh/main_table_experiments/README.md`，使用官方代码链路，不与 InterRec 共享环境。
