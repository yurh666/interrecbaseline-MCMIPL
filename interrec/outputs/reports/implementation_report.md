# InterRec 实现细节报告

> **报告目的**：说明当前代码实现状态、各模块如何工作、以及"能跑 smoke test"和"跑出论文级结果"之间的具体差距。

---

## 一、当前状态总览


| 模块                          | 当前状态               | 能否运行 | 论文级质量？      |
| --------------------------- | ------------------ | ---- | ----------- |
| 数据预处理                       | ✅ 完整实现             | ✅    | ✅           |
| TF-IDF+SVD embedding        | ✅ 完整实现             | ✅    | ⚠️ 质量低于神经网络 |
| BM25 检索 baseline            | ✅ 完整实现             | ✅    | ✅（纯 BM25）   |
| Bayesian belief state       | ✅ 完整实现             | ✅    | ✅           |
| Uncertainty direction       | ✅ 完整实现             | ✅    | ✅           |
| IG / VOI 框架                 | ✅ 完整实现             | ✅    | ✅           |
| 模拟用户 (theta*)               | ✅ 完整实现             | ✅    | ✅           |
| Laplace belief update       | ✅ 完整实现             | ✅    | ✅           |
| 日志系统                        | ✅ 完整实现             | ✅    | ✅           |
| Direction Translator (LLM)  | ⚠️ mock/rule-based | ✅    | ❌ 需要真实 LLM  |
| Hypothesis Generator (LLM)  | ⚠️ mock/rule-based | ✅    | ❌ 需要真实 LLM  |
| Option Writer (LLM)         | ⚠️ template        | ✅    | ❌ 需要真实 LLM  |
| BGE-M3 embedding            | ❌ 接口占位             | ❌    | -           |
| LLM Planning baselines      | ❌ stub 占位          | ❌    | -           |
| Interactive Agent baselines | ❌ stub 占位          | ❌    | -           |
| 消融实验                        | ❌ 未实现              | ❌    | -           |


**结论：代码框架完整，最小闭环可运行。要跑出论文指标，需要接入真实 LLM 和更好的 Embedding。**

---

## 二、环境已配置的内容

```
Python 3.13.12 (Anaconda)
GPU: NVIDIA GeForce RTX 3060 Ti, 8192 MiB
已安装：numpy 2.4.4 / pandas 3.0.2 / scikit-learn 1.8.0 /
        rank-bm25 0.2.2 / tqdm 4.67.3 / pyyaml 6.0.3 /
        scipy (numpy 依赖，已可用)
sentence-transformers: 正在安装中（后台）
```

⚠️ **已知问题**：本机的 `numpy 2.4.4` 使用 `scipy_openblas64`（64-bit integer BLAS），导致 `np.linalg.inv/solve` 异常慢（128×128 矩阵需 400ms，正常应 < 1ms）。
**已修复**：所有矩阵运算全部改用 `scipy.linalg.cho_factor/cho_solve`，实际速度恢复正常（约 0.8s/turn）。

---

## 三、数据准备（已完成）

### 数据来源

- **数据集**：hetrec2011-lastfm-2k（公开数据集，已下载）
- **下载地址**：`https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip`
- **物品粒度**：artist（艺术家）作为 item，非单曲

### 切分结果

```
总用户：1892 → 过滤后 1648 用户
总 item：17632 → 过滤后 2665 个高频 artist
总交互：67098 条（已切分）
```

### 切分比例（严格执行，无泄露）

```
observed_history: 前 40%
future_train:     后 60% 中的 70% ≈ 总量 42%
future_valid:     后 60% 中的 10% ≈ 总量  6%
future_test:      后 60% 中的 20% ≈ 总量 12%
```

---

## 四、各模块实现细节

### 4.1 Embedding（当前：TF-IDF+SVD）

**实现**：TF-IDF（max_features=50000, ngram=(1,2)）+ TruncatedSVD（dim=128）

**Item 文本格式**：

```
title: {artist_name}
artist: {artist_name}
category: 
tags: {tag1} {tag2} ...    ← 来自 user_taggedartists
description: {artist_name}. Tags: {tags}
```

**已产出文件**：

```
data/processed/item_embeddings.npy  shape=(2665, 128)
data/processed/item_id_to_index.json
data/processed/item_index.pkl
```

**与论文级的差距**：TF-IDF 只做词频匹配，无法捕捉语义相似度（如 "rock" vs "metal" 被视为完全不同）。换 BGE 后 embedding 质量会显著提升。

---

### 4.2 Bayesian Belief State（✅ 完整）

```python
b_t(θ) = N(θ | μ_t, Σ_t)
μ_t ∈ R^128,  Σ_t ∈ R^{128×128}
```

- **初始化**：`μ_0 = normalize(Σ_i log(1+play_count_i) * e_i)`，仅用 observed_history
- **熵**：`H = 0.5*(d*log(2πe) + log|Σ|)`，用 scipy cho_factor 计算 logdet
- **特征分解**：用 `scipy.linalg.eigh` 提取 top-K 主方向

---

### 4.3 Uncertainty Direction + Anchor（✅ 完整）

```
对 Σ_t 做特征分解，取 top-K 特征向量 v_k
对每个方向：s_{ik} = <e_i - μ_t, v_k>
I_k^+ = top-5 items（正向）
I_k^- = bottom-5 items（负向）
```

---

### 4.4 Hypothesis Generation（⚠️ 当前 mock）

**当前（mock/rule-based）**：

- 每个不确定性方向生成 1 个正向 + 1 个负向假设
- `text_description = "User may prefer: {positive_side}"`
- `feature_signature = positive_side.split()[:4]`

**论文级需要**：接入真实 LLM，基于 anchor item 列表和用户历史摘要，动态推理出语义丰富、context-aware 的 intent 假设。

---

### 4.5 IG / VOI 计算（✅ 完整）

**MC 估计 P(option_i)**：

```python
# 从 N(μ, Σ) 采样 S 个 θ^(s)（Cholesky 采样，~8ms）
P̂(h_i) = (1/S) Σ_s softmax(<θ^(s), v_k>/τ)_i
```

**Laplace 后验更新**：Newton 迭代（3步）求 MAP，用 `scipy.cho_factor` 解方程，避免 `np.linalg.inv`。

**entropy_only 模式**：IG 估计时只需 `log|Σ_new| = -log|H_neg|`，无需形成完整 Σ，通过 Cholesky logdet 直接算出。

**VOI 决策**：`VOI = IG(Q*) - c_ask`，仅 `VOI > 0` 才提问。

---

### 4.6 模拟用户（✅ 完整）

```python
# θ* 来源严格对应评估阶段（test/valid/train）
theta* = normalize(Σ_i log(1+play_count_i) * e_i)  # 仅用 future_test

# 选择概率
sim_i = cos(θ*, v_{option_i})
logit_none = none_bias + (none_threshold - max_sim) / τ_none
probs = softmax([logit_1, ..., logit_n, logit_none])
selected = argmax(probs)  # deterministic_argmax（正式实验默认）
```

---

### 4.7 日志系统（✅ 完整）

每次实验自动产生：

```
experiments/runs/{timestamp}_{dataset}_{method}_seed{N}/
  config.yaml          ← 完整配置快照
  environment.txt      ← Python 版本 + 包列表
  git_commit.txt       ← Git commit
  full_log.jsonl       ← 每轮详细 JSON 日志
  metrics.csv          ← 聚合指标（1行）
  metrics_by_turn.csv  ← 每用户每轮指标
  run_summary.json     ← 运行摘要
  artifacts/           ← 大型张量（option vectors 等）
```

---

## 五、真正跑出论文级结果需要的准备

### 5.1 必须有真实 LLM API

**影响模块**：DirectionTranslator、HypothesisGenerator、OptionWriter

**如何接入**（修改 `configs/default.yaml`）：

```yaml
llm:
  mode: openai          # 改为 openai 或 anthropic
  provider: openai
```

**设置环境变量**：

```bash
export OPENAI_API_KEY="sk-..."
```

**推荐模型**：`gpt-4o-mini`（成本低）或 `gpt-4o`（质量高）。

**当前 mock 的影响**：

- Direction 描述是 "items like: 123, 456"，而非真实语义（"upbeat electronic music for workouts"）
- Hypothesis 是 positive_side 文字的简单重复，不反映真实 intent 推理
- 这会导致 IG 计算偏差、选项质量差，最终 HitRate 偏低

---

### 5.2 需要更好的 Embedding（推荐 BGE-M3）

**当前问题**：TF-IDF 对 LastFM 的 artist tag（数字 ID 或短词）效果差，embedding 语义分辨力弱。

**准备 GPU 环境**：本机已有 RTX 3060 Ti（8G），直接可用。

**安装**（正在安装中）：

```bash
pip install sentence-transformers
```

**切换方式**（修改 `configs/default.yaml`）：

```yaml
embedding:
  mode: bge
  bge_model: BAAI/bge-m3   # 或 BAAI/bge-large-zh, BAAI/bge-small-en
  dim: 128                  # BGE-M3 原始 1024 维，SVD 降到 128
```

**注意**：BAAI/bge-m3 原始维度 1024，首次运行会自动从 HuggingFace 下载（约 2.4GB）。8G 显存足够运行推理（batch=32）。

---

### 5.3 需要真实 LastFM 有 track-level 数据（可选但推荐）

**当前问题**：hetrec2011 只有 artist 粒度，没有 track 名。artist-level 推荐粒度较粗。

**推荐替代**：使用 MCMIPL 官方已处理的 `lastfm_star` 数据（`/home/yurh/main_table_experiments/...`），item 粒度更细，还包含 tag graph。

已提供转换脚本：

```bash
python scripts/convert_mcmipl_lastfm.py \
    --mcmipl-dir /home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL \
    --dataset lastfm_star \
    --out data/raw
```

**注意**：MCMIPL 的 item_fea 只有 tag ID（整数），没有 artist/track 名，TF-IDF 效果仍然有限，仍需真实 LLM 配合 tag ID 解释。

---

### 5.4 运行速度预估


| 配置                              | 速度           | 适合场景     |
| ------------------------------- | ------------ | -------- |
| mock LLM + TF-IDF + mc=64       | ~4s/user（5轮） | 调试、验证流程  |
| mock LLM + TF-IDF + mc=128      | ~8s/user     | 超参搜索     |
| real LLM (GPT-4o-mini) + TF-IDF | ~15-30s/user | 含 API 延迟 |
| real LLM + BGE-M3 + mc=128      | ~20-40s/user | 论文实验     |


**全量实验（1648 用户 × 5 轮）**：

- mock 模式：约 1.8 小时
- real LLM 模式：约 7-14 小时（含 API 延迟）

**建议**：正式实验先用 `--max-users 200` 做验证集调参，最终 test 跑全量。

---

### 5.5 对比指标对齐（与 MCMIPL 比较）

MCMIPL 使用的指标是：`SR@5 / SR@10 / SR@15 / AvgT / hDCG`。

当前 InterRec 日志记录的是：`HitRate@5/@10 / NDCG@10 / MRR@10`。

**需要补充实现**（在 `src/recommendation/metrics.py` 中添加）：

- `SR@K`（Success Rate at K turns）= 前 K 轮内是否至少命中 1 次
- `AvgT`（平均成功轮次）
- `hDCG`（hierarchical DCG，需要 MCMIPL 的定义）

这部分可以在论文写作阶段按需补充，当前框架已预留扩展接口。

---

## 六、运行命令速查

```bash
cd /home/yurh/interrec

# === 一次性准备（已完成）===
# 数据已下载并处理完毕：data/processed/ 目录已就绪
# embedding 已构建：data/processed/item_embeddings.npy 已就绪

# === 验证流程跑通（mock 模式，约 20 分钟）===
python scripts/run_main_experiment.py \
    --config configs/default.yaml \
    --max-users 50 --seed 42

# === BM25 baseline（约 10 秒）===
python scripts/run_main_experiment.py \
    --config configs/default.yaml \
    --method bm25 --seed 42

# === 切换到 BGE-M3 embedding（需 GPU + ~30 分钟首次下载）===
python scripts/build_item_embeddings.py \
    --config configs/default.yaml --mode bge

# === 接入真实 LLM（需设置 OPENAI_API_KEY）===
# 修改 configs/default.yaml: llm.mode: openai
python scripts/run_main_experiment.py \
    --config configs/default.yaml \
    --max-users 200 --seed 42

# === 查看结果 ===
cat experiments/runs/<run_id>/run_summary.json
cat experiments/runs/<run_id>/metrics.csv
```

---

## 七、目前还没实现的内容（明确列出）

以下内容当前是 stub 占位，不会影响 InterRec 主流程运行，但论文需要：


| 内容                           | 文件                                          | 工作量估计   |
| ---------------------------- | ------------------------------------------- | ------- |
| BGE-M3 baseline              | `src/baselines/bge_m3_baseline.py`          | 小（接口已有） |
| BGE-Reranker baseline        | `src/embedding/bge_reranker.py`             | 中       |
| Zero-shot / CoT / ReAct LLM  | `src/agents/llm_planning_baselines.py`      | 中-大     |
| TAIRA/InteRecAgent baselines | `src/agents/interactive_agent_baselines.py` | 大       |
| No-IG 消融                     | `src/ablations/no_information_gain.py`      | 小       |
| No-Belief 消融                 | `src/ablations/no_bayesian_belief.py`       | 中       |
| SR@K / AvgT / hDCG 指标        | `src/recommendation/metrics.py`             | 小       |
| 自动 report 完整版                | `src/logging/report_generator.py`           | 中       |


---

## 八、当前 mock 透明性声明

所有以下模块在 `experiments/runs/*/report.md` 中均有记录：


| 模块                  | implementation_mode | 对论文结果的影响                       |
| ------------------- | ------------------- | ------------------------------ |
| LLMClient           | `mock_llm`          | **高**：选项无语义，IG 偏差，HitRate 偏低   |
| DirectionTranslator | `mock`              | **高**：方向描述是 item_id 拼接，无真实语义   |
| HypothesisGenerator | `mock`              | **高**：假设不反映 context，多样性低       |
| OptionWriter        | `template`          | **中**：选项文本质量低                  |
| Item Encoder        | `tfidf_svd`         | **中**：语义相似度低，影响 belief 初始化和 IG |


