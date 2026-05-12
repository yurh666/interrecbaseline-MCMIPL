# InterRec Baseline 复现 Prompt

> **这份文档的定位**：这是一个可以直接作为 Prompt 给 AI Agent 使用的上下文文档。
> 拿到这份文档后，在它后面追加"请帮我实现 XXX baseline"，AI 就能知道所有必要的背景信息，
> 直接写出与 InterRec 框架兼容的 baseline 代码。
>
> **项目路径**：`/home/yurh/interrec`（下文所有相对路径均以此为根）

---

## 一、项目背景

我们正在研究一个名为 **InterRec** 的交互式推荐系统，核心思想是用贝叶斯信念（Gaussian Belief）
维护用户偏好不确定性，通过 LLM Agent 生成选择题，用信息增益（VOI）决定何时提问，
用 Laplace 近似做贝叶斯后验更新。

现在需要**实现若干 baseline 方法**与 InterRec 在相同数据、相同切分、相同指标下进行比较。

已有代码框架：`/home/yurh/interrec/src/`，包含数据加载、embedding、评估指标、日志系统等
公共模块，**baseline 直接复用即可，不要重新实现**。

---

## 二、可直接复用的已有模块

### 2.1 数据加载

```python
import sys
sys.path.insert(0, '/home/yurh/interrec')
from src.data.dataset import InterRecDataset
from src.utils.config import load_config

cfg = load_config('/home/yurh/interrec/configs/shared_experiment_config.yaml')
dataset = InterRecDataset.load('/home/yurh/interrec/data/processed')

# dataset.sessions      → list of session dicts（每个用户一个）
# dataset.items         → pd.DataFrame，2665 行，cols: item_id, artist_name, tags, description...
# dataset.interactions  → pd.DataFrame，67098 行，cols: user_id, item_id, timestamp, play_count
```

每个 session 的结构（来自 `data/processed/sessions.json`）：
```python
session = {
    "user_id": "10",
    "episode_id": "10_session",
    "observed_history": ["454", "453", "51", ...],   # 前 40%，可使用
    "future_train":  ["12", "34", ...],              # 42%，仅用于训练
    "future_valid":  ["56", ...],                    # 6%，仅用于调参
    "future_test":   ["78", "90", ...],              # 12%，仅用于最终评估
}
# future_test 是评估的 ground truth，绝对不能提前给系统看
```

### 2.2 Embedding Index（TF-IDF+SVD，128维，已构建）

```python
from src.embedding.index import EmbeddingIndex

index = EmbeddingIndex.load('/home/yurh/interrec/data/processed')
# index.embeddings          → np.ndarray, shape=(2665, 128)
# index.item_ids            → list[str], 2665 个 item_id
# index.item_id_to_index    → dict[str, int]

# 向量检索（余弦相似度，返回 top_k 个 (item_id, score)）
results = index.search(query_vec, top_k=10, exclude={'seen_item_1', 'seen_item_2'})

# 取单个 item 的向量
vec = index.vector_for('454')

# 批量取向量
vecs = index.vectors_for(['454', '51', '12'])  # shape=(3, 128)
```

### 2.3 BM25 Index（关键词检索）

```python
from src.embedding.bm25_index import BM25Index

bm25 = BM25Index.build(dataset.items)
results = bm25.search("rock alternative indie", top_k=10, exclude=set())
# returns: [('item_id', score), ...]
```

### 2.4 评估指标

```python
from src.recommendation.metrics import ranking_metrics

relevant = set(str(x) for x in session['future_test'])  # ground truth
recommended = ['item_1', 'item_2', ..., 'item_10']       # 你的推荐结果

m = ranking_metrics(recommended, relevant, k=10)
# → {'HitRate@10': 0.0, 'NDCG@10': 0.0, 'MRR@10': 0.0}
m5 = ranking_metrics(recommended, relevant, k=5)
# → {'HitRate@5': 0.0, 'NDCG@5': 0.0, 'MRR@5': 0.0}
```

### 2.5 LLM Client（支持 mock 和真实 API，统一接口）

```python
from src.agents.llm_client import LLMClient

llm = LLMClient(
    mode=cfg['llm']['mode'],       # 'mock' 或 'openai'
    provider=cfg['llm']['provider'],
)

# 发送消息（mock 模式返回预设 JSON，不需要 API key）
response = llm.chat([
    {"role": "system", "content": "You are a music recommender."},
    {"role": "user", "content": "..."},
])

# 直接解析 JSON 响应
data = llm.structured_json([...])
```

### 2.6 日志系统

```python
from src.logging.run_logger import RunLogger
from src.utils.time import make_run_id

run_id = make_run_id('lastfm', 'my_baseline_name', seed=42)
logger = RunLogger(run_id, base_dir='/home/yurh/interrec/experiments/runs')
logger.write_config(cfg)
logger.write_environment()

# 每轮记录（详见第五节的日志格式要求）
logger.log_turn(turn_log_dict)

# 最终写入指标
logger.write_metrics(aggregate_dict, all_results=[])
```

### 2.7 工具函数

```python
from src.utils.math import l2_normalize, softmax, cosine, weighted_average
from src.utils.seed import set_seed

set_seed(42)  # 固定随机种子
```

---

## 三、数据说明

### 数据集信息

| 项目 | 值 |
|------|----|
| 数据集 | hetrec2011-lastfm-2k |
| 物品粒度 | Artist（艺术家）非单曲 |
| 过滤后用户数 | 1648 |
| 过滤后物品数 | 2665 |
| 总交互数 | 67098 |
| 平均每用户交互 | 40.7 |

### 切分规则（已执行，勿修改）

```
observed_history : 前 40%   → 系统可见，用于 profile 初始化
future_train     : 约 42%   → 仅用于训练/调参
future_valid     : 约  6%   → 仅用于验证/超参选择
future_test      : 约 12%   → 【唯一评估来源】，绝对不可提前暴露给系统
```

### Items 文件列说明（`data/processed/items.csv`）

| 列 | 内容 | 示例 |
|----|------|------|
| `item_id` | artistID（字符串） | `"454"` |
| `title` | 艺术家名 | `"Radiohead"` |
| `artist_name` | 艺术家名（同上） | `"Radiohead"` |
| `tags` | 用户标注的 tag（空格分隔） | `"alternative rock indie"` |
| `description` | 自动生成的描述文本 | `"Radiohead. Tags: alternative rock indie"` |

---

## 四、评估协议（所有方法统一）

### 4.1 主要指标

| 指标 | 定义 | 说明 |
|------|------|------|
| `HitRate@K` | top-K 推荐中是否命中 future_test 中任意物品 | K=5,10 |
| `NDCG@K` | 位置加权的命中率 | K=10 |
| `MRR@K` | 最高命中位置的倒数 | K=10 |

### 4.2 与 MCMIPL 对比所需指标（对话式方法使用）

MCMIPL 的对话轮次上限是 **15 轮**，成功条件是系统在 top-10 推荐中命中用户目标物品。

| 指标 | 定义 |
|------|------|
| `SR@5` | 前 5 轮内成功率 |
| `SR@10` | 前 10 轮内成功率 |
| `SR@15` | 前 15 轮内成功率（主指标） |
| `AvgT` | 成功用户平均轮次（未成功用户计为 max_turns） |
| `hDCG` | `1/log2(t+3) + (1/log2(t+2)-1/log2(t+3)) / log2(done+1)`，t 是成功轮次 |

**MCMIPL 当前已复现的参考结果（seed 0/1/2 均值）**：
```
SR@5=0.417,  SR@10=0.773,  SR@15=0.860,  AvgT=7.19,  hDCG=0.349
```

### 4.3 评估用的 ground truth

```python
# 对于非对话式方法（只推荐一次）：
relevant = set(str(x) for x in session['future_test'])

# 对于对话式方法（多轮推荐）：
# 每轮都用 future_test 作为 target，在 max_turns 轮内命中即为成功
```

---

## 五、日志格式（必须遵守）

每个 baseline 的 `full_log.jsonl` 每行一条记录，**至少包含以下字段**：

```json
{
  "run_id":  "20260507_210000_lastfm_bm25_seed42",
  "method":  "bm25_baseline",
  "dataset": "lastfm",
  "seed":    42,
  "user_id": "10",
  "turn":    1,

  "data_split": {
    "observed_history_count": 18,
    "theta_star_source": "future_test",
    "future_test_count": 7
  },

  "implementation_modes": {
    "item_encoder":   "tfidf_svd",
    "llm":            "not_used",
    "baseline_mode":  "official"
  },

  "recommendation": {
    "recommended_items": ["454", "51", "12", "..."],
    "HitRate@5":  0.0,
    "HitRate@10": 0.0,
    "NDCG@10":    0.0,
    "MRR@10":     0.0
  },

  "metrics": {
    "HitRate@10":      0.0,
    "NDCG@10":         0.0,
    "MRR@10":          0.0,
    "ask_count_so_far": 0
  }
}
```

`metrics.csv` 最终格式：
```csv
method,dataset,seed,encoder_mode,llm_mode,HitRate@5,HitRate@10,NDCG@10,MRR@10,SR@5,SR@10,SR@15,AvgT,hDCG,ask_count,n_users
bm25_baseline,lastfm,42,tfidf_svd,not_used,0.12,0.23,0.08,0.07,,,,,, 0,1648
```

---

## 六、已实现的 Baseline 参考

以下 baseline 已在 `src/simulation/experiment_runner.py` 中实现，可参考其写法：

### BM25 Baseline（已实现，参考代码）

```python
# src/simulation/experiment_runner.py → run_bm25_baseline()
def run_bm25_baseline(cfg, dataset, index, logger, bm25_index):
    for session in sessions:
        observed = session['observed_history']
        query = " ".join(str(x) for x in observed[-20:])   # 用最近 20 条历史构建 query
        relevant = set(str(x) for x in session['future_test'])
        seen = set(str(x) for x in observed)
        results = bm25_index.search(query, top_k=10, exclude=seen)
        rec_ids = [r[0] for r in results]
        m = ranking_metrics(rec_ids, relevant, k=10)
        # ... log and aggregate
```

运行方式：
```bash
cd /home/yurh/interrec
python scripts/run_main_experiment.py --method bm25 --seed 42
```

---

## 七、新 Baseline 的实现模板

实现一个新 baseline，需要创建 `src/baselines/{method_name}.py`，并在
`scripts/run_main_experiment.py` 中注册。

**最小实现模板**：

```python
# src/baselines/my_baseline.py
from __future__ import annotations
from typing import Any
from tqdm import tqdm

from src.data.dataset import InterRecDataset
from src.embedding.index import EmbeddingIndex
from src.logging.run_logger import RunLogger
from src.recommendation.metrics import ranking_metrics
from src.utils.time import make_run_id


def run_my_baseline(
    cfg: dict[str, Any],
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    logger: RunLogger,
    **kwargs,
) -> dict[str, Any]:
    """实现你的 baseline 逻辑，返回聚合指标 dict。"""
    sim_cfg = cfg.get('simulation', {})
    split_key = 'future_test'   # 评估固定用 future_test
    top_k = int(cfg.get('recommendation', {}).get('top_k', 10))
    max_users = cfg.get('simulation', {}).get('max_users') or len(dataset.sessions)

    all_hr, all_ndcg, all_mrr = [], [], []

    for session in tqdm(dataset.sessions[:max_users], desc='My Baseline'):
        observed = session['observed_history']
        relevant = set(str(x) for x in session.get(split_key, []))
        seen = set(str(x) for x in observed)

        # ── 实现你的推荐逻辑 ──────────────────────────────
        rec_ids = my_recommend(session, observed, index, seen, top_k)
        # ──────────────────────────────────────────────────

        m = ranking_metrics(rec_ids, relevant, k=top_k)
        all_hr.append(m[f'HitRate@{top_k}'])
        all_ndcg.append(m[f'NDCG@{top_k}'])
        all_mrr.append(m[f'MRR@{top_k}'])

        # 记录日志（每轮）
        logger.log_turn({
            'run_id': logger.run_id,
            'method': 'my_baseline',
            'dataset': cfg.get('dataset', {}).get('name', 'lastfm'),
            'seed': cfg.get('seed', 42),
            'user_id': session['user_id'],
            'turn': 1,
            'data_split': {
                'observed_history_count': len(observed),
                'theta_star_source': split_key,
                'future_test_count': len(session.get('future_test', [])),
            },
            'implementation_modes': {
                'item_encoder': cfg.get('embedding', {}).get('mode', 'tfidf_svd'),
                'llm': 'not_used',
                'baseline_mode': 'reproduction',  # official / reproduction / simplified
            },
            'recommendation': {
                'recommended_items': rec_ids,
                **ranking_metrics(rec_ids, relevant, k=5),
                **m,
            },
            'metrics': {**m, 'ask_count_so_far': 0},
        })

    def mean(lst): return float(sum(lst) / max(len(lst), 1))

    aggregate = {
        'method': 'my_baseline',
        'dataset': cfg.get('dataset', {}).get('name', 'lastfm'),
        'seed': cfg.get('seed', 42),
        'encoder_mode': cfg.get('embedding', {}).get('mode', 'tfidf_svd'),
        'llm_mode': 'not_used',
        f'HitRate@{top_k}': mean(all_hr),
        f'NDCG@{top_k}': mean(all_ndcg),
        f'MRR@{top_k}': mean(all_mrr),
        'ask_count': 0,
        'n_users': len(all_hr),
    }
    logger.write_metrics(aggregate, [])
    return aggregate
```

---

## 八、各类 Baseline 的实现要点

### A. 纯检索 Baseline（BM25 / BGE-M3 / Reranker）

- **不进行任何交互**，不提问，不更新信念
- 只用 `observed_history` 构建 query
- 评估时只算 `HitRate@K / NDCG@K / MRR@K`（无 SR 相关指标）
- `ask_count = 0`

**BM25** 已实现，参考 `src/simulation/experiment_runner.py:run_bm25_baseline()`

**BGE-M3 Dense Retrieval**：
```python
# 用 observed_history 的 embedding 均值作为 query vector
from src.utils.math import l2_normalize, weighted_average
vecs = index.vectors_for(session['observed_history'])
weights = dataset.weights_for_items(session['user_id'], session['observed_history'])
query_vec = l2_normalize(weighted_average(vecs, weights).reshape(1,-1))[0]
results = index.search(query_vec, top_k=10, exclude=seen)
```

### B. LLM Planning Baseline（Zero-shot / CoT / ReAct 等）

- 使用 `LLMClient`（`src/agents/llm_client.py`）
- Prompt 里只能包含 `observed_history` 中的物品，**不能包含** `future_test` 物品
- 每次 LLM 调用必须在日志里记录 `implementation_modes.llm = "gpt-4o-mini"` 等
- 非对话式（单次推荐）：和纯检索一样，`ask_count = 0`
- 对话式（多轮）：需记录每轮的 `ask_count_so_far`

### C. 对话式交互 Baseline（TAIRA / InteRecAgent / MACRS）

- 多轮结构：每轮可以提问或推荐
- 每轮都要记录 `turn`、`ask_count_so_far`
- 最大轮次与 InterRec 对齐：`max_turns: 15`（来自共享配置）
- 需要计算 `SR@K / AvgT / hDCG`（成功 = top-10 推荐命中 future_test）

---

## 九、运行和提交结果

### 运行命令

```bash
cd /home/yurh/interrec
# 设置 API Key（如果需要 LLM）
export OPENAI_API_KEY="sk-..."

# 运行（改成你的 baseline 名称和脚本）
python scripts/run_main_experiment.py \
    --config configs/shared_experiment_config.yaml \
    --method my_baseline \
    --seed 42

# 正式实验需要跑 3 个 seed
for seed in 0 1 2; do
    python scripts/run_main_experiment.py \
        --config configs/shared_experiment_config.yaml \
        --method my_baseline --seed $seed
done
```

### 结果汇总

```bash
cd /home/yurh/interrec
python3 -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob('experiments/runs/*/metrics.csv')]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('outputs/metrics/main_table.csv', index=False)
print(combined[['method','seed','HitRate@10','NDCG@10','MRR@10']].to_string(index=False))
"
```

---

## 十、约束清单

实现任何 baseline 必须满足以下所有条件：

```
【数据使用】
  ✅ 评估必须使用 future_test 作为 ground truth
  ✅ 训练/调参只能用 future_train / future_valid
  ✅ observed_history 可以完整使用
  ❌ future_test 物品不能出现在任何 prompt 或模型输入中

【配置固定】
  ✅ 使用 /home/yurh/interrec/configs/shared_experiment_config.yaml
  ✅ seed 跑 0、1、2 三次，报告均值 ± 标准差
  ✅ top_k = 10（主表），评估 K=5 和 K=10

【日志要求】
  ✅ full_log.jsonl 必须有第五节规定的字段
  ✅ implementation_modes.baseline_mode 必须填写
  ✅ LLM 模型名必须记录（不用 LLM 填 "not_used"）

【代码要求】
  ✅ 放在 src/baselines/{method_name}.py
  ✅ 直接复用已有的 EmbeddingIndex / BM25Index / RunLogger / ranking_metrics
  ✅ 不修改 data/processed/ 中的任何文件
  ❌ 不硬编码 API Key
  ❌ 不修改 src/recommendation/metrics.py（指标定义统一）
```
