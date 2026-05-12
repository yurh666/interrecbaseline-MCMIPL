# InterRec 实验环境接入手册

> **本文档用途**：你拿到这份文档时，数据、embedding、配置文件都已准备好。按照本手册操作，可以直接把你的方法接入到 InterRec 的统一评估流程里，不需要重新处理数据或配置环境。

---

## 一、服务器基本信息

```
主机：ubuntu-gpu
GPU：NVIDIA GeForce RTX 3060 Ti（8192 MiB）
Python：3.13.12（Anaconda）
工作目录：/home/yurh/interrec
MCMIPL 所在目录：/home/yurh/main_table_experiments
Choice Diagnostic 目录：/home/yurh/choice_diagnostic
```

---

## 二、已完成的准备工作（无需重做）

以下所有文件均已就绪，可以直接使用：

```
✅ /home/yurh/interrec/data/raw/hetrec2011-lastfm-2k/   ← 原始数据
✅ /home/yurh/interrec/data/raw/interactions.csv         ← 原始转换后交互
✅ /home/yurh/interrec/data/raw/items.csv                ← 原始转换后物品
✅ /home/yurh/interrec/data/processed/interactions.csv   ← 过滤后交互（1648用户）
✅ /home/yurh/interrec/data/processed/items.csv          ← 过滤后物品（2665 artists）
✅ /home/yurh/interrec/data/processed/user_splits.json   ← 用户切分（40/42/6/12%）
✅ /home/yurh/interrec/data/processed/sessions.json      ← Session 列表
✅ /home/yurh/interrec/data/processed/item_embeddings.npy  ← TF-IDF+SVD 128维
✅ /home/yurh/interrec/data/processed/item_id_to_index.json
✅ /home/yurh/interrec/data/processed/item_index.pkl
✅ /home/yurh/interrec/configs/shared_experiment_config.yaml  ← 共享配置
✅ MCMIPL 复现结果（seed 0/1/2）见 main_table_experiments/comparison/results/
```

---

## 三、环境安装

### 3.1 当前基础环境（已安装）

```bash
# 已安装的包
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
rank-bm25==0.2.2
tqdm==4.67.3
pyyaml==6.0.3
scipy（numpy 依赖，已可用）
sentence-transformers==5.4.1
```

### 3.2 安装 InterRec 依赖

```bash
cd /home/yurh/interrec
pip install -r requirements.txt
```

### 3.3 验证安装

```bash
cd /home/yurh/interrec
python3 smoke_test.py
# 预期输出：All smoke tests passed!
```

### 3.4 ⚠️ 已知的 numpy 性能问题和解决方案

本机 `numpy 2.4.4` 使用 `scipy_openblas64`（64-bit integer BLAS），导致：
- `np.linalg.inv(128×128)` 耗时 400ms（正常应 <1ms）
- `np.random.multivariate_normal(128, n=64)` 耗时 240ms

**已在 InterRec 代码中修复**，全部改用 `scipy.linalg.cho_factor/cho_solve`（耗时 0.17ms）。

**如果你在实现 baseline 时需要矩阵运算，请使用：**
```python
from scipy import linalg as spla
# 代替 np.linalg.inv:
c = spla.cho_factor(A, check_finite=False)
inv_A = spla.cho_solve(c, np.eye(A.shape[0]), check_finite=False)
# 代替 np.random.multivariate_normal:
L = spla.cholesky(Sigma, lower=True, check_finite=False)
samples = mu + np.random.randn(n, d) @ L.T
```

---

## 四、加载共享数据（代码示例）

### 4.1 加载共享配置

```python
import sys
sys.path.insert(0, '/home/yurh/interrec')
from src.utils.config import load_config

cfg = load_config('/home/yurh/interrec/configs/shared_experiment_config.yaml')
# 数据路径
processed_dir = cfg['data']['processed_dir']  # /home/yurh/interrec/data/processed
```

### 4.2 加载数据集

```python
from src.data.dataset import InterRecDataset

dataset = InterRecDataset.load('/home/yurh/interrec/data/processed')
print(f"用户数：{len(dataset.sessions)}")       # 1648
print(f"物品数：{len(dataset.items)}")          # 2665
print(f"交互数：{len(dataset.interactions)}")   # 67098

# 访问一个用户的数据
session = dataset.sessions[0]
print(session['user_id'])                       # "10"
print(len(session['observed_history']))         # 约 18 条
print(len(session['future_test']))              # 约 7 条（评估用）
```

### 4.3 加载 Embedding Index

```python
from src.embedding.index import EmbeddingIndex

index = EmbeddingIndex.load('/home/yurh/interrec/data/processed')
print(index.embeddings.shape)  # (2665, 128)

# 获取某个 item 的 embedding
vec = index.vector_for('454')  # artist_id 454 = Radiohead

# 向量检索 top-10
results = index.search(query_vec, top_k=10, exclude={'seen_item_1', 'seen_item_2'})
# returns: [('item_id', score), ...]
```

### 4.4 加载 BM25 Index

```python
from src.embedding.bm25_index import BM25Index

bm25 = BM25Index.build(dataset.items)  # 约 2 秒
results = bm25.search("rock alternative", top_k=10)
```

### 4.5 计算评估指标

```python
from src.recommendation.metrics import ranking_metrics

# relevant = future_test 中的 item_id 集合
relevant = set(str(x) for x in session['future_test'])
recommended = ['item_1', 'item_2', ..., 'item_10']
metrics = ranking_metrics(recommended, relevant, k=10)
# {'HitRate@10': 0.0, 'NDCG@10': 0.0, 'MRR@10': 0.0}
```

---

## 五、接入 LLM API

### 5.1 设置 API 密钥（仅需执行一次）

```bash
# 写入 ~/.bashrc 或当次 session 执行
export OPENAI_API_KEY="sk-..."
# 如果使用 API 代理：
export OPENAI_BASE_URL="https://your-proxy.com/v1"
```

### 5.2 在代码中使用统一 LLM Client

```python
from src.agents.llm_client import LLMClient

# Mock 模式（调试用，不需要 API key）
llm = LLMClient(mode='mock')

# 真实 OpenAI 模式
llm = LLMClient(mode='openai', provider='openai')

# 发送请求
response = llm.chat([
    {"role": "system", "content": "You are a music recommender."},
    {"role": "user", "content": "Recommend artists similar to Radiohead."}
])

# 解析 JSON 响应
data = llm.structured_json([...])
```

### 5.3 在 Config 中切换模式（推荐方式）

修改 `configs/shared_experiment_config.yaml`：
```yaml
llm:
  mode: openai      # 从 mock 改为 openai
  provider: openai
  model: gpt-4o-mini
```

然后在代码中：
```python
cfg = load_config('configs/shared_experiment_config.yaml')
llm = LLMClient(
    mode=cfg['llm']['mode'],
    provider=cfg['llm']['provider'],
)
```

---

## 六、使用日志系统

### 6.1 创建一次 Run

```python
from src.logging.run_logger import RunLogger
from src.utils.time import make_run_id

run_id = make_run_id('lastfm', 'my_baseline', seed=42)
# → "20260507_210000_lastfm_my_baseline_seed42"

logger = RunLogger(run_id, base_dir='/home/yurh/interrec/experiments/runs')
logger.write_config(cfg)           # 保存配置快照
logger.write_environment()         # 保存 Python 环境信息
logger.write_git_info('/home/yurh/interrec')
```

### 6.2 记录每轮日志

```python
# 每一轮结束后调用（格式必须包含以下字段）
turn_log = {
    "run_id": run_id,
    "method": "my_baseline",
    "dataset": "lastfm",
    "seed": 42,
    "user_id": session['user_id'],
    "turn": turn_num,
    "data_split": {
        "observed_history_count": len(session['observed_history']),
        "theta_star_source": "future_test",
    },
    "implementation_modes": {
        "item_encoder": "tfidf_svd",
        "llm": "mock_llm",       # 或 "gpt-4o-mini" 等
        "baseline_mode": "reproduction",
    },
    "recommendation": {
        "recommended_items": [...],
        "HitRate@10": 0.0,
        "NDCG@10": 0.0,
        "MRR@10": 0.0,
    },
    "metrics": {
        "HitRate@10": 0.0,
        "NDCG@10": 0.0,
        "MRR@10": 0.0,
        "ask_count_so_far": 0,
    }
}
logger.log_turn(turn_log)
```

### 6.3 写入最终指标

```python
aggregate = {
    "method": "my_baseline",
    "dataset": "lastfm",
    "seed": 42,
    "encoder_mode": "tfidf_svd",
    "llm_mode": "mock_llm",
    "HitRate@5": 0.12,
    "HitRate@10": 0.23,
    "NDCG@10": 0.08,
    "MRR@10": 0.07,
    "ask_count": 0,
    "n_users": 1648,
}
logger.write_metrics(aggregate, all_results=[])
```

---

## 七、已有 Baseline 的运行方式

### 7.1 BM25 Retrieval Baseline（已实现，可直接运行）

```bash
cd /home/yurh/interrec
python scripts/run_main_experiment.py \
    --config configs/shared_experiment_config.yaml \
    --method bm25 \
    --seed 42
# 约 10 秒
```

### 7.2 InterRec 主方法（Mock LLM 模式）

```bash
cd /home/yurh/interrec
python scripts/run_main_experiment.py \
    --config configs/shared_experiment_config.yaml \
    --max-users 50 \
    --seed 42
# 约 3-4 分钟（50用户 × 5轮）
```

### 7.3 MCMIPL（已有复现结果，无需重跑）

结果已保存在：
```
/home/yurh/main_table_experiments/comparison/results/mcmipl_main_table.csv
SR@5=0.417, SR@10=0.773, SR@15=0.860, AvgT=7.19, hDCG=0.349（3-seed均值）
```

如需重跑：
```bash
cd /home/yurh/main_table_experiments
conda activate mcmipl-reproduce
bash baselines/mcmipl_official/scripts/run_train.sh LAST_FM_STAR 0
bash baselines/mcmipl_official/scripts/run_eval.sh LAST_FM_STAR 100 0
```

### 7.4 BGE-M3 Embedding（sentence-transformers 已安装）

```bash
cd /home/yurh/interrec
# 首次运行会下载约 2.4GB 模型
python scripts/build_item_embeddings.py \
    --config configs/shared_experiment_config.yaml \
    --mode bge
# 产出：data/processed/item_embeddings.npy（BGE-M3 版本）
```

---

## 八、主表汇总

```bash
cd /home/yurh/interrec

python3 - <<'EOF'
import pandas as pd, glob, os

run_dirs = glob.glob('experiments/runs/*/metrics.csv')
dfs = []
for f in run_dirs:
    df = pd.read_csv(f)
    df['run_dir'] = os.path.dirname(f)
    dfs.append(df)

if not dfs:
    print("No runs found yet.")
else:
    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    combined.to_csv('outputs/metrics/main_table.csv', index=False)
    # mean ± std across seeds
    num_cols = combined.select_dtypes('number').columns.tolist()
    grouped = combined.groupby('method')[num_cols].agg(['mean','std'])
    grouped.to_csv('outputs/metrics/main_table_mean_std.csv')
    print(combined[['method','dataset','seed','HitRate@10','NDCG@10','MRR@10']].to_string(index=False))
EOF
```

---

## 九、关键约束备忘卡

```
╔══════════════════════════════════════════════════════════╗
║  必须遵守的规则（违反会导致论文结果不可信）              ║
╠══════════════════════════════════════════════════════════╣
║  ✅ 评估只用 future_test                                 ║
║  ✅ theta* 来源必须记录在日志中                          ║
║  ✅ 所有 run 使用 shared_experiment_config.yaml          ║
║  ✅ seed 必须覆盖 0、1、2 三个值                         ║
║  ✅ LLM 模型名必须记录在 implementation_modes           ║
║  ✅ baseline_mode 必须是 official/reproduction           ║
║  ❌ 禁止用 future_test 训练或调参                       ║
║  ❌ 禁止修改 data/processed/ 中的文件                   ║
║  ❌ 禁止不同方法用不同的用户子集评估                    ║
║  ❌ 禁止在代码里硬编码 API key                          ║
╚══════════════════════════════════════════════════════════╝
```
