# Baseline（MCMIPL）复现与方法对比分析报告

> **撰写依据**：`/home/yurh/interact/methodnew.md`（本方方法草稿）、  
> `/home/yurh/main_table_experiments/comparison/results/reports/report_LAST_FM_STAR.md`、`mcmipl_log_metrics` 对原始 `logs/train_*.log` 的解析。  
> **说明**：Yelp 数值以 `collect_results.py` 从 `logs/train_YELP_STAR_s*.log` 解析为准，见专报 **`report_YELP_STAR_three_seed_unified_honest.md`**。

---

## 零、当前实验进度摘要

| 数据集 | Seed 0 | Seed 1 | Seed 2 |
|--------|--------|--------|--------|
| LAST_FM_STAR | ✅ | ✅ | ✅ |
| YELP_STAR | ✅ | ✅ | ✅ |

**Yelp 三 seed 表 + mean±std**：`comparison/results/reports/report_YELP_STAR_three_seed_unified_honest.md`（`python3 comparison/collect_results.py` 重扫后与 `mcmipl_YELP_STAR_s*.json` 对齐）。  

**脚注（仅自检）**：seed 2 约在 **40/50** RL 步处停，数字仍按与 0/1 相同规则从日志择优；要写「满 50 步」对外稿时再补跑 seed 2。

**Baseline 范围**：**LAST_FM_STAR + YELP_STAR**，各三 seed；BOOK/MOVIE 时序协议 §3.2 单列。

---

# 数据集一：LAST_FM_STAR（已完成，信息完整）

## 1. Baseline（MCMIPL）思路 vs 本方方法思路的主要差别

**MCMIPL（WWW 2022，本复现即官方实现）**：对话式推荐中，系统将交互建模为强化学习（如 DQN）策略在多轮会话上的决策；在每轮可进行 **多项选择题式** 的属性/偏好探询（固定选项结构、`choice_num=4` 等），并结合 **TransE + GCN** 类图表征与 multi-interest 设计，学习目标是在轮次上限内需 **命中推荐列表中的目标物品**（Success Rate）。

**本方方法（见 `methodnew.md`）**：

1. **显式belief**：用连续向量偏好 \(\theta\) 的高斯后验 \(\mathcal{N}(\mu_t,\Sigma_t)\) 表征不确定性；推荐由期望效用 + 覆盖率型探索加分得到。  
2. **What / When / How**：  
   - *What to ask*：不再依赖与数据schema强绑定的**静态**intent集合，而是由 LLM Agent 在高不确定方向 \(\Sigma_t\) 特征向量语义化（锚点商品）的基础上**在线生成**候选 intent hypothesis；  
   - *When to ask*：基于 **Monte Carlo 近似 + Laplace**，用期望信息增益与提问成本 \(c_{\text{ask}}\)（VOI）决定是否提问（可少问、问好问）；  
   - *How to ask*：将 hypothesis 转为用户可读选项并做中立性检查；回答通过 softmax 似然 + Laplace 更新 \(\mu,\Sigma\)（兜底则通胀不确定性）。  

**一言以蔽之**：MCMIPL 用 **端到端离散动作RL**（含题库式MCQ）学习探询——推荐节律；本方用 **结构化贝叶斯信念 + LLM生成的语义假设 + 信息价值决策**把“问什么 / 何时问”从纯RL中剥离并理论化。**二者不在同一可实现栈上**：MCMIPL 在用户模拟器上与官方指标对齐；本方若要在 CRS 上与 MCMIPL 并排，须在**同一环境/划分/度量**下单独实现并跑通。

---

## 2. 实验结果指标、含义与合理性（及对论文量级）

本节数值来自 **`report_LAST_FM_STAR.md`** 中跨三 seed、「各次评测整体均值中按 SR@15 择优」后与官方实现一致的汇总。

### 2.1 指标含义

| 指标 | 含义（本环境） |
|------|----------------|
| **SR@K（K∈{5,10,15}）** | 在至多 `max_turn=15` 轮对话结束前，任一时刻推荐列表出现时，**目标物品是否出现在长度为 K 的推荐列表中的成功率**（对测试集中用户会话取平均）。K 越大越容易成功；**SR@15 为该任务最常用的主成功率**。 |
| **AvgT** | 对上述“成功会话”统计的**平均成功轮数**；越小通常表示以更短对话完成任务（在成功率不降的前提下更有意义）。 |
| **hDCG** | 对推荐序列位置加权的增益型指标（与 Rank 等综合；具体实现以官方 evaluator 为准），反映排序质量。**与 SR 侧重点不同**。 |
| **reward（训练日志）** | RL 训练中环境返回，用于优化策略；论文主表以 **离线测试集 SR 系列**为准，不要将含 `Total epoch_uesr` 的批次均值与主表混用（见原有报告第一章）。 |

### 2.2 数值与是否合理、与论文量级是否一致

**跨三 seed 汇总（节选）**：SR@15 = **0.8410 ± 0.0025**；SR@5 ≈ **0.417**；AvgT ≈ **7.37**；hDCG ≈ **0.342**。  

**与原论文量级**：原有复现报告指出论文 Table 2 上 LastFM-Star 约 **0.874**，本实现低约 **3.3%**。在 CRS 文献与官方代码多种 **eval_num、评测用户数、随机种子与框架版本**差异下，**几个百分点的偏移通常可解释为合理复现区间**；但若要在审稿中作为主表对齐，建议在文中列出：checkpoint 选取规则（与源码一致时已满足）、是否与论文完全相同的评测子采样与步长。

---

## 3. 实施细节：哪些严格按 baseline、为本方方法改过什么、改动能否严谨支持结论

### 3.1 对本 baseline 分支（MCMIPL 复现实验）

依照仓库约定与文档：

- **完全按官方实现**：动作空间；用户模拟器；reward；**train/valid/test 划分（官方随机划分比例）**；TransE/graph 流程；DQN RL 结构与论文设定下的 MCQ 机制；（见 `report_LAST_FM_STAR.md` §二、§二末「数据边界」）。
- **未为“适配本方方法”改写核心算法**：本流水线目的是 **主表可用的强 baseline 数字**，不是半官方魔改。
- **与 InterRec/本方方法并排时的边界**：若 InterRec / 本方使用 **按时间序列切分的 future 预测评测协议**，则与 MCMIPL **官方划分必然不等价**。并排数字时需单独声明：**属协议差异，不能仅从 SR 差额推断算法优劣**。要严谨得出“本方更好”，须在 **转换后同一数据集、同一 rollout 规则** 上重跑双方（或单方+官方模拟器对齐接口）。

### 3.2 BOOK / MOVIE：为与 InterRec「时序协议」可比而准备 MCMIPL 数据（方法层面补充）

上节 §3.1 针对 **LAST_FM_STAR / YELP_STAR** 等：**训练划分沿用官方随机切分**。另有一条与方法论文对齐的路径，用于 **BOOK、MOVIE** 在「**observed / future 时序切分**」下与 InterRec、本方方法做**同一协议**对比——**不改变 MCMIPL 核心 RL、MCQ、模拟器与指标定义**，只改变 **进入 RL 前的交互划分及随后依赖图的中间产物**。

**流程（`interrec/` 与 MCMIPL 仓库协同）**：

1. **原始 MCMIPL 合并字典 → InterRec 风格 CSV**：`interrec/scripts/convert_mcmipl_to_interrec_csv.py`（列表顺序作为时间代理）。
2. **InterRec 预处理**：`python interrec/scripts/preprocess_dataset.py --config configs/preprocess_mcmipl_book.yaml`（及 `..._movie.yaml`），在统一的 `observed_ratio` 与 future 三段比例下生成 `sessions.json`、`user_splits.json` 等。
3. **Movie 与 Book 的阈值差异（仅属数据分布适配，不是改切分公式）**：由 MCMIPL 导出的 **movie** 交互中，单用户序列极短（例如整表最大交互次数约 **18**），无法同时满足与 Book/LastFM 相同的 **`min_user_interactions=20`** 与在 `observed_ratio=0.4` 下所需的 **`min_observed_interactions=10`**（否则迭代过滤后交互为空）。因此在 **`configs/preprocess_mcmipl_movie.yaml` only** 中将阈值放宽为 **`min_user_interactions=15`、`min_observed_interactions=6`** 等与本分布相容的设定；**时间轴上 observed 与未来三段的相对比例不变**。Book 仍可采用与 InterRec 主实验一致的较严阈值。
4. **写回官方 MCMIPL 数据目录**：`export_interrec_sessions_to_mcmipl_book_movie.py` 按 session 覆盖各域下的 **`UI_Interaction_data/review_dict_{train,valid,test}.json`** 与 **`UI_data/{train,test}.pkl`**；**默认不覆盖** **`fea_item/*.pkl`**（物品侧特征文件除非另行重算，保持与原版一致）。
5. **图与交互对齐**：覆盖交互文件后，必须在 MCMIPL 根目录对 **BOOK、MOVIE 分别** 重跑 **`graph_init.py`**，使 **user–item–feature 图及 tmp 中缓存** 与**新区分**一致；若仅用旧图、新 `pkl`，会出现拓扑与训练目标不匹配。**实现注意**：官方 `graph_init.py` 若在 argparse 中未列出 `MOVIE`，需将 **`MOVIE` 纳入 `--data_name` 的合法取值**，否则无法对 movie 域执行图重建（属脚本接口修补，不改变算法）。

**与 §3.1 的关系**：§3.1 = **官方随机划分 baseline**；§3.2 = **为公平对比 InterRec 时序方法而构造的 BOOK/MOVIE 数据管线**。主表或正文并排时须**标明所用划分**（官方随机 vs 时序 future），避免混读 SR。按 **§零** 约定，**本轮 baseline 复现在 Yelp 完成后即不再自动接 BOOK/MOVIE 全量训练**；本节仅保留方法与数据准备说明，供后续单独立项使用。

---

## 4. 从结果看我方 method 要达到什么才算“压住 baseline”，理论上能否更好

### 4.1 实证上建议的判定标准（与 MCMIPL 并排时）

在 **完全一致**的评测协议下，若要令人信服：

1. **主指标**：SR@15 **稳定高于** baseline，且最好能报告 **均值±多 seed** 与显著性检验（或置信区间收窄）。单个 seed 反超不够。  
2. **效率**：在 SR@15 **不降**前提下，AvgT **显著更低**会更强——对应“更会问（VOI）、少冗余轮次”。  
3. **次要**：Rank / hDCG 若一致提升可增强“排序质量好”的主张，但审稿通常仍最看 SR。  
4. **成本与稳定性**：本方依赖 LLM 时，须在论文中交代 **latency、费用、中立性校验失败率**；否则会削弱“可比性”叙事。

粗略经验：若在 LastFM-Star 上对 MCMIPL 的差距为 **≤3–4 个百分点量级**，单方方法展示出 **≥2–5 个点**的持续优势且方差可控，才较容易在主表故事中站住——具体阈值仍取决于审稿与显著性。

### 4.2 理论上是否“应该”优于 MCMIPL？

**可能存在优势的情形**：  
静态 MCQ/离散策略空间对 **intent 粒度与语义覆盖**的表达有限；当用户真实偏好落在 **题库未覆盖的组合**或需要 **语义细粒度分叉**时，RL 要学习到正确探询成本高。本方将 **语义假设空间**交由 LLM 生成，并由 **belief 的不确定性方向与 VOI** 决定问与不问，因而在“假设空间更可塑、冗余提问惩罚明确”的条件下，有机会在 **等价轮上限**内更快收敛到正确检索区域。

**不保证占优的原因**：  
（1）若环境/奖励与 softmax 语义选择模型不匹配，belief 更新会偏；  
（2）LLM 生成不稳定或中立性校验频繁失败 → 噪声反超 RL；  
（3）MCMIPL 在长期大规模 RL 训练中已隐含学习 **接近最优的提问节奏**，而本方若 \(c_{\text{ask}}\) 或采样近似不准会导致 **过少/过多提问**；  
（4）**若没有统一模拟器与用户模型**，两方数字不可严格比较——理论优势无法落成统计结论。

结论：**理论上有合理动机，但是否胜出是实证问题**；必须 **协议对齐 + 多 seed + 消融（去掉 VOI / 去掉 LLM 仅用固定意图等）** 才能严谨地说“是本方法的机制带来增益”。

---

# 数据集二：YELP_STAR

数值来源：`mcmipl_log_metrics` 从完整 eval 均值行中按 **SR@15** 取最优（与 `RL_model` 内 `SR15_best` 一致）。三 seed 并排与 **mean±std** 见 **`report_YELP_STAR_three_seed_unified_honest.md`**；重扫命令：`python3 comparison/collect_results.py`。

### 逐 seed（择优 checkpoint）

| Seed | SR@15 | SR@10 | SR@5 | AvgT |
|------|------:|------:|-----:|-----:|
| 0 | 0.5052 | 0.3284 | 0.0968 | 11.79 |
| 1 | 0.5740 | 0.3788 | 0.0936 | 11.46 |
| 2 | 0.4480 | 0.3240 | 0.1676 | 11.53 |

**三 seed**：SR@15 **0.509 ± 0.052**。

下列小节结构平行于 LAST_FM（从简）。

---

## 1′. Yelp 上下文中 baseline vs 本方方法

与 LastFM 相同：**MCMIPL**仍是 **离散动作 RL + 固定结构 MCQ**；**本方**仍为 **Gaussian belief + Agent 语义假设 + VOI**。Yelp attribute 更丰富、语义空间更大，对本方“可生成语义假设”的一侧可能更友善，但同时也更考验锚点语义化是否稳定——**仅凭 baseline 的数字无法推导**，需跑本方。

---

## 2′. 指标含义与合理性（Yelp）

与 §2 相同定义，仅 **评测用户规模通常为 2500**（参见原报告技术性说明）。

**量级**：论文 Yelp-Star 二手摘要常在 SR@15 ≈ 0.48 附近；本复现三 seed SR@15 汇总 **0.509 ± 0.052**，seed 间有差异，主表脚注可写 mean±std。

---

## 3′. 实施细节（Yelp）

与 LastFM：**官方数据与代码路径**，`max_steps=50`（与 LastFM 的 100 不同属官方设定）；**未发现为本方 method 改写核心 RL/MCMIPL 的实现**——第二数据集仍为纯 baseline。**与 InterRec/本方对齐时同样需声明划分与模拟器一致性**。若主表另含 **BOOK / MOVIE** 上「时序 future」与 InterRec 对齐的 MCMIPL baseline，数据准备路径见上文 **§3.2**（与本节沿用 **官方随机划分** 属不同协议，勿混读数值）。

---

## 4′. 我方要在 Yelp 做到什么才算强、理论上为何可能更好

判别标准与 §4.1 **相同**。Yelp **SR 绝对值低于 LastFM** 是常态；比较应 **数据集内**：相对 MCMIPL 复制的提升 + AvgT 是否占优。

理论上，若 Yelp 用户需求更碎片化，**题库式离散 MCQ**更频繁出现“语义不对题”；本方通过不确定方向驱动的 **按需生成语义选项**有望在 **等价轮上限**减轻错配。**关键实验**仍为：对齐协议的多 seed SR、AvgT；以及关掉 LLM（或改用固定模版假设）消融，观察 SR 坍塌程度以证明增益来自方法论。

---

## 维护备注

数据更新：`cd main_table_experiments && python3 comparison/collect_results.py`。**附**：seed 2 若以后要「严格满 50 步」再对齐，可重跑 `run_mcmipl.sh YELP_STAR 2 50 …`。
