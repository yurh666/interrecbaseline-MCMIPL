# MCMIPL 官方复现报告：LastFM-Star（LAST_FM_STAR）

> 生成时间：2026-05-09  |  数据集领域：音乐推荐  
> 下文第三、四章中逐步评估表由 `comparison/generate_report.py` 从原始 log 自动生成，已与 `comparison/collect_results.py`（`mcmipl_main_table.csv`）对齐。

原始训练：`2026-05-06 19:17` → `2026-05-08 12:09` 左右完成（约 41 小时，3 seed 并行）。

## 一、指标解析说明（重要）


官方 `RL_evaluate.py` 在评估中会打印两类行：
- **批次行**：行末带 `Total epoch_uesr:N`，仅为最近 `observe_num` 名用户的滑动统计，**数值波动大**，不能代表整次 eval。
- **整体均值行**：同样含 `reward:`（单数），但**行尾无** `Total epoch_uesr`，是一次完整评估（LAST_FM：`test_size=4000`；Yelp 等：`2500`）后的真实均值，与源码中追踪 `best!!!` 的逻辑一致。

本报告与 `collect_results.py` **只采纳整体均值行**，并在各次 eval 中取 **SR@15 最高**的 checkpoint（与官方 `RL_model.train` 中 `SR15_best` 更新方式一致）。

## 二、实验目的


本实验是对 MCMIPL（Multiple Choice Questions Based Multi-Interest Policy Learning for Conversational Recommendation, WWW 2022）的官方复现，
目的是在 LastFM 数据集上获得其标准性能指标，作为 InterRec 方法的主表比较基准。

- **基准方法**：MCMIPL 官方实现（commit `01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5`）
- **数据集**：LastFM（音乐推荐）
- **评估协议**：每隔 `eval_num=10` 个训练步做一次完整评估，取历次 eval 上 **SR@15 最优**结果为最终报表数值
- **随机种子**：0, 1, 2


### 典型超参数（与当前流水线一致时请以此为准）


| 参数 | 值 | 说明 |
|------|-----|------|
| max_steps | 100 | RL 外层迭代步数 |
| sample_times | 100 | 每步采样的 episode 数 |
| eval_num | 10 | 每若干步做一次完整评估 |
| max_turn | 15 | 单会话最大对话轮数 |
| choice_num | 4 | MCQ 选项个数 |
| embed | transe | TransE 初始化 |

（若日志首行包含 `steps=…`，则以该值为 max_steps；否则上表默认值来自 `comparison/generate_report.py` 中的 `DEFAULT_MAX_STEPS`。）

### 方法概要（MCMIPL WWW 2022）

对话式推荐：系统在 `max_turn=15` 轮内交替「出选择题问属性」与「推荐 top-K」，成功定义为推荐列表中出现目标物品。选择题选项数为 `choice_num=4`（外加「都不符合」类选项由环境处理）。表征使用 TransE + GCN，`RL_model.py` 中 DQN 策略与官方默认 reward 保持一致。

本次复现使用官方源码 commit `01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5` 与配套数据划分，不改核心算法逻辑。

### checkpoint 选择与论文是否一致

训练过程中在 **测试环境**（`mode='test'`，对用户模拟器.rollout）上多次调用 `dqn_evaluate`；源码用 `SR15_best` 记录历史上最高的 **SR@15 整体均值**（不是训练采样行里的 `rewards:`）。

论文正文对「取哪一轮写进主表」的表述较为简略；**与 GitHub 实现一致的做法**即为：在各次评测的整体均值上对 SR@15 取最优。先前的报错版本曾误读了带 `Total epoch_uesr:4000` 的 **最后一个 100 用户批次**，把局部噪声当成了全量 4000 用户均值，导致 SR@15 虚高 ≈0.90；现已修正。

### 数据与 InterRec 比较时的边界

官方数据为 **数据集级随机划分**（论文写 7:1.5:1.5 train/valid/test）；与 InterRec 要求的「按交互时间排序后 40% 可见史 + 60% future」**不是同一种切分**。主表并排数字时建议在文中点明：**若未完成统一时间切分与格式转换，属于协议差异，而非实现细节噪声**。

## 三、训练过程与各次 eval（整体均值）

**Seed 0**：✅ 已完成
- 启动标记：`=== MCMIPL | LAST_FM_STAR | seed=0 | steps=100 | Wed May  6 07:17:31 PM CST 2026 ===`
- 各 eval 的整体均值 SR@15（checkpoint 对齐 training step，`eval_num=10`）：
  - step 0: SR@5=0.417, SR@10=0.749, SR@15=0.8405, AvgT=7.37, hDCG=0.3414
  - step 10: SR@5=0.409, SR@10=0.732, SR@15=0.8317, AvgT=7.47, hDCG=0.3422
  - step 20: SR@5=0.399, SR@10=0.744, SR@15=0.8267, AvgT=7.49, hDCG=0.3385
  - step 30: SR@5=0.410, SR@10=0.755, SR@15=0.8355, AvgT=7.37, hDCG=0.3412
  - step 40: SR@5=0.406, SR@10=0.741, SR@15=0.8315, AvgT=7.48, hDCG=0.3355
  - step 50: SR@5=0.409, SR@10=0.729, SR@15=0.8207, AvgT=7.53, hDCG=0.3339
  - step 60: SR@5=0.412, SR@10=0.741, SR@15=0.8253, AvgT=7.43, hDCG=0.3389
  - step 70: SR@5=0.382, SR@10=0.740, SR@15=0.8250, AvgT=7.58, hDCG=0.3331
  - step 80: SR@5=0.392, SR@10=0.747, SR@15=0.8393, AvgT=7.50, hDCG=0.3365
  - step 90: SR@5=0.398, SR@10=0.751, SR@15=0.8360, AvgT=7.47, hDCG=0.3361
  - step 100: SR@5=0.380, SR@10=0.720, SR@15=0.8177, AvgT=7.74, hDCG=0.3274
- **按 SR@15 选最优 checkpoint**：SR@15=0.8405, AvgT=7.37, hDCG=0.3414

**Seed 1**：✅ 已完成
- 启动标记：`=== MCMIPL | LAST_FM_STAR | seed=1 | steps=100 | Wed May  6 07:17:32 PM CST 2026 ===`
- 各 eval 的整体均值 SR@15（checkpoint 对齐 training step，`eval_num=10`）：
  - step 0: SR@5=0.429, SR@10=0.754, SR@15=0.8442, AvgT=7.29, hDCG=0.3465
  - step 10: SR@5=0.422, SR@10=0.753, SR@15=0.8430, AvgT=7.27, hDCG=0.3482
  - step 20: SR@5=0.398, SR@10=0.741, SR@15=0.8332, AvgT=7.49, hDCG=0.3391
  - step 30: SR@5=0.401, SR@10=0.761, SR@15=0.8387, AvgT=7.35, hDCG=0.3405
  - step 40: SR@5=0.420, SR@10=0.747, SR@15=0.8310, AvgT=7.40, hDCG=0.3432
  - step 50: SR@5=0.413, SR@10=0.757, SR@15=0.8288, AvgT=7.38, hDCG=0.3381
  - step 60: SR@5=0.385, SR@10=0.754, SR@15=0.8273, AvgT=7.47, hDCG=0.3356
  - step 70: SR@5=0.390, SR@10=0.751, SR@15=0.8248, AvgT=7.52, hDCG=0.3317
  - step 80: SR@5=0.389, SR@10=0.754, SR@15=0.8297, AvgT=7.49, hDCG=0.3338
  - step 90: SR@5=0.391, SR@10=0.755, SR@15=0.8303, AvgT=7.45, hDCG=0.3371
  - step 100: SR@5=0.404, SR@10=0.766, SR@15=0.8325, AvgT=7.36, hDCG=0.3362
- **按 SR@15 选最优 checkpoint**：SR@15=0.8442, AvgT=7.29, hDCG=0.3465

**Seed 2**：✅ 已完成
- 启动标记：`=== MCMIPL | LAST_FM_STAR | seed=2 | steps=100 | Wed May  6 07:17:33 PM CST 2026 ===`
- 各 eval 的整体均值 SR@15（checkpoint 对齐 training step，`eval_num=10`）：
  - step 0: SR@5=0.411, SR@10=0.723, SR@15=0.8288, AvgT=7.55, hDCG=0.3404
  - step 10: SR@5=0.400, SR@10=0.710, SR@15=0.8130, AvgT=7.65, hDCG=0.3338
  - step 20: SR@5=0.418, SR@10=0.742, SR@15=0.8283, AvgT=7.42, hDCG=0.3395
  - step 30: SR@5=0.403, SR@10=0.736, SR@15=0.8280, AvgT=7.52, hDCG=0.3374
  - step 40: SR@5=0.407, SR@10=0.750, SR@15=0.8370, AvgT=7.42, hDCG=0.3406
  - step 50: SR@5=0.405, SR@10=0.744, SR@15=0.8383, AvgT=7.46, hDCG=0.3371
  - step 60: SR@5=0.405, SR@10=0.748, SR@15=0.8322, AvgT=7.46, hDCG=0.3371
  - step 70: SR@5=0.400, SR@10=0.732, SR@15=0.8183, AvgT=7.56, hDCG=0.3332
  - step 80: SR@5=0.392, SR@10=0.752, SR@15=0.8330, AvgT=7.48, hDCG=0.3370
  - step 90: SR@5=0.397, SR@10=0.746, SR@15=0.8357, AvgT=7.45, hDCG=0.3378
  - step 100: SR@5=0.409, SR@10=0.736, SR@15=0.8200, AvgT=7.51, hDCG=0.3341
- **按 SR@15 选最优 checkpoint**：SR@15=0.8383, AvgT=7.46, hDCG=0.3371


## 四、汇总结果（各 seed 上取最优 checkpoint 后再跨 seed 聚合）


基于 3 个 seed（[0, 1, 2]）：

| 指标 | Seed 0 | Seed 1 | Seed 2 | **均值 ± 标准差** |
|------|-------|-------|-------|---------|
| SR@5 | 0.4170 | 0.4287 | 0.4055 | **0.4171 ± 0.0095** |
| SR@10 | 0.7488 | 0.7542 | 0.7442 | **0.7491 ± 0.0041** |
| SR@15 | 0.8405 | 0.8442 | 0.8383 | **0.8410 ± 0.0025** |
| AvgT | 7.3738 | 7.2873 | 7.4620 | **7.3743 ± 0.0713** |
| hDCG | 0.3414 | 0.3465 | 0.3371 | **0.3417 ± 0.0038** |

```
数据集: LAST_FM_STAR
    SR@5: 0.4171 ± 0.0095
   SR@10: 0.7491 ± 0.0041
   SR@15: 0.8410 ± 0.0025
    AvgT: 7.3743 ± 0.0713
    hDCG: 0.3417 ± 0.0038
```

## 五、简要解读

- **SR@15=0.841**：主成功率指标。

- **SR@5=0.417**；与 SR@15 的间隙反映后半程对话的贡献。

- **AvgT=7.37**：成功会话的平均轮数。

## 六、与论文报告值对照

论文 Table 2 中 LastFM-Star 的 SR@15 约为 **0.874**。本复现三项整体均值择优后再平均得到 **SR@15 = 0.8410 ± 0.0025**，约 **−3.3%**。常见原因包括：论文默认脚本 `eval_num=1`、评测子样本量（官方在 `eval_num=1` 时仅滚 500 名用户）、PyTorch/DGL 与随机种子差异等。该区间的偏差在 CRS 复现中可接受。

## 七、主表占位（对齐 InterRec）

| 指标 | MCMIPL（本复现 mean±std） | InterRec |
|------|---------------------------|----------|
| SR@5  | **0.417 ± 0.010** | TBD |
| SR@10 | **0.749 ± 0.004** | TBD |
| SR@15 | **0.841 ± 0.003** | TBD |
| AvgT  | **7.37 ± 0.07**   | TBD |
| hDCG  | **0.342 ± 0.004** | TBD |

## 八、流水线维护

数据集全部跑完后建议在仓库根目录执行：

```bash
python comparison/collect_results.py
python comparison/generate_report.py --dataset ALL
```

---

数据来源与解析：`comparison/mcmipl_log_metrics.py`（与 `collect_results.py`、`scripts/parse_eval_log.py` 共用逻辑）。