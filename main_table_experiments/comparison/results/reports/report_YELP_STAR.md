# MCMIPL 复现报告：Yelp (YELP_STAR)

> 生成时间：2026-05-09 11:37  |  数据集领域：餐厅/商家推荐

## 一、指标解析说明（重要）


官方 `RL_evaluate.py` 在评估中会打印两类行：
- **批次行**：行末带 `Total epoch_uesr:N`，仅为最近 `observe_num` 名用户的滑动统计，**数值波动大**，不能代表整次 eval。
- **整体均值行**：同样含 `reward:`（单数），但**行尾无** `Total epoch_uesr`，是一次完整评估（LAST_FM：`test_size=4000`；Yelp 等：`2500`）后的真实均值，与源码中追踪 `best!!!` 的逻辑一致。

本报告与 `collect_results.py` **只采纳整体均值行**，并在各次 eval 中取 **SR@15 最高**的 checkpoint（与官方 `RL_model.train` 中 `SR15_best` 更新方式一致）。

## 二、实验目的


本实验是对 MCMIPL（Multiple Choice Questions Based Multi-Interest Policy Learning for Conversational Recommendation, WWW 2022）的官方复现，
目的是在 Yelp 数据集上获得其标准性能指标，作为 InterRec 方法的主表比较基准。

- **基准方法**：MCMIPL 官方实现（commit `01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5`）
- **数据集**：Yelp（餐厅/商家推荐）
- **评估协议**：每隔 `eval_num=10` 个训练步做一次完整评估，取历次 eval 上 **SR@15 最优**结果为最终报表数值
- **随机种子**：0


### 典型超参数（与当前流水线一致时请以此为准）


| 参数 | 值 | 说明 |
|------|-----|------|
| max_steps | 50 | RL 外层迭代步数 |
| sample_times | 100 | 每步采样的 episode 数 |
| eval_num | 10 | 每若干步做一次完整评估 |
| max_turn | 15 | 单会话最大对话轮数 |
| choice_num | 4 | MCQ 选项个数 |
| embed | transe | TransE 初始化 |

（若单日誌首行包含 `steps=…`，则以该值为 max_steps；否则上表默认值来自 `comparison/generate_report.py` 中的 `DEFAULT_MAX_STEPS`。）

## 三、训练过程与各次 eval（整体均值）

**Seed 0**：⏳ 进行中（训练步进度 20/50）
- 启动标记：`未知`
- 各 eval 的整体均值 SR@15（checkpoint 对齐 training step，`eval_num=10`）：
  - step 0: SR@5=0.142, SR@10=0.294, SR@15=0.4180, AvgT=11.96, hDCG=0.1490
  - step 10: SR@5=0.176, SR@10=0.309, SR@15=0.4188, AvgT=11.61, hDCG=0.1696
- **按 SR@15 选最优 checkpoint**：SR@15=0.4188, AvgT=11.61, hDCG=0.1696

**Seed 1**：⏳ 进行中（训练步进度 0/50）
- 启动标记：`未知`
- 暂无 eval 整体均值输出

**Seed 2**：⏳ 进行中（训练步进度 0/50）
- 启动标记：`未知`
- 暂无 eval 整体均值输出


## 四、汇总结果（各 seed 上取最优 checkpoint 后再跨 seed 聚合）


基于 1 个 seed（[0]）：

| 指标 | Seed 0 | **均值 ± 标准差** |
|------|-------|---------|
| SR@5 | 0.1756 | **0.1756 ± 0.0000** |
| SR@10 | 0.3092 | **0.3092 ± 0.0000** |
| SR@15 | 0.4188 | **0.4188 ± 0.0000** |
| AvgT | 11.6148 | **11.6148 ± 0.0000** |
| hDCG | 0.1696 | **0.1696 ± 0.0000** |

```
数据集: YELP_STAR
    SR@5: 0.1756 ± 0.0000
   SR@10: 0.3092 ± 0.0000
   SR@15: 0.4188 ± 0.0000
    AvgT: 11.6148 ± 0.0000
    hDCG: 0.1696 ± 0.0000
```

## 五、简要解读

- **SR@15=0.419**：主成功率指标。

- **SR@5=0.176**；与 SR@15 的间隙反映后半程对话的贡献。

- **AvgT=11.61**：成功会话的平均轮数。


---
数据来源：`comparison/mcmipl_log_metrics.py`（与 `collect_results.py` 共用）。