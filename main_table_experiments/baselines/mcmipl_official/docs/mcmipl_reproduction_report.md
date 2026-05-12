# MCMIPL 主表 Baseline 复现报告

## 1. 复现目标

本实验用于论文主表 baseline。目标是完整复现官方 MCMIPL：

```text
official repo
  -> official data
  -> graph_init.py
  -> TransE / OpenKE graph embeddings
  -> RL_model.py training
  -> evaluate.py evaluation
  -> official metrics
```

本报告明确区分主表 baseline 与选择题机制预实验：预实验结果不能替代本主表 baseline。

## 2. 官方 Repo

- URL: `https://github.com/ZYM6-6/MCMIPL`
- 本地路径: `main_table_experiments/baselines/mcmipl_official/MCMIPL`
- Commit hash: `01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5`

## 3. 环境版本

官方 README 要求：

- Python 3.7.9
- PyTorch 1.7.1
- DGL 0.6.0

已提供：

- `baselines/mcmipl_official/environment.yml`
- `baselines/mcmipl_official/Dockerfile`
- `baselines/mcmipl_official/scripts/setup_env.sh`

实际安装版本将记录到：

```text
baselines/mcmipl_official/results/raw_logs/env_versions.txt
```

## 4. 数据集

官方 README 提到：

- `lastfm_start` / code: `LAST_FM_STAR`
- `yelp_star` / code: `YELP_STAR`
- `Amazon-Book` / code: `BOOK`
- `MovieLens` / code: `MOVIE`

优先复现顺序：

1. `LAST_FM_STAR`
2. `YELP_STAR`
3. `MOVIE`（若官方数据和 graph_init 兼容性问题解决）

主表复现要求使用官方 released data 和官方 split，不允许用预实验处理的小样本替代。

## 5. 运行命令

环境：

```bash
cd /home/yurh/main_table_experiments/baselines/mcmipl_official
bash scripts/setup_env.sh
```

数据检查：

```bash
bash scripts/prepare_data.sh LAST_FM_STAR
```

Graph init：

```bash
bash scripts/run_graph_init.sh LAST_FM_STAR
```

训练：

```bash
bash scripts/run_train.sh LAST_FM_STAR 0
bash scripts/run_train.sh LAST_FM_STAR 1
bash scripts/run_train.sh LAST_FM_STAR 2
```

评估：

```bash
bash scripts/run_eval.sh LAST_FM_STAR <checkpoint_epoch> 0
bash scripts/run_eval.sh LAST_FM_STAR <checkpoint_epoch> 1
bash scripts/run_eval.sh LAST_FM_STAR <checkpoint_epoch> 2
```

一键运行：

```bash
CHECKPOINT_EPOCH=100 bash scripts/run_all.sh LAST_FM_STAR
```

## 6. 是否使用官方默认参数

当前脚本默认保留官方参数，仅显式传入：

- `--data_name`
- `--seed`
- `--load_rl_epoch`

不修改：

- action space；
- user simulator；
- reward；
- evaluation metrics；
- train/test split。

## 7. Checkpoint epoch

官方训练默认：

- `max_steps=100`
- `save_num=10`

因此可评估 checkpoint epoch 通常为 `10, 20, ..., 100`。正式主表应按照官方论文或 validation 表现选择 checkpoint，并在本节记录。

当前默认 placeholder：

```text
checkpoint_epoch = 100
```

## 8. 每个 seed 的结果

结果 JSON 输出到：

```text
baselines/mcmipl_official/results/metrics/eval_<data_name>_<seed>.json
```

待真实运行后填入：

| Dataset | Seed | Epoch | SR@5 | SR@10 | SR@15 | AvgT | hDCG |
|---|---:|---:|---:|---:|---:|---:|---:|
| TBD | 0 | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | 2 | TBD | TBD | TBD | TBD | TBD | TBD |

## 9. Mean ± Std

使用：

```bash
cd /home/yurh/main_table_experiments/comparison
python collect_results.py
python make_main_table.py
```

输出：

```text
comparison/results/main_table.csv
comparison/results/main_table_mean_std.csv
```

## 10. 与论文报告结果是否一致

待真实运行后比较。若不一致，优先检查：

1. 官方数据版本；
2. TransE / OpenKE embedding 是否一致；
3. checkpoint epoch 是否一致；
4. PyTorch / DGL 版本；
5. GPU / CPU 数值差异；
6. 官方代码 bug 或未说明参数。

## 11. 已发现的复现风险

1. `evaluate.py` 中 `SR15_best`、`SR5_best` 等变量未初始化，可能直接触发 `NameError`。
2. `graph_init.py` 中 `MOVIE` 有分支但不在 argparse choices 中。
3. 官方 README 未给出 OpenKE / TransE 完整训练命令，只说明 embedding 放置位置。
4. 当前 repo 中可能不包含完整 `UI_data/*.pkl` 或 `tmp/<dataset>/embeds/transe.pkl`。
5. 老版本 DGL/PyTorch 在当前系统上可能需要 conda 或 Docker。

## 12. Patch 记录

当前未对官方核心算法做 patch。

如后续必须 patch：

- 只能做兼容性修复或 logging；
- 必须生成 `patches/*.patch`；
- 必须说明是否影响算法；
- 主表结果优先使用无 logging 或确认 logging 不影响的版本。

## 13. 当前状态

已完成：

- 官方 repo clone；
- code review；
- environment.yml / Dockerfile / setup_env.sh；
- data / graph_init / train / eval / run_all 脚本；
- eval log parser；
- comparison 汇总与统计检验脚本。

待完成：

- 安装官方环境；
- 获取/确认官方 released data 与 TransE embeddings；
- 运行至少 seed=0 的 smoke test；
- 正式跑 seed=0,1,2；
- 填入结果并生成主表。
