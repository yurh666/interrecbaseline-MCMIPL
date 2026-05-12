# MCMIPL 官方代码 Review

官方仓库：https://github.com/ZYM6-6/MCMIPL  
本地路径：`main_table_experiments/baselines/mcmipl_official/MCMIPL`  
当前 commit：`01b7dd672331fc58b67a9ec3ba3dfa4a02f31bd5`  
论文：Multiple Choice Questions based Multi-Interest Policy Learning for Conversational Recommendation, WWW 2022

## 1. 官方环境

README 声明的环境为：

- Python 3.7.9
- PyTorch 1.7.1
- DGL 0.6.0

代码还使用：

- numpy / scipy / pandas；
- tqdm；
- easydict；
- ipdb；
- tkinter `_flatten`；
- OpenKE / TransE 预训练 embedding 产物。

## 2. 官方支持的数据集

README 提到 released data 包括：

- `lastfm_start`，代码中常量为 `LAST_FM_STAR`，目录为 `data/lastfm_star`；
- `yelp_star`，代码常量为 `YELP_STAR`；
- `Amazon-Book`，代码常量为 `BOOK`，目录为 `data/book`；
- `MovieLens`，代码常量为 `MOVIE`，目录为 `data/movie`。

`utils.py` 中的目录映射为：

```python
DATA_DIR = {
    LAST_FM_STAR: './data/lastfm_star',
    YELP_STAR: './data/yelp_star',
    BOOK: './data/book',
    MOVIE: './data/movie',
}
TMP_DIR = {
    LAST_FM_STAR: './tmp/last_fm_star',
    YELP_STAR: './tmp/yelp_star',
    BOOK: './tmp/book',
    MOVIE: './tmp/movie',
}
```

注意：`graph_init.py` 的 argparse choices 写成 `[LAST_FM_STAR, YELP_STAR, BOOK]`，虽然代码里有 `MOVIE` 分支，但默认 choices 没包含 `MOVIE`，这可能需要官方兼容性 patch 才能运行 MovieLens graph init。

## 3. 数据目录格式

官方 README 要求把数据放到：

```text
MCMIPL/data/<data_name>
```

从代码看，环境会读取以下数据：

```text
data/<data_name>/UI_Interaction_data/review_dict_train.json
data/<data_name>/UI_Interaction_data/review_dict_valid.json
data/<data_name>/UI_Interaction_data/review_dict_test.json
data/<data_name>/UI_data/train.pkl
data/<data_name>/UI_data/test.pkl
data/<data_name>/Graph_generate_data/...
```

`YELP_STAR` 的 `Graph_generate_data` 需要：

- `user_item.json`
- `user_dict.json`
- `item_dict-original_tag.json`
- `item_dict-merged_tag.json`
- `first-layer_merged_tag_map.json`
- `second-layer_oringinal_tag_map.json`
- `2-layer taxonomy.json`

`LAST_FM_STAR` 的 graph 类需要：

- `user_friends.pkl`
- `user_like.pkl`
- `item_fea.pkl`
- `fea_large.pkl`

## 4. graph_init.py 做了什么

`graph_init.py` 负责从官方数据构造 `dataset.pkl` 和 `kg.pkl`：

1. 选择数据集对应的 Dataset 类；
2. 加载官方 processed data；
3. 保存 dataset 到 `tmp/<dataset>/dataset.pkl`；
4. 构造 graph object；
5. 保存 kg 到 `tmp/<dataset>/kg.pkl`。

对 `BOOK` 和 `MOVIE`，代码路径较特殊：先直接构造 `BookGraph()` / `MovieGraph()` 并写入 `tmp/book/kg.pkl` / `tmp/movie/kg.pkl`，再构造 Dataset。

## 5. TransE / OpenKE embedding 准备

README 要求：

```text
Use TransE [OpenKE] to pretrain graph embeddings.
Put pretrained embeddings under /tmp/<data_name>/embeds/.
```

但代码实际从相对路径读取：

```python
TMP_DIR[dataset] + '/embeds/' + f'{embed}.pkl'
```

默认 `embed='transe'`，因此需要：

```text
MCMIPL/tmp/<dataset>/embeds/transe.pkl
```

`transe.pkl` 需要包含：

```python
{
  'ui_emb': numpy.ndarray,
  'feature_emb': numpy.ndarray
}
```

如果找不到 embedding，环境代码会 fallback 到随机 `nn.Embedding`，但主表复现不能依赖随机 fallback，必须使用官方或按官方 OpenKE 流程生成的 TransE embeddings。

## 6. RL_model.py 训练流程

主入口为：

```bash
python RL_model.py --data_name <data_name>
```

默认重要参数：

- `seed=1`
- `max_turn=15`
- `sample_times=100`
- `max_steps=100`
- `eval_num=1`
- `save_num=10`
- `choice_num=4`
- `cand_num=10`
- `cand_item_num=10`
- `embed=transe`
- `seq=transformer`
- `gcn=True`（通过 `--gcn action='store_false'` 实现，不传即 True）

训练流程：

1. 加载 `kg.pkl` 和 `dataset.pkl`；
2. 初始化 `MultiChoiceRecommendEnv(..., mode='train')`；
3. 用 `construct_graph.get_graph()` 构造 DGL heterograph；
4. 拼接 `ui_embeds`、`feature_emb` 和 padding embedding；
5. 初始化 `GraphEncoder` 和 DQN agent；
6. 采样训练 episode；
7. 每轮由 agent 在 ask feature 和 recommend item action 中选择；
8. 环境执行 `env.step()`，返回 reward / done；
9. 存入 prioritized replay memory；
10. 调用 `agent.optimize_model()` 更新 DQN；
11. 每 `eval_num` 步调用 `dqn_evaluate()`；
12. 每 `save_num` 步保存 policy model 到 `tmp/<dataset>/RL-agent/`。

## 7. evaluate.py 评估流程

官方 README 用法：

```bash
python evaluate.py --data_name <data_name> --load_rl_epoch <checkpoint_epoch>
```

评估流程：

1. 加载 `kg.pkl` 和 `dataset.pkl`；
2. 初始化 test env；
3. 构造 graph encoder 和 agent；
4. 从 `tmp/<dataset>/RL-agent/<filename>-epoch-<epoch>.pkl` 加载 checkpoint；
5. 调用 `RL/RL_evaluate.py::dqn_evaluate()`；
6. 输出并保存 `SR5 / SR10 / SR15 / AvgT / Rank / reward`。

风险：当前官方 `evaluate.py` 在 `evaluate()` 函数中使用 `SR15_best`、`SR5_best` 等变量，但没有在函数内或全局初始化。若直接运行触发 `NameError`，需要做兼容性 patch：初始化这些变量或删除 best 判断打印。该 patch 只影响日志打印，不应改变 `dqn_evaluate()` 计算逻辑。

## 8. User simulator 逻辑

环境类为 `RL/env_multi_choice_question.py::MultiChoiceRecommendEnv`。

每个 episode 的用户目标：

- train 模式：随机选择用户，再从 `u_multi[str(user_id)]` 中选择 target item set；
- test 模式：遍历 `test.pkl` 中展开的 `(user, target_item_set)`。

`reset()` 中：

1. 从 target item set 取所有 target items 的共同属性作为初始可接受属性候选；
2. 随机选择一个 feature 作为初始用户偏好 feature；
3. 将该 feature 放入 `user_acc_feature`；
4. 基于该 feature 更新 candidate items；
5. 计算 reachable features 和 state graph。

用户接受 feature 的逻辑在 `_ask_update()`：

```python
if asked_feature in self.feature_groundtrue:
    accept
else:
    reject
```

其中 `feature_groundtrue` 是 target item set 的 feature 并集。

## 9. Multiple choice question 逻辑

在 `step()` 中，如果 action 是 feature：

1. agent 给出 `sorted_actions`；
2. 环境将小 feature 映射到大 feature；
3. 按排序位置累积大 feature score；
4. 选择得分最高的大 feature；
5. 取该大 feature 下前 `choice_num` 个 small features 作为 `asked_feature`；
6. 调用 `_ask_update(asked_feature)` 模拟用户对多个 feature 的接受/拒绝。

因此 MCMIPL 的 multiple choice question 是 attribute-instance based：一次 ask 中包含多个 small feature choices，并按 target item features 决定接受/拒绝。

## 10. Candidate item set update

`_update_cand_items(asked_feature, acc_rej)` 中：

- 对 accepted feature：候选集合与包含该 feature 的 item 集合取交集；
- 对 rejected feature：候选集合中移除包含该 feature 的 items；
- 如果存在 accepted items，则以 accepted item set 为主，再去掉 rejected item set；
- 更新后用 `_item_score()` 对 candidate items 排序。

推荐失败后，`_recommend_update()` 会从候选集合中删除已推荐但未命中的 items。

## 11. Reward

环境中的 reward 字典：

```python
ask_suc: 0.01
ask_fail: -0.1
rec_suc: 1
rec_fail: -0.1
until_T: -0.3
cand_none: -0.1
```

训练时 episode 累计 reward 用于日志统计，DQN 每步使用即时 reward。

## 12. Success 判断

推荐动作会取排序后的前 `rec_num=10` 个 item。`_recommend_update()` 判断：

```python
hit = any(target_item in recom_items for target_item in self.target_item)
```

如果命中：

- reward = `rec_suc = 1`
- done = 命中 item 在推荐列表中的位置 + 1

如果未命中：

- reward = `rec_fail = -0.1`
- done = 0

## 13. SR@5 / SR@10 / SR@15 / AvgT / hDCG 指标

指标在 `RL/RL_evaluate.py::dqn_evaluate()` 中计算。

如果 episode 在第 `t` 个 step done 且 reward=1：

- `t < 5`：SR5、SR10、SR15 都加 1；
- `5 <= t < 10`：SR10、SR15 加 1；
- `t >= 10`：SR15 加 1。

`AvgT` 累加 `t+1`。

`Rank` 计算公式为：

```python
Rank += 1/log(t+3,2) + (1/log(t+2,2)-1/log(t+3,2))/log(done+1,2)
```

`utils.save_rl_mtric()` 将 `Rank` 保存为 `hDCG`。因此主表里应将代码输出 `Rank` 映射为 `hDCG`。

评估时：

- `LAST_FM_STAR`：`eval_num==1` 时 test size 为 500，否则 4000；
- 其他数据集：`eval_num==1` 时 test size 为 500，否则 2500。

## 14. 随机种子

代码提供 `set_random_seed(seed)`：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if cuda
```

`MultiChoiceRecommendEnv.__init__()` 和训练/评估入口都会调用 seed 设置。主表应至少跑 `seed=0,1,2`。

## 15. 可能遇到的依赖和复现问题

1. 官方依赖较旧：Python 3.7.9、PyTorch 1.7.1、DGL 0.6.0 在新系统上可能安装困难。
2. `evaluate.py` 存在未定义 `SR15_best` 等变量的风险。
3. `graph_init.py` 中 `MOVIE` 有代码分支但不在 argparse choices 中。
4. 官方 README 没提供 OpenKE / TransE 的完整训练命令，只说明 embedding 放置路径。
5. 当前 cloned repo 包含部分 data json，但不一定包含全部 `UI_data/*.pkl`、`Graph_generate_data/*.pkl` 和 `tmp/<dataset>/embeds/transe.pkl`。
6. 若缺少官方 released data 或 embeddings，不能用自造数据替代主表 baseline，只能记录阻塞并等待官方数据。
