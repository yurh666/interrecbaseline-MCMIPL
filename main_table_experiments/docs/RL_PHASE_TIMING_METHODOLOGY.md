# RL 阶段计时方法论（GPU vs CPU 与瓶颈）

本文档说明仓库内 **RL（`RL_model.py`）** 的监测方式及**可作出的严谨结论**，不包含 **TransE / OpenKE**（独立流程）。

## 1. 我们在测什么

- **对象**：`train()` 内墙钟时间，拆成两类：
  - **`eval_s`**：所有 **`dqn_evaluate`** 调用时间之和。内部是 `tqdm(range(test_size))`，对每个测试用户跑完整多轮对话直至 `done`，含环境逻辑与 `select_action(..., is_test=True)`。
  - **`train_sampling_s`**：每个训练 step 里 **`for i_episode in range(sample_times)`** 整段耗时之和（含 `env.reset` / `env.step`、`select_action`、`optimize_model`、目标网络更新等）。

启用方式：运行前设置 **`MCMIPL_RL_PHASE_TIMINGS=1`**。结束时打印一行：

`[MCMIPL_RL_PHASE_TIMINGS] device=... eval_s=... train_sampling_s=... eval_pct=... train_sampling_pct=... eval_calls=...`

**未测到的部分**：图/数据 `load_kg`、`load_dataset`、首构 `GraphEncoder`、TransE 加载、进程启动等（占比较短时可忽略；若要可在外层再包一层 `time`）。

## 2. 快速对比脚本（GPU vs CPU 墙钟）

```bash
cd main_table_experiments
bash scripts/profile_rl_gpu_cpu_compare.sh
```

默认用 **`MCMIPL_RL_PROFILE_TEST_USERS=200`** 缩小评测集（**仅用于剖析**，主表实验请**不要**设置该环境变量）。

脚本会对同一短配置各跑一遍 **GPU**（`unset MCMIPL_FORCE_CPU`）与 **CPU**（`MCMIPL_FORCE_CPU=1`），并把日志写入 `logs/rl_gpu_cpu_profile_<时间戳>.log`。

## 3. 能严格论证什么、不能论证什么

### 3.1 可论证（在「同一配置、同一机器」前提下）

- **评测段的规模**：若 **`eval_pct` 长期显著高于 `train_sampling_pct`**，则在该配置上 **总墙钟对「评测规模（test_size × eval 次数）」敏感**；此时即便 GPU 加速神经部分，**对整段 RL 的加速比也有上界**（评测里环境与用户循环仍在 CPU/Python）。
- **训练采样段**：若 **`train_sampling_pct` 高** 且 **`wall_clock_cpu / wall_clock_gpu` 明显大于 1**，则 **GPU 对训练步有益**，值得保留 GPU 跑 RL。
- **二者可比性**：同一 `PROF_MAX_STEPS`、`PROF_SAMPLE_TIMES`、同一 `MCMIPL_RL_PROFILE_TEST_USERS` 下比较 GPU/CPU **墙钟** 与 **`[MCMIPL_RL_PHASE_TIMINGS]`**，是**内部一致**的；可重复跑取中位数减少噪声。

### 3.2 不能仅凭短跑「一锤定音」的

- **缩小 `MCMIPL_RL_PROFILE_TEST_USERS`** 会改变 **eval 与 sampling 的相对占比**；全量 **2500 用户/次** 时 **`eval_s` 通常更大**。短跑结论应表述为：**在剖析配置下的占比与加速比**；全量需长程一次或保守外推。
- **「GPU 完全没用」**：即使墙钟接近，GPU 仍可能降低 **CPU 占用**、避免 **同一 Python 进程里** 过多大张量在 CPU 上算；结论应写为 **「加速比接近 1」或「收益小于阈值」**，而不是物理上零差异。
- **TransE**：本监测**不涉及**；TransE 是否上 GPU需单独测 OpenKE。

## 4. 若结论为「除 TransE 外 GPU 收益有限」时的思路调整（建议）

若多次全量或外推后仍表明 **评测主导** 且 **GPU/CPU 墙钟比接近 1**：

- 可考虑：**TransE 仍用 GPU**；**RL 用 CPU 机或低价核多机器** 跑，以节省 GPU 机时。
- 或：**不减正确性前提下** 减少评测频率 / 评测用户数（属于 **改 protocol**，需与论文/基线定义一致，不能悄悄改）。

以上均需 **你们用本脚本 + 全量设置** 自证后再定策略。
