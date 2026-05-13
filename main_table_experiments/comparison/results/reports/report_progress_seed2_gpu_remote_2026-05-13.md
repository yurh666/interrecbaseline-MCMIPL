# 主表进度、GPU 与远程续跑说明（快照：2026-05-13）

本文档答复三件事：**seed2 完成情况**、**GPU 开销集中在哪里**、**如何在另一台机器上拿到完整 MCMIPL 并完成「剩下两个数据集」续跑**。  
数据源：`main_table_experiments/logs/train_*.log`、`ps`、`RL_model.py`/`agent.py` 源码阅读。

---

## 1. 第二个数据集 · seed 2（YELP_STAR）是否跑完？

**结论：未完成。**

| 条目 | 状态 |
|------|------|
| `LAST_FM_STAR` · seed `2` | **已完成**。日志含 `=== DONE: LAST_FM_STAR seed=2 ===`，并保存 `epoch-100` 策略权重。 |
| `YELP_STAR` · seed `2` | **未完成**。未见 `=== DONE: YELP_STAR seed=2 ===`（`run_mcmipl.sh` 在 `RL_model.py` **正常退出后**才会 `tee` 该行）。 |

**佐证（约 2026-05-13 10:17 CST 快照）：**

- 仍在运行的命令形如：  
  `python -u RL_model.py --data_name YELP_STAR --seed 2 --gpu 0 --max_steps 50 --sample_times 100 ...`
- `RL_model.py` 中每个 `train_step` 末尾会打印一行：  
  `loss : … in epoch_uesr 100`（此处 `epoch_uesr` 实为 `sample_times`）。  
  当前日志中该类行累计 **40** 条 → 已结束 **40** 个大步，`max_steps=50` 下还剩 **至多 10** 大步（另含当前步内未到 100 的 sampling  tqdm）。
- 日志尾部可见 `sampling: … \| 38/100` 等，说明正处在某一大步内部的 **蒙特卡洛对话采样**，尚未进入该大步的收尾与下一轮。

**对您「跑完则停掉后面两个数据集」的答复：**

- 因 **尚未跑完**，**不应**在此处终止当前 YELP seed2（否则得不到可归档日志与 DONE 行）。  
- 本机未发现独立的 **BOOK / MOVIE** 全量 RL 进程或 `cron` 排队任务（仅见上述 **YELP_STAR seed2** 单进程）；**没有需要额外杀掉的后两个数据集任务**——若之后在 shell/队列里接了 BOOK/MOVIE，请以 `pgrep -af RL_model.py` / `BOOK` / `MOVIE` 自检。

---

## 2. 若未完成：YELP seed2 大概还要多久？

方法：用「剩余大步数 × 历史单步量级」粗略估计。

- **已知**：`sample_times=100`，每大步需跑 **100** 轮对话式 episode；单步末尾才打印 `loss : …`。  
  单次 `sampling: 100%` 在近期日志常见 **约 20–35 分钟/步** 量级（与 GPU、特征维、用户仿真长度有关）。
- **剩余**：已完成 **40/50** 大步 → 约 **10** 个大步 + **当前大步**的余量（采样进度条未完成部分）。

**粗估：** 在完成当前大步剩余采样后，尚需约 **≈ 3.5–7 小时** 跑满 50 步（含每隔 `eval_num` 次的 `dqn_evaluate`：测试集上对 **2500** 条链路滚动评估，有时也会接近 **1 h/次**量级）。  

**更准确做法：** 再等 **1–2 个完整大步**，用两段 `loss :` 行之间的墙钟差 × 剩余步数，自行修正区间。

完成后请在本机核对：

```bash
grep -E '^=== DONE: YELP_STAR seed=2' /home/yurh/main_table_experiments/logs/train_YELP_STAR_s2.log
```

出现后执行 `comparison/collect_results.py`（若有）并重扫 CSV/报告。

---

## 3. GPU 主要花在什么环节？

**总览：** 本流水线里 GPU 核心是 **PyTorch**：**RL 阶段（`RL_model.py`）**；**图表征 / GCN-Q 网络与前向**。其余大量是 **CPU**（仿真环境、图谱与数据载入、预处理脚本等）。

### 3.1 必须 / 强烈建议 GPU：`RL_model.py`（LAST_FM_STAR、YELP_STAR、BOOK、MOVIE 同类）

源码要点：

- `os.environ['CUDA_VISIBLE_DEVICES']=args.gpu`，`args.device=torch.device('cuda')`（CUDA 可用时）。
- **`GraphEncoder`（multi-interest / GCN）**与 **两套 DQN（policy_net / target_net）**均在 `cuda`：`agent.optimize_model`、`select_action`、`gcn_net([state])` 等始终在 GPU 上算。
- **每一对话步**会做：GCN + DQN + replay 优化 → **GPU 利用率高**，尤其 `sample_times × max_steps` 双重大循环。
- **`dqn_evaluate`**（定期在测试子集 **2500** 条上 rollout）同样是同一套 GPU 模型前向推理，步数多时也会长时间占卡。

### 3.2 常为 CPU：`graph_init.py` / 构图与数据导出

构图、pickle/json 读写、多数 **TransE/OpenKE** 的官方脚本若配置为 CPU 版则不占 GPU。**若你改用 GPU 版 OpenKE，则 Embedding 训练阶段也会显著吃 GPU**，与当前 conda 镜像有关。

### 3.3 常为 CPU：InterRec 侧（BOOK/MOVIE 时序协议）

`preprocess_dataset.py`、`export_interrec_sessions_to_mcmipl_*`：**TF-IDF/SVD、会话构造、CSV** 等以 **numpy/sklearn/scipy** 为主，通常在 **CPU**；一般不抢 RL 进程的 GPU。

**实践建议：** 同一台机子上 **别让两条 `RL_model.py` 共用一张卡**（易 OOM/极慢）；InterRec + `graph_init` 可与 RL 串行或使用另一张卡/另一台机器。

---

## 4. GitHub / 快照同步说明

与本报告同批更新的内容（若已 rsync/copy 到 **`interrecbaseline-MCMIPL`**）：  

- `comparison/results/reports/report_progress_seed2_gpu_remote_2026-05-13.md`（本文件）  
- `docs/PROMPT_REMOTE_SERVER_MCMIPL_BOOK_MOVIE.md`（远程续跑可复制 Prompt）

在仓库根执行：

```bash
cd /path/to/interrecbaseline-MCMIPL
git add -A && git status
git commit -m "Docs: seed2 progress, GPU usage, remote BOOK/MOVIE runbook"
git push origin main
```

（若远端已有历史而你本地改过同一提交，按需 `git pull --rebase` 或解决冲突后再推。）

---

## 5. 「剩下两个数据集」指什么？

与既有「主表两数据集」（LAST_FM + Yelp）区分开：本文档后续的 **Prompt** 中「剩下两个」指 **`BOOK` + `MOVIE`**（在你此前约定中与 **InterRec 时序预处理 → `graph_init` → TransE → `RL_model.py`** 同一套链路）。  
LAST_FM_STAR / YELP_STAR 在主表语境下：**前者三 seed 已齐；后者差 seed 2 收尾**。  

远程服务器上可先 **克隆带数据与脚本** 的 `interrecbaseline-MCMIPL`，再等本机 YELP s2 DONE 后主表数值完全锁定，或直接在新机器上从零按 Prompt 对齐 BOOK/MOVIE。
