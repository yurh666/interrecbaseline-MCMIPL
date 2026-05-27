# 新服务器 Prompt：MCMIPL RL 补跑（BOOK / LAST_FM / MOVIE seed 0,1）

> 复制整段给 AI / 协作者。仓库：<https://github.com/yurh666/interrecbaseline-MCMIPL>

---

## 任务说明

在**新服务器**上克隆本仓库，**只跑 Phase B RL**（不重做 Phase A / graph_init），串行补齐：

1. **BOOK** seed 0 → seed 1  
2. **LAST_FM_STAR** seed 0 → seed 1  
3. **MOVIE** seed 0 → seed 1  

**YELP_STAR 不在本任务**（在另一台 CPU 机上跑 seed 0→1→2）。

每个 seed 结束后必须 **立即归档** 到 `experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/<slug>/seed_<N>/`，避免 checkpoint 覆盖。

---

## 步骤 1：克隆与版本

```bash
git clone https://github.com/yurh666/interrecbaseline-MCMIPL.git
cd interrecbaseline-MCMIPL
git checkout main && git pull origin main
git log -1 --oneline
test -f main_table_experiments/rl_supplement_three_datasets/run_supplement_s01_serial.sh && echo OK
```

---

## 步骤 2：检查代码与中间结果（只读）

确认存在：

| 检查项 | 路径 |
|--------|------|
| Phase-A 归档 | `experiments/mcmipl_interrec_protocol_eval/archives/phase_a/{book,last_fm_star,movie}/` 含 `dataset.pkl`, `kg.pkl`, `embeds/transe.pkl` |
| seed_2 参考 | `archives/checkpoints/*/seed_2/MANIFEST.txt` |
| 归档脚本 | `main_table_experiments/scripts/mcmipl_{archive_seed,run_seed_with_archive,seed_archive_status,lane_three_datasets_serial}.sh` |
| seed 分目录 | `main_table_experiments/baselines/mcmipl_official/MCMIPL/utils.py` 支持 `MCMIPL_RL_SEED` → `tmp/<slug>/RL-agent/seed_<N>/` |

```bash
for slug in book last_fm_star movie; do
  ls -la experiments/mcmipl_interrec_protocol_eval/archives/phase_a/$slug/
done
bash main_table_experiments/scripts/mcmipl_seed_archive_status.sh BOOK 0 || echo "BOOK s0 needs run"
```

---

## 步骤 3：仅 RL 环境（CPU 即可）

```bash
# 示例：venv（按本机调整路径）
python3 -m venv ~/venvs/mcmipl-cpu
source ~/venvs/mcmipl-cpu/bin/activate
pip install -r main_table_experiments/baselines/mcmipl_official/requirements.txt  # 或 environment.yml 导出

export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON="$(which python)"
export CUDA_VISIBLE_DEVICES=""
export MCMIPL_SAVE_NUM=10

python -c "import torch,dgl; print('torch',torch.__version__,'dgl ok')"
```

**不需要** GPU、**不需要** OpenKE 重训（`transe.pkl` 从归档恢复）。

---

## 步骤 4：恢复 Phase-A 到 tmp

```bash
cd interrecbaseline-MCMIPL
bash main_table_experiments/rl_supplement_three_datasets/restore_phase_a_to_tmp.sh
ls -la main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/book/embeds/transe.pkl
```

---

## 步骤 5：串行补跑（推荐 tmux）

```bash
cd interrecbaseline-MCMIPL
# 前台调试：
# RUN_FOREGROUND=1 bash main_table_experiments/rl_supplement_three_datasets/run_supplement_s01_serial.sh

# 默认 tmux：
bash main_table_experiments/rl_supplement_three_datasets/run_supplement_s01_serial.sh
tmux attach -t mcmipl_rl_supplement_3ds
```

监控：

```bash
tail -f main_table_experiments/logs/lane_three_datasets_*.log
cat experiments/mcmipl_interrec_protocol_eval/archives/run_registry.jsonl
```

---

## 步骤 6：验收

```bash
for ds in BOOK LAST_FM_STAR MOVIE; do
  for s in 0 1; do
    bash main_table_experiments/scripts/mcmipl_seed_archive_status.sh "$ds" "$s" && echo OK_$ds$s || echo FAIL_$ds$s
  done
done
ls experiments/mcmipl_interrec_protocol_eval/archives/checkpoints/book/seed_0/RL-agent/
```

通过后 `git add` 仅 **archives 新增** + **MANIFEST/TIMING**（勿提交 `tmp/**/*.pkl`），push 回 GitHub。

---

## 禁止事项

- 不要与 YELP 训练并行抢同一台机（若同机则只跑本 prompt 三数据集或只跑 YELP，二选一）
- 不要删除 `archives/checkpoints/*/seed_2/` 参考副本
- 不要跳过 per-seed 归档

---

## 参考文档

- `main_table_experiments/rl_supplement_three_datasets/MANIFEST.md`
- `experiments/mcmipl_interrec_protocol_eval/archives/RUN_STORAGE_LAYOUT.md`
- `docs/HANDOFF_RUNBOOK.md`（Phase A/B 总览）
