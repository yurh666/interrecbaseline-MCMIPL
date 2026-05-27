# MCMIPL Ready Eval-Only 命令模板

**生成时间：** 2026-05-27  
**状态：** 当前 **无** dataset×seed 满足 ready — 以下仅为模板，**请勿现在执行**。

---

## 重要：CPU 参数写法（非 `--gpu -1`）

本仓库 `RL_model.py` / `evaluate.py` **不支持** `--gpu -1`。

CPU 模式正确写法：

```bash
export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON=/home/yrh666/venvs/mcmipl-cpu/bin/python
export CUDA_VISIBLE_DEVICES=""
```

`run_mcmipl.sh` 在 `MCMIPL_FORCE_CPU=1` 时已自动选择 CPU python 并隐藏 GPU。  
`evaluate.py` 使用 `--gpu ""` 且需 `torch.cuda.is_available()==False` 才会走 CPU（建议同样 `export MCMIPL_FORCE_CPU=1` 或 `CUDA_VISIBLE_DEVICES=""`）。

---

## 模板 A：官方 MCMIPL test（`evaluate.py`，非 InterRec kg_item 协议）

**前置：** Phase B 完成 + 对应 seed 的 `load_rl_epoch` checkpoint 在盘。

```bash
cd /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL

export MCMIPL_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=""

/home/yrh666/venvs/mcmipl-cpu/bin/python evaluate.py \
  --data_name LAST_FM_STAR \
  --load_rl_epoch 50 \
  --seed 2 \
  --gpu "" \
  --embed transe \
  --seq transformer \
  --mode test \
  --eval_num 10 \
  --cand_num 10 \
  --cand_item_num 10
```

将 `LAST_FM_STAR` / `seed` / `load_rl_epoch` 替换为 BOOK、MOVIE、YELP_STAR 及对应 seed。  
**注意：** 仅当该 seed 的 checkpoint 未被覆盖时才有效（当前磁盘多为 seed 2）。

---

## 模板 B：期望的 InterRec 协议 eval（脚本尚不存在）

`scripts/eval_mcmipl_on_interrec_protocol.py` **在 CPU 环境不存在**。  
不可猜测 `--valid_catalog_mode kg_item_all` 等参数。

待脚本实现并部署到：

`/home/yrh666/interrecbaseline-MCMIPL/scripts/eval_mcmipl_on_interrec_protocol.py`

预期形态（**待脚本确认后使用**）：

```bash
cd /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments/baselines/mcmipl_official/MCMIPL

export MCMIPL_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=""

/home/yrh666/venvs/mcmipl-cpu/bin/python \
  /home/yrh666/interrecbaseline-MCMIPL/scripts/eval_mcmipl_on_interrec_protocol.py \
  --data_name YELP_STAR \
  --load_rl_epoch 50 \
  --seed 0 \
  --embed transe \
  --seq transformer \
  --cand_num 10 \
  --cand_item_num 10 \
  --raw_ranked_preview_k 200 \
  --item_only_scan_depth 200 \
  --valid_catalog_mode kg_item_all \
  --export_predictions_jsonl \
  /home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_interrec_protocol_eval/kgitem_eval_yelp_star/mcmipl_predictions_kgitem_yelp_star_full.jsonl
```

（原模板中的 `--gpu -1` 应改为 `MCMIPL_FORCE_CPU=1`，除非脚本作者另行实现 `-1`。）

---

## 模板 C：Yelp Phase B 完成后单 seed 训练命令（非 eval）

当前 screen 会自动顺序跑 Yelp s0→s1→s2。若需手动补跑：

```bash
cd /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments
export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON=/home/yrh666/venvs/mcmipl-cpu/bin/python
export MCMIPL_SAVE_NUM=10
bash run_mcmipl.sh YELP_STAR 0 50 100 10
```

---

## 模板 D：MOVIE seed=1 补跑（Yelp 流水线结束后）

```bash
cd /home/yrh666/interrecbaseline-MCMIPL/main_table_experiments
export MCMIPL_FORCE_CPU=1
export MCMIPL_CPU_PYTHON=/home/yrh666/venvs/mcmipl-cpu/bin/python
export MCMIPL_SAVE_NUM=10
bash run_mcmipl.sh MOVIE 1 50 100 10
```

**不要**对 s1 使用 `MCMIPL_LOAD_RL_EPOCH` 从 s0/s2 续跑 — s1 应完整独立训练。

---

## 模板 E：每个 seed 完成后归档 checkpoint（强烈建议）

避免覆盖，在每个 seed DONE 后执行：

```bash
SLUG=book   # or last_fm_star, movie, yelp_star
SEED=0
ARCH=/home/yrh666/interrecbaseline-MCMIPL/experiments/mcmipl_phase_b_archives/${SLUG}/seed_${SEED}
mkdir -p "$ARCH"
cp -a main_table_experiments/baselines/mcmipl_official/MCMIPL/tmp/${SLUG}/RL-agent "$ARCH/"
```
