#!/bin/bash
# 按顺序跑完所有数据集，每个数据集跑 3 个 seed（串行）
# 用法: bash run_all_datasets.sh
# 在 tmux 里运行，自动接力

PYTHON=/home/yurh/.conda/envs/mcmipl-reproduce/bin/python
MCMIPL=/home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL
LOGS=/home/yurh/main_table_experiments/logs
# LAST_FM_STAR seed0 实测：SR@15 在 step10 就达到峰值（0.90），之后不再上升，
# 故后续三个数据集将 max_steps 从 100 缩短到 50，节省约 50% 训练时间。
# eval_num 保持 10（每 10 步评估一次），确保能采到 step10 的峰值点；
# 如改成 eval_num=25，第一次 eval 在 step25，会错过早期最优 checkpoint。
MAX_STEPS=50
SAMPLE_TIMES=100
EVAL_NUM=10
# 数据若要按 InterRec 顺序走「导出 CSV → preprocess_dataset」，见
# baselines/mcmipl_official/scripts/prepare_data.sh 里 EXPORT_INTERREC_CSV /
# interrec/scripts/convert_mcmipl_to_interrec_csv.py（与训练脚本独立）。

mkdir -p "$LOGS"

run_one() {
    local DATASET=$1
    local SEED=$2
    local LOG="$LOGS/train_${DATASET}_s${SEED}.log"

    # 如果已经跑完就跳过
    if grep -q "DONE:" "$LOG" 2>/dev/null; then
        echo "[SKIP] $DATASET seed=$SEED already done"
        return
    fi

    echo "========================================"
    echo "[START] $DATASET | seed=$SEED | $(date)"
    echo "========================================"

    cd "$MCMIPL" || exit 1
    $PYTHON -u RL_model.py \
        --data_name "$DATASET" \
        --embed transe \
        --seed "$SEED" \
        --gpu 0 \
        --max_steps $MAX_STEPS \
        --sample_times $SAMPLE_TIMES \
        --attr_num 20 \
        --choice_num 4 \
        --max_turn 15 \
        --eval_num $EVAL_NUM \
        --save_num $MAX_STEPS \
        2>&1 | tee -a "$LOG"

    echo "=== DONE: $DATASET seed=$SEED at $(date) ===" | tee -a "$LOG"
    echo ""
}

# =============================================
# LAST_FM_STAR 已在 tmux 里并行跑，等它完成后
# 本脚本接着跑 YELP_STAR, BOOK, MOVIE（3 seed 串行）
# =============================================

echo "=========================================="
echo "  等待 LAST_FM_STAR 三个 seed 全部完成..."
echo "=========================================="
while true; do
    done_count=0
    for s in 0 1 2; do
        if grep -q "DONE:" "$LOGS/train_LAST_FM_STAR_s${s}.log" 2>/dev/null; then
            done_count=$((done_count + 1))
        fi
    done
    if [ "$done_count" -eq 3 ]; then
        echo "LAST_FM_STAR 全部完成！开始 YELP_STAR..."
        break
    fi
    echo "  [$(date +%H:%M)] LAST_FM_STAR 完成 $done_count/3 个 seed，继续等待..."
    sleep 300
done

# YELP_STAR
for s in 0 1 2; do
    run_one YELP_STAR $s
done

# BOOK
for s in 0 1 2; do
    run_one BOOK $s
done

# MOVIE
for s in 0 1 2; do
    run_one MOVIE $s
done

echo ""
echo "=========================================="
echo "  全部 4 个数据集 × 3 seed 已完成！"
echo "  结束时间: $(date)"
echo "=========================================="
