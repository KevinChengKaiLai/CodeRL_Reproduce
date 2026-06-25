#!/bin/bash
# Round 5B — Weak baseline: ckpt-3000 warmstart + round1 data (SFT rollouts vs SFT greedy).
#
# Hypothesis: stronger warmstart (ckpt-3000 > SFT) with healthier data (13.2% positive)
# will reach higher peak than Round 4 (SFT start + same data, peak=1.20% at ckpt-5000).
# No code changes needed — just different model_path and data_suffix.

mkdir -p logs

echo "[$(date +%H:%M)] Round 5B: ckpt-3000 warmstart, round1 data (weak SFT baseline), LR=5e-6, gnorm=0.1"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model codet5-large \
    --model_path models/rl_actor_best_ckpt3000 \
    --tuning_mode rl \
    --train-path data/APPS/train/ \
    --data-suffix _round1 \
    --save_dir codet5-large_rl_round5b_ckpt3000_weakbaseline_lr5e-06_gnorm0.1_relreturns \
    --relative_returns \
    --epochs 3 \
    --batch-size-per-replica 2 \
    --grad-acc-steps 32 \
    --lr 5e-6 \
    --max_grad_norm 0.1 \
    --save-freq 1000 \
    --log-freq 10 \
    --save_total_limit 5 \
    --fp16 \
    2>&1 | tee logs/train_rl_round5b.log

echo "[$(date +%H:%M)] Round 5B complete."


