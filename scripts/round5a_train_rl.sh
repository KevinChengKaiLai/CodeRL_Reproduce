#!/bin/bash
# Round 5A — Positive-only filter from ckpt-3000, round1 data (SFT rollouts + round1 critic).
#
# Hypothesis: masking negative/zero relative-reward samples to 0 gradient
# forces model to only learn from cases where stochastic > greedy.
# Round1 data has ~13.2% positive samples (healthier than round2's 2.1%).
# Using --data-suffix _round1 for consistency with 5B/5C/5E (single variable: gradient filter).

mkdir -p logs

echo "[$(date +%H:%M)] Round 5A: positive_only from ckpt-3000, round1 data, LR=5e-6, gnorm=0.1"

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model codet5-large \
    --model_path models/rl_actor_best_ckpt3000 \
    --tuning_mode rl \
    --train-path data/APPS/train/ \
    --data-suffix _round1 \
    --save_dir codet5-large_rl_round5a_ckpt3000_posonly_lr5e-06_gnorm0.1_relreturns_r1data \
    --relative_returns \
    --positive_only \
    --epochs 3 \
    --batch-size-per-replica 2 \
    --grad-acc-steps 32 \
    --lr 5e-6 \
    --max_grad_norm 0.1 \
    --save-freq 1000 \
    --log-freq 10 \
    --save_total_limit 5 \
    --fp16 \
    2>&1 | tee logs/train_rl_round5a_r1data.log

echo "[$(date +%H:%M)] Round 5A complete."
