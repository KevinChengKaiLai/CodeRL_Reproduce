#!/bin/bash
# Round 5C — Negative-only: ckpt-3000 warmstart + round1 data, only train on negative relative reward.
#
# Hypothesis: masking positive/zero relative-reward samples to 0 gradient forces model
# to only learn "don't do worse than greedy". Round1 data has ~86.8% negative samples.
# Using --data-suffix _round1 for consistency with 5A/5B/5E (single variable: gradient filter).

mkdir -p logs

echo "[$(date +%H:%M)] Round 5C: negative_only from ckpt-3000, round1 data, LR=5e-6, gnorm=0.1"

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model codet5-large \
    --model_path models/rl_actor_best_ckpt3000 \
    --tuning_mode rl \
    --train-path data/APPS/train/ \
    --data-suffix _round1 \
    --save_dir codet5-large_rl_round5c_ckpt3000_negonly_lr5e-06_gnorm0.1_relreturns_r1data \
    --relative_returns \
    --negative_only \
    --epochs 3 \
    --batch-size-per-replica 2 \
    --grad-acc-steps 32 \
    --lr 5e-6 \
    --max_grad_norm 0.1 \
    --save-freq 1000 \
    --log-freq 10 \
    --save_total_limit 5 \
    --fp16 \
    2>&1 | tee logs/train_rl_round5c_r1data.log

echo "[$(date +%H:%M)] Round 5C complete."
