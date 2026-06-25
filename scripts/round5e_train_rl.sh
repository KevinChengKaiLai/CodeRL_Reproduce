#!/bin/bash
# Round 5E — Positive-only from ckpt-3000, round1 data, max_grad_norm=1.0.
# Identical to 5A except max_grad_norm=1.0 (vs 5A's 0.1).
# Both use round1 data for fair comparison (single variable: grad norm clipping).
#
# Hypothesis: larger gradient headroom lets model learn faster from ~13.2% positive
# samples in round1 data. Risk: RL gradient spikes may destabilize training.

mkdir -p logs

echo "[$(date +%H:%M)] Round 5E: positive_only from ckpt-3000, round1 data, LR=5e-6, gnorm=1.0"

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model codet5-large \
    --model_path models/rl_actor_best_ckpt3000 \
    --tuning_mode rl \
    --train-path data/APPS/train/ \
    --data-suffix _round1 \
    --save_dir codet5-large_rl_round5e_ckpt3000_posonly_lr5e-06_gnorm1.0_relreturns_r1data \
    --relative_returns \
    --positive_only \
    --epochs 3 \
    --batch-size-per-replica 2 \
    --grad-acc-steps 32 \
    --lr 5e-6 \
    --max_grad_norm 1.0 \
    --save-freq 1000 \
    --log-freq 10 \
    --save_total_limit 5 \
    --fp16 \
    2>&1 | tee logs/train_rl_round5e_r1data.log

echo "[$(date +%H:%M)] Round 5E complete."
