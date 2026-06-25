#!/bin/bash
# Round 5D — SFT rollouts vs ckpt-3000 greedy baseline, ckpt-3000 warmstart.
#
# vs Round 5B: same rollout (SFT stochastic, _round1), same warmstart (ckpt-3000),
#              but baseline changed from SFT greedy → ckpt-3000 greedy (stronger baseline).
# Hypothesis: harder baseline reduces positive sample rate below 5B's 13.2%,
#             testing whether a tougher comparative signal helps or hurts stability.

mkdir -p logs

echo "[$(date +%H:%M)] Round 5D: SFT rollouts, ckpt-3000 baseline, ckpt-3000 warmstart, LR=5e-6, gnorm=0.1"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model codet5-large \
    --model_path models/rl_actor_best_ckpt3000 \
    --tuning_mode rl \
    --train-path data/APPS/train/ \
    --data-suffix _round1 \
    --baseline-suffix "" \
    --save_dir codet5-large_rl_round5d_ckpt3000_sftrollout_ckpt3kbaseline_lr5e-06_gnorm0.1_relreturns \
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
    2>&1 | tee logs/train_rl_round5d.log

echo "[$(date +%H:%M)] Round 5D complete."
