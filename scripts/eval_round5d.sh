#!/bin/bash
# Eval all checkpoints for Round 5D (SFT rollout + ckpt-3000 baseline, ckpt-3000 warmstart)
SAVE_DIR="exps/codet5-large_rl_round5d_ckpt3000_sftrollout_ckpt3kbaseline_lr5e-06_gnorm0.1_relreturns"
GPU=1

for CKPT in 1000 2000 3000 4000; do
    CODES_DIR="outputs/sampled_code/codes_round5d_ckpt_${CKPT}"
    RESULTS_DIR="outputs/unit_test_score/test_results_round5d_ckpt_${CKPT}"

    if [ ! "$(ls -A ${CODES_DIR} 2>/dev/null)" ]; then
        echo "[$(date +%H:%M)] Round 5D ckpt-${CKPT}: generating..."
        mkdir -p ${CODES_DIR}
        CUDA_VISIBLE_DEVICES=${GPU} python generate.py \
            --model_path ${SAVE_DIR}/checkpoint-${CKPT} \
            --tokenizer_path models/codet5-base-tokenizer \
            --test_path data/APPS/test/ \
            --output_path ${CODES_DIR} \
            -s 0 -e 500 --num_seqs 10 --num_seqs_per_iter 10 --temperature 0.6
    else
        echo "[$(date +%H:%M)] Round 5D ckpt-${CKPT}: codes exist ($(ls ${CODES_DIR} | wc -l) files), skipping."
    fi

    echo "[$(date +%H:%M)] Round 5D ckpt-${CKPT}: running unit tests..."
    mkdir -p ${RESULTS_DIR}
    index=0
    for (( prob=0; prob<500; prob++ )); do
        index=$(( index + 1 ))
        (
            ulimit -v 16000000
            timeout 300 python test_one_solution.py \
                --code_path ${CODES_DIR}/ \
                --output_path ${RESULTS_DIR}/ \
                --test_path data/APPS/test/ \
                --example_tests 0 \
                --i ${prob} 2>/dev/null
        ) &
        if [ $(( index % 16 )) -eq 0 ]; then wait; fi
    done
    wait

    echo "[$(date +%H:%M)] Round 5D ckpt-${CKPT}: $(ls ${RESULTS_DIR}/*.pkl 2>/dev/null | wc -l)/500 pkl"
    PYTHONPATH=/data_new/chengkai/CodeRL_Reproduce python scripts/LAB_GPU_REDO/eval_performance.py \
        ${RESULTS_DIR} ${CODES_DIR}
    echo "[$(date +%H:%M)] Round 5D ckpt-${CKPT}: done."
done

echo "[$(date +%H:%M)] All Round 5D evals complete."
