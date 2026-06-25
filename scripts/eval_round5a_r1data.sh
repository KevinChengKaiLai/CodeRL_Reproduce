#!/bin/bash
# Eval all checkpoints for Round 5A r1data (positive_only, ckpt-3000 warmstart, round1 data)
# Introductory split: indices 4000-4499 (500 problems)
SAVE_DIR="exps/codet5-large_rl_round5a_ckpt3000_posonly_lr5e-06_gnorm0.1_relreturns_r1data"
GPU=1

for CKPT in 1000 2000 3000 4000; do
    CODES_DIR="outputs/sampled_code/codes_round5a_r1data_ckpt_${CKPT}_intro"
    RESULTS_DIR="outputs/unit_test_score/test_results_round5a_r1data_ckpt_${CKPT}_intro"

    if [ ! "$(ls -A ${CODES_DIR} 2>/dev/null)" ]; then
        echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: generating..."
        mkdir -p ${CODES_DIR}
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        CUDA_VISIBLE_DEVICES=${GPU} python generate.py \
            --model_path ${SAVE_DIR}/checkpoint-${CKPT} \
            --tokenizer_path models/codet5-base-tokenizer \
            --test_path data/APPS/test/ \
            --output_path ${CODES_DIR} \
            -s 4000 -e 4500 \
            --num_seqs 10 \
            --num_seqs_per_iter 10 \
            --temperature 0.6
    else
        echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: codes exist ($(ls ${CODES_DIR} | wc -l) files), skipping generate."
    fi

    echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: running unit tests..."
    mkdir -p ${RESULTS_DIR}
    index=0
    for (( prob=0; prob<500; prob++ )); do  # --i is offset into intro problems (0-499)
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
        if [ $(( index % 16 )) -eq 0 ]; then
            wait
        fi
    done
    wait

    echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: $(ls ${RESULTS_DIR}/*.pkl 2>/dev/null | wc -l)/500 pkl files"
    echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: computing pass@k..."
    PYTHONPATH=/data_new/chengkai/CodeRL_Reproduce python scripts/LAB_GPU_REDO/eval_performance.py ${RESULTS_DIR} ${CODES_DIR}
    echo "[$(date +%H:%M)] Round 5A r1data ckpt-${CKPT}: done."
done

echo "[$(date +%H:%M)] All Round 5A r1data evals complete."
