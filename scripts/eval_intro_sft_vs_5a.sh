#!/bin/bash
# Eval SFT baseline + Round 5A checkpoints on APPS Introductory split
# Introductory problems: indices 4000-4499 (500 problems)
# n=10, temp=0.6

mkdir -p logs

# ── SFT baseline (GPU 0) ──────────────────────────────────────────────────────
run_eval() {
    local TAG=$1
    local MODEL_PATH=$2
    local GPU=$3
    local CODES_DIR="outputs/sampled_code/codes_${TAG}_intro"
    local RESULTS_DIR="outputs/unit_test_score/test_results_${TAG}_intro"

    mkdir -p ${CODES_DIR} ${RESULTS_DIR}

    if [ ! "$(ls -A ${CODES_DIR} 2>/dev/null)" ]; then
        echo "[$(date +%H:%M)] ${TAG}: generating on GPU${GPU}..."
        CUDA_VISIBLE_DEVICES=${GPU} python generate.py \
            --model_path ${MODEL_PATH} \
            --tokenizer_path models/codet5-base-tokenizer \
            --test_path data/APPS/test/ \
            --output_path ${CODES_DIR} \
            -s 4000 -e 4500 \
            --num_seqs 10 \
            --num_seqs_per_iter 10 \
            --temperature 0.6
    else
        echo "[$(date +%H:%M)] ${TAG}: codes exist ($(ls ${CODES_DIR} | wc -l) files), skipping generate."
    fi

    echo "[$(date +%H:%M)] ${TAG}: running unit tests..."
    index=0
    for (( prob=4000; prob<4500; prob++ )); do
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

    echo "[$(date +%H:%M)] ${TAG}: $(ls ${RESULTS_DIR}/*.pkl 2>/dev/null | wc -l)/500 pkl files"
    echo "[$(date +%H:%M)] ${TAG}: computing pass@k..."
    python scripts/LAB_GPU_REDO/eval_performance.py ${RESULTS_DIR} ${CODES_DIR}
    echo "[$(date +%H:%M)] ${TAG}: done."
}

# SFT on GPU 0
run_eval "sft" "models/sft_actor" 0 &
SFT_PID=$!

# 5A ckpts on GPU 1 (sequential — one at a time to avoid OOM)
(
    SAVE_DIR="exps/codet5-large_rl_round5a_ckpt3000_posonly_lr5e-06_gnorm0.1_relreturns"
    for CKPT in 1000 2000 3000 4000; do
        run_eval "round5a_ckpt${CKPT}" "${SAVE_DIR}/checkpoint-${CKPT}" 1
    done
) &
ROUND5A_PID=$!

wait ${SFT_PID}
echo "[$(date +%H:%M)] SFT eval complete."
wait ${ROUND5A_PID}
echo "[$(date +%H:%M)] Round 5A eval complete."

echo ""
echo "=== Summary ==="
for TAG in sft round5a_ckpt1000 round5a_ckpt2000 round5a_ckpt3000 round5a_ckpt4000; do
    RESULTS_DIR="outputs/unit_test_score/test_results_${TAG}_intro"
    if [ -f "${RESULTS_DIR}/eval_summary.json" ]; then
        echo "--- ${TAG} ---"
        python3 -c "
import json
d = json.load(open('${RESULTS_DIR}/eval_summary.json'))
pk = d['pass_at_k']
oc = d['outcome_pcts']
print(f'  pass@1={pk.get(\"1\",0):.2f}%  pass@5={pk.get(\"5\",0):.2f}%  pass@10={pk.get(\"10\",0):.2f}%')
print(f'  CE%={oc[\"CompileError\"]:.1f}  RE%={oc[\"RuntimeError\"]:.1f}  FT%={oc[\"FailedTest\"]:.1f}  PT%={oc[\"PassedTest\"]:.2f}')
print(f'  Uniqueness={d[\"uniqueness_mean_pct\"]:.1f}%')
"
    fi
done
