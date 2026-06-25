#!/bin/bash
# Parallel eval pipeline for relrl ckpts 12000, 13000, 14000.
#
# Strategy:
#   GPU0: ckpt-13000 (starts immediately)
#   GPU1: ckpt-12000 (sampling already in progress, PID 3962078)
#         then ckpt-14000 (after ckpt-12000 eval finishes)
#
# Generation timing is printed after each sampling run as a collapse diagnostic:
#   healthy model ~ 5-15s/problem, collapsed model ~ 30-60s/problem.

CKPT_BASE="exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns"
TOKENIZER="models/codet5-base-tokenizer"
TEST_PATH="data/APPS/test/"

# PID of the existing sequential eval script (parent of ckpt-12000 generate.py).
# Killing it with SIGTERM leaves the foreground child (generate.py) running as an orphan.
OLD_EVAL_PID=3962049
# PID of the ckpt-12000 generate.py process already running.
CKPT12_GENERATE_PID=3962078

# ---------------------------------------------------------------------------
run_unit_tests_and_eval() {
    local ckpt=$1
    local codes_dir="outputs/sampled_code/codes_relrl_ckpt_${ckpt}"
    local results_dir="outputs/unit_test_score/test_results_relrl_ckpt_${ckpt}"

    echo "=== [UNIT TESTS] ckpt${ckpt} ==="
    mkdir -p "${results_dir}"
    local idx=0
    for (( prob=0; prob<500; prob++ )); do
        idx=$(( idx + 1 ))
        ( timeout 300 python test_one_solution.py \
            --code_path "${codes_dir}/" \
            --output_path "${results_dir}/" \
            --test_path "${TEST_PATH}" \
            --example_tests 0 \
            --i "${prob}" ) &
        if (( idx % 16 == 0 )); then wait; fi
    done
    wait
    echo "[UNIT TESTS DONE] ckpt${ckpt}: $(ls "${results_dir}/" | wc -l)/500"

    echo "=== [EVAL] ckpt${ckpt} ==="
    PYTHONPATH=/data_new/chengkai/CodeRL_Reproduce python scripts/LAB_GPU_REDO/eval_performance.py \
        "${results_dir}/" \
        "${codes_dir}/"
    echo "[EVAL DONE] ckpt${ckpt}"
}

# Sample a checkpoint on the given GPU, print timing, then run unit tests + eval.
# Skips everything if eval_summary.txt already exists.
sample_and_eval() {
    local ckpt=$1
    local gpu=$2
    local codes_dir="outputs/sampled_code/codes_relrl_ckpt_${ckpt}"
    local results_dir="outputs/unit_test_score/test_results_relrl_ckpt_${ckpt}"

    if [ -f "${results_dir}/eval_summary.txt" ]; then
        echo "[SKIP] ckpt${ckpt} already evaluated."
        return
    fi

    echo "=== [SAMPLING] ckpt${ckpt} on GPU${gpu} ==="
    mkdir -p "${codes_dir}"

    local t0
    t0=$(date +%s)

    CUDA_VISIBLE_DEVICES=${gpu} python generate.py \
        --model_path "${CKPT_BASE}/checkpoint-${ckpt}" \
        --tokenizer_path "${TOKENIZER}" \
        --test_path "${TEST_PATH}" \
        --output_path "${codes_dir}/" \
        -s 0 -e 500 \
        --num_seqs 10 \
        --num_seqs_per_iter 10 \
        --temperature 0.6

    local t1
    t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local n_solutions=$(( 500 * 10 ))
    echo "[TIMING] ckpt${ckpt} sampling: ${elapsed}s total | $(( elapsed / 500 ))s/problem | $(( elapsed / n_solutions ))s/solution"
    echo "[SAMPLING DONE] ckpt${ckpt}"

    run_unit_tests_and_eval "${ckpt}"
}

# ---------------------------------------------------------------------------
echo "=== eval_relrl_parallel.sh starting at $(date) ==="
echo "GPU0: ckpt-13000 (starting now)"
echo "GPU1: ckpt-12000 (already sampling, PID ${CKPT12_GENERATE_PID}) → then ckpt-14000"
echo ""

# Step 1: Kill the sequential eval_relrl_ckpts.sh parent to prevent conflicts.
# SIGTERM does not propagate to its foreground child (generate.py), which survives.
if kill -0 ${OLD_EVAL_PID} 2>/dev/null; then
    echo "[SETUP] Killing old eval script (PID ${OLD_EVAL_PID}) to prevent conflicts..."
    kill ${OLD_EVAL_PID} 2>/dev/null || true
    sleep 2
fi

# Verify the ckpt-12000 generate.py is still alive.
if ! kill -0 ${CKPT12_GENERATE_PID} 2>/dev/null; then
    echo "[INFO] ckpt-12000 generate.py (PID ${CKPT12_GENERATE_PID}) is no longer running."
    echo "       Sampling may already be complete — proceeding to unit tests."
fi

# Step 2: Start ckpt-13000 on GPU0 in the background.
(
    sample_and_eval 13000 0
) &
PID_GPU0=$!
echo "[LAUNCHED] ckpt-13000 pipeline on GPU0 (subshell PID ${PID_GPU0})"

# Step 3: Wait for the existing ckpt-12000 generate.py to finish.
echo "[WAITING] Polling for ckpt-12000 sampling (PID ${CKPT12_GENERATE_PID}) to finish..."
while kill -0 ${CKPT12_GENERATE_PID} 2>/dev/null; do
    echo "  [$(date +%H:%M)] ckpt-12000: $(ls outputs/sampled_code/codes_relrl_ckpt_12000/ 2>/dev/null | wc -l)/500 done"
    sleep 60
done
echo "[$(date +%H:%M)] ckpt-12000 sampling finished."

# Step 4: Unit tests + eval for ckpt-12000 (main thread, GPU1 now free).
results_12="outputs/unit_test_score/test_results_relrl_ckpt_12000"
if [ -f "${results_12}/eval_summary.txt" ]; then
    echo "[SKIP] ckpt-12000 already evaluated."
else
    run_unit_tests_and_eval 12000
fi

# Step 5: GPU1 is free — start ckpt-14000 on GPU1 in background.
(
    sample_and_eval 14000 1
) &
PID_GPU1=$!
echo "[LAUNCHED] ckpt-14000 pipeline on GPU1 (subshell PID ${PID_GPU1})"

# Step 6: Wait for both background pipelines to finish.
wait ${PID_GPU0}
echo "[$(date +%H:%M)] ckpt-13000 pipeline complete."

wait ${PID_GPU1}
echo "[$(date +%H:%M)] ckpt-14000 pipeline complete."

# Step 7: Summary table.
echo ""
echo "============================================================"
echo "  Final summary: relrl ckpts 12000–14000"
echo "============================================================"
for ckpt in 12000 13000 14000; do
    summary="outputs/unit_test_score/test_results_relrl_ckpt_${ckpt}/eval_summary.txt"
    if [ -f "${summary}" ]; then
        pass1=$(grep  "pass@1 "        "${summary}" | awk '{print $3}')
        pass5=$(grep  "pass@5 "        "${summary}" | awk '{print $3}')
        pass10=$(grep "pass@10"        "${summary}" | awk '{print $3}')
        unique=$(grep "Mean uniqueness" "${summary}" | awk '{print $5}')
        echo "  ckpt${ckpt}: pass@1=${pass1}  pass@5=${pass5}  pass@10=${pass10}  uniqueness=${unique}"
    else
        echo "  ckpt${ckpt}: eval_summary.txt not found — eval may have failed"
    fi
done
echo "============================================================"
echo "=== eval_relrl_parallel.sh finished at $(date) ==="
