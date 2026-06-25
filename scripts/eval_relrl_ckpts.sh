#!/bin/bash
# Evaluate all available checkpoints from the relative-returns RL run.
# Safe to re-run: skips checkpoints whose eval_summary.txt already exists.
# Sampling on GPU1; unit tests on CPU (16 parallel workers).

CKPT_BASE="exps/codet5-large_rl_bs4x16_lr2e-05_newcritic_relreturns"
TOKENIZER="models/codet5-base-tokenizer"
TEST_PATH="data/APPS/test/"

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

sample_checkpoint() {
    local ckpt=$1
    local codes_dir="outputs/sampled_code/codes_relrl_ckpt_${ckpt}"
    echo "=== [SAMPLING] ckpt${ckpt} on GPU1 ==="
    mkdir -p "${codes_dir}"
    CUDA_VISIBLE_DEVICES=1 python generate.py \
        --model_path "${CKPT_BASE}/checkpoint-${ckpt}" \
        --tokenizer_path "${TOKENIZER}" \
        --test_path "${TEST_PATH}" \
        --output_path "${codes_dir}/" \
        -s 0 -e 500 \
        --num_seqs 10 \
        --num_seqs_per_iter 10 \
        --temperature 0.6
    echo "[SAMPLING DONE] ckpt${ckpt}"
}

# Discover all available checkpoints in the save dir
if [ ! -d "${CKPT_BASE}" ]; then
    echo "Save dir not found: ${CKPT_BASE}"
    echo "Run training first: bash scripts/LAB_GPU_REDO/train_rl_actor.sh"
    exit 1
fi

mapfile -t CKPT_DIRS < <(find "${CKPT_BASE}" -maxdepth 1 -name "checkpoint-*" -type d | sort -V)

if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    echo "No checkpoints found in ${CKPT_BASE} yet. Check back later."
    exit 0
fi

echo "Found ${#CKPT_DIRS[@]} checkpoint(s) to evaluate."
EVALUATED=()

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    ckpt=$(basename "${ckpt_dir}" | sed 's/checkpoint-//')
    results_dir="outputs/unit_test_score/test_results_relrl_ckpt_${ckpt}"

    if [ -f "${results_dir}/eval_summary.txt" ]; then
        echo "[SKIP] ckpt${ckpt} already evaluated."
        EVALUATED+=("${ckpt}")
        continue
    fi

    sample_checkpoint "${ckpt}"
    run_unit_tests_and_eval "${ckpt}"
    EVALUATED+=("${ckpt}")
done

echo ""
echo "============================================================"
echo "  Summary: pass@k across evaluated checkpoints (relrl run)"
echo "============================================================"
for ckpt in "${EVALUATED[@]}"; do
    summary="outputs/unit_test_score/test_results_relrl_ckpt_${ckpt}/eval_summary.txt"
    if [ -f "${summary}" ]; then
        pass1=$(grep  "pass@1 "      "${summary}" | awk '{print $3}')
        pass5=$(grep  "pass@5 "      "${summary}" | awk '{print $3}')
        pass10=$(grep "pass@10"      "${summary}" | awk '{print $3}')
        unique=$(grep "Mean uniqueness" "${summary}" | awk '{print $5}')
        echo "  ckpt${ckpt}: pass@1=${pass1}  pass@5=${pass5}  pass@10=${pass10}  uniqueness=${unique}"
    fi
done
echo "============================================================"
