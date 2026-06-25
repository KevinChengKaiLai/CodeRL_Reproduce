#!/bin/bash
# Full pipeline to generate baseline_solutions.json for all 5000 train problems.
# Step 1: Greedy sample (temperature=0, num_seqs=1) from SFT actor on both GPUs
# Step 2: Run unit tests on baseline codes
# Step 3: Write baseline_solutions.json per problem into data/APPS/train/

set -e
mkdir -p outputs/sampled_code/codes_baseline outputs/unit_test_score/test_results_baseline logs

# ── Step 1: Greedy sampling ───────────────────────────────────────────────────
echo "=== [STEP 1] Greedy sampling from SFT actor ==="

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path models/sft_actor \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ \
    --output_path outputs/sampled_code/codes_baseline/ \
    -s 0 -e 2500 \
    --num_seqs 1 \
    --num_seqs_per_iter 1 \
    --temperature 0 \
    > logs/baseline_sample_gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python generate.py \
    --model_path models/sft_actor \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ \
    --output_path outputs/sampled_code/codes_baseline/ \
    -s 2500 -e 5000 \
    --num_seqs 1 \
    --num_seqs_per_iter 1 \
    --temperature 0 \
    > logs/baseline_sample_gpu1.log 2>&1 &
PID1=$!

wait $PID0 && echo "[DONE] Sampling GPU0" || echo "[FAILED] Sampling GPU0"
wait $PID1 && echo "[DONE] Sampling GPU1" || echo "[FAILED] Sampling GPU1"
echo "Codes generated: $(ls outputs/sampled_code/codes_baseline/ | wc -l)/5000"

# ── Step 2: Unit tests ────────────────────────────────────────────────────────
echo "=== [STEP 2] Running unit tests on baseline codes ==="

index=0
for (( i=0; i<5000; i++ )); do
    index=$(( index + 1 ))
    (
    timeout 300 python test_one_solution.py \
        --code_path outputs/sampled_code/codes_baseline/ \
        --output_path outputs/unit_test_score/test_results_baseline/ \
        --test_path data/APPS/train/ \
        --example_tests 0 \
        --i $i
    ) &
    if (( index % 16 == 0 )); then wait; fi
done
wait
echo "[DONE] Unit tests: $(ls outputs/unit_test_score/test_results_baseline/ | wc -l)/5000"

# ── Step 3: Write baseline_solutions.json ─────────────────────────────────────
echo "=== [STEP 3] Writing baseline_solutions.json per problem ==="

PYTHONPATH=/data_new/chengkai/CodeRL_Reproduce python scripts/make_baseline_solutions.py \
    --codes_dir outputs/sampled_code/codes_baseline/ \
    --results_dir outputs/unit_test_score/test_results_baseline/ \
    --train_path data/APPS/train/

echo "=== ALL DONE — baseline_solutions.json ready for --relative_returns ==="
