mkdir -p logs

# Restart with num_seqs_per_iter=10 (5x speedup, ~8GB VRAM expected)
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path models/sft_actor \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ \
    --output_path outputs/codes \
    -s 0 -e 2500 \
    --num_seqs 20 \
    --num_seqs_per_iter 10 \
    --temperature 0.6 \
    > logs/generate_train_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python generate.py \
    --model_path models/sft_actor \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ \
    --output_path outputs/codes \
    -s 2500 -e 5000 \
    --num_seqs 20 \
    --num_seqs_per_iter 10 \
    --temperature 0.6 \
    > logs/generate_train_gpu1.log 2>&1 &