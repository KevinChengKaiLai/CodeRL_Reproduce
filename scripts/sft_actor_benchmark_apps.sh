

python generate.py \
    --model_path models/sft_actor \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/APPS/test/ \
    --output_path outputs/codes_sft/ \
    -s 0 -e 500 \
    --num_seqs 10 \
    --num_seqs_per_iter 10 \
    --temperature 0.6

    
