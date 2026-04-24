
# no --gt_solution because we are generate critix scores from SFT actor sample

python generate.py \
    --model_path exps/codet5-small_critic_bs16x16_lr5e-05/checkpoint-8000 \
    --tokenizer_path models/codet5-base-tokenizer \
    --test_path data/APPS/train/ \
    --output_path data/APPS/train/ \
    --critic_scores \
    --start 0 \
    --end 5000