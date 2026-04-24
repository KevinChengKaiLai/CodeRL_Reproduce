
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_path models/sft_actor/ \
    --save_dir exps/codet5-large_rl_bs4x16_lr2e-05_newcritic \
    --train-path data/APPS/train/ \
    --tuning_mode rl \
    --epochs 10 \
    --lr 2e-5 \
    --batch_size_per_replica 4 \
    --grad_acc_steps 16 \
    --save_freq 1000 \
    --log_freq 10 \
    --save_total_limit 10 \
    --fp16