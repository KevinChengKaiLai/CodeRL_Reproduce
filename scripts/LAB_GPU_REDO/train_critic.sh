# Kevin made in  04/22/2026

python train.py \
    --model codet5-base \
    --tuning_mode critic \
    --train-path data/APPS/train/ \
    --model_path models/codet5-base \
    --epochs 10  \
    --lr 5e-5 \
    --batch-size-per-replica 32 \
    --grad-acc-steps 8 \
    --save-freq 1000 \
    --log-freq 10 \
    --save_total_limit 2 \
    --log-freq 10 \
    --fp16 
    