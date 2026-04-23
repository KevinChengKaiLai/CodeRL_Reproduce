# Kevin made in  04/22/2026


# Add CUDA_VISIBLE_DEVICES=0 because the dataparallel has some issue on this GPU
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --model codet5-small \
  --model_path models/codet5-small \
  --tuning_mode critic \
  --train-path data/APPS/train/ \
  --epochs 10 \
  --lr 5e-5 \
  --batch-size-per-replica 16 \
  --grad-acc-steps 16 \
  --save-freq 1000 \
  --log-freq 10 \
  --save_total_limit 2 \
  --fp16