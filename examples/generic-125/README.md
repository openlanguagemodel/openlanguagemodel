# Basic training

python -u train.py --epochs 3 --batch_size 16

# With custom settings

python -u train.py --epochs 5 --save_every 500 --early_stopping_patience 10

# Resume training

python -u train.py --resume checkpoints/checkpoint_latest.pt

# Background training with logging

nohup python -u train.py --epochs 10 > logs/train.log 2>&1 &
