


#!/bin/bash

cd /root/autodl-tmp/AccFlow || exit 1
python train.py \
    model=accflow \
    optimizer.lr=2e-4 \
    epochs=10 \
    batch_size=5 \
    '+target.accumulate_probs=[0.3,0.3,0.3]' \
    num_frames=5 \
    loss_fn=accflowsupLoss \
    train_aug=True \
    'voxel_size=[0.15, 0.15, 0.15]' \
    'point_cloud_range=[-38.4, -38.4, -3.3, 38.4, 38.4, 3.3]' \
    +optimizer.scheduler.name=WarmupCosLR \
    +optimizer.scheduler.max_lr=2e-4 \
    +optimizer.scheduler.total_steps=20000\
    wandb_mode=disabled\
    val_check_interval=0.05 \
# python train.py model=deltaflow optimizer.lr=2e-3 epochs=20 batch_size=2 num_frames=5 loss_fn=deflowLoss train_aug=True "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3.3, 38.4, 38.4, 3.3]" +optimizer.scheduler.name=WarmupCosLR +optimizer.scheduler.max_lr=2e-3 +optimizer.scheduler.total_steps=20000
