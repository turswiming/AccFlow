cd /root/autodl-tmp/AccFlow
python find_lr.py model=accflow loss_fn=accflowsupLoss \
    '+target.accumulate_probs=[1,0,0]' \
    num_frames=5 batch_size=5\
    'voxel_size=[0.15, 0.15, 0.15]' \
    batch_size=5 \
    'point_cloud_range=[-38.4, -38.4, -3.3, 38.4, 38.4, 3.3]' \
    accumulate_grad_batches=4\