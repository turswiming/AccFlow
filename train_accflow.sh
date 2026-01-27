#!/bin/bash
# Train AccFlow with self-supervised accumulated error loss
# AccFlow: Uses history frames (like DeltaFlow) + accumulated error training
#
# num_frames=5 means:
#   - Training (HDF5DatasetAccFlow): 8 frames total
#     pch3, pch2, pch1, pc0, pc1, pc2, pc3, pc4
#     (3 history + 2 current + 4 future for accumulated error)
#   - Validation (HDF5Dataset): 5 frames (history only)
#     pch3, pch2, pch1, pc0, pc1
#
# General formula for num_frames=N:
#   num_history = N - 2
#   num_future = 4 (training only) - pc2, pc3, pc4, pc5
#   total_train_frames = 2*N - 2

cd /workspace/OpenSceneFlow && \
source /opt/miniforge3/etc/profile.d/conda.sh && \
conda activate opensf && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/venv/opensf/lib/python3.8/site-packages/torch/lib && \
python train.py \
    model=accflow \
    loss_fn=accflowLoss \
    '+add_seloss={chamfer_dis:1.0,static_flow_loss:1.0,dynamic_chamfer_dis:1.0,cluster_based_pc0pc1:1.0}' \
    optimizer.lr=2e-4 \
    epochs=15 \
    batch_size=4 \
    num_frames=5 \
    '+target.accumulate_probs=[0,3,0.3,0.3]' \
    'voxel_size=[0.15,0.15,0.15]' \
    'point_cloud_range=[-38.4,-38.4,-3.15,38.4,38.4,3.15]' \
    val_check_interval=0.05 \
    wandb_mode=disabled\
    accumulate_grad_batches=5\
    checkpoint="/workspace/OpenSceneFlow/logs/jobs/accflow-accflow-0/12-28-22-34best5frame/checkpoints/08_accflow-accflow-v2.ckpt"
