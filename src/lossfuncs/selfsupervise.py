"""
# Created: 2023-07-17 00:00
# Updated: 2025-08-07 00:01
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow)
# * HiMo (https://kin-zhang.github.io/HiMo)
#
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Define the self-supervised (without GT) loss function for training.
#
"""
import torch
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4

def seflowppLoss(res_dict, timer=None):
    pch1_label = res_dict['pch1_labels']
    pc0_label = res_dict['pc0_labels']
    pc1_label = res_dict['pc1_labels']

    pch1 = res_dict['pch1']
    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow
    pseduo_pch1from0 = pc0 - est_flow

    unique_labels = torch.unique(pc0_label)
    pc0_dynamic = pc0[pc0_label > 0]
    pc1_dynamic = pc1[pc1_label > 0]

    # fpc1_dynamic = pseudo_pc1from0[pc0_label > 0]
    # NOTE(Qingwen): since we set THREADS_PER_BLOCK is 256
    have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc1_dynamic.shape[0] > 256)

    # first item loss: chamfer distance
    # timer[5][1].start("MyCUDAChamferDis")
    chamfer_dis = MyCUDAChamferDis(pseudo_pc1from0, pc1, truncate_dist=TRUNCATED_DIST) + MyCUDAChamferDis(pseduo_pch1from0, pch1, truncate_dist=TRUNCATED_DIST)
    # timer[5][1].stop()
    
    # second item loss: dynamic chamfer distance
    # timer[5][2].start("DynamicChamferDistance")
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    if have_dynamic_cluster:
        dynamic_chamfer_dis += MyCUDAChamferDis(pseudo_pc1from0[pc0_label > 0], pc1_dynamic, truncate_dist=TRUNCATED_DIST)
        if pch1[pch1_label > 0].shape[0] > 256:
            dynamic_chamfer_dis += MyCUDAChamferDis(pseduo_pch1from0[pc0_label > 0], pch1[pch1_label > 0], truncate_dist=TRUNCATED_DIST)
    # timer[5][2].stop()

    # third item loss: exclude static points' flow
    # NOTE(Qingwen): add in the later part on label==0
    static_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    
    # fourth item loss: same label points' flow should be the same
    # timer[5][3].start("SameClusterLoss")
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)
    for label in unique_labels:
        mask = pc0_label == label
        if label == 0:
            # Eq. 6 in the SeFlow paper
            static_cluster_loss += torch.linalg.vector_norm(est_flow[mask, :], dim=-1).mean()
        # NOTE(Qingwen) 2025-04-23: label=1 is dynamic but no cluster id satisfied
        elif label > 1 and have_dynamic_cluster:
            cluster_id_flow = est_flow[mask, :]
            cluster_nnd = raw_dist0[mask]
            if cluster_nnd.shape[0] <= 0:
                continue

            # Eq. 8 in the SeFlow paper
            sorted_idxs = torch.argsort(cluster_nnd, descending=True)
            nearby_label = pc1_label[raw_idx0[mask][sorted_idxs]] # nonzero means dynamic in label
            non_zero_valid_indices = torch.nonzero(nearby_label > 0)
            if non_zero_valid_indices.shape[0] <= 0:
                continue
            max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]
            
            # Eq. 9 in the SeFlow paper
            max_flow = pc1[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

            # Eq. 10 in the SeFlow paper
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)))
    
    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean() # Eq. 11 in the SeFlow paper
    elif have_dynamic_cluster:
        moved_cluster_loss = torch.mean(raw_dist0[raw_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_dist1[raw_dist1 <= TRUNCATED_DIST])
    # timer[5][3].stop()

    res_loss = {
        'chamfer_dis': chamfer_dis / 2.0,
        'dynamic_chamfer_dis': dynamic_chamfer_dis / 2.0,
        'static_flow_loss': static_cluster_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }
    return res_loss

def seflowLoss(res_dict, timer=None):
    pc0_label = res_dict['pc0_labels']
    pc1_label = res_dict['pc1_labels']

    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow

    unique_labels = torch.unique(pc0_label)
    pc0_dynamic = pc0[pc0_label > 0]
    pc1_dynamic = pc1[pc1_label > 0]
    # fpc1_dynamic = pseudo_pc1from0[pc0_label > 0]
    # NOTE(Qingwen): since we set THREADS_PER_BLOCK is 256
    have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc1_dynamic.shape[0] > 256)

    # first item loss: chamfer distance
    # timer[5][1].start("MyCUDAChamferDis")
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    est_dist0, est_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc1from0, pc1)
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    chamfer_dis = torch.mean(est_dist0[est_dist0 <= TRUNCATED_DIST]) + torch.mean(est_dist1[est_dist1 <= TRUNCATED_DIST])
    # timer[5][1].stop()
    
    # second item loss: dynamic chamfer distance
    # timer[5][2].start("DynamicChamferDistance")
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    if have_dynamic_cluster:
        dynamic_chamfer_dis += MyCUDAChamferDis(pseudo_pc1from0[pc0_label>0], pc1_dynamic, truncate_dist=TRUNCATED_DIST)
    # timer[5][2].stop()

    # third item loss: exclude static points' flow
    # NOTE(Qingwen): add in the later part on label==0
    static_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    
    # fourth item loss: same label points' flow should be the same
    # timer[5][3].start("SameClusterLoss")
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)
    for label in unique_labels:
        mask = pc0_label == label
        if label == 0:
            # Eq. 6 in the paper
            static_cluster_loss += torch.linalg.vector_norm(est_flow[mask, :], dim=-1).mean()
        # NOTE(Qingwen) 2025-04-23: label=1 is dynamic but no cluster id satisfied
        elif label > 1 and have_dynamic_cluster:
            cluster_id_flow = est_flow[mask, :]
            cluster_nnd = raw_dist0[mask]
            if cluster_nnd.shape[0] <= 0:
                continue

            # Eq. 8 in the paper
            sorted_idxs = torch.argsort(cluster_nnd, descending=True)
            nearby_label = pc1_label[raw_idx0[mask][sorted_idxs]] # nonzero means dynamic in label
            non_zero_valid_indices = torch.nonzero(nearby_label > 0)
            if non_zero_valid_indices.shape[0] <= 0:
                continue
            max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]
            
            # Eq. 9 in the paper
            max_flow = pc1[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

            # Eq. 10 in the paper
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)))
    
    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean() # Eq. 11 in the paper
    elif have_dynamic_cluster:
        moved_cluster_loss = torch.mean(raw_dist0[raw_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_dist1[raw_dist1 <= TRUNCATED_DIST])
    # timer[5][3].stop()

    res_loss = {
        'chamfer_dis': chamfer_dis,
        'dynamic_chamfer_dis': dynamic_chamfer_dis,
        'static_flow_loss': static_cluster_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }
    return res_loss


def accflowLoss(res_dict, timer=None):
    """
    Self-supervised loss for AccumulateErrorFlow.

    The accumulated flow from pc0 is compared against the target frame (pc_final)
    using Chamfer distance. This allows self-supervised training without GT flow.

    res_dict should contain:
    - pc0: source point cloud [N, 3]
    - pc_target: target point cloud (final frame) [M, 3]
    - est_flow: accumulated flow from pc0 [N, 3]
    - pc0_labels (optional): dynamic labels for pc0 (label > 1 = cluster id, used for small object weighting)
    - pc_target_labels (optional): dynamic labels for target frame
    """
    pc0 = res_dict['pc0']
    pc_target = res_dict['pc_target']  # Final frame point cloud
    est_flow = res_dict['est_flow']  # Accumulated flow

    # Warped pc0 using accumulated flow
    pseudo_pc_target = pc0 + est_flow

    # Main loss: Chamfer distance between warped pc0 and real target
    # Use weighted chamfer distance for small objects
    raw_dist0, raw_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc_target, pc_target)
    # Apply truncation
    raw_dist0 = torch.clamp(raw_dist0, max=TRUNCATED_DIST)
    raw_dist1 = torch.clamp(raw_dist1, max=TRUNCATED_DIST)
    # Apply small object weight to pc0 side
    chamfer_dis = raw_dist0.mean() + raw_dist1.mean()

    # Optional: dynamic-specific losses if labels are provided
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    static_flow_loss = torch.tensor(0.0, device=est_flow.device)
    cluster_loss = torch.tensor(0.0, device=est_flow.device)

    if 'pc0_labels' in res_dict and 'pc_target_labels' in res_dict:
        pc0_label = res_dict['pc0_labels']
        pc_target_label = res_dict['pc_target_labels']

        pc0_dynamic = pc0[pc0_label > 0]
        pc_target_dynamic = pc_target[pc_target_label > 0]

        have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc_target_dynamic.shape[0] > 256)

        if have_dynamic_cluster:
            # Dynamic chamfer distance with small object weight
            dynamic_mask = pc0_label > 0
            dynamic_raw_dist0, dynamic_raw_dist1, _, _ = MyCUDAChamferDis.disid_res(
                pseudo_pc_target[dynamic_mask],
                pc_target_dynamic
            )
            dynamic_raw_dist0 = torch.clamp(dynamic_raw_dist0, max=TRUNCATED_DIST)
            dynamic_raw_dist1 = torch.clamp(dynamic_raw_dist1, max=TRUNCATED_DIST)
            # Apply small object weight
            dynamic_chamfer_dis = dynamic_raw_dist0.mean() + dynamic_raw_dist1.mean()

        # Static flow should be small (ego-motion compensated)
        static_mask = pc0_label == 0
        if static_mask.sum() > 0:
            static_flow_loss = torch.linalg.vector_norm(est_flow[static_mask], dim=-1).mean()

        # Cluster consistency loss: same cluster should have similar flow
        # Similar to SeFlow's cluster_based_pc0pc1
        unique_labels = torch.unique(pc0_label)
        raw_chamfer_dist0, raw_chamfer_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc_target)
        moved_cluster_norms = torch.tensor([], device=est_flow.device)

        for label in unique_labels:
            if label > 1 and have_dynamic_cluster:  # label > 1 means dynamic with valid cluster id
                mask = pc0_label == label
                cluster_id_flow = est_flow[mask, :]
                cluster_nnd = raw_chamfer_dist0[mask]

                if cluster_nnd.shape[0] <= 0:
                    continue

                # Find the point with max distance to target (likely edge point)
                sorted_idxs = torch.argsort(cluster_nnd, descending=True)
                nearby_label = pc_target_label[raw_idx0[mask][sorted_idxs]]
                non_zero_valid_indices = torch.nonzero(nearby_label > 0)

                if non_zero_valid_indices.shape[0] <= 0:
                    continue

                max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]

                # Reference flow from nearest neighbor
                max_flow = pc_target[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

                # All points in cluster should have similar flow
                moved_cluster_norms = torch.cat((
                    moved_cluster_norms,
                    torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)
                ))

        if moved_cluster_norms.shape[0] > 0:
            cluster_loss = moved_cluster_norms.mean()
        elif have_dynamic_cluster:
            cluster_loss = torch.mean(raw_chamfer_dist0[raw_chamfer_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_chamfer_dist1[raw_chamfer_dist1 <= TRUNCATED_DIST])

    res_loss = {
        'chamfer_dis': chamfer_dis,
        'dynamic_chamfer_dis': dynamic_chamfer_dis,
        'static_flow_loss': static_flow_loss,
        'cluster_based_pc0pc1': cluster_loss,
    }
    return res_loss
